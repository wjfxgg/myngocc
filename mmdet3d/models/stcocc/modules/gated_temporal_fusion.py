import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.utils import TORCH_VERSION, digit_version

from mmcv.runner import BaseModule, force_fp32
from mmcv.utils import ext_loader
import math
from mmdet3d.models.builder import HEADS
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])
@HEADS.register_module()
class VisibilityGatedFusion(BaseModule):
    def __init__(
        self,
        top_k=None,  # 稀疏采样的"前景体素数量"（只融合top_k个高占用概率的体素，减少计算量）
        history_num=8,  # 历史帧数量（融合最近8帧的特征）
        single_bev_num_channels=None,  # 单帧BEV特征的通道数（如256）
        foreground_idx=None,  # 前景类别索引（未实际使用，保留原代码兼容）
        num_classes=17,  # 占用预测的类别数（如17类障碍物）
        occ_embedims=32,  # 占用预测的嵌入维度（将类别概率映射到32维向量）
        # 新增：门控机制参数
        vis_gate_feat_dim=32,  # 可见性特征的编码维度（未实际使用，属设计残留，可忽略）
        **kwargs  # 兼容其他未定义参数
):
        super(VisibilityGatedFusion, self).__init__()  # 调用父类BaseModule的初始化
        # 1. 保存基础参数（实例变量，后续方法可调用）
        self.single_bev_num_channels = single_bev_num_channels  # 单帧BEV通道数
        self.history_bev = None  # 缓存历史帧BEV特征（初始为None，第一帧后赋值）
        self.history_last_bev = None  # 缓存上一帧最终融合后的BEV特征
        self.history_forward_augs = None  # 缓存历史帧的"姿态变换矩阵"（如BDAM矩阵，用于坐标对齐）
        self.history_vis = None  # 新增：缓存历史帧的可见性概率（[bs, history_num, z, h, w]）
        self.history_num = history_num  # 历史帧数量
        self.history_seq_ids = None  # 缓存每个样本的"序列ID"（区分不同视频序列）
        self.history_sweep_time = None  # 缓存每个样本的"历史帧计数"（记录已缓存多少帧）
        self.history_cam_sweep_freq = 0.5  # 帧间隔时间（秒，未实际使用，保留原逻辑）
        self.top_k = top_k  # 稀疏采样的前景体素数
        self.foreground_idx = foreground_idx  # 前景类别索引（兼容原代码）

        # 2. 占用预测嵌入层（保留原逻辑：将类别概率映射为低维向量，用于融合）
        self.occ_embedding = nn.Sequential(  # 序列层：Linear→Softplus→Linear
            nn.Linear(num_classes, occ_embedims),  # 类别概率（17维）→嵌入维度（32维）
            nn.Softplus(),  # 激活函数：比ReLU平滑，避免梯度消失
            nn.Linear(occ_embedims, occ_embedims),  # 32维→32维，进一步压缩特征
        )

        # 3. 新增：可见性特征编码层（设计残留，forward中未调用，可忽略）
        self.vis_encoder = nn.Sequential(
            nn.Linear(6, vis_gate_feat_dim),  # 输入：3D坐标(x,y,z)+相机投影特征(3)→32维
            nn.ReLU(),  # 激活函数
            nn.Linear(vis_gate_feat_dim, 2),  # 32维→2维（对应V_prev和V_curr）
            nn.Sigmoid()  # 压缩到[0,1]，符合概率范围
        )

        # 4. 新增：门控机制的可学习参数（用ParameterDict包装，可被优化器更新）
        self.gate_params = nn.ParameterDict({
            # 历史特征门控公式：w_hist = sigmoid(α*V_prev - β*V_curr + γ)
            'alpha': nn.Parameter(torch.tensor(5.0)),  # α：历史可见性的"增益系数"（初始5.0，放大V_prev的影响）
            'beta': nn.Parameter(torch.tensor(5.0)),   # β：当前可见性的"抑制系数"（初始5.0，削弱V_curr的影响）
            'gamma': nn.Parameter(torch.tensor(0.0)),  # γ：偏置项（调整门控的基础输出）

            # 当前特征门控公式：w_curr = sigmoid(δ*V_curr - ε*V_prev + ζ)
            'delta': nn.Parameter(torch.tensor(5.0)),  # δ：当前可见性的"增益系数"
            'epsilon': nn.Parameter(torch.tensor(5.0)),# ε：历史可见性的"抑制系数"
            'zeta': nn.Parameter(torch.tensor(0.0)),   # ζ：偏置项

            # 协同增强系数：增强"历史+当前都可见"的特征权重
            'eta': nn.Parameter(torch.tensor(0.2))     # η：初始0.2，控制增强幅度
        })

        # 5. 融合线性层（修改输入维度：原代码拼接多帧，现用门控加权后单帧+嵌入，故维度减少）
        self.history_fusion_linear = nn.Sequential(  # 前景融合层
            nn.Linear(single_bev_num_channels + occ_embedims, single_bev_num_channels),  # （BEV通道数+嵌入维度）→BEV通道数
            nn.Softplus(),  # 激活函数
            nn.Linear(single_bev_num_channels, single_bev_num_channels),  # 保持通道数，细化特征
        )
        self.history_fusion_bg_linear = nn.Sequential(  # 背景融合层（逻辑同前景，参数独立）
            nn.Linear(single_bev_num_channels + occ_embedims, single_bev_num_channels),
            nn.Softplus(),
            nn.Linear(single_bev_num_channels, single_bev_num_channels),
        )

    def generate_grid(self, curr_bev):
        """生成3D体素的(x,y,z)坐标网格，用于后续可见性计算和坐标变换"""
        n, c_, z, h, w = curr_bev.shape  # curr_bev形状：[ batch_size, 通道数, 高度(z), 宽度(h), 长度(w) ]
        # 1. 生成单轴坐标（linspace：在[0, 维度-1]间生成"维度数"个均匀点）
        xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device)  # x轴（长度方向）：[w]
        ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device)  # y轴（宽度方向）：[h]
        zs = torch.linspace(0, z - 1, z, dtype=curr_bev.dtype, device=curr_bev.device)  # z轴（高度方向）：[z]
        # 2. 生成3D网格（meshgrid：将单轴坐标组合成网格，indexing='xy'表示按(x,y)顺序）
        x_grid, y_grid, z_grid = torch.meshgrid(xs, ys, zs, indexing='xy')  # 输出：[h, w, z]（注意维度顺序）
        # 3. 组合坐标并调整维度（x,y,z→[h, w, z, 3]）
        grid = torch.stack((x_grid, y_grid, z_grid), dim=-1).permute(1, 0, 2, 3)  # stack后[h,w,z,3]，permute保持维度逻辑
        # 4. 扩展到batch维度（每个样本用相同网格，故expand(n, ...)）
        grid = grid.unsqueeze(0).expand(n, -1, -1, -1, -1)  # 最终形状：[n, h, w, z, 3]
        return grid

    def compute_visibility(self, grid, cam_params, is_history=False):
        """
        计算体素可见性概率V（连续值，[0,1]）
        Args:
            grid: [n, h, w, z, 3] 体素3D坐标（x,y,z）
            cam_params: dict 相机参数（内参K、外参T_ego2cam、图像尺寸img_shape）
            is_history: bool 是否计算历史帧的可见性（决定用当前/历史相机参数）
        Returns:
            vis_prob: [n, h, w, z] 可见性概率（越高=越可能被观测）
        """
        n, h, w, z, _ = grid.shape  # 解析网格形状
        # 1. 获取相机参数（内参K：3x3，外参T_ego2cam：4x4，将自车坐标系→相机坐标系）
        K = cam_params['K']  # [n, 3, 3] 相机内参（焦距、主点坐标）
        # 选择外参：当前帧用T_ego2cam，历史帧用T_ego2cam_prev
        T_ego2cam = cam_params['T_ego2cam'] if not is_history else cam_params['T_ego2cam_prev']  # [n, 4, 4]

        # 2. 体素坐标→齐次坐标（3D→4D，方便仿射变换：平移+旋转）
        grid_flat = grid.reshape(n, -1, 3)  # 展平：[n, h*w*z, 3]（N=h*w*z，每个样本的所有体素）
        # 齐次坐标：在最后加1列"1"（表示点坐标，区别于向量）
        grid_hom = torch.cat([grid_flat, torch.ones_like(grid_flat[..., :1])], dim=-1)  # [n, N, 4]

        # 3. 自车坐标系→相机坐标系（用外参T_ego2cam变换）
        # bmm：批量矩阵乘法（每个样本独立计算），permute调整维度适配矩阵乘法
        cam_coords = torch.bmm(T_ego2cam[:, :3, :4], grid_hom.permute(0, 2, 1)).permute(0, 2, 1)  # [n, N, 3]
        # 注：T_ego2cam[:, :3, :4]取前3行（忽略齐次项），结果是相机坐标系下的3D坐标（x,y,z：z为深度）

        # 4. 相机坐标系→图像平面（用内参K投影，得到像素坐标(u,v)）
        img_coords = torch.bmm(K, cam_coords.permute(0, 2, 1)).permute(0, 2, 1)  # [n, N, 3]
        # 透视除法：像素坐标(u,v) = (x/z, y/z)（z为深度，避免远近距离导致的坐标偏移）
        img_xy = img_coords[..., :2] / (img_coords[..., 2:3] + 1e-8)  # [n, N, 2]（+1e-8避免除0）

        # 5. 判断可见性的2个核心条件（均为bool→float，便于后续计算）
        # 条件1：深度>0（体素在相机"前方"，相机无法观测后方物体）
        depth_valid = (cam_coords[..., 2] > 0).float()  # [n, N]（1=有效，0=无效）
        # 条件2：像素坐标(u,v)在图像范围内（不超出图像高宽）
        h_img, w_img = cam_params['img_shape']  # 图像尺寸（如[1280, 1920]）
        u_valid = (img_xy[..., 0] >= 0) & (img_xy[..., 0] < w_img)  # u在[0, w_img)
        v_valid = (img_xy[..., 1] >= 0) & (img_xy[..., 1] < h_img)  # v在[0, h_img)
        img_valid = (u_valid & v_valid).float()  # [n, N]（1=在范围内，0=超出）

        # 6. 融合条件→可见性概率（平滑处理，保证可微）
        vis_prob = depth_valid * img_valid  # [n, N]（1=同时满足两个条件，0=至少一个不满足）
        # sigmoid平滑：将{0,1}硬阈值转为[0,1]连续值，增强模型鲁棒性（10是放大系数，提升区分度）
        vis_prob = F.sigmoid(10 * (vis_prob - 0.5))  # 输出：[n, N]
        # 7. 还原为原网格形状（匹配输入grid的维度）
        return vis_prob.reshape(n, h, w, z)  # [n, h, w, z]

    def compute_gate_weights(self, V_prev, V_curr):
        """
        计算历史特征权重w_hist和当前特征权重w_curr（sigmoid门控，连续可微）
        Args:
            V_prev: [n, num_samples] 历史可见性概率（num_samples=top_k或bg_k）
            V_curr: [n, num_samples] 当前可见性概率
        Returns:
            w_hist: [n, num_samples, 1] 历史特征权重（最后一维用于匹配特征维度）
            w_curr: [n, num_samples, 1] 当前特征权重
        """
        # 1. 计算历史特征权重的logits（未激活的原始值）
        # 公式：w_hist_logits = α*V_prev - β*V_curr + γ
        w_hist_logits = self.gate_params['alpha'] * V_prev - self.gate_params['beta'] * V_curr + self.gate_params['gamma']
        # sigmoid激活：将logits压缩到[0,1]，unsqueeze(-1)加维度（适配特征的通道维度）
        w_hist = torch.sigmoid(w_hist_logits).unsqueeze(-1)  # [n, num_samples, 1]

        # 2. 计算当前特征权重的logits（对称公式）
        # 公式：w_curr_logits = δ*V_curr - ε*V_prev + ζ
        w_curr_logits = self.gate_params['delta'] * V_curr - self.gate_params['epsilon'] * V_prev + self.gate_params['zeta']
        w_curr = torch.sigmoid(w_curr_logits).unsqueeze(-1)  # [n, num_samples, 1]

        return w_hist, w_curr

    @force_fp32()  # 装饰器：强制用float32计算（避免半精度训练时的精度损失）
    def forward(self, curr_bev, cam_params, history_fusion_params, dx, bx, history_last_bev=None, last_occ_pred=None, nonempty_prob=None):
        # 输入参数解析（关键）：
        # curr_bev: [bs, c, z, h, w] 当前帧BEV特征（bs=batch_size，c=通道数）
        # cam_params: dict 相机参数（含K、T_ego2cam、T_ego2cam_prev、img_shape等）
        # history_fusion_params: dict 历史融合参数（seq_ids=序列ID、start_of_sequence=是否新序列、curr_to_prev_ego_rt=帧间变换）
        # dx: 体素大小（如[0.5, 0.5, 0.5]），用于坐标转换
        # bx: 体素偏移量（如[0, 0, 0]），用于坐标转换
        # last_occ_pred: [bs, z, h, w, num_classes] 上一帧的占用预测结果（类别概率）
        # nonempty_prob: [bs, z, h, w] 体素被占用的概率（1=占用，0=空闲）

        # 0、检查特征类型（是否为3D体素特征：5维=体素，4维=BEV平面）
        voxel_feat = True if len(curr_bev.shape) == 5 else False
        bs, c_, z, h, w = curr_bev.shape  # 解析当前帧特征形状

        # 1、处理序列信息（保留原逻辑：区分"新序列"和"旧序列"，避免跨序列融合）
        # 序列ID：区分不同视频序列（如样本1属于序列A，样本2属于序列B）
        if type(history_fusion_params['sequence_group_idx']) is list:
            seq_ids = history_fusion_params['sequence_group_idx'][0]  # 兼容列表格式输入
        else:
            seq_ids = history_fusion_params['sequence_group_idx']  # [bs]
        # 是否为新序列的第一帧（start_of_sequence=True：需重置历史缓存）
        if type(history_fusion_params['start_of_sequence']) is list:
            start_of_sequence = history_fusion_params['start_of_sequence'][0]  # [bs]（bool）
        else:
            start_of_sequence = history_fusion_params['start_of_sequence']
        # 帧间变换矩阵：当前帧→历史帧的自车姿态变换（用于历史特征对齐）
        if type(history_fusion_params['curr_to_prev_ego_rt']) is list:
            curr_to_prev_ego_rt = history_fusion_params['curr_to_prev_ego_rt'][0]  # [bs, 4, 4]
        else:
            curr_to_prev_ego_rt = history_fusion_params['curr_to_prev_ego_rt']
        forward_augs = cam_params[-1]  # 姿态增强矩阵（如BDAM，用于BEV特征的旋转/平移对齐）

        # 2、初始化历史缓存（第一次调用forward时，history_bev为None）
        if self.history_bev is None:
            # 历史BEV特征：将当前帧重复history_num次（初始无历史，用当前帧填充）
            self.history_bev = curr_bev.repeat(1, self.history_num, 1, 1, 1).detach()  # [bs, history_num*c, z, h, w]
            # 历史可见性：初始为0（无历史观测，默认不可见）
            self.history_vis = torch.zeros(bs, self.history_num, z, h, w, device=curr_bev.device).detach()  # [bs, history_num, z, h, w]
            # 其他缓存：姿态变换、序列ID、帧计数
            self.history_forward_augs = forward_augs.clone().detach()  # [bs, 4, 4]
            self.history_seq_ids = seq_ids.clone().detach()  # [bs]
            self.history_sweep_time = curr_bev.new_zeros(bs, self.history_num).detach()  # [bs, history_num]（计数）

        # 3、处理新序列（start_of_sequence=True的样本，重置其历史缓存）
        self.history_sweep_time += 1  # 所有样本的历史帧计数+1
        if start_of_sequence.sum() > 0:  # 存在新序列样本
            # 重置新序列的历史BEV（用当前帧填充）
            self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_num, 1, 1, 1).detach()
            # 重置新序列的历史可见性（0=不可见）
            self.history_vis[start_of_sequence] = 0.0
            # 重置姿态变换、序列ID、帧计数
            self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence].detach()
            self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence].detach()
            self.history_sweep_time[start_of_sequence] = 0  # 新序列计数从0开始

        # 4、生成体素网格坐标（用于可见性计算和历史特征采样）
        grid_3d = self.generate_grid(curr_bev)  # [bs, h, w, z, 3]（每个体素的x,y,z）
        # 生成"体素→BEV"的坐标转换矩阵（复用原代码，确保坐标对齐）
        feat2bev = self.generate_feat2bev(grid_3d, dx, bx)  # [1, 4, 4]（所有样本共用）

        # 5、历史特征坐标对齐（将历史帧特征映射到当前帧坐标系）
        # 帧间变换矩阵rt_flow：历史帧→当前帧的完整变换（含姿态增强）
        rt_flow = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_ego_rt @ 
                torch.inverse(forward_augs) @ feat2bev)  # [bs, 4, 4]
        # 网格坐标→齐次坐标（加1列1），并应用rt_flow变换
        grid = rt_flow.view(bs, 1, 1, 1, 4, 4) @ torch.cat([grid_3d, torch.ones_like(grid_3d[..., :1])], dim=-1).unsqueeze(-1)
        # 归一化坐标：适配F.grid_sample的输入要求（坐标需在[-1,1]范围内）
        normalize_factor = torch.tensor([w - 1.0, h - 1.0, z - 1.0], dtype=curr_bev.dtype, device=curr_bev.device)
        grid_sampler = grid[..., :3, 0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0  # [bs, h, w, z, 3]

        # 6、采样历史BEV特征（从对齐后的坐标中提取历史特征）
        mc = self.history_bev.shape[1]  # 历史特征总通道数：history_num * c_（如8*256=2048）
        sampled_history_bev = F.grid_sample(
            self.history_bev.reshape(bs, mc, z, h, w),  # 输入特征：[bs, mc, z, h, w]
            grid_sampler.permute(0, 3, 1, 2, 4),  # grid形状适配：[bs, z, h, w, 3]（grid_sample要求的顺序）
            align_corners=True,  # 对齐网格角点（提升采样精度）
            mode='bilinear'  # 双线性采样（平滑过渡）
        )  # 输出：[bs, mc, z, h, w]（采样后的历史特征，与当前帧维度一致）

        # 7、计算可见性概率V_prev（历史）和V_curr（当前）
        # 7.1 当前可见性V_curr：当前帧相机能否观测到每个体素
        V_curr = self.compute_visibility(
            grid_3d, 
            cam_params={
                'K': cam_params['K'],  # 当前帧相机内参
                'T_ego2cam': cam_params['T_ego2cam'],  # 当前帧外参
                'img_shape': cam_params['img_shape']  # 图像尺寸
            }, 
            is_history=False
        )  # [bs, h, w, z]

        # 7.2 历史可见性V_prev：历史帧相机能否观测到（对齐到当前帧的）体素
        V_prev = self.compute_visibility(
            grid_3d,  # 已对齐到当前帧的体素坐标
            cam_params={
                'K': cam_params['K_prev'],  # 历史帧相机内参
                'T_ego2cam_prev': cam_params['T_ego2cam_prev'],  # 历史帧外参
                'img_shape': cam_params['img_shape']
            }, 
            is_history=True
        )  # [bs, h, w, z]

        # 8、稀疏采样（只融合"前景"和"背景"体素，减少计算量）
        # 8.1 展平特征和可见性（便于按索引采样）
        curr_bev_flat = curr_bev.reshape(bs, c_, -1)  # 当前特征：[bs, c_, h*w*z]
        history_bev_flat = sampled_history_bev.reshape(bs, mc, -1)  # 历史特征：[bs, mc, h*w*z]
        V_prev_flat = V_prev.reshape(bs, -1)  # 历史可见性：[bs, h*w*z]
        V_curr_flat = V_curr.reshape(bs, -1)  # 当前可见性：[bs, h*w*z]
        nonempty_prob_flat = nonempty_prob.reshape(bs, -1)  # 占用概率：[bs, h*w*z]
        total_voxels = nonempty_prob_flat.shape[1]  # 总体素数：h*w*z

        # 8.2 采样前景和背景索引（基于nonempty_prob：占用概率）
        # 前景：top_k个"高占用概率"体素（nonempty_prob大=有物体）
        fg_indices = torch.topk(nonempty_prob_flat, self.top_k, dim=1)[1]  # [bs, top_k]（索引）
        # 背景：(总体素数-top_k)个"低占用概率"体素（1-nonempty_prob大=无物体）
        bg_indices = torch.topk(1 - nonempty_prob_flat, total_voxels - self.top_k, dim=1)[1]  # [bs, bg_k]（bg_k=total_voxels-top_k）

        # 9、门控融合（分前景/背景，避免相互干扰）
        # 9.1 前景融合（核心：高优先级，需精准）
        # ① 提取前景的特征、可见性（用gather按索引采样）
        # 历史前景特征：[bs, mc, top_k]
        fg_history_feat = torch.gather(history_bev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, mc, 1))
        # 当前前景特征：[bs, c_, top_k]
        fg_curr_feat = torch.gather(curr_bev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, c_, 1))
        # 历史前景可见性：[bs, top_k]
        fg_V_prev = torch.gather(V_prev_flat, dim=1, index=fg_indices)
        # 当前前景可见性：[bs, top_k]
        fg_V_curr = torch.gather(V_curr_flat, dim=1, index=fg_indices)

        # ② 计算前景门控权重
        fg_w_hist, fg_w_curr = self.compute_gate_weights(fg_V_prev, fg_V_curr)  # [bs, top_k, 1]

        # ③ 前景占用嵌入（将类别概率映射为向量）
        # 展平上一帧占用预测：[bs, h*w*z, num_classes]
        last_occ_pred_flat = last_occ_pred.permute(0, 3, 2, 1, 4).reshape(bs, -1, num_classes)
        # 采样前景占用预测并嵌入：[bs, top_k, occ_embedims]
        fg_occ_embed = self.occ_embedding(torch.gather(last_occ_pred_flat, dim=1, index=fg_indices.unsqueeze(-1).repeat(1, 1, num_classes)))
        fg_occ_embed = fg_occ_embed.permute(0, 2, 1)  # 调整维度：[bs, occ_embedims, top_k]

        # ④ 前景特征融合（加权求和+协同增强）
        # 加权求和：历史特征*w_hist + 当前特征*w_curr（permute调整维度匹配）
        fg_fused = (fg_w_hist * fg_history_feat.permute(0, 2, 1) +  # [bs, top_k, mc] * [bs, top_k, 1]
                    fg_w_curr * fg_curr_feat.permute(0, 2, 1))    # [bs, top_k, c_] * [bs, top_k, 1]
        # 协同增强项：仅当"历史+当前都可见"时增强（min(V_prev,V_curr)表示两者的最小可见性）
        fg_coeff = self.gate_params['eta'] * torch.min(fg_V_prev, fg_V_curr).unsqueeze(-1)  # [bs, top_k, 1]
        fg_fused += fg_coeff * (fg_history_feat.permute(0, 2, 1) * fg_curr_feat.permute(0, 2, 1))  # 元素乘法交互
        # 线性变换：融合特征+占用嵌入→最终前景特征（调整维度匹配）
        fg_fused = torch.cat([fg_fused, fg_occ_embed.permute(0, 2, 1)], dim=-1)  # 拼接：[bs, top_k, mc + occ_embedims]
        fg_fused = self.history_fusion_linear(fg_fused).permute(0, 2, 1)  # [bs, c_, top_k]（还原通道维度）

        # 9.2 背景融合（逻辑同前景，简化计算：历史特征取一半通道）
        # ① 提取背景特征和可见性
        bg_history_feat = torch.gather(history_bev_flat[:, :mc//2], dim=2, index=bg_indices.unsqueeze(1).repeat(1, mc//2, 1))  # [bs, mc//2, bg_k]
        bg_curr_feat = torch.gather(curr_bev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, c_, 1))  # [bs, c_, bg_k]
        bg_V_prev = torch.gather(V_prev_flat, dim=1, index=bg_indices)  # [bs, bg_k]
        bg_V_curr = torch.gather(V_curr_flat, dim=1, index=bg_indices)  # [bs, bg_k]

        # ② 计算背景门控权重
        bg_w_hist, bg_w_curr = self.compute_gate_weights(bg_V_prev, bg_V_curr)  # [bs, bg_k, 1]

        # ③ 背景占用嵌入
        bg_occ_embed = self.occ_embedding(torch.gather(last_occ_pred_flat, dim=1, index=bg_indices.unsqueeze(-1).repeat(1, 1, num_classes)))  # [bs, bg_k, occ_embedims]
        bg_occ_embed = bg_occ_embed.permute(0, 2, 1)  # [bs, occ_embedims, bg_k]

        # ④ 背景特征融合
        bg_fused = (bg_w_hist * bg_history_feat.permute(0, 2, 1) + 
                    bg_w_curr * bg_curr_feat.permute(0, 2, 1))
        bg_coeff = self.gate_params['eta'] * torch.min(bg_V_prev, bg_V_curr).unsqueeze(-1)
        bg_fused += bg_coeff * (bg_history_feat.permute(0, 2, 1) * bg_curr_feat.permute(0, 2, 1))
        bg_fused = torch.cat([bg_fused, bg_occ_embed.permute(0, 2, 1)], dim=-1)
        bg_fused = self.history_fusion_bg_linear(bg_fused).permute(0, 2, 1)  # [bs, c_, bg_k]

        # 10、更新当前BEV特征（将融合后的前景/背景特征加回原位置）
        curr_bev_updated = curr_bev_flat.clone()  # 复制当前特征（避免修改原始数据）
        # 前景特征更新：scatter_add_按索引累加（将fg_fused加到curr_bev_updated的前景位置）
        curr_bev_updated.scatter_add_(dim=2, index=fg_indices.unsqueeze(1).repeat(1, c_, 1), src=fg_fused)
        # 背景特征更新：同理累加背景融合特征
        curr_bev_updated.scatter_add_(dim=2, index=bg_indices.unsqueeze(1).repeat(1, c_, 1), src=bg_fused)
        # 还原为3D体素形状
        curr_bev_updated = curr_bev_updated.reshape(bs, c_, z, h, w)  # [bs, c_, z, h, w]

        # 11、更新历史缓存（为下一帧准备）
        # 历史BEV滚动更新：去掉最旧的一帧（前c_通道），加入当前帧（detach避免计算图累积）
        self.history_bev = torch.cat([self.history_bev[:, c_:], curr_bev.detach()], dim=1)
        # 历史可见性滚动更新：去掉最旧的一帧，加入当前帧可见性
        self.history_vis = torch.cat([self.history_vis[:, 1:], V_curr.unsqueeze(1).detach()], dim=1)
        # 更新姿态变换和上一帧融合特征
        self.history_forward_augs = forward_augs.clone().detach()
        self.history_last_bev = curr_bev_updated.detach()

        # 返回最终融合后的当前帧特征
        return curr_bev_updated.clone()

    # 复用原generate_feat2bev函数
    def generate_feat2bev(self, grid, dx, bx):
        """复用原代码：生成4x4矩阵，将体素的(x,y,z)坐标转换到BEV空间（对齐体素中心）"""
        feat2bev = torch.zeros((4, 4), dtype=grid.dtype).to(grid)  # 初始化4x4零矩阵
        # 对角线：体素大小（dx[0]=x轴体素宽，dx[1]=y轴体素宽，dx[2]=z轴体素高）
        feat2bev[0, 0] = dx[0]
        feat2bev[1, 1] = dx[1]
        feat2bev[2, 2] = dx[2]
        # 平移项：确保体素"中心"与坐标对齐（bx是体素偏移，减dx/2是中心校准）
        feat2bev[0, 3] = bx[0] - dx[0] / 2.
        feat2bev[1, 3] = bx[1] - dx[1] / 2.
        feat2bev[2, 3] = bx[2] - dx[2] / 2.
        feat2bev[3, 3] = 1  # 齐次坐标的最后一行（固定为1）
        feat2bev = feat2bev.view(1, 4, 4)  # 扩展为[1,4,4]，适配batch维度
        return feat2bev







@HEADS.register_module()
class VisibilityGatedFusion1(BaseModule):
    def __init__(
            self,
            top_k=None,
            history_num=8,
            single_bev_num_channels=None,
            foreground_idx=None,
            num_classes=17,
            occ_embedims=32,
            img_shape=[256, 704],  # 新增：图像尺寸（高，宽），需根据实际数据调整
            num_cams=6,  # 新增：相机数量（根据cam_params[0].shape[1]=6设置）
            **kwargs
    ):
        super(VisibilityGatedFusion1, self).__init__()
        self.single_bev_num_channels = single_bev_num_channels
        self.history_bev = None
        self.history_last_bev = None
        self.history_forward_augs = None  # 历史BDAM矩阵
        self.history_vis = None  # 历史可见性概率
        self.history_num = history_num
        self.history_seq_ids = None
        self.history_sweep_time = None
        self.history_cam_sweep_freq = 0.5
        self.top_k = top_k
        self.foreground_idx = foreground_idx

        # 新增：相机参数相关（当前+历史）
        self.img_shape = img_shape  # 图像尺寸（h_img, w_img）
        self.num_cams = num_cams    # 相机数量（6）
        # 历史相机参数缓存：内参（K_prev）和外参（T_ego2cam_prev）
        self.history_cam_intrins = None  # [bs, history_num, num_cams, 4, 4]
        self.history_cam_extrins = None  # [bs, history_num, num_cams, 4, 4]

        # 占用预测嵌入（保留原逻辑）
        self.occ_embedding = nn.Sequential(
            nn.Linear(num_classes, occ_embedims),
            nn.Softplus(),
            nn.Linear(occ_embedims, occ_embedims),
        )

        # 门控机制可学习参数（保留原逻辑）
        self.gate_params = nn.ParameterDict({
            'alpha': nn.Parameter(torch.tensor(5.0)),
            'beta': nn.Parameter(torch.tensor(5.0)),
            'gamma': nn.Parameter(torch.tensor(0.0)),
            'delta': nn.Parameter(torch.tensor(5.0)),
            'epsilon': nn.Parameter(torch.tensor(5.0)),
            'zeta': nn.Parameter(torch.tensor(0.0)),
            'eta': nn.Parameter(torch.tensor(0.2))
        })
        self.rt_vis_calculator = EfficientRayTracingVisibility(nonempty_thresh=0.1)
        # 融合线性层（保留原逻辑）
        self.history_fusion_linear = nn.Sequential(
            nn.Linear(single_bev_num_channels + occ_embedims, single_bev_num_channels),
            nn.Softplus(),
            nn.Linear(single_bev_num_channels, single_bev_num_channels),
        )
        self.history_fusion_bg_linear = nn.Sequential(
            nn.Linear(single_bev_num_channels + occ_embedims, single_bev_num_channels),
            nn.Softplus(),
            nn.Linear(single_bev_num_channels, single_bev_num_channels),
        )

    def generate_grid(self, curr_bev):
        """生成3D体素网格坐标（x,y,z），保留原逻辑"""
        n, c_, z, h, w = curr_bev.shape
        xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device)
        ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device)
        zs = torch.linspace(0, z - 1, z, dtype=curr_bev.dtype, device=curr_bev.device)
        x_grid, y_grid, z_grid = torch.meshgrid(xs, ys, zs, indexing='xy')
        grid = torch.stack((x_grid, y_grid, z_grid), dim=-1).permute(1, 0, 2, 3)
        grid = grid.unsqueeze(0).expand(n, -1, -1, -1, -1)  # [bs, h, w, z, 3]
        return grid

    def compute_visibility(self, grid, cam_intrins, cam_extrins, img_shape):
        """
        升级：支持多相机可见性计算（取所有相机的最大可见性）
        Args:
            grid: [bs, h, w, z, 3] 体素3D坐标
            cam_intrins: [bs, num_cams, 4, 4] 多相机内参（当前/历史）
            cam_extrins: [bs, num_cams, 4, 4] 多相机外参（当前/历史）
            img_shape: [2] 图像尺寸（h_img, w_img）
        Returns:
            vis_prob: [bs, h, w, z] 体素可见性概率（多相机取max）
        """
        bs, h, w, z, _ = grid.shape
        h_img, w_img = img_shape
        device = grid.device

        # 1. 扩展网格维度以匹配多相机：[bs, num_cams, h, w, z, 3]
        grid_cam = grid.unsqueeze(1).expand(bs, self.num_cams, h, w, z, 3)
        # 展平为：[bs*num_cams, h*w*z, 3]（便于批量计算）
        grid_flat = grid_cam.reshape(-1, h*w*z, 3)
        # 转为齐次坐标：[bs*num_cams, h*w*z, 4]
        grid_hom = torch.cat([grid_flat, torch.ones_like(grid_flat[..., :1])], dim=-1)

        # 2. 处理相机参数：展平为[bs*num_cams, 4, 4]
        cam_intrins_flat = cam_intrins.reshape(-1, 4, 4)  # 内参（4x4，实际有效部分是3x3）
        cam_extrins_flat = cam_extrins.reshape(-1, 4, 4)  # 外参（相机→自车）

        # 3. 自车坐标系 → 相机坐标系（外参变换）
        # 外参是相机→自车，需逆变换得到自车→相机：inv(cam_extrins)
        extrins_inv = torch.inverse(cam_extrins_flat)
        # 变换：[bs*num_cams, h*w*z, 3]（取前3行，忽略齐次项）
        cam_coords = torch.bmm(extrins_inv[:, :3, :4], grid_hom.permute(0, 2, 1)).permute(0, 2, 1)  # [bs*num_cams, h*w*z, 3]

        # 4. 相机坐标系 → 图像平面（内参投影）
        # 关键修复：内参取前3行、前3列（3x3有效部分），与相机坐标（3维）匹配
        img_coords = torch.bmm(cam_intrins_flat[:, :3, :3], cam_coords.permute(0, 2, 1)).permute(0, 2, 1)  # [bs*num_cams, h*w*z, 3]
        # 透视除法：(u, v) = (x/z, y/z)，z为相机坐标系下的深度
        depth = cam_coords[..., 2:3] + 1e-8  # 避免除0
        img_xy = img_coords[..., :2] / depth  # [bs*num_cams, h*w*z, 2]（u, v）

        # 5. 判断可见性（两个条件）
        # 条件1：深度>0（体素在相机前方）
        depth_valid = (cam_coords[..., 2] > 0).float()  # [bs*num_cams, h*w*z]
        # 条件2：像素坐标在图像范围内（u∈[0, w_img)，v∈[0, h_img)）
        u_valid = (img_xy[..., 0] >= 0) & (img_xy[..., 0] < w_img)
        v_valid = (img_xy[..., 1] >= 0) & (img_xy[..., 1] < h_img)
        img_valid = (u_valid & v_valid).float()  # [bs*num_cams, h*w*z]

        # 6. 计算单相机可见性概率（平滑处理）
        cam_vis = depth_valid * img_valid  # [bs*num_cams, h*w*z]
        cam_vis = F.sigmoid(10 * (cam_vis - 0.5))  # 压缩到[0,1]

        # 7. 多相机聚合：取每个体素的最大可见性（一个相机看到即算可见）
        cam_vis = cam_vis.reshape(bs, self.num_cams, h*w*z)  # [bs, num_cams, h*w*z]
        vis_prob = cam_vis.max(dim=1)[0]  # [bs, h*w*z]（多相机取max）

        # 8. 还原为网格形状
        return vis_prob.reshape(bs, h, w, z)  # [bs, h, w, z]

    def compute_gate_weights(self, V_prev, V_curr):
        """计算门控权重，保留原逻辑"""
        w_hist_logits = self.gate_params['alpha'] * V_prev - self.gate_params['beta'] * V_curr + self.gate_params['gamma']
        w_hist = torch.sigmoid(w_hist_logits).unsqueeze(-1)

        w_curr_logits = self.gate_params['delta'] * V_curr - self.gate_params['epsilon'] * V_prev + self.gate_params['zeta']
        w_curr = torch.sigmoid(w_curr_logits).unsqueeze(-1)

        return w_hist, w_curr

    def generate_feat2bev(self, grid, dx, bx):
        """生成体素→BEV转换矩阵，保留原逻辑"""
        feat2bev = torch.zeros((4, 4), dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = dx[0]
        feat2bev[1, 1] = dx[1]
        feat2bev[2, 2] = dx[2]
        feat2bev[0, 3] = bx[0] - dx[0] / 2.
        feat2bev[1, 3] = bx[1] - dx[1] / 2.
        feat2bev[2, 3] = bx[2] - dx[2] / 2.
        feat2bev[3, 3] = 1
        feat2bev = feat2bev.view(1, 4, 4)
        return feat2bev

    @force_fp32()
    def forward(self, curr_bev, cam_params, history_fusion_params, dx, bx, history_last_bev=None, last_occ_pred=None, nonempty_prob=None):
        # 输入参数解析
        curr_cam_extrins = cam_params[0]
        curr_cam_intrins = cam_params[2]
        forward_augs = cam_params[4]

        # 0、基础参数解析 + 打印（不变）
        voxel_feat = True if len(curr_bev.shape) == 5 else False
        bs, c_, z, h, w = curr_bev.shape
        device = curr_bev.device
        print(f"=== 基础参数 ===")
        print(f"curr_bev.shape: {curr_bev.shape} → (bs, c_, z, h, w) = ({bs}, {c_}, {z}, {h}, {w})")
        print(f"history_num: {self.history_num} → 历史总通道数 mc 应为 {self.history_num} * {c_} = {self.history_num * c_}")

        # 1、处理序列信息（必须保留！不能省略！）
        # 从 history_fusion_params 中提取关键变量：seq_ids、start_of_sequence、curr_to_prev_ego_rt
        if isinstance(history_fusion_params['sequence_group_idx'], list):
            seq_ids = history_fusion_params['sequence_group_idx'][0]  # 兼容列表格式输入
        else:
            seq_ids = history_fusion_params['sequence_group_idx']  # [bs]（序列ID）
        
        if isinstance(history_fusion_params['start_of_sequence'], list):
            start_of_sequence = history_fusion_params['start_of_sequence'][0]  # [bs]（是否新序列）
        else:
            start_of_sequence = history_fusion_params['start_of_sequence']
        
        if isinstance(history_fusion_params['curr_to_prev_ego_rt'], list):
            curr_to_prev_ego_rt = history_fusion_params['curr_to_prev_ego_rt'][0]  # [bs, 4, 4]（关键：帧间姿态变换矩阵）
        else:
            curr_to_prev_ego_rt = history_fusion_params['curr_to_prev_ego_rt']
        print(f"\n=== 序列信息 ===")
        print(f"curr_to_prev_ego_rt.shape: {curr_to_prev_ego_rt.shape} → (bs, 4, 4)")  # 验证变量是否正确定义

        # 2、初始化历史缓存 + 打印（补全 seq_ids 相关代码）
        if self.history_bev is None:
            self.history_bev = curr_bev.repeat(1, self.history_num, 1, 1, 1).detach()
            self.history_vis = torch.zeros(bs, self.history_num, z, h, w, device=device).detach()
            self.history_forward_augs = forward_augs.clone().detach()
            self.history_seq_ids = seq_ids.clone().detach()  # 之前可能因 seq_ids 未定义导致注释，现在补回
            self.history_sweep_time = curr_bev.new_zeros(bs, self.history_num).detach()
            self.history_cam_intrins = curr_cam_intrins.unsqueeze(1).repeat(1, self.history_num, 1, 1, 1).detach()
            self.history_cam_extrins = curr_cam_extrins.unsqueeze(1).repeat(1, self.history_num, 1, 1, 1).detach()
            
            print(f"\n=== 历史缓存初始化 ===")
            print(f"history_bev.shape: {self.history_bev.shape} → (bs, history_num*c_, z, h, w)")
            print(f"history_cam_intrins.shape: {self.history_cam_intrins.shape} → (bs, history_num, num_cams, 4, 4)")

        # 3、处理新序列（补全 start_of_sequence 相关逻辑，避免新序列时变量异常）
        self.history_sweep_time += 1
        if start_of_sequence.sum() > 0:
            # 重置新序列的BEV缓存
            self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_num, 1, 1, 1).detach()
            self.history_vis[start_of_sequence] = 0.0
            self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence].detach()
            self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence].detach()
            self.history_sweep_time[start_of_sequence] = 0
            # 重置新序列的相机参数缓存
            self.history_cam_intrins[start_of_sequence] = curr_cam_intrins[start_of_sequence].unsqueeze(1).repeat(1, self.history_num, 1, 1, 1).detach()
            self.history_cam_extrins[start_of_sequence] = curr_cam_extrins[start_of_sequence].unsqueeze(1).repeat(1, self.history_num, 1, 1, 1).detach()
        print(f"\n=== 新序列处理 ===")
        print(f"start_of_sequence.sum(): {start_of_sequence.sum()} → 0=无新序列，>0=有新序列需重置缓存")

        # 4、生成体素网格 + 打印（不变）
        grid_3d = self.generate_grid(curr_bev)
        feat2bev = self.generate_feat2bev(grid_3d, dx, bx)
        print(f"\n=== 体素网格 ===")
        print(f"grid_3d.shape: {grid_3d.shape} → (bs, h, w, z, 3)")

        # 5、历史特征坐标对齐 + 打印（现在 curr_to_prev_ego_rt 已定义，可正常使用）
        rt_flow = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_ego_rt @ torch.inverse(forward_augs) @ feat2bev)
        grid = rt_flow.view(bs, 1, 1, 1, 4, 4) @ torch.cat([grid_3d, torch.ones_like(grid_3d[..., :1])], dim=-1).unsqueeze(-1)
        normalize_factor = torch.tensor([w - 1.0, h - 1.0, z - 1.0], dtype=curr_bev.dtype, device=device)
        grid_sampler = grid[..., :3, 0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0
        print(f"\n=== 历史特征对齐 ===")
        print(f"grid_sampler.shape: {grid_sampler.shape} → (bs, h, w, z, 3)")

        # 6、采样历史BEV特征 + 打印
        mc = self.history_bev.shape[1]
        sampled_history_bev = F.grid_sample(
            self.history_bev.reshape(bs, mc, z, h, w),
            grid_sampler.permute(0, 3, 1, 2, 4),
            align_corners=True,
            mode='bilinear'
        )
        print(f"\n=== 历史特征采样 ===")
        print(f"mc (历史总通道): {mc}")
        print(f"sampled_history_bev.shape: {sampled_history_bev.shape} → (bs, mc, z, h, w)")

        # 7、可见性概率 + 打印
        # V_curr = self.compute_visibility(grid_3d, curr_cam_intrins, curr_cam_extrins, self.img_shape)

        # 新增：动态计算体素尺寸（voxel_size）
        # 步骤1：从配置中获取基础体素尺寸（x/y/z方向的step）
        # 注意：实际项目中可从模型配置中读取，这里简化为直接解析配置
        base_voxel_x = 0.8  # grid_config['x'][2]
        base_voxel_y = 0.8  # grid_config['y'][2]
        base_voxel_z = 0.8  # grid_config['z'][2]

        # 步骤2：确定当前尺度倍数（根据curr_bev的空间尺寸推断）
        # 基础网格尺寸（1x尺度）：x/y方向各 (40 - (-40))/0.8 = 100体素（对应配置中的bev_h_=100, bev_w_=100）
        bs, c_, z, h, w = curr_bev.shape
        base_grid_h = 100  # 1x尺度下的h尺寸（配置中的bev_h_=100）
        scale_factor = base_grid_h / h  # 尺度倍数 = 基础尺寸 / 当前尺寸（如h=50 → 100/50=2 → 1/2尺度）
        scale_factor = round(scale_factor)  # 确保是整数（1、2、4）
        print(f"当前尺度倍数：{scale_factor}x（基础网格h={base_grid_h}，当前h={h}）")

        # 步骤3：计算当前体素尺寸（基础尺寸 × 尺度倍数）
        curr_voxel_x = base_voxel_x * scale_factor
        curr_voxel_y = base_voxel_y * scale_factor
        curr_voxel_z = base_voxel_z * scale_factor  # z方向通常与x/y同尺度变化

        # 步骤4：转换为单元素张量（避免原报错）
        voxel_size = torch.tensor([curr_voxel_x, curr_voxel_y, curr_voxel_z], device=device)
        print(f"动态计算的体素尺寸：voxel_size={voxel_size}")

        # 步骤5：更新dx和bx（与当前体素尺寸保持一致）
        dx = curr_voxel_x
        bx = curr_voxel_z

        # 7、可见性概率计算（使用动态体素尺寸）
        V_curr = self.rt_vis_calculator(
            grid=grid_3d,
            cam_intrins=curr_cam_intrins,
            cam_extrins=curr_cam_extrins,
            img_shape=self.img_shape,
            nonempty_prob=nonempty_prob,
            # 动态计算体素网格范围（基础范围不变，与体素尺寸无关）
            voxel_min=torch.tensor([-40.0, -40.0, -1.0], device=device),  # grid_config['x'][0], grid_config['y'][0], grid_config['z'][0]
            voxel_max=torch.tensor([40.0, 40.0, 5.4], device=device),     # grid_config['x'][1], grid_config['y'][1], grid_config['z'][1]
            voxel_size=voxel_size  # 使用动态计算的体素尺寸
        )
        # visualizer = VisibilityVisualizer()
        # # 可视化当前帧可见性（V_curr），保存到工作目录
        # visualizer.plot_visibility(V_curr, save_path="curr_frame_visibility.png")
        prev_cam_intrins = self.history_cam_intrins[:, -1, :, :, :]
        prev_cam_extrins = self.history_cam_extrins[:, -1, :, :, :]
        V_prev = self.compute_visibility(grid_3d, prev_cam_intrins, prev_cam_extrins, self.img_shape)
        print(f"\n=== 可见性概率 ===")
        print(f"V_curr.shape: {V_curr.shape} → (bs, h, w, z)")
        print(f"V_prev.shape: {V_prev.shape} → (bs, h, w, z)")

        # 8、稀疏采样 + 打印
        curr_bev_flat = curr_bev.reshape(bs, c_, -1)
        history_bev_flat = sampled_history_bev.reshape(bs, mc, -1)
        V_prev_flat = V_prev.reshape(bs, -1)
        V_curr_flat = V_curr.reshape(bs, -1)
        nonempty_prob_flat = nonempty_prob.reshape(bs, -1)
        total_voxels = nonempty_prob_flat.shape[1]
        fg_indices = torch.topk(nonempty_prob_flat, self.top_k, dim=1)[1]
        bg_indices = torch.topk(1 - nonempty_prob_flat, total_voxels - self.top_k, dim=1)[1]
        print(f"\n=== 稀疏采样 ===")
        print(f"curr_bev_flat.shape: {curr_bev_flat.shape} → (bs, c_, total_voxels)")
        print(f"history_bev_flat.shape: {history_bev_flat.shape} → (bs, mc, total_voxels)")
        print(f"fg_indices.shape: {fg_indices.shape} → (bs, top_k)")
        print(f"bg_indices.shape: {bg_indices.shape} → (bs, total_voxels - top_k)")



        # 9、门控融合 + 打印（重点：修复维度不匹配）
        # 9.1 前景融合
        fg_history_feat = torch.gather(history_bev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, mc, 1))  # [3, 384, 500]
        fg_curr_feat = torch.gather(curr_bev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, c_, 1))        # [3, 96, 500]
        fg_V_prev = torch.gather(V_prev_flat, dim=1, index=fg_indices)  # [3, 500]
        fg_V_curr = torch.gather(V_curr_flat, dim=1, index=fg_indices)  # [3, 500]
        print(f"\n=== 前景特征 ===")
        print(f"fg_history_feat.shape: {fg_history_feat.shape} → (bs, mc, top_k) = (3, 384, 500)")
        print(f"fg_curr_feat.shape: {fg_curr_feat.shape} → (bs, c_, top_k) = (3, 96, 500)")

        # 核心修改1：历史特征拆分为“时间维度×单帧通道”
        # mc = history_num * c_ → 拆分为 (bs, history_num, c_, top_k)
        fg_history_feat_time = fg_history_feat.reshape(bs, self.history_num, c_, self.top_k)  # [3, 4, 96, 500]
        print(f"fg_history_feat_time.shape (拆分后): {fg_history_feat_time.shape} → (bs, history_num, c_, top_k)")

        # 核心修改2：时间加权聚合（越新的历史帧权重越高，压缩到单帧通道）
        # 生成时间衰减权重（如指数衰减，权重随时间步递减）
        time_weights = torch.exp(-0.5 * torch.arange(self.history_num, device=device)).view(1, self.history_num, 1, 1)  # [1, 4, 1, 1]
        print(f"time_weights: {time_weights.squeeze()} → 时间衰减权重")
        # 加权求和：按时间维度聚合，去掉时间维度 → 通道数变为96
        fg_history_feat_agg = (fg_history_feat_time * time_weights).sum(dim=1)  # [3, 96, 500]（与当前特征通道一致）
        print(f"fg_history_feat_agg.shape (聚合后): {fg_history_feat_agg.shape} → (bs, c_, top_k) = (3, 96, 500)")

        # 核心修改3：门控权重计算（历史可见性也需时间聚合，与特征权重一致）
        # 历史可见性扩展时间维度后聚合
        fg_V_prev_time = fg_V_prev.unsqueeze(1).repeat(1, self.history_num, 1)  # [3, 4, 500]
        fg_V_prev_agg = (fg_V_prev_time * time_weights.squeeze(-1)).sum(dim=1)  # [3, 500]（聚合后历史可见性）
        fg_w_hist, fg_w_curr = self.compute_gate_weights(fg_V_prev_agg, fg_V_curr)  # [3, 500, 1]
        print(f"fg_w_hist.shape: {fg_w_hist.shape} → (bs, top_k, 1)")

        # 后续步骤：融合前维度验证（现在通道已匹配）
        last_occ_pred_flat = last_occ_pred.permute(0, 3, 2, 1, 4).reshape(bs, -1, 18)
        fg_occ_embed = self.occ_embedding(torch.gather(last_occ_pred_flat, dim=1, index=fg_indices.unsqueeze(-1).repeat(1, 1, 18)))  # [3, 500, 32]
        fg_occ_embed = fg_occ_embed.permute(0, 2, 1)  # [3, 32, 500]
        print(f"fg_occ_embed.shape: {fg_occ_embed.shape} → (bs, occ_embedims, top_k)")

        # 融合前维度检查（关键：现在两者最后一维均为96）
        print(f"融合前维度检查（修复后）:")
        fg_history_agg_perm = fg_history_feat_agg.permute(0, 2, 1)  # [3, 500, 96]
        fg_curr_perm = fg_curr_feat.permute(0, 2, 1)                # [3, 500, 96]
        print(f"fg_history_agg_perm.shape: {fg_history_agg_perm.shape} → (bs, top_k, c_)")
        print(f"fg_curr_perm.shape: {fg_curr_perm.shape} → (bs, top_k, c_)")

        # 现在维度匹配，可正常融合
        fg_fused = (fg_w_hist * fg_history_agg_perm + fg_w_curr * fg_curr_perm)  # [3, 500, 96]
        fg_coeff = self.gate_params['eta'] * torch.min(fg_V_prev_agg, fg_V_curr).unsqueeze(-1)  # [3, 500, 1]
        fg_fused += fg_coeff * (fg_history_agg_perm * fg_curr_perm)  # 元素乘法交互（维度匹配）
        fg_fused = torch.cat([fg_fused, fg_occ_embed.permute(0, 2, 1)], dim=-1)  # [3, 500, 96+32=128]
        fg_fused = self.history_fusion_linear(fg_fused).permute(0, 2, 1)  # [3, 96, 500]（还原通道维度）
        print(f"fg_fused.shape (融合后): {fg_fused.shape} → (bs, c_, top_k) = (3, 96, 500)")

        # 9.2 背景融合（同理修复：历史背景特征通道192→96）
        bg_history_feat = torch.gather(history_bev_flat[:, :mc//2], dim=2, index=bg_indices.unsqueeze(1).repeat(1, mc//2, 1))  # [3, 192, 750]
        bg_curr_feat = torch.gather(curr_bev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, c_, 1))  # [3, 96, 750]
        print(f"\n=== 背景特征 ===")
        print(f"bg_history_feat.shape: {bg_history_feat.shape} → (bs, mc//2, bg_k) = (3, 192, 750)")

        # 背景历史特征拆分+聚合（mc//2 = 4×48 → 拆分为4帧，每帧48通道，聚合后pad到96）
        bg_history_feat_time = bg_history_feat.reshape(bs, self.history_num, c_//2, -1)  # [3, 4, 48, 750]
        bg_history_feat_agg = (bg_history_feat_time * time_weights).sum(dim=1)  # [3, 48, 750]
        # pad到96通道（与当前背景特征匹配）
        bg_history_agg_perm = F.pad(
            bg_history_feat_agg.permute(0, 2, 1),  # [3, 750, 48]
            (0, c_ - c_//2, 0, 0, 0, 0)  # 右侧pad 48个0 → [3, 750, 96]
        )
        bg_curr_perm = bg_curr_feat.permute(0, 2, 1)  # [3, 750, 96]

        # 背景门控权重（历史可见性聚合）
        bg_V_prev = torch.gather(V_prev_flat, dim=1, index=bg_indices)  # [3, 750]
        bg_V_prev_time = bg_V_prev.unsqueeze(1).repeat(1, self.history_num, 1)  # [3, 4, 750]
        bg_V_prev_agg = (bg_V_prev_time * time_weights.squeeze(-1)).sum(dim=1)  # [3, 750]
        bg_w_hist, bg_w_curr = self.compute_gate_weights(bg_V_prev_agg, torch.gather(V_curr_flat, dim=1, index=bg_indices))  # [3, 750, 1]

        # 背景融合（维度已匹配）
        bg_fused = (bg_w_hist * bg_history_agg_perm + bg_w_curr * bg_curr_perm)  # [3, 750, 96]
        bg_occ_embed = self.occ_embedding(torch.gather(last_occ_pred_flat, dim=1, index=bg_indices.unsqueeze(-1).repeat(1, 1, 18)))  # [3, 750, 32]
        bg_fused = torch.cat([bg_fused, bg_occ_embed], dim=-1)  # [3, 750, 128]
        bg_fused = self.history_fusion_bg_linear(bg_fused).permute(0, 2, 1)  # [3, 96, 750]
        print(f"bg_fused.shape (融合后): {bg_fused.shape} → (bs, c_, bg_k) = (3, 96, 750)")

        # 10、更新当前BEV特征 + 打印
        curr_bev_updated = curr_bev_flat.clone()
        curr_bev_updated.scatter_add_(dim=2, index=fg_indices.unsqueeze(1).repeat(1, c_, 1), src=fg_fused)
        curr_bev_updated.scatter_add_(dim=2, index=bg_indices.unsqueeze(1).repeat(1, c_, 1), src=bg_fused)
        curr_bev_updated = curr_bev_updated.reshape(bs, c_, z, h, w)
        print(f"\n=== 最终输出 ===")
        print(f"curr_bev_updated.shape: {curr_bev_updated.shape} → (bs, c_, z, h, w)")

        # 11、更新历史缓存（略）

        return curr_bev_updated.clone()



class VisibilityVisualizer:
    def __init__(self, figsize=(15, 5)):
        self.figsize = figsize  # 画布大小
        self.cmap = 'viridis'   # 颜色映射（蓝→黄：低→高可见性）

    def plot_visibility(self, vis_prob, save_path=None):
        """
        可视化体素可见性分布
        Args:
            vis_prob: torch.Tensor → [bs, h, w, z]，compute_visibility的输出
            save_path: str → 图片保存路径（如"visibility_plot.png"，None则不保存）
        """
        # 1. 数据预处理：取第1个样本（bs=0），转为numpy数组
        vis_np = vis_prob[0].detach().cpu().numpy()  # [h, w, z]
        h, w, z = vis_np.shape
        print(f"可视化数据维度：h={h}, w={w}, z={z}（z为高度层数）")

        # 2. 创建子图：按z层排列（1行z列，或自动调整行列）
        cols = min(z, 5)  # 每行最多显示5个z层，避免画面过宽
        rows = (z + cols - 1) // cols  # 计算需要的行数
        fig, axes = plt.subplots(rows, cols, figsize=self.figsize)
        axes = axes.flatten()  # 展平axes，方便遍历

        # 3. 遍历每个z层，绘制热力图
        for z_idx in range(z):
            ax = axes[z_idx]
            # 提取当前z层的可见性数据（[h, w]）
            vis_z = vis_np[..., z_idx]
            # 绘制热力图
            im = ax.imshow(vis_z, cmap=self.cmap, vmin=0, vmax=1)
            # 设置标题（显示当前z层）
            ax.set_title(f'Visibility - Z Layer: {z_idx}', fontsize=10)
            # 隐藏坐标轴刻度（非必需，根据需求调整）
            ax.set_xticks([])
            ax.set_yticks([])

        # 4. 隐藏多余的子图（若z不能被cols整除）
        for z_idx in range(z, len(axes)):
            axes[z_idx].set_visible(False)

        # 5. 添加颜色条（统一标注可见性概率）
        cbar = fig.colorbar(im, ax=axes[:z], shrink=0.6, aspect=20)
        cbar.set_label('Visibility Probability (0=invisible, 1=visible)', fontsize=12)

        # 6. 调整布局，避免重叠
        plt.tight_layout()

        # 7. 保存或显示图片
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visibility plot saved to: {save_path}")
        plt.show()


class EfficientRayTracingVisibility:
    def __init__(self, nonempty_thresh=0.1, max_step_ratio=1.2):
        self.nonempty_thresh = nonempty_thresh  # 稀疏采样阈值
        self.max_step_ratio = max_step_ratio    # 最大步数系数（对角线的1.2倍）

    def build_voxel_hash(self, voxel_coords, nonempty_prob):
        """
        构建体素坐标→非空概率的哈希表（空间索引）
        Args:
            voxel_coords: [N, 3] 所有稀疏体素的坐标（x, y, z，自车坐标系）
            nonempty_prob: [N] 对应体素的非空概率
        Returns:
            hash_table: dict，键为 tuple(x,y,z)，值为非空概率
        """
        hash_table = defaultdict(float)
        coords_np = voxel_coords.cpu().numpy()
        prob_np = nonempty_prob.cpu().numpy()
        for coord, prob in zip(coords_np, prob_np):
            hash_table[(int(coord[0]), int(coord[1]), int(coord[2]))] = prob
        return hash_table

    def batch_ray_voxel_intersection(self, ray_origins, ray_dirs, voxel_min, voxel_max, voxel_size, max_steps):
        """
        批量计算射线经过的体素（向量化实现）
        Args:
            ray_origins: [B, 3] 射线起点（自车坐标系，B为稀疏体素数量）
            ray_dirs: [B, 3] 射线方向（单位向量，自车坐标系）
            voxel_min/max: [3] 体素网格范围
            voxel_size: [3] 体素尺寸
            max_steps: int 最大步进数
        Returns:
            all_voxels: [B, max_steps, 3] 每条射线经过的体素坐标（填充-1表示无效）
        """
        B = ray_origins.shape[0]
        device = ray_origins.device

        # 1. 计算射线与体素网格的有效t范围（向量化）
        t_min = torch.zeros(B, device=device)
        t_max = torch.full((B,), 1e6, device=device)
        for i in range(3):
            dir_i = ray_dirs[:, i]
            mask = dir_i != 0  # 排除方向为0的射线
            t1 = (voxel_min[i] - ray_origins[mask, i]) / dir_i[mask]
            t2 = (voxel_max[i] - ray_origins[mask, i]) / dir_i[mask]
            t_min[mask] = torch.max(t_min[mask], torch.min(t1, t2))
            t_max[mask] = torch.min(t_max[mask], torch.max(t1, t2))
        valid_ray_mask = t_min < t_max  # [B] 有效射线掩码

        # 2. 初始化步进参数
        current_t = t_min.clone()
        all_voxels = torch.full((B, max_steps, 3), -1, dtype=torch.int32, device=device)  # 填充-1表示无效

        # 3. 批量步进计算体素（循环max_steps次，而非逐个体素）
        for step in range(max_steps):
            # 计算当前t对应的空间坐标
            pos = ray_origins + current_t.unsqueeze(1) * ray_dirs  # [B, 3]
            # 转换为体素坐标（仅对有效射线计算）
            voxel = torch.round((pos[valid_ray_mask] - voxel_min) / voxel_size).int()  # [V, 3]，V为当前有效射线数
            all_voxels[valid_ray_mask, step] = voxel
            # 更新下一次t（沿最快到达体素边界的轴步进）
            for i in range(3):
                dir_i = ray_dirs[:, i]
                mask = (dir_i != 0) & valid_ray_mask
                next_t = current_t[mask] + (voxel_size[i] / torch.abs(dir_i[mask]))
                current_t[mask] = torch.min(current_t[mask], next_t)
            # 更新有效射线掩码（t未超出max且未离开体素网格）
            valid_ray_mask = valid_ray_mask & (current_t < t_max) & (step < max_steps - 1)
            if not valid_ray_mask.any():
                break  # 所有射线均无效，提前退出

        return all_voxels

    def __call__(self, grid, cam_intrins, cam_extrins, img_shape, nonempty_prob, voxel_min, voxel_max, voxel_size):
        """
        高效光线追踪可见性计算（集成四大优化）
        Args:
            grid: [bs, h, w, z, 3] 体素中心坐标（自车坐标系）
            cam_intrins: [bs, num_cams, 4, 4] 相机内参
            cam_extrins: [bs, num_cams, 4, 4] 相机外参（自车→相机）
            img_shape: [2] 图像尺寸 (h_img, w_img)
            nonempty_prob: [bs, z, h, w] 非空概率
            voxel_min/max: [3] 体素网格范围
            voxel_size: [3] 体素尺寸 (dx, dy, dz)
        Returns:
            rt_vis: [bs, h, w, z] 可见性概率（0~1）
        """
        bs, h, w, z, _ = grid.shape
        num_cams = cam_intrins.shape[1]
        device = grid.device
        rt_vis = torch.zeros(bs, h, w, z, device=device)

        # 计算最大步数（体素网格对角线长度 / 最小体素尺寸 * 系数）
        diag_length = torch.norm(voxel_max - voxel_min)  # 网格对角线长度
        min_voxel_size = torch.min(voxel_size)
        max_steps = int(diag_length / min_voxel_size * self.max_step_ratio) + 1
        print(f"光线追踪参数：max_steps={max_steps}, 体素尺寸={voxel_size}")

        for b in range(bs):
            # ------------------- 优化1：稀疏采样 -------------------
            # 1.1 获取非空体素的索引和坐标
            nonempty_mask = nonempty_prob[b] > self.nonempty_thresh  # [z, h, w]
            if not nonempty_mask.any():
                continue  # 无有效体素，跳过
            z_idx, h_idx, w_idx = torch.where(nonempty_mask)  # 非空体素的索引
            sparse_voxels = grid[b, h_idx, w_idx, z_idx]  # [N, 3]，N为稀疏体素数量
            
            # 新增：定义N为稀疏体素的数量（关键修复！）
            N = sparse_voxels.shape[0]  # N = 稀疏体素的数量
            if N == 0:
                continue  # 避免空体素导致的后续错误
            
            sparse_nonempty = nonempty_prob[b, z_idx, h_idx, w_idx]  # [N]

            # 1.2 构建空间哈希表
            voxel_hash = self.build_voxel_hash(sparse_voxels, sparse_nonempty)

            # ------------------- 优化2：批量处理所有稀疏体素的射线 -------------------
            for cam_idx in range(num_cams):
                # 2.1 相机参数（外参、内参、光心）
                extrin = cam_extrins[b, cam_idx]  # [4,4] 自车→相机
                intrin = cam_intrins[b, cam_idx]  # [4,4]
                cam_center_ego = torch.inverse(extrin)[:3, 3]  # 相机光心（自车坐标系）

                # 2.2 批量生成射线（此时N已定义，可正常使用）
                ray_origins = cam_center_ego.unsqueeze(0).repeat(N, 1)  # [N, 3]
                ray_dirs = sparse_voxels - cam_center_ego.unsqueeze(0)  # [N, 3]
                ray_dirs = F.normalize(ray_dirs, dim=1)  # 单位化方向向量

                # 2.3 批量过滤视场外体素（提前排除无需计算的射线）
                # 转换体素到相机坐标系
                sparse_voxels_hom = torch.cat([sparse_voxels, torch.ones(N, 1, device=device)], dim=1)  # [N,4]
                voxels_cam = (extrin @ sparse_voxels_hom.T).T[:, :3]  # [N,3]（相机坐标系）
                # 深度>0且像素在图像内
                depth_valid = voxels_cam[:, 2] > 0
                img_coord = (intrin[:3, :3] @ voxels_cam.T).T[:, :2] / voxels_cam[:, 2:3]  # [N,2] (u,v)
                img_valid = (img_coord[:, 0] >= 0) & (img_coord[:, 0] < img_shape[1]) & \
                            (img_coord[:, 1] >= 0) & (img_coord[:, 1] < img_shape[0])
                valid_mask = depth_valid & img_valid  # [N] 有效体素掩码
                if not valid_mask.any():
                    continue
                # 筛选有效射线
                valid_rays = valid_mask.nonzero().squeeze(1)
                valid_origins = ray_origins[valid_rays]
                valid_dirs = ray_dirs[valid_rays]
                valid_voxels = sparse_voxels[valid_rays]  # [M,3]，M为有效体素数量
                valid_indices = (z_idx[valid_rays], h_idx[valid_rays], w_idx[valid_rays])  # 原始网格索引

                # 2.4 批量计算射线经过的体素（优化2：向量化；优化4：限制max_steps）
                ray_voxels = self.batch_ray_voxel_intersection(
                    ray_origins=valid_origins,
                    ray_dirs=valid_dirs,
                    voxel_min=voxel_min,
                    voxel_max=voxel_max,
                    voxel_size=voxel_size,
                    max_steps=max_steps
                )  # [M, max_steps, 3]

                # 2.5 批量检测遮挡（利用哈希表快速查询）
                # 转换目标体素为坐标键
                target_coords = [tuple(map(int, coord.cpu().numpy())) for coord in valid_voxels]
                # 遍历每条射线的体素
                visible_mask = torch.ones(M, dtype=torch.bool, device=device)
                for m in range(M):
                    target_key = target_coords[m]
                    # 检查射线经过的体素是否有遮挡物
                    for step in range(max_steps):
                        voxel = ray_voxels[m, step]
                        if voxel[0] == -1:
                            break  # 射线已离开网格
                        voxel_key = (voxel[0].item(), voxel[1].item(), voxel[2].item())
                        if voxel_key == target_key:
                            continue  # 跳过目标体素
                        # 哈希表查询非空概率（优化3：O(1)查询）
                        if voxel_hash.get(voxel_key, 0.0) > self.nonempty_thresh:
                            visible_mask[m] = False
                            break  # 被遮挡，跳出循环

                # 2.6 更新可见性（多相机取最大）
                visible_values = sparse_nonempty[valid_rays][visible_mask]  # 可见体素的非空概率
                z_idx_vis, h_idx_vis, w_idx_vis = (
                    valid_indices[0][visible_mask],
                    valid_indices[1][visible_mask],
                    valid_indices[2][visible_mask]
                )
                rt_vis[b, h_idx_vis, w_idx_vis, z_idx_vis] = torch.max(
                    rt_vis[b, h_idx_vis, w_idx_vis, z_idx_vis],
                    visible_values
                )

        # 归一化到[0,1]
        return torch.sigmoid(5 * (rt_vis - 0.5))


@HEADS.register_module()
class GatedTemporalFusion2(BaseModule):
    def __init__(
        self,
        history_num=4,
        top_k=500,
        single_bev_num_channels=96,
        occ_embedims=32,
        num_classes=18,
        # 可见性门控参数
        vis_theta=0.3,
        vis_beta=10.0,
        vis_gamma=0.3,
        vis_sigma=0.1,
        # 射线追踪参数
        nonempty_thresh=0.1,
        max_step_ratio=1.2,
        **kwargs
    ):
        super().__init__(** kwargs)
        # 基础参数
        self.history_num = history_num
        self.top_k = top_k
        self.single_bev_num_channels = single_bev_num_channels
        self.occ_embedims = occ_embedims
        
        # 可见性门控参数
        self.vis_theta = vis_theta  # 可见性阈值
        self.vis_beta = vis_beta    # 敏感度参数
        self.vis_gamma = vis_gamma  # 场景4固定权重
        self.vis_sigma = vis_sigma  # 软化参数
        
        # 历史特征缓存
        self.history_bev = None
        self.history_cam_intrins = None
        self.history_cam_extrins = None
        
        # 网络层
        self.occ_embedding = nn.Sequential(
            nn.Linear(num_classes, occ_embedims),
            nn.Softplus(),
            nn.Linear(occ_embedims, occ_embedims),
        )
        self.history_fusion_linear = nn.Sequential(
            nn.Linear(single_bev_num_channels + occ_embedims, single_bev_num_channels),
            nn.Softplus(),
            nn.Linear(single_bev_num_channels, single_bev_num_channels),
        )
        self.history_fusion_bg_linear = nn.Sequential(
            nn.Linear(single_bev_num_channels + occ_embedims, single_bev_num_channels),
            nn.Softplus(),
            nn.Linear(single_bev_num_channels, single_bev_num_channels),
        )
        
        # 可见性计算组件
        self.rt_vis_calculator = EfficientRayTracingVisibility(
            nonempty_thresh=nonempty_thresh,
            max_step_ratio=max_step_ratio
        )
        self.img_shape = (900, 1600)  # 默认图像尺寸，可根据实际数据调整

    def compute_alpha_unified(self, V_curr, V_prev):
        """统一计算当前帧融合权重α（覆盖四场景）"""
        eps = 1e-8
        # 1. 基础动态权重σ_base（场景1-3）
        ratio = V_curr / (V_curr + V_prev + eps)
        sigma_base = torch.sigmoid(self.vis_beta * (ratio - 0.5))
        
        # 2. 场景4软化掩码σ_both
        mask_curr = torch.sigmoid(-(V_curr - self.vis_theta) / self.vis_sigma)
        mask_prev = torch.sigmoid(-(V_prev - self.vis_theta) / self.vis_sigma)
        sigma_both = mask_curr * mask_prev
        
        # 3. 最终权重计算
        alpha = (1 - sigma_both) * sigma_base + sigma_both * self.vis_gamma
        return alpha.unsqueeze(-1)  # [bs, N, 1]

    def compute_gate_weights(self, V_prev_agg, V_curr):
        """计算历史和当前帧的门控权重"""
        alpha = self.compute_alpha_unified(V_curr, V_prev_agg)
        return 1 - alpha, alpha  # 历史权重，当前权重

    def generate_grid(self, curr_bev, voxel_min, voxel_max, voxel_size):
        """生成体素中心坐标网格（自车坐标系）"""
        bs, c_, z, h, w = curr_bev.shape
        device = curr_bev.device
        
        # 计算体素中心坐标
        x_coords = torch.linspace(
            voxel_min[0] + voxel_size[0]/2, 
            voxel_max[0] - voxel_size[0]/2, 
            w, device=device
        )
        y_coords = torch.linspace(
            voxel_min[1] + voxel_size[1]/2, 
            voxel_max[1] - voxel_size[1]/2, 
            h, device=device
        )
        z_coords = torch.linspace(
            voxel_min[2] + voxel_size[2]/2, 
            voxel_max[2] - voxel_size[2]/2, 
            z, device=device
        )
        
        # 生成网格并扩展维度 [bs, h, w, z, 3]
        x_grid, y_grid, z_grid = torch.meshgrid(x_coords, y_coords, z_coords, indexing='xy')
        grid = torch.stack([x_grid, y_grid, z_grid], dim=-1)  # [w, h, z, 3]
        grid = grid.permute(1, 0, 2, 3)  # [h, w, z, 3]
        return grid.unsqueeze(0).repeat(bs, 1, 1, 1, 1)  # [bs, h, w, z, 3]

    @force_fp32()
    def forward(
        self,
        curr_bev,
        cam_intrins,
        cam_extrins,
        dx,
        bx,
        nonempty_prob,
        last_occ_pred,
        history_fusion_params
    ):
        """
        Args:
            curr_bev: [bs, c, z, h, w] 当前帧BEV特征
            cam_intrins: [bs, num_cams, 4, 4] 相机内参
            cam_extrins: [bs, num_cams, 4, 4] 相机外参
            dx: 体素尺寸
            bx: 体素偏移
            nonempty_prob: [bs, z, h, w] 体素非空概率
            last_occ_pred: [bs, z, h, w, num_classes] 上一时刻 occupancy 预测
            history_fusion_params: 历史融合参数（包含序列信息等）
        Returns:
            curr_bev_updated: [bs, c, z, h, w] 融合后BEV特征
        """
        bs, c_, z, h, w = curr_bev.shape
        device = curr_bev.device
        mc = self.history_num * c_  # 历史特征总通道数

        # 1. 初始化历史缓存
        if self.history_bev is None:
            self.history_bev = curr_bev.repeat(1, self.history_num, 1, 1, 1)  # [bs, mc, z, h, w]
            self.history_cam_intrins = cam_intrins.unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)
            self.history_cam_extrins = cam_extrins.unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)

        # 2. 处理新序列（根据序列索引重置历史）
        start_of_sequence = history_fusion_params.get('start_of_sequence', torch.zeros(bs, dtype=torch.bool, device=device))
        if start_of_sequence.any():
            self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_num, 1, 1, 1)
            self.history_cam_intrins[start_of_sequence] = cam_intrins[start_of_sequence].unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)
            self.history_cam_extrins[start_of_sequence] = cam_extrins[start_of_sequence].unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)

        # 3. 动态计算体素参数
        voxel_size = dx  # [3] 体素尺寸 (x, y, z)
        voxel_min = torch.tensor([-40.0, -40.0, -1.0], device=device)  # 体素网格最小值
        voxel_max = torch.tensor([40.0, 40.0, 5.4], device=device)    # 体素网格最大值

        # 4. 生成体素坐标网格
        grid_3d = self.generate_grid(curr_bev, voxel_min, voxel_max, voxel_size)  # [bs, h, w, z, 3]

        # 5. 采样历史特征（通过运动补偿）
        # 这里简化实现，实际应包含网格采样逻辑（如使用F.grid_sample）
        sampled_history_bev = self.history_bev  # 假设已通过运动补偿采样

        # 6. 计算可见性概率
        V_curr = self.rt_vis_calculator(
            grid=grid_3d,
            cam_intrins=cam_intrins,
            cam_extrins=cam_extrins,
            img_shape=self.img_shape,
            nonempty_prob=nonempty_prob,
            voxel_min=voxel_min,
            voxel_max=voxel_max,
            voxel_size=voxel_size
        )  # [bs, h, w, z]

        # 历史可见性（取最近一帧历史的可见性）
        prev_cam_intrins = self.history_cam_intrins[:, -1]  # [bs, num_cams, 4, 4]
        prev_cam_extrins = self.history_cam_extrins[:, -1]  # [bs, num_cams, 4, 4]
        V_prev = self.rt_vis_calculator(
            grid=grid_3d,
            cam_intrins=prev_cam_intrins,
            cam_extrins=prev_cam_extrins,
            img_shape=self.img_shape,
            nonempty_prob=nonempty_prob,
            voxel_min=voxel_min,
            voxel_max=voxel_max,
            voxel_size=voxel_size
        )  # [bs, h, w, z]

        # 7. 稀疏采样（前景/背景分离）
        curr_bev_flat = curr_bev.reshape(bs, c_, -1)  # [bs, c_, h*w*z]
        history_bev_flat = sampled_history_bev.reshape(bs, mc, -1)  # [bs, mc, h*w*z]
        V_prev_flat = V_prev.reshape(bs, -1)  # [bs, h*w*z]
        V_curr_flat = V_curr.reshape(bs, -1)  # [bs, h*w*z]
        nonempty_prob_flat = nonempty_prob.reshape(bs, -1)  # [bs, h*w*z]
        total_voxels = nonempty_prob_flat.shape[1]

        # 前景/背景索引
        fg_indices = torch.topk(nonempty_prob_flat, self.top_k, dim=1)[1]  # [bs, top_k]
        bg_indices = torch.topk(1 - nonempty_prob_flat, total_voxels - self.top_k, dim=1)[1]  # [bs, total_voxels-top_k]

        # 8. 前景融合
        # 8.1 提取前景特征
        fg_history_feat = torch.gather(
            history_bev_flat, dim=2, 
            index=fg_indices.unsqueeze(1).repeat(1, mc, 1)
        )  # [bs, mc, top_k]
        fg_curr_feat = torch.gather(
            curr_bev_flat, dim=2, 
            index=fg_indices.unsqueeze(1).repeat(1, c_, 1)
        )  # [bs, c_, top_k]

        # 8.2 历史特征时间聚合
        fg_history_feat_time = fg_history_feat.reshape(bs, self.history_num, c_, self.top_k)  # [bs, T, c_, K]
        time_weights = torch.exp(-0.5 * torch.arange(self.history_num, device=device)).view(1, self.history_num, 1, 1)  # 时间衰减权重
        fg_history_agg = (fg_history_feat_time * time_weights).sum(dim=1)  # [bs, c_, K]

        # 8.3 可见性聚合与门控权重
        fg_V_prev = torch.gather(V_prev_flat, dim=1, index=fg_indices)  # [bs, K]
        fg_V_curr = torch.gather(V_curr_flat, dim=1, index=fg_indices)  # [bs, K]
        fg_V_prev_time = fg_V_prev.unsqueeze(1).repeat(1, self.history_num, 1)  # [bs, T, K]
        fg_V_prev_agg = (fg_V_prev_time * time_weights.squeeze(-1)).sum(dim=1)  # [bs, K]
        fg_w_hist, fg_w_curr = self.compute_gate_weights(fg_V_prev_agg, fg_V_curr)  # [bs, K, 1]

        # 8.4 前景特征融合
        fg_history_agg_perm = fg_history_agg.permute(0, 2, 1)  # [bs, K, c_]
        fg_curr_perm = fg_curr_feat.permute(0, 2, 1)  # [bs, K, c_]
        fg_fused = fg_w_hist * fg_history_agg_perm + fg_w_curr * fg_curr_perm  # [bs, K, c_]

        # 融合occupancy嵌入
        fg_occ_embed = self.occ_embedding(torch.gather(
            last_occ_pred.permute(0, 3, 2, 1, 4).reshape(bs, -1, last_occ_pred.shape[-1]),
            dim=1,
            index=fg_indices.unsqueeze(-1).repeat(1, 1, last_occ_pred.shape[-1])
        )).permute(0, 2, 1)  # [bs, occ_embedims, K]
        fg_fused = torch.cat([fg_fused, fg_occ_embed.permute(0, 2, 1)], dim=-1)  # [bs, K, c_+occ_embedims]
        fg_fused = self.history_fusion_linear(fg_fused).permute(0, 2, 1)  # [bs, c_, K]

        # 9. 背景融合（与前景逻辑类似）
        bg_history_feat = torch.gather(
            history_bev_flat[:, :mc//2], dim=2,
            index=bg_indices.unsqueeze(1).repeat(1, mc//2, 1)
        )  # [bs, mc//2, bg_k]
        bg_curr_feat = torch.gather(
            curr_bev_flat, dim=2,
            index=bg_indices.unsqueeze(1).repeat(1, c_, 1)
        )  # [bs, c_, bg_k]

        # 背景历史特征聚合（通道数调整）
        bg_history_feat_time = bg_history_feat.reshape(bs, self.history_num, c_//2, -1)  # [bs, T, c_//2, bg_k]
        bg_history_agg = (bg_history_feat_time * time_weights).sum(dim=1)  # [bs, c_//2, bg_k]
        bg_history_agg_perm = F.pad(
            bg_history_agg.permute(0, 2, 1),
            (0, c_ - c_//2, 0, 0)  # 补齐通道至c_
        )  # [bs, bg_k, c_]

        # 背景可见性与门控
        bg_V_prev = torch.gather(V_prev_flat, dim=1, index=bg_indices)  # [bs, bg_k]
        bg_V_curr = torch.gather(V_curr_flat, dim=1, index=bg_indices)  # [bs, bg_k]
        bg_V_prev_time = bg_V_prev.unsqueeze(1).repeat(1, self.history_num, 1)  # [bs, T, bg_k]
        bg_V_prev_agg = (bg_V_prev_time * time_weights.squeeze(-1)).sum(dim=1)  # [bs, bg_k]
        bg_w_hist, bg_w_curr = self.compute_gate_weights(bg_V_prev_agg, bg_V_curr)  # [bs, bg_k, 1]

        # 背景特征融合
        bg_curr_perm = bg_curr_feat.permute(0, 2, 1)  # [bs, bg_k, c_]
        bg_fused = bg_w_hist * bg_history_agg_perm + bg_w_curr * bg_curr_perm  # [bs, bg_k, c_]
        
        bg_occ_embed = self.occ_embedding(torch.gather(
            last_occ_pred.permute(0, 3, 2, 1, 4).reshape(bs, -1, last_occ_pred.shape[-1]),
            dim=1,
            index=bg_indices.unsqueeze(-1).repeat(1, 1, last_occ_pred.shape[-1])
        ))  # [bs, bg_k, occ_embedims]
        bg_fused = torch.cat([bg_fused, bg_occ_embed], dim=-1)  # [bs, bg_k, c_+occ_embedims]
        bg_fused = self.history_fusion_bg_linear(bg_fused).permute(0, 2, 1)  # [bs, c_, bg_k]

        # 10. 更新当前BEV特征
        curr_bev_updated = curr_bev_flat.clone()
        curr_bev_updated.scatter_add_(
            dim=2,
            index=fg_indices.unsqueeze(1).repeat(1, c_, 1),
            src=fg_fused
        )
        curr_bev_updated.scatter_add_(
            dim=2,
            index=bg_indices.unsqueeze(1).repeat(1, c_, 1),
            src=bg_fused
        )
        curr_bev_updated = curr_bev_updated.reshape(bs, c_, z, h, w)

        # 11. 更新历史缓存
        self.history_bev = torch.cat([
            self.history_bev[:, c_:, ...],  # 移除最早的历史帧
            curr_bev  # 添加当前帧作为最新历史
        ], dim=1)
        self.history_cam_intrins = torch.cat([
            self.history_cam_intrins[:, 1:, ...],
            cam_intrins.unsqueeze(1)
        ], dim=1)
        self.history_cam_extrins = torch.cat([
            self.history_cam_extrins[:, 1:, ...],
            cam_extrins.unsqueeze(1)
        ], dim=1)

        return curr_bev_updated




@HEADS.register_module()
class GatedTemporalFusion3(BaseModule):
    def __init__(
        self,
        history_num=4,
        depth_sampler_embed_dims=256,
        depth_sampler_num_heads=8,
        depth_sampler_num_levels=4,
        depth_sampler_num_points=4,
        im2col_step=64,
        top_k=500,
        single_bev_num_channels=96,
        occ_embedims=32,
        num_classes=18,
        vis_theta=0.3,
        vis_beta=10.0,
        vis_gamma=0.3,
        vis_sigma=0.1,
        nonempty_thresh=0.1,
        max_step_ratio=1.2,
        **kwargs
    ):
        super(GatedTemporalFusion3, self).__init__()

        # 基础参数
        self.history_num = history_num
        self.top_k = top_k
        self.single_bev_num_channels = single_bev_num_channels
        self.occ_embedims = occ_embedims
        
        # 可见性门控参数
        self.vis_theta = vis_theta  # 可见性阈值
        self.vis_beta = vis_beta    # 敏感度参数
        self.vis_gamma = vis_gamma  # 场景4固定权重
        self.vis_sigma = vis_sigma  # 软化参数
        
        # 历史特征缓存（新增history_last_bev）
        self.history_bev = None  # 缓存历史多帧BEV特征
        self.history_last_bev = None  # 缓存上一帧最终融合后的BEV特征（关键新增）
        self.history_cam_intrins = None
        self.history_cam_extrins = None
        
        # 网络层（保持不变）
        self.occ_embedding = nn.Sequential(
            nn.Linear(num_classes, occ_embedims),
            nn.Softplus(),
            nn.Linear(occ_embedims, occ_embedims),
        )
        self.history_fusion_linear = nn.Sequential(
            nn.Linear(single_bev_num_channels + occ_embedims, single_bev_num_channels),
            nn.Softplus(),
            nn.Linear(single_bev_num_channels, single_bev_num_channels),
        )
        self.history_fusion_bg_linear = nn.Sequential(
            nn.Linear(single_bev_num_channels + occ_embedims, single_bev_num_channels),
            nn.Softplus(),
            nn.Linear(single_bev_num_channels, single_bev_num_channels),
        )
        
        # 可见性计算组件（保持不变）
        self.rt_vis_calculator = EfficientRayTracingVisibility(
            nonempty_thresh=nonempty_thresh,
            max_step_ratio=max_step_ratio
        )
        self.img_shape = (900, 1600)  # 默认图像尺寸

        self.depth_sampler = DeformableDepthSampler(
            embed_dims=depth_sampler_embed_dims,
            num_heads=depth_sampler_num_heads,
            num_levels=depth_sampler_num_levels,
            num_points=depth_sampler_num_points
        )
        self.history_forward_augs = None  # 用于缓存历史帧的变换矩阵（如BDAM矩阵）
        self.im2col_step = im2col_step
        # self.voxel_encoder = nn.Linear(single_bev_num_channels, depth_sampler_embed_dims)
        self.dbound = [1.0, 45.0, 0.5]
        self.pc_range = [-40, -40, -1.0, 40, 40, 5.4]
        self.final_dim = (256, 704)
    def compute_visibility(self, grid, cam_intrins, cam_extrins, img_shape, img_feats, spatial_shapes):
        """
        升级：结合可变形注意力采样的深度值优化可见性计算
        Args:
            img_feats: 多尺度图像特征 [bs, num_levels, c, h, w]
            spatial_shapes: 图像特征尺度 [num_levels, 2] (h, w)
            其他参数同原函数
        Returns:
            vis_prob: [bs, h, w, z] 优化后的可见性概率
        """
        bs, h, w, z, _ = grid.shape
        h_img, w_img = img_shape
        device = grid.device
        num_voxels = h * w * z  # 体素总数

        # 1. 原有相机投影逻辑（计算图像坐标和初始可见性）
        # 1.1 体素坐标→相机坐标→图像坐标
        grid_cam = grid.unsqueeze(1).expand(bs, self.num_cams, h, w, z, 3)  # [bs, num_cams, h, w, z, 3]
        grid_flat = grid_cam.reshape(-1, num_voxels, 3)  # [bs*num_cams, N, 3]
        grid_hom = torch.cat([grid_flat, torch.ones_like(grid_flat[..., :1])], dim=-1)  # [bs*num_cams, N, 4]

        cam_intrins_flat = cam_intrins.reshape(-1, 4, 4)  # [bs*num_cams, 4, 4]
        cam_extrins_flat = cam_extrins.reshape(-1, 4, 4)  # [bs*num_cams, 4, 4]
        extrins_inv = torch.inverse(cam_extrins_flat)
        cam_coords = torch.bmm(extrins_inv[:, :3, :4], grid_hom.permute(0, 2, 1)).permute(0, 2, 1)  # [bs*num_cams, N, 3]
        depth = cam_coords[..., 2:3] + 1e-8  # 相机坐标系下的深度

        # 1.2 计算初始可见性（原逻辑）
        img_coords = torch.bmm(cam_intrins_flat[:, :3, :3], cam_coords.permute(0, 2, 1)).permute(0, 2, 1)  # [bs*num_cams, N, 3]
        img_xy = img_coords[..., :2] / depth  # [bs*num_cams, N, 2] (u, v)
        depth_valid = (cam_coords[..., 2] > 0).float()  # [bs*num_cams, N]
        u_valid = (img_xy[..., 0] >= 0) & (img_xy[..., 0] < w_img)
        v_valid = (img_xy[..., 1] >= 0) & (img_xy[..., 1] < h_img)
        img_valid = (u_valid & v_valid).float()  # [bs*num_cams, N]
        initial_vis = depth_valid * img_valid  # [bs*num_cams, N]

        # 2. 可变形注意力深度采样
        # 2.1 准备输入：体素特征编码
        # voxel_feat = self.voxel_encoder(self.curr_bev_feat)  # [bs, c_, z, h, w] → [bs, z*h*w, embed_dims]（需提前展平体素特征）
        # voxel_feat = voxel_feat.reshape(bs, num_voxels, -1)  # [bs, N, embed_dims]

        # 2.2 生成参考点（归一化到[0,1]）
        norm_img_xy = img_xy / torch.tensor([w_img, h_img], device=device).view(1, 1, 2)  # [bs*num_cams, N, 2]
        # 取主相机（如第0个相机）的参考点作为采样基准
        ref_points = norm_img_xy.reshape(bs, self.num_cams, num_voxels, 2)[:, 0]  # [bs, N, 2]
        ref_points = ref_points.unsqueeze(2).repeat(1, 1, self.depth_sampler.num_levels, 1)  # [bs, N, L, 2]

        # 2.3 采样深度特征
        sampled_depth = self.depth_sampler(
            query=voxel_feat,
            value=img_feats,  # 多尺度图像特征 [bs, L, c, h, w]
            reference_points=ref_points,
            spatial_shapes=spatial_shapes
        )  # [bs, N]

        # 3. 结合采样深度优化可见性
        # 3.1 深度一致性校验：采样深度与相机投影深度的差异
        cam_depth = depth.reshape(bs, self.num_cams, num_voxels)[:, 0]  # 主相机的投影深度 [bs, N]
        depth_diff = torch.abs(sampled_depth - cam_depth) / (cam_depth + 1e-8)  # 相对深度差
        depth_consistent = (depth_diff < 0.3).float()  # 深度差小于30%则认为有效

        # 3.2 融合可见性：初始可见性 × 深度一致性
        initial_vis = initial_vis.reshape(bs, self.num_cams, num_voxels).max(dim=1)[0]  # [bs, N]（多相机取max）
        vis_prob = initial_vis * depth_consistent  # [bs, N]

        # 4. 还原形状
        return vis_prob.reshape(bs, h, w, z)  # [bs, h, w, z]

    def compute_alpha_unified(self, V_curr, V_prev):
        """统一计算当前帧融合权重α（覆盖四场景）"""
        eps = 1e-8
        # 1. 基础动态权重σ_base（场景1-3）
        ratio = V_curr / (V_curr + V_prev + eps)
        sigma_base = torch.sigmoid(self.vis_beta * (ratio - 0.5))
        
        # 2. 场景4软化掩码σ_both
        mask_curr = torch.sigmoid(-(V_curr - self.vis_theta) / self.vis_sigma)
        mask_prev = torch.sigmoid(-(V_prev - self.vis_theta) / self.vis_sigma)
        sigma_both = mask_curr * mask_prev
        
        # 3. 最终权重计算
        alpha = (1 - sigma_both) * sigma_base + sigma_both * self.vis_gamma
        return alpha.unsqueeze(-1)  # [bs, N, 1]

    def compute_gate_weights(self, V_prev_agg, V_curr):
        """计算历史和当前帧的门控权重"""
        alpha = self.compute_alpha_unified(V_curr, V_prev_agg)
        return 1 - alpha, alpha  # 历史权重，当前权重

    def generate_grid(self, curr_bev, voxel_min, voxel_max, voxel_size):
        """生成体素中心坐标网格（自车坐标系）"""
        bs, c_, z, h, w = curr_bev.shape
        device = curr_bev.device
        
        # 计算体素中心坐标
        x_coords = torch.linspace(
            voxel_min[0] + voxel_size[0]/2, 
            voxel_max[0] - voxel_size[0]/2, 
            w, device=device
        )
        y_coords = torch.linspace(
            voxel_min[1] + voxel_size[1]/2, 
            voxel_max[1] - voxel_size[1]/2, 
            h, device=device
        )
        z_coords = torch.linspace(
            voxel_min[2] + voxel_size[2]/2, 
            voxel_max[2] - voxel_size[2]/2, 
            z, device=device
        )
        
        # 生成网格并扩展维度 [bs, h, w, z, 3]
        x_grid, y_grid, z_grid = torch.meshgrid(x_coords, y_coords, z_coords, indexing='xy')
        grid = torch.stack([x_grid, y_grid, z_grid], dim=-1)  # [w, h, z, 3]
        grid = grid.permute(1, 0, 2, 3)  # [h, w, z, 3]
        return grid.unsqueeze(0).repeat(bs, 1, 1, 1, 1)  # [bs, h, w, z, 3]


    def get_reference_points(self, H, W, Z=None, num_points_in_pillar =4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
        torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    @force_fp32(apply_to=('reference_points', 'cam_params'))
    def point_sampling(self, reference_points, pc_range, img_metas, cam_params=None):
        # prepare for point sampling
        lidar2img = []
        ego2lidar = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])     # lidar2img update the post aug in the loading pipeline
            ego2lidar.append(img_meta['ego2lidar'])
        lidar2img = torch.stack(lidar2img, dim=0).to(reference_points.device)
        ego2lidar = torch.stack(ego2lidar, dim=0).to(reference_points.device)

        sensor2egos, ego2globals, intrins, post_augs, bda_mat = cam_params
        num_cam = sensor2egos.size(1)
        ogfH, ogfW = self.final_dim

        # reference_points defines in the bev space, [bs, D, hxw, 3]
        # change reference_points from bev-ego coordinate to ego coordinate
        reference_points = reference_points.clone()
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
                                     (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
                                     (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
                                     (pc_range[5] - pc_range[2]) + pc_range[2]

        # prepare for point sampling
        reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
        reference_points = reference_points.permute(1, 0, 2, 3)  # shape: (num_points_in_pillar,bs,h*w,4)
        D, B, num_query = reference_points.size()[:3]  # D=num_points_in_pillar , num_query=h*w
        reference_points = reference_points.view(D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)  # shape: (num_points_in_pillar,bs,num_cam,h*w,4)
        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)
        ego2lidar = ego2lidar.view(1, B, 1, 1, 4, 4).repeat(D, 1, num_cam, num_query, 1, 1)
        inverse_bda = bda_mat.view(1, B, 1, 1, 4, 4).repeat(D, 1, num_cam, num_query, 1, 1)

        # change reference_points from ego coordinate to img coordinate
        eps = 1e-5
        reference_points_cam = (lidar2img @ ego2lidar @ inverse_bda @ reference_points).squeeze(-1)   # [num_points_in_pillar, bs, num_cam, num_query=h*w, 4]
        reference_points_depth = reference_points_cam[..., 2:3]
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(reference_points_depth, torch.ones_like(reference_points_depth) * eps)

        # Bug!!
        # Correct normalize is
        # reference_points_cam[..., 0] /= ogfW
        # reference_points_cam[..., 1] /= ogfH
        # But for reproducing our results, we use the following normalization
        reference_points_cam[..., 0] /= ogfW
        reference_points_cam[..., 1] /= ogfH

        bev_mask = (reference_points_depth > eps)
        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)                  # shape: (num_cam, bs,h*w, num_points_in_pillar, 2)
        reference_points_depth = reference_points_depth.permute(2, 1, 3, 0, 4)              # shape: (num_cam, bs,h*w, num_points_in_pillar, 1)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)                        # shape: (num_cam, bs,h*w, num_points_in_pillar)

        return reference_points_cam, reference_points_depth, bev_mask


    @force_fp32()
    def forward(self, curr_bev, cam_params, history_fusion_params, dx, bx, history_last_bev=None, last_occ_pred=None, nonempty_prob=None, img_feats=None, spatial_shapes=None,pred_img_depth=None,**kwargs):
        """
        Args:
            curr_bev: [bs, c, z, h, w] 当前帧BEV特征
            cam_params: 相机参数列表，包含外参、内参等
            history_fusion_params: 历史融合参数（包含序列信息等）
            dx: 体素尺寸 (x, y, z)
            bx: 体素偏移
            nonempty_prob: [bs, z, h, w] 体素非空概率
            last_occ_pred: [bs, z, h, w, num_classes] 上一时刻 occupancy 预测
            img_feats: 多尺度图像特征 [bs, num_levels, c, h, w]（新增，用于深度采样）
            spatial_shapes: 图像特征尺度 [num_levels, 2] (h, w)（新增，用于深度采样）
        Returns:
            curr_bev_updated: [bs, c, z, h, w] 融合后BEV特征
        """
        
        # -------------------------- 1. 解析参数后打印核心形状 --------------------------
        # 解析相机参数
        curr_cam_extrins = cam_params[0]  # [bs, num_cams, 4, 4]
        curr_cam_intrins = cam_params[2]  # [bs, num_cams, 4, 4]
        forward_augs = cam_params[4]      # [bs, 4, 4] 前向变换矩阵
        self.num_cams = curr_cam_extrins.shape[1]  # 从外参中获取相机数量
        bs, c_, z, h, w = curr_bev.shape
        device = curr_bev.device
        mc = self.history_num * c_        # 历史特征总通道数
        self.history_forward_augs = forward_augs.clone()


        ref_3d = self.get_reference_points(
            h, w, z, z, dim='3d', bs=bs, device=device, dtype=curr_bev.dtype) # torch.Size([3, 2, 625, 3])
        ref_2d = self.get_reference_points(
            h, w, dim='2d', bs=bs, device=device, dtype=curr_bev.dtype) #torch.Size([3, 625, 1, 2])
        slots = torch.zeros(list([ref_3d.shape[0],ref_3d.shape[2],ref_3d.shape[1]])+[88]).to(ref_3d)
        reference_points_cam, reference_points_depth, bev_mask = self.point_sampling(ref_3d, self.pc_range, img_metas=kwargs['img_metas'], cam_params=cam_params)
        indexes = [[] for _ in range(bs)]
        spatial_shapes =[]
        spatial_shapes.append([16, 44])
        spatial_shapes = torch.tensor(spatial_shapes).to(device)
        pred_img_depth = pred_img_depth.view(bs * 6, -1, spatial_shapes[0][0], spatial_shapes[0][1])
        pred_img_depth = pred_img_depth.flatten(2).permute(0, 2, 1)  
        max_len = 0
        for j in range(bs):
            for i, mask_per_img in enumerate(bev_mask):
                index_query_per_img = mask_per_img[j].sum(-1).nonzero().squeeze(-1)
                if len(index_query_per_img) == 0:
                    index_query_per_img = bev_mask[i][j].sum(-1).nonzero().squeeze(-1)[0:1]
                indexes[j].append(index_query_per_img)
                # for batch operation, we need to pad the indexes to the same length
                max_len = max(max_len, len(index_query_per_img))
        reference_points_cam_rebatch = reference_points_cam.new_zeros([bs, self.num_cams, max_len, z, 2])
        reference_points_depth_rebatch = reference_points_depth.new_zeros([bs, self.num_cams, max_len, z, 1])

        for j in range(bs):
            for i, (reference_points_per_img, reference_points_depth_per_img) in enumerate(zip(reference_points_cam, reference_points_depth)):
                index_query_per_img = indexes[j][i]
                reference_points_cam_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]
                reference_points_depth_rebatch[j, i, :len(index_query_per_img)] = reference_points_depth_per_img[j, index_query_per_img]

        #use deformble attn
        depth_reference_points = reference_points_cam_rebatch.reshape(bs*6, max_len*z, 1, 1, 1, 2).contiguous()
        depth_attention_weights = torch.ones_like(depth_reference_points[..., 0]).contiguous()
        pred_img_depth = pred_img_depth.unsqueeze(2).contiguous()
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))



        bev_query_depth_rebatch = (reference_points_depth_rebatch- self.dbound[0])/ self.dbound[2]
        bev_query_depth_rebatch = torch.clip(torch.floor(bev_query_depth_rebatch), 0, 88-1).to(torch.long)
        bev_query_depth_rebatch = F.one_hot(bev_query_depth_rebatch.squeeze(-1),
                                   num_classes=88)

        depth_output = MultiScaleDeformableAttnFunction_fp32.apply(pred_img_depth, spatial_shapes,level_start_index,depth_reference_points,depth_attention_weights, self.im2col_step)
        depth_output = depth_output.reshape(bs,6, max_len,z, -1)   # [bs*num_cam, num_query, num_Z_anchors, C]
        # reference_points_depth_rebatch

        # depth_output[...,0]+=1e-9
        # print("depth_output.sum(-1).shape")
        # print(depth_output.sum(-1).shape)
        # print("depth_output.sum(-1)")
        # print(depth_output.sum(-1))
        # print("print((depth_output.sum(-1)==0).sum())")
        # print((depth_output.sum(-1)==0).sum())
        # depth_output =depth_output/depth_output.sum(-1)[...,None] #bs,xy,z,D

        # print("depth_output.sum(-1).shape")
        # print(depth_output.sum(-1).shape)
        # print("depth_output.sum(-1)")
        # print(depth_output.sum(-1))
        # print("print((depth_output.sum(-1)==1).sum())")
        # print("depth_output")
        # print(depth_output)


        # depth_output = (1-depth_output.cumsum(dim=-1))


        # # print("print((depth_output[...,-1]==0).sum())")
        # # print((depth_output[...,-1]==0).sum())
        # depth_output = bev_query_depth_rebatch*depth_output

        #恢复depth_output的shape
        # fix_depth_output = depth_output.new_zeros([bs, 6, w*h, z, 88])
        # for j in range(bs):
        #     for i in range(6):
        #         index_query_per_img = indexes[j][i]
        #         fix_depth_output[j, i, index_query_per_img] = depth_output[j, i, :len(index_query_per_img)]

        for j in range(bs):
            for i in range(6):
                index_query_per_img = indexes[j][i]
                slots[j, index_query_per_img] += depth_output[j, i, :len(index_query_per_img)]

        # depth_sum = fix_depth_output.sum(dim=-1).view(3,6,25,25,2)
        # bs = depth_sum.shape[0]       # 3
        # num_cam = depth_sum.shape[1]  # 6
        # z_layers = depth_sum.shape[4] # 2
        # H, W = depth_sum.shape[2], depth_sum.shape[3]  # 25,25

        # # 颜色映射：0值用黑色，非0值用渐变色
        # cmap = plt.cm.viridis
        # cmap.set_bad(color='black')  # 0值标记为黑色

        # # 设置子图布局（不变）
        # fig, axes = plt.subplots(
        #     nrows=bs, ncols=num_cam * z_layers,
        #     figsize=(30, 8),
        #     squeeze=False
        # )
        # fig.suptitle('fix_depth_output.sum(-1) 可视化（黑色=0值，颜色=非0值）', fontsize=16, y=0.98)


        # # -------------------------- 2. 循环绘制每个子图（核心修改：添加 .detach()） --------------------------
        # for batch_idx in range(bs):
        #     for cam_idx in range(num_cam):
        #         for z_idx in range(z_layers):
        #             col_idx = cam_idx * z_layers + z_idx
        #             ax = axes[batch_idx, col_idx]
                    
        #             # 核心修改：先 detach() 切断计算图，再转 cpu 和 numpy
        #             data = depth_sum[batch_idx, cam_idx, :, :, z_idx].detach().cpu().numpy()
        #             data[data == 0] = np.nan  # 0值替换为NaN，显示为黑色
                    
        #             # 绘制热力图（修改 vmin/vmax：同样添加 .detach()）
        #             im = ax.imshow(
        #                 data, 
        #                 cmap=cmap, 
        #                 aspect='auto',
        #                 # 关键修改：depth_sum 先 detach 再转 numpy，确保不影响梯度
        #                 vmin=np.nanmin(depth_sum.detach().cpu().numpy()),
        #                 vmax=np.nanmax(depth_sum.detach().cpu().numpy())
        #             )
                    
        #             # 子图标题和坐标轴（不变）
        #             ax.set_title(
        #                 f'Batch{batch_idx+1}\nCam{cam_idx+1} Z{z_idx+1}',
        #                 fontsize=10, pad=5
        #             )
        #             ax.set_xticks([])
        #             ax.set_yticks([])


        # # -------------------------- 3. 添加颜色条（不变） --------------------------
        # cbar = fig.colorbar(
        #     im, 
        #     ax=axes.ravel().tolist(),
        #     shrink=0.8,
        #     pad=0.02
        # )
        # cbar.set_label('Sum of Depth Bins (D=88)', fontsize=12)


        # # -------------------------- 4. 调整布局并保存（不变） --------------------------
        # plt.tight_layout(rect=[0, 0, 0.98, 0.95])
        # plt.savefig('depth_sum_visualization.png', dpi=300, bbox_inches='tight')
        # plt.show()

        # output
        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None,None]
        slots[...,0]+=1e-9
        print("slots.sum(-1).shape")
        print(slots.sum(-1).shape)
        print("slots.sum(-1)")
        print(slots.sum(-1))
        print("print((slots.sum(-1)==0).sum())")
        print((slots.sum(-1)==0).sum())
        slots =slots/slots.sum(-1)[...,None] #bs,xy,z,D

        print("slots.sum(-1).shape")
        print(slots.sum(-1).shape)
        print("slots.sum(-1)")
        print(slots.sum(-1))
        print("print((slots.sum(-1)==1).sum())")
        print("slots")
        print(slots)


        slots = (1-slots.cumsum(dim=-1))

        print("print((slots[...,-1]==0).sum())")
        print((slots[...,-1]==0).sum())

        print("slots")
        print(slots)
        print("print((slots[...,-1]<0.01).sum())")
        print((slots[...,-1]<0.01).sum())

        #TODO 这里的对于边界值的考虑，从0开始还是从1开始？




        # slots = self.output_proj(slots)
        #TODO 上面这里需要检查一下
        #这里相当于两次softmax，可能会导致分布变得不够尖锐，需要进一步确认

        #到这里slots就是可见性的概率分布了
        #这里先尝试使用期望值进行计算，使得可微分

        #然后再采用stc的原始离散计算方法，


        # 打印解析后关键变量形状
        print("="*50)
        print("1. 解析参数后核心变量形状：")
        print(f"curr_bev: {curr_bev.shape} (预期：[bs, c, z, h, w])")
        print(f"curr_cam_extrins: {curr_cam_extrins.shape} (预期：[bs, num_cams, 4, 4])")
        print(f"curr_cam_intrins: {curr_cam_intrins.shape} (预期：[bs, num_cams, 4, 4])")
        print(f"forward_augs: {forward_augs.shape} (预期：[bs, 4, 4])")
        print(f"dx: {dx.shape if hasattr(dx, 'shape') else type(dx)} (预期：[3])")
        print(f"bx: {bx.shape if hasattr(bx, 'shape') else type(bx)} (预期：[3])")
        print(f"bs: {bs}, c_: {c_}, z: {z}, h: {h}, w: {w} (BEV特征维度)")
        print("="*50)

        if type(history_fusion_params['sequence_group_idx']) is list:
            seq_ids = history_fusion_params['sequence_group_idx'][0]
        else:
            seq_ids = history_fusion_params['sequence_group_idx']
        if type(history_fusion_params['start_of_sequence']) is list:
            start_of_sequence = history_fusion_params['start_of_sequence'][0]
        else:
            start_of_sequence = history_fusion_params['start_of_sequence']
        if type(history_fusion_params['curr_to_prev_ego_rt']) is list:
            curr_to_prev_ego_rt = history_fusion_params['curr_to_prev_ego_rt'][0]
        else:
            curr_to_prev_ego_rt = history_fusion_params['curr_to_prev_ego_rt']
        forward_augs = cam_params[-1]  # bda

        # check seq_ids > 0
        assert (seq_ids >= 0).all()
        # -------------------------- 2. 初始化历史缓存后打印 --------------------------
        if self.history_bev is None:
            # self.history_bev = curr_bev.repeat(1, self.history_num, 1, 1, 1)  # [bs, mc, z, h, w]
            self.history_cam_intrins = curr_cam_intrins.unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)  # [bs, T, num_cams, 4, 4]
            self.history_cam_extrins = curr_cam_extrins.unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)  # [bs, T, num_cams, 4, 4]
            self.history_bev = curr_bev.clone()
            self.history_forward_augs = forward_augs.clone()
            self.history_seq_ids = seq_ids.clone()
            self.history_bev = curr_bev.repeat(1, self.history_num, 1, 1, 1)
            self.history_sweep_time = curr_bev.new_zeros(curr_bev.shape[0], self.history_num)
        self.history_bev = self.history_bev.detach()
        self.history_sweep_time += 1

        # 打印历史缓存形状
        print("\n2. 历史缓存初始化后形状：")
        print(f"history_bev: {self.history_bev.shape} (预期：[bs, mc, z, h, w]，mc={mc})")
        print(f"history_cam_intrins: {self.history_cam_intrins.shape} (预期：[bs, history_num, num_cams, 4, 4])")
        print(f"history_cam_extrins: {self.history_cam_extrins.shape} (预期：[bs, history_num, num_cams, 4, 4])")

        # -------------------------- 3. 生成网格和BEV变换矩阵后打印 --------------------------
        # 处理新序列（略，不影响维度）
        # start_of_sequence = history_fusion_params.get('start_of_sequence', torch.zeros(bs, dtype=torch.bool, device=device))
        if start_of_sequence.sum()>0:
            self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_num, 1, 1, 1)
            self.history_cam_intrins[start_of_sequence] = curr_cam_intrins[start_of_sequence].unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)
            self.history_cam_extrins[start_of_sequence] = curr_cam_extrins[start_of_sequence].unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)
            self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]
            self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
            self.history_sweep_time[start_of_sequence] = 0  # zero the new sequence timestep starts

        # 生成体素网格和BEV变换矩阵
        grid = self.generate_grid(curr_bev)
        grid_3d = grid
        feat2bev = self.generate_feat2bev(grid, dx, bx)

        # 打印网格和变换矩阵形状
        print("\n3. 生成网格和BEV变换矩阵后形状：")
        print(f"grid_3d (体素网格): {grid_3d.shape} (关键！预期：[bs, h, w, z, 3] 或 [bs, w, h, z, 3])")
        print(f"feat2bev (BEV变换矩阵): {feat2bev.shape} (预期：[bs, 4, 4])")

        # -------------------------- 4. 运动补偿矩阵计算后打印 --------------------------
        # 获取帧间姿态变换
        # curr_to_prev_ego_rt = history_fusion_params.get('curr_to_prev_ego_rt', torch.eye(4, device=device).unsqueeze(0).repeat(bs, 1, 1))
        # 计算RT流（坐标变换矩阵）
        rt_flow = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_ego_rt @ torch.inverse(forward_augs) @ feat2bev)
        # 生成齐次网格
        # 在forward函数中，生成grid_hom的位置修正：
        # grid_3d = self.generate_grid(curr_bev)  # 现在形状：[3, 25, 25, 2, 3]（bs, h, w, z, 3）
        # # 生成齐次坐标（x,y,z,1），并添加最后一个维度（用于矩阵乘法）
        # grid_hom = torch.cat([
        #     grid_3d,  # [3,25,25,2,3]
        #     torch.ones_like(grid_3d[..., :1])  # [3,25,25,2,1]（补充1作为齐次坐标）
        # ], dim=-1).unsqueeze(-1)  # 最终形状：[3,25,25,2,4,1]（符合预期）
        # # 打印运动补偿相关形状（矩阵乘法前关键检查）
        # print("\n4. 运动补偿矩阵计算后形状（矩阵乘法前）：")
        # print(f"curr_to_prev_ego_rt (帧间姿态): {curr_to_prev_ego_rt.shape} (预期：[bs, 4, 4])")
        # print(f"rt_flow (变换流): {rt_flow.shape} (预期：[bs, 4, 4])")
        # print(f"grid_hom (齐次网格): {grid_hom.shape} (关键！预期：[bs, h, w, z, 4, 1]，需与rt_flow广播匹配)")
        # print(f"rt_flow.view后: {rt_flow.view(bs, 1, 1, 1, 4, 4).shape} (预期：[bs, 1, 1, 1, 4, 4])")

        # # -------------------------- 5. 网格变换后打印（解决之前维度错的核心） --------------------------
        # try:
        #     grid_transformed = rt_flow.view(bs, 1, 1, 1, 4, 4) @ grid_hom  # 矩阵乘法：[bs, h, w, z, 4, 1]
        #     print("\n5. 网格变换后形状（矩阵乘法成功！）：")
        #     print(f"grid_transformed: {grid_transformed.shape} (预期：[bs, h, w, z, 4, 1])")
        # except RuntimeError as e:
        #     print(f"\n5. 网格变换矩阵乘法报错！错误信息：{str(e)}")
        #     print(f"  - rt_flow.view形状: {rt_flow.view(bs, 1, 1, 1, 4, 4).shape}")
        #     print(f"  - grid_hom形状: {grid_hom.shape}")
        #     print("  提示：需确保grid_hom的第1-4维度与rt_flow.view的第2-5维度匹配（广播规则）")
        #     raise e  # 继续抛出错误，方便定位
        bs, mc, z, h, w = self.history_bev.shape
        n, c_, z, h, w = curr_bev.shape
        grid = rt_flow.view(n, 1, 1, 1, 4, 4) @ grid
        # -------------------------- 6. 采样网格生成后打印 --------------------------
        # 生成采样网格（归一化到[-1,1]，适配F.grid_sample）
        normalize_factor = torch.tensor([w - 1.0, h - 1.0, z - 1.0], dtype=curr_bev.dtype, device=device)
        # grid_sampler = grid_transformed[..., :3, 0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0  # [bs, h, w, z, 3]
        # # 调整采样网格维度（适配F.grid_sample输入：[bs, z, h, w, 3]）
        # grid_sampler_permuted = grid_sampler.permute(0, 3, 1, 2, 4)  # 交换z和h/w维度
        grid = grid[:, :, :, :, :3, 0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0   # grid order is x, y, z


        print("\n6. 采样网格生成后形状：")
        # print(f"grid_sampler (归一化后): {grid_sampler.shape} (预期：[bs, h, w, z, 3])")
        # print(f"grid_sampler_permuted (适配采样): {grid_sampler_permuted.shape} (预期：[bs, z, h, w, 3])")

        # -------------------------- 7. 历史BEV采样后打印 --------------------------
        # 采样历史BEV特征
        sampled_history_bev = F.grid_sample(
            self.history_bev.reshape(bs, mc, z, h, w),  # 输入：[bs, mc, z, h, w]
            grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4),
            align_corners=True,
            mode='bilinear'
        )

        print("\n7. 历史BEV采样后形状：")
        print(f"history_bev.reshape: {self.history_bev.reshape(bs, mc, z, h, w).shape} (预期：[bs, mc, z, h, w])")
        print(f"sampled_history_bev: {sampled_history_bev.shape} (预期：[bs, mc, z, h, w])")

        # -------------------------- 8. 可见性计算后打印 --------------------------
        # 计算当前帧可见性
        # V_curr = self.compute_visibility(
        #     grid_3d, 
        #     cam_intrins=curr_cam_intrins,
        #     cam_extrins=curr_cam_extrins,
        #     img_shape=self.img_shape,
        #     img_feats=img_feats,
        #     spatial_shapes=spatial_shapes
        # )

        print("V_curr (当前可见性).  "*3)
        V_curr = slots
        # 计算历史帧可见性
        prev_cam_intrins = self.history_cam_intrins[:, -1]
        prev_cam_extrins = self.history_cam_extrins[:, -1]
        # V_prev = self.compute_visibility(
        #     grid_3d,
        #     cam_intrins=prev_cam_intrins,
        #     cam_extrins=prev_cam_extrins,
        #     img_shape=self.img_shape,
        #     img_feats=img_feats,
        #     spatial_shapes=spatial_shapes
        # )

        V_prev = slots

        print("\n8. 可见性计算后形状：")
        print(f"V_curr (当前可见性): {V_curr.shape} (预期：[bs, h, w, z])")
        print(f"V_prev (历史可见性): {V_prev.shape} (预期：[bs, h, w, z])")

        # -------------------------- 9. 稀疏采样前展平变量打印 --------------------------
        # 展平变量（用于稀疏采样）
        curr_bev_flat = curr_bev.reshape(bs, c_, -1)  # [bs, c_, N], N=h*w*z
        history_bev_flat = sampled_history_bev.reshape(bs, mc, -1)  # [bs, mc, N]
        V_prev_flat = V_prev.reshape(bs, -1)  # [bs, N]
        V_curr_flat = V_curr.reshape(bs, -1)  # [bs, N]
        nonempty_prob_flat = nonempty_prob.reshape(bs, -1)  # [bs, N]
        total_voxels = nonempty_prob_flat.shape[1]

        print("\n9. 稀疏采样前展平变量形状：")
        print(f"curr_bev_flat: {curr_bev_flat.shape} (预期：[bs, c_, N], N={total_voxels})")
        print(f"history_bev_flat: {history_bev_flat.shape} (预期：[bs, mc, N])")
        print(f"nonempty_prob_flat: {nonempty_prob_flat.shape} (预期：[bs, N])")
        print(f"total_voxels (h*w*z): {total_voxels} (预期：{h*w*z})")

        # -------------------------- 10. 前景/背景索引及融合后打印（可选，确认后续维度） --------------------------
        # 生成前景/背景索引
        fg_indices = torch.topk(nonempty_prob_flat, self.top_k, dim=1)[1]
        bg_indices = torch.topk(1 - nonempty_prob_flat, total_voxels - self.top_k, dim=1)[1]
        # 提取前景特征（示例，其他融合步骤类似）
        fg_history_feat = torch.gather(history_bev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, mc, 1))

        print("\n10. 前景/背景索引及特征提取后形状：")
        print(f"fg_indices (前景索引): {fg_indices.shape} (预期：[bs, top_k])")
        print(f"bg_indices (背景索引): {bg_indices.shape} (预期：[bs, N-top_k])")
        print(f"fg_history_feat (前景历史特征): {fg_history_feat.shape} (预期：[bs, mc, top_k])")
        print("="*50)

        # -------------------------- 后续原有逻辑（略，维度已通过打印确认） --------------------------
        # 8. 前景融合（原有代码）
        # 9. 背景融合（原有代码）
        # 10. 更新当前BEV特征（原有代码）
        # curr_bev_flat = curr_bev.reshape(bs, c_, -1)  # [bs, c_, N] N=h*w*z
        # history_bev_flat = sampled_history_bev.reshape(bs, mc, -1)  # [bs, mc, N]
        # V_prev_flat = V_prev.reshape(bs, -1)  # [bs, N]
        # V_curr_flat = V_curr.reshape(bs, -1)  # [bs, N]
        # nonempty_prob_flat = nonempty_prob.reshape(bs, -1)  # [bs, N]
        # total_voxels = nonempty_prob_flat.shape[1]

        # fg_indices = torch.topk(nonempty_prob_flat, self.top_k, dim=1)[1]  # [bs, top_k]
        # bg_indices = torch.topk(1 - nonempty_prob_flat, total_voxels - self.top_k, dim=1)[1]  # [bs, N-top_k]

        # # 前景特征提取
        # fg_history_feat = torch.gather(history_bev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, mc, 1))  # [bs, mc, top_k]
        fg_curr_feat = torch.gather(curr_bev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, c_, 1))  # [bs, c_, top_k]

        # 历史特征时间聚合
        fg_history_feat_time = fg_history_feat.reshape(bs, self.history_num, c_, self.top_k)  # [bs, T, c_, K]
        time_weights = torch.exp(-0.5 * torch.arange(self.history_num, device=device)).view(1, self.history_num, 1, 1)
        fg_history_agg = (fg_history_feat_time * time_weights).sum(dim=1)  # [bs, c_, K]

        # 可见性聚合与门控
        fg_V_prev = torch.gather(V_prev_flat, dim=1, index=fg_indices)  # [bs, K]
        fg_V_curr = torch.gather(V_curr_flat, dim=1, index=fg_indices)  # [bs, K]
        fg_V_prev_time = fg_V_prev.unsqueeze(1).repeat(1, self.history_num, 1)  # [bs, T, K]
        fg_V_prev_agg = (fg_V_prev_time * time_weights.squeeze(-1)).sum(dim=1)  # [bs, K]
        fg_w_hist, fg_w_curr = self.compute_gate_weights(fg_V_prev_agg, fg_V_curr)  # [bs, K, 1]

        # 前景融合
        fg_history_agg_perm = fg_history_agg.permute(0, 2, 1)  # [bs, K, c_]
        fg_curr_perm = fg_curr_feat.permute(0, 2, 1)  # [bs, K, c_]
        fg_fused = fg_w_hist * fg_history_agg_perm + fg_w_curr * fg_curr_perm  # [bs, K, c_]

        # occupancy嵌入融合
        last_occ_reshaped = last_occ_pred.permute(0, 3, 2, 1, 4).reshape(bs, -1, last_occ_pred.shape[-1])  # [bs, N, num_classes]
        fg_occ_feat = torch.gather(last_occ_reshaped, dim=1, index=fg_indices.unsqueeze(-1).repeat(1, 1, last_occ_pred.shape[-1]))  # [bs, K, num_classes]
        fg_occ_embed = self.occ_embedding(fg_occ_feat).permute(0, 2, 1)  # [bs, occ_embedims, K]
        fg_fused = torch.cat([fg_fused, fg_occ_embed.permute(0, 2, 1)], dim=-1)  # [bs, K, c_+occ_embedims]
        fg_fused = self.history_fusion_linear(fg_fused).permute(0, 2, 1)  # [bs, c_, K]

        # 背景融合（原有代码）
        bg_history_feat = torch.gather(history_bev_flat[:, :mc//2], dim=2, index=bg_indices.unsqueeze(1).repeat(1, mc//2, 1))  # [bs, mc//2, bg_k]
        bg_curr_feat = torch.gather(curr_bev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, c_, 1))  # [bs, c_, bg_k]

        bg_history_feat_time = bg_history_feat.reshape(bs, self.history_num, c_//2, -1)  # [bs, T, c_//2, bg_k]
        bg_history_agg = (bg_history_feat_time * time_weights).sum(dim=1)  # [bs, c_//2, bg_k]
        bg_history_agg_perm = F.pad(bg_history_agg.permute(0, 2, 1), (0, c_ - c_//2, 0, 0))  # [bs, bg_k, c_]

        bg_V_prev = torch.gather(V_prev_flat, dim=1, index=bg_indices)  # [bs, bg_k]
        bg_V_curr = torch.gather(V_curr_flat, dim=1, index=bg_indices)  # [bs, bg_k]
        bg_V_prev_time = bg_V_prev.unsqueeze(1).repeat(1, self.history_num, 1)  # [bs, T, bg_k]
        bg_V_prev_agg = (bg_V_prev_time * time_weights.squeeze(-1)).sum(dim=1)  # [bs, bg_k]
        bg_w_hist, bg_w_curr = self.compute_gate_weights(bg_V_prev_agg, bg_V_curr)  # [bs, bg_k, 1]

        bg_curr_perm = bg_curr_feat.permute(0, 2, 1)  # [bs, bg_k, c_]
        bg_fused = bg_w_hist * bg_history_agg_perm + bg_w_curr * bg_curr_perm  # [bs, bg_k, c_]
        bg_occ_feat = torch.gather(last_occ_reshaped, dim=1, index=bg_indices.unsqueeze(-1).repeat(1, 1, last_occ_pred.shape[-1]))  # [bs, bg_k, num_classes]
        bg_occ_embed = self.occ_embedding(bg_occ_feat).permute(0, 2, 1)  # [bs, occ_embedims, bg_k]
        bg_fused = torch.cat([bg_fused, bg_occ_embed.permute(0, 2, 1)], dim=-1)  # [bs, bg_k, c_+occ_embedims]
        bg_fused = self.history_fusion_bg_linear(bg_fused).permute(0, 2, 1)  # [bs, c_, bg_k]

        # 更新当前BEV
        curr_bev_updated = curr_bev_flat.clone()
        curr_bev_updated.scatter_add_(dim=2, index=fg_indices.unsqueeze(1).repeat(1, c_, 1), src=fg_fused)
        curr_bev_updated.scatter_add_(dim=2, index=bg_indices.unsqueeze(1).repeat(1, c_, 1), src=bg_fused)
        curr_bev_updated = curr_bev_updated.reshape(bs, c_, z, h, w)  # 恢复原形状

        # 更新历史缓存
        self.history_last_bev = curr_bev_updated.detach().clone()
        self.history_bev = torch.cat([curr_bev,sampled_history_bev[:, :-c_, ...]], dim=1).detach()
        self.history_cam_intrins = torch.cat([self.history_cam_intrins[:, 1:, ...], curr_cam_intrins.unsqueeze(1)], dim=1).detach()
        self.history_cam_extrins = torch.cat([self.history_cam_extrins[:, 1:, ...], curr_cam_extrins.unsqueeze(1)], dim=1).detach()

        return curr_bev_updated

    def generate_grid(self, curr_bev):
        n, c_, z, h, w = curr_bev.shape
        # Generate grid
        xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w, 1).expand(h, w, z)
        ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1, 1).expand(h, w, z)
        zs = torch.linspace(0, z - 1, z, dtype=curr_bev.dtype, device=curr_bev.device).view(1, 1, z).expand(h, w, z)
        grid = torch.stack((xs, ys, zs, torch.ones_like(xs)), -1).view(1, h, w, z, 4).expand(n, h, w, z, 4).view(n, h,w, z, 4, 1)
        return grid

    def generate_feat2bev(self, grid, dx, bx):
        feat2bev = torch.zeros((4, 4), dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = dx[0]
        feat2bev[1, 1] = dx[1]
        feat2bev[2, 2] = dx[2]
        feat2bev[0, 3] = bx[0] - dx[0] / 2.
        feat2bev[1, 3] = bx[1] - dx[1] / 2.
        feat2bev[2, 3] = bx[2] - dx[2] / 2.
        feat2bev[3, 3] = 1
        feat2bev = feat2bev.view(1, 4, 4)
        return feat2bev


@HEADS.register_module()
class GatedTemporalFusion4(BaseModule):
    def __init__(
        self,
        history_num=4,
        depth_sampler_embed_dims=256,
        depth_sampler_num_heads=8,
        depth_sampler_num_levels=4,
        depth_sampler_num_points=4,
        im2col_step=64,
        top_k=500,
        single_bev_num_channels=96,
        occ_embedims=32,
        num_classes=18,
        vis_theta=0.3,
        vis_beta=10.0,
        vis_gamma=0.3,
        vis_sigma=0.1,
        nonempty_thresh=0.1,
        max_step_ratio=1.2,
        **kwargs
    ):
        super(GatedTemporalFusion4, self).__init__()

        # 基础参数
        self.history_num = history_num
        self.top_k = top_k
        self.single_bev_num_channels = single_bev_num_channels
        self.occ_embedims = occ_embedims
        
        # 可见性门控参数
        self.vis_theta = vis_theta  # 可见性阈值
        self.vis_beta = vis_beta    # 敏感度参数
        self.vis_gamma = vis_gamma  # 场景4固定权重
        self.vis_sigma = vis_sigma  # 软化参数
        
        # 历史特征缓存（新增history_last_bev）
        self.history_bev = None  # 缓存历史多帧BEV特征
        self.history_last_bev = None  # 缓存上一帧最终融合后的BEV特征（关键新增）
        self.history_cam_intrins = None
        self.history_cam_extrins = None
        
        # 网络层（保持不变）
        self.occ_embedding = nn.Sequential(
            nn.Linear(num_classes, occ_embedims),
            nn.Softplus(),
            nn.Linear(occ_embedims, occ_embedims),
        )
        self.history_fusion_linear = nn.Sequential(
            nn.Linear(single_bev_num_channels + occ_embedims, single_bev_num_channels),
            nn.Softplus(),
            nn.Linear(single_bev_num_channels, single_bev_num_channels),
        )
        self.history_fusion_bg_linear = nn.Sequential(
            nn.Linear(single_bev_num_channels + occ_embedims, single_bev_num_channels),
            nn.Softplus(),
            nn.Linear(single_bev_num_channels, single_bev_num_channels),
        )
        
        # 可见性计算组件（保持不变）
        self.rt_vis_calculator = EfficientRayTracingVisibility(
            nonempty_thresh=nonempty_thresh,
            max_step_ratio=max_step_ratio
        )
        self.img_shape = (900, 1600)  # 默认图像尺寸

        self.depth_sampler = DeformableDepthSampler(
            embed_dims=depth_sampler_embed_dims,
            num_heads=depth_sampler_num_heads,
            num_levels=depth_sampler_num_levels,
            num_points=depth_sampler_num_points
        )
        self.history_forward_augs = None  # 用于缓存历史帧的变换矩阵（如BDAM矩阵）
        self.im2col_step = im2col_step
        # self.voxel_encoder = nn.Linear(single_bev_num_channels, depth_sampler_embed_dims)

        self.pc_range = [-40, -40, -1.0, 40, 40, 5.4]
        self.final_dim = (256, 704)
    def compute_visibility(self, grid, cam_intrins, cam_extrins, img_shape, img_feats, spatial_shapes):
        """
        升级：结合可变形注意力采样的深度值优化可见性计算
        Args:
            img_feats: 多尺度图像特征 [bs, num_levels, c, h, w]
            spatial_shapes: 图像特征尺度 [num_levels, 2] (h, w)
            其他参数同原函数
        Returns:
            vis_prob: [bs, h, w, z] 优化后的可见性概率
        """
        bs, h, w, z, _ = grid.shape
        h_img, w_img = img_shape
        device = grid.device
        num_voxels = h * w * z  # 体素总数

        # 1. 原有相机投影逻辑（计算图像坐标和初始可见性）
        # 1.1 体素坐标→相机坐标→图像坐标
        grid_cam = grid.unsqueeze(1).expand(bs, self.num_cams, h, w, z, 3)  # [bs, num_cams, h, w, z, 3]
        grid_flat = grid_cam.reshape(-1, num_voxels, 3)  # [bs*num_cams, N, 3]
        grid_hom = torch.cat([grid_flat, torch.ones_like(grid_flat[..., :1])], dim=-1)  # [bs*num_cams, N, 4]

        cam_intrins_flat = cam_intrins.reshape(-1, 4, 4)  # [bs*num_cams, 4, 4]
        cam_extrins_flat = cam_extrins.reshape(-1, 4, 4)  # [bs*num_cams, 4, 4]
        extrins_inv = torch.inverse(cam_extrins_flat)
        cam_coords = torch.bmm(extrins_inv[:, :3, :4], grid_hom.permute(0, 2, 1)).permute(0, 2, 1)  # [bs*num_cams, N, 3]
        depth = cam_coords[..., 2:3] + 1e-8  # 相机坐标系下的深度

        # 1.2 计算初始可见性（原逻辑）
        img_coords = torch.bmm(cam_intrins_flat[:, :3, :3], cam_coords.permute(0, 2, 1)).permute(0, 2, 1)  # [bs*num_cams, N, 3]
        img_xy = img_coords[..., :2] / depth  # [bs*num_cams, N, 2] (u, v)
        depth_valid = (cam_coords[..., 2] > 0).float()  # [bs*num_cams, N]
        u_valid = (img_xy[..., 0] >= 0) & (img_xy[..., 0] < w_img)
        v_valid = (img_xy[..., 1] >= 0) & (img_xy[..., 1] < h_img)
        img_valid = (u_valid & v_valid).float()  # [bs*num_cams, N]
        initial_vis = depth_valid * img_valid  # [bs*num_cams, N]

        # 2. 可变形注意力深度采样
        # 2.1 准备输入：体素特征编码
        # voxel_feat = self.voxel_encoder(self.curr_bev_feat)  # [bs, c_, z, h, w] → [bs, z*h*w, embed_dims]（需提前展平体素特征）
        # voxel_feat = voxel_feat.reshape(bs, num_voxels, -1)  # [bs, N, embed_dims]

        # 2.2 生成参考点（归一化到[0,1]）
        norm_img_xy = img_xy / torch.tensor([w_img, h_img], device=device).view(1, 1, 2)  # [bs*num_cams, N, 2]
        # 取主相机（如第0个相机）的参考点作为采样基准
        ref_points = norm_img_xy.reshape(bs, self.num_cams, num_voxels, 2)[:, 0]  # [bs, N, 2]
        ref_points = ref_points.unsqueeze(2).repeat(1, 1, self.depth_sampler.num_levels, 1)  # [bs, N, L, 2]

        # 2.3 采样深度特征
        sampled_depth = self.depth_sampler(
            query=voxel_feat,
            value=img_feats,  # 多尺度图像特征 [bs, L, c, h, w]
            reference_points=ref_points,
            spatial_shapes=spatial_shapes
        )  # [bs, N]

        # 3. 结合采样深度优化可见性
        # 3.1 深度一致性校验：采样深度与相机投影深度的差异
        cam_depth = depth.reshape(bs, self.num_cams, num_voxels)[:, 0]  # 主相机的投影深度 [bs, N]
        depth_diff = torch.abs(sampled_depth - cam_depth) / (cam_depth + 1e-8)  # 相对深度差
        depth_consistent = (depth_diff < 0.3).float()  # 深度差小于30%则认为有效

        # 3.2 融合可见性：初始可见性 × 深度一致性
        initial_vis = initial_vis.reshape(bs, self.num_cams, num_voxels).max(dim=1)[0]  # [bs, N]（多相机取max）
        vis_prob = initial_vis * depth_consistent  # [bs, N]

        # 4. 还原形状
        return vis_prob.reshape(bs, h, w, z)  # [bs, h, w, z]

    def compute_alpha_unified(self, V_curr, V_prev):
        """统一计算当前帧融合权重α（覆盖四场景）"""
        eps = 1e-8
        # 1. 基础动态权重σ_base（场景1-3）
        ratio = V_curr / (V_curr + V_prev + eps)
        sigma_base = torch.sigmoid(self.vis_beta * (ratio - 0.5))
        
        # 2. 场景4软化掩码σ_both
        mask_curr = torch.sigmoid(-(V_curr - self.vis_theta) / self.vis_sigma)
        mask_prev = torch.sigmoid(-(V_prev - self.vis_theta) / self.vis_sigma)
        sigma_both = mask_curr * mask_prev
        
        # 3. 最终权重计算
        alpha = (1 - sigma_both) * sigma_base + sigma_both * self.vis_gamma
        return alpha.unsqueeze(-1)  # [bs, N, 1]

    def compute_gate_weights(self, V_prev_agg, V_curr):
        """计算历史和当前帧的门控权重"""
        alpha = self.compute_alpha_unified(V_curr, V_prev_agg)
        return 1 - alpha, alpha  # 历史权重，当前权重

    def generate_grid(self, curr_bev, voxel_min, voxel_max, voxel_size):
        """生成体素中心坐标网格（自车坐标系）"""
        bs, c_, z, h, w = curr_bev.shape
        device = curr_bev.device
        
        # 计算体素中心坐标
        x_coords = torch.linspace(
            voxel_min[0] + voxel_size[0]/2, 
            voxel_max[0] - voxel_size[0]/2, 
            w, device=device
        )
        y_coords = torch.linspace(
            voxel_min[1] + voxel_size[1]/2, 
            voxel_max[1] - voxel_size[1]/2, 
            h, device=device
        )
        z_coords = torch.linspace(
            voxel_min[2] + voxel_size[2]/2, 
            voxel_max[2] - voxel_size[2]/2, 
            z, device=device
        )
        
        # 生成网格并扩展维度 [bs, h, w, z, 3]
        x_grid, y_grid, z_grid = torch.meshgrid(x_coords, y_coords, z_coords, indexing='xy')
        grid = torch.stack([x_grid, y_grid, z_grid], dim=-1)  # [w, h, z, 3]
        grid = grid.permute(1, 0, 2, 3)  # [h, w, z, 3]
        return grid.unsqueeze(0).repeat(bs, 1, 1, 1, 1)  # [bs, h, w, z, 3]


    def get_reference_points(self, H, W, Z=None, num_points_in_pillar =4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
        torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    @force_fp32(apply_to=('reference_points', 'cam_params'))
    def point_sampling(self, reference_points, pc_range, img_metas, cam_params=None):
        # prepare for point sampling
        lidar2img = []
        ego2lidar = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])     # lidar2img update the post aug in the loading pipeline
            ego2lidar.append(img_meta['ego2lidar'])
        lidar2img = torch.stack(lidar2img, dim=0).to(reference_points.device)
        ego2lidar = torch.stack(ego2lidar, dim=0).to(reference_points.device)

        sensor2egos, ego2globals, intrins, post_augs, bda_mat = cam_params
        num_cam = sensor2egos.size(1)
        ogfH, ogfW = self.final_dim

        # reference_points defines in the bev space, [bs, D, hxw, 3]
        # change reference_points from bev-ego coordinate to ego coordinate
        reference_points = reference_points.clone()
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
                                     (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
                                     (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
                                     (pc_range[5] - pc_range[2]) + pc_range[2]

        # prepare for point sampling
        reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
        reference_points = reference_points.permute(1, 0, 2, 3)  # shape: (num_points_in_pillar,bs,h*w,4)
        D, B, num_query = reference_points.size()[:3]  # D=num_points_in_pillar , num_query=h*w
        reference_points = reference_points.view(D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)  # shape: (num_points_in_pillar,bs,num_cam,h*w,4)
        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)
        ego2lidar = ego2lidar.view(1, B, 1, 1, 4, 4).repeat(D, 1, num_cam, num_query, 1, 1)
        inverse_bda = bda_mat.view(1, B, 1, 1, 4, 4).repeat(D, 1, num_cam, num_query, 1, 1)

        # change reference_points from ego coordinate to img coordinate
        eps = 1e-5
        reference_points_cam = (lidar2img @ ego2lidar @ inverse_bda @ reference_points).squeeze(-1)   # [num_points_in_pillar, bs, num_cam, num_query=h*w, 4]
        reference_points_depth = reference_points_cam[..., 2:3]
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(reference_points_depth, torch.ones_like(reference_points_depth) * eps)

        # Bug!!
        # Correct normalize is
        # reference_points_cam[..., 0] /= ogfW
        # reference_points_cam[..., 1] /= ogfH
        # But for reproducing our results, we use the following normalization
        reference_points_cam[..., 0] /= ogfW
        reference_points_cam[..., 1] /= ogfH

        bev_mask = (reference_points_depth > eps)
        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)                  # shape: (num_cam, bs,h*w, num_points_in_pillar, 2)
        reference_points_depth = reference_points_depth.permute(2, 1, 3, 0, 4)              # shape: (num_cam, bs,h*w, num_points_in_pillar, 1)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)                        # shape: (num_cam, bs,h*w, num_points_in_pillar)

        return reference_points_cam, reference_points_depth, bev_mask


    @force_fp32()
    def forward(self, curr_bev, cam_params, history_fusion_params, dx, bx, history_last_bev=None, last_occ_pred=None, nonempty_prob=None, img_feats=None, spatial_shapes=None,pred_img_depth=None,**kwargs):
        """
        Args:
            curr_bev: [bs, c, z, h, w] 当前帧BEV特征
            cam_params: 相机参数列表，包含外参、内参等
            history_fusion_params: 历史融合参数（包含序列信息等）
            dx: 体素尺寸 (x, y, z)
            bx: 体素偏移
            nonempty_prob: [bs, z, h, w] 体素非空概率
            last_occ_pred: [bs, z, h, w, num_classes] 上一时刻 occupancy 预测
            img_feats: 多尺度图像特征 [bs, num_levels, c, h, w]（新增，用于深度采样）
            spatial_shapes: 图像特征尺度 [num_levels, 2] (h, w)（新增，用于深度采样）
        Returns:
            curr_bev_updated: [bs, c, z, h, w] 融合后BEV特征
        """
        
        # -------------------------- 1. 解析参数后打印核心形状 --------------------------
        # 解析相机参数
        curr_cam_extrins = cam_params[0]  # [bs, num_cams, 4, 4]
        curr_cam_intrins = cam_params[2]  # [bs, num_cams, 4, 4]
        forward_augs = cam_params[4]      # [bs, 4, 4] 前向变换矩阵
        self.num_cams = curr_cam_extrins.shape[1]  # 从外参中获取相机数量
        bs, c_, z, h, w = curr_bev.shape
        device = curr_bev.device
        mc = self.history_num * c_        # 历史特征总通道数
        self.history_forward_augs = forward_augs.clone()


        ref_3d = self.get_reference_points(
            h, w, self.pc_range[5] - self.pc_range[2], z, dim='3d', bs=bs, device=device, dtype=curr_bev.dtype) # torch.Size([3, 2, 625, 3])
        ref_2d = self.get_reference_points(
            h, w, dim='2d', bs=bs, device=device, dtype=curr_bev.dtype) #torch.Size([3, 625, 1, 2])
        reference_points_cam, reference_points_depth, bev_mask = self.point_sampling(ref_3d, self.pc_range, img_metas=kwargs['img_metas'], cam_params=cam_params)
        reference_points_cam = reference_points_cam.reshape( self.num_cams,bs, -1, 2)
        reference_points_depth = reference_points_depth.reshape( self.num_cams,bs, -1, 1)
        bev_mask = bev_mask.reshape( self.num_cams,bs, -1,1)
        indexes = [[] for _ in range(bs)]
        spatial_shapes =[]
        spatial_shapes.append([16, 44])
        spatial_shapes = torch.tensor(spatial_shapes).to(device)
        pred_img_depth = pred_img_depth.view(bs * 6, -1, spatial_shapes[0][0], spatial_shapes[0][1])
        pred_img_depth = pred_img_depth.flatten(2).permute(0, 2, 1)  
        max_len = 0
        slots = torch.zeros(list([ref_3d.shape[0],ref_3d.shape[1]])+[88]).to(ref_3d)
        for j in range(bs):
            for i, mask_per_img in enumerate(bev_mask):
                index_query_per_img = mask_per_img[j].sum(-1).nonzero().squeeze(-1)
                if len(index_query_per_img) == 0:
                    index_query_per_img = bev_mask[i][j].sum(-1).nonzero().squeeze(-1)[0:1]
                indexes[j].append(index_query_per_img)
                # for batch operation, we need to pad the indexes to the same length
                max_len = max(max_len, len(index_query_per_img))
        reference_points_cam_rebatch = reference_points_cam.new_zeros([bs, self.num_cams, max_len, 2])
        reference_points_depth_rebatch = reference_points_depth.new_zeros([bs, self.num_cams, max_len, 1])

        for j in range(bs):
            for i, (reference_points_per_img, reference_points_depth_per_img) in enumerate(zip(reference_points_cam, reference_points_depth)):
                index_query_per_img = indexes[j][i]
                reference_points_cam_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]
                reference_points_depth_rebatch[j, i, :len(index_query_per_img)] = reference_points_depth_per_img[j, index_query_per_img]

        #use deformble attn
        depth_reference_points = reference_points_cam_rebatch.reshape(bs*6, max_len, 1, 1, 1, 2).contiguous()
        depth_attention_weights = torch.ones_like(depth_reference_points[..., 0]).contiguous()
        pred_img_depth = pred_img_depth.unsqueeze(2).contiguous()
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        depth_output = MultiScaleDeformableAttnFunction_fp32.apply(pred_img_depth, spatial_shapes,level_start_index,depth_reference_points,depth_attention_weights, self.im2col_step)
        depth_output = depth_output.reshape(bs,6, max_len,z, -1)   # [bs*num_cam, num_query, num_Z_anchors, C]


        

        for j in range(bs):
            for i in range(6):
                index_query_per_img = indexes[j][i]
                slots[j, index_query_per_img] += depth_output[j, i, :len(index_query_per_img)]

        # output
        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None,None]
        slots = (1-slots.softmax(dim=-1).cumsum(dim=-1))+1e-6
        # slots = self.output_proj(slots)

        #到这里slots就是可见性的概率分布了
        


        # 打印解析后关键变量形状
        print("="*50)
        print("1. 解析参数后核心变量形状：")
        print(f"curr_bev: {curr_bev.shape} (预期：[bs, c, z, h, w])")
        print(f"curr_cam_extrins: {curr_cam_extrins.shape} (预期：[bs, num_cams, 4, 4])")
        print(f"curr_cam_intrins: {curr_cam_intrins.shape} (预期：[bs, num_cams, 4, 4])")
        print(f"forward_augs: {forward_augs.shape} (预期：[bs, 4, 4])")
        print(f"dx: {dx.shape if hasattr(dx, 'shape') else type(dx)} (预期：[3])")
        print(f"bx: {bx.shape if hasattr(bx, 'shape') else type(bx)} (预期：[3])")
        print(f"bs: {bs}, c_: {c_}, z: {z}, h: {h}, w: {w} (BEV特征维度)")
        print("="*50)

        # -------------------------- 2. 初始化历史缓存后打印 --------------------------
        if self.history_bev is None:
            self.history_bev = curr_bev.repeat(1, self.history_num, 1, 1, 1)  # [bs, mc, z, h, w]
            self.history_cam_intrins = curr_cam_intrins.unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)  # [bs, T, num_cams, 4, 4]
            self.history_cam_extrins = curr_cam_extrins.unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)  # [bs, T, num_cams, 4, 4]

        # 打印历史缓存形状
        print("\n2. 历史缓存初始化后形状：")
        print(f"history_bev: {self.history_bev.shape} (预期：[bs, mc, z, h, w]，mc={mc})")
        print(f"history_cam_intrins: {self.history_cam_intrins.shape} (预期：[bs, history_num, num_cams, 4, 4])")
        print(f"history_cam_extrins: {self.history_cam_extrins.shape} (预期：[bs, history_num, num_cams, 4, 4])")

        # -------------------------- 3. 生成网格和BEV变换矩阵后打印 --------------------------
        # 处理新序列（略，不影响维度）
        start_of_sequence = history_fusion_params.get('start_of_sequence', torch.zeros(bs, dtype=torch.bool, device=device))
        if start_of_sequence.any():
            self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_num, 1, 1, 1)
            self.history_cam_intrins[start_of_sequence] = curr_cam_intrins[start_of_sequence].unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)
            self.history_cam_extrins[start_of_sequence] = curr_cam_extrins[start_of_sequence].unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)

        # 生成体素网格和BEV变换矩阵
        grid = self.generate_grid(curr_bev)
        grid_3d = grid
        feat2bev = self.generate_feat2bev(grid, dx, bx)

        # 打印网格和变换矩阵形状
        print("\n3. 生成网格和BEV变换矩阵后形状：")
        print(f"grid_3d (体素网格): {grid_3d.shape} (关键！预期：[bs, h, w, z, 3] 或 [bs, w, h, z, 3])")
        print(f"feat2bev (BEV变换矩阵): {feat2bev.shape} (预期：[bs, 4, 4])")

        # -------------------------- 4. 运动补偿矩阵计算后打印 --------------------------
        # 获取帧间姿态变换
        curr_to_prev_ego_rt = history_fusion_params.get('curr_to_prev_ego_rt', torch.eye(4, device=device).unsqueeze(0).repeat(bs, 1, 1))
        # 计算RT流（坐标变换矩阵）
        rt_flow = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_ego_rt @ torch.inverse(forward_augs) @ feat2bev)
        # 生成齐次网格
        # 在forward函数中，生成grid_hom的位置修正：
        grid_3d = self.generate_grid(curr_bev)  # 现在形状：[3, 25, 25, 2, 3]（bs, h, w, z, 3）
        # 生成齐次坐标（x,y,z,1），并添加最后一个维度（用于矩阵乘法）
        grid_hom = torch.cat([
            grid_3d,  # [3,25,25,2,3]
            torch.ones_like(grid_3d[..., :1])  # [3,25,25,2,1]（补充1作为齐次坐标）
        ], dim=-1).unsqueeze(-1)  # 最终形状：[3,25,25,2,4,1]（符合预期）
        # 打印运动补偿相关形状（矩阵乘法前关键检查）
        print("\n4. 运动补偿矩阵计算后形状（矩阵乘法前）：")
        print(f"curr_to_prev_ego_rt (帧间姿态): {curr_to_prev_ego_rt.shape} (预期：[bs, 4, 4])")
        print(f"rt_flow (变换流): {rt_flow.shape} (预期：[bs, 4, 4])")
        print(f"grid_hom (齐次网格): {grid_hom.shape} (关键！预期：[bs, h, w, z, 4, 1]，需与rt_flow广播匹配)")
        print(f"rt_flow.view后: {rt_flow.view(bs, 1, 1, 1, 4, 4).shape} (预期：[bs, 1, 1, 1, 4, 4])")

        # -------------------------- 5. 网格变换后打印（解决之前维度错的核心） --------------------------
        try:
            grid_transformed = rt_flow.view(bs, 1, 1, 1, 4, 4) @ grid_hom  # 矩阵乘法：[bs, h, w, z, 4, 1]
            print("\n5. 网格变换后形状（矩阵乘法成功！）：")
            print(f"grid_transformed: {grid_transformed.shape} (预期：[bs, h, w, z, 4, 1])")
        except RuntimeError as e:
            print(f"\n5. 网格变换矩阵乘法报错！错误信息：{str(e)}")
            print(f"  - rt_flow.view形状: {rt_flow.view(bs, 1, 1, 1, 4, 4).shape}")
            print(f"  - grid_hom形状: {grid_hom.shape}")
            print("  提示：需确保grid_hom的第1-4维度与rt_flow.view的第2-5维度匹配（广播规则）")
            raise e  # 继续抛出错误，方便定位

        # -------------------------- 6. 采样网格生成后打印 --------------------------
        # 生成采样网格（归一化到[-1,1]，适配F.grid_sample）
        normalize_factor = torch.tensor([w - 1.0, h - 1.0, z - 1.0], dtype=curr_bev.dtype, device=device)
        grid_sampler = grid_transformed[..., :3, 0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0  # [bs, h, w, z, 3]
        # 调整采样网格维度（适配F.grid_sample输入：[bs, z, h, w, 3]）
        grid_sampler_permuted = grid_sampler.permute(0, 3, 1, 2, 4)  # 交换z和h/w维度

        print("\n6. 采样网格生成后形状：")
        print(f"grid_sampler (归一化后): {grid_sampler.shape} (预期：[bs, h, w, z, 3])")
        print(f"grid_sampler_permuted (适配采样): {grid_sampler_permuted.shape} (预期：[bs, z, h, w, 3])")

        # -------------------------- 7. 历史BEV采样后打印 --------------------------
        # 采样历史BEV特征
        sampled_history_bev = F.grid_sample(
            self.history_bev.reshape(bs, mc, z, h, w),  # 输入：[bs, mc, z, h, w]
            grid_sampler_permuted,
            align_corners=True,
            mode='bilinear'
        )

        print("\n7. 历史BEV采样后形状：")
        print(f"history_bev.reshape: {self.history_bev.reshape(bs, mc, z, h, w).shape} (预期：[bs, mc, z, h, w])")
        print(f"sampled_history_bev: {sampled_history_bev.shape} (预期：[bs, mc, z, h, w])")

        # -------------------------- 8. 可见性计算后打印 --------------------------
        # 计算当前帧可见性
        V_curr = self.compute_visibility(
            grid_3d, 
            cam_intrins=curr_cam_intrins,
            cam_extrins=curr_cam_extrins,
            img_shape=self.img_shape,
            img_feats=img_feats,
            spatial_shapes=spatial_shapes
        )
        # 计算历史帧可见性
        prev_cam_intrins = self.history_cam_intrins[:, -1]
        prev_cam_extrins = self.history_cam_extrins[:, -1]
        V_prev = self.compute_visibility(
            grid_3d,
            cam_intrins=prev_cam_intrins,
            cam_extrins=prev_cam_extrins,
            img_shape=self.img_shape,
            img_feats=img_feats,
            spatial_shapes=spatial_shapes
        )

        print("\n8. 可见性计算后形状：")
        print(f"V_curr (当前可见性): {V_curr.shape} (预期：[bs, h, w, z])")
        print(f"V_prev (历史可见性): {V_prev.shape} (预期：[bs, h, w, z])")

        # -------------------------- 9. 稀疏采样前展平变量打印 --------------------------
        # 展平变量（用于稀疏采样）
        curr_bev_flat = curr_bev.reshape(bs, c_, -1)  # [bs, c_, N], N=h*w*z
        history_bev_flat = sampled_history_bev.reshape(bs, mc, -1)  # [bs, mc, N]
        V_prev_flat = V_prev.reshape(bs, -1)  # [bs, N]
        V_curr_flat = V_curr.reshape(bs, -1)  # [bs, N]
        nonempty_prob_flat = nonempty_prob.reshape(bs, -1)  # [bs, N]
        total_voxels = nonempty_prob_flat.shape[1]

        print("\n9. 稀疏采样前展平变量形状：")
        print(f"curr_bev_flat: {curr_bev_flat.shape} (预期：[bs, c_, N], N={total_voxels})")
        print(f"history_bev_flat: {history_bev_flat.shape} (预期：[bs, mc, N])")
        print(f"nonempty_prob_flat: {nonempty_prob_flat.shape} (预期：[bs, N])")
        print(f"total_voxels (h*w*z): {total_voxels} (预期：{h*w*z})")

        # -------------------------- 10. 前景/背景索引及融合后打印（可选，确认后续维度） --------------------------
        # 生成前景/背景索引
        fg_indices = torch.topk(nonempty_prob_flat, self.top_k, dim=1)[1]
        bg_indices = torch.topk(1 - nonempty_prob_flat, total_voxels - self.top_k, dim=1)[1]
        # 提取前景特征（示例，其他融合步骤类似）
        fg_history_feat = torch.gather(history_bev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, mc, 1))

        print("\n10. 前景/背景索引及特征提取后形状：")
        print(f"fg_indices (前景索引): {fg_indices.shape} (预期：[bs, top_k])")
        print(f"bg_indices (背景索引): {bg_indices.shape} (预期：[bs, N-top_k])")
        print(f"fg_history_feat (前景历史特征): {fg_history_feat.shape} (预期：[bs, mc, top_k])")
        print("="*50)

        # -------------------------- 后续原有逻辑（略，维度已通过打印确认） --------------------------
        # 8. 前景融合（原有代码）
        # 9. 背景融合（原有代码）
        # 10. 更新当前BEV特征（原有代码）
        curr_bev_flat = curr_bev.reshape(bs, c_, -1)  # [bs, c_, N] N=h*w*z
        history_bev_flat = sampled_history_bev.reshape(bs, mc, -1)  # [bs, mc, N]
        V_prev_flat = V_prev.reshape(bs, -1)  # [bs, N]
        V_curr_flat = V_curr.reshape(bs, -1)  # [bs, N]
        nonempty_prob_flat = nonempty_prob.reshape(bs, -1)  # [bs, N]
        total_voxels = nonempty_prob_flat.shape[1]

        fg_indices = torch.topk(nonempty_prob_flat, self.top_k, dim=1)[1]  # [bs, top_k]
        bg_indices = torch.topk(1 - nonempty_prob_flat, total_voxels - self.top_k, dim=1)[1]  # [bs, N-top_k]

        # 前景特征提取
        fg_history_feat = torch.gather(history_bev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, mc, 1))  # [bs, mc, top_k]
        fg_curr_feat = torch.gather(curr_bev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, c_, 1))  # [bs, c_, top_k]

        # 历史特征时间聚合
        fg_history_feat_time = fg_history_feat.reshape(bs, self.history_num, c_, self.top_k)  # [bs, T, c_, K]
        time_weights = torch.exp(-0.5 * torch.arange(self.history_num, device=device)).view(1, self.history_num, 1, 1)
        fg_history_agg = (fg_history_feat_time * time_weights).sum(dim=1)  # [bs, c_, K]

        # 可见性聚合与门控
        fg_V_prev = torch.gather(V_prev_flat, dim=1, index=fg_indices)  # [bs, K]
        fg_V_curr = torch.gather(V_curr_flat, dim=1, index=fg_indices)  # [bs, K]
        fg_V_prev_time = fg_V_prev.unsqueeze(1).repeat(1, self.history_num, 1)  # [bs, T, K]
        fg_V_prev_agg = (fg_V_prev_time * time_weights.squeeze(-1)).sum(dim=1)  # [bs, K]
        fg_w_hist, fg_w_curr = self.compute_gate_weights(fg_V_prev_agg, fg_V_curr)  # [bs, K, 1]

        # 前景融合
        fg_history_agg_perm = fg_history_agg.permute(0, 2, 1)  # [bs, K, c_]
        fg_curr_perm = fg_curr_feat.permute(0, 2, 1)  # [bs, K, c_]
        fg_fused = fg_w_hist * fg_history_agg_perm + fg_w_curr * fg_curr_perm  # [bs, K, c_]

        # occupancy嵌入融合
        last_occ_reshaped = last_occ_pred.permute(0, 3, 2, 1, 4).reshape(bs, -1, last_occ_pred.shape[-1])  # [bs, N, num_classes]
        fg_occ_feat = torch.gather(last_occ_reshaped, dim=1, index=fg_indices.unsqueeze(-1).repeat(1, 1, last_occ_pred.shape[-1]))  # [bs, K, num_classes]
        fg_occ_embed = self.occ_embedding(fg_occ_feat).permute(0, 2, 1)  # [bs, occ_embedims, K]
        fg_fused = torch.cat([fg_fused, fg_occ_embed.permute(0, 2, 1)], dim=-1)  # [bs, K, c_+occ_embedims]
        fg_fused = self.history_fusion_linear(fg_fused).permute(0, 2, 1)  # [bs, c_, K]

        # 背景融合（原有代码）
        bg_history_feat = torch.gather(history_bev_flat[:, :mc//2], dim=2, index=bg_indices.unsqueeze(1).repeat(1, mc//2, 1))  # [bs, mc//2, bg_k]
        bg_curr_feat = torch.gather(curr_bev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, c_, 1))  # [bs, c_, bg_k]

        bg_history_feat_time = bg_history_feat.reshape(bs, self.history_num, c_//2, -1)  # [bs, T, c_//2, bg_k]
        bg_history_agg = (bg_history_feat_time * time_weights).sum(dim=1)  # [bs, c_//2, bg_k]
        bg_history_agg_perm = F.pad(bg_history_agg.permute(0, 2, 1), (0, c_ - c_//2, 0, 0))  # [bs, bg_k, c_]

        bg_V_prev = torch.gather(V_prev_flat, dim=1, index=bg_indices)  # [bs, bg_k]
        bg_V_curr = torch.gather(V_curr_flat, dim=1, index=bg_indices)  # [bs, bg_k]
        bg_V_prev_time = bg_V_prev.unsqueeze(1).repeat(1, self.history_num, 1)  # [bs, T, bg_k]
        bg_V_prev_agg = (bg_V_prev_time * time_weights.squeeze(-1)).sum(dim=1)  # [bs, bg_k]
        bg_w_hist, bg_w_curr = self.compute_gate_weights(bg_V_prev_agg, bg_V_curr)  # [bs, bg_k, 1]

        bg_curr_perm = bg_curr_feat.permute(0, 2, 1)  # [bs, bg_k, c_]
        bg_fused = bg_w_hist * bg_history_agg_perm + bg_w_curr * bg_curr_perm  # [bs, bg_k, c_]
        bg_occ_feat = torch.gather(last_occ_reshaped, dim=1, index=bg_indices.unsqueeze(-1).repeat(1, 1, last_occ_pred.shape[-1]))  # [bs, bg_k, num_classes]
        bg_occ_embed = self.occ_embedding(bg_occ_feat).permute(0, 2, 1)  # [bs, occ_embedims, bg_k]
        bg_fused = torch.cat([bg_fused, bg_occ_embed.permute(0, 2, 1)], dim=-1)  # [bs, bg_k, c_+occ_embedims]
        bg_fused = self.history_fusion_bg_linear(bg_fused).permute(0, 2, 1)  # [bs, c_, bg_k]

        # 更新当前BEV
        curr_bev_updated = curr_bev_flat.clone()
        curr_bev_updated.scatter_add_(dim=2, index=fg_indices.unsqueeze(1).repeat(1, c_, 1), src=fg_fused)
        curr_bev_updated.scatter_add_(dim=2, index=bg_indices.unsqueeze(1).repeat(1, c_, 1), src=bg_fused)
        curr_bev_updated = curr_bev_updated.reshape(bs, c_, z, h, w)  # 恢复原形状

        # 更新历史缓存
        self.history_last_bev = curr_bev_updated.detach().clone()
        self.history_bev = torch.cat([self.history_bev[:, c_:, ...], curr_bev], dim=1).detach()
        self.history_cam_intrins = torch.cat([self.history_cam_intrins[:, 1:, ...], curr_cam_intrins.unsqueeze(1)], dim=1).detach()
        self.history_cam_extrins = torch.cat([self.history_cam_extrins[:, 1:, ...], curr_cam_extrins.unsqueeze(1)], dim=1).detach()

        return curr_bev_updated

    def generate_grid(self, curr_bev):
        """生成3D体素网格坐标（非齐次，不含多余维度）"""
        # curr_bev形状：[bs, c, z, h, w] → 解析维度
        bs, c, z_dim, h_dim, w_dim = curr_bev.shape  # z_dim=2, h_dim=25, w_dim=25（对应你的打印结果）
        device = curr_bev.device

        # 生成x（宽）、y（高）、z（深）轴坐标（范围0到维度-1）
        x = torch.linspace(0, w_dim - 1, w_dim, device=device)  # [25]（w=25）
        y = torch.linspace(0, h_dim - 1, h_dim, device=device)  # [25]（h=25）
        z = torch.linspace(0, z_dim - 1, z_dim, device=device)  # [2]（z=2）

        # 生成网格（indexing='ij'确保坐标顺序正确）
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')  # 形状均为 [25, 25, 2]（w, h, z）

        # 合并为3D坐标，并扩展批次维度
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # [25, 25, 2, 3]（w, h, z, 3）
        grid = grid.unsqueeze(0).repeat(bs, 1, 1, 1, 1)  # [3, 25, 25, 2, 3]（bs, w, h, z, 3）

        # 调整维度顺序为 [bs, h, w, z, 3]（匹配后续h=25, w=25的逻辑）
        grid = grid.permute(0, 2, 1, 3, 4)  # [3, 25, 25, 2, 3]（bs, h, w, z, 3）
        return grid

    def generate_feat2bev(self, grid, dx, bx):
        """生成特征到BEV的变换矩阵（适配批次维度）"""
        bs = grid.shape[0]  # 从网格中获取批次大小（3）
        device = grid.device

        # 初始化变换矩阵（单位矩阵）
        feat2bev = torch.eye(4, device=device).unsqueeze(0).repeat(bs, 1, 1)  # [3, 4, 4]（关键：按批次重复）

        # 填充缩放（体素尺寸）和偏移（bx）
        feat2bev[:, 0, 0] = dx[0]  # x轴缩放
        feat2bev[:, 1, 1] = dx[1]  # y轴缩放
        feat2bev[:, 2, 2] = dx[2]  # z轴缩放
        feat2bev[:, 0, 3] = bx[0]  # x轴偏移
        feat2bev[:, 1, 3] = bx[1]  # y轴偏移
        feat2bev[:, 2, 3] = bx[2]  # z轴偏移
        return feat2bev

class DeformableDepthSampler(nn.Module):
    def __init__(self, embed_dims=256, num_heads=8, num_levels=4, num_points=4):
        super().__init__()
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        # 采样偏移预测（输入：体素特征/坐标编码，输出：采样点偏移）
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        # 注意力权重预测
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        # 初始化偏移量（参考BEVFormer的螺旋初始化）
        self._init_weights()

    def _init_weights(self):
        # constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        self.sampling_offsets.bias.data = grid_init.view(-1)
        # constant_init(self.attention_weights, 0.)

    def forward(self, query, value, reference_points, spatial_shapes):
        """
        Args:
            query: 体素特征编码 [bs, num_voxels, embed_dims]
            value: 多尺度图像特征 [bs, num_levels, c, h, w] -> 展平后 [bs, num_levels*h*w, c]
            reference_points: 体素投影到图像的初始坐标 [bs, num_voxels, num_levels, 2]（归一化到[0,1]）
            spatial_shapes: 各尺度图像的高宽 [num_levels, 2] (h, w)
        Returns:
            sampled_depth: 采样得到的深度特征 [bs, num_voxels]
        """
        bs, num_voxels, _ = query.shape
        num_levels = self.num_levels
        num_heads = self.num_heads
        num_points = self.num_points

        # 1. 预测采样偏移和注意力权重
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_voxels, num_heads, num_levels, num_points, 2)  # [bs, N, H, L, P, 2]
        attention_weights = self.attention_weights(query).view(
            bs, num_voxels, num_heads, num_levels * num_points)  # [bs, N, H, L*P]
        attention_weights = attention_weights.softmax(-1).view(
            bs, num_voxels, num_heads, num_levels, num_points)  # [bs, N, H, L, P]

        # 2. 计算采样位置（参考点 + 偏移，归一化到图像坐标）
        offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)  # [L, 2] (w, h)
        sampling_locations = reference_points[:, :, None, :, None, :] + \
                             sampling_offsets / offset_normalizer[None, None, None, :, None, :]  # [bs, N, H, L, P, 2]

        # 3. 多尺度图像特征展平（适配采样函数）
        value_flat = []
        level_start_index = [0]
        for i in range(num_levels):
            h, w = spatial_shapes[i]
            value_flat.append(value[:, i].flatten(2).transpose(1, 2))  # [bs, h*w, c]
            level_start_index.append(level_start_index[-1] + h * w)
        value_flat = torch.cat(value_flat, dim=1)  # [bs, total_num_pixels, c]
        level_start_index = torch.tensor(level_start_index, device=value.device)

        # 4. 调用可变形注意力采样函数（复用MultiScaleDeformableAttnFunction）
        # from tfd.mmdet3d.models.stcocc.view_transformation.backward_projection.bevformer_utils import MultiScaleDeformableAttnFunction_fp32
        sampled_feat = MultiScaleDeformableAttnFunction_fp32.apply(
            value_flat, spatial_shapes, level_start_index,
            reference_points, attention_weights,
            sampling_locations, self.num_heads, self.im2col_step
        )  # [bs, num_voxels, c]

        # 5. 从采样特征中提取深度信息（假设最后一维包含深度相关特征）
        sampled_depth = sampled_feat.mean(dim=-1)  # 简化：取特征均值作为深度估计
        return sampled_depth






@HEADS.register_module()
class GatedTemporalFusion5(BaseModule):
    def __init__(
        self,
        history_num=4,
        depth_sampler_embed_dims=256,
        depth_sampler_num_heads=8,
        depth_sampler_num_levels=4,
        depth_sampler_num_points=4,
        im2col_step=64,
        top_k=500,
        single_bev_num_channels=96,
        occ_embedims=32,
        num_classes=18,
        vis_theta=0.3,
        vis_beta=10.0,
        vis_gamma=0.4,
        vis_sigma=0.1,
        nonempty_thresh=0.1,
        max_step_ratio=1.2,
        **kwargs
    ):
        super(GatedTemporalFusion5, self).__init__()

        # 基础参数
        self.history_num = history_num
        self.top_k = top_k
        self.single_bev_num_channels = single_bev_num_channels
        self.occ_embedims = occ_embedims
        self.fg_scale = nn.Parameter(torch.full((history_num,), 10.0))
        self.bg_scale = nn.Parameter(torch.full((history_num,), 10.0))
        # 可见性门控参数
        # self.vis_theta = vis_theta  # 可见性阈值
        # self.vis_beta = vis_beta    # 敏感度参数
        # self.vis_gamma = vis_gamma  # 场景4固定权重
        # self.vis_sigma = vis_sigma  # 软化参数
        self.vis_theta = nn.Parameter(torch.tensor(vis_theta))   # 可见性阈值
        self.vis_beta = nn.Parameter(torch.tensor(vis_beta))     # 敏感度参数
        self.vis_gamma = nn.Parameter(torch.tensor(vis_gamma))   # 场景4固定权重
        self.vis_sigma = nn.Parameter(torch.tensor(vis_sigma))  

        
        # 历史特征缓存（新增history_last_bev）
        self.history_bev = None  # 缓存历史多帧BEV特征
        self.history_last_bev = None  # 缓存上一帧最终融合后的BEV特征（关键新增）
        self.history_cam_intrins = None
        self.history_cam_extrins = None
        
        # 网络层（保持不变）
        self.occ_embedding = nn.Sequential(
            nn.Linear(num_classes, occ_embedims),
            nn.Softplus(),
            nn.Linear(occ_embedims, occ_embedims),
        )
        self.history_fusion_linear = nn.Sequential(
            nn.Linear(single_bev_num_channels + occ_embedims, single_bev_num_channels),
            nn.Softplus(),
            nn.Linear(single_bev_num_channels, single_bev_num_channels),
        )
        self.history_fusion_bg_linear = nn.Sequential(
            nn.Linear(single_bev_num_channels + occ_embedims, single_bev_num_channels),
            nn.Softplus(),
            nn.Linear(single_bev_num_channels, single_bev_num_channels),
        )
        
        # 可见性计算组件（保持不变）
        self.rt_vis_calculator = EfficientRayTracingVisibility(
            nonempty_thresh=nonempty_thresh,
            max_step_ratio=max_step_ratio
        )
        # self.img_shape = (900, 1600)  # 默认图像尺寸

        self.depth_sampler = DeformableDepthSampler(
            embed_dims=depth_sampler_embed_dims,
            num_heads=depth_sampler_num_heads,
            num_levels=depth_sampler_num_levels,
            num_points=depth_sampler_num_points
        )
        self.history_forward_augs = None  # 用于缓存历史帧的变换矩阵（如BDAM矩阵）
        self.im2col_step = im2col_step
        # self.voxel_encoder = nn.Linear(single_bev_num_channels, depth_sampler_embed_dims)
        self.dbound = [1.0, 45.0, 0.5]
        self.pc_range = [-40, -40, -1.0, 40, 40, 5.4]
        self.final_dim = (256, 704)
    def compute_visibility(self, grid, cam_intrins, cam_extrins, img_shape, img_feats, spatial_shapes):
        """
        升级：结合可变形注意力采样的深度值优化可见性计算
        Args:
            img_feats: 多尺度图像特征 [bs, num_levels, c, h, w]
            spatial_shapes: 图像特征尺度 [num_levels, 2] (h, w)
            其他参数同原函数
        Returns:
            vis_prob: [bs, h, w, z] 优化后的可见性概率
        """
        bs, h, w, z, _ = grid.shape
        h_img, w_img = img_shape
        device = grid.device
        num_voxels = h * w * z  # 体素总数

        # 1. 原有相机投影逻辑（计算图像坐标和初始可见性）
        # 1.1 体素坐标→相机坐标→图像坐标
        grid_cam = grid.unsqueeze(1).expand(bs, self.num_cams, h, w, z, 3)  # [bs, num_cams, h, w, z, 3]
        grid_flat = grid_cam.reshape(-1, num_voxels, 3)  # [bs*num_cams, N, 3]
        grid_hom = torch.cat([grid_flat, torch.ones_like(grid_flat[..., :1])], dim=-1)  # [bs*num_cams, N, 4]

        cam_intrins_flat = cam_intrins.reshape(-1, 4, 4)  # [bs*num_cams, 4, 4]
        cam_extrins_flat = cam_extrins.reshape(-1, 4, 4)  # [bs*num_cams, 4, 4]
        extrins_inv = torch.inverse(cam_extrins_flat)
        cam_coords = torch.bmm(extrins_inv[:, :3, :4], grid_hom.permute(0, 2, 1)).permute(0, 2, 1)  # [bs*num_cams, N, 3]
        depth = cam_coords[..., 2:3] + 1e-8  # 相机坐标系下的深度

        # 1.2 计算初始可见性（原逻辑）
        img_coords = torch.bmm(cam_intrins_flat[:, :3, :3], cam_coords.permute(0, 2, 1)).permute(0, 2, 1)  # [bs*num_cams, N, 3]
        img_xy = img_coords[..., :2] / depth  # [bs*num_cams, N, 2] (u, v)
        depth_valid = (cam_coords[..., 2] > 0).float()  # [bs*num_cams, N]
        u_valid = (img_xy[..., 0] >= 0) & (img_xy[..., 0] < w_img)
        v_valid = (img_xy[..., 1] >= 0) & (img_xy[..., 1] < h_img)
        img_valid = (u_valid & v_valid).float()  # [bs*num_cams, N]
        initial_vis = depth_valid * img_valid  # [bs*num_cams, N]

        # 2. 可变形注意力深度采样
        # 2.1 准备输入：体素特征编码
        # voxel_feat = self.voxel_encoder(self.curr_bev_feat)  # [bs, c_, z, h, w] → [bs, z*h*w, embed_dims]（需提前展平体素特征）
        # voxel_feat = voxel_feat.reshape(bs, num_voxels, -1)  # [bs, N, embed_dims]

        # 2.2 生成参考点（归一化到[0,1]）
        norm_img_xy = img_xy / torch.tensor([w_img, h_img], device=device).view(1, 1, 2)  # [bs*num_cams, N, 2]
        # 取主相机（如第0个相机）的参考点作为采样基准
        ref_points = norm_img_xy.reshape(bs, self.num_cams, num_voxels, 2)[:, 0]  # [bs, N, 2]
        ref_points = ref_points.unsqueeze(2).repeat(1, 1, self.depth_sampler.num_levels, 1)  # [bs, N, L, 2]

        # 2.3 采样深度特征
        sampled_depth = self.depth_sampler(
            query=voxel_feat,
            value=img_feats,  # 多尺度图像特征 [bs, L, c, h, w]
            reference_points=ref_points,
            spatial_shapes=spatial_shapes
        )  # [bs, N]

        # 3. 结合采样深度优化可见性
        # 3.1 深度一致性校验：采样深度与相机投影深度的差异
        cam_depth = depth.reshape(bs, self.num_cams, num_voxels)[:, 0]  # 主相机的投影深度 [bs, N]
        depth_diff = torch.abs(sampled_depth - cam_depth) / (cam_depth + 1e-8)  # 相对深度差
        depth_consistent = (depth_diff < 0.3).float()  # 深度差小于30%则认为有效

        # 3.2 融合可见性：初始可见性 × 深度一致性
        initial_vis = initial_vis.reshape(bs, self.num_cams, num_voxels).max(dim=1)[0]  # [bs, N]（多相机取max）
        vis_prob = initial_vis * depth_consistent  # [bs, N]

        # 4. 还原形状
        return vis_prob.reshape(bs, h, w, z)  # [bs, h, w, z]

    def compute_alpha_unified(self, V_curr, V_prev):
        """统一计算当前帧融合权重α（覆盖四场景）"""
        eps = 1e-8
        # 1. 基础动态权重σ_base（场景1-3）
        ratio = V_curr / (V_curr + V_prev + eps)
        sigma_base = torch.sigmoid(self.vis_beta * (ratio - 0.5))
        
        # 2. 场景4软化掩码σ_both
        mask_curr = torch.sigmoid(-(V_curr - self.vis_theta) / self.vis_sigma)
        mask_prev = torch.sigmoid(-(V_prev - self.vis_theta) / self.vis_sigma)
        sigma_both = mask_curr * mask_prev
        
        # 3. 最终权重计算
        alpha = (1 - sigma_both) * sigma_base + sigma_both * self.vis_gamma
        return alpha.unsqueeze(-1)  # [bs, N, 1]

    def compute_gate_weights(self, V_prev_agg, V_curr):
        """计算历史和当前帧的门控权重"""
        alpha = self.compute_alpha_unified(V_curr, V_prev_agg)
        return 1 - alpha, alpha  # 历史权重，当前权重

    def generate_grid(self, curr_bev, voxel_min, voxel_max, voxel_size):
        """生成体素中心坐标网格（自车坐标系）"""
        bs, c_, z, h, w = curr_bev.shape
        device = curr_bev.device
        
        # 计算体素中心坐标
        x_coords = torch.linspace(
            voxel_min[0] + voxel_size[0]/2, 
            voxel_max[0] - voxel_size[0]/2, 
            w, device=device
        )
        y_coords = torch.linspace(
            voxel_min[1] + voxel_size[1]/2, 
            voxel_max[1] - voxel_size[1]/2, 
            h, device=device
        )
        z_coords = torch.linspace(
            voxel_min[2] + voxel_size[2]/2, 
            voxel_max[2] - voxel_size[2]/2, 
            z, device=device
        )
        
        # 生成网格并扩展维度 [bs, h, w, z, 3]
        x_grid, y_grid, z_grid = torch.meshgrid(x_coords, y_coords, z_coords, indexing='xy')
        grid = torch.stack([x_grid, y_grid, z_grid], dim=-1)  # [w, h, z, 3]
        grid = grid.permute(1, 0, 2, 3)  # [h, w, z, 3]
        return grid.unsqueeze(0).repeat(bs, 1, 1, 1, 1)  # [bs, h, w, z, 3]


    def get_reference_points(self, H, W, Z=None, num_points_in_pillar =4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
        torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    @force_fp32(apply_to=('reference_points', 'cam_params'))
    def point_sampling(self, reference_points, pc_range, img_metas, cam_params=None):
        # prepare for point sampling
        lidar2img = []
        ego2lidar = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])     # lidar2img update the post aug in the loading pipeline
            ego2lidar.append(img_meta['ego2lidar'])
        lidar2img = torch.stack(lidar2img, dim=0).to(reference_points.device)
        ego2lidar = torch.stack(ego2lidar, dim=0).to(reference_points.device)

        sensor2egos, ego2globals, intrins, post_augs, bda_mat = cam_params
        num_cam = sensor2egos.size(1)
        ogfH, ogfW = self.final_dim

        # reference_points defines in the bev space, [bs, D, hxw, 3]
        # change reference_points from bev-ego coordinate to ego coordinate
        reference_points = reference_points.clone()
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
                                     (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
                                     (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
                                     (pc_range[5] - pc_range[2]) + pc_range[2]

        # prepare for point sampling
        reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
        reference_points = reference_points.permute(1, 0, 2, 3)  # shape: (num_points_in_pillar,bs,h*w,4)
        D, B, num_query = reference_points.size()[:3]  # D=num_points_in_pillar , num_query=h*w
        reference_points = reference_points.view(D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)  # shape: (num_points_in_pillar,bs,num_cam,h*w,4)
        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)
        ego2lidar = ego2lidar.view(1, B, 1, 1, 4, 4).repeat(D, 1, num_cam, num_query, 1, 1)
        inverse_bda = bda_mat.view(1, B, 1, 1, 4, 4).repeat(D, 1, num_cam, num_query, 1, 1)

        # change reference_points from ego coordinate to img coordinate
        eps = 1e-5
        reference_points_cam = (lidar2img @ ego2lidar @ inverse_bda @ reference_points).squeeze(-1)   # [num_points_in_pillar, bs, num_cam, num_query=h*w, 4]
        reference_points_depth = reference_points_cam[..., 2:3]
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(reference_points_depth, torch.ones_like(reference_points_depth) * eps)

        # Bug!!
        # Correct normalize is
        # reference_points_cam[..., 0] /= ogfW
        # reference_points_cam[..., 1] /= ogfH
        # But for reproducing our results, we use the following normalization
        reference_points_cam[..., 0] /= ogfW
        reference_points_cam[..., 1] /= ogfH

        bev_mask = (reference_points_depth > eps)
        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)                  # shape: (num_cam, bs,h*w, num_points_in_pillar, 2)
        reference_points_depth = reference_points_depth.permute(2, 1, 3, 0, 4)              # shape: (num_cam, bs,h*w, num_points_in_pillar, 1)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)                        # shape: (num_cam, bs,h*w, num_points_in_pillar)

        return reference_points_cam, reference_points_depth, bev_mask


    @force_fp32()
    def forward(self, curr_bev, cam_params, history_fusion_params, dx, bx, history_last_bev=None, last_occ_pred=None, nonempty_prob=None, img_feats=None, spatial_shapes=None,pred_img_depth=None,**kwargs):
        """
        Args:
            curr_bev: [bs, c, z, h, w] 当前帧BEV特征
            cam_params: 相机参数列表，包含外参、内参等
            history_fusion_params: 历史融合参数（包含序列信息等）
            dx: 体素尺寸 (x, y, z)
            bx: 体素偏移
            nonempty_prob: [bs, z, h, w] 体素非空概率
            last_occ_pred: [bs, z, h, w, num_classes] 上一时刻 occupancy 预测
            img_feats: 多尺度图像特征 [bs, num_levels, c, h, w]（新增，用于深度采样）
            spatial_shapes: 图像特征尺度 [num_levels, 2] (h, w)（新增，用于深度采样）
        Returns:
            curr_bev_updated: [bs, c, z, h, w] 融合后BEV特征
        """
        
        # -------------------------- 1. 解析参数后打印核心形状 --------------------------
        # 解析相机参数
        curr_cam_extrins = cam_params[0]  # [bs, num_cams, 4, 4]
        curr_cam_intrins = cam_params[2]  # [bs, num_cams, 4, 4]
        forward_augs = cam_params[4]      # [bs, 4, 4] 前向变换矩阵
        self.num_cams = curr_cam_extrins.shape[1]  # 从外参中获取相机数量
        bs, c_, z, h, w = curr_bev.shape
        device = curr_bev.device
        mc = self.history_num * c_        # 历史特征总通道数
        # self.history_forward_augs = forward_augs.clone()


        ref_3d = self.get_reference_points(
            h, w, z, z, dim='3d', bs=bs, device=device, dtype=curr_bev.dtype) # torch.Size([3, 2, 625, 3]) #[bs,z,yx,3(x,y,z)]
        # ref_2d = self.get_reference_points(
        #     h, w, dim='2d', bs=bs, device=device, dtype=curr_bev.dtype) #torch.Size([3, 625, 1, 2])
        slots = torch.zeros(list([ref_3d.shape[0],ref_3d.shape[2],ref_3d.shape[1]])).to(ref_3d)
        reference_points_cam, reference_points_depth, bev_mask = self.point_sampling(ref_3d, self.pc_range, img_metas=kwargs['img_metas'], cam_params=cam_params)
        indexes = [[] for _ in range(bs)]
        spatial_shapes =[]
        spatial_shapes.append([16, 44])
        spatial_shapes = torch.tensor(spatial_shapes).to(device)
        pred_img_depth = pred_img_depth.view(bs * 6, -1, spatial_shapes[0][0], spatial_shapes[0][1])
        pred_img_depth = pred_img_depth.flatten(2).permute(0, 2, 1)  
        max_len = 0
        for j in range(bs):
            for i, mask_per_img in enumerate(bev_mask):
                index_query_per_img = mask_per_img[j].sum(-1).nonzero().squeeze(-1)
                if len(index_query_per_img) == 0:
                    index_query_per_img = bev_mask[i][j].sum(-1).nonzero().squeeze(-1)[0:1]
                indexes[j].append(index_query_per_img)
                # for batch operation, we need to pad the indexes to the same length
                max_len = max(max_len, len(index_query_per_img))
        reference_points_cam_rebatch = reference_points_cam.new_zeros([bs, self.num_cams, max_len, z, 2])
        reference_points_depth_rebatch = reference_points_depth.new_zeros([bs, self.num_cams, max_len, z, 1])

        for j in range(bs):
            for i, (reference_points_per_img, reference_points_depth_per_img) in enumerate(zip(reference_points_cam, reference_points_depth)):
                index_query_per_img = indexes[j][i]
                reference_points_cam_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]
                reference_points_depth_rebatch[j, i, :len(index_query_per_img)] = reference_points_depth_per_img[j, index_query_per_img]

        #use deformble attn
        depth_reference_points = reference_points_cam_rebatch.reshape(bs*6, max_len*z, 1, 1, 1, 2).contiguous()
        depth_attention_weights = torch.ones_like(depth_reference_points[..., 0]).contiguous()
        pred_img_depth = pred_img_depth.unsqueeze(2).contiguous()
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))



        bev_query_depth_rebatch = (reference_points_depth_rebatch- self.dbound[0])/ self.dbound[2]
        bev_query_depth_rebatch = torch.clip(torch.floor(bev_query_depth_rebatch), 0, 88-1).to(torch.long)
        bev_query_depth_rebatch = F.one_hot(bev_query_depth_rebatch.squeeze(-1),
                                   num_classes=88)

        depth_output = MultiScaleDeformableAttnFunction_fp32.apply(pred_img_depth, spatial_shapes,level_start_index,depth_reference_points,depth_attention_weights, self.im2col_step)
        depth_output = depth_output.reshape(bs,6, max_len,z, -1)   # [bs*num_cam, num_query, num_Z_anchors, C]
        # reference_points_depth_rebatch

        increment = torch.zeros_like(depth_output)
        increment[..., 0] = 1e-9  # 非原地赋值（创建新张量）
        depth_output = depth_output + increment
        print("depth_output.sum(-1).shape")
        print(depth_output.sum(-1).shape)
        print("depth_output.sum(-1)")
        print(depth_output.sum(-1))
        print("print((depth_output.sum(-1)==0).sum())")
        print((depth_output.sum(-1)==0).sum())
        depth_output =depth_output/depth_output.sum(-1)[...,None] #bs,xy,z,D

        print("depth_output.sum(-1).shape")
        print(depth_output.sum(-1).shape)
        print("depth_output.sum(-1)")
        print(depth_output.sum(-1))
        print("print((depth_output.sum(-1)==1).sum())")
        print((depth_output.sum(-1)>=0.99).sum())
        print("depth_output")
        # print(depth_output)


        depth_output = (1-depth_output.cumsum(dim=-1))


        # print("print((depth_output[...,-1]==0).sum())")
        # print((depth_output[...,-1]==0).sum())
        depth_output = (bev_query_depth_rebatch*depth_output).sum(-1)

        #恢复depth_output的shape
        # fix_depth_output = depth_output.new_zeros([bs, 6, w*h, z, 88])
        # for j in range(bs):
        #     for i in range(6):
        #         index_query_per_img = indexes[j][i]
        #         fix_depth_output[j, i, index_query_per_img] = depth_output[j, i, :len(index_query_per_img)]

        for j in range(bs):
            for i in range(6):
                index_query_per_img = indexes[j][i]
                slots[j, index_query_per_img] = torch.max(slots[j, index_query_per_img],depth_output[j, i, :len(index_query_per_img)])

        # depth_sum = fix_depth_output.sum(dim=-1).view(3,6,25,25,2)
        # bs = depth_sum.shape[0]       # 3
        # num_cam = depth_sum.shape[1]  # 6
        # z_layers = depth_sum.shape[4] # 2
        # H, W = depth_sum.shape[2], depth_sum.shape[3]  # 25,25

        # # 颜色映射：0值用黑色，非0值用渐变色
        # cmap = plt.cm.viridis
        # cmap.set_bad(color='black')  # 0值标记为黑色

        # # 设置子图布局（不变）
        # fig, axes = plt.subplots(
        #     nrows=bs, ncols=num_cam * z_layers,
        #     figsize=(30, 8),
        #     squeeze=False
        # )
        # fig.suptitle('fix_depth_output.sum(-1) 可视化（黑色=0值，颜色=非0值）', fontsize=16, y=0.98)


        # # -------------------------- 2. 循环绘制每个子图（核心修改：添加 .detach()） --------------------------
        # for batch_idx in range(bs):
        #     for cam_idx in range(num_cam):
        #         for z_idx in range(z_layers):
        #             col_idx = cam_idx * z_layers + z_idx
        #             ax = axes[batch_idx, col_idx]
                    
        #             # 核心修改：先 detach() 切断计算图，再转 cpu 和 numpy
        #             data = depth_sum[batch_idx, cam_idx, :, :, z_idx].detach().cpu().numpy()
        #             data[data == 0] = np.nan  # 0值替换为NaN，显示为黑色
                    
        #             # 绘制热力图（修改 vmin/vmax：同样添加 .detach()）
        #             im = ax.imshow(
        #                 data, 
        #                 cmap=cmap, 
        #                 aspect='auto',
        #                 # 关键修改：depth_sum 先 detach 再转 numpy，确保不影响梯度
        #                 vmin=np.nanmin(depth_sum.detach().cpu().numpy()),
        #                 vmax=np.nanmax(depth_sum.detach().cpu().numpy())
        #             )
                    
        #             # 子图标题和坐标轴（不变）
        #             ax.set_title(
        #                 f'Batch{batch_idx+1}\nCam{cam_idx+1} Z{z_idx+1}',
        #                 fontsize=10, pad=5
        #             )
        #             ax.set_xticks([])
        #             ax.set_yticks([])


        # # -------------------------- 3. 添加颜色条（不变） --------------------------
        # cbar = fig.colorbar(
        #     im, 
        #     ax=axes.ravel().tolist(),
        #     shrink=0.8,
        #     pad=0.02
        # )
        # cbar.set_label('Sum of Depth Bins (D=88)', fontsize=12)


        # # -------------------------- 4. 调整布局并保存（不变） --------------------------
        # plt.tight_layout(rect=[0, 0, 0.98, 0.95])
        # plt.savefig('depth_sum_visualization.png', dpi=300, bbox_inches='tight')
        # plt.show()

        # output

        #计数更新，建立在纸上的假设成立的基础上
        # count = bev_mask.sum(-1) > 0
        # count = count.permute(1, 2, 0).sum(-1)
        # count = torch.clamp(count, min=1.0)
        # slots = slots / count[..., None]

        print("slots.shape")
        print(slots.shape)
        print("slots")  
        # print(slots)
        # zzzzz=1/0

        V_curr =slots.view(bs, 1, h, w,z).permute(0, 1, 4, 2, 3)
        # slots[...,0]+=1e-9
        # print("slots.sum(-1).shape")
        # print(slots.sum(-1).shape)
        # print("slots.sum(-1)")
        # print(slots.sum(-1))
        # print("print((slots.sum(-1)==0).sum())")
        # print((slots.sum(-1)==0).sum())
        # slots =slots/slots.sum(-1)[...,None] #bs,xy,z,D

        # print("slots.sum(-1).shape")
        # print(slots.sum(-1).shape)
        # print("slots.sum(-1)")
        # print(slots.sum(-1))
        # print("print((slots.sum(-1)==1).sum())")
        # print("slots")
        # print(slots)


        # slots = (1-slots.cumsum(dim=-1))

        # print("print((slots[...,-1]==0).sum())")
        # print((slots[...,-1]==0).sum())

        # print("slots")
        # print(slots)
        # print("print((slots[...,-1]<0.01).sum())")
        # print((slots[...,-1]<0.01).sum())

        #TODO 这里的对于边界值的考虑，从0开始还是从1开始？




        # slots = self.output_proj(slots)
        #TODO 上面这里需要检查一下
        #这里相当于两次softmax，可能会导致分布变得不够尖锐，需要进一步确认

        #到这里slots就是可见性的概率分布了
        #这里先尝试使用期望值进行计算，使得可微分

        #然后再采用stc的原始离散计算方法，


        # 打印解析后关键变量形状
        print("="*50)
        print("1. 解析参数后核心变量形状：")
        print(f"curr_bev: {curr_bev.shape} (预期：[bs, c, z, h, w])")
        print(f"curr_cam_extrins: {curr_cam_extrins.shape} (预期：[bs, num_cams, 4, 4])")
        print(f"curr_cam_intrins: {curr_cam_intrins.shape} (预期：[bs, num_cams, 4, 4])")
        print(f"forward_augs: {forward_augs.shape} (预期：[bs, 4, 4])")
        print(f"dx: {dx.shape if hasattr(dx, 'shape') else type(dx)} (预期：[3])")
        print(f"bx: {bx.shape if hasattr(bx, 'shape') else type(bx)} (预期：[3])")
        print(f"bs: {bs}, c_: {c_}, z: {z}, h: {h}, w: {w} (BEV特征维度)")
        print("="*50)

        if type(history_fusion_params['sequence_group_idx']) is list:
            seq_ids = history_fusion_params['sequence_group_idx'][0]
        else:
            seq_ids = history_fusion_params['sequence_group_idx']
        if type(history_fusion_params['start_of_sequence']) is list:
            start_of_sequence = history_fusion_params['start_of_sequence'][0]
        else:
            start_of_sequence = history_fusion_params['start_of_sequence']
        if type(history_fusion_params['curr_to_prev_ego_rt']) is list:
            curr_to_prev_ego_rt = history_fusion_params['curr_to_prev_ego_rt'][0]
        else:
            curr_to_prev_ego_rt = history_fusion_params['curr_to_prev_ego_rt']
        forward_augs = cam_params[-1]  # bda

        # check seq_ids > 0
        assert (seq_ids >= 0).all()
        # -------------------------- 2. 初始化历史缓存后打印 --------------------------
        if self.history_bev is None:
            # self.history_bev = curr_bev.repeat(1, self.history_num, 1, 1, 1)  # [bs, mc, z, h, w]
            # self.history_cam_intrins = curr_cam_intrins.unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)  # [bs, T, num_cams, 4, 4]
            # self.history_cam_extrins = curr_cam_extrins.unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)  # [bs, T, num_cams, 4, 4]
            # self.history_bev = curr_bev.clone()
            self.history_forward_augs = forward_augs.clone()
            self.history_seq_ids = seq_ids.clone()
            self.history_bev = curr_bev.repeat(1, self.history_num, 1, 1, 1)
            self.history_sweep_time = curr_bev.new_zeros(curr_bev.shape[0], self.history_num)
            self.history_visibility = V_curr.repeat(1, self.history_num, 1, 1, 1)
        self.history_bev = self.history_bev.detach()
        self.history_visibility = self.history_visibility.detach()
        self.history_sweep_time += 1

        # 打印历史缓存形状
        print("\n2. 历史缓存初始化后形状：")
        print(f"history_bev: {self.history_bev.shape} (预期：[bs, mc, z, h, w]，mc={mc})")
        # print(f"history_cam_intrins: {self.history_cam_intrins.shape} (预期：[bs, history_num, num_cams, 4, 4])")
        # print(f"history_cam_extrins: {self.history_cam_extrins.shape} (预期：[bs, history_num, num_cams, 4, 4])")

        # -------------------------- 3. 生成网格和BEV变换矩阵后打印 --------------------------
        # 处理新序列（略，不影响维度）
        # start_of_sequence = history_fusion_params.get('start_of_sequence', torch.zeros(bs, dtype=torch.bool, device=device))
        if start_of_sequence.sum()>0:
            self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_num, 1, 1, 1)
            # self.history_cam_intrins[start_of_sequence] = curr_cam_intrins[start_of_sequence].unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)
            # self.history_cam_extrins[start_of_sequence] = curr_cam_extrins[start_of_sequence].unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)
            self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]
            self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
            self.history_sweep_time[start_of_sequence] = 0  # zero the new sequence timestep starts
            self.history_visibility[start_of_sequence] = V_curr[start_of_sequence].repeat(1, self.history_num, 1, 1, 1)

        # 生成体素网格和BEV变换矩阵
        grid = self.generate_grid(curr_bev) #[bs,y,x,z,4]
        grid_3d = grid
        feat2bev = self.generate_feat2bev(grid, dx, bx)

        # 打印网格和变换矩阵形状
        print("\n3. 生成网格和BEV变换矩阵后形状：")
        print(f"grid_3d (体素网格): {grid_3d.shape} (关键！预期：[bs, h, w, z, 3] 或 [bs, w, h, z, 3])")
        print(f"feat2bev (BEV变换矩阵): {feat2bev.shape} (预期：[bs, 4, 4])")

        # -------------------------- 4. 运动补偿矩阵计算后打印 --------------------------
        # 获取帧间姿态变换
        # curr_to_prev_ego_rt = history_fusion_params.get('curr_to_prev_ego_rt', torch.eye(4, device=device).unsqueeze(0).repeat(bs, 1, 1))
        # 计算RT流（坐标变换矩阵）
        rt_flow = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_ego_rt @ torch.inverse(forward_augs) @ feat2bev)
        # 生成齐次网格
        # 在forward函数中，生成grid_hom的位置修正：
        # grid_3d = self.generate_grid(curr_bev)  # 现在形状：[3, 25, 25, 2, 3]（bs, h, w, z, 3）
        # # 生成齐次坐标（x,y,z,1），并添加最后一个维度（用于矩阵乘法）
        # grid_hom = torch.cat([
        #     grid_3d,  # [3,25,25,2,3]
        #     torch.ones_like(grid_3d[..., :1])  # [3,25,25,2,1]（补充1作为齐次坐标）
        # ], dim=-1).unsqueeze(-1)  # 最终形状：[3,25,25,2,4,1]（符合预期）
        # # 打印运动补偿相关形状（矩阵乘法前关键检查）
        # print("\n4. 运动补偿矩阵计算后形状（矩阵乘法前）：")
        # print(f"curr_to_prev_ego_rt (帧间姿态): {curr_to_prev_ego_rt.shape} (预期：[bs, 4, 4])")
        # print(f"rt_flow (变换流): {rt_flow.shape} (预期：[bs, 4, 4])")
        # print(f"grid_hom (齐次网格): {grid_hom.shape} (关键！预期：[bs, h, w, z, 4, 1]，需与rt_flow广播匹配)")
        # print(f"rt_flow.view后: {rt_flow.view(bs, 1, 1, 1, 4, 4).shape} (预期：[bs, 1, 1, 1, 4, 4])")

        # # -------------------------- 5. 网格变换后打印（解决之前维度错的核心） --------------------------
        # try:
        #     grid_transformed = rt_flow.view(bs, 1, 1, 1, 4, 4) @ grid_hom  # 矩阵乘法：[bs, h, w, z, 4, 1]
        #     print("\n5. 网格变换后形状（矩阵乘法成功！）：")
        #     print(f"grid_transformed: {grid_transformed.shape} (预期：[bs, h, w, z, 4, 1])")
        # except RuntimeError as e:
        #     print(f"\n5. 网格变换矩阵乘法报错！错误信息：{str(e)}")
        #     print(f"  - rt_flow.view形状: {rt_flow.view(bs, 1, 1, 1, 4, 4).shape}")
        #     print(f"  - grid_hom形状: {grid_hom.shape}")
        #     print("  提示：需确保grid_hom的第1-4维度与rt_flow.view的第2-5维度匹配（广播规则）")
        #     raise e  # 继续抛出错误，方便定位
        bs, mc, z, h, w = self.history_bev.shape
        n, c_, z, h, w = curr_bev.shape
        grid = rt_flow.view(n, 1, 1, 1, 4, 4) @ grid
        # -------------------------- 6. 采样网格生成后打印 --------------------------
        # 生成采样网格（归一化到[-1,1]，适配F.grid_sample）
        normalize_factor = torch.tensor([w - 1.0, h - 1.0, z - 1.0], dtype=curr_bev.dtype, device=device)
        # grid_sampler = grid_transformed[..., :3, 0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0  # [bs, h, w, z, 3]
        # # 调整采样网格维度（适配F.grid_sample输入：[bs, z, h, w, 3]）
        # grid_sampler_permuted = grid_sampler.permute(0, 3, 1, 2, 4)  # 交换z和h/w维度
        grid = grid[:, :, :, :, :3, 0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0   # grid order is x, y, z


        print("\n6. 采样网格生成后形状：")
        # print(f"grid_sampler (归一化后): {grid_sampler.shape} (预期：[bs, h, w, z, 3])")
        # print(f"grid_sampler_permuted (适配采样): {grid_sampler_permuted.shape} (预期：[bs, z, h, w, 3])")

        # -------------------------- 7. 历史BEV采样后打印 --------------------------
        # 采样历史BEV特征
        sampled_history_bev = F.grid_sample(
            self.history_bev.reshape(bs, mc, z, h, w),  # 输入：[bs, mc, z, h, w]
            grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4),
            align_corners=True,
            mode='bilinear'
        )
        sampled_history_visibility = F.grid_sample(
            self.history_visibility.reshape(bs, self.history_num, z, h, w),  # 输入：[bs, mc, z, h, w]
            grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4),
            align_corners=True,
            mode='nearest'
        )
        print("\n7. 历史BEV采样后形状：")
        print(f"history_bev.reshape: {self.history_bev.reshape(bs, mc, z, h, w).shape} (预期：[bs, mc, z, h, w])")
        print(f"sampled_history_bev: {sampled_history_bev.shape} (预期：[bs, mc, z, h, w])")

        # -------------------------- 8. 可见性计算后打印 --------------------------
        # 计算当前帧可见性
        # V_curr = self.compute_visibility(
        #     grid_3d, 
        #     cam_intrins=curr_cam_intrins,
        #     cam_extrins=curr_cam_extrins,
        #     img_shape=self.img_shape,
        #     img_feats=img_feats,
        #     spatial_shapes=spatial_shapes
        # )

        print("V_curr (当前可见性).  "*3)
        # V_curr = slots
        # 计算历史帧可见性
        # prev_cam_intrins = self.history_cam_intrins[:, -1]
        # prev_cam_extrins = self.history_cam_extrins[:, -1]
        # V_prev = self.compute_visibility(
        #     grid_3d,
        #     cam_intrins=prev_cam_intrins,
        #     cam_extrins=prev_cam_extrins,
        #     img_shape=self.img_shape,
        #     img_feats=img_feats,
        #     spatial_shapes=spatial_shapes
        # )

        V_prev = sampled_history_visibility #bs,4,z,h,w

        print("\n8. 可见性计算后形状：")
        print(f"V_curr (当前可见性): {V_curr.shape} (预期：[bs, h, w, z])")
        print(f"V_prev (历史可见性): {V_prev.shape} (预期：[bs, h, w, z])")

        # -------------------------- 9. 稀疏采样前展平变量打印 --------------------------
        # 展平变量（用于稀疏采样）
        curr_bev_flat = curr_bev.reshape(bs, c_, -1)  # [bs, c_, N], N=h*w*z
        history_bev_flat = sampled_history_bev.reshape(bs, mc, -1)  # [bs, mc, N]
        V_prev_flat = V_prev.reshape(bs,self.history_num, -1)  # [bs, 4,N]
        V_curr_flat = V_curr.reshape(bs, 1,-1)  # [bs, 1,N]
        nonempty_prob_flat = nonempty_prob.reshape(bs, -1)  # [bs, N]
        print("print(nonempty_prob_flat.shape)")
        print(nonempty_prob_flat.shape)
        total_voxels = nonempty_prob_flat.shape[1]

        print("\n9. 稀疏采样前展平变量形状：")
        print(f"curr_bev_flat: {curr_bev_flat.shape} (预期：[bs, c_, N], N={total_voxels})")
        print(f"history_bev_flat: {history_bev_flat.shape} (预期：[bs, mc, N])")
        print(f"nonempty_prob_flat: {nonempty_prob_flat.shape} (预期：[bs, N])")
        print(f"total_voxels (h*w*z): {total_voxels} (预期：{h*w*z})")

        # -------------------------- 10. 前景/背景索引及融合后打印（可选，确认后续维度） --------------------------
        # 生成前景/背景索引
        fg_indices = torch.topk(nonempty_prob_flat, self.top_k, dim=1)[1]
        bg_indices = torch.topk(1 - nonempty_prob_flat, total_voxels - self.top_k, dim=1)[1]
        # 提取前景特征（示例，其他融合步骤类似）
        fg_history_feat = torch.gather(history_bev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, mc, 1))

        print("\n10. 前景/背景索引及特征提取后形状：")
        print(f"fg_indices (前景索引): {fg_indices.shape} (预期：[bs, top_k])")
        print(f"bg_indices (背景索引): {bg_indices.shape} (预期：[bs, N-top_k])")
        print(f"fg_history_feat (前景历史特征): {fg_history_feat.shape} (预期：[bs, mc, top_k])")
        print("="*50)

        # -------------------------- 后续原有逻辑（略，维度已通过打印确认） --------------------------
        # 8. 前景融合（原有代码）
        # 9. 背景融合（原有代码）
        # 10. 更新当前BEV特征（原有代码）
        # curr_bev_flat = curr_bev.reshape(bs, c_, -1)  # [bs, c_, N] N=h*w*z
        # history_bev_flat = sampled_history_bev.reshape(bs, mc, -1)  # [bs, mc, N]
        # V_prev_flat = V_prev.reshape(bs, -1)  # [bs, N]
        # V_curr_flat = V_curr.reshape(bs, -1)  # [bs, N]
        # nonempty_prob_flat = nonempty_prob.reshape(bs, -1)  # [bs, N]
        # total_voxels = nonempty_prob_flat.shape[1]

        # fg_indices = torch.topk(nonempty_prob_flat, self.top_k, dim=1)[1]  # [bs, top_k]
        # bg_indices = torch.topk(1 - nonempty_prob_flat, total_voxels - self.top_k, dim=1)[1]  # [bs, N-top_k]

        # # 前景特征提取
        # fg_history_feat = torch.gather(history_bev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, mc, 1))  # [bs, mc, top_k]
        fg_curr_feat = torch.gather(curr_bev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, c_, 1))  # [bs, c_, top_k]

        # 历史特征时间聚合
        fg_history_feat_time = fg_history_feat.reshape(bs, self.history_num, c_, self.top_k)  # [bs, T, c_, K]
        #TODO 后续可以把time_weights也乘进去
        
        # time_weights = torch.exp(-0.5 * torch.arange(self.history_num, device=device)).view(1, self.history_num, 1, 1)

        fg_V_prev = torch.gather(V_prev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, self.history_num, 1))  # [bs, 4,K]
        fg_V_curr = torch.gather(V_curr_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, 1, 1))  # [bs, 1,K]
        # fg_time_vis_weights = fg_V_prev/(fg_V_prev.sum(dim=1).unsqueeze(1)+1e-10 ) # [bs, 4,K]
        fg_time_vis_weights = (fg_V_prev*(self.fg_scale.view(1,self.history_num,1))).softmax(dim=1)
        print("print(fg_history_feat_time.shape)")
        print(fg_history_feat_time.shape)
        print("print(fg_time_vis_weights.shape)")
        print(fg_time_vis_weights.shape)
        # fg_history_agg = (fg_history_feat_time * fg_time_vis_weights.unsqueeze(2)).sum(dim=1)  # [bs, c_, K]).sum(dim=1)  # [bs, c_, K]]
        fg_history_agg = (fg_history_feat_time * fg_time_vis_weights.unsqueeze(2)).sum(dim=1)  # [bs, c_, K]).sum(dim=1)  # [bs, c_, K]]


        print("print(fg_history_agg.shape)")
        print(fg_history_agg.shape)

        # 可见性聚合与门控
        
        # fg_V_prev_time = fg_V_prev.unsqueeze(1).repeat(1, self.history_num, 1)  # [bs, T, K]
        # fg_V_prev_agg = (fg_V_prev_time * time_weights.squeeze(-1)).sum(dim=1)  # [bs, K]
        fg_V_prev_agg = fg_V_prev.max(dim=1)[0]  # [bs, K]
        fg_w_hist, fg_w_curr = self.compute_gate_weights(fg_V_prev_agg, fg_V_curr.squeeze(1))  # [bs, K, 1]

        # 前景融合
        fg_history_agg_perm = fg_history_agg.permute(0, 2, 1)  # [bs, K, c_]
        fg_curr_perm = fg_curr_feat.permute(0, 2, 1)  # [bs, K, c_]
        fg_fused = fg_w_hist * fg_history_agg_perm + fg_w_curr * fg_curr_perm  # [bs, K, c_]

        # occupancy嵌入融合
        last_occ_reshaped = last_occ_pred.permute(0, 3, 2, 1, 4).reshape(bs, -1, last_occ_pred.shape[-1])  # [bs, N, num_classes]
        fg_occ_feat = torch.gather(last_occ_reshaped, dim=1, index=fg_indices.unsqueeze(-1).repeat(1, 1, last_occ_pred.shape[-1]))  # [bs, K, num_classes]
        fg_occ_embed = self.occ_embedding(fg_occ_feat).permute(0, 2, 1)  # [bs, occ_embedims, K]
        fg_fused = torch.cat([fg_fused, fg_occ_embed.permute(0, 2, 1)], dim=-1)  # [bs, K, c_+occ_embedims]
        fg_fused = self.history_fusion_linear(fg_fused).permute(0, 2, 1)  # [bs, c_, K]

        # 背景融合（原有代码）
        bg_history_feat = torch.gather(history_bev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, mc, 1))  # [bs, mc//2, bg_k]
        bg_curr_feat = torch.gather(curr_bev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, c_, 1))  # [bs, c_, bg_k]

        bg_history_feat_time = bg_history_feat.reshape(bs, self.history_num, c_, -1)  # [bs, T, c_//2, bg_k]
        # bg_history_agg = (bg_history_feat_time * time_weights).sum(dim=1)  # [bs, c_//2, bg_k]
        bg_V_prev = torch.gather(V_prev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, self.history_num, 1))  # [bs, bg_k]
        bg_V_curr = torch.gather(V_curr_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, 1, 1))  # [bs, bg_k]
        #TODO 这个10的超参数？ 调整成可学习？
        # bg_time_vis_weights = bg_V_prev/(bg_V_prev.sum(dim=1).unsqueeze(1) +1e-10) # [bs, 4,K]
        bg_time_vis_weights =(bg_V_prev*(self.bg_scale.view(1,self.history_num,1))).softmax(dim=1)
        bg_history_agg = (bg_history_feat_time * bg_time_vis_weights.unsqueeze(2)).sum(dim=1)  # [bs, c_, K]).sum(dim=1)  # [bs, c_, K]]



        # bg_history_agg_perm = F.pad(bg_history_agg.permute(0, 2, 1), (0, c_ - c_//2, 0, 0))  # [bs, bg_k, c_]

        bg_history_agg_perm = bg_history_agg.permute(0, 2, 1)  # [bs, bg_k, c_]
        # bg_V_prev_time = bg_V_prev.unsqueeze(1).repeat(1, self.history_num, 1)  # [bs, T, bg_k]
        # bg_V_prev_agg = (bg_V_prev_time * time_weights.squeeze(-1)).sum(dim=1)  # [bs, bg_k]
        bg_V_prev_agg = bg_V_prev.max(dim=1)[0]  # [bs, bg_k]
        bg_w_hist, bg_w_curr = self.compute_gate_weights(bg_V_prev_agg, bg_V_curr.squeeze(1))  # [bs, bg_k, 1]

        bg_curr_perm = bg_curr_feat.permute(0, 2, 1)  # [bs, bg_k, c_]
        # print("*"*50)
        # print("print(bg_w_hist.shape)")
        # print(bg_w_hist.shape)
        # print("print(bg_w_curr.shape)")
        # print(bg_w_curr.shape)
        # print("print(bg_history_agg_perm.shape)")
        # print(bg_history_agg_perm.shape)
        # print("print(bg_curr_perm.shape)")        
        # print(bg_curr_perm.shape)

        # # 断言批次大小一致
        # assert bg_w_hist.shape[0] == bg_history_agg_perm.shape[0] == bg_w_curr.shape[0] == bg_curr_perm.shape[0], \
        #     f"批次大小不匹配: {bg_w_hist.shape[0]}, {bg_history_agg_perm.shape[0]}, {bg_w_curr.shape[0]}, {bg_curr_perm.shape[0]}"

        # # 断言第二维度（bg_k）一致
        # assert bg_w_hist.shape[1] == bg_history_agg_perm.shape[1] == bg_w_curr.shape[1] == bg_curr_perm.shape[1], \
        #     f"bg_k维度不匹配: {bg_w_hist.shape[1]}, {bg_history_agg_perm.shape[1]}, {bg_w_curr.shape[1]}, {bg_curr_perm.shape[1]}"

        # # 断言第三维度（c_）匹配（bg_w_hist和bg_w_curr的第三维为1，不影响广播）
        # assert bg_history_agg_perm.shape[2] == bg_curr_perm.shape[2], \
        #     f"特征维度c_不匹配: {bg_history_agg_perm.shape[2]} vs {bg_curr_perm.shape[2]}"

        # print("bg_w_hist dtype:", bg_w_hist.dtype)
        # print("bg_history_agg_perm dtype:", bg_history_agg_perm.dtype)
        # print("bg_w_curr dtype:", bg_w_curr.dtype)
        # print("bg_curr_perm dtype:", bg_curr_perm.dtype)


        # print("bg_w_hist device:", bg_w_hist.device)
        # print("bg_history_agg_perm device:", bg_history_agg_perm.device)
        # print("bg_w_curr device:", bg_w_curr.device)
        # print("bg_curr_perm device:", bg_curr_perm.device)


        bg_fused = bg_w_hist * bg_history_agg_perm + bg_w_curr * bg_curr_perm  # [bs, bg_k, c_]
        # 先验证乘法是否正常
        # temp1 = bg_w_hist * bg_history_agg_perm
        # temp2 = bg_w_curr * bg_curr_perm
        # 再验证加法是否正常

        # bg_w_hist = bg_w_hist.contiguous()
        # bg_history_agg_perm = bg_history_agg_perm.contiguous()
        # bg_w_curr = bg_w_curr.contiguous()
        # bg_curr_perm = bg_curr_perm.contiguous()

        # # 重新计算
        # temp1 = bg_w_hist * bg_history_agg_perm
        # temp2 = bg_w_curr * bg_curr_perm
        # temp1 = temp1.contiguous()
        # temp2 = temp2.contiguous()
        # bg_fused = temp1 + temp2


        # 转移所有张量到CPU
        # bg_w_hist_cpu = bg_w_hist.cpu()
        # bg_history_agg_perm_cpu = bg_history_agg_perm.cpu()
        # bg_w_curr_cpu = bg_w_curr.cpu()
        # bg_curr_perm_cpu = bg_curr_perm.cpu()

        # # 分步执行运算
        # try:
        #     temp1_cpu = bg_w_hist_cpu * bg_history_agg_perm_cpu
        #     temp2_cpu = bg_w_curr_cpu * bg_curr_perm_cpu
        #     bg_fused_cpu = temp1_cpu + temp2_cpu
        #     print(bg_fused_cpu)
        #     print(bg_fused_cpu.shape)
        #     print("CPU运算成功，无明显错误")
        # except Exception as e:
        #     print(f"CPU运算报错：{e}")  # 此处会显示具体错误原因（如值异常）


        # bg_fused = temp1.clone() + temp2.clone()
        # 1/0
        bg_occ_feat = torch.gather(last_occ_reshaped, dim=1, index=bg_indices.unsqueeze(-1).repeat(1, 1, last_occ_pred.shape[-1]))  # [bs, bg_k, num_classes]
        bg_occ_embed = self.occ_embedding(bg_occ_feat).permute(0, 2, 1)  # [bs, occ_embedims, bg_k]
        bg_fused = torch.cat([bg_fused, bg_occ_embed.permute(0, 2, 1)], dim=-1)  # [bs, bg_k, c_+occ_embedims]
        bg_fused = self.history_fusion_bg_linear(bg_fused).permute(0, 2, 1)  # [bs, c_, bg_k]

        # 更新当前BEV
        curr_bev_updated = curr_bev_flat.clone()
        curr_bev_updated.scatter_add_(dim=2, index=fg_indices.unsqueeze(1).repeat(1, c_, 1), src=fg_fused)
        curr_bev_updated.scatter_add_(dim=2, index=bg_indices.unsqueeze(1).repeat(1, c_, 1), src=bg_fused)
        curr_bev_updated = curr_bev_updated.reshape(bs, c_, z, h, w)  # 恢复原形状

        # 更新历史缓存
        self.history_last_bev = curr_bev_updated.detach().clone()
        self.history_bev = torch.cat([curr_bev,sampled_history_bev[:, :-c_, ...]], dim=1).detach()
        # self.history_cam_intrins = torch.cat([curr_cam_intrins.unsqueeze(1),self.history_cam_intrins[:, :-1, ...]], dim=1).detach()
        # self.history_cam_extrins = torch.cat([curr_cam_extrins.unsqueeze(1),self.history_cam_extrins[:, 1-1:, ...]], dim=1).detach()
        self.history_visibility =torch.cat([V_curr, V_prev[:, :-1, ...]],dim=1).detach().half()
        self.history_forward_augs = forward_augs.clone()

        return curr_bev_updated

    def generate_grid(self, curr_bev):
        n, c_, z, h, w = curr_bev.shape
        # Generate grid
        xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w, 1).expand(h, w, z)
        ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1, 1).expand(h, w, z)
        zs = torch.linspace(0, z - 1, z, dtype=curr_bev.dtype, device=curr_bev.device).view(1, 1, z).expand(h, w, z)
        grid = torch.stack((xs, ys, zs, torch.ones_like(xs)), -1).view(1, h, w, z, 4).expand(n, h, w, z, 4).view(n, h,w, z, 4, 1)
        return grid

    def generate_feat2bev(self, grid, dx, bx):
        feat2bev = torch.zeros((4, 4), dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = dx[0]
        feat2bev[1, 1] = dx[1]
        feat2bev[2, 2] = dx[2]
        feat2bev[0, 3] = bx[0] - dx[0] / 2.
        feat2bev[1, 3] = bx[1] - dx[1] / 2.
        feat2bev[2, 3] = bx[2] - dx[2] / 2.
        feat2bev[3, 3] = 1
        feat2bev = feat2bev.view(1, 4, 4)
        return feat2bev




@HEADS.register_module()
class GatedTemporalFusion6(BaseModule):
    def __init__(
        self,
        history_num=4,
        depth_sampler_embed_dims=256,
        depth_sampler_num_heads=8,
        depth_sampler_num_levels=4,
        depth_sampler_num_points=4,
        im2col_step=64,
        top_k=500,
        single_bev_num_channels=96,
        occ_embedims=32,
        num_classes=18,
        vis_theta=0.28,
        vis_beta=10.0,
        vis_gamma=0.4,
        vis_sigma=0.1,
        nonempty_thresh=0.1,
        max_step_ratio=1.2,
        **kwargs
    ):
        super(GatedTemporalFusion6, self).__init__()

        # 基础参数
        self.history_num = history_num
        self.top_k = top_k
        self.single_bev_num_channels = single_bev_num_channels
        self.occ_embedims = occ_embedims
        self.fg_scale = nn.Parameter(torch.full((history_num,), 10.0))
        self.bg_scale = nn.Parameter(torch.full((history_num,), 10.0))
        # 可见性门控参数
        # self.vis_theta = vis_theta  # 可见性阈值
        # self.vis_beta = vis_beta    # 敏感度参数
        # self.vis_gamma = vis_gamma  # 场景4固定权重
        # self.vis_sigma = vis_sigma  # 软化参数
        self.vis_theta = nn.Parameter(torch.tensor(vis_theta))   # 可见性阈值
        self.vis_beta = nn.Parameter(torch.tensor(vis_beta))     # 敏感度参数
        self.vis_gamma = nn.Parameter(torch.tensor(vis_gamma))   # 场景4固定权重
        self.vis_sigma = nn.Parameter(torch.tensor(vis_sigma))  

        
        # 历史特征缓存（新增history_last_bev）
        self.history_bev = None  # 缓存历史多帧BEV特征
        self.history_last_bev = None  # 缓存上一帧最终融合后的BEV特征（关键新增）
        self.history_cam_intrins = None
        self.history_cam_extrins = None
        
        # 网络层（保持不变）
        self.occ_embedding = nn.Sequential(
            nn.Linear(num_classes, occ_embedims),
            nn.Softplus(),
            nn.Linear(occ_embedims, occ_embedims),
        )
        self.history_fusion_linear = nn.Sequential(
            nn.Linear(single_bev_num_channels + occ_embedims, single_bev_num_channels),
            nn.Softplus(),
            nn.Linear(single_bev_num_channels, single_bev_num_channels),
        )
        self.history_fusion_bg_linear = nn.Sequential(
            nn.Linear(single_bev_num_channels + occ_embedims, single_bev_num_channels),
            nn.Softplus(),
            nn.Linear(single_bev_num_channels, single_bev_num_channels),
        )
        
        # 可见性计算组件（保持不变）
        self.rt_vis_calculator = EfficientRayTracingVisibility(
            nonempty_thresh=nonempty_thresh,
            max_step_ratio=max_step_ratio
        )
        # self.img_shape = (900, 1600)  # 默认图像尺寸

        # self.depth_sampler = DeformableDepthSampler(
        #     embed_dims=depth_sampler_embed_dims,
        #     num_heads=depth_sampler_num_heads,
        #     num_levels=depth_sampler_num_levels,
        #     num_points=depth_sampler_num_points
        # )
        self.history_forward_augs = None  # 用于缓存历史帧的变换矩阵（如BDAM矩阵）
        self.im2col_step = im2col_step
        # self.voxel_encoder = nn.Linear(single_bev_num_channels, depth_sampler_embed_dims)
        self.dbound = [1.0, 45.0, 0.5]
        self.pc_range = [-40, -40, -1.0, 40, 40, 5.4]
        self.final_dim = (256, 704)
    def compute_visibility(self, grid, cam_intrins, cam_extrins, img_shape, img_feats, spatial_shapes):
        """
        升级：结合可变形注意力采样的深度值优化可见性计算
        Args:
            img_feats: 多尺度图像特征 [bs, num_levels, c, h, w]
            spatial_shapes: 图像特征尺度 [num_levels, 2] (h, w)
            其他参数同原函数
        Returns:
            vis_prob: [bs, h, w, z] 优化后的可见性概率
        """
        bs, h, w, z, _ = grid.shape
        h_img, w_img = img_shape
        device = grid.device
        num_voxels = h * w * z  # 体素总数

        # 1. 原有相机投影逻辑（计算图像坐标和初始可见性）
        # 1.1 体素坐标→相机坐标→图像坐标
        grid_cam = grid.unsqueeze(1).expand(bs, self.num_cams, h, w, z, 3)  # [bs, num_cams, h, w, z, 3]
        grid_flat = grid_cam.reshape(-1, num_voxels, 3)  # [bs*num_cams, N, 3]
        grid_hom = torch.cat([grid_flat, torch.ones_like(grid_flat[..., :1])], dim=-1)  # [bs*num_cams, N, 4]

        cam_intrins_flat = cam_intrins.reshape(-1, 4, 4)  # [bs*num_cams, 4, 4]
        cam_extrins_flat = cam_extrins.reshape(-1, 4, 4)  # [bs*num_cams, 4, 4]
        extrins_inv = torch.inverse(cam_extrins_flat)
        cam_coords = torch.bmm(extrins_inv[:, :3, :4], grid_hom.permute(0, 2, 1)).permute(0, 2, 1)  # [bs*num_cams, N, 3]
        depth = cam_coords[..., 2:3] + 1e-8  # 相机坐标系下的深度

        # 1.2 计算初始可见性（原逻辑）
        img_coords = torch.bmm(cam_intrins_flat[:, :3, :3], cam_coords.permute(0, 2, 1)).permute(0, 2, 1)  # [bs*num_cams, N, 3]
        img_xy = img_coords[..., :2] / depth  # [bs*num_cams, N, 2] (u, v)
        depth_valid = (cam_coords[..., 2] > 0).float()  # [bs*num_cams, N]
        u_valid = (img_xy[..., 0] >= 0) & (img_xy[..., 0] < w_img)
        v_valid = (img_xy[..., 1] >= 0) & (img_xy[..., 1] < h_img)
        img_valid = (u_valid & v_valid).float()  # [bs*num_cams, N]
        initial_vis = depth_valid * img_valid  # [bs*num_cams, N]

        # 2. 可变形注意力深度采样
        # 2.1 准备输入：体素特征编码
        # voxel_feat = self.voxel_encoder(self.curr_bev_feat)  # [bs, c_, z, h, w] → [bs, z*h*w, embed_dims]（需提前展平体素特征）
        # voxel_feat = voxel_feat.reshape(bs, num_voxels, -1)  # [bs, N, embed_dims]

        # 2.2 生成参考点（归一化到[0,1]）
        norm_img_xy = img_xy / torch.tensor([w_img, h_img], device=device).view(1, 1, 2)  # [bs*num_cams, N, 2]
        # 取主相机（如第0个相机）的参考点作为采样基准
        ref_points = norm_img_xy.reshape(bs, self.num_cams, num_voxels, 2)[:, 0]  # [bs, N, 2]
        ref_points = ref_points.unsqueeze(2).repeat(1, 1, self.depth_sampler.num_levels, 1)  # [bs, N, L, 2]

        # 2.3 采样深度特征
        sampled_depth = self.depth_sampler(
            query=voxel_feat,
            value=img_feats,  # 多尺度图像特征 [bs, L, c, h, w]
            reference_points=ref_points,
            spatial_shapes=spatial_shapes
        )  # [bs, N]

        # 3. 结合采样深度优化可见性
        # 3.1 深度一致性校验：采样深度与相机投影深度的差异
        cam_depth = depth.reshape(bs, self.num_cams, num_voxels)[:, 0]  # 主相机的投影深度 [bs, N]
        depth_diff = torch.abs(sampled_depth - cam_depth) / (cam_depth + 1e-8)  # 相对深度差
        depth_consistent = (depth_diff < 0.3).float()  # 深度差小于30%则认为有效

        # 3.2 融合可见性：初始可见性 × 深度一致性
        initial_vis = initial_vis.reshape(bs, self.num_cams, num_voxels).max(dim=1)[0]  # [bs, N]（多相机取max）
        vis_prob = initial_vis * depth_consistent  # [bs, N]

        # 4. 还原形状
        return vis_prob.reshape(bs, h, w, z)  # [bs, h, w, z]

    def compute_alpha_unified(self, V_curr, V_prev):
        """统一计算当前帧融合权重α（覆盖四场景）"""
        eps = 1e-8
        # 1. 基础动态权重σ_base（场景1-3）
        ratio = V_curr / (V_curr + V_prev + eps)
        sigma_base = torch.sigmoid(self.vis_beta * (ratio - 0.5))
        
        # 2. 场景4软化掩码σ_both
        mask_curr = torch.sigmoid(-(V_curr - self.vis_theta) / self.vis_sigma)
        mask_prev = torch.sigmoid(-(V_prev - self.vis_theta) / self.vis_sigma)
        sigma_both = mask_curr * mask_prev
        
        # 3. 最终权重计算
        alpha = (1 - sigma_both) * sigma_base + sigma_both * self.vis_gamma
        return alpha.unsqueeze(-1)  # [bs, N, 1]

    def compute_gate_weights(self, V_prev_agg, V_curr):
        """计算历史和当前帧的门控权重"""
        alpha = self.compute_alpha_unified(V_curr, V_prev_agg)
        return 1 - alpha, alpha  # 历史权重，当前权重

    def generate_grid(self, curr_bev, voxel_min, voxel_max, voxel_size):
        """生成体素中心坐标网格（自车坐标系）"""
        bs, c_, z, h, w = curr_bev.shape
        device = curr_bev.device
        
        # 计算体素中心坐标
        x_coords = torch.linspace(
            voxel_min[0] + voxel_size[0]/2, 
            voxel_max[0] - voxel_size[0]/2, 
            w, device=device
        )
        y_coords = torch.linspace(
            voxel_min[1] + voxel_size[1]/2, 
            voxel_max[1] - voxel_size[1]/2, 
            h, device=device
        )
        z_coords = torch.linspace(
            voxel_min[2] + voxel_size[2]/2, 
            voxel_max[2] - voxel_size[2]/2, 
            z, device=device
        )
        
        # 生成网格并扩展维度 [bs, h, w, z, 3]
        x_grid, y_grid, z_grid = torch.meshgrid(x_coords, y_coords, z_coords, indexing='xy')
        grid = torch.stack([x_grid, y_grid, z_grid], dim=-1)  # [w, h, z, 3]
        grid = grid.permute(1, 0, 2, 3)  # [h, w, z, 3]
        return grid.unsqueeze(0).repeat(bs, 1, 1, 1, 1)  # [bs, h, w, z, 3]


    def get_reference_points(self, H, W, Z=None, num_points_in_pillar =4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
        torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    @force_fp32(apply_to=('reference_points', 'cam_params'))
    def point_sampling(self, reference_points, pc_range, img_metas, cam_params=None):
        # prepare for point sampling
        lidar2img = []
        ego2lidar = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])     # lidar2img update the post aug in the loading pipeline
            ego2lidar.append(img_meta['ego2lidar'])
        lidar2img = torch.stack(lidar2img, dim=0).to(reference_points.device)
        ego2lidar = torch.stack(ego2lidar, dim=0).to(reference_points.device)

        sensor2egos, ego2globals, intrins, post_augs, bda_mat = cam_params
        num_cam = sensor2egos.size(1)
        ogfH, ogfW = self.final_dim

        # reference_points defines in the bev space, [bs, D, hxw, 3]
        # change reference_points from bev-ego coordinate to ego coordinate
        reference_points = reference_points.clone()
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
                                     (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
                                     (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
                                     (pc_range[5] - pc_range[2]) + pc_range[2]

        # prepare for point sampling
        reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
        reference_points = reference_points.permute(1, 0, 2, 3)  # shape: (num_points_in_pillar,bs,h*w,4)
        D, B, num_query = reference_points.size()[:3]  # D=num_points_in_pillar , num_query=h*w
        reference_points = reference_points.view(D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)  # shape: (num_points_in_pillar,bs,num_cam,h*w,4)
        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)
        ego2lidar = ego2lidar.view(1, B, 1, 1, 4, 4).repeat(D, 1, num_cam, num_query, 1, 1)
        inverse_bda = bda_mat.view(1, B, 1, 1, 4, 4).repeat(D, 1, num_cam, num_query, 1, 1)

        # change reference_points from ego coordinate to img coordinate
        eps = 1e-5
        reference_points_cam = (lidar2img @ ego2lidar @ inverse_bda @ reference_points).squeeze(-1)   # [num_points_in_pillar, bs, num_cam, num_query=h*w, 4]
        reference_points_depth = reference_points_cam[..., 2:3]
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(reference_points_depth, torch.ones_like(reference_points_depth) * eps)

        # Bug!!
        # Correct normalize is
        # reference_points_cam[..., 0] /= ogfW
        # reference_points_cam[..., 1] /= ogfH
        # But for reproducing our results, we use the following normalization
        reference_points_cam[..., 0] /= ogfW
        reference_points_cam[..., 1] /= ogfH

        bev_mask = (reference_points_depth > eps)
        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)                  # shape: (num_cam, bs,h*w, num_points_in_pillar, 2)
        reference_points_depth = reference_points_depth.permute(2, 1, 3, 0, 4)              # shape: (num_cam, bs,h*w, num_points_in_pillar, 1)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)                        # shape: (num_cam, bs,h*w, num_points_in_pillar)

        return reference_points_cam, reference_points_depth, bev_mask


    @force_fp32()
    def forward(self, curr_bev, cam_params, history_fusion_params, dx, bx, history_last_bev=None, last_occ_pred=None, nonempty_prob=None, img_feats=None, spatial_shapes=None,pred_img_depth=None,**kwargs):
        """
        Args:
            curr_bev: [bs, c, z, h, w] 当前帧BEV特征
            cam_params: 相机参数列表，包含外参、内参等
            history_fusion_params: 历史融合参数（包含序列信息等）
            dx: 体素尺寸 (x, y, z)
            bx: 体素偏移
            nonempty_prob: [bs, z, h, w] 体素非空概率
            last_occ_pred: [bs, z, h, w, num_classes] 上一时刻 occupancy 预测
            img_feats: 多尺度图像特征 [bs, num_levels, c, h, w]（新增，用于深度采样）
            spatial_shapes: 图像特征尺度 [num_levels, 2] (h, w)（新增，用于深度采样）
        Returns:
            curr_bev_updated: [bs, c, z, h, w] 融合后BEV特征
        """
        # print(self.history_num)
        # print("print(self.fg_scale)")
        # print(self.fg_scale) 
        # print("print(self.bg_scale)")
        # print(self.bg_scale)
        # # 可见性门控参数
        # # self.vis_theta = vis_theta  # 可见性阈值
        # # self.vis_beta = vis_beta    # 敏感度参数
        # # self.vis_gamma = vis_gamma  # 场景4固定权重
        # # self.vis_sigma = vis_sigma  # 软化参数
        # print("print(self.vis_theta # 可见性阈值)")
        # print(self.vis_theta )  # 可见性阈值
        # print("print(self.vis_beta) # 敏感度参数")
        # print(self.vis_beta)      # 敏感度参数
        # print("print(self.vis_gamma) # 场景4固定权重")
        # print(self.vis_gamma)  # 场景4固定权重
        # print("print(self.vis_sigma) # 软化参数")
        # print(self.vis_sigma) 
        if torch.rand(1).item() < 1/2000:
            print(self.history_num)
            print("print(self.fg_scale)")
            print(self.fg_scale) 
            print("print(self.bg_scale)")
            print(self.bg_scale)
            # 可见性门控参数
            print("print(self.vis_theta # 可见性阈值)")
            print(self.vis_theta)  # 可见性阈值
            print("print(self.vis_beta) # 敏感度参数")
            print(self.vis_beta)      # 敏感度参数
            print("print(self.vis_gamma) # 场景4固定权重")
            print(self.vis_gamma)  # 场景4固定权重
            print("print(self.vis_sigma) # 软化参数")
            print(self.vis_sigma)  


        # -------------------------- 1. 解析参数后打印核心形状 --------------------------
        # 解析相机参数
        curr_cam_extrins = cam_params[0]  # [bs, num_cams, 4, 4]
        curr_cam_intrins = cam_params[2]  # [bs, num_cams, 4, 4]
        forward_augs = cam_params[4]      # [bs, 4, 4] 前向变换矩阵
        self.num_cams = curr_cam_extrins.shape[1]  # 从外参中获取相机数量
        bs, c_, z, h, w = curr_bev.shape
        device = curr_bev.device
        mc = self.history_num * c_        # 历史特征总通道数
        # self.history_forward_augs = forward_augs.clone()


        ref_3d = self.get_reference_points(
            h, w, z, z, dim='3d', bs=bs, device=device, dtype=curr_bev.dtype) # torch.Size([3, 2, 625, 3]) #[bs,z,yx,3(x,y,z)]
        # ref_2d = self.get_reference_points(
        #     h, w, dim='2d', bs=bs, device=device, dtype=curr_bev.dtype) #torch.Size([3, 625, 1, 2])
        slots = torch.zeros(list([ref_3d.shape[0],ref_3d.shape[2],ref_3d.shape[1]])).to(ref_3d)
        reference_points_cam, reference_points_depth, bev_mask = self.point_sampling(ref_3d, self.pc_range, img_metas=kwargs['img_metas'], cam_params=cam_params)
        indexes = [[] for _ in range(bs)]
        spatial_shapes =[]
        spatial_shapes.append([16, 44])
        spatial_shapes = torch.tensor(spatial_shapes).to(device)
        pred_img_depth = pred_img_depth.view(bs * 6, -1, spatial_shapes[0][0], spatial_shapes[0][1])
        pred_img_depth = pred_img_depth.flatten(2).permute(0, 2, 1)  
        max_len = 0
        for j in range(bs):
            for i, mask_per_img in enumerate(bev_mask):
                index_query_per_img = mask_per_img[j].sum(-1).nonzero().squeeze(-1)
                if len(index_query_per_img) == 0:
                    index_query_per_img = bev_mask[i][j].sum(-1).nonzero().squeeze(-1)[0:1]
                indexes[j].append(index_query_per_img)
                # for batch operation, we need to pad the indexes to the same length
                max_len = max(max_len, len(index_query_per_img))
        reference_points_cam_rebatch = reference_points_cam.new_zeros([bs, self.num_cams, max_len, z, 2])
        reference_points_depth_rebatch = reference_points_depth.new_zeros([bs, self.num_cams, max_len, z, 1])

        for j in range(bs):
            for i, (reference_points_per_img, reference_points_depth_per_img) in enumerate(zip(reference_points_cam, reference_points_depth)):
                index_query_per_img = indexes[j][i]
                reference_points_cam_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]
                reference_points_depth_rebatch[j, i, :len(index_query_per_img)] = reference_points_depth_per_img[j, index_query_per_img]

        #use deformble attn
        depth_reference_points = reference_points_cam_rebatch.reshape(bs*6, max_len*z, 1, 1, 1, 2).contiguous()
        depth_attention_weights = torch.ones_like(depth_reference_points[..., 0]).contiguous()
        pred_img_depth = pred_img_depth.unsqueeze(2).contiguous()
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))



        bev_query_depth_rebatch = (reference_points_depth_rebatch- self.dbound[0])/ self.dbound[2]
        bev_query_depth_rebatch = torch.clip(torch.floor(bev_query_depth_rebatch), 0, 88-1).to(torch.long)
        bev_query_depth_rebatch = F.one_hot(bev_query_depth_rebatch.squeeze(-1),
                                   num_classes=88)

        depth_output = MultiScaleDeformableAttnFunction_fp32.apply(pred_img_depth, spatial_shapes,level_start_index,depth_reference_points,depth_attention_weights, self.im2col_step)
        depth_output = depth_output.reshape(bs,6, max_len,z, -1)   # [bs*num_cam, num_query, num_Z_anchors, C]
        # reference_points_depth_rebatch

        increment = torch.zeros_like(depth_output)
        # increment[..., 0] = 1e-9  # 非原地赋值（创建新张量）
        # depth_output = depth_output + increment
        depth_output = depth_output + torch.cat([(torch.zeros_like(depth_output[...,:1]) + 1e-9),torch.zeros_like(depth_output[...,1:])],dim=-1)

        # print("depth_output.sum(-1).shape")
        # print(depth_output.sum(-1).shape)
        # print("depth_output.sum(-1)")
        # print(depth_output.sum(-1))
        # print("print((depth_output.sum(-1)==0).sum())")
        # print((depth_output.sum(-1)==0).sum())
        depth_output =depth_output/depth_output.sum(-1)[...,None] #bs,xy,z,D

        # print("depth_output.sum(-1).shape")
        # print(depth_output.sum(-1).shape)
        # print("depth_output.sum(-1)")
        # print(depth_output.sum(-1))
        # print("print((depth_output.sum(-1)==1).sum())")
        # print((depth_output.sum(-1)>=0.99).sum())
        # print("depth_output")
        # print(depth_output)


        depth_output = (1-depth_output.cumsum(dim=-1))


        # print("print((depth_output[...,-1]==0).sum())")
        # print((depth_output[...,-1]==0).sum())
        depth_output = (bev_query_depth_rebatch*depth_output).sum(-1)

        #恢复depth_output的shape
        # fix_depth_output = depth_output.new_zeros([bs, 6, w*h, z, 88])
        # for j in range(bs):
        #     for i in range(6):
        #         index_query_per_img = indexes[j][i]
        #         fix_depth_output[j, i, index_query_per_img] = depth_output[j, i, :len(index_query_per_img)]

        for j in range(bs):
            for i in range(6):
                index_query_per_img = indexes[j][i]
                slots[j, index_query_per_img] = torch.max(slots[j, index_query_per_img],depth_output[j, i, :len(index_query_per_img)])

        # depth_sum = fix_depth_output.sum(dim=-1).view(3,6,25,25,2)
        # bs = depth_sum.shape[0]       # 3
        # num_cam = depth_sum.shape[1]  # 6
        # z_layers = depth_sum.shape[4] # 2
        # H, W = depth_sum.shape[2], depth_sum.shape[3]  # 25,25

        # # 颜色映射：0值用黑色，非0值用渐变色
        # cmap = plt.cm.viridis
        # cmap.set_bad(color='black')  # 0值标记为黑色

        # # 设置子图布局（不变）
        # fig, axes = plt.subplots(
        #     nrows=bs, ncols=num_cam * z_layers,
        #     figsize=(30, 8),
        #     squeeze=False
        # )
        # fig.suptitle('fix_depth_output.sum(-1) 可视化（黑色=0值，颜色=非0值）', fontsize=16, y=0.98)


        # # -------------------------- 2. 循环绘制每个子图（核心修改：添加 .detach()） --------------------------
        # for batch_idx in range(bs):
        #     for cam_idx in range(num_cam):
        #         for z_idx in range(z_layers):
        #             col_idx = cam_idx * z_layers + z_idx
        #             ax = axes[batch_idx, col_idx]
                    
        #             # 核心修改：先 detach() 切断计算图，再转 cpu 和 numpy
        #             data = depth_sum[batch_idx, cam_idx, :, :, z_idx].detach().cpu().numpy()
        #             data[data == 0] = np.nan  # 0值替换为NaN，显示为黑色
                    
        #             # 绘制热力图（修改 vmin/vmax：同样添加 .detach()）
        #             im = ax.imshow(
        #                 data, 
        #                 cmap=cmap, 
        #                 aspect='auto',
        #                 # 关键修改：depth_sum 先 detach 再转 numpy，确保不影响梯度
        #                 vmin=np.nanmin(depth_sum.detach().cpu().numpy()),
        #                 vmax=np.nanmax(depth_sum.detach().cpu().numpy())
        #             )
                    
        #             # 子图标题和坐标轴（不变）
        #             ax.set_title(
        #                 f'Batch{batch_idx+1}\nCam{cam_idx+1} Z{z_idx+1}',
        #                 fontsize=10, pad=5
        #             )
        #             ax.set_xticks([])
        #             ax.set_yticks([])


        # # -------------------------- 3. 添加颜色条（不变） --------------------------
        # cbar = fig.colorbar(
        #     im, 
        #     ax=axes.ravel().tolist(),
        #     shrink=0.8,
        #     pad=0.02
        # )
        # cbar.set_label('Sum of Depth Bins (D=88)', fontsize=12)


        # # -------------------------- 4. 调整布局并保存（不变） --------------------------
        # plt.tight_layout(rect=[0, 0, 0.98, 0.95])
        # plt.savefig('depth_sum_visualization.png', dpi=300, bbox_inches='tight')
        # plt.show()

        # output

        #计数更新，建立在纸上的假设成立的基础上
        # count = bev_mask.sum(-1) > 0
        # count = count.permute(1, 2, 0).sum(-1)
        # count = torch.clamp(count, min=1.0)
        # slots = slots / count[..., None]

        # print("slots.shape")
        # print(slots.shape)
        # print("slots")  
        # print(slots)
        # zzzzz=1/0

        V_curr =slots.view(bs, 1, h, w,z).permute(0, 1, 4, 2, 3)
        # slots[...,0]+=1e-9
        # print("slots.sum(-1).shape")
        # print(slots.sum(-1).shape)
        # print("slots.sum(-1)")
        # print(slots.sum(-1))
        # print("print((slots.sum(-1)==0).sum())")
        # print((slots.sum(-1)==0).sum())
        # slots =slots/slots.sum(-1)[...,None] #bs,xy,z,D

        # print("slots.sum(-1).shape")
        # print(slots.sum(-1).shape)
        # print("slots.sum(-1)")
        # print(slots.sum(-1))
        # print("print((slots.sum(-1)==1).sum())")
        # print("slots")
        # print(slots)


        # slots = (1-slots.cumsum(dim=-1))

        # print("print((slots[...,-1]==0).sum())")
        # print((slots[...,-1]==0).sum())

        # print("slots")
        # print(slots)
        # print("print((slots[...,-1]<0.01).sum())")
        # print((slots[...,-1]<0.01).sum())

        #TODO 这里的对于边界值的考虑，从0开始还是从1开始？




        # slots = self.output_proj(slots)
        #TODO 上面这里需要检查一下
        #这里相当于两次softmax，可能会导致分布变得不够尖锐，需要进一步确认

        #到这里slots就是可见性的概率分布了
        #这里先尝试使用期望值进行计算，使得可微分

        #然后再采用stc的原始离散计算方法，


        # 打印解析后关键变量形状
        # print("="*50)
        # print("1. 解析参数后核心变量形状：")
        # print(f"curr_bev: {curr_bev.shape} (预期：[bs, c, z, h, w])")
        # print(f"curr_cam_extrins: {curr_cam_extrins.shape} (预期：[bs, num_cams, 4, 4])")
        # print(f"curr_cam_intrins: {curr_cam_intrins.shape} (预期：[bs, num_cams, 4, 4])")
        # print(f"forward_augs: {forward_augs.shape} (预期：[bs, 4, 4])")
        # print(f"dx: {dx.shape if hasattr(dx, 'shape') else type(dx)} (预期：[3])")
        # print(f"bx: {bx.shape if hasattr(bx, 'shape') else type(bx)} (预期：[3])")
        # print(f"bs: {bs}, c_: {c_}, z: {z}, h: {h}, w: {w} (BEV特征维度)")
        # print("="*50)

        if type(history_fusion_params['sequence_group_idx']) is list:
            seq_ids = history_fusion_params['sequence_group_idx'][0]
        else:
            seq_ids = history_fusion_params['sequence_group_idx']
        if type(history_fusion_params['start_of_sequence']) is list:
            start_of_sequence = history_fusion_params['start_of_sequence'][0]
        else:
            start_of_sequence = history_fusion_params['start_of_sequence']
        if type(history_fusion_params['curr_to_prev_ego_rt']) is list:
            curr_to_prev_ego_rt = history_fusion_params['curr_to_prev_ego_rt'][0]
        else:
            curr_to_prev_ego_rt = history_fusion_params['curr_to_prev_ego_rt']
        forward_augs = cam_params[-1]  # bda

        # check seq_ids > 0
        assert (seq_ids >= 0).all()
        # -------------------------- 2. 初始化历史缓存后打印 --------------------------
        if self.history_bev is None:
            # self.history_bev = curr_bev.repeat(1, self.history_num, 1, 1, 1)  # [bs, mc, z, h, w]
            # self.history_cam_intrins = curr_cam_intrins.unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)  # [bs, T, num_cams, 4, 4]
            # self.history_cam_extrins = curr_cam_extrins.unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)  # [bs, T, num_cams, 4, 4]
            # self.history_bev = curr_bev.clone()
            self.history_forward_augs = forward_augs.clone()
            self.history_seq_ids = seq_ids.clone()
            self.history_bev = curr_bev.repeat(1, self.history_num, 1, 1, 1)
            self.history_sweep_time = curr_bev.new_zeros(curr_bev.shape[0], self.history_num)
            self.history_visibility = V_curr.repeat(1, self.history_num, 1, 1, 1).half()
        self.history_bev = self.history_bev.detach()
        self.history_visibility = self.history_visibility.detach().half()
        self.history_sweep_time += 1

        # 打印历史缓存形状
        # print("\n2. 历史缓存初始化后形状：")
        # print(f"history_bev: {self.history_bev.shape} (预期：[bs, mc, z, h, w]，mc={mc})")
        # print(f"history_cam_intrins: {self.history_cam_intrins.shape} (预期：[bs, history_num, num_cams, 4, 4])")
        # print(f"history_cam_extrins: {self.history_cam_extrins.shape} (预期：[bs, history_num, num_cams, 4, 4])")

        # -------------------------- 3. 生成网格和BEV变换矩阵后打印 --------------------------
        # 处理新序列（略，不影响维度）
        # start_of_sequence = history_fusion_params.get('start_of_sequence', torch.zeros(bs, dtype=torch.bool, device=device))
        if start_of_sequence.sum()>0:
            self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_num, 1, 1, 1)
            # self.history_cam_intrins[start_of_sequence] = curr_cam_intrins[start_of_sequence].unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)
            # self.history_cam_extrins[start_of_sequence] = curr_cam_extrins[start_of_sequence].unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)
            self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]
            self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
            self.history_sweep_time[start_of_sequence] = 0  # zero the new sequence timestep starts
            self.history_visibility[start_of_sequence] = V_curr[start_of_sequence].repeat(1, self.history_num, 1, 1, 1).half()

        # 生成体素网格和BEV变换矩阵
        grid = self.generate_grid(curr_bev) #[bs,y,x,z,4]
        grid_3d = grid
        feat2bev = self.generate_feat2bev(grid, dx, bx)

        # 打印网格和变换矩阵形状
        # print("\n3. 生成网格和BEV变换矩阵后形状：")
        # print(f"grid_3d (体素网格): {grid_3d.shape} (关键！预期：[bs, h, w, z, 3] 或 [bs, w, h, z, 3])")
        # print(f"feat2bev (BEV变换矩阵): {feat2bev.shape} (预期：[bs, 4, 4])")

        # -------------------------- 4. 运动补偿矩阵计算后打印 --------------------------
        # 获取帧间姿态变换
        # curr_to_prev_ego_rt = history_fusion_params.get('curr_to_prev_ego_rt', torch.eye(4, device=device).unsqueeze(0).repeat(bs, 1, 1))
        # 计算RT流（坐标变换矩阵）
        rt_flow = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_ego_rt @ torch.inverse(forward_augs) @ feat2bev)
        # 生成齐次网格
        # 在forward函数中，生成grid_hom的位置修正：
        # grid_3d = self.generate_grid(curr_bev)  # 现在形状：[3, 25, 25, 2, 3]（bs, h, w, z, 3）
        # # 生成齐次坐标（x,y,z,1），并添加最后一个维度（用于矩阵乘法）
        # grid_hom = torch.cat([
        #     grid_3d,  # [3,25,25,2,3]
        #     torch.ones_like(grid_3d[..., :1])  # [3,25,25,2,1]（补充1作为齐次坐标）
        # ], dim=-1).unsqueeze(-1)  # 最终形状：[3,25,25,2,4,1]（符合预期）
        # # 打印运动补偿相关形状（矩阵乘法前关键检查）
        # print("\n4. 运动补偿矩阵计算后形状（矩阵乘法前）：")
        # print(f"curr_to_prev_ego_rt (帧间姿态): {curr_to_prev_ego_rt.shape} (预期：[bs, 4, 4])")
        # print(f"rt_flow (变换流): {rt_flow.shape} (预期：[bs, 4, 4])")
        # print(f"grid_hom (齐次网格): {grid_hom.shape} (关键！预期：[bs, h, w, z, 4, 1]，需与rt_flow广播匹配)")
        # print(f"rt_flow.view后: {rt_flow.view(bs, 1, 1, 1, 4, 4).shape} (预期：[bs, 1, 1, 1, 4, 4])")

        # # -------------------------- 5. 网格变换后打印（解决之前维度错的核心） --------------------------
        # try:
        #     grid_transformed = rt_flow.view(bs, 1, 1, 1, 4, 4) @ grid_hom  # 矩阵乘法：[bs, h, w, z, 4, 1]
        #     print("\n5. 网格变换后形状（矩阵乘法成功！）：")
        #     print(f"grid_transformed: {grid_transformed.shape} (预期：[bs, h, w, z, 4, 1])")
        # except RuntimeError as e:
        #     print(f"\n5. 网格变换矩阵乘法报错！错误信息：{str(e)}")
        #     print(f"  - rt_flow.view形状: {rt_flow.view(bs, 1, 1, 1, 4, 4).shape}")
        #     print(f"  - grid_hom形状: {grid_hom.shape}")
        #     print("  提示：需确保grid_hom的第1-4维度与rt_flow.view的第2-5维度匹配（广播规则）")
        #     raise e  # 继续抛出错误，方便定位
        bs, mc, z, h, w = self.history_bev.shape
        n, c_, z, h, w = curr_bev.shape
        grid = rt_flow.view(n, 1, 1, 1, 4, 4) @ grid
        # -------------------------- 6. 采样网格生成后打印 --------------------------
        # 生成采样网格（归一化到[-1,1]，适配F.grid_sample）
        normalize_factor = torch.tensor([w - 1.0, h - 1.0, z - 1.0], dtype=curr_bev.dtype, device=device)
        # grid_sampler = grid_transformed[..., :3, 0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0  # [bs, h, w, z, 3]
        # # 调整采样网格维度（适配F.grid_sample输入：[bs, z, h, w, 3]）
        # grid_sampler_permuted = grid_sampler.permute(0, 3, 1, 2, 4)  # 交换z和h/w维度
        grid = grid[:, :, :, :, :3, 0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0   # grid order is x, y, z


        # print("\n6. 采样网格生成后形状：")
        # print(f"grid_sampler (归一化后): {grid_sampler.shape} (预期：[bs, h, w, z, 3])")
        # print(f"grid_sampler_permuted (适配采样): {grid_sampler_permuted.shape} (预期：[bs, z, h, w, 3])")

        # -------------------------- 7. 历史BEV采样后打印 --------------------------
        # 采样历史BEV特征
        sampled_history_bev = F.grid_sample(
            self.history_bev.reshape(bs, mc, z, h, w),  # 输入：[bs, mc, z, h, w]
            grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4),
            align_corners=True,
            mode='bilinear'
        )
        sampled_history_visibility = F.grid_sample(
            self.history_visibility.reshape(bs, self.history_num, z, h, w).half(),  # 输入：[bs, mc, z, h, w]
            grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4).half(),
            align_corners=True,
            mode='nearest'
        )
        # print("\n7. 历史BEV采样后形状：")
        # print(f"history_bev.reshape: {self.history_bev.reshape(bs, mc, z, h, w).shape} (预期：[bs, mc, z, h, w])")
        # print(f"sampled_history_bev: {sampled_history_bev.shape} (预期：[bs, mc, z, h, w])")

        # -------------------------- 8. 可见性计算后打印 --------------------------
        # 计算当前帧可见性
        # V_curr = self.compute_visibility(
        #     grid_3d, 
        #     cam_intrins=curr_cam_intrins,
        #     cam_extrins=curr_cam_extrins,
        #     img_shape=self.img_shape,
        #     img_feats=img_feats,
        #     spatial_shapes=spatial_shapes
        # )

        # print("V_curr (当前可见性).  "*3)
        # V_curr = slots
        # 计算历史帧可见性
        # prev_cam_intrins = self.history_cam_intrins[:, -1]
        # prev_cam_extrins = self.history_cam_extrins[:, -1]
        # V_prev = self.compute_visibility(
        #     grid_3d,
        #     cam_intrins=prev_cam_intrins,
        #     cam_extrins=prev_cam_extrins,
        #     img_shape=self.img_shape,
        #     img_feats=img_feats,
        #     spatial_shapes=spatial_shapes
        # )

        V_prev = sampled_history_visibility #bs,4,z,h,w

        # print("\n8. 可见性计算后形状：")
        # print(f"V_curr (当前可见性): {V_curr.shape} (预期：[bs, h, w, z])")
        # print(f"V_prev (历史可见性): {V_prev.shape} (预期：[bs, h, w, z])")

        # -------------------------- 9. 稀疏采样前展平变量打印 --------------------------
        # 展平变量（用于稀疏采样）
        curr_bev_flat = curr_bev.reshape(bs, c_, -1)  # [bs, c_, N], N=h*w*z
        history_bev_flat = sampled_history_bev.reshape(bs, mc, -1)  # [bs, mc, N]
        V_prev_flat = V_prev.reshape(bs,self.history_num, -1)  # [bs, 4,N]
        V_curr_flat = V_curr.reshape(bs, 1,-1)  # [bs, 1,N]
        nonempty_prob_flat = nonempty_prob.reshape(bs, -1)  # [bs, N]
        # print("print(nonempty_prob_flat.shape)")
        # print(nonempty_prob_flat.shape)
        total_voxels = nonempty_prob_flat.shape[1]

        # print("\n9. 稀疏采样前展平变量形状：")
        # print(f"curr_bev_flat: {curr_bev_flat.shape} (预期：[bs, c_, N], N={total_voxels})")
        # print(f"history_bev_flat: {history_bev_flat.shape} (预期：[bs, mc, N])")
        # print(f"nonempty_prob_flat: {nonempty_prob_flat.shape} (预期：[bs, N])")
        # print(f"total_voxels (h*w*z): {total_voxels} (预期：{h*w*z})")

        # -------------------------- 10. 前景/背景索引及融合后打印（可选，确认后续维度） --------------------------
        # 生成前景/背景索引
        fg_indices = torch.topk(nonempty_prob_flat, self.top_k, dim=1)[1]
        bg_indices = torch.topk(1 - nonempty_prob_flat, total_voxels - self.top_k, dim=1)[1]
        # 提取前景特征（示例，其他融合步骤类似）
        fg_history_feat = torch.gather(history_bev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, mc, 1))

        # print("\n10. 前景/背景索引及特征提取后形状：")
        # print(f"fg_indices (前景索引): {fg_indices.shape} (预期：[bs, top_k])")
        # print(f"bg_indices (背景索引): {bg_indices.shape} (预期：[bs, N-top_k])")
        # print(f"fg_history_feat (前景历史特征): {fg_history_feat.shape} (预期：[bs, mc, top_k])")
        # print("="*50)

        # -------------------------- 后续原有逻辑（略，维度已通过打印确认） --------------------------
        # 8. 前景融合（原有代码）
        # 9. 背景融合（原有代码）
        # 10. 更新当前BEV特征（原有代码）
        # curr_bev_flat = curr_bev.reshape(bs, c_, -1)  # [bs, c_, N] N=h*w*z
        # history_bev_flat = sampled_history_bev.reshape(bs, mc, -1)  # [bs, mc, N]
        # V_prev_flat = V_prev.reshape(bs, -1)  # [bs, N]
        # V_curr_flat = V_curr.reshape(bs, -1)  # [bs, N]
        # nonempty_prob_flat = nonempty_prob.reshape(bs, -1)  # [bs, N]
        # total_voxels = nonempty_prob_flat.shape[1]

        # fg_indices = torch.topk(nonempty_prob_flat, self.top_k, dim=1)[1]  # [bs, top_k]
        # bg_indices = torch.topk(1 - nonempty_prob_flat, total_voxels - self.top_k, dim=1)[1]  # [bs, N-top_k]

        # # 前景特征提取
        # fg_history_feat = torch.gather(history_bev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, mc, 1))  # [bs, mc, top_k]
        fg_curr_feat = torch.gather(curr_bev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, c_, 1))  # [bs, c_, top_k]

        # 历史特征时间聚合
        fg_history_feat_time = fg_history_feat.reshape(bs, self.history_num, c_, self.top_k)  # [bs, T, c_, K]
        #TODO 后续可以把time_weights也乘进去
        
        # time_weights = torch.exp(-0.5 * torch.arange(self.history_num, device=device)).view(1, self.history_num, 1, 1)

        fg_V_prev = torch.gather(V_prev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, self.history_num, 1))  # [bs, 4,K]
        fg_V_curr = torch.gather(V_curr_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, 1, 1))  # [bs, 1,K]
        # fg_time_vis_weights = fg_V_prev/(fg_V_prev.sum(dim=1).unsqueeze(1)+1e-10 ) # [bs, 4,K]
        fg_time_vis_weights = (fg_V_prev*(self.fg_scale.view(1,self.history_num,1))).softmax(dim=1)
        # print("print(fg_history_feat_time.shape)")
        # print(fg_history_feat_time.shape)
        # print("print(fg_time_vis_weights.shape)")
        # print(fg_time_vis_weights.shape)
        # fg_history_agg = (fg_history_feat_time * fg_time_vis_weights.unsqueeze(2)).sum(dim=1)  # [bs, c_, K]).sum(dim=1)  # [bs, c_, K]]
        fg_history_agg = (fg_history_feat_time * fg_time_vis_weights.unsqueeze(2)).sum(dim=1)  # [bs, c_, K]).sum(dim=1)  # [bs, c_, K]]


        # print("print(fg_history_agg.shape)")
        # print(fg_history_agg.shape)

        # 可见性聚合与门控
        
        # fg_V_prev_time = fg_V_prev.unsqueeze(1).repeat(1, self.history_num, 1)  # [bs, T, K]
        # fg_V_prev_agg = (fg_V_prev_time * time_weights.squeeze(-1)).sum(dim=1)  # [bs, K]
        fg_V_prev_agg = fg_V_prev.max(dim=1)[0]  # [bs, K]
        fg_w_hist, fg_w_curr = self.compute_gate_weights(fg_V_prev_agg, fg_V_curr.squeeze(1))  # [bs, K, 1]

        # 前景融合
        fg_history_agg_perm = fg_history_agg.permute(0, 2, 1)  # [bs, K, c_]
        fg_curr_perm = fg_curr_feat.permute(0, 2, 1)  # [bs, K, c_]
        fg_fused = fg_w_hist * fg_history_agg_perm + fg_w_curr * fg_curr_perm  # [bs, K, c_]

        # occupancy嵌入融合
        last_occ_reshaped = last_occ_pred.permute(0, 3, 2, 1, 4).reshape(bs, -1, last_occ_pred.shape[-1])  # [bs, N, num_classes]
        fg_occ_feat = torch.gather(last_occ_reshaped, dim=1, index=fg_indices.unsqueeze(-1).repeat(1, 1, last_occ_pred.shape[-1]))  # [bs, K, num_classes]
        fg_occ_embed = self.occ_embedding(fg_occ_feat).permute(0, 2, 1)  # [bs, occ_embedims, K]
        fg_fused = torch.cat([fg_fused, fg_occ_embed.permute(0, 2, 1)], dim=-1)  # [bs, K, c_+occ_embedims]
        fg_fused = self.history_fusion_linear(fg_fused).permute(0, 2, 1)  # [bs, c_, K]

        # 背景融合（原有代码）
        bg_history_feat = torch.gather(history_bev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, mc, 1))  # [bs, mc//2, bg_k]
        bg_curr_feat = torch.gather(curr_bev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, c_, 1))  # [bs, c_, bg_k]

        bg_history_feat_time = bg_history_feat.reshape(bs, self.history_num, c_, -1)  # [bs, T, c_//2, bg_k]
        # bg_history_agg = (bg_history_feat_time * time_weights).sum(dim=1)  # [bs, c_//2, bg_k]
        bg_V_prev = torch.gather(V_prev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, self.history_num, 1))  # [bs, bg_k]
        bg_V_curr = torch.gather(V_curr_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, 1, 1))  # [bs, bg_k]
        #TODO 这个10的超参数？ 调整成可学习？
        # bg_time_vis_weights = bg_V_prev/(bg_V_prev.sum(dim=1).unsqueeze(1) +1e-10) # [bs, 4,K]
        bg_time_vis_weights =(bg_V_prev*(self.bg_scale.view(1,self.history_num,1))).softmax(dim=1)
        bg_history_agg = (bg_history_feat_time * bg_time_vis_weights.unsqueeze(2)).sum(dim=1)  # [bs, c_, K]).sum(dim=1)  # [bs, c_, K]]



        # bg_history_agg_perm = F.pad(bg_history_agg.permute(0, 2, 1), (0, c_ - c_//2, 0, 0))  # [bs, bg_k, c_]

        bg_history_agg_perm = bg_history_agg.permute(0, 2, 1)  # [bs, bg_k, c_]
        # bg_V_prev_time = bg_V_prev.unsqueeze(1).repeat(1, self.history_num, 1)  # [bs, T, bg_k]
        # bg_V_prev_agg = (bg_V_prev_time * time_weights.squeeze(-1)).sum(dim=1)  # [bs, bg_k]
        bg_V_prev_agg = bg_V_prev.max(dim=1)[0]  # [bs, bg_k]
        bg_w_hist, bg_w_curr = self.compute_gate_weights(bg_V_prev_agg, bg_V_curr.squeeze(1))  # [bs, bg_k, 1]

        bg_curr_perm = bg_curr_feat.permute(0, 2, 1)  # [bs, bg_k, c_]
        # print("*"*50)
        # print("print(bg_w_hist.shape)")
        # print(bg_w_hist.shape)
        # print("print(bg_w_curr.shape)")
        # print(bg_w_curr.shape)
        # print("print(bg_history_agg_perm.shape)")
        # print(bg_history_agg_perm.shape)
        # print("print(bg_curr_perm.shape)")        
        # print(bg_curr_perm.shape)

        # # 断言批次大小一致
        # assert bg_w_hist.shape[0] == bg_history_agg_perm.shape[0] == bg_w_curr.shape[0] == bg_curr_perm.shape[0], \
        #     f"批次大小不匹配: {bg_w_hist.shape[0]}, {bg_history_agg_perm.shape[0]}, {bg_w_curr.shape[0]}, {bg_curr_perm.shape[0]}"

        # # 断言第二维度（bg_k）一致
        # assert bg_w_hist.shape[1] == bg_history_agg_perm.shape[1] == bg_w_curr.shape[1] == bg_curr_perm.shape[1], \
        #     f"bg_k维度不匹配: {bg_w_hist.shape[1]}, {bg_history_agg_perm.shape[1]}, {bg_w_curr.shape[1]}, {bg_curr_perm.shape[1]}"

        # # 断言第三维度（c_）匹配（bg_w_hist和bg_w_curr的第三维为1，不影响广播）
        # assert bg_history_agg_perm.shape[2] == bg_curr_perm.shape[2], \
        #     f"特征维度c_不匹配: {bg_history_agg_perm.shape[2]} vs {bg_curr_perm.shape[2]}"

        # print("bg_w_hist dtype:", bg_w_hist.dtype)
        # print("bg_history_agg_perm dtype:", bg_history_agg_perm.dtype)
        # print("bg_w_curr dtype:", bg_w_curr.dtype)
        # print("bg_curr_perm dtype:", bg_curr_perm.dtype)


        # print("bg_w_hist device:", bg_w_hist.device)
        # print("bg_history_agg_perm device:", bg_history_agg_perm.device)
        # print("bg_w_curr device:", bg_w_curr.device)
        # print("bg_curr_perm device:", bg_curr_perm.device)


        bg_fused = bg_w_hist * bg_history_agg_perm + bg_w_curr * bg_curr_perm  # [bs, bg_k, c_]
        # 先验证乘法是否正常
        # temp1 = bg_w_hist * bg_history_agg_perm
        # temp2 = bg_w_curr * bg_curr_perm
        # 再验证加法是否正常

        # bg_w_hist = bg_w_hist.contiguous()
        # bg_history_agg_perm = bg_history_agg_perm.contiguous()
        # bg_w_curr = bg_w_curr.contiguous()
        # bg_curr_perm = bg_curr_perm.contiguous()

        # # 重新计算
        # temp1 = bg_w_hist * bg_history_agg_perm
        # temp2 = bg_w_curr * bg_curr_perm
        # temp1 = temp1.contiguous()
        # temp2 = temp2.contiguous()
        # bg_fused = temp1 + temp2


        # 转移所有张量到CPU
        # bg_w_hist_cpu = bg_w_hist.cpu()
        # bg_history_agg_perm_cpu = bg_history_agg_perm.cpu()
        # bg_w_curr_cpu = bg_w_curr.cpu()
        # bg_curr_perm_cpu = bg_curr_perm.cpu()

        # # 分步执行运算
        # try:
        #     temp1_cpu = bg_w_hist_cpu * bg_history_agg_perm_cpu
        #     temp2_cpu = bg_w_curr_cpu * bg_curr_perm_cpu
        #     bg_fused_cpu = temp1_cpu + temp2_cpu
        #     print(bg_fused_cpu)
        #     print(bg_fused_cpu.shape)
        #     print("CPU运算成功，无明显错误")
        # except Exception as e:
        #     print(f"CPU运算报错：{e}")  # 此处会显示具体错误原因（如值异常）


        # bg_fused = temp1.clone() + temp2.clone()
        # 1/0
        bg_occ_feat = torch.gather(last_occ_reshaped, dim=1, index=bg_indices.unsqueeze(-1).repeat(1, 1, last_occ_pred.shape[-1]))  # [bs, bg_k, num_classes]
        bg_occ_embed = self.occ_embedding(bg_occ_feat).permute(0, 2, 1)  # [bs, occ_embedims, bg_k]
        bg_fused = torch.cat([bg_fused, bg_occ_embed.permute(0, 2, 1)], dim=-1)  # [bs, bg_k, c_+occ_embedims]
        bg_fused = self.history_fusion_bg_linear(bg_fused).permute(0, 2, 1)  # [bs, c_, bg_k]

        # 更新当前BEV
        curr_bev_updated = curr_bev_flat.clone()
        curr_bev_updated.scatter_add_(dim=2, index=fg_indices.unsqueeze(1).repeat(1, c_, 1), src=fg_fused)
        curr_bev_updated.scatter_add_(dim=2, index=bg_indices.unsqueeze(1).repeat(1, c_, 1), src=bg_fused)
        curr_bev_updated = curr_bev_updated.reshape(bs, c_, z, h, w)  # 恢复原形状

        # 更新历史缓存
        self.history_last_bev = curr_bev_updated.detach().clone()
        self.history_bev = torch.cat([curr_bev,sampled_history_bev[:, :-c_, ...]], dim=1).detach()
        # self.history_cam_intrins = torch.cat([curr_cam_intrins.unsqueeze(1),self.history_cam_intrins[:, :-1, ...]], dim=1).detach()
        # self.history_cam_extrins = torch.cat([curr_cam_extrins.unsqueeze(1),self.history_cam_extrins[:, 1-1:, ...]], dim=1).detach()
        self.history_visibility =torch.cat([V_curr, V_prev[:, :-1, ...]],dim=1).detach()
        self.history_forward_augs = forward_augs.clone()

        return curr_bev_updated

    def generate_grid(self, curr_bev):
        n, c_, z, h, w = curr_bev.shape
        # Generate grid
        xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w, 1).expand(h, w, z)
        ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1, 1).expand(h, w, z)
        zs = torch.linspace(0, z - 1, z, dtype=curr_bev.dtype, device=curr_bev.device).view(1, 1, z).expand(h, w, z)
        grid = torch.stack((xs, ys, zs, torch.ones_like(xs)), -1).view(1, h, w, z, 4).expand(n, h, w, z, 4).view(n, h,w, z, 4, 1)
        return grid

    def generate_feat2bev(self, grid, dx, bx):
        feat2bev = torch.zeros((4, 4), dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = dx[0]
        feat2bev[1, 1] = dx[1]
        feat2bev[2, 2] = dx[2]
        feat2bev[0, 3] = bx[0] - dx[0] / 2.
        feat2bev[1, 3] = bx[1] - dx[1] / 2.
        feat2bev[2, 3] = bx[2] - dx[2] / 2.
        feat2bev[3, 3] = 1
        feat2bev = feat2bev.view(1, 4, 4)
        return feat2bev





@HEADS.register_module()
class GatedTemporalFusion6_cat(BaseModule):
    def __init__(
        self,
        history_num=4,
        depth_sampler_embed_dims=256,
        depth_sampler_num_heads=8,
        depth_sampler_num_levels=4,
        depth_sampler_num_points=4,
        im2col_step=64,
        top_k=500,
        single_bev_num_channels=96,
        occ_embedims=32,
        num_classes=18,
        vis_theta=0.28,
        vis_beta=10.0,
        vis_gamma=0.4,
        vis_sigma=0.1,
        nonempty_thresh=0.1,
        max_step_ratio=1.2,
        **kwargs
    ):
        super(GatedTemporalFusion6_cat, self).__init__()

        # 基础参数
        self.history_num = history_num
        self.top_k = top_k
        self.single_bev_num_channels = single_bev_num_channels
        self.occ_embedims = occ_embedims
        # self.fg_scale = nn.Parameter(torch.full((history_num,), 10.0))
        # self.bg_scale = nn.Parameter(torch.full((history_num,), 10.0))
        # 可见性门控参数
        # self.vis_theta = vis_theta  # 可见性阈值
        # self.vis_beta = vis_beta    # 敏感度参数
        # self.vis_gamma = vis_gamma  # 场景4固定权重
        # self.vis_sigma = vis_sigma  # 软化参数
        # self.vis_theta = nn.Parameter(torch.tensor(vis_theta))   # 可见性阈值
        # self.vis_beta = nn.Parameter(torch.tensor(vis_beta))     # 敏感度参数
        # self.vis_gamma = nn.Parameter(torch.tensor(vis_gamma))   # 场景4固定权重
        # self.vis_sigma = nn.Parameter(torch.tensor(vis_sigma))  

        
        # 历史特征缓存（新增history_last_bev）
        self.history_bev = None  # 缓存历史多帧BEV特征
        self.history_last_bev = None  # 缓存上一帧最终融合后的BEV特征（关键新增）
        self.history_cam_intrins = None
        self.history_cam_extrins = None
        
        # 网络层（保持不变）
        self.occ_embedding = nn.Sequential(
            nn.Linear(num_classes, occ_embedims),
            nn.Softplus(),
            nn.Linear(occ_embedims, occ_embedims),
        )
        self.history_fusion_linear = nn.Sequential(
            nn.Linear(single_bev_num_channels * (history_num + 1) + occ_embedims, single_bev_num_channels),
            nn.Softplus(),
            nn.Linear(single_bev_num_channels, single_bev_num_channels),
        )
        self.history_fusion_bg_linear = nn.Sequential(
            nn.Linear(single_bev_num_channels * (history_num//2 + 1) + occ_embedims, single_bev_num_channels),
            nn.Softplus(),
            nn.Linear(single_bev_num_channels, single_bev_num_channels),
        )

        
        # 可见性计算组件（保持不变）
        self.rt_vis_calculator = EfficientRayTracingVisibility(
            nonempty_thresh=nonempty_thresh,
            max_step_ratio=max_step_ratio
        )
        # self.img_shape = (900, 1600)  # 默认图像尺寸

        # self.depth_sampler = DeformableDepthSampler(
        #     embed_dims=depth_sampler_embed_dims,
        #     num_heads=depth_sampler_num_heads,
        #     num_levels=depth_sampler_num_levels,
        #     num_points=depth_sampler_num_points
        # )
        self.history_forward_augs = None  # 用于缓存历史帧的变换矩阵（如BDAM矩阵）
        self.im2col_step = im2col_step
        # self.voxel_encoder = nn.Linear(single_bev_num_channels, depth_sampler_embed_dims)
        self.dbound = [1.0, 45.0, 0.5]
        self.pc_range = [-40, -40, -1.0, 40, 40, 5.4]
        self.final_dim = (256, 704)
    def compute_visibility(self, grid, cam_intrins, cam_extrins, img_shape, img_feats, spatial_shapes):
        """
        升级：结合可变形注意力采样的深度值优化可见性计算
        Args:
            img_feats: 多尺度图像特征 [bs, num_levels, c, h, w]
            spatial_shapes: 图像特征尺度 [num_levels, 2] (h, w)
            其他参数同原函数
        Returns:
            vis_prob: [bs, h, w, z] 优化后的可见性概率
        """
        bs, h, w, z, _ = grid.shape
        h_img, w_img = img_shape
        device = grid.device
        num_voxels = h * w * z  # 体素总数

        # 1. 原有相机投影逻辑（计算图像坐标和初始可见性）
        # 1.1 体素坐标→相机坐标→图像坐标
        grid_cam = grid.unsqueeze(1).expand(bs, self.num_cams, h, w, z, 3)  # [bs, num_cams, h, w, z, 3]
        grid_flat = grid_cam.reshape(-1, num_voxels, 3)  # [bs*num_cams, N, 3]
        grid_hom = torch.cat([grid_flat, torch.ones_like(grid_flat[..., :1])], dim=-1)  # [bs*num_cams, N, 4]

        cam_intrins_flat = cam_intrins.reshape(-1, 4, 4)  # [bs*num_cams, 4, 4]
        cam_extrins_flat = cam_extrins.reshape(-1, 4, 4)  # [bs*num_cams, 4, 4]
        extrins_inv = torch.inverse(cam_extrins_flat)
        cam_coords = torch.bmm(extrins_inv[:, :3, :4], grid_hom.permute(0, 2, 1)).permute(0, 2, 1)  # [bs*num_cams, N, 3]
        depth = cam_coords[..., 2:3] + 1e-8  # 相机坐标系下的深度

        # 1.2 计算初始可见性（原逻辑）
        img_coords = torch.bmm(cam_intrins_flat[:, :3, :3], cam_coords.permute(0, 2, 1)).permute(0, 2, 1)  # [bs*num_cams, N, 3]
        img_xy = img_coords[..., :2] / depth  # [bs*num_cams, N, 2] (u, v)
        depth_valid = (cam_coords[..., 2] > 0).float()  # [bs*num_cams, N]
        u_valid = (img_xy[..., 0] >= 0) & (img_xy[..., 0] < w_img)
        v_valid = (img_xy[..., 1] >= 0) & (img_xy[..., 1] < h_img)
        img_valid = (u_valid & v_valid).float()  # [bs*num_cams, N]
        initial_vis = depth_valid * img_valid  # [bs*num_cams, N]

        # 2. 可变形注意力深度采样
        # 2.1 准备输入：体素特征编码
        # voxel_feat = self.voxel_encoder(self.curr_bev_feat)  # [bs, c_, z, h, w] → [bs, z*h*w, embed_dims]（需提前展平体素特征）
        # voxel_feat = voxel_feat.reshape(bs, num_voxels, -1)  # [bs, N, embed_dims]

        # 2.2 生成参考点（归一化到[0,1]）
        norm_img_xy = img_xy / torch.tensor([w_img, h_img], device=device).view(1, 1, 2)  # [bs*num_cams, N, 2]
        # 取主相机（如第0个相机）的参考点作为采样基准
        ref_points = norm_img_xy.reshape(bs, self.num_cams, num_voxels, 2)[:, 0]  # [bs, N, 2]
        ref_points = ref_points.unsqueeze(2).repeat(1, 1, self.depth_sampler.num_levels, 1)  # [bs, N, L, 2]

        # 2.3 采样深度特征
        sampled_depth = self.depth_sampler(
            query=voxel_feat,
            value=img_feats,  # 多尺度图像特征 [bs, L, c, h, w]
            reference_points=ref_points,
            spatial_shapes=spatial_shapes
        )  # [bs, N]

        # 3. 结合采样深度优化可见性
        # 3.1 深度一致性校验：采样深度与相机投影深度的差异
        cam_depth = depth.reshape(bs, self.num_cams, num_voxels)[:, 0]  # 主相机的投影深度 [bs, N]
        depth_diff = torch.abs(sampled_depth - cam_depth) / (cam_depth + 1e-8)  # 相对深度差
        depth_consistent = (depth_diff < 0.3).float()  # 深度差小于30%则认为有效

        # 3.2 融合可见性：初始可见性 × 深度一致性
        initial_vis = initial_vis.reshape(bs, self.num_cams, num_voxels).max(dim=1)[0]  # [bs, N]（多相机取max）
        vis_prob = initial_vis * depth_consistent  # [bs, N]

        # 4. 还原形状
        return vis_prob.reshape(bs, h, w, z)  # [bs, h, w, z]

    def compute_alpha_unified(self, V_curr, V_prev):
        """统一计算当前帧融合权重α（覆盖四场景）"""
        eps = 1e-8
        # 1. 基础动态权重σ_base（场景1-3）
        ratio = V_curr / (V_curr + V_prev + eps)
        sigma_base = torch.sigmoid(self.vis_beta * (ratio - 0.5))
        
        # 2. 场景4软化掩码σ_both
        mask_curr = torch.sigmoid(-(V_curr - self.vis_theta) / self.vis_sigma)
        mask_prev = torch.sigmoid(-(V_prev - self.vis_theta) / self.vis_sigma)
        sigma_both = mask_curr * mask_prev
        
        # 3. 最终权重计算
        alpha = (1 - sigma_both) * sigma_base + sigma_both * self.vis_gamma
        return alpha.unsqueeze(-1)  # [bs, N, 1]

    def compute_gate_weights(self, V_prev_agg, V_curr):
        """计算历史和当前帧的门控权重"""
        alpha = self.compute_alpha_unified(V_curr, V_prev_agg)
        return 1 - alpha, alpha  # 历史权重，当前权重

    def generate_grid(self, curr_bev, voxel_min, voxel_max, voxel_size):
        """生成体素中心坐标网格（自车坐标系）"""
        bs, c_, z, h, w = curr_bev.shape
        device = curr_bev.device
        
        # 计算体素中心坐标
        x_coords = torch.linspace(
            voxel_min[0] + voxel_size[0]/2, 
            voxel_max[0] - voxel_size[0]/2, 
            w, device=device
        )
        y_coords = torch.linspace(
            voxel_min[1] + voxel_size[1]/2, 
            voxel_max[1] - voxel_size[1]/2, 
            h, device=device
        )
        z_coords = torch.linspace(
            voxel_min[2] + voxel_size[2]/2, 
            voxel_max[2] - voxel_size[2]/2, 
            z, device=device
        )
        
        # 生成网格并扩展维度 [bs, h, w, z, 3]
        x_grid, y_grid, z_grid = torch.meshgrid(x_coords, y_coords, z_coords, indexing='xy')
        grid = torch.stack([x_grid, y_grid, z_grid], dim=-1)  # [w, h, z, 3]
        grid = grid.permute(1, 0, 2, 3)  # [h, w, z, 3]
        return grid.unsqueeze(0).repeat(bs, 1, 1, 1, 1)  # [bs, h, w, z, 3]


    def get_reference_points(self, H, W, Z=None, num_points_in_pillar =4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
        torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    @force_fp32(apply_to=('reference_points', 'cam_params'))
    def point_sampling(self, reference_points, pc_range, img_metas, cam_params=None):
        # prepare for point sampling
        lidar2img = []
        ego2lidar = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])     # lidar2img update the post aug in the loading pipeline
            ego2lidar.append(img_meta['ego2lidar'])
        lidar2img = torch.stack(lidar2img, dim=0).to(reference_points.device)
        ego2lidar = torch.stack(ego2lidar, dim=0).to(reference_points.device)

        sensor2egos, ego2globals, intrins, post_augs, bda_mat = cam_params
        num_cam = sensor2egos.size(1)
        ogfH, ogfW = self.final_dim

        # reference_points defines in the bev space, [bs, D, hxw, 3]
        # change reference_points from bev-ego coordinate to ego coordinate
        reference_points = reference_points.clone()
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
                                     (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
                                     (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
                                     (pc_range[5] - pc_range[2]) + pc_range[2]

        # prepare for point sampling
        reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
        reference_points = reference_points.permute(1, 0, 2, 3)  # shape: (num_points_in_pillar,bs,h*w,4)
        D, B, num_query = reference_points.size()[:3]  # D=num_points_in_pillar , num_query=h*w
        reference_points = reference_points.view(D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)  # shape: (num_points_in_pillar,bs,num_cam,h*w,4)
        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)
        ego2lidar = ego2lidar.view(1, B, 1, 1, 4, 4).repeat(D, 1, num_cam, num_query, 1, 1)
        inverse_bda = bda_mat.view(1, B, 1, 1, 4, 4).repeat(D, 1, num_cam, num_query, 1, 1)

        # change reference_points from ego coordinate to img coordinate
        eps = 1e-5
        reference_points_cam = (lidar2img @ ego2lidar @ inverse_bda @ reference_points).squeeze(-1)   # [num_points_in_pillar, bs, num_cam, num_query=h*w, 4]
        reference_points_depth = reference_points_cam[..., 2:3]
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(reference_points_depth, torch.ones_like(reference_points_depth) * eps)

        # Bug!!
        # Correct normalize is
        # reference_points_cam[..., 0] /= ogfW
        # reference_points_cam[..., 1] /= ogfH
        # But for reproducing our results, we use the following normalization
        reference_points_cam[..., 0] /= ogfW
        reference_points_cam[..., 1] /= ogfH

        bev_mask = (reference_points_depth > eps)
        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)                  # shape: (num_cam, bs,h*w, num_points_in_pillar, 2)
        reference_points_depth = reference_points_depth.permute(2, 1, 3, 0, 4)              # shape: (num_cam, bs,h*w, num_points_in_pillar, 1)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)                        # shape: (num_cam, bs,h*w, num_points_in_pillar)

        return reference_points_cam, reference_points_depth, bev_mask


    @force_fp32()
    def forward(self, curr_bev, cam_params, history_fusion_params, dx, bx, history_last_bev=None, last_occ_pred=None, nonempty_prob=None, img_feats=None, spatial_shapes=None,pred_img_depth=None,**kwargs):
        """
        Args:
            curr_bev: [bs, c, z, h, w] 当前帧BEV特征
            cam_params: 相机参数列表，包含外参、内参等
            history_fusion_params: 历史融合参数（包含序列信息等）
            dx: 体素尺寸 (x, y, z)
            bx: 体素偏移
            nonempty_prob: [bs, z, h, w] 体素非空概率
            last_occ_pred: [bs, z, h, w, num_classes] 上一时刻 occupancy 预测
            img_feats: 多尺度图像特征 [bs, num_levels, c, h, w]（新增，用于深度采样）
            spatial_shapes: 图像特征尺度 [num_levels, 2] (h, w)（新增，用于深度采样）
        Returns:
            curr_bev_updated: [bs, c, z, h, w] 融合后BEV特征
        """
        # print(self.history_num)
        # print("print(self.fg_scale)")
        # print(self.fg_scale) 
        # print("print(self.bg_scale)")
        # print(self.bg_scale)
        # # 可见性门控参数
        # # self.vis_theta = vis_theta  # 可见性阈值
        # # self.vis_beta = vis_beta    # 敏感度参数
        # # self.vis_gamma = vis_gamma  # 场景4固定权重
        # # self.vis_sigma = vis_sigma  # 软化参数
        # print("print(self.vis_theta # 可见性阈值)")
        # print(self.vis_theta )  # 可见性阈值
        # print("print(self.vis_beta) # 敏感度参数")
        # print(self.vis_beta)      # 敏感度参数
        # print("print(self.vis_gamma) # 场景4固定权重")
        # print(self.vis_gamma)  # 场景4固定权重
        # print("print(self.vis_sigma) # 软化参数")
        # print(self.vis_sigma) 


        # if torch.rand(1).item() < 1/2000:
        #     print(self.history_num)
        #     print("print(self.fg_scale)")
        #     print(self.fg_scale) 
        #     print("print(self.bg_scale)")
        #     print(self.bg_scale)
        #     # 可见性门控参数
        #     print("print(self.vis_theta # 可见性阈值)")
        #     print(self.vis_theta)  # 可见性阈值
        #     print("print(self.vis_beta) # 敏感度参数")
        #     print(self.vis_beta)      # 敏感度参数
        #     print("print(self.vis_gamma) # 场景4固定权重")
        #     print(self.vis_gamma)  # 场景4固定权重
        #     print("print(self.vis_sigma) # 软化参数")
        #     print(self.vis_sigma)  


        # -------------------------- 1. 解析参数后打印核心形状 --------------------------
        # 解析相机参数
        curr_cam_extrins = cam_params[0]  # [bs, num_cams, 4, 4]
        curr_cam_intrins = cam_params[2]  # [bs, num_cams, 4, 4]
        forward_augs = cam_params[4]      # [bs, 4, 4] 前向变换矩阵
        self.num_cams = curr_cam_extrins.shape[1]  # 从外参中获取相机数量
        bs, c_, z, h, w = curr_bev.shape
        device = curr_bev.device
        mc = self.history_num * c_        # 历史特征总通道数
        # self.history_forward_augs = forward_augs.clone()


        ref_3d = self.get_reference_points(
            h, w, z, z, dim='3d', bs=bs, device=device, dtype=curr_bev.dtype) # torch.Size([3, 2, 625, 3]) #[bs,z,yx,3(x,y,z)]
        # ref_2d = self.get_reference_points(
        #     h, w, dim='2d', bs=bs, device=device, dtype=curr_bev.dtype) #torch.Size([3, 625, 1, 2])
        slots = torch.zeros(list([ref_3d.shape[0],ref_3d.shape[2],ref_3d.shape[1]])).to(ref_3d)
        reference_points_cam, reference_points_depth, bev_mask = self.point_sampling(ref_3d, self.pc_range, img_metas=kwargs['img_metas'], cam_params=cam_params)
        indexes = [[] for _ in range(bs)]
        spatial_shapes =[]
        spatial_shapes.append([16, 44])
        spatial_shapes = torch.tensor(spatial_shapes).to(device)
        pred_img_depth = pred_img_depth.view(bs * 6, -1, spatial_shapes[0][0], spatial_shapes[0][1])
        pred_img_depth = pred_img_depth.flatten(2).permute(0, 2, 1)  
        max_len = 0
        for j in range(bs):
            for i, mask_per_img in enumerate(bev_mask):
                index_query_per_img = mask_per_img[j].sum(-1).nonzero().squeeze(-1)
                if len(index_query_per_img) == 0:
                    index_query_per_img = bev_mask[i][j].sum(-1).nonzero().squeeze(-1)[0:1]
                indexes[j].append(index_query_per_img)
                # for batch operation, we need to pad the indexes to the same length
                max_len = max(max_len, len(index_query_per_img))
        reference_points_cam_rebatch = reference_points_cam.new_zeros([bs, self.num_cams, max_len, z, 2])
        reference_points_depth_rebatch = reference_points_depth.new_zeros([bs, self.num_cams, max_len, z, 1])

        for j in range(bs):
            for i, (reference_points_per_img, reference_points_depth_per_img) in enumerate(zip(reference_points_cam, reference_points_depth)):
                index_query_per_img = indexes[j][i]
                reference_points_cam_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]
                reference_points_depth_rebatch[j, i, :len(index_query_per_img)] = reference_points_depth_per_img[j, index_query_per_img]

        #use deformble attn
        depth_reference_points = reference_points_cam_rebatch.reshape(bs*6, max_len*z, 1, 1, 1, 2).contiguous()
        depth_attention_weights = torch.ones_like(depth_reference_points[..., 0]).contiguous()
        pred_img_depth = pred_img_depth.unsqueeze(2).contiguous()
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))



        bev_query_depth_rebatch = (reference_points_depth_rebatch- self.dbound[0])/ self.dbound[2]
        bev_query_depth_rebatch = torch.clip(torch.floor(bev_query_depth_rebatch), 0, 88-1).to(torch.long)
        bev_query_depth_rebatch = F.one_hot(bev_query_depth_rebatch.squeeze(-1),
                                   num_classes=88)

        depth_output = MultiScaleDeformableAttnFunction_fp32.apply(pred_img_depth, spatial_shapes,level_start_index,depth_reference_points,depth_attention_weights, self.im2col_step)
        depth_output = depth_output.reshape(bs,6, max_len,z, -1)   # [bs*num_cam, num_query, num_Z_anchors, C]
        # reference_points_depth_rebatch

        increment = torch.zeros_like(depth_output)
        # increment[..., 0] = 1e-9  # 非原地赋值（创建新张量）
        # depth_output = depth_output + increment
        depth_output = depth_output + torch.cat([(torch.zeros_like(depth_output[...,:1]) + 1e-9),torch.zeros_like(depth_output[...,1:])],dim=-1)

        # print("depth_output.sum(-1).shape")
        # print(depth_output.sum(-1).shape)
        # print("depth_output.sum(-1)")
        # print(depth_output.sum(-1))
        # print("print((depth_output.sum(-1)==0).sum())")
        # print((depth_output.sum(-1)==0).sum())
        depth_output =depth_output/depth_output.sum(-1)[...,None] #bs,xy,z,D

        # print("depth_output.sum(-1).shape")
        # print(depth_output.sum(-1).shape)
        # print("depth_output.sum(-1)")
        # print(depth_output.sum(-1))
        # print("print((depth_output.sum(-1)==1).sum())")
        # print((depth_output.sum(-1)>=0.99).sum())
        # print("depth_output")
        # print(depth_output)


        depth_output = (1-depth_output.cumsum(dim=-1))
        depth_output =torch.cat([torch.ones_like(depth_output[...,0:1]),depth_output[...,0:-1]],dim=-1)

        # print("print((depth_output[...,-1]==0).sum())")
        # print((depth_output[...,-1]==0).sum())
        depth_output = (bev_query_depth_rebatch*depth_output).sum(-1)

        #恢复depth_output的shape
        # fix_depth_output = depth_output.new_zeros([bs, 6, w*h, z, 88])
        # for j in range(bs):
        #     for i in range(6):
        #         index_query_per_img = indexes[j][i]
        #         fix_depth_output[j, i, index_query_per_img] = depth_output[j, i, :len(index_query_per_img)]

        for j in range(bs):
            for i in range(6):
                index_query_per_img = indexes[j][i]
                slots[j, index_query_per_img] = torch.max(slots[j, index_query_per_img],depth_output[j, i, :len(index_query_per_img)])

        # depth_sum = fix_depth_output.sum(dim=-1).view(3,6,25,25,2)
        # bs = depth_sum.shape[0]       # 3
        # num_cam = depth_sum.shape[1]  # 6
        # z_layers = depth_sum.shape[4] # 2
        # H, W = depth_sum.shape[2], depth_sum.shape[3]  # 25,25

        # # 颜色映射：0值用黑色，非0值用渐变色
        # cmap = plt.cm.viridis
        # cmap.set_bad(color='black')  # 0值标记为黑色

        # # 设置子图布局（不变）
        # fig, axes = plt.subplots(
        #     nrows=bs, ncols=num_cam * z_layers,
        #     figsize=(30, 8),
        #     squeeze=False
        # )
        # fig.suptitle('fix_depth_output.sum(-1) 可视化（黑色=0值，颜色=非0值）', fontsize=16, y=0.98)


        # # -------------------------- 2. 循环绘制每个子图（核心修改：添加 .detach()） --------------------------
        # for batch_idx in range(bs):
        #     for cam_idx in range(num_cam):
        #         for z_idx in range(z_layers):
        #             col_idx = cam_idx * z_layers + z_idx
        #             ax = axes[batch_idx, col_idx]
                    
        #             # 核心修改：先 detach() 切断计算图，再转 cpu 和 numpy
        #             data = depth_sum[batch_idx, cam_idx, :, :, z_idx].detach().cpu().numpy()
        #             data[data == 0] = np.nan  # 0值替换为NaN，显示为黑色
                    
        #             # 绘制热力图（修改 vmin/vmax：同样添加 .detach()）
        #             im = ax.imshow(
        #                 data, 
        #                 cmap=cmap, 
        #                 aspect='auto',
        #                 # 关键修改：depth_sum 先 detach 再转 numpy，确保不影响梯度
        #                 vmin=np.nanmin(depth_sum.detach().cpu().numpy()),
        #                 vmax=np.nanmax(depth_sum.detach().cpu().numpy())
        #             )
                    
        #             # 子图标题和坐标轴（不变）
        #             ax.set_title(
        #                 f'Batch{batch_idx+1}\nCam{cam_idx+1} Z{z_idx+1}',
        #                 fontsize=10, pad=5
        #             )
        #             ax.set_xticks([])
        #             ax.set_yticks([])


        # # -------------------------- 3. 添加颜色条（不变） --------------------------
        # cbar = fig.colorbar(
        #     im, 
        #     ax=axes.ravel().tolist(),
        #     shrink=0.8,
        #     pad=0.02
        # )
        # cbar.set_label('Sum of Depth Bins (D=88)', fontsize=12)


        # # -------------------------- 4. 调整布局并保存（不变） --------------------------
        # plt.tight_layout(rect=[0, 0, 0.98, 0.95])
        # plt.savefig('depth_sum_visualization.png', dpi=300, bbox_inches='tight')
        # plt.show()

        # output

        #计数更新，建立在纸上的假设成立的基础上
        # count = bev_mask.sum(-1) > 0
        # count = count.permute(1, 2, 0).sum(-1)
        # count = torch.clamp(count, min=1.0)
        # slots = slots / count[..., None]

        # print("slots.shape")
        # print(slots.shape)
        # print("slots")  
        # print(slots)
        # zzzzz=1/0

        V_curr =slots.view(bs, 1, h, w,z).permute(0, 1, 4, 2, 3)
        # slots[...,0]+=1e-9
        # print("slots.sum(-1).shape")
        # print(slots.sum(-1).shape)
        # print("slots.sum(-1)")
        # print(slots.sum(-1))
        # print("print((slots.sum(-1)==0).sum())")
        # print((slots.sum(-1)==0).sum())
        # slots =slots/slots.sum(-1)[...,None] #bs,xy,z,D

        # print("slots.sum(-1).shape")
        # print(slots.sum(-1).shape)
        # print("slots.sum(-1)")
        # print(slots.sum(-1))
        # print("print((slots.sum(-1)==1).sum())")
        # print("slots")
        # print(slots)


        # slots = (1-slots.cumsum(dim=-1))

        # print("print((slots[...,-1]==0).sum())")
        # print((slots[...,-1]==0).sum())

        # print("slots")
        # print(slots)
        # print("print((slots[...,-1]<0.01).sum())")
        # print((slots[...,-1]<0.01).sum())

        #TODO 这里的对于边界值的考虑，从0开始还是从1开始？




        # slots = self.output_proj(slots)
        #TODO 上面这里需要检查一下
        #这里相当于两次softmax，可能会导致分布变得不够尖锐，需要进一步确认

        #到这里slots就是可见性的概率分布了
        #这里先尝试使用期望值进行计算，使得可微分

        #然后再采用stc的原始离散计算方法，


        # 打印解析后关键变量形状
        # print("="*50)
        # print("1. 解析参数后核心变量形状：")
        # print(f"curr_bev: {curr_bev.shape} (预期：[bs, c, z, h, w])")
        # print(f"curr_cam_extrins: {curr_cam_extrins.shape} (预期：[bs, num_cams, 4, 4])")
        # print(f"curr_cam_intrins: {curr_cam_intrins.shape} (预期：[bs, num_cams, 4, 4])")
        # print(f"forward_augs: {forward_augs.shape} (预期：[bs, 4, 4])")
        # print(f"dx: {dx.shape if hasattr(dx, 'shape') else type(dx)} (预期：[3])")
        # print(f"bx: {bx.shape if hasattr(bx, 'shape') else type(bx)} (预期：[3])")
        # print(f"bs: {bs}, c_: {c_}, z: {z}, h: {h}, w: {w} (BEV特征维度)")
        # print("="*50)

        if type(history_fusion_params['sequence_group_idx']) is list:
            seq_ids = history_fusion_params['sequence_group_idx'][0]
        else:
            seq_ids = history_fusion_params['sequence_group_idx']
        if type(history_fusion_params['start_of_sequence']) is list:
            start_of_sequence = history_fusion_params['start_of_sequence'][0]
        else:
            start_of_sequence = history_fusion_params['start_of_sequence']
        if type(history_fusion_params['curr_to_prev_ego_rt']) is list:
            curr_to_prev_ego_rt = history_fusion_params['curr_to_prev_ego_rt'][0]
        else:
            curr_to_prev_ego_rt = history_fusion_params['curr_to_prev_ego_rt']
        forward_augs = cam_params[-1]  # bda

        # check seq_ids > 0
        assert (seq_ids >= 0).all()
        # -------------------------- 2. 初始化历史缓存后打印 --------------------------
        if self.history_bev is None:
            # self.history_bev = curr_bev.repeat(1, self.history_num, 1, 1, 1)  # [bs, mc, z, h, w]
            # self.history_cam_intrins = curr_cam_intrins.unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)  # [bs, T, num_cams, 4, 4]
            # self.history_cam_extrins = curr_cam_extrins.unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)  # [bs, T, num_cams, 4, 4]
            # self.history_bev = curr_bev.clone()
            self.history_forward_augs = forward_augs.clone()
            self.history_seq_ids = seq_ids.clone()
            self.history_bev = curr_bev.repeat(1, self.history_num, 1, 1, 1)
            self.history_sweep_time = curr_bev.new_zeros(curr_bev.shape[0], self.history_num)
            self.history_visibility = V_curr.repeat(1, self.history_num, 1, 1, 1).half()
        self.history_bev = self.history_bev.detach()
        self.history_visibility = self.history_visibility.detach().half()
        self.history_sweep_time += 1

        # 打印历史缓存形状
        # print("\n2. 历史缓存初始化后形状：")
        # print(f"history_bev: {self.history_bev.shape} (预期：[bs, mc, z, h, w]，mc={mc})")
        # print(f"history_cam_intrins: {self.history_cam_intrins.shape} (预期：[bs, history_num, num_cams, 4, 4])")
        # print(f"history_cam_extrins: {self.history_cam_extrins.shape} (预期：[bs, history_num, num_cams, 4, 4])")

        # -------------------------- 3. 生成网格和BEV变换矩阵后打印 --------------------------
        # 处理新序列（略，不影响维度）
        # start_of_sequence = history_fusion_params.get('start_of_sequence', torch.zeros(bs, dtype=torch.bool, device=device))
        if start_of_sequence.sum()>0:
            self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_num, 1, 1, 1)
            # self.history_cam_intrins[start_of_sequence] = curr_cam_intrins[start_of_sequence].unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)
            # self.history_cam_extrins[start_of_sequence] = curr_cam_extrins[start_of_sequence].unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)
            self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]
            self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
            self.history_sweep_time[start_of_sequence] = 0  # zero the new sequence timestep starts
            self.history_visibility[start_of_sequence] = V_curr[start_of_sequence].repeat(1, self.history_num, 1, 1, 1).half()

        # 生成体素网格和BEV变换矩阵
        grid = self.generate_grid(curr_bev) #[bs,y,x,z,4]
        grid_3d = grid
        feat2bev = self.generate_feat2bev(grid, dx, bx)

        # 打印网格和变换矩阵形状
        # print("\n3. 生成网格和BEV变换矩阵后形状：")
        # print(f"grid_3d (体素网格): {grid_3d.shape} (关键！预期：[bs, h, w, z, 3] 或 [bs, w, h, z, 3])")
        # print(f"feat2bev (BEV变换矩阵): {feat2bev.shape} (预期：[bs, 4, 4])")

        # -------------------------- 4. 运动补偿矩阵计算后打印 --------------------------
        # 获取帧间姿态变换
        # curr_to_prev_ego_rt = history_fusion_params.get('curr_to_prev_ego_rt', torch.eye(4, device=device).unsqueeze(0).repeat(bs, 1, 1))
        # 计算RT流（坐标变换矩阵）
        rt_flow = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_ego_rt @ torch.inverse(forward_augs) @ feat2bev)
        # 生成齐次网格
        # 在forward函数中，生成grid_hom的位置修正：
        # grid_3d = self.generate_grid(curr_bev)  # 现在形状：[3, 25, 25, 2, 3]（bs, h, w, z, 3）
        # # 生成齐次坐标（x,y,z,1），并添加最后一个维度（用于矩阵乘法）
        # grid_hom = torch.cat([
        #     grid_3d,  # [3,25,25,2,3]
        #     torch.ones_like(grid_3d[..., :1])  # [3,25,25,2,1]（补充1作为齐次坐标）
        # ], dim=-1).unsqueeze(-1)  # 最终形状：[3,25,25,2,4,1]（符合预期）
        # # 打印运动补偿相关形状（矩阵乘法前关键检查）
        # print("\n4. 运动补偿矩阵计算后形状（矩阵乘法前）：")
        # print(f"curr_to_prev_ego_rt (帧间姿态): {curr_to_prev_ego_rt.shape} (预期：[bs, 4, 4])")
        # print(f"rt_flow (变换流): {rt_flow.shape} (预期：[bs, 4, 4])")
        # print(f"grid_hom (齐次网格): {grid_hom.shape} (关键！预期：[bs, h, w, z, 4, 1]，需与rt_flow广播匹配)")
        # print(f"rt_flow.view后: {rt_flow.view(bs, 1, 1, 1, 4, 4).shape} (预期：[bs, 1, 1, 1, 4, 4])")

        # # -------------------------- 5. 网格变换后打印（解决之前维度错的核心） --------------------------
        # try:
        #     grid_transformed = rt_flow.view(bs, 1, 1, 1, 4, 4) @ grid_hom  # 矩阵乘法：[bs, h, w, z, 4, 1]
        #     print("\n5. 网格变换后形状（矩阵乘法成功！）：")
        #     print(f"grid_transformed: {grid_transformed.shape} (预期：[bs, h, w, z, 4, 1])")
        # except RuntimeError as e:
        #     print(f"\n5. 网格变换矩阵乘法报错！错误信息：{str(e)}")
        #     print(f"  - rt_flow.view形状: {rt_flow.view(bs, 1, 1, 1, 4, 4).shape}")
        #     print(f"  - grid_hom形状: {grid_hom.shape}")
        #     print("  提示：需确保grid_hom的第1-4维度与rt_flow.view的第2-5维度匹配（广播规则）")
        #     raise e  # 继续抛出错误，方便定位
        bs, mc, z, h, w = self.history_bev.shape
        n, c_, z, h, w = curr_bev.shape
        grid = rt_flow.view(n, 1, 1, 1, 4, 4) @ grid
        # -------------------------- 6. 采样网格生成后打印 --------------------------
        # 生成采样网格（归一化到[-1,1]，适配F.grid_sample）
        normalize_factor = torch.tensor([w - 1.0, h - 1.0, z - 1.0], dtype=curr_bev.dtype, device=device)
        # grid_sampler = grid_transformed[..., :3, 0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0  # [bs, h, w, z, 3]
        # # 调整采样网格维度（适配F.grid_sample输入：[bs, z, h, w, 3]）
        # grid_sampler_permuted = grid_sampler.permute(0, 3, 1, 2, 4)  # 交换z和h/w维度
        grid = grid[:, :, :, :, :3, 0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0   # grid order is x, y, z


        # print("\n6. 采样网格生成后形状：")
        # print(f"grid_sampler (归一化后): {grid_sampler.shape} (预期：[bs, h, w, z, 3])")
        # print(f"grid_sampler_permuted (适配采样): {grid_sampler_permuted.shape} (预期：[bs, z, h, w, 3])")

        # -------------------------- 7. 历史BEV采样后打印 --------------------------
        # 采样历史BEV特征
        sampled_history_bev = F.grid_sample(
            self.history_bev.reshape(bs, mc, z, h, w),  # 输入：[bs, mc, z, h, w]
            grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4),
            align_corners=True,
            mode='bilinear'
        )
        sampled_history_visibility = F.grid_sample(
            self.history_visibility.reshape(bs, self.history_num, z, h, w).half(),  # 输入：[bs, mc, z, h, w]
            grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4).half(),
            align_corners=True,
            mode='nearest'
        )
        # print("\n7. 历史BEV采样后形状：")
        # print(f"history_bev.reshape: {self.history_bev.reshape(bs, mc, z, h, w).shape} (预期：[bs, mc, z, h, w])")
        # print(f"sampled_history_bev: {sampled_history_bev.shape} (预期：[bs, mc, z, h, w])")

        # -------------------------- 8. 可见性计算后打印 --------------------------
        # 计算当前帧可见性
        # V_curr = self.compute_visibility(
        #     grid_3d, 
        #     cam_intrins=curr_cam_intrins,
        #     cam_extrins=curr_cam_extrins,
        #     img_shape=self.img_shape,
        #     img_feats=img_feats,
        #     spatial_shapes=spatial_shapes
        # )

        # print("V_curr (当前可见性).  "*3)
        # V_curr = slots
        # 计算历史帧可见性
        # prev_cam_intrins = self.history_cam_intrins[:, -1]
        # prev_cam_extrins = self.history_cam_extrins[:, -1]
        # V_prev = self.compute_visibility(
        #     grid_3d,
        #     cam_intrins=prev_cam_intrins,
        #     cam_extrins=prev_cam_extrins,
        #     img_shape=self.img_shape,
        #     img_feats=img_feats,
        #     spatial_shapes=spatial_shapes
        # )

        V_prev = sampled_history_visibility #bs,4,z,h,w

        # print("\n8. 可见性计算后形状：")
        # print(f"V_curr (当前可见性): {V_curr.shape} (预期：[bs, h, w, z])")
        # print(f"V_prev (历史可见性): {V_prev.shape} (预期：[bs, h, w, z])")

        # -------------------------- 9. 稀疏采样前展平变量打印 --------------------------
        # 展平变量（用于稀疏采样）
        curr_bev_flat = curr_bev.reshape(bs, c_, -1)  # [bs, c_, N], N=h*w*z
        history_bev_flat = sampled_history_bev.reshape(bs, mc, -1)  # [bs, mc, N]
        V_prev_flat = V_prev.reshape(bs,self.history_num, -1)  # [bs, 4,N]
        V_curr_flat = V_curr.reshape(bs, 1,-1)  # [bs, 1,N]
        nonempty_prob_flat = nonempty_prob.reshape(bs, -1)  # [bs, N]
        # print("print(nonempty_prob_flat.shape)")
        # print(nonempty_prob_flat.shape)
        total_voxels = nonempty_prob_flat.shape[1]

        # print("\n9. 稀疏采样前展平变量形状：")
        # print(f"curr_bev_flat: {curr_bev_flat.shape} (预期：[bs, c_, N], N={total_voxels})")
        # print(f"history_bev_flat: {history_bev_flat.shape} (预期：[bs, mc, N])")
        # print(f"nonempty_prob_flat: {nonempty_prob_flat.shape} (预期：[bs, N])")
        # print(f"total_voxels (h*w*z): {total_voxels} (预期：{h*w*z})")

        # -------------------------- 10. 前景/背景索引及融合后打印（可选，确认后续维度） --------------------------
        # 生成前景/背景索引
        fg_indices = torch.topk(nonempty_prob_flat, self.top_k, dim=1)[1]
        bg_indices = torch.topk(1 - nonempty_prob_flat, total_voxels - self.top_k, dim=1)[1]
        # 提取前景特征（示例，其他融合步骤类似）
        fg_history_feat = torch.gather(history_bev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, mc, 1))

        # print("\n10. 前景/背景索引及特征提取后形状：")
        # print(f"fg_indices (前景索引): {fg_indices.shape} (预期：[bs, top_k])")
        # print(f"bg_indices (背景索引): {bg_indices.shape} (预期：[bs, N-top_k])")
        # print(f"fg_history_feat (前景历史特征): {fg_history_feat.shape} (预期：[bs, mc, top_k])")
        # print("="*50)

        # -------------------------- 后续原有逻辑（略，维度已通过打印确认） --------------------------
        # 8. 前景融合（原有代码）
        # 9. 背景融合（原有代码）
        # 10. 更新当前BEV特征（原有代码）
        # curr_bev_flat = curr_bev.reshape(bs, c_, -1)  # [bs, c_, N] N=h*w*z
        # history_bev_flat = sampled_history_bev.reshape(bs, mc, -1)  # [bs, mc, N]
        # V_prev_flat = V_prev.reshape(bs, -1)  # [bs, N]
        # V_curr_flat = V_curr.reshape(bs, -1)  # [bs, N]
        # nonempty_prob_flat = nonempty_prob.reshape(bs, -1)  # [bs, N]
        # total_voxels = nonempty_prob_flat.shape[1]

        # fg_indices = torch.topk(nonempty_prob_flat, self.top_k, dim=1)[1]  # [bs, top_k]
        # bg_indices = torch.topk(1 - nonempty_prob_flat, total_voxels - self.top_k, dim=1)[1]  # [bs, N-top_k]

        # # 前景特征提取
        # fg_history_feat = torch.gather(history_bev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, mc, 1))  # [bs, mc, top_k]
        fg_curr_feat = torch.gather(curr_bev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, c_, 1))  # [bs, c_, top_k]

        # 历史特征时间聚合
        # fg_history_feat_time = fg_history_feat.reshape(bs, self.history_num, c_, self.top_k)  # [bs, T, c_, K]
        #TODO 后续可以把time_weights也乘进去


        # time_weights = torch.exp(-0.5 * torch.arange(self.history_num, device=device)).view(1, self.history_num, 1, 1)

        fg_V_prev = torch.gather(V_prev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, self.history_num, 1))  # [bs, 4,K]
        fg_V_curr = torch.gather(V_curr_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, 1, 1))  # [bs, 1,K]
        # fg_time_vis_weights = fg_V_prev/(fg_V_prev.sum(dim=1).unsqueeze(1)+1e-10 ) # [bs, 4,K]
        #下面进行替换，不用显示提取权重，不计算 softmax
        # fg_time_vis_weights = (fg_V_prev*(self.fg_scale.view(1,self.history_num,1))).softmax(dim=1)
        # fg_time_vis_weights = (fg_V_prev*(self.fg_scale.view(1,self.history_num,1)))
        # print("print(last_occ_pred.shape)")
        # print(last_occ_pred.shape) #bs,x,y,z,num_classes ->bs,z,y,x,num_classes
        last_occ_pred = last_occ_pred.permute(0, 3, 2, 1, 4).reshape(bs, -1, last_occ_pred.shape[-1])  # [bs, N, num_classes]

        fg_occ_feat = torch.gather(last_occ_pred, dim=1, index=fg_indices.unsqueeze(-1).repeat(1, 1, last_occ_pred.shape[-1]))  # [bs, K, num_classes]

        fg_occ_embed = self.occ_embedding(fg_occ_feat).permute(0, 2, 1)  # [bs, occ_embedims, K]

        # print("print(fg_V_curr.shape)")
        # print(fg_V_curr.shape) # bs,1,K
        # print("print(fg_curr_feat.shape)")
        # print(fg_curr_feat.shape) #torch.Size([2, 96, 500])  bs,c_,K
        # print("print(fg_V_prev.shape)") 
        # print(fg_V_prev.shape) #torch.Size([2, 4, 500])  bs,4,K
        # print("print(fg_history_feat.shape)")
        # print(fg_history_feat.shape) #torch.Size([2, 384, 500])  bs,mc,K
        # print("print(fg_occ_embed.shape)")
        # print(fg_occ_embed.shape) #torch.Size([2, 32, 500])  bs,occ_embedims,K
        fg_fused = torch.cat([ fg_V_curr * fg_curr_feat,(fg_V_prev.unsqueeze(2) * fg_history_feat.view(bs, self.history_num, c_, self.top_k)).reshape(bs, self.history_num*c_, self.top_k) , fg_occ_embed], dim=1).permute(0, 2, 1)

        fg_fused = self.history_fusion_linear(fg_fused).permute(0, 2, 1)  # [bs, c_, K]


        bg_V_prev = torch.gather(V_prev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, self.history_num//2, 1))  # [bs, bg_k]
        bg_V_curr = torch.gather(V_curr_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, 1, 1))  # [bs, bg_k]
        bg_history_feat = torch.gather(history_bev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, mc//2, 1))  # [bs, mc//2, bg_k]
        bg_curr_feat = torch.gather(curr_bev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, c_, 1))  # [bs, c_, bg_k]
        bg_occ_feat = torch.gather(last_occ_pred, dim=1, index=bg_indices.unsqueeze(-1).repeat(1, 1, last_occ_pred.shape[-1]))  # [bs, bg_k, num_classes]
        bg_occ_embed = self.occ_embedding(bg_occ_feat).permute(0, 2, 1)  # [bs, occ_embedims, bg_k]

        # print(bg_V_curr.shape)
        # print("print(bg_V_curr.shape)")
        # print(bg_V_curr.shape)
        # print("print(bg_curr_feat.shape)")
        # print(bg_curr_feat.shape)
        # print("print(bg_V_prev.shape)")
        # print(bg_V_prev.shape)
        # print("print(bg_history_feat.shape)")
        # print(bg_history_feat.shape)
        # print("print(bg_occ_embed.shape)")
        # print(bg_occ_embed.shape)
        bg_fused = torch.cat([ bg_V_curr * bg_curr_feat,(bg_V_prev.unsqueeze(2) * bg_history_feat.view(bs, self.history_num//2, c_, total_voxels - self.top_k)).reshape(bs, self.history_num*c_//2, total_voxels - self.top_k), bg_occ_embed], dim=1).permute(0, 2, 1)
        # print("print(bg_fused.shape)")
        # print(bg_fused.shape)
        

        
        bg_fused = self.history_fusion_bg_linear(bg_fused).permute(0, 2, 1)  # [bs, c_, bg_k]

        # print()
        #TODO TODO 后续用senet实现一个通道注意力的版本  这个可能会更好？


        # # print("print(fg_history_feat_time.shape)")
        # # print(fg_history_feat_time.shape)
        # # print("print(fg_time_vis_weights.shape)")
        # # print(fg_time_vis_weights.shape)
        # # fg_history_agg = (fg_history_feat_time * fg_time_vis_weights.unsqueeze(2)).sum(dim=1)  # [bs, c_, K]).sum(dim=1)  # [bs, c_, K]]
        # fg_history_agg = (fg_history_feat_time * fg_time_vis_weights.unsqueeze(2)).sum(dim=1)  # [bs, c_, K]).sum(dim=1)  # [bs, c_, K]]


        # # print("print(fg_history_agg.shape)")
        # # print(fg_history_agg.shape)

        # # 可见性聚合与门控
        
        # # fg_V_prev_time = fg_V_prev.unsqueeze(1).repeat(1, self.history_num, 1)  # [bs, T, K]
        # # fg_V_prev_agg = (fg_V_prev_time * time_weights.squeeze(-1)).sum(dim=1)  # [bs, K]
        # fg_V_prev_agg = fg_V_prev.max(dim=1)[0]  # [bs, K]
        # fg_w_hist, fg_w_curr = self.compute_gate_weights(fg_V_prev_agg, fg_V_curr.squeeze(1))  # [bs, K, 1]

        # # 前景融合
        # fg_history_agg_perm = fg_history_agg.permute(0, 2, 1)  # [bs, K, c_]
        # fg_curr_perm = fg_curr_feat.permute(0, 2, 1)  # [bs, K, c_]
        # fg_fused = fg_w_hist * fg_history_agg_perm + fg_w_curr * fg_curr_perm  # [bs, K, c_]

        # # occupancy嵌入融合
        # last_occ_reshaped = last_occ_pred.permute(0, 3, 2, 1, 4).reshape(bs, -1, last_occ_pred.shape[-1])  # [bs, N, num_classes]
        # fg_occ_feat = torch.gather(last_occ_reshaped, dim=1, index=fg_indices.unsqueeze(-1).repeat(1, 1, last_occ_pred.shape[-1]))  # [bs, K, num_classes]
        # fg_occ_embed = self.occ_embedding(fg_occ_feat).permute(0, 2, 1)  # [bs, occ_embedims, K]
        # fg_fused = torch.cat([fg_fused, fg_occ_embed.permute(0, 2, 1)], dim=-1)  # [bs, K, c_+occ_embedims]
        # fg_fused = self.history_fusion_linear(fg_fused).permute(0, 2, 1)  # [bs, c_, K]

        # # 背景融合（原有代码）
        # bg_history_feat = torch.gather(history_bev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, mc, 1))  # [bs, mc//2, bg_k]
        # bg_curr_feat = torch.gather(curr_bev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, c_, 1))  # [bs, c_, bg_k]

        # bg_history_feat_time = bg_history_feat.reshape(bs, self.history_num, c_, -1)  # [bs, T, c_//2, bg_k]
        # # bg_history_agg = (bg_history_feat_time * time_weights).sum(dim=1)  # [bs, c_//2, bg_k]
        # bg_V_prev = torch.gather(V_prev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, self.history_num, 1))  # [bs, bg_k]
        # bg_V_curr = torch.gather(V_curr_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, 1, 1))  # [bs, bg_k]
        # #TODO 这个10的超参数？ 调整成可学习？
        # # bg_time_vis_weights = bg_V_prev/(bg_V_prev.sum(dim=1).unsqueeze(1) +1e-10) # [bs, 4,K]
        # bg_time_vis_weights =(bg_V_prev*(self.bg_scale.view(1,self.history_num,1))).softmax(dim=1)
        # bg_history_agg = (bg_history_feat_time * bg_time_vis_weights.unsqueeze(2)).sum(dim=1)  # [bs, c_, K]).sum(dim=1)  # [bs, c_, K]]



        # # bg_history_agg_perm = F.pad(bg_history_agg.permute(0, 2, 1), (0, c_ - c_//2, 0, 0))  # [bs, bg_k, c_]

        # bg_history_agg_perm = bg_history_agg.permute(0, 2, 1)  # [bs, bg_k, c_]
        # # bg_V_prev_time = bg_V_prev.unsqueeze(1).repeat(1, self.history_num, 1)  # [bs, T, bg_k]
        # # bg_V_prev_agg = (bg_V_prev_time * time_weights.squeeze(-1)).sum(dim=1)  # [bs, bg_k]
        # bg_V_prev_agg = bg_V_prev.max(dim=1)[0]  # [bs, bg_k]
        # bg_w_hist, bg_w_curr = self.compute_gate_weights(bg_V_prev_agg, bg_V_curr.squeeze(1))  # [bs, bg_k, 1]

        # bg_curr_perm = bg_curr_feat.permute(0, 2, 1)  # [bs, bg_k, c_]
        # # print("*"*50)
        # # print("print(bg_w_hist.shape)")
        # # print(bg_w_hist.shape)
        # # print("print(bg_w_curr.shape)")
        # # print(bg_w_curr.shape)
        # # print("print(bg_history_agg_perm.shape)")
        # # print(bg_history_agg_perm.shape)
        # # print("print(bg_curr_perm.shape)")        
        # # print(bg_curr_perm.shape)

        # # # 断言批次大小一致
        # # assert bg_w_hist.shape[0] == bg_history_agg_perm.shape[0] == bg_w_curr.shape[0] == bg_curr_perm.shape[0], \
        # #     f"批次大小不匹配: {bg_w_hist.shape[0]}, {bg_history_agg_perm.shape[0]}, {bg_w_curr.shape[0]}, {bg_curr_perm.shape[0]}"

        # # # 断言第二维度（bg_k）一致
        # # assert bg_w_hist.shape[1] == bg_history_agg_perm.shape[1] == bg_w_curr.shape[1] == bg_curr_perm.shape[1], \
        # #     f"bg_k维度不匹配: {bg_w_hist.shape[1]}, {bg_history_agg_perm.shape[1]}, {bg_w_curr.shape[1]}, {bg_curr_perm.shape[1]}"

        # # # 断言第三维度（c_）匹配（bg_w_hist和bg_w_curr的第三维为1，不影响广播）
        # # assert bg_history_agg_perm.shape[2] == bg_curr_perm.shape[2], \
        # #     f"特征维度c_不匹配: {bg_history_agg_perm.shape[2]} vs {bg_curr_perm.shape[2]}"

        # # print("bg_w_hist dtype:", bg_w_hist.dtype)
        # # print("bg_history_agg_perm dtype:", bg_history_agg_perm.dtype)
        # # print("bg_w_curr dtype:", bg_w_curr.dtype)
        # # print("bg_curr_perm dtype:", bg_curr_perm.dtype)


        # # print("bg_w_hist device:", bg_w_hist.device)
        # # print("bg_history_agg_perm device:", bg_history_agg_perm.device)
        # # print("bg_w_curr device:", bg_w_curr.device)
        # # print("bg_curr_perm device:", bg_curr_perm.device)


        # bg_fused = bg_w_hist * bg_history_agg_perm + bg_w_curr * bg_curr_perm  # [bs, bg_k, c_]
        # # 先验证乘法是否正常
        # # temp1 = bg_w_hist * bg_history_agg_perm
        # # temp2 = bg_w_curr * bg_curr_perm
        # # 再验证加法是否正常

        # # bg_w_hist = bg_w_hist.contiguous()
        # # bg_history_agg_perm = bg_history_agg_perm.contiguous()
        # # bg_w_curr = bg_w_curr.contiguous()
        # # bg_curr_perm = bg_curr_perm.contiguous()

        # # # 重新计算
        # # temp1 = bg_w_hist * bg_history_agg_perm
        # # temp2 = bg_w_curr * bg_curr_perm
        # # temp1 = temp1.contiguous()
        # # temp2 = temp2.contiguous()
        # # bg_fused = temp1 + temp2


        # # 转移所有张量到CPU
        # # bg_w_hist_cpu = bg_w_hist.cpu()
        # # bg_history_agg_perm_cpu = bg_history_agg_perm.cpu()
        # # bg_w_curr_cpu = bg_w_curr.cpu()
        # # bg_curr_perm_cpu = bg_curr_perm.cpu()

        # # # 分步执行运算
        # # try:
        # #     temp1_cpu = bg_w_hist_cpu * bg_history_agg_perm_cpu
        # #     temp2_cpu = bg_w_curr_cpu * bg_curr_perm_cpu
        # #     bg_fused_cpu = temp1_cpu + temp2_cpu
        # #     print(bg_fused_cpu)
        # #     print(bg_fused_cpu.shape)
        # #     print("CPU运算成功，无明显错误")
        # # except Exception as e:
        # #     print(f"CPU运算报错：{e}")  # 此处会显示具体错误原因（如值异常）


        # # bg_fused = temp1.clone() + temp2.clone()
        # # 1/0
        # bg_occ_feat = torch.gather(last_occ_reshaped, dim=1, index=bg_indices.unsqueeze(-1).repeat(1, 1, last_occ_pred.shape[-1]))  # [bs, bg_k, num_classes]
        # bg_occ_embed = self.occ_embedding(bg_occ_feat).permute(0, 2, 1)  # [bs, occ_embedims, bg_k]
        # bg_fused = torch.cat([bg_fused, bg_occ_embed.permute(0, 2, 1)], dim=-1)  # [bs, bg_k, c_+occ_embedims]
        # bg_fused = self.history_fusion_bg_linear(bg_fused).permute(0, 2, 1)  # [bs, c_, bg_k]

        # 更新当前BEV
        curr_bev_updated = curr_bev_flat.clone()
        curr_bev_updated.scatter_add_(dim=2, index=fg_indices.unsqueeze(1).repeat(1, c_, 1), src=fg_fused)
        curr_bev_updated.scatter_add_(dim=2, index=bg_indices.unsqueeze(1).repeat(1, c_, 1), src=bg_fused)
        curr_bev_updated = curr_bev_updated.reshape(bs, c_, z, h, w)  # 恢复原形状

        # 更新历史缓存
        self.history_last_bev = curr_bev_updated.detach().clone()
        self.history_bev = torch.cat([curr_bev,sampled_history_bev[:, :-c_, ...]], dim=1).detach()
        # self.history_cam_intrins = torch.cat([curr_cam_intrins.unsqueeze(1),self.history_cam_intrins[:, :-1, ...]], dim=1).detach()
        # self.history_cam_extrins = torch.cat([curr_cam_extrins.unsqueeze(1),self.history_cam_extrins[:, 1-1:, ...]], dim=1).detach()
        self.history_visibility =torch.cat([V_curr, V_prev[:, :-1, ...]],dim=1).detach()
        self.history_forward_augs = forward_augs.clone()

        return curr_bev_updated

    def generate_grid(self, curr_bev):
        n, c_, z, h, w = curr_bev.shape
        # Generate grid
        xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w, 1).expand(h, w, z)
        ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1, 1).expand(h, w, z)
        zs = torch.linspace(0, z - 1, z, dtype=curr_bev.dtype, device=curr_bev.device).view(1, 1, z).expand(h, w, z)
        grid = torch.stack((xs, ys, zs, torch.ones_like(xs)), -1).view(1, h, w, z, 4).expand(n, h, w, z, 4).view(n, h,w, z, 4, 1)
        return grid

    def generate_feat2bev(self, grid, dx, bx):
        feat2bev = torch.zeros((4, 4), dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = dx[0]
        feat2bev[1, 1] = dx[1]
        feat2bev[2, 2] = dx[2]
        feat2bev[0, 3] = bx[0] - dx[0] / 2.
        feat2bev[1, 3] = bx[1] - dx[1] / 2.
        feat2bev[2, 3] = bx[2] - dx[2] / 2.
        feat2bev[3, 3] = 1
        feat2bev = feat2bev.view(1, 4, 4)
        return feat2bev


@HEADS.register_module()
class GatedTemporalFusion6_SE(BaseModule):
    def __init__(
        self,
        history_num=4,
        depth_sampler_embed_dims=256,
        depth_sampler_num_heads=8,
        depth_sampler_num_levels=4,
        depth_sampler_num_points=4,
        im2col_step=64,
        top_k=500,
        single_bev_num_channels=96,
        occ_embedims=32,
        num_classes=18,
        vis_theta=0.28,
        vis_beta=10.0,
        vis_gamma=0.4,
        vis_sigma=0.1,
        nonempty_thresh=0.1,
        max_step_ratio=1.2,
        **kwargs
    ):
        super(GatedTemporalFusion6_SE, self).__init__()

        # 基础参数
        self.history_num = history_num
        self.top_k = top_k
        self.single_bev_num_channels = single_bev_num_channels
        self.occ_embedims = occ_embedims
        # self.fg_scale = nn.Parameter(torch.full((history_num,), 10.0))
        # self.bg_scale = nn.Parameter(torch.full((history_num,), 10.0))
        # 可见性门控参数
        # self.vis_theta = vis_theta  # 可见性阈值
        # self.vis_beta = vis_beta    # 敏感度参数
        # self.vis_gamma = vis_gamma  # 场景4固定权重
        # self.vis_sigma = vis_sigma  # 软化参数
        # self.vis_theta = nn.Parameter(torch.tensor(vis_theta))   # 可见性阈值
        # self.vis_beta = nn.Parameter(torch.tensor(vis_beta))     # 敏感度参数
        # self.vis_gamma = nn.Parameter(torch.tensor(vis_gamma))   # 场景4固定权重
        # self.vis_sigma = nn.Parameter(torch.tensor(vis_sigma))  

        
        # 历史特征缓存（新增history_last_bev）
        self.history_bev = None  # 缓存历史多帧BEV特征
        self.history_last_bev = None  # 缓存上一帧最终融合后的BEV特征（关键新增）
        self.history_cam_intrins = None
        self.history_cam_extrins = None
        
        # 网络层（保持不变）
        self.occ_embedding = nn.Sequential(
            nn.Linear(num_classes, occ_embedims),
            nn.Softplus(),
            nn.Linear(occ_embedims, occ_embedims),
        )
        self.history_fusion_linear = nn.Sequential(
            nn.Linear(single_bev_num_channels * (history_num + 1) + occ_embedims, single_bev_num_channels),
            nn.Softplus(),
            nn.Linear(single_bev_num_channels, single_bev_num_channels),
        )
        self.history_fusion_bg_linear = nn.Sequential(
            nn.Linear(single_bev_num_channels * (history_num//2 + 1) + occ_embedims, single_bev_num_channels),
            nn.Softplus(),
            nn.Linear(single_bev_num_channels, single_bev_num_channels),
        )

        
        # 可见性计算组件（保持不变）
        self.rt_vis_calculator = EfficientRayTracingVisibility(
            nonempty_thresh=nonempty_thresh,
            max_step_ratio=max_step_ratio
        )
        # self.img_shape = (900, 1600)  # 默认图像尺寸

        # self.depth_sampler = DeformableDepthSampler(
        #     embed_dims=depth_sampler_embed_dims,
        #     num_heads=depth_sampler_num_heads,
        #     num_levels=depth_sampler_num_levels,
        #     num_points=depth_sampler_num_points
        # )
        self.history_forward_augs = None  # 用于缓存历史帧的变换矩阵（如BDAM矩阵）
        self.im2col_step = im2col_step
        # self.voxel_encoder = nn.Linear(single_bev_num_channels, depth_sampler_embed_dims)
        self.dbound = [1.0, 45.0, 0.5]
        self.pc_range = [-40, -40, -1.0, 40, 40, 5.4]
        self.final_dim = (256, 704)
    def compute_visibility(self, grid, cam_intrins, cam_extrins, img_shape, img_feats, spatial_shapes):
        """
        升级：结合可变形注意力采样的深度值优化可见性计算
        Args:
            img_feats: 多尺度图像特征 [bs, num_levels, c, h, w]
            spatial_shapes: 图像特征尺度 [num_levels, 2] (h, w)
            其他参数同原函数
        Returns:
            vis_prob: [bs, h, w, z] 优化后的可见性概率
        """
        bs, h, w, z, _ = grid.shape
        h_img, w_img = img_shape
        device = grid.device
        num_voxels = h * w * z  # 体素总数

        # 1. 原有相机投影逻辑（计算图像坐标和初始可见性）
        # 1.1 体素坐标→相机坐标→图像坐标
        grid_cam = grid.unsqueeze(1).expand(bs, self.num_cams, h, w, z, 3)  # [bs, num_cams, h, w, z, 3]
        grid_flat = grid_cam.reshape(-1, num_voxels, 3)  # [bs*num_cams, N, 3]
        grid_hom = torch.cat([grid_flat, torch.ones_like(grid_flat[..., :1])], dim=-1)  # [bs*num_cams, N, 4]

        cam_intrins_flat = cam_intrins.reshape(-1, 4, 4)  # [bs*num_cams, 4, 4]
        cam_extrins_flat = cam_extrins.reshape(-1, 4, 4)  # [bs*num_cams, 4, 4]
        extrins_inv = torch.inverse(cam_extrins_flat)
        cam_coords = torch.bmm(extrins_inv[:, :3, :4], grid_hom.permute(0, 2, 1)).permute(0, 2, 1)  # [bs*num_cams, N, 3]
        depth = cam_coords[..., 2:3] + 1e-8  # 相机坐标系下的深度

        # 1.2 计算初始可见性（原逻辑）
        img_coords = torch.bmm(cam_intrins_flat[:, :3, :3], cam_coords.permute(0, 2, 1)).permute(0, 2, 1)  # [bs*num_cams, N, 3]
        img_xy = img_coords[..., :2] / depth  # [bs*num_cams, N, 2] (u, v)
        depth_valid = (cam_coords[..., 2] > 0).float()  # [bs*num_cams, N]
        u_valid = (img_xy[..., 0] >= 0) & (img_xy[..., 0] < w_img)
        v_valid = (img_xy[..., 1] >= 0) & (img_xy[..., 1] < h_img)
        img_valid = (u_valid & v_valid).float()  # [bs*num_cams, N]
        initial_vis = depth_valid * img_valid  # [bs*num_cams, N]

        # 2. 可变形注意力深度采样
        # 2.1 准备输入：体素特征编码
        # voxel_feat = self.voxel_encoder(self.curr_bev_feat)  # [bs, c_, z, h, w] → [bs, z*h*w, embed_dims]（需提前展平体素特征）
        # voxel_feat = voxel_feat.reshape(bs, num_voxels, -1)  # [bs, N, embed_dims]

        # 2.2 生成参考点（归一化到[0,1]）
        norm_img_xy = img_xy / torch.tensor([w_img, h_img], device=device).view(1, 1, 2)  # [bs*num_cams, N, 2]
        # 取主相机（如第0个相机）的参考点作为采样基准
        ref_points = norm_img_xy.reshape(bs, self.num_cams, num_voxels, 2)[:, 0]  # [bs, N, 2]
        ref_points = ref_points.unsqueeze(2).repeat(1, 1, self.depth_sampler.num_levels, 1)  # [bs, N, L, 2]

        # 2.3 采样深度特征
        sampled_depth = self.depth_sampler(
            query=voxel_feat,
            value=img_feats,  # 多尺度图像特征 [bs, L, c, h, w]
            reference_points=ref_points,
            spatial_shapes=spatial_shapes
        )  # [bs, N]

        # 3. 结合采样深度优化可见性
        # 3.1 深度一致性校验：采样深度与相机投影深度的差异
        cam_depth = depth.reshape(bs, self.num_cams, num_voxels)[:, 0]  # 主相机的投影深度 [bs, N]
        depth_diff = torch.abs(sampled_depth - cam_depth) / (cam_depth + 1e-8)  # 相对深度差
        depth_consistent = (depth_diff < 0.3).float()  # 深度差小于30%则认为有效

        # 3.2 融合可见性：初始可见性 × 深度一致性
        initial_vis = initial_vis.reshape(bs, self.num_cams, num_voxels).max(dim=1)[0]  # [bs, N]（多相机取max）
        vis_prob = initial_vis * depth_consistent  # [bs, N]

        # 4. 还原形状
        return vis_prob.reshape(bs, h, w, z)  # [bs, h, w, z]

    def compute_alpha_unified(self, V_curr, V_prev):
        """统一计算当前帧融合权重α（覆盖四场景）"""
        eps = 1e-8
        # 1. 基础动态权重σ_base（场景1-3）
        ratio = V_curr / (V_curr + V_prev + eps)
        sigma_base = torch.sigmoid(self.vis_beta * (ratio - 0.5))
        
        # 2. 场景4软化掩码σ_both
        mask_curr = torch.sigmoid(-(V_curr - self.vis_theta) / self.vis_sigma)
        mask_prev = torch.sigmoid(-(V_prev - self.vis_theta) / self.vis_sigma)
        sigma_both = mask_curr * mask_prev
        
        # 3. 最终权重计算
        alpha = (1 - sigma_both) * sigma_base + sigma_both * self.vis_gamma
        return alpha.unsqueeze(-1)  # [bs, N, 1]

    def compute_gate_weights(self, V_prev_agg, V_curr):
        """计算历史和当前帧的门控权重"""
        alpha = self.compute_alpha_unified(V_curr, V_prev_agg)
        return 1 - alpha, alpha  # 历史权重，当前权重

    def generate_grid(self, curr_bev, voxel_min, voxel_max, voxel_size):
        """生成体素中心坐标网格（自车坐标系）"""
        bs, c_, z, h, w = curr_bev.shape
        device = curr_bev.device
        
        # 计算体素中心坐标
        x_coords = torch.linspace(
            voxel_min[0] + voxel_size[0]/2, 
            voxel_max[0] - voxel_size[0]/2, 
            w, device=device
        )
        y_coords = torch.linspace(
            voxel_min[1] + voxel_size[1]/2, 
            voxel_max[1] - voxel_size[1]/2, 
            h, device=device
        )
        z_coords = torch.linspace(
            voxel_min[2] + voxel_size[2]/2, 
            voxel_max[2] - voxel_size[2]/2, 
            z, device=device
        )
        
        # 生成网格并扩展维度 [bs, h, w, z, 3]
        x_grid, y_grid, z_grid = torch.meshgrid(x_coords, y_coords, z_coords, indexing='xy')
        grid = torch.stack([x_grid, y_grid, z_grid], dim=-1)  # [w, h, z, 3]
        grid = grid.permute(1, 0, 2, 3)  # [h, w, z, 3]
        return grid.unsqueeze(0).repeat(bs, 1, 1, 1, 1)  # [bs, h, w, z, 3]


    def get_reference_points(self, H, W, Z=None, num_points_in_pillar =4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
        torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    @force_fp32(apply_to=('reference_points', 'cam_params'))
    def point_sampling(self, reference_points, pc_range, img_metas, cam_params=None):
        # prepare for point sampling
        lidar2img = []
        ego2lidar = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])     # lidar2img update the post aug in the loading pipeline
            ego2lidar.append(img_meta['ego2lidar'])
        lidar2img = torch.stack(lidar2img, dim=0).to(reference_points.device)
        ego2lidar = torch.stack(ego2lidar, dim=0).to(reference_points.device)

        sensor2egos, ego2globals, intrins, post_augs, bda_mat = cam_params
        num_cam = sensor2egos.size(1)
        ogfH, ogfW = self.final_dim

        # reference_points defines in the bev space, [bs, D, hxw, 3]
        # change reference_points from bev-ego coordinate to ego coordinate
        reference_points = reference_points.clone()
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
                                     (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
                                     (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
                                     (pc_range[5] - pc_range[2]) + pc_range[2]

        # prepare for point sampling
        reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
        reference_points = reference_points.permute(1, 0, 2, 3)  # shape: (num_points_in_pillar,bs,h*w,4)
        D, B, num_query = reference_points.size()[:3]  # D=num_points_in_pillar , num_query=h*w
        reference_points = reference_points.view(D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)  # shape: (num_points_in_pillar,bs,num_cam,h*w,4)
        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)
        ego2lidar = ego2lidar.view(1, B, 1, 1, 4, 4).repeat(D, 1, num_cam, num_query, 1, 1)
        inverse_bda = bda_mat.view(1, B, 1, 1, 4, 4).repeat(D, 1, num_cam, num_query, 1, 1)

        # change reference_points from ego coordinate to img coordinate
        eps = 1e-5
        reference_points_cam = (lidar2img @ ego2lidar @ inverse_bda @ reference_points).squeeze(-1)   # [num_points_in_pillar, bs, num_cam, num_query=h*w, 4]
        reference_points_depth = reference_points_cam[..., 2:3]
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(reference_points_depth, torch.ones_like(reference_points_depth) * eps)

        # Bug!!
        # Correct normalize is
        # reference_points_cam[..., 0] /= ogfW
        # reference_points_cam[..., 1] /= ogfH
        # But for reproducing our results, we use the following normalization
        reference_points_cam[..., 0] /= ogfW
        reference_points_cam[..., 1] /= ogfH

        bev_mask = (reference_points_depth > eps)
        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)                  # shape: (num_cam, bs,h*w, num_points_in_pillar, 2)
        reference_points_depth = reference_points_depth.permute(2, 1, 3, 0, 4)              # shape: (num_cam, bs,h*w, num_points_in_pillar, 1)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)                        # shape: (num_cam, bs,h*w, num_points_in_pillar)

        return reference_points_cam, reference_points_depth, bev_mask


    @force_fp32()
    def forward(self, curr_bev, cam_params, history_fusion_params, dx, bx, history_last_bev=None, last_occ_pred=None, nonempty_prob=None, img_feats=None, spatial_shapes=None,pred_img_depth=None,**kwargs):
        """
        Args:
            curr_bev: [bs, c, z, h, w] 当前帧BEV特征
            cam_params: 相机参数列表，包含外参、内参等
            history_fusion_params: 历史融合参数（包含序列信息等）
            dx: 体素尺寸 (x, y, z)
            bx: 体素偏移
            nonempty_prob: [bs, z, h, w] 体素非空概率
            last_occ_pred: [bs, z, h, w, num_classes] 上一时刻 occupancy 预测
            img_feats: 多尺度图像特征 [bs, num_levels, c, h, w]（新增，用于深度采样）
            spatial_shapes: 图像特征尺度 [num_levels, 2] (h, w)（新增，用于深度采样）
        Returns:
            curr_bev_updated: [bs, c, z, h, w] 融合后BEV特征
        """
        # print(self.history_num)
        # print("print(self.fg_scale)")
        # print(self.fg_scale) 
        # print("print(self.bg_scale)")
        # print(self.bg_scale)
        # # 可见性门控参数
        # # self.vis_theta = vis_theta  # 可见性阈值
        # # self.vis_beta = vis_beta    # 敏感度参数
        # # self.vis_gamma = vis_gamma  # 场景4固定权重
        # # self.vis_sigma = vis_sigma  # 软化参数
        # print("print(self.vis_theta # 可见性阈值)")
        # print(self.vis_theta )  # 可见性阈值
        # print("print(self.vis_beta) # 敏感度参数")
        # print(self.vis_beta)      # 敏感度参数
        # print("print(self.vis_gamma) # 场景4固定权重")
        # print(self.vis_gamma)  # 场景4固定权重
        # print("print(self.vis_sigma) # 软化参数")
        # print(self.vis_sigma) 


        # if torch.rand(1).item() < 1/2000:
        #     print(self.history_num)
        #     print("print(self.fg_scale)")
        #     print(self.fg_scale) 
        #     print("print(self.bg_scale)")
        #     print(self.bg_scale)
        #     # 可见性门控参数
        #     print("print(self.vis_theta # 可见性阈值)")
        #     print(self.vis_theta)  # 可见性阈值
        #     print("print(self.vis_beta) # 敏感度参数")
        #     print(self.vis_beta)      # 敏感度参数
        #     print("print(self.vis_gamma) # 场景4固定权重")
        #     print(self.vis_gamma)  # 场景4固定权重
        #     print("print(self.vis_sigma) # 软化参数")
        #     print(self.vis_sigma)  


        # -------------------------- 1. 解析参数后打印核心形状 --------------------------
        # 解析相机参数
        curr_cam_extrins = cam_params[0]  # [bs, num_cams, 4, 4]
        curr_cam_intrins = cam_params[2]  # [bs, num_cams, 4, 4]
        forward_augs = cam_params[4]      # [bs, 4, 4] 前向变换矩阵
        self.num_cams = curr_cam_extrins.shape[1]  # 从外参中获取相机数量
        bs, c_, z, h, w = curr_bev.shape
        device = curr_bev.device
        mc = self.history_num * c_        # 历史特征总通道数
        # self.history_forward_augs = forward_augs.clone()


        ref_3d = self.get_reference_points(
            h, w, z, z, dim='3d', bs=bs, device=device, dtype=curr_bev.dtype) # torch.Size([3, 2, 625, 3]) #[bs,z,yx,3(x,y,z)]
        # ref_2d = self.get_reference_points(
        #     h, w, dim='2d', bs=bs, device=device, dtype=curr_bev.dtype) #torch.Size([3, 625, 1, 2])
        slots = torch.zeros(list([ref_3d.shape[0],ref_3d.shape[2],ref_3d.shape[1]])).to(ref_3d)
        reference_points_cam, reference_points_depth, bev_mask = self.point_sampling(ref_3d, self.pc_range, img_metas=kwargs['img_metas'], cam_params=cam_params)
        indexes = [[] for _ in range(bs)]
        spatial_shapes =[]
        spatial_shapes.append([16, 44])
        spatial_shapes = torch.tensor(spatial_shapes).to(device)
        pred_img_depth = pred_img_depth.view(bs * 6, -1, spatial_shapes[0][0], spatial_shapes[0][1])
        pred_img_depth = pred_img_depth.flatten(2).permute(0, 2, 1)  
        max_len = 0
        for j in range(bs):
            for i, mask_per_img in enumerate(bev_mask):
                index_query_per_img = mask_per_img[j].sum(-1).nonzero().squeeze(-1)
                if len(index_query_per_img) == 0:
                    index_query_per_img = bev_mask[i][j].sum(-1).nonzero().squeeze(-1)[0:1]
                indexes[j].append(index_query_per_img)
                # for batch operation, we need to pad the indexes to the same length
                max_len = max(max_len, len(index_query_per_img))
        reference_points_cam_rebatch = reference_points_cam.new_zeros([bs, self.num_cams, max_len, z, 2])
        reference_points_depth_rebatch = reference_points_depth.new_zeros([bs, self.num_cams, max_len, z, 1])

        for j in range(bs):
            for i, (reference_points_per_img, reference_points_depth_per_img) in enumerate(zip(reference_points_cam, reference_points_depth)):
                index_query_per_img = indexes[j][i]
                reference_points_cam_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]
                reference_points_depth_rebatch[j, i, :len(index_query_per_img)] = reference_points_depth_per_img[j, index_query_per_img]

        #use deformble attn
        depth_reference_points = reference_points_cam_rebatch.reshape(bs*6, max_len*z, 1, 1, 1, 2).contiguous()
        depth_attention_weights = torch.ones_like(depth_reference_points[..., 0]).contiguous()
        pred_img_depth = pred_img_depth.unsqueeze(2).contiguous()
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))



        bev_query_depth_rebatch = (reference_points_depth_rebatch- self.dbound[0])/ self.dbound[2]
        bev_query_depth_rebatch = torch.clip(torch.floor(bev_query_depth_rebatch), 0, 88-1).to(torch.long)
        bev_query_depth_rebatch = F.one_hot(bev_query_depth_rebatch.squeeze(-1),
                                   num_classes=88)

        depth_output = MultiScaleDeformableAttnFunction_fp32.apply(pred_img_depth, spatial_shapes,level_start_index,depth_reference_points,depth_attention_weights, self.im2col_step)
        depth_output = depth_output.reshape(bs,6, max_len,z, -1)   # [bs*num_cam, num_query, num_Z_anchors, C]
        # reference_points_depth_rebatch

        increment = torch.zeros_like(depth_output)
        # increment[..., 0] = 1e-9  # 非原地赋值（创建新张量）
        # depth_output = depth_output + increment
        depth_output = depth_output + torch.cat([(torch.zeros_like(depth_output[...,:1]) + 1e-9),torch.zeros_like(depth_output[...,1:])],dim=-1)

        # print("depth_output.sum(-1).shape")
        # print(depth_output.sum(-1).shape)
        # print("depth_output.sum(-1)")
        # print(depth_output.sum(-1))
        # print("print((depth_output.sum(-1)==0).sum())")
        # print((depth_output.sum(-1)==0).sum())
        depth_output =depth_output/depth_output.sum(-1)[...,None] #bs,xy,z,D

        # print("depth_output.sum(-1).shape")
        # print(depth_output.sum(-1).shape)
        # print("depth_output.sum(-1)")
        # print(depth_output.sum(-1))
        # print("print((depth_output.sum(-1)==1).sum())")
        # print((depth_output.sum(-1)>=0.99).sum())
        # print("depth_output")
        # print(depth_output)


        depth_output = (1-depth_output.cumsum(dim=-1))
        depth_output =torch.cat([torch.ones_like(depth_output[...,0:1]),depth_output[...,0:-1]],dim=-1)

        # print("print((depth_output[...,-1]==0).sum())")
        # print((depth_output[...,-1]==0).sum())
        depth_output = (bev_query_depth_rebatch*depth_output).sum(-1)

        #恢复depth_output的shape
        # fix_depth_output = depth_output.new_zeros([bs, 6, w*h, z, 88])
        # for j in range(bs):
        #     for i in range(6):
        #         index_query_per_img = indexes[j][i]
        #         fix_depth_output[j, i, index_query_per_img] = depth_output[j, i, :len(index_query_per_img)]

        for j in range(bs):
            for i in range(6):
                index_query_per_img = indexes[j][i]
                slots[j, index_query_per_img] = torch.max(slots[j, index_query_per_img],depth_output[j, i, :len(index_query_per_img)])

        # depth_sum = fix_depth_output.sum(dim=-1).view(3,6,25,25,2)
        # bs = depth_sum.shape[0]       # 3
        # num_cam = depth_sum.shape[1]  # 6
        # z_layers = depth_sum.shape[4] # 2
        # H, W = depth_sum.shape[2], depth_sum.shape[3]  # 25,25

        # # 颜色映射：0值用黑色，非0值用渐变色
        # cmap = plt.cm.viridis
        # cmap.set_bad(color='black')  # 0值标记为黑色

        # # 设置子图布局（不变）
        # fig, axes = plt.subplots(
        #     nrows=bs, ncols=num_cam * z_layers,
        #     figsize=(30, 8),
        #     squeeze=False
        # )
        # fig.suptitle('fix_depth_output.sum(-1) 可视化（黑色=0值，颜色=非0值）', fontsize=16, y=0.98)


        # # -------------------------- 2. 循环绘制每个子图（核心修改：添加 .detach()） --------------------------
        # for batch_idx in range(bs):
        #     for cam_idx in range(num_cam):
        #         for z_idx in range(z_layers):
        #             col_idx = cam_idx * z_layers + z_idx
        #             ax = axes[batch_idx, col_idx]
                    
        #             # 核心修改：先 detach() 切断计算图，再转 cpu 和 numpy
        #             data = depth_sum[batch_idx, cam_idx, :, :, z_idx].detach().cpu().numpy()
        #             data[data == 0] = np.nan  # 0值替换为NaN，显示为黑色
                    
        #             # 绘制热力图（修改 vmin/vmax：同样添加 .detach()）
        #             im = ax.imshow(
        #                 data, 
        #                 cmap=cmap, 
        #                 aspect='auto',
        #                 # 关键修改：depth_sum 先 detach 再转 numpy，确保不影响梯度
        #                 vmin=np.nanmin(depth_sum.detach().cpu().numpy()),
        #                 vmax=np.nanmax(depth_sum.detach().cpu().numpy())
        #             )
                    
        #             # 子图标题和坐标轴（不变）
        #             ax.set_title(
        #                 f'Batch{batch_idx+1}\nCam{cam_idx+1} Z{z_idx+1}',
        #                 fontsize=10, pad=5
        #             )
        #             ax.set_xticks([])
        #             ax.set_yticks([])


        # # -------------------------- 3. 添加颜色条（不变） --------------------------
        # cbar = fig.colorbar(
        #     im, 
        #     ax=axes.ravel().tolist(),
        #     shrink=0.8,
        #     pad=0.02
        # )
        # cbar.set_label('Sum of Depth Bins (D=88)', fontsize=12)


        # # -------------------------- 4. 调整布局并保存（不变） --------------------------
        # plt.tight_layout(rect=[0, 0, 0.98, 0.95])
        # plt.savefig('depth_sum_visualization.png', dpi=300, bbox_inches='tight')
        # plt.show()

        # output

        #计数更新，建立在纸上的假设成立的基础上
        # count = bev_mask.sum(-1) > 0
        # count = count.permute(1, 2, 0).sum(-1)
        # count = torch.clamp(count, min=1.0)
        # slots = slots / count[..., None]

        # print("slots.shape")
        # print(slots.shape)
        # print("slots")  
        # print(slots)
        # zzzzz=1/0

        V_curr =slots.view(bs, 1, h, w,z).permute(0, 1, 4, 2, 3)
        # slots[...,0]+=1e-9
        # print("slots.sum(-1).shape")
        # print(slots.sum(-1).shape)
        # print("slots.sum(-1)")
        # print(slots.sum(-1))
        # print("print((slots.sum(-1)==0).sum())")
        # print((slots.sum(-1)==0).sum())
        # slots =slots/slots.sum(-1)[...,None] #bs,xy,z,D

        # print("slots.sum(-1).shape")
        # print(slots.sum(-1).shape)
        # print("slots.sum(-1)")
        # print(slots.sum(-1))
        # print("print((slots.sum(-1)==1).sum())")
        # print("slots")
        # print(slots)


        # slots = (1-slots.cumsum(dim=-1))

        # print("print((slots[...,-1]==0).sum())")
        # print((slots[...,-1]==0).sum())

        # print("slots")
        # print(slots)
        # print("print((slots[...,-1]<0.01).sum())")
        # print((slots[...,-1]<0.01).sum())

        #TODO 这里的对于边界值的考虑，从0开始还是从1开始？




        # slots = self.output_proj(slots)
        #TODO 上面这里需要检查一下
        #这里相当于两次softmax，可能会导致分布变得不够尖锐，需要进一步确认

        #到这里slots就是可见性的概率分布了
        #这里先尝试使用期望值进行计算，使得可微分

        #然后再采用stc的原始离散计算方法，


        # 打印解析后关键变量形状
        # print("="*50)
        # print("1. 解析参数后核心变量形状：")
        # print(f"curr_bev: {curr_bev.shape} (预期：[bs, c, z, h, w])")
        # print(f"curr_cam_extrins: {curr_cam_extrins.shape} (预期：[bs, num_cams, 4, 4])")
        # print(f"curr_cam_intrins: {curr_cam_intrins.shape} (预期：[bs, num_cams, 4, 4])")
        # print(f"forward_augs: {forward_augs.shape} (预期：[bs, 4, 4])")
        # print(f"dx: {dx.shape if hasattr(dx, 'shape') else type(dx)} (预期：[3])")
        # print(f"bx: {bx.shape if hasattr(bx, 'shape') else type(bx)} (预期：[3])")
        # print(f"bs: {bs}, c_: {c_}, z: {z}, h: {h}, w: {w} (BEV特征维度)")
        # print("="*50)

        if type(history_fusion_params['sequence_group_idx']) is list:
            seq_ids = history_fusion_params['sequence_group_idx'][0]
        else:
            seq_ids = history_fusion_params['sequence_group_idx']
        if type(history_fusion_params['start_of_sequence']) is list:
            start_of_sequence = history_fusion_params['start_of_sequence'][0]
        else:
            start_of_sequence = history_fusion_params['start_of_sequence']
        if type(history_fusion_params['curr_to_prev_ego_rt']) is list:
            curr_to_prev_ego_rt = history_fusion_params['curr_to_prev_ego_rt'][0]
        else:
            curr_to_prev_ego_rt = history_fusion_params['curr_to_prev_ego_rt']
        forward_augs = cam_params[-1]  # bda

        # check seq_ids > 0
        assert (seq_ids >= 0).all()
        # -------------------------- 2. 初始化历史缓存后打印 --------------------------
        if self.history_bev is None:
            # self.history_bev = curr_bev.repeat(1, self.history_num, 1, 1, 1)  # [bs, mc, z, h, w]
            # self.history_cam_intrins = curr_cam_intrins.unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)  # [bs, T, num_cams, 4, 4]
            # self.history_cam_extrins = curr_cam_extrins.unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)  # [bs, T, num_cams, 4, 4]
            # self.history_bev = curr_bev.clone()
            self.history_forward_augs = forward_augs.clone()
            self.history_seq_ids = seq_ids.clone()
            self.history_bev = curr_bev.repeat(1, self.history_num, 1, 1, 1)
            self.history_sweep_time = curr_bev.new_zeros(curr_bev.shape[0], self.history_num)
            self.history_visibility = V_curr.repeat(1, self.history_num, 1, 1, 1).half()
        self.history_bev = self.history_bev.detach()
        self.history_visibility = self.history_visibility.detach().half()
        self.history_sweep_time += 1

        # 打印历史缓存形状
        # print("\n2. 历史缓存初始化后形状：")
        # print(f"history_bev: {self.history_bev.shape} (预期：[bs, mc, z, h, w]，mc={mc})")
        # print(f"history_cam_intrins: {self.history_cam_intrins.shape} (预期：[bs, history_num, num_cams, 4, 4])")
        # print(f"history_cam_extrins: {self.history_cam_extrins.shape} (预期：[bs, history_num, num_cams, 4, 4])")

        # -------------------------- 3. 生成网格和BEV变换矩阵后打印 --------------------------
        # 处理新序列（略，不影响维度）
        # start_of_sequence = history_fusion_params.get('start_of_sequence', torch.zeros(bs, dtype=torch.bool, device=device))
        if start_of_sequence.sum()>0:
            self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_num, 1, 1, 1)
            # self.history_cam_intrins[start_of_sequence] = curr_cam_intrins[start_of_sequence].unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)
            # self.history_cam_extrins[start_of_sequence] = curr_cam_extrins[start_of_sequence].unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)
            self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]
            self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
            self.history_sweep_time[start_of_sequence] = 0  # zero the new sequence timestep starts
            self.history_visibility[start_of_sequence] = V_curr[start_of_sequence].repeat(1, self.history_num, 1, 1, 1).half()

        # 生成体素网格和BEV变换矩阵
        grid = self.generate_grid(curr_bev) #[bs,y,x,z,4]
        grid_3d = grid
        feat2bev = self.generate_feat2bev(grid, dx, bx)

        # 打印网格和变换矩阵形状
        # print("\n3. 生成网格和BEV变换矩阵后形状：")
        # print(f"grid_3d (体素网格): {grid_3d.shape} (关键！预期：[bs, h, w, z, 3] 或 [bs, w, h, z, 3])")
        # print(f"feat2bev (BEV变换矩阵): {feat2bev.shape} (预期：[bs, 4, 4])")

        # -------------------------- 4. 运动补偿矩阵计算后打印 --------------------------
        # 获取帧间姿态变换
        # curr_to_prev_ego_rt = history_fusion_params.get('curr_to_prev_ego_rt', torch.eye(4, device=device).unsqueeze(0).repeat(bs, 1, 1))
        # 计算RT流（坐标变换矩阵）
        rt_flow = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_ego_rt @ torch.inverse(forward_augs) @ feat2bev)
        # 生成齐次网格
        # 在forward函数中，生成grid_hom的位置修正：
        # grid_3d = self.generate_grid(curr_bev)  # 现在形状：[3, 25, 25, 2, 3]（bs, h, w, z, 3）
        # # 生成齐次坐标（x,y,z,1），并添加最后一个维度（用于矩阵乘法）
        # grid_hom = torch.cat([
        #     grid_3d,  # [3,25,25,2,3]
        #     torch.ones_like(grid_3d[..., :1])  # [3,25,25,2,1]（补充1作为齐次坐标）
        # ], dim=-1).unsqueeze(-1)  # 最终形状：[3,25,25,2,4,1]（符合预期）
        # # 打印运动补偿相关形状（矩阵乘法前关键检查）
        # print("\n4. 运动补偿矩阵计算后形状（矩阵乘法前）：")
        # print(f"curr_to_prev_ego_rt (帧间姿态): {curr_to_prev_ego_rt.shape} (预期：[bs, 4, 4])")
        # print(f"rt_flow (变换流): {rt_flow.shape} (预期：[bs, 4, 4])")
        # print(f"grid_hom (齐次网格): {grid_hom.shape} (关键！预期：[bs, h, w, z, 4, 1]，需与rt_flow广播匹配)")
        # print(f"rt_flow.view后: {rt_flow.view(bs, 1, 1, 1, 4, 4).shape} (预期：[bs, 1, 1, 1, 4, 4])")

        # # -------------------------- 5. 网格变换后打印（解决之前维度错的核心） --------------------------
        # try:
        #     grid_transformed = rt_flow.view(bs, 1, 1, 1, 4, 4) @ grid_hom  # 矩阵乘法：[bs, h, w, z, 4, 1]
        #     print("\n5. 网格变换后形状（矩阵乘法成功！）：")
        #     print(f"grid_transformed: {grid_transformed.shape} (预期：[bs, h, w, z, 4, 1])")
        # except RuntimeError as e:
        #     print(f"\n5. 网格变换矩阵乘法报错！错误信息：{str(e)}")
        #     print(f"  - rt_flow.view形状: {rt_flow.view(bs, 1, 1, 1, 4, 4).shape}")
        #     print(f"  - grid_hom形状: {grid_hom.shape}")
        #     print("  提示：需确保grid_hom的第1-4维度与rt_flow.view的第2-5维度匹配（广播规则）")
        #     raise e  # 继续抛出错误，方便定位
        bs, mc, z, h, w = self.history_bev.shape
        n, c_, z, h, w = curr_bev.shape
        grid = rt_flow.view(n, 1, 1, 1, 4, 4) @ grid
        # -------------------------- 6. 采样网格生成后打印 --------------------------
        # 生成采样网格（归一化到[-1,1]，适配F.grid_sample）
        normalize_factor = torch.tensor([w - 1.0, h - 1.0, z - 1.0], dtype=curr_bev.dtype, device=device)
        # grid_sampler = grid_transformed[..., :3, 0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0  # [bs, h, w, z, 3]
        # # 调整采样网格维度（适配F.grid_sample输入：[bs, z, h, w, 3]）
        # grid_sampler_permuted = grid_sampler.permute(0, 3, 1, 2, 4)  # 交换z和h/w维度
        grid = grid[:, :, :, :, :3, 0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0   # grid order is x, y, z


        # print("\n6. 采样网格生成后形状：")
        # print(f"grid_sampler (归一化后): {grid_sampler.shape} (预期：[bs, h, w, z, 3])")
        # print(f"grid_sampler_permuted (适配采样): {grid_sampler_permuted.shape} (预期：[bs, z, h, w, 3])")

        # -------------------------- 7. 历史BEV采样后打印 --------------------------
        # 采样历史BEV特征
        sampled_history_bev = F.grid_sample(
            self.history_bev.reshape(bs, mc, z, h, w),  # 输入：[bs, mc, z, h, w]
            grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4),
            align_corners=True,
            mode='bilinear'
        )
        sampled_history_visibility = F.grid_sample(
            self.history_visibility.reshape(bs, self.history_num, z, h, w).half(),  # 输入：[bs, mc, z, h, w]
            grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4).half(),
            align_corners=True,
            mode='nearest'
        )
        # print("\n7. 历史BEV采样后形状：")
        # print(f"history_bev.reshape: {self.history_bev.reshape(bs, mc, z, h, w).shape} (预期：[bs, mc, z, h, w])")
        # print(f"sampled_history_bev: {sampled_history_bev.shape} (预期：[bs, mc, z, h, w])")

        # -------------------------- 8. 可见性计算后打印 --------------------------
        # 计算当前帧可见性
        # V_curr = self.compute_visibility(
        #     grid_3d, 
        #     cam_intrins=curr_cam_intrins,
        #     cam_extrins=curr_cam_extrins,
        #     img_shape=self.img_shape,
        #     img_feats=img_feats,
        #     spatial_shapes=spatial_shapes
        # )

        # print("V_curr (当前可见性).  "*3)
        # V_curr = slots
        # 计算历史帧可见性
        # prev_cam_intrins = self.history_cam_intrins[:, -1]
        # prev_cam_extrins = self.history_cam_extrins[:, -1]
        # V_prev = self.compute_visibility(
        #     grid_3d,
        #     cam_intrins=prev_cam_intrins,
        #     cam_extrins=prev_cam_extrins,
        #     img_shape=self.img_shape,
        #     img_feats=img_feats,
        #     spatial_shapes=spatial_shapes
        # )

        V_prev = sampled_history_visibility #bs,4,z,h,w

        # print("\n8. 可见性计算后形状：")
        # print(f"V_curr (当前可见性): {V_curr.shape} (预期：[bs, h, w, z])")
        # print(f"V_prev (历史可见性): {V_prev.shape} (预期：[bs, h, w, z])")

        # -------------------------- 9. 稀疏采样前展平变量打印 --------------------------
        # 展平变量（用于稀疏采样）
        curr_bev_flat = curr_bev.reshape(bs, c_, -1)  # [bs, c_, N], N=h*w*z
        history_bev_flat = sampled_history_bev.reshape(bs, mc, -1)  # [bs, mc, N]
        V_prev_flat = V_prev.reshape(bs,self.history_num, -1)  # [bs, 4,N]
        V_curr_flat = V_curr.reshape(bs, 1,-1)  # [bs, 1,N]
        nonempty_prob_flat = nonempty_prob.reshape(bs, -1)  # [bs, N]
        # print("print(nonempty_prob_flat.shape)")
        # print(nonempty_prob_flat.shape)
        total_voxels = nonempty_prob_flat.shape[1]

        # print("\n9. 稀疏采样前展平变量形状：")
        # print(f"curr_bev_flat: {curr_bev_flat.shape} (预期：[bs, c_, N], N={total_voxels})")
        # print(f"history_bev_flat: {history_bev_flat.shape} (预期：[bs, mc, N])")
        # print(f"nonempty_prob_flat: {nonempty_prob_flat.shape} (预期：[bs, N])")
        # print(f"total_voxels (h*w*z): {total_voxels} (预期：{h*w*z})")

        # -------------------------- 10. 前景/背景索引及融合后打印（可选，确认后续维度） --------------------------
        # 生成前景/背景索引
        fg_indices = torch.topk(nonempty_prob_flat, self.top_k, dim=1)[1]
        bg_indices = torch.topk(1 - nonempty_prob_flat, total_voxels - self.top_k, dim=1)[1]
        # 提取前景特征（示例，其他融合步骤类似）
        fg_history_feat = torch.gather(history_bev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, mc, 1))

        # print("\n10. 前景/背景索引及特征提取后形状：")
        # print(f"fg_indices (前景索引): {fg_indices.shape} (预期：[bs, top_k])")
        # print(f"bg_indices (背景索引): {bg_indices.shape} (预期：[bs, N-top_k])")
        # print(f"fg_history_feat (前景历史特征): {fg_history_feat.shape} (预期：[bs, mc, top_k])")
        # print("="*50)

        # -------------------------- 后续原有逻辑（略，维度已通过打印确认） --------------------------
        # 8. 前景融合（原有代码）
        # 9. 背景融合（原有代码）
        # 10. 更新当前BEV特征（原有代码）
        # curr_bev_flat = curr_bev.reshape(bs, c_, -1)  # [bs, c_, N] N=h*w*z
        # history_bev_flat = sampled_history_bev.reshape(bs, mc, -1)  # [bs, mc, N]
        # V_prev_flat = V_prev.reshape(bs, -1)  # [bs, N]
        # V_curr_flat = V_curr.reshape(bs, -1)  # [bs, N]
        # nonempty_prob_flat = nonempty_prob.reshape(bs, -1)  # [bs, N]
        # total_voxels = nonempty_prob_flat.shape[1]

        # fg_indices = torch.topk(nonempty_prob_flat, self.top_k, dim=1)[1]  # [bs, top_k]
        # bg_indices = torch.topk(1 - nonempty_prob_flat, total_voxels - self.top_k, dim=1)[1]  # [bs, N-top_k]

        # # 前景特征提取
        # fg_history_feat = torch.gather(history_bev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, mc, 1))  # [bs, mc, top_k]
        fg_curr_feat = torch.gather(curr_bev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, c_, 1))  # [bs, c_, top_k]

        # 历史特征时间聚合
        # fg_history_feat_time = fg_history_feat.reshape(bs, self.history_num, c_, self.top_k)  # [bs, T, c_, K]
        #TODO 后续可以把time_weights也乘进去


        # time_weights = torch.exp(-0.5 * torch.arange(self.history_num, device=device)).view(1, self.history_num, 1, 1)

        fg_V_prev = torch.gather(V_prev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, self.history_num, 1))  # [bs, 4,K]
        fg_V_curr = torch.gather(V_curr_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, 1, 1))  # [bs, 1,K]
        # fg_time_vis_weights = fg_V_prev/(fg_V_prev.sum(dim=1).unsqueeze(1)+1e-10 ) # [bs, 4,K]
        #下面进行替换，不用显示提取权重，不计算 softmax
        # fg_time_vis_weights = (fg_V_prev*(self.fg_scale.view(1,self.history_num,1))).softmax(dim=1)
        # fg_time_vis_weights = (fg_V_prev*(self.fg_scale.view(1,self.history_num,1)))
        print("print(last_occ_pred.shape)")
        print(last_occ_pred.shape)
        last_occ_pred = last_occ_pred.permute(0, 3, 2, 1, 4).reshape(bs, -1, last_occ_pred.shape[-1])  # [bs, N, num_classes]

        fg_occ_feat = torch.gather(last_occ_pred, dim=1, index=fg_indices.unsqueeze(-1).repeat(1, 1, last_occ_pred.shape[-1]))  # [bs, K, num_classes]

        fg_occ_embed = self.occ_embedding(fg_occ_feat).permute(0, 2, 1)  # [bs, occ_embedims, K]

        print("print(fg_V_curr.shape)")
        print(fg_V_curr.shape)
        print("print(fg_curr_feat.shape)")
        print(fg_curr_feat.shape)
        print("print(fg_V_prev.shape)")
        print(fg_V_prev.shape)
        print("print(fg_history_feat.shape)")
        print(fg_history_feat.shape)
        print("print(fg_occ_embed.shape)")
        print(fg_occ_embed.shape)
        fg_fused = torch.cat([ (fg_V_curr * fg_curr_feat.permute(0, 2, 1)).permute(0, 2, 1),(fg_V_prev.unsqueeze(2) * fg_history_feat.view(bs, self.history_num, c_, self.top_k)).reshape(bs, self.history_num*c_, self.top_k).permute(0, 2, 1) , fg_occ_embed.permute(0, 2, 1)], dim=-1)

        fg_fused = self.history_fusion_linear(fg_fused).permute(0, 2, 1)  # [bs, c_, K]


        bg_V_prev = torch.gather(V_prev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, self.history_num, 1))  # [bs, bg_k]
        bg_V_curr = torch.gather(V_curr_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, 1, 1))  # [bs, bg_k]
        bg_history_feat = torch.gather(history_bev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, mc, 1))  # [bs, mc//2, bg_k]
        bg_curr_feat = torch.gather(curr_bev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, c_, 1))  # [bs, c_, bg_k]
        bg_occ_feat = torch.gather(last_occ_reshaped, dim=1, index=bg_indices.unsqueeze(-1).repeat(1, 1, last_occ_pred.shape[-1]))  # [bs, bg_k, num_classes]
        bg_occ_embed = self.occ_embedding(bg_occ_feat).permute(0, 2, 1)  # [bs, occ_embedims, bg_k]

        # print(bg_V_curr.shape)
        # print
        bg_fused = torch.cat([ (bg_V_curr * bg_curr_feat.permute(0, 2, 1)).permute(0, 2, 1),(bg_V_prev.unsqueeze(2) * bg_history_feat.view(bs, self.history_num, c_, self.top_k)).reshape(bs, self.history_num*c_, self.top_k).permute(0, 2, 1) , bg_occ_embed.permute(0, 2, 1)], dim=-1)


        #TODO TODO 后续用senet实现一个通道注意力的版本  这个可能会更好？


        # # print("print(fg_history_feat_time.shape)")
        # # print(fg_history_feat_time.shape)
        # # print("print(fg_time_vis_weights.shape)")
        # # print(fg_time_vis_weights.shape)
        # # fg_history_agg = (fg_history_feat_time * fg_time_vis_weights.unsqueeze(2)).sum(dim=1)  # [bs, c_, K]).sum(dim=1)  # [bs, c_, K]]
        # fg_history_agg = (fg_history_feat_time * fg_time_vis_weights.unsqueeze(2)).sum(dim=1)  # [bs, c_, K]).sum(dim=1)  # [bs, c_, K]]


        # # print("print(fg_history_agg.shape)")
        # # print(fg_history_agg.shape)

        # # 可见性聚合与门控
        
        # # fg_V_prev_time = fg_V_prev.unsqueeze(1).repeat(1, self.history_num, 1)  # [bs, T, K]
        # # fg_V_prev_agg = (fg_V_prev_time * time_weights.squeeze(-1)).sum(dim=1)  # [bs, K]
        # fg_V_prev_agg = fg_V_prev.max(dim=1)[0]  # [bs, K]
        # fg_w_hist, fg_w_curr = self.compute_gate_weights(fg_V_prev_agg, fg_V_curr.squeeze(1))  # [bs, K, 1]

        # # 前景融合
        # fg_history_agg_perm = fg_history_agg.permute(0, 2, 1)  # [bs, K, c_]
        # fg_curr_perm = fg_curr_feat.permute(0, 2, 1)  # [bs, K, c_]
        # fg_fused = fg_w_hist * fg_history_agg_perm + fg_w_curr * fg_curr_perm  # [bs, K, c_]

        # # occupancy嵌入融合
        # last_occ_reshaped = last_occ_pred.permute(0, 3, 2, 1, 4).reshape(bs, -1, last_occ_pred.shape[-1])  # [bs, N, num_classes]
        # fg_occ_feat = torch.gather(last_occ_reshaped, dim=1, index=fg_indices.unsqueeze(-1).repeat(1, 1, last_occ_pred.shape[-1]))  # [bs, K, num_classes]
        # fg_occ_embed = self.occ_embedding(fg_occ_feat).permute(0, 2, 1)  # [bs, occ_embedims, K]
        # fg_fused = torch.cat([fg_fused, fg_occ_embed.permute(0, 2, 1)], dim=-1)  # [bs, K, c_+occ_embedims]
        # fg_fused = self.history_fusion_linear(fg_fused).permute(0, 2, 1)  # [bs, c_, K]

        # # 背景融合（原有代码）
        # bg_history_feat = torch.gather(history_bev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, mc, 1))  # [bs, mc//2, bg_k]
        # bg_curr_feat = torch.gather(curr_bev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, c_, 1))  # [bs, c_, bg_k]

        # bg_history_feat_time = bg_history_feat.reshape(bs, self.history_num, c_, -1)  # [bs, T, c_//2, bg_k]
        # # bg_history_agg = (bg_history_feat_time * time_weights).sum(dim=1)  # [bs, c_//2, bg_k]
        # bg_V_prev = torch.gather(V_prev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, self.history_num, 1))  # [bs, bg_k]
        # bg_V_curr = torch.gather(V_curr_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, 1, 1))  # [bs, bg_k]
        # #TODO 这个10的超参数？ 调整成可学习？
        # # bg_time_vis_weights = bg_V_prev/(bg_V_prev.sum(dim=1).unsqueeze(1) +1e-10) # [bs, 4,K]
        # bg_time_vis_weights =(bg_V_prev*(self.bg_scale.view(1,self.history_num,1))).softmax(dim=1)
        # bg_history_agg = (bg_history_feat_time * bg_time_vis_weights.unsqueeze(2)).sum(dim=1)  # [bs, c_, K]).sum(dim=1)  # [bs, c_, K]]



        # # bg_history_agg_perm = F.pad(bg_history_agg.permute(0, 2, 1), (0, c_ - c_//2, 0, 0))  # [bs, bg_k, c_]

        # bg_history_agg_perm = bg_history_agg.permute(0, 2, 1)  # [bs, bg_k, c_]
        # # bg_V_prev_time = bg_V_prev.unsqueeze(1).repeat(1, self.history_num, 1)  # [bs, T, bg_k]
        # # bg_V_prev_agg = (bg_V_prev_time * time_weights.squeeze(-1)).sum(dim=1)  # [bs, bg_k]
        # bg_V_prev_agg = bg_V_prev.max(dim=1)[0]  # [bs, bg_k]
        # bg_w_hist, bg_w_curr = self.compute_gate_weights(bg_V_prev_agg, bg_V_curr.squeeze(1))  # [bs, bg_k, 1]

        # bg_curr_perm = bg_curr_feat.permute(0, 2, 1)  # [bs, bg_k, c_]
        # # print("*"*50)
        # # print("print(bg_w_hist.shape)")
        # # print(bg_w_hist.shape)
        # # print("print(bg_w_curr.shape)")
        # # print(bg_w_curr.shape)
        # # print("print(bg_history_agg_perm.shape)")
        # # print(bg_history_agg_perm.shape)
        # # print("print(bg_curr_perm.shape)")        
        # # print(bg_curr_perm.shape)

        # # # 断言批次大小一致
        # # assert bg_w_hist.shape[0] == bg_history_agg_perm.shape[0] == bg_w_curr.shape[0] == bg_curr_perm.shape[0], \
        # #     f"批次大小不匹配: {bg_w_hist.shape[0]}, {bg_history_agg_perm.shape[0]}, {bg_w_curr.shape[0]}, {bg_curr_perm.shape[0]}"

        # # # 断言第二维度（bg_k）一致
        # # assert bg_w_hist.shape[1] == bg_history_agg_perm.shape[1] == bg_w_curr.shape[1] == bg_curr_perm.shape[1], \
        # #     f"bg_k维度不匹配: {bg_w_hist.shape[1]}, {bg_history_agg_perm.shape[1]}, {bg_w_curr.shape[1]}, {bg_curr_perm.shape[1]}"

        # # # 断言第三维度（c_）匹配（bg_w_hist和bg_w_curr的第三维为1，不影响广播）
        # # assert bg_history_agg_perm.shape[2] == bg_curr_perm.shape[2], \
        # #     f"特征维度c_不匹配: {bg_history_agg_perm.shape[2]} vs {bg_curr_perm.shape[2]}"

        # # print("bg_w_hist dtype:", bg_w_hist.dtype)
        # # print("bg_history_agg_perm dtype:", bg_history_agg_perm.dtype)
        # # print("bg_w_curr dtype:", bg_w_curr.dtype)
        # # print("bg_curr_perm dtype:", bg_curr_perm.dtype)


        # # print("bg_w_hist device:", bg_w_hist.device)
        # # print("bg_history_agg_perm device:", bg_history_agg_perm.device)
        # # print("bg_w_curr device:", bg_w_curr.device)
        # # print("bg_curr_perm device:", bg_curr_perm.device)


        # bg_fused = bg_w_hist * bg_history_agg_perm + bg_w_curr * bg_curr_perm  # [bs, bg_k, c_]
        # # 先验证乘法是否正常
        # # temp1 = bg_w_hist * bg_history_agg_perm
        # # temp2 = bg_w_curr * bg_curr_perm
        # # 再验证加法是否正常

        # # bg_w_hist = bg_w_hist.contiguous()
        # # bg_history_agg_perm = bg_history_agg_perm.contiguous()
        # # bg_w_curr = bg_w_curr.contiguous()
        # # bg_curr_perm = bg_curr_perm.contiguous()

        # # # 重新计算
        # # temp1 = bg_w_hist * bg_history_agg_perm
        # # temp2 = bg_w_curr * bg_curr_perm
        # # temp1 = temp1.contiguous()
        # # temp2 = temp2.contiguous()
        # # bg_fused = temp1 + temp2


        # # 转移所有张量到CPU
        # # bg_w_hist_cpu = bg_w_hist.cpu()
        # # bg_history_agg_perm_cpu = bg_history_agg_perm.cpu()
        # # bg_w_curr_cpu = bg_w_curr.cpu()
        # # bg_curr_perm_cpu = bg_curr_perm.cpu()

        # # # 分步执行运算
        # # try:
        # #     temp1_cpu = bg_w_hist_cpu * bg_history_agg_perm_cpu
        # #     temp2_cpu = bg_w_curr_cpu * bg_curr_perm_cpu
        # #     bg_fused_cpu = temp1_cpu + temp2_cpu
        # #     print(bg_fused_cpu)
        # #     print(bg_fused_cpu.shape)
        # #     print("CPU运算成功，无明显错误")
        # # except Exception as e:
        # #     print(f"CPU运算报错：{e}")  # 此处会显示具体错误原因（如值异常）


        # # bg_fused = temp1.clone() + temp2.clone()
        # # 1/0
        # bg_occ_feat = torch.gather(last_occ_reshaped, dim=1, index=bg_indices.unsqueeze(-1).repeat(1, 1, last_occ_pred.shape[-1]))  # [bs, bg_k, num_classes]
        # bg_occ_embed = self.occ_embedding(bg_occ_feat).permute(0, 2, 1)  # [bs, occ_embedims, bg_k]
        # bg_fused = torch.cat([bg_fused, bg_occ_embed.permute(0, 2, 1)], dim=-1)  # [bs, bg_k, c_+occ_embedims]
        # bg_fused = self.history_fusion_bg_linear(bg_fused).permute(0, 2, 1)  # [bs, c_, bg_k]

        # 更新当前BEV
        curr_bev_updated = curr_bev_flat.clone()
        curr_bev_updated.scatter_add_(dim=2, index=fg_indices.unsqueeze(1).repeat(1, c_, 1), src=fg_fused)
        curr_bev_updated.scatter_add_(dim=2, index=bg_indices.unsqueeze(1).repeat(1, c_, 1), src=bg_fused)
        curr_bev_updated = curr_bev_updated.reshape(bs, c_, z, h, w)  # 恢复原形状

        # 更新历史缓存
        self.history_last_bev = curr_bev_updated.detach().clone()
        self.history_bev = torch.cat([curr_bev,sampled_history_bev[:, :-c_, ...]], dim=1).detach()
        # self.history_cam_intrins = torch.cat([curr_cam_intrins.unsqueeze(1),self.history_cam_intrins[:, :-1, ...]], dim=1).detach()
        # self.history_cam_extrins = torch.cat([curr_cam_extrins.unsqueeze(1),self.history_cam_extrins[:, 1-1:, ...]], dim=1).detach()
        self.history_visibility =torch.cat([V_curr, V_prev[:, :-1, ...]],dim=1).detach()
        self.history_forward_augs = forward_augs.clone()

        return curr_bev_updated

    def generate_grid(self, curr_bev):
        n, c_, z, h, w = curr_bev.shape
        # Generate grid
        xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w, 1).expand(h, w, z)
        ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1, 1).expand(h, w, z)
        zs = torch.linspace(0, z - 1, z, dtype=curr_bev.dtype, device=curr_bev.device).view(1, 1, z).expand(h, w, z)
        grid = torch.stack((xs, ys, zs, torch.ones_like(xs)), -1).view(1, h, w, z, 4).expand(n, h, w, z, 4).view(n, h,w, z, 4, 1)
        return grid

    def generate_feat2bev(self, grid, dx, bx):
        feat2bev = torch.zeros((4, 4), dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = dx[0]
        feat2bev[1, 1] = dx[1]
        feat2bev[2, 2] = dx[2]
        feat2bev[0, 3] = bx[0] - dx[0] / 2.
        feat2bev[1, 3] = bx[1] - dx[1] / 2.
        feat2bev[2, 3] = bx[2] - dx[2] / 2.
        feat2bev[3, 3] = 1
        feat2bev = feat2bev.view(1, 4, 4)
        return feat2bev


@HEADS.register_module()
class GatedTemporalFusion6_SE_channel(BaseModule):
    def __init__(
        self,
        history_num=4,
        depth_sampler_embed_dims=256,
        depth_sampler_num_heads=8,
        depth_sampler_num_levels=4,
        depth_sampler_num_points=4,
        im2col_step=64,
        top_k=500,
        single_bev_num_channels=96,
        occ_embedims=32,
        num_classes=18,
        vis_theta=0.28,
        vis_beta=10.0,
        vis_gamma=0.4,
        vis_sigma=0.1,
        nonempty_thresh=0.1,
        max_step_ratio=1.2,
        **kwargs
    ):
        super(GatedTemporalFusion6_SE_channel, self).__init__()

        # 基础参数
        self.history_num = history_num
        self.top_k = top_k
        self.single_bev_num_channels = single_bev_num_channels
        self.occ_embedims = occ_embedims
        # self.fg_scale = nn.Parameter(torch.full((history_num,), 10.0))
        # self.bg_scale = nn.Parameter(torch.full((history_num,), 10.0))
        # 可见性门控参数
        # self.vis_theta = vis_theta  # 可见性阈值
        # self.vis_beta = vis_beta    # 敏感度参数
        # self.vis_gamma = vis_gamma  # 场景4固定权重
        # self.vis_sigma = vis_sigma  # 软化参数
        # self.vis_theta = nn.Parameter(torch.tensor(vis_theta))   # 可见性阈值
        # self.vis_beta = nn.Parameter(torch.tensor(vis_beta))     # 敏感度参数
        # self.vis_gamma = nn.Parameter(torch.tensor(vis_gamma))   # 场景4固定权重
        # self.vis_sigma = nn.Parameter(torch.tensor(vis_sigma))  

        
        # 历史特征缓存（新增history_last_bev）
        self.history_bev = None  # 缓存历史多帧BEV特征
        self.history_last_bev = None  # 缓存上一帧最终融合后的BEV特征（关键新增）
        self.history_cam_intrins = None
        self.history_cam_extrins = None
        
        # 网络层（保持不变）
        self.occ_embedding = nn.Sequential(
            nn.Linear(num_classes, occ_embedims),
            nn.Softplus(),
            nn.Linear(occ_embedims, occ_embedims),
        )
        self.history_fusion_linear = nn.Sequential(
            nn.Linear(single_bev_num_channels * (history_num + 1) + occ_embedims, single_bev_num_channels),
            nn.Softplus(),
            nn.Linear(single_bev_num_channels, single_bev_num_channels),
        )
        self.history_fusion_bg_linear = nn.Sequential(
            nn.Linear(single_bev_num_channels * (history_num//2 + 1) + occ_embedims, single_bev_num_channels),
            nn.Softplus(),
            nn.Linear(single_bev_num_channels, single_bev_num_channels),
        )

        
        # 可见性计算组件（保持不变）
        self.rt_vis_calculator = EfficientRayTracingVisibility(
            nonempty_thresh=nonempty_thresh,
            max_step_ratio=max_step_ratio
        )
        # self.img_shape = (900, 1600)  # 默认图像尺寸

        # self.depth_sampler = DeformableDepthSampler(
        #     embed_dims=depth_sampler_embed_dims,
        #     num_heads=depth_sampler_num_heads,
        #     num_levels=depth_sampler_num_levels,
        #     num_points=depth_sampler_num_points
        # )
        self.history_forward_augs = None  # 用于缓存历史帧的变换矩阵（如BDAM矩阵）
        self.im2col_step = im2col_step
        # self.voxel_encoder = nn.Linear(single_bev_num_channels, depth_sampler_embed_dims)
        self.dbound = [1.0, 45.0, 0.5]
        self.pc_range = [-40, -40, -1.0, 40, 40, 5.4]
        self.final_dim = (256, 704)
    def compute_visibility(self, grid, cam_intrins, cam_extrins, img_shape, img_feats, spatial_shapes):
        """
        升级：结合可变形注意力采样的深度值优化可见性计算
        Args:
            img_feats: 多尺度图像特征 [bs, num_levels, c, h, w]
            spatial_shapes: 图像特征尺度 [num_levels, 2] (h, w)
            其他参数同原函数
        Returns:
            vis_prob: [bs, h, w, z] 优化后的可见性概率
        """
        bs, h, w, z, _ = grid.shape
        h_img, w_img = img_shape
        device = grid.device
        num_voxels = h * w * z  # 体素总数

        # 1. 原有相机投影逻辑（计算图像坐标和初始可见性）
        # 1.1 体素坐标→相机坐标→图像坐标
        grid_cam = grid.unsqueeze(1).expand(bs, self.num_cams, h, w, z, 3)  # [bs, num_cams, h, w, z, 3]
        grid_flat = grid_cam.reshape(-1, num_voxels, 3)  # [bs*num_cams, N, 3]
        grid_hom = torch.cat([grid_flat, torch.ones_like(grid_flat[..., :1])], dim=-1)  # [bs*num_cams, N, 4]

        cam_intrins_flat = cam_intrins.reshape(-1, 4, 4)  # [bs*num_cams, 4, 4]
        cam_extrins_flat = cam_extrins.reshape(-1, 4, 4)  # [bs*num_cams, 4, 4]
        extrins_inv = torch.inverse(cam_extrins_flat)
        cam_coords = torch.bmm(extrins_inv[:, :3, :4], grid_hom.permute(0, 2, 1)).permute(0, 2, 1)  # [bs*num_cams, N, 3]
        depth = cam_coords[..., 2:3] + 1e-8  # 相机坐标系下的深度

        # 1.2 计算初始可见性（原逻辑）
        img_coords = torch.bmm(cam_intrins_flat[:, :3, :3], cam_coords.permute(0, 2, 1)).permute(0, 2, 1)  # [bs*num_cams, N, 3]
        img_xy = img_coords[..., :2] / depth  # [bs*num_cams, N, 2] (u, v)
        depth_valid = (cam_coords[..., 2] > 0).float()  # [bs*num_cams, N]
        u_valid = (img_xy[..., 0] >= 0) & (img_xy[..., 0] < w_img)
        v_valid = (img_xy[..., 1] >= 0) & (img_xy[..., 1] < h_img)
        img_valid = (u_valid & v_valid).float()  # [bs*num_cams, N]
        initial_vis = depth_valid * img_valid  # [bs*num_cams, N]

        # 2. 可变形注意力深度采样
        # 2.1 准备输入：体素特征编码
        # voxel_feat = self.voxel_encoder(self.curr_bev_feat)  # [bs, c_, z, h, w] → [bs, z*h*w, embed_dims]（需提前展平体素特征）
        # voxel_feat = voxel_feat.reshape(bs, num_voxels, -1)  # [bs, N, embed_dims]

        # 2.2 生成参考点（归一化到[0,1]）
        norm_img_xy = img_xy / torch.tensor([w_img, h_img], device=device).view(1, 1, 2)  # [bs*num_cams, N, 2]
        # 取主相机（如第0个相机）的参考点作为采样基准
        ref_points = norm_img_xy.reshape(bs, self.num_cams, num_voxels, 2)[:, 0]  # [bs, N, 2]
        ref_points = ref_points.unsqueeze(2).repeat(1, 1, self.depth_sampler.num_levels, 1)  # [bs, N, L, 2]

        # 2.3 采样深度特征
        sampled_depth = self.depth_sampler(
            query=voxel_feat,
            value=img_feats,  # 多尺度图像特征 [bs, L, c, h, w]
            reference_points=ref_points,
            spatial_shapes=spatial_shapes
        )  # [bs, N]

        # 3. 结合采样深度优化可见性
        # 3.1 深度一致性校验：采样深度与相机投影深度的差异
        cam_depth = depth.reshape(bs, self.num_cams, num_voxels)[:, 0]  # 主相机的投影深度 [bs, N]
        depth_diff = torch.abs(sampled_depth - cam_depth) / (cam_depth + 1e-8)  # 相对深度差
        depth_consistent = (depth_diff < 0.3).float()  # 深度差小于30%则认为有效

        # 3.2 融合可见性：初始可见性 × 深度一致性
        initial_vis = initial_vis.reshape(bs, self.num_cams, num_voxels).max(dim=1)[0]  # [bs, N]（多相机取max）
        vis_prob = initial_vis * depth_consistent  # [bs, N]

        # 4. 还原形状
        return vis_prob.reshape(bs, h, w, z)  # [bs, h, w, z]

    def compute_alpha_unified(self, V_curr, V_prev):
        """统一计算当前帧融合权重α（覆盖四场景）"""
        eps = 1e-8
        # 1. 基础动态权重σ_base（场景1-3）
        ratio = V_curr / (V_curr + V_prev + eps)
        sigma_base = torch.sigmoid(self.vis_beta * (ratio - 0.5))
        
        # 2. 场景4软化掩码σ_both
        mask_curr = torch.sigmoid(-(V_curr - self.vis_theta) / self.vis_sigma)
        mask_prev = torch.sigmoid(-(V_prev - self.vis_theta) / self.vis_sigma)
        sigma_both = mask_curr * mask_prev
        
        # 3. 最终权重计算
        alpha = (1 - sigma_both) * sigma_base + sigma_both * self.vis_gamma
        return alpha.unsqueeze(-1)  # [bs, N, 1]

    def compute_gate_weights(self, V_prev_agg, V_curr):
        """计算历史和当前帧的门控权重"""
        alpha = self.compute_alpha_unified(V_curr, V_prev_agg)
        return 1 - alpha, alpha  # 历史权重，当前权重

    def generate_grid(self, curr_bev, voxel_min, voxel_max, voxel_size):
        """生成体素中心坐标网格（自车坐标系）"""
        bs, c_, z, h, w = curr_bev.shape
        device = curr_bev.device
        
        # 计算体素中心坐标
        x_coords = torch.linspace(
            voxel_min[0] + voxel_size[0]/2, 
            voxel_max[0] - voxel_size[0]/2, 
            w, device=device
        )
        y_coords = torch.linspace(
            voxel_min[1] + voxel_size[1]/2, 
            voxel_max[1] - voxel_size[1]/2, 
            h, device=device
        )
        z_coords = torch.linspace(
            voxel_min[2] + voxel_size[2]/2, 
            voxel_max[2] - voxel_size[2]/2, 
            z, device=device
        )
        
        # 生成网格并扩展维度 [bs, h, w, z, 3]
        x_grid, y_grid, z_grid = torch.meshgrid(x_coords, y_coords, z_coords, indexing='xy')
        grid = torch.stack([x_grid, y_grid, z_grid], dim=-1)  # [w, h, z, 3]
        grid = grid.permute(1, 0, 2, 3)  # [h, w, z, 3]
        return grid.unsqueeze(0).repeat(bs, 1, 1, 1, 1)  # [bs, h, w, z, 3]


    def get_reference_points(self, H, W, Z=None, num_points_in_pillar =4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
        torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    @force_fp32(apply_to=('reference_points', 'cam_params'))
    def point_sampling(self, reference_points, pc_range, img_metas, cam_params=None):
        # prepare for point sampling
        lidar2img = []
        ego2lidar = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])     # lidar2img update the post aug in the loading pipeline
            ego2lidar.append(img_meta['ego2lidar'])
        lidar2img = torch.stack(lidar2img, dim=0).to(reference_points.device)
        ego2lidar = torch.stack(ego2lidar, dim=0).to(reference_points.device)

        sensor2egos, ego2globals, intrins, post_augs, bda_mat = cam_params
        num_cam = sensor2egos.size(1)
        ogfH, ogfW = self.final_dim

        # reference_points defines in the bev space, [bs, D, hxw, 3]
        # change reference_points from bev-ego coordinate to ego coordinate
        reference_points = reference_points.clone()
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
                                     (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
                                     (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
                                     (pc_range[5] - pc_range[2]) + pc_range[2]

        # prepare for point sampling
        reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
        reference_points = reference_points.permute(1, 0, 2, 3)  # shape: (num_points_in_pillar,bs,h*w,4)
        D, B, num_query = reference_points.size()[:3]  # D=num_points_in_pillar , num_query=h*w
        reference_points = reference_points.view(D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)  # shape: (num_points_in_pillar,bs,num_cam,h*w,4)
        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)
        ego2lidar = ego2lidar.view(1, B, 1, 1, 4, 4).repeat(D, 1, num_cam, num_query, 1, 1)
        inverse_bda = bda_mat.view(1, B, 1, 1, 4, 4).repeat(D, 1, num_cam, num_query, 1, 1)

        # change reference_points from ego coordinate to img coordinate
        eps = 1e-5
        reference_points_cam = (lidar2img @ ego2lidar @ inverse_bda @ reference_points).squeeze(-1)   # [num_points_in_pillar, bs, num_cam, num_query=h*w, 4]
        reference_points_depth = reference_points_cam[..., 2:3]
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(reference_points_depth, torch.ones_like(reference_points_depth) * eps)

        # Bug!!
        # Correct normalize is
        # reference_points_cam[..., 0] /= ogfW
        # reference_points_cam[..., 1] /= ogfH
        # But for reproducing our results, we use the following normalization
        reference_points_cam[..., 0] /= ogfW
        reference_points_cam[..., 1] /= ogfH

        bev_mask = (reference_points_depth > eps)
        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)                  # shape: (num_cam, bs,h*w, num_points_in_pillar, 2)
        reference_points_depth = reference_points_depth.permute(2, 1, 3, 0, 4)              # shape: (num_cam, bs,h*w, num_points_in_pillar, 1)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)                        # shape: (num_cam, bs,h*w, num_points_in_pillar)

        return reference_points_cam, reference_points_depth, bev_mask


    @force_fp32()
    def forward(self, curr_bev, cam_params, history_fusion_params, dx, bx, history_last_bev=None, last_occ_pred=None, nonempty_prob=None, img_feats=None, spatial_shapes=None,pred_img_depth=None,**kwargs):
        """
        Args:
            curr_bev: [bs, c, z, h, w] 当前帧BEV特征
            cam_params: 相机参数列表，包含外参、内参等
            history_fusion_params: 历史融合参数（包含序列信息等）
            dx: 体素尺寸 (x, y, z)
            bx: 体素偏移
            nonempty_prob: [bs, z, h, w] 体素非空概率
            last_occ_pred: [bs, z, h, w, num_classes] 上一时刻 occupancy 预测
            img_feats: 多尺度图像特征 [bs, num_levels, c, h, w]（新增，用于深度采样）
            spatial_shapes: 图像特征尺度 [num_levels, 2] (h, w)（新增，用于深度采样）
        Returns:
            curr_bev_updated: [bs, c, z, h, w] 融合后BEV特征
        """
        # print(self.history_num)
        # print("print(self.fg_scale)")
        # print(self.fg_scale) 
        # print("print(self.bg_scale)")
        # print(self.bg_scale)
        # # 可见性门控参数
        # # self.vis_theta = vis_theta  # 可见性阈值
        # # self.vis_beta = vis_beta    # 敏感度参数
        # # self.vis_gamma = vis_gamma  # 场景4固定权重
        # # self.vis_sigma = vis_sigma  # 软化参数
        # print("print(self.vis_theta # 可见性阈值)")
        # print(self.vis_theta )  # 可见性阈值
        # print("print(self.vis_beta) # 敏感度参数")
        # print(self.vis_beta)      # 敏感度参数
        # print("print(self.vis_gamma) # 场景4固定权重")
        # print(self.vis_gamma)  # 场景4固定权重
        # print("print(self.vis_sigma) # 软化参数")
        # print(self.vis_sigma) 


        # if torch.rand(1).item() < 1/2000:
        #     print(self.history_num)
        #     print("print(self.fg_scale)")
        #     print(self.fg_scale) 
        #     print("print(self.bg_scale)")
        #     print(self.bg_scale)
        #     # 可见性门控参数
        #     print("print(self.vis_theta # 可见性阈值)")
        #     print(self.vis_theta)  # 可见性阈值
        #     print("print(self.vis_beta) # 敏感度参数")
        #     print(self.vis_beta)      # 敏感度参数
        #     print("print(self.vis_gamma) # 场景4固定权重")
        #     print(self.vis_gamma)  # 场景4固定权重
        #     print("print(self.vis_sigma) # 软化参数")
        #     print(self.vis_sigma)  


        # -------------------------- 1. 解析参数后打印核心形状 --------------------------
        # 解析相机参数
        curr_cam_extrins = cam_params[0]  # [bs, num_cams, 4, 4]
        curr_cam_intrins = cam_params[2]  # [bs, num_cams, 4, 4]
        forward_augs = cam_params[4]      # [bs, 4, 4] 前向变换矩阵
        self.num_cams = curr_cam_extrins.shape[1]  # 从外参中获取相机数量
        bs, c_, z, h, w = curr_bev.shape
        device = curr_bev.device
        mc = self.history_num * c_        # 历史特征总通道数
        # self.history_forward_augs = forward_augs.clone()


        ref_3d = self.get_reference_points(
            h, w, z, z, dim='3d', bs=bs, device=device, dtype=curr_bev.dtype) # torch.Size([3, 2, 625, 3]) #[bs,z,yx,3(x,y,z)]
        # ref_2d = self.get_reference_points(
        #     h, w, dim='2d', bs=bs, device=device, dtype=curr_bev.dtype) #torch.Size([3, 625, 1, 2])
        slots = torch.zeros(list([ref_3d.shape[0],ref_3d.shape[2],ref_3d.shape[1]])).to(ref_3d)
        reference_points_cam, reference_points_depth, bev_mask = self.point_sampling(ref_3d, self.pc_range, img_metas=kwargs['img_metas'], cam_params=cam_params)
        indexes = [[] for _ in range(bs)]
        spatial_shapes =[]
        spatial_shapes.append([16, 44])
        spatial_shapes = torch.tensor(spatial_shapes).to(device)
        pred_img_depth = pred_img_depth.view(bs * 6, -1, spatial_shapes[0][0], spatial_shapes[0][1])
        pred_img_depth = pred_img_depth.flatten(2).permute(0, 2, 1)  
        max_len = 0
        for j in range(bs):
            for i, mask_per_img in enumerate(bev_mask):
                index_query_per_img = mask_per_img[j].sum(-1).nonzero().squeeze(-1)
                if len(index_query_per_img) == 0:
                    index_query_per_img = bev_mask[i][j].sum(-1).nonzero().squeeze(-1)[0:1]
                indexes[j].append(index_query_per_img)
                # for batch operation, we need to pad the indexes to the same length
                max_len = max(max_len, len(index_query_per_img))
        reference_points_cam_rebatch = reference_points_cam.new_zeros([bs, self.num_cams, max_len, z, 2])
        reference_points_depth_rebatch = reference_points_depth.new_zeros([bs, self.num_cams, max_len, z, 1])

        for j in range(bs):
            for i, (reference_points_per_img, reference_points_depth_per_img) in enumerate(zip(reference_points_cam, reference_points_depth)):
                index_query_per_img = indexes[j][i]
                reference_points_cam_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]
                reference_points_depth_rebatch[j, i, :len(index_query_per_img)] = reference_points_depth_per_img[j, index_query_per_img]

        #use deformble attn
        depth_reference_points = reference_points_cam_rebatch.reshape(bs*6, max_len*z, 1, 1, 1, 2).contiguous()
        depth_attention_weights = torch.ones_like(depth_reference_points[..., 0]).contiguous()
        pred_img_depth = pred_img_depth.unsqueeze(2).contiguous()
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))



        bev_query_depth_rebatch = (reference_points_depth_rebatch- self.dbound[0])/ self.dbound[2]
        bev_query_depth_rebatch = torch.clip(torch.floor(bev_query_depth_rebatch), 0, 88-1).to(torch.long)
        bev_query_depth_rebatch = F.one_hot(bev_query_depth_rebatch.squeeze(-1),
                                   num_classes=88)

        depth_output = MultiScaleDeformableAttnFunction_fp32.apply(pred_img_depth, spatial_shapes,level_start_index,depth_reference_points,depth_attention_weights, self.im2col_step)
        depth_output = depth_output.reshape(bs,6, max_len,z, -1)   # [bs*num_cam, num_query, num_Z_anchors, C]
        # reference_points_depth_rebatch

        increment = torch.zeros_like(depth_output)
        # increment[..., 0] = 1e-9  # 非原地赋值（创建新张量）
        # depth_output = depth_output + increment
        depth_output = depth_output + torch.cat([(torch.zeros_like(depth_output[...,:1]) + 1e-9),torch.zeros_like(depth_output[...,1:])],dim=-1)

        # print("depth_output.sum(-1).shape")
        # print(depth_output.sum(-1).shape)
        # print("depth_output.sum(-1)")
        # print(depth_output.sum(-1))
        # print("print((depth_output.sum(-1)==0).sum())")
        # print((depth_output.sum(-1)==0).sum())
        depth_output =depth_output/depth_output.sum(-1)[...,None] #bs,xy,z,D

        # print("depth_output.sum(-1).shape")
        # print(depth_output.sum(-1).shape)
        # print("depth_output.sum(-1)")
        # print(depth_output.sum(-1))
        # print("print((depth_output.sum(-1)==1).sum())")
        # print((depth_output.sum(-1)>=0.99).sum())
        # print("depth_output")
        # print(depth_output)


        depth_output = (1-depth_output.cumsum(dim=-1))
        depth_output =torch.cat([torch.ones_like(depth_output[...,0:1]),depth_output[...,0:-1]],dim=-1)

        # print("print((depth_output[...,-1]==0).sum())")
        # print((depth_output[...,-1]==0).sum())
        depth_output = (bev_query_depth_rebatch*depth_output).sum(-1)

        #恢复depth_output的shape
        # fix_depth_output = depth_output.new_zeros([bs, 6, w*h, z, 88])
        # for j in range(bs):
        #     for i in range(6):
        #         index_query_per_img = indexes[j][i]
        #         fix_depth_output[j, i, index_query_per_img] = depth_output[j, i, :len(index_query_per_img)]

        for j in range(bs):
            for i in range(6):
                index_query_per_img = indexes[j][i]
                slots[j, index_query_per_img] = torch.max(slots[j, index_query_per_img],depth_output[j, i, :len(index_query_per_img)])

        # depth_sum = fix_depth_output.sum(dim=-1).view(3,6,25,25,2)
        # bs = depth_sum.shape[0]       # 3
        # num_cam = depth_sum.shape[1]  # 6
        # z_layers = depth_sum.shape[4] # 2
        # H, W = depth_sum.shape[2], depth_sum.shape[3]  # 25,25

        # # 颜色映射：0值用黑色，非0值用渐变色
        # cmap = plt.cm.viridis
        # cmap.set_bad(color='black')  # 0值标记为黑色

        # # 设置子图布局（不变）
        # fig, axes = plt.subplots(
        #     nrows=bs, ncols=num_cam * z_layers,
        #     figsize=(30, 8),
        #     squeeze=False
        # )
        # fig.suptitle('fix_depth_output.sum(-1) 可视化（黑色=0值，颜色=非0值）', fontsize=16, y=0.98)


        # # -------------------------- 2. 循环绘制每个子图（核心修改：添加 .detach()） --------------------------
        # for batch_idx in range(bs):
        #     for cam_idx in range(num_cam):
        #         for z_idx in range(z_layers):
        #             col_idx = cam_idx * z_layers + z_idx
        #             ax = axes[batch_idx, col_idx]
                    
        #             # 核心修改：先 detach() 切断计算图，再转 cpu 和 numpy
        #             data = depth_sum[batch_idx, cam_idx, :, :, z_idx].detach().cpu().numpy()
        #             data[data == 0] = np.nan  # 0值替换为NaN，显示为黑色
                    
        #             # 绘制热力图（修改 vmin/vmax：同样添加 .detach()）
        #             im = ax.imshow(
        #                 data, 
        #                 cmap=cmap, 
        #                 aspect='auto',
        #                 # 关键修改：depth_sum 先 detach 再转 numpy，确保不影响梯度
        #                 vmin=np.nanmin(depth_sum.detach().cpu().numpy()),
        #                 vmax=np.nanmax(depth_sum.detach().cpu().numpy())
        #             )
                    
        #             # 子图标题和坐标轴（不变）
        #             ax.set_title(
        #                 f'Batch{batch_idx+1}\nCam{cam_idx+1} Z{z_idx+1}',
        #                 fontsize=10, pad=5
        #             )
        #             ax.set_xticks([])
        #             ax.set_yticks([])


        # # -------------------------- 3. 添加颜色条（不变） --------------------------
        # cbar = fig.colorbar(
        #     im, 
        #     ax=axes.ravel().tolist(),
        #     shrink=0.8,
        #     pad=0.02
        # )
        # cbar.set_label('Sum of Depth Bins (D=88)', fontsize=12)


        # # -------------------------- 4. 调整布局并保存（不变） --------------------------
        # plt.tight_layout(rect=[0, 0, 0.98, 0.95])
        # plt.savefig('depth_sum_visualization.png', dpi=300, bbox_inches='tight')
        # plt.show()

        # output

        #计数更新，建立在纸上的假设成立的基础上
        # count = bev_mask.sum(-1) > 0
        # count = count.permute(1, 2, 0).sum(-1)
        # count = torch.clamp(count, min=1.0)
        # slots = slots / count[..., None]

        # print("slots.shape")
        # print(slots.shape)
        # print("slots")  
        # print(slots)
        # zzzzz=1/0

        V_curr =slots.view(bs, 1, h, w,z).permute(0, 1, 4, 2, 3)
        # slots[...,0]+=1e-9
        # print("slots.sum(-1).shape")
        # print(slots.sum(-1).shape)
        # print("slots.sum(-1)")
        # print(slots.sum(-1))
        # print("print((slots.sum(-1)==0).sum())")
        # print((slots.sum(-1)==0).sum())
        # slots =slots/slots.sum(-1)[...,None] #bs,xy,z,D

        # print("slots.sum(-1).shape")
        # print(slots.sum(-1).shape)
        # print("slots.sum(-1)")
        # print(slots.sum(-1))
        # print("print((slots.sum(-1)==1).sum())")
        # print("slots")
        # print(slots)


        # slots = (1-slots.cumsum(dim=-1))

        # print("print((slots[...,-1]==0).sum())")
        # print((slots[...,-1]==0).sum())

        # print("slots")
        # print(slots)
        # print("print((slots[...,-1]<0.01).sum())")
        # print((slots[...,-1]<0.01).sum())

        #TODO 这里的对于边界值的考虑，从0开始还是从1开始？




        # slots = self.output_proj(slots)
        #TODO 上面这里需要检查一下
        #这里相当于两次softmax，可能会导致分布变得不够尖锐，需要进一步确认

        #到这里slots就是可见性的概率分布了
        #这里先尝试使用期望值进行计算，使得可微分

        #然后再采用stc的原始离散计算方法，


        # 打印解析后关键变量形状
        # print("="*50)
        # print("1. 解析参数后核心变量形状：")
        # print(f"curr_bev: {curr_bev.shape} (预期：[bs, c, z, h, w])")
        # print(f"curr_cam_extrins: {curr_cam_extrins.shape} (预期：[bs, num_cams, 4, 4])")
        # print(f"curr_cam_intrins: {curr_cam_intrins.shape} (预期：[bs, num_cams, 4, 4])")
        # print(f"forward_augs: {forward_augs.shape} (预期：[bs, 4, 4])")
        # print(f"dx: {dx.shape if hasattr(dx, 'shape') else type(dx)} (预期：[3])")
        # print(f"bx: {bx.shape if hasattr(bx, 'shape') else type(bx)} (预期：[3])")
        # print(f"bs: {bs}, c_: {c_}, z: {z}, h: {h}, w: {w} (BEV特征维度)")
        # print("="*50)

        if type(history_fusion_params['sequence_group_idx']) is list:
            seq_ids = history_fusion_params['sequence_group_idx'][0]
        else:
            seq_ids = history_fusion_params['sequence_group_idx']
        if type(history_fusion_params['start_of_sequence']) is list:
            start_of_sequence = history_fusion_params['start_of_sequence'][0]
        else:
            start_of_sequence = history_fusion_params['start_of_sequence']
        if type(history_fusion_params['curr_to_prev_ego_rt']) is list:
            curr_to_prev_ego_rt = history_fusion_params['curr_to_prev_ego_rt'][0]
        else:
            curr_to_prev_ego_rt = history_fusion_params['curr_to_prev_ego_rt']
        forward_augs = cam_params[-1]  # bda

        # check seq_ids > 0
        assert (seq_ids >= 0).all()
        # -------------------------- 2. 初始化历史缓存后打印 --------------------------
        if self.history_bev is None:
            # self.history_bev = curr_bev.repeat(1, self.history_num, 1, 1, 1)  # [bs, mc, z, h, w]
            # self.history_cam_intrins = curr_cam_intrins.unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)  # [bs, T, num_cams, 4, 4]
            # self.history_cam_extrins = curr_cam_extrins.unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)  # [bs, T, num_cams, 4, 4]
            # self.history_bev = curr_bev.clone()
            self.history_forward_augs = forward_augs.clone()
            self.history_seq_ids = seq_ids.clone()
            self.history_bev = curr_bev.repeat(1, self.history_num, 1, 1, 1)
            self.history_sweep_time = curr_bev.new_zeros(curr_bev.shape[0], self.history_num)
            self.history_visibility = V_curr.repeat(1, self.history_num, 1, 1, 1).half()
        self.history_bev = self.history_bev.detach()
        self.history_visibility = self.history_visibility.detach().half()
        self.history_sweep_time += 1

        # 打印历史缓存形状
        # print("\n2. 历史缓存初始化后形状：")
        # print(f"history_bev: {self.history_bev.shape} (预期：[bs, mc, z, h, w]，mc={mc})")
        # print(f"history_cam_intrins: {self.history_cam_intrins.shape} (预期：[bs, history_num, num_cams, 4, 4])")
        # print(f"history_cam_extrins: {self.history_cam_extrins.shape} (预期：[bs, history_num, num_cams, 4, 4])")

        # -------------------------- 3. 生成网格和BEV变换矩阵后打印 --------------------------
        # 处理新序列（略，不影响维度）
        # start_of_sequence = history_fusion_params.get('start_of_sequence', torch.zeros(bs, dtype=torch.bool, device=device))
        if start_of_sequence.sum()>0:
            self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_num, 1, 1, 1)
            # self.history_cam_intrins[start_of_sequence] = curr_cam_intrins[start_of_sequence].unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)
            # self.history_cam_extrins[start_of_sequence] = curr_cam_extrins[start_of_sequence].unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)
            self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]
            self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
            self.history_sweep_time[start_of_sequence] = 0  # zero the new sequence timestep starts
            self.history_visibility[start_of_sequence] = V_curr[start_of_sequence].repeat(1, self.history_num, 1, 1, 1).half()

        # 生成体素网格和BEV变换矩阵
        grid = self.generate_grid(curr_bev) #[bs,y,x,z,4]
        grid_3d = grid
        feat2bev = self.generate_feat2bev(grid, dx, bx)

        # 打印网格和变换矩阵形状
        # print("\n3. 生成网格和BEV变换矩阵后形状：")
        # print(f"grid_3d (体素网格): {grid_3d.shape} (关键！预期：[bs, h, w, z, 3] 或 [bs, w, h, z, 3])")
        # print(f"feat2bev (BEV变换矩阵): {feat2bev.shape} (预期：[bs, 4, 4])")

        # -------------------------- 4. 运动补偿矩阵计算后打印 --------------------------
        # 获取帧间姿态变换
        # curr_to_prev_ego_rt = history_fusion_params.get('curr_to_prev_ego_rt', torch.eye(4, device=device).unsqueeze(0).repeat(bs, 1, 1))
        # 计算RT流（坐标变换矩阵）
        rt_flow = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_ego_rt @ torch.inverse(forward_augs) @ feat2bev)
        # 生成齐次网格
        # 在forward函数中，生成grid_hom的位置修正：
        # grid_3d = self.generate_grid(curr_bev)  # 现在形状：[3, 25, 25, 2, 3]（bs, h, w, z, 3）
        # # 生成齐次坐标（x,y,z,1），并添加最后一个维度（用于矩阵乘法）
        # grid_hom = torch.cat([
        #     grid_3d,  # [3,25,25,2,3]
        #     torch.ones_like(grid_3d[..., :1])  # [3,25,25,2,1]（补充1作为齐次坐标）
        # ], dim=-1).unsqueeze(-1)  # 最终形状：[3,25,25,2,4,1]（符合预期）
        # # 打印运动补偿相关形状（矩阵乘法前关键检查）
        # print("\n4. 运动补偿矩阵计算后形状（矩阵乘法前）：")
        # print(f"curr_to_prev_ego_rt (帧间姿态): {curr_to_prev_ego_rt.shape} (预期：[bs, 4, 4])")
        # print(f"rt_flow (变换流): {rt_flow.shape} (预期：[bs, 4, 4])")
        # print(f"grid_hom (齐次网格): {grid_hom.shape} (关键！预期：[bs, h, w, z, 4, 1]，需与rt_flow广播匹配)")
        # print(f"rt_flow.view后: {rt_flow.view(bs, 1, 1, 1, 4, 4).shape} (预期：[bs, 1, 1, 1, 4, 4])")

        # # -------------------------- 5. 网格变换后打印（解决之前维度错的核心） --------------------------
        # try:
        #     grid_transformed = rt_flow.view(bs, 1, 1, 1, 4, 4) @ grid_hom  # 矩阵乘法：[bs, h, w, z, 4, 1]
        #     print("\n5. 网格变换后形状（矩阵乘法成功！）：")
        #     print(f"grid_transformed: {grid_transformed.shape} (预期：[bs, h, w, z, 4, 1])")
        # except RuntimeError as e:
        #     print(f"\n5. 网格变换矩阵乘法报错！错误信息：{str(e)}")
        #     print(f"  - rt_flow.view形状: {rt_flow.view(bs, 1, 1, 1, 4, 4).shape}")
        #     print(f"  - grid_hom形状: {grid_hom.shape}")
        #     print("  提示：需确保grid_hom的第1-4维度与rt_flow.view的第2-5维度匹配（广播规则）")
        #     raise e  # 继续抛出错误，方便定位
        bs, mc, z, h, w = self.history_bev.shape
        n, c_, z, h, w = curr_bev.shape
        grid = rt_flow.view(n, 1, 1, 1, 4, 4) @ grid
        # -------------------------- 6. 采样网格生成后打印 --------------------------
        # 生成采样网格（归一化到[-1,1]，适配F.grid_sample）
        normalize_factor = torch.tensor([w - 1.0, h - 1.0, z - 1.0], dtype=curr_bev.dtype, device=device)
        # grid_sampler = grid_transformed[..., :3, 0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0  # [bs, h, w, z, 3]
        # # 调整采样网格维度（适配F.grid_sample输入：[bs, z, h, w, 3]）
        # grid_sampler_permuted = grid_sampler.permute(0, 3, 1, 2, 4)  # 交换z和h/w维度
        grid = grid[:, :, :, :, :3, 0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0   # grid order is x, y, z


        # print("\n6. 采样网格生成后形状：")
        # print(f"grid_sampler (归一化后): {grid_sampler.shape} (预期：[bs, h, w, z, 3])")
        # print(f"grid_sampler_permuted (适配采样): {grid_sampler_permuted.shape} (预期：[bs, z, h, w, 3])")

        # -------------------------- 7. 历史BEV采样后打印 --------------------------
        # 采样历史BEV特征
        sampled_history_bev = F.grid_sample(
            self.history_bev.reshape(bs, mc, z, h, w),  # 输入：[bs, mc, z, h, w]
            grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4),
            align_corners=True,
            mode='bilinear'
        )
        sampled_history_visibility = F.grid_sample(
            self.history_visibility.reshape(bs, self.history_num, z, h, w).half(),  # 输入：[bs, mc, z, h, w]
            grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4).half(),
            align_corners=True,
            mode='nearest'
        )
        # print("\n7. 历史BEV采样后形状：")
        # print(f"history_bev.reshape: {self.history_bev.reshape(bs, mc, z, h, w).shape} (预期：[bs, mc, z, h, w])")
        # print(f"sampled_history_bev: {sampled_history_bev.shape} (预期：[bs, mc, z, h, w])")

        # -------------------------- 8. 可见性计算后打印 --------------------------
        # 计算当前帧可见性
        # V_curr = self.compute_visibility(
        #     grid_3d, 
        #     cam_intrins=curr_cam_intrins,
        #     cam_extrins=curr_cam_extrins,
        #     img_shape=self.img_shape,
        #     img_feats=img_feats,
        #     spatial_shapes=spatial_shapes
        # )

        # print("V_curr (当前可见性).  "*3)
        # V_curr = slots
        # 计算历史帧可见性
        # prev_cam_intrins = self.history_cam_intrins[:, -1]
        # prev_cam_extrins = self.history_cam_extrins[:, -1]
        # V_prev = self.compute_visibility(
        #     grid_3d,
        #     cam_intrins=prev_cam_intrins,
        #     cam_extrins=prev_cam_extrins,
        #     img_shape=self.img_shape,
        #     img_feats=img_feats,
        #     spatial_shapes=spatial_shapes
        # )

        V_prev = sampled_history_visibility #bs,4,z,h,w

        # print("\n8. 可见性计算后形状：")
        # print(f"V_curr (当前可见性): {V_curr.shape} (预期：[bs, h, w, z])")
        # print(f"V_prev (历史可见性): {V_prev.shape} (预期：[bs, h, w, z])")

        # -------------------------- 9. 稀疏采样前展平变量打印 --------------------------
        # 展平变量（用于稀疏采样）
        curr_bev_flat = curr_bev.reshape(bs, c_, -1)  # [bs, c_, N], N=h*w*z
        history_bev_flat = sampled_history_bev.reshape(bs, mc, -1)  # [bs, mc, N]
        V_prev_flat = V_prev.reshape(bs,self.history_num, -1)  # [bs, 4,N]
        V_curr_flat = V_curr.reshape(bs, 1,-1)  # [bs, 1,N]
        nonempty_prob_flat = nonempty_prob.reshape(bs, -1)  # [bs, N]
        # print("print(nonempty_prob_flat.shape)")
        # print(nonempty_prob_flat.shape)
        total_voxels = nonempty_prob_flat.shape[1]

        # print("\n9. 稀疏采样前展平变量形状：")
        # print(f"curr_bev_flat: {curr_bev_flat.shape} (预期：[bs, c_, N], N={total_voxels})")
        # print(f"history_bev_flat: {history_bev_flat.shape} (预期：[bs, mc, N])")
        # print(f"nonempty_prob_flat: {nonempty_prob_flat.shape} (预期：[bs, N])")
        # print(f"total_voxels (h*w*z): {total_voxels} (预期：{h*w*z})")

        # -------------------------- 10. 前景/背景索引及融合后打印（可选，确认后续维度） --------------------------
        # 生成前景/背景索引
        fg_indices = torch.topk(nonempty_prob_flat, self.top_k, dim=1)[1]
        bg_indices = torch.topk(1 - nonempty_prob_flat, total_voxels - self.top_k, dim=1)[1]
        # 提取前景特征（示例，其他融合步骤类似）
        fg_history_feat = torch.gather(history_bev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, mc, 1))

        # print("\n10. 前景/背景索引及特征提取后形状：")
        # print(f"fg_indices (前景索引): {fg_indices.shape} (预期：[bs, top_k])")
        # print(f"bg_indices (背景索引): {bg_indices.shape} (预期：[bs, N-top_k])")
        # print(f"fg_history_feat (前景历史特征): {fg_history_feat.shape} (预期：[bs, mc, top_k])")
        # print("="*50)

        # -------------------------- 后续原有逻辑（略，维度已通过打印确认） --------------------------
        # 8. 前景融合（原有代码）
        # 9. 背景融合（原有代码）
        # 10. 更新当前BEV特征（原有代码）
        # curr_bev_flat = curr_bev.reshape(bs, c_, -1)  # [bs, c_, N] N=h*w*z
        # history_bev_flat = sampled_history_bev.reshape(bs, mc, -1)  # [bs, mc, N]
        # V_prev_flat = V_prev.reshape(bs, -1)  # [bs, N]
        # V_curr_flat = V_curr.reshape(bs, -1)  # [bs, N]
        # nonempty_prob_flat = nonempty_prob.reshape(bs, -1)  # [bs, N]
        # total_voxels = nonempty_prob_flat.shape[1]

        # fg_indices = torch.topk(nonempty_prob_flat, self.top_k, dim=1)[1]  # [bs, top_k]
        # bg_indices = torch.topk(1 - nonempty_prob_flat, total_voxels - self.top_k, dim=1)[1]  # [bs, N-top_k]

        # # 前景特征提取
        # fg_history_feat = torch.gather(history_bev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, mc, 1))  # [bs, mc, top_k]
        fg_curr_feat = torch.gather(curr_bev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, c_, 1))  # [bs, c_, top_k]

        # 历史特征时间聚合
        # fg_history_feat_time = fg_history_feat.reshape(bs, self.history_num, c_, self.top_k)  # [bs, T, c_, K]
        #TODO 后续可以把time_weights也乘进去


        # time_weights = torch.exp(-0.5 * torch.arange(self.history_num, device=device)).view(1, self.history_num, 1, 1)

        fg_V_prev = torch.gather(V_prev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, self.history_num, 1))  # [bs, 4,K]
        fg_V_curr = torch.gather(V_curr_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, 1, 1))  # [bs, 1,K]
        # fg_time_vis_weights = fg_V_prev/(fg_V_prev.sum(dim=1).unsqueeze(1)+1e-10 ) # [bs, 4,K]
        #下面进行替换，不用显示提取权重，不计算 softmax
        # fg_time_vis_weights = (fg_V_prev*(self.fg_scale.view(1,self.history_num,1))).softmax(dim=1)
        # fg_time_vis_weights = (fg_V_prev*(self.fg_scale.view(1,self.history_num,1)))
        print("print(last_occ_pred.shape)")
        print(last_occ_pred.shape)
        last_occ_pred = last_occ_pred.permute(0, 3, 2, 1, 4).reshape(bs, -1, last_occ_pred.shape[-1])  # [bs, N, num_classes]

        fg_occ_feat = torch.gather(last_occ_pred, dim=1, index=fg_indices.unsqueeze(-1).repeat(1, 1, last_occ_pred.shape[-1]))  # [bs, K, num_classes]

        fg_occ_embed = self.occ_embedding(fg_occ_feat).permute(0, 2, 1)  # [bs, occ_embedims, K]

        print("print(fg_V_curr.shape)")
        print(fg_V_curr.shape)
        print("print(fg_curr_feat.shape)")
        print(fg_curr_feat.shape)
        print("print(fg_V_prev.shape)")
        print(fg_V_prev.shape)
        print("print(fg_history_feat.shape)")
        print(fg_history_feat.shape)
        print("print(fg_occ_embed.shape)")
        print(fg_occ_embed.shape)
        fg_fused = torch.cat([ (fg_V_curr * fg_curr_feat.permute(0, 2, 1)).permute(0, 2, 1),(fg_V_prev.unsqueeze(2) * fg_history_feat.view(bs, self.history_num, c_, self.top_k)).reshape(bs, self.history_num*c_, self.top_k).permute(0, 2, 1) , fg_occ_embed.permute(0, 2, 1)], dim=-1)

        fg_fused = self.history_fusion_linear(fg_fused).permute(0, 2, 1)  # [bs, c_, K]


        bg_V_prev = torch.gather(V_prev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, self.history_num, 1))  # [bs, bg_k]
        bg_V_curr = torch.gather(V_curr_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, 1, 1))  # [bs, bg_k]
        bg_history_feat = torch.gather(history_bev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, mc, 1))  # [bs, mc//2, bg_k]
        bg_curr_feat = torch.gather(curr_bev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, c_, 1))  # [bs, c_, bg_k]
        bg_occ_feat = torch.gather(last_occ_reshaped, dim=1, index=bg_indices.unsqueeze(-1).repeat(1, 1, last_occ_pred.shape[-1]))  # [bs, bg_k, num_classes]
        bg_occ_embed = self.occ_embedding(bg_occ_feat).permute(0, 2, 1)  # [bs, occ_embedims, bg_k]

        # print(bg_V_curr.shape)
        # print
        bg_fused = torch.cat([ (bg_V_curr * bg_curr_feat.permute(0, 2, 1)).permute(0, 2, 1),(bg_V_prev.unsqueeze(2) * bg_history_feat.view(bs, self.history_num, c_, self.top_k)).reshape(bs, self.history_num*c_, self.top_k).permute(0, 2, 1) , bg_occ_embed.permute(0, 2, 1)], dim=-1)


        #TODO TODO 后续用senet实现一个通道注意力的版本  这个可能会更好？


        # # print("print(fg_history_feat_time.shape)")
        # # print(fg_history_feat_time.shape)
        # # print("print(fg_time_vis_weights.shape)")
        # # print(fg_time_vis_weights.shape)
        # # fg_history_agg = (fg_history_feat_time * fg_time_vis_weights.unsqueeze(2)).sum(dim=1)  # [bs, c_, K]).sum(dim=1)  # [bs, c_, K]]
        # fg_history_agg = (fg_history_feat_time * fg_time_vis_weights.unsqueeze(2)).sum(dim=1)  # [bs, c_, K]).sum(dim=1)  # [bs, c_, K]]


        # # print("print(fg_history_agg.shape)")
        # # print(fg_history_agg.shape)

        # # 可见性聚合与门控
        
        # # fg_V_prev_time = fg_V_prev.unsqueeze(1).repeat(1, self.history_num, 1)  # [bs, T, K]
        # # fg_V_prev_agg = (fg_V_prev_time * time_weights.squeeze(-1)).sum(dim=1)  # [bs, K]
        # fg_V_prev_agg = fg_V_prev.max(dim=1)[0]  # [bs, K]
        # fg_w_hist, fg_w_curr = self.compute_gate_weights(fg_V_prev_agg, fg_V_curr.squeeze(1))  # [bs, K, 1]

        # # 前景融合
        # fg_history_agg_perm = fg_history_agg.permute(0, 2, 1)  # [bs, K, c_]
        # fg_curr_perm = fg_curr_feat.permute(0, 2, 1)  # [bs, K, c_]
        # fg_fused = fg_w_hist * fg_history_agg_perm + fg_w_curr * fg_curr_perm  # [bs, K, c_]

        # # occupancy嵌入融合
        # last_occ_reshaped = last_occ_pred.permute(0, 3, 2, 1, 4).reshape(bs, -1, last_occ_pred.shape[-1])  # [bs, N, num_classes]
        # fg_occ_feat = torch.gather(last_occ_reshaped, dim=1, index=fg_indices.unsqueeze(-1).repeat(1, 1, last_occ_pred.shape[-1]))  # [bs, K, num_classes]
        # fg_occ_embed = self.occ_embedding(fg_occ_feat).permute(0, 2, 1)  # [bs, occ_embedims, K]
        # fg_fused = torch.cat([fg_fused, fg_occ_embed.permute(0, 2, 1)], dim=-1)  # [bs, K, c_+occ_embedims]
        # fg_fused = self.history_fusion_linear(fg_fused).permute(0, 2, 1)  # [bs, c_, K]

        # # 背景融合（原有代码）
        # bg_history_feat = torch.gather(history_bev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, mc, 1))  # [bs, mc//2, bg_k]
        # bg_curr_feat = torch.gather(curr_bev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, c_, 1))  # [bs, c_, bg_k]

        # bg_history_feat_time = bg_history_feat.reshape(bs, self.history_num, c_, -1)  # [bs, T, c_//2, bg_k]
        # # bg_history_agg = (bg_history_feat_time * time_weights).sum(dim=1)  # [bs, c_//2, bg_k]
        # bg_V_prev = torch.gather(V_prev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, self.history_num, 1))  # [bs, bg_k]
        # bg_V_curr = torch.gather(V_curr_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, 1, 1))  # [bs, bg_k]
        # #TODO 这个10的超参数？ 调整成可学习？
        # # bg_time_vis_weights = bg_V_prev/(bg_V_prev.sum(dim=1).unsqueeze(1) +1e-10) # [bs, 4,K]
        # bg_time_vis_weights =(bg_V_prev*(self.bg_scale.view(1,self.history_num,1))).softmax(dim=1)
        # bg_history_agg = (bg_history_feat_time * bg_time_vis_weights.unsqueeze(2)).sum(dim=1)  # [bs, c_, K]).sum(dim=1)  # [bs, c_, K]]



        # # bg_history_agg_perm = F.pad(bg_history_agg.permute(0, 2, 1), (0, c_ - c_//2, 0, 0))  # [bs, bg_k, c_]

        # bg_history_agg_perm = bg_history_agg.permute(0, 2, 1)  # [bs, bg_k, c_]
        # # bg_V_prev_time = bg_V_prev.unsqueeze(1).repeat(1, self.history_num, 1)  # [bs, T, bg_k]
        # # bg_V_prev_agg = (bg_V_prev_time * time_weights.squeeze(-1)).sum(dim=1)  # [bs, bg_k]
        # bg_V_prev_agg = bg_V_prev.max(dim=1)[0]  # [bs, bg_k]
        # bg_w_hist, bg_w_curr = self.compute_gate_weights(bg_V_prev_agg, bg_V_curr.squeeze(1))  # [bs, bg_k, 1]

        # bg_curr_perm = bg_curr_feat.permute(0, 2, 1)  # [bs, bg_k, c_]
        # # print("*"*50)
        # # print("print(bg_w_hist.shape)")
        # # print(bg_w_hist.shape)
        # # print("print(bg_w_curr.shape)")
        # # print(bg_w_curr.shape)
        # # print("print(bg_history_agg_perm.shape)")
        # # print(bg_history_agg_perm.shape)
        # # print("print(bg_curr_perm.shape)")        
        # # print(bg_curr_perm.shape)

        # # # 断言批次大小一致
        # # assert bg_w_hist.shape[0] == bg_history_agg_perm.shape[0] == bg_w_curr.shape[0] == bg_curr_perm.shape[0], \
        # #     f"批次大小不匹配: {bg_w_hist.shape[0]}, {bg_history_agg_perm.shape[0]}, {bg_w_curr.shape[0]}, {bg_curr_perm.shape[0]}"

        # # # 断言第二维度（bg_k）一致
        # # assert bg_w_hist.shape[1] == bg_history_agg_perm.shape[1] == bg_w_curr.shape[1] == bg_curr_perm.shape[1], \
        # #     f"bg_k维度不匹配: {bg_w_hist.shape[1]}, {bg_history_agg_perm.shape[1]}, {bg_w_curr.shape[1]}, {bg_curr_perm.shape[1]}"

        # # # 断言第三维度（c_）匹配（bg_w_hist和bg_w_curr的第三维为1，不影响广播）
        # # assert bg_history_agg_perm.shape[2] == bg_curr_perm.shape[2], \
        # #     f"特征维度c_不匹配: {bg_history_agg_perm.shape[2]} vs {bg_curr_perm.shape[2]}"

        # # print("bg_w_hist dtype:", bg_w_hist.dtype)
        # # print("bg_history_agg_perm dtype:", bg_history_agg_perm.dtype)
        # # print("bg_w_curr dtype:", bg_w_curr.dtype)
        # # print("bg_curr_perm dtype:", bg_curr_perm.dtype)


        # # print("bg_w_hist device:", bg_w_hist.device)
        # # print("bg_history_agg_perm device:", bg_history_agg_perm.device)
        # # print("bg_w_curr device:", bg_w_curr.device)
        # # print("bg_curr_perm device:", bg_curr_perm.device)


        # bg_fused = bg_w_hist * bg_history_agg_perm + bg_w_curr * bg_curr_perm  # [bs, bg_k, c_]
        # # 先验证乘法是否正常
        # # temp1 = bg_w_hist * bg_history_agg_perm
        # # temp2 = bg_w_curr * bg_curr_perm
        # # 再验证加法是否正常

        # # bg_w_hist = bg_w_hist.contiguous()
        # # bg_history_agg_perm = bg_history_agg_perm.contiguous()
        # # bg_w_curr = bg_w_curr.contiguous()
        # # bg_curr_perm = bg_curr_perm.contiguous()

        # # # 重新计算
        # # temp1 = bg_w_hist * bg_history_agg_perm
        # # temp2 = bg_w_curr * bg_curr_perm
        # # temp1 = temp1.contiguous()
        # # temp2 = temp2.contiguous()
        # # bg_fused = temp1 + temp2


        # # 转移所有张量到CPU
        # # bg_w_hist_cpu = bg_w_hist.cpu()
        # # bg_history_agg_perm_cpu = bg_history_agg_perm.cpu()
        # # bg_w_curr_cpu = bg_w_curr.cpu()
        # # bg_curr_perm_cpu = bg_curr_perm.cpu()

        # # # 分步执行运算
        # # try:
        # #     temp1_cpu = bg_w_hist_cpu * bg_history_agg_perm_cpu
        # #     temp2_cpu = bg_w_curr_cpu * bg_curr_perm_cpu
        # #     bg_fused_cpu = temp1_cpu + temp2_cpu
        # #     print(bg_fused_cpu)
        # #     print(bg_fused_cpu.shape)
        # #     print("CPU运算成功，无明显错误")
        # # except Exception as e:
        # #     print(f"CPU运算报错：{e}")  # 此处会显示具体错误原因（如值异常）


        # # bg_fused = temp1.clone() + temp2.clone()
        # # 1/0
        # bg_occ_feat = torch.gather(last_occ_reshaped, dim=1, index=bg_indices.unsqueeze(-1).repeat(1, 1, last_occ_pred.shape[-1]))  # [bs, bg_k, num_classes]
        # bg_occ_embed = self.occ_embedding(bg_occ_feat).permute(0, 2, 1)  # [bs, occ_embedims, bg_k]
        # bg_fused = torch.cat([bg_fused, bg_occ_embed.permute(0, 2, 1)], dim=-1)  # [bs, bg_k, c_+occ_embedims]
        # bg_fused = self.history_fusion_bg_linear(bg_fused).permute(0, 2, 1)  # [bs, c_, bg_k]

        # 更新当前BEV
        curr_bev_updated = curr_bev_flat.clone()
        curr_bev_updated.scatter_add_(dim=2, index=fg_indices.unsqueeze(1).repeat(1, c_, 1), src=fg_fused)
        curr_bev_updated.scatter_add_(dim=2, index=bg_indices.unsqueeze(1).repeat(1, c_, 1), src=bg_fused)
        curr_bev_updated = curr_bev_updated.reshape(bs, c_, z, h, w)  # 恢复原形状

        # 更新历史缓存
        self.history_last_bev = curr_bev_updated.detach().clone()
        self.history_bev = torch.cat([curr_bev,sampled_history_bev[:, :-c_, ...]], dim=1).detach()
        # self.history_cam_intrins = torch.cat([curr_cam_intrins.unsqueeze(1),self.history_cam_intrins[:, :-1, ...]], dim=1).detach()
        # self.history_cam_extrins = torch.cat([curr_cam_extrins.unsqueeze(1),self.history_cam_extrins[:, 1-1:, ...]], dim=1).detach()
        self.history_visibility =torch.cat([V_curr, V_prev[:, :-1, ...]],dim=1).detach()
        self.history_forward_augs = forward_augs.clone()

        return curr_bev_updated

    def generate_grid(self, curr_bev):
        n, c_, z, h, w = curr_bev.shape
        # Generate grid
        xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w, 1).expand(h, w, z)
        ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1, 1).expand(h, w, z)
        zs = torch.linspace(0, z - 1, z, dtype=curr_bev.dtype, device=curr_bev.device).view(1, 1, z).expand(h, w, z)
        grid = torch.stack((xs, ys, zs, torch.ones_like(xs)), -1).view(1, h, w, z, 4).expand(n, h, w, z, 4).view(n, h,w, z, 4, 1)
        return grid

    def generate_feat2bev(self, grid, dx, bx):
        feat2bev = torch.zeros((4, 4), dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = dx[0]
        feat2bev[1, 1] = dx[1]
        feat2bev[2, 2] = dx[2]
        feat2bev[0, 3] = bx[0] - dx[0] / 2.
        feat2bev[1, 3] = bx[1] - dx[1] / 2.
        feat2bev[2, 3] = bx[2] - dx[2] / 2.
        feat2bev[3, 3] = 1
        feat2bev = feat2bev.view(1, 4, 4)
        return feat2bev



@HEADS.register_module()
class GatedTemporalFusion6_not(BaseModule):
    def __init__(
        self,
        history_num=4,
        depth_sampler_embed_dims=256,
        depth_sampler_num_heads=8,
        depth_sampler_num_levels=4,
        depth_sampler_num_points=4,
        im2col_step=64,
        top_k=500,
        single_bev_num_channels=96,
        occ_embedims=32,
        num_classes=18,
        vis_theta=0.28,
        vis_beta=10.0,
        vis_gamma=0.4,
        vis_sigma=0.1,
        nonempty_thresh=0.1,
        max_step_ratio=1.2,
        **kwargs
    ):
        super(GatedTemporalFusion6_not, self).__init__()

        # 基础参数
        self.history_num = history_num
        self.top_k = top_k
        self.single_bev_num_channels = single_bev_num_channels
        self.occ_embedims = occ_embedims
        self.fg_scale = nn.Parameter(torch.full((history_num,), 10.0))
        self.bg_scale = nn.Parameter(torch.full((history_num,), 10.0))
        # 可见性门控参数
        # self.vis_theta = vis_theta  # 可见性阈值
        # self.vis_beta = vis_beta    # 敏感度参数
        # self.vis_gamma = vis_gamma  # 场景4固定权重
        # self.vis_sigma = vis_sigma  # 软化参数
        self.vis_theta = nn.Parameter(torch.tensor(vis_theta))   # 可见性阈值
        self.vis_beta = nn.Parameter(torch.tensor(vis_beta))     # 敏感度参数
        self.vis_gamma = nn.Parameter(torch.tensor(vis_gamma))   # 场景4固定权重
        self.vis_sigma = nn.Parameter(torch.tensor(vis_sigma))  

        
        # 历史特征缓存（新增history_last_bev）
        self.history_bev = None  # 缓存历史多帧BEV特征
        self.history_last_bev = None  # 缓存上一帧最终融合后的BEV特征（关键新增）
        self.history_cam_intrins = None
        self.history_cam_extrins = None
        
        # 网络层（保持不变）
        self.occ_embedding = nn.Sequential(
            nn.Linear(num_classes, occ_embedims),
            nn.Softplus(),
            nn.Linear(occ_embedims, occ_embedims),
        )
        self.history_fusion_linear = nn.Sequential(
            nn.Linear(single_bev_num_channels + occ_embedims, single_bev_num_channels),
            nn.Softplus(),
            nn.Linear(single_bev_num_channels, single_bev_num_channels),
        )
        self.history_fusion_bg_linear = nn.Sequential(
            nn.Linear(single_bev_num_channels + occ_embedims, single_bev_num_channels),
            nn.Softplus(),
            nn.Linear(single_bev_num_channels, single_bev_num_channels),
        )
        
        # 可见性计算组件（保持不变）
        self.rt_vis_calculator = EfficientRayTracingVisibility(
            nonempty_thresh=nonempty_thresh,
            max_step_ratio=max_step_ratio
        )
        # self.img_shape = (900, 1600)  # 默认图像尺寸

        # self.depth_sampler = DeformableDepthSampler(
        #     embed_dims=depth_sampler_embed_dims,
        #     num_heads=depth_sampler_num_heads,
        #     num_levels=depth_sampler_num_levels,
        #     num_points=depth_sampler_num_points
        # )
        self.history_forward_augs = None  # 用于缓存历史帧的变换矩阵（如BDAM矩阵）
        self.im2col_step = im2col_step
        # self.voxel_encoder = nn.Linear(single_bev_num_channels, depth_sampler_embed_dims)
        self.dbound = [1.0, 45.0, 0.5]
        self.pc_range = [-40, -40, -1.0, 40, 40, 5.4]
        self.final_dim = (256, 704)
    def compute_visibility(self, grid, cam_intrins, cam_extrins, img_shape, img_feats, spatial_shapes):
        """
        升级：结合可变形注意力采样的深度值优化可见性计算
        Args:
            img_feats: 多尺度图像特征 [bs, num_levels, c, h, w]
            spatial_shapes: 图像特征尺度 [num_levels, 2] (h, w)
            其他参数同原函数
        Returns:
            vis_prob: [bs, h, w, z] 优化后的可见性概率
        """
        bs, h, w, z, _ = grid.shape
        h_img, w_img = img_shape
        device = grid.device
        num_voxels = h * w * z  # 体素总数

        # 1. 原有相机投影逻辑（计算图像坐标和初始可见性）
        # 1.1 体素坐标→相机坐标→图像坐标
        grid_cam = grid.unsqueeze(1).expand(bs, self.num_cams, h, w, z, 3)  # [bs, num_cams, h, w, z, 3]
        grid_flat = grid_cam.reshape(-1, num_voxels, 3)  # [bs*num_cams, N, 3]
        grid_hom = torch.cat([grid_flat, torch.ones_like(grid_flat[..., :1])], dim=-1)  # [bs*num_cams, N, 4]

        cam_intrins_flat = cam_intrins.reshape(-1, 4, 4)  # [bs*num_cams, 4, 4]
        cam_extrins_flat = cam_extrins.reshape(-1, 4, 4)  # [bs*num_cams, 4, 4]
        extrins_inv = torch.inverse(cam_extrins_flat)
        cam_coords = torch.bmm(extrins_inv[:, :3, :4], grid_hom.permute(0, 2, 1)).permute(0, 2, 1)  # [bs*num_cams, N, 3]
        depth = cam_coords[..., 2:3] + 1e-8  # 相机坐标系下的深度

        # 1.2 计算初始可见性（原逻辑）
        img_coords = torch.bmm(cam_intrins_flat[:, :3, :3], cam_coords.permute(0, 2, 1)).permute(0, 2, 1)  # [bs*num_cams, N, 3]
        img_xy = img_coords[..., :2] / depth  # [bs*num_cams, N, 2] (u, v)
        depth_valid = (cam_coords[..., 2] > 0).float()  # [bs*num_cams, N]
        u_valid = (img_xy[..., 0] >= 0) & (img_xy[..., 0] < w_img)
        v_valid = (img_xy[..., 1] >= 0) & (img_xy[..., 1] < h_img)
        img_valid = (u_valid & v_valid).float()  # [bs*num_cams, N]
        initial_vis = depth_valid * img_valid  # [bs*num_cams, N]

        # 2. 可变形注意力深度采样
        # 2.1 准备输入：体素特征编码
        # voxel_feat = self.voxel_encoder(self.curr_bev_feat)  # [bs, c_, z, h, w] → [bs, z*h*w, embed_dims]（需提前展平体素特征）
        # voxel_feat = voxel_feat.reshape(bs, num_voxels, -1)  # [bs, N, embed_dims]

        # 2.2 生成参考点（归一化到[0,1]）
        norm_img_xy = img_xy / torch.tensor([w_img, h_img], device=device).view(1, 1, 2)  # [bs*num_cams, N, 2]
        # 取主相机（如第0个相机）的参考点作为采样基准
        ref_points = norm_img_xy.reshape(bs, self.num_cams, num_voxels, 2)[:, 0]  # [bs, N, 2]
        ref_points = ref_points.unsqueeze(2).repeat(1, 1, self.depth_sampler.num_levels, 1)  # [bs, N, L, 2]

        # 2.3 采样深度特征
        sampled_depth = self.depth_sampler(
            query=voxel_feat,
            value=img_feats,  # 多尺度图像特征 [bs, L, c, h, w]
            reference_points=ref_points,
            spatial_shapes=spatial_shapes
        )  # [bs, N]

        # 3. 结合采样深度优化可见性
        # 3.1 深度一致性校验：采样深度与相机投影深度的差异
        cam_depth = depth.reshape(bs, self.num_cams, num_voxels)[:, 0]  # 主相机的投影深度 [bs, N]
        depth_diff = torch.abs(sampled_depth - cam_depth) / (cam_depth + 1e-8)  # 相对深度差
        depth_consistent = (depth_diff < 0.3).float()  # 深度差小于30%则认为有效

        # 3.2 融合可见性：初始可见性 × 深度一致性
        initial_vis = initial_vis.reshape(bs, self.num_cams, num_voxels).max(dim=1)[0]  # [bs, N]（多相机取max）
        vis_prob = initial_vis * depth_consistent  # [bs, N]

        # 4. 还原形状
        return vis_prob.reshape(bs, h, w, z)  # [bs, h, w, z]

    def compute_alpha_unified(self, V_curr, V_prev):
        """统一计算当前帧融合权重α（覆盖四场景）"""
        eps = 1e-8
        # 1. 基础动态权重σ_base（场景1-3）
        ratio = V_curr / (V_curr + V_prev + eps)
        sigma_base = torch.sigmoid(self.vis_beta * (ratio - 0.5))
        
        # 2. 场景4软化掩码σ_both
        mask_curr = torch.sigmoid(-(V_curr - self.vis_theta) / self.vis_sigma)
        mask_prev = torch.sigmoid(-(V_prev - self.vis_theta) / self.vis_sigma)
        sigma_both = mask_curr * mask_prev
        
        # 3. 最终权重计算
        alpha = (1 - sigma_both) * sigma_base + sigma_both * self.vis_gamma
        return alpha.unsqueeze(-1)  # [bs, N, 1]

    def compute_gate_weights(self, V_prev_agg, V_curr):
        """计算历史和当前帧的门控权重"""
        alpha = self.compute_alpha_unified(V_curr, V_prev_agg)
        return 1 - alpha, alpha  # 历史权重，当前权重

    def generate_grid(self, curr_bev, voxel_min, voxel_max, voxel_size):
        """生成体素中心坐标网格（自车坐标系）"""
        bs, c_, z, h, w = curr_bev.shape
        device = curr_bev.device
        
        # 计算体素中心坐标
        x_coords = torch.linspace(
            voxel_min[0] + voxel_size[0]/2, 
            voxel_max[0] - voxel_size[0]/2, 
            w, device=device
        )
        y_coords = torch.linspace(
            voxel_min[1] + voxel_size[1]/2, 
            voxel_max[1] - voxel_size[1]/2, 
            h, device=device
        )
        z_coords = torch.linspace(
            voxel_min[2] + voxel_size[2]/2, 
            voxel_max[2] - voxel_size[2]/2, 
            z, device=device
        )
        
        # 生成网格并扩展维度 [bs, h, w, z, 3]
        x_grid, y_grid, z_grid = torch.meshgrid(x_coords, y_coords, z_coords, indexing='xy')
        grid = torch.stack([x_grid, y_grid, z_grid], dim=-1)  # [w, h, z, 3]
        grid = grid.permute(1, 0, 2, 3)  # [h, w, z, 3]
        return grid.unsqueeze(0).repeat(bs, 1, 1, 1, 1)  # [bs, h, w, z, 3]


    def get_reference_points(self, H, W, Z=None, num_points_in_pillar =4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
        torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    @force_fp32(apply_to=('reference_points', 'cam_params'))
    def point_sampling(self, reference_points, pc_range, img_metas, cam_params=None):
        # prepare for point sampling
        lidar2img = []
        ego2lidar = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])     # lidar2img update the post aug in the loading pipeline
            ego2lidar.append(img_meta['ego2lidar'])
        lidar2img = torch.stack(lidar2img, dim=0).to(reference_points.device)
        ego2lidar = torch.stack(ego2lidar, dim=0).to(reference_points.device)

        sensor2egos, ego2globals, intrins, post_augs, bda_mat = cam_params
        num_cam = sensor2egos.size(1)
        ogfH, ogfW = self.final_dim

        # reference_points defines in the bev space, [bs, D, hxw, 3]
        # change reference_points from bev-ego coordinate to ego coordinate
        reference_points = reference_points.clone()
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
                                     (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
                                     (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
                                     (pc_range[5] - pc_range[2]) + pc_range[2]

        # prepare for point sampling
        reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
        reference_points = reference_points.permute(1, 0, 2, 3)  # shape: (num_points_in_pillar,bs,h*w,4)
        D, B, num_query = reference_points.size()[:3]  # D=num_points_in_pillar , num_query=h*w
        reference_points = reference_points.view(D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)  # shape: (num_points_in_pillar,bs,num_cam,h*w,4)
        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)
        ego2lidar = ego2lidar.view(1, B, 1, 1, 4, 4).repeat(D, 1, num_cam, num_query, 1, 1)
        inverse_bda = bda_mat.view(1, B, 1, 1, 4, 4).repeat(D, 1, num_cam, num_query, 1, 1)

        # change reference_points from ego coordinate to img coordinate
        eps = 1e-5
        reference_points_cam = (lidar2img @ ego2lidar @ inverse_bda @ reference_points).squeeze(-1)   # [num_points_in_pillar, bs, num_cam, num_query=h*w, 4]
        reference_points_depth = reference_points_cam[..., 2:3]
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(reference_points_depth, torch.ones_like(reference_points_depth) * eps)

        # Bug!!
        # Correct normalize is
        # reference_points_cam[..., 0] /= ogfW
        # reference_points_cam[..., 1] /= ogfH
        # But for reproducing our results, we use the following normalization
        reference_points_cam[..., 0] /= ogfW
        reference_points_cam[..., 1] /= ogfH

        bev_mask = (reference_points_depth > eps)
        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)                  # shape: (num_cam, bs,h*w, num_points_in_pillar, 2)
        reference_points_depth = reference_points_depth.permute(2, 1, 3, 0, 4)              # shape: (num_cam, bs,h*w, num_points_in_pillar, 1)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)                        # shape: (num_cam, bs,h*w, num_points_in_pillar)

        return reference_points_cam, reference_points_depth, bev_mask


    @force_fp32()
    def forward(self, curr_bev, cam_params, history_fusion_params, dx, bx, history_last_bev=None, last_occ_pred=None, nonempty_prob=None, img_feats=None, spatial_shapes=None,pred_img_depth=None,**kwargs):
        """
        Args:
            curr_bev: [bs, c, z, h, w] 当前帧BEV特征
            cam_params: 相机参数列表，包含外参、内参等
            history_fusion_params: 历史融合参数（包含序列信息等）
            dx: 体素尺寸 (x, y, z)
            bx: 体素偏移
            nonempty_prob: [bs, z, h, w] 体素非空概率
            last_occ_pred: [bs, z, h, w, num_classes] 上一时刻 occupancy 预测
            img_feats: 多尺度图像特征 [bs, num_levels, c, h, w]（新增，用于深度采样）
            spatial_shapes: 图像特征尺度 [num_levels, 2] (h, w)（新增，用于深度采样）
        Returns:
            curr_bev_updated: [bs, c, z, h, w] 融合后BEV特征
        """
        # print(self.history_num)
        # print("print(self.fg_scale)")
        # print(self.fg_scale) 
        # print("print(self.bg_scale)")
        # print(self.bg_scale)
        # # 可见性门控参数
        # # self.vis_theta = vis_theta  # 可见性阈值
        # # self.vis_beta = vis_beta    # 敏感度参数
        # # self.vis_gamma = vis_gamma  # 场景4固定权重
        # # self.vis_sigma = vis_sigma  # 软化参数
        # print("print(self.vis_theta # 可见性阈值)")
        # print(self.vis_theta )  # 可见性阈值
        # print("print(self.vis_beta) # 敏感度参数")
        # print(self.vis_beta)      # 敏感度参数
        # print("print(self.vis_gamma) # 场景4固定权重")
        # print(self.vis_gamma)  # 场景4固定权重
        # print("print(self.vis_sigma) # 软化参数")
        # print(self.vis_sigma) 
        if torch.rand(1).item() < 1/2000:
            print(self.history_num)
            print("print(self.fg_scale)")
            print(self.fg_scale) 
            print("print(self.bg_scale)")
            print(self.bg_scale)
            # 可见性门控参数
            print("print(self.vis_theta # 可见性阈值)")
            print(self.vis_theta)  # 可见性阈值
            print("print(self.vis_beta) # 敏感度参数")
            print(self.vis_beta)      # 敏感度参数
            print("print(self.vis_gamma) # 场景4固定权重")
            print(self.vis_gamma)  # 场景4固定权重
            print("print(self.vis_sigma) # 软化参数")
            print(self.vis_sigma)  


        # -------------------------- 1. 解析参数后打印核心形状 --------------------------
        # 解析相机参数
        curr_cam_extrins = cam_params[0]  # [bs, num_cams, 4, 4]
        curr_cam_intrins = cam_params[2]  # [bs, num_cams, 4, 4]
        forward_augs = cam_params[4]      # [bs, 4, 4] 前向变换矩阵
        self.num_cams = curr_cam_extrins.shape[1]  # 从外参中获取相机数量
        bs, c_, z, h, w = curr_bev.shape
        device = curr_bev.device
        mc = self.history_num * c_        # 历史特征总通道数
        # self.history_forward_augs = forward_augs.clone()


        ref_3d = self.get_reference_points(
            h, w, z, z, dim='3d', bs=bs, device=device, dtype=curr_bev.dtype) # torch.Size([3, 2, 625, 3]) #[bs,z,yx,3(x,y,z)]
        # ref_2d = self.get_reference_points(
        #     h, w, dim='2d', bs=bs, device=device, dtype=curr_bev.dtype) #torch.Size([3, 625, 1, 2])
        slots = torch.zeros(list([ref_3d.shape[0],ref_3d.shape[2],ref_3d.shape[1]])).to(ref_3d)
        reference_points_cam, reference_points_depth, bev_mask = self.point_sampling(ref_3d, self.pc_range, img_metas=kwargs['img_metas'], cam_params=cam_params)
        indexes = [[] for _ in range(bs)]
        spatial_shapes =[]
        spatial_shapes.append([16, 44])
        spatial_shapes = torch.tensor(spatial_shapes).to(device)
        pred_img_depth = pred_img_depth.view(bs * 6, -1, spatial_shapes[0][0], spatial_shapes[0][1])
        pred_img_depth = pred_img_depth.flatten(2).permute(0, 2, 1)  
        max_len = 0
        for j in range(bs):
            for i, mask_per_img in enumerate(bev_mask):
                index_query_per_img = mask_per_img[j].sum(-1).nonzero().squeeze(-1)
                if len(index_query_per_img) == 0:
                    index_query_per_img = bev_mask[i][j].sum(-1).nonzero().squeeze(-1)[0:1]
                indexes[j].append(index_query_per_img)
                # for batch operation, we need to pad the indexes to the same length
                max_len = max(max_len, len(index_query_per_img))
        reference_points_cam_rebatch = reference_points_cam.new_zeros([bs, self.num_cams, max_len, z, 2])
        reference_points_depth_rebatch = reference_points_depth.new_zeros([bs, self.num_cams, max_len, z, 1])

        for j in range(bs):
            for i, (reference_points_per_img, reference_points_depth_per_img) in enumerate(zip(reference_points_cam, reference_points_depth)):
                index_query_per_img = indexes[j][i]
                reference_points_cam_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]
                reference_points_depth_rebatch[j, i, :len(index_query_per_img)] = reference_points_depth_per_img[j, index_query_per_img]

        #use deformble attn
        depth_reference_points = reference_points_cam_rebatch.reshape(bs*6, max_len*z, 1, 1, 1, 2).contiguous()
        depth_attention_weights = torch.ones_like(depth_reference_points[..., 0]).contiguous()
        pred_img_depth = pred_img_depth.unsqueeze(2).contiguous()
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))



        bev_query_depth_rebatch = (reference_points_depth_rebatch- self.dbound[0])/ self.dbound[2]
        bev_query_depth_rebatch = torch.clip(torch.floor(bev_query_depth_rebatch), 0, 88-1).to(torch.long)
        bev_query_depth_rebatch = F.one_hot(bev_query_depth_rebatch.squeeze(-1),
                                   num_classes=88)

        depth_output = MultiScaleDeformableAttnFunction_fp32.apply(pred_img_depth, spatial_shapes,level_start_index,depth_reference_points,depth_attention_weights, self.im2col_step)
        depth_output = depth_output.reshape(bs,6, max_len,z, -1)   # [bs*num_cam, num_query, num_Z_anchors, C]
        # reference_points_depth_rebatch

        increment = torch.zeros_like(depth_output)
        # increment[..., 0] = 1e-9  # 非原地赋值（创建新张量）
        # depth_output = depth_output + increment
        depth_output = depth_output + torch.cat([(torch.zeros_like(depth_output[...,:1]) + 1e-9),torch.zeros_like(depth_output[...,1:])],dim=-1)

        # print("depth_output.sum(-1).shape")
        # print(depth_output.sum(-1).shape)
        # print("depth_output.sum(-1)")
        # print(depth_output.sum(-1))
        # print("print((depth_output.sum(-1)==0).sum())")
        # print((depth_output.sum(-1)==0).sum())
        depth_output =depth_output/depth_output.sum(-1)[...,None] #bs,xy,z,D

        # print("depth_output.sum(-1).shape")
        # print(depth_output.sum(-1).shape)
        # print("depth_output.sum(-1)")
        # print(depth_output.sum(-1))
        # print("print((depth_output.sum(-1)==1).sum())")
        # print((depth_output.sum(-1)>=0.99).sum())
        # print("depth_output")
        # print(depth_output)


        depth_output = (1-depth_output.cumsum(dim=-1))


        # print("print((depth_output[...,-1]==0).sum())")
        # print((depth_output[...,-1]==0).sum())
        depth_output = (bev_query_depth_rebatch*depth_output).sum(-1)

        #恢复depth_output的shape
        # fix_depth_output = depth_output.new_zeros([bs, 6, w*h, z, 88])
        # for j in range(bs):
        #     for i in range(6):
        #         index_query_per_img = indexes[j][i]
        #         fix_depth_output[j, i, index_query_per_img] = depth_output[j, i, :len(index_query_per_img)]

        for j in range(bs):
            for i in range(6):
                index_query_per_img = indexes[j][i]
                slots[j, index_query_per_img] = torch.max(slots[j, index_query_per_img],depth_output[j, i, :len(index_query_per_img)])

        # depth_sum = fix_depth_output.sum(dim=-1).view(3,6,25,25,2)
        # bs = depth_sum.shape[0]       # 3
        # num_cam = depth_sum.shape[1]  # 6
        # z_layers = depth_sum.shape[4] # 2
        # H, W = depth_sum.shape[2], depth_sum.shape[3]  # 25,25

        # # 颜色映射：0值用黑色，非0值用渐变色
        # cmap = plt.cm.viridis
        # cmap.set_bad(color='black')  # 0值标记为黑色

        # # 设置子图布局（不变）
        # fig, axes = plt.subplots(
        #     nrows=bs, ncols=num_cam * z_layers,
        #     figsize=(30, 8),
        #     squeeze=False
        # )
        # fig.suptitle('fix_depth_output.sum(-1) 可视化（黑色=0值，颜色=非0值）', fontsize=16, y=0.98)


        # # -------------------------- 2. 循环绘制每个子图（核心修改：添加 .detach()） --------------------------
        # for batch_idx in range(bs):
        #     for cam_idx in range(num_cam):
        #         for z_idx in range(z_layers):
        #             col_idx = cam_idx * z_layers + z_idx
        #             ax = axes[batch_idx, col_idx]
                    
        #             # 核心修改：先 detach() 切断计算图，再转 cpu 和 numpy
        #             data = depth_sum[batch_idx, cam_idx, :, :, z_idx].detach().cpu().numpy()
        #             data[data == 0] = np.nan  # 0值替换为NaN，显示为黑色
                    
        #             # 绘制热力图（修改 vmin/vmax：同样添加 .detach()）
        #             im = ax.imshow(
        #                 data, 
        #                 cmap=cmap, 
        #                 aspect='auto',
        #                 # 关键修改：depth_sum 先 detach 再转 numpy，确保不影响梯度
        #                 vmin=np.nanmin(depth_sum.detach().cpu().numpy()),
        #                 vmax=np.nanmax(depth_sum.detach().cpu().numpy())
        #             )
                    
        #             # 子图标题和坐标轴（不变）
        #             ax.set_title(
        #                 f'Batch{batch_idx+1}\nCam{cam_idx+1} Z{z_idx+1}',
        #                 fontsize=10, pad=5
        #             )
        #             ax.set_xticks([])
        #             ax.set_yticks([])


        # # -------------------------- 3. 添加颜色条（不变） --------------------------
        # cbar = fig.colorbar(
        #     im, 
        #     ax=axes.ravel().tolist(),
        #     shrink=0.8,
        #     pad=0.02
        # )
        # cbar.set_label('Sum of Depth Bins (D=88)', fontsize=12)


        # # -------------------------- 4. 调整布局并保存（不变） --------------------------
        # plt.tight_layout(rect=[0, 0, 0.98, 0.95])
        # plt.savefig('depth_sum_visualization.png', dpi=300, bbox_inches='tight')
        # plt.show()

        # output

        #计数更新，建立在纸上的假设成立的基础上
        # count = bev_mask.sum(-1) > 0
        # count = count.permute(1, 2, 0).sum(-1)
        # count = torch.clamp(count, min=1.0)
        # slots = slots / count[..., None]

        # print("slots.shape")
        # print(slots.shape)
        # print("slots")  
        # print(slots)
        # zzzzz=1/0

        V_curr =slots.view(bs, 1, h, w,z).permute(0, 1, 4, 2, 3)
        # slots[...,0]+=1e-9
        # print("slots.sum(-1).shape")
        # print(slots.sum(-1).shape)
        # print("slots.sum(-1)")
        # print(slots.sum(-1))
        # print("print((slots.sum(-1)==0).sum())")
        # print((slots.sum(-1)==0).sum())
        # slots =slots/slots.sum(-1)[...,None] #bs,xy,z,D

        # print("slots.sum(-1).shape")
        # print(slots.sum(-1).shape)
        # print("slots.sum(-1)")
        # print(slots.sum(-1))
        # print("print((slots.sum(-1)==1).sum())")
        # print("slots")
        # print(slots)


        # slots = (1-slots.cumsum(dim=-1))

        # print("print((slots[...,-1]==0).sum())")
        # print((slots[...,-1]==0).sum())

        # print("slots")
        # print(slots)
        # print("print((slots[...,-1]<0.01).sum())")
        # print((slots[...,-1]<0.01).sum())

        #TODO 这里的对于边界值的考虑，从0开始还是从1开始？




        # slots = self.output_proj(slots)
        #TODO 上面这里需要检查一下
        #这里相当于两次softmax，可能会导致分布变得不够尖锐，需要进一步确认

        #到这里slots就是可见性的概率分布了
        #这里先尝试使用期望值进行计算，使得可微分

        #然后再采用stc的原始离散计算方法，


        # 打印解析后关键变量形状
        # print("="*50)
        # print("1. 解析参数后核心变量形状：")
        # print(f"curr_bev: {curr_bev.shape} (预期：[bs, c, z, h, w])")
        # print(f"curr_cam_extrins: {curr_cam_extrins.shape} (预期：[bs, num_cams, 4, 4])")
        # print(f"curr_cam_intrins: {curr_cam_intrins.shape} (预期：[bs, num_cams, 4, 4])")
        # print(f"forward_augs: {forward_augs.shape} (预期：[bs, 4, 4])")
        # print(f"dx: {dx.shape if hasattr(dx, 'shape') else type(dx)} (预期：[3])")
        # print(f"bx: {bx.shape if hasattr(bx, 'shape') else type(bx)} (预期：[3])")
        # print(f"bs: {bs}, c_: {c_}, z: {z}, h: {h}, w: {w} (BEV特征维度)")
        # print("="*50)

        if type(history_fusion_params['sequence_group_idx']) is list:
            seq_ids = history_fusion_params['sequence_group_idx'][0]
        else:
            seq_ids = history_fusion_params['sequence_group_idx']
        if type(history_fusion_params['start_of_sequence']) is list:
            start_of_sequence = history_fusion_params['start_of_sequence'][0]
        else:
            start_of_sequence = history_fusion_params['start_of_sequence']
        if type(history_fusion_params['curr_to_prev_ego_rt']) is list:
            curr_to_prev_ego_rt = history_fusion_params['curr_to_prev_ego_rt'][0]
        else:
            curr_to_prev_ego_rt = history_fusion_params['curr_to_prev_ego_rt']
        forward_augs = cam_params[-1]  # bda

        # check seq_ids > 0
        assert (seq_ids >= 0).all()
        # -------------------------- 2. 初始化历史缓存后打印 --------------------------
        if self.history_bev is None:
            # self.history_bev = curr_bev.repeat(1, self.history_num, 1, 1, 1)  # [bs, mc, z, h, w]
            # self.history_cam_intrins = curr_cam_intrins.unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)  # [bs, T, num_cams, 4, 4]
            # self.history_cam_extrins = curr_cam_extrins.unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)  # [bs, T, num_cams, 4, 4]
            # self.history_bev = curr_bev.clone()
            self.history_forward_augs = forward_augs.clone()
            self.history_seq_ids = seq_ids.clone()
            self.history_bev = curr_bev.repeat(1, self.history_num, 1, 1, 1)
            self.history_sweep_time = curr_bev.new_zeros(curr_bev.shape[0], self.history_num)
            self.history_visibility = V_curr.repeat(1, self.history_num, 1, 1, 1).half()
        self.history_bev = self.history_bev.detach()
        self.history_visibility = self.history_visibility.detach().half()
        self.history_sweep_time += 1

        # 打印历史缓存形状
        # print("\n2. 历史缓存初始化后形状：")
        # print(f"history_bev: {self.history_bev.shape} (预期：[bs, mc, z, h, w]，mc={mc})")
        # print(f"history_cam_intrins: {self.history_cam_intrins.shape} (预期：[bs, history_num, num_cams, 4, 4])")
        # print(f"history_cam_extrins: {self.history_cam_extrins.shape} (预期：[bs, history_num, num_cams, 4, 4])")

        # -------------------------- 3. 生成网格和BEV变换矩阵后打印 --------------------------
        # 处理新序列（略，不影响维度）
        # start_of_sequence = history_fusion_params.get('start_of_sequence', torch.zeros(bs, dtype=torch.bool, device=device))
        if start_of_sequence.sum()>0:
            self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_num, 1, 1, 1)
            # self.history_cam_intrins[start_of_sequence] = curr_cam_intrins[start_of_sequence].unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)
            # self.history_cam_extrins[start_of_sequence] = curr_cam_extrins[start_of_sequence].unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)
            self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]
            self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
            self.history_sweep_time[start_of_sequence] = 0  # zero the new sequence timestep starts
            self.history_visibility[start_of_sequence] = V_curr[start_of_sequence].repeat(1, self.history_num, 1, 1, 1).half()

        # 生成体素网格和BEV变换矩阵
        grid = self.generate_grid(curr_bev) #[bs,y,x,z,4]
        grid_3d = grid
        feat2bev = self.generate_feat2bev(grid, dx, bx)

        # 打印网格和变换矩阵形状
        # print("\n3. 生成网格和BEV变换矩阵后形状：")
        # print(f"grid_3d (体素网格): {grid_3d.shape} (关键！预期：[bs, h, w, z, 3] 或 [bs, w, h, z, 3])")
        # print(f"feat2bev (BEV变换矩阵): {feat2bev.shape} (预期：[bs, 4, 4])")

        # -------------------------- 4. 运动补偿矩阵计算后打印 --------------------------
        # 获取帧间姿态变换
        # curr_to_prev_ego_rt = history_fusion_params.get('curr_to_prev_ego_rt', torch.eye(4, device=device).unsqueeze(0).repeat(bs, 1, 1))
        # 计算RT流（坐标变换矩阵）
        rt_flow = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_ego_rt @ torch.inverse(forward_augs) @ feat2bev)
        # 生成齐次网格
        # 在forward函数中，生成grid_hom的位置修正：
        # grid_3d = self.generate_grid(curr_bev)  # 现在形状：[3, 25, 25, 2, 3]（bs, h, w, z, 3）
        # # 生成齐次坐标（x,y,z,1），并添加最后一个维度（用于矩阵乘法）
        # grid_hom = torch.cat([
        #     grid_3d,  # [3,25,25,2,3]
        #     torch.ones_like(grid_3d[..., :1])  # [3,25,25,2,1]（补充1作为齐次坐标）
        # ], dim=-1).unsqueeze(-1)  # 最终形状：[3,25,25,2,4,1]（符合预期）
        # # 打印运动补偿相关形状（矩阵乘法前关键检查）
        # print("\n4. 运动补偿矩阵计算后形状（矩阵乘法前）：")
        # print(f"curr_to_prev_ego_rt (帧间姿态): {curr_to_prev_ego_rt.shape} (预期：[bs, 4, 4])")
        # print(f"rt_flow (变换流): {rt_flow.shape} (预期：[bs, 4, 4])")
        # print(f"grid_hom (齐次网格): {grid_hom.shape} (关键！预期：[bs, h, w, z, 4, 1]，需与rt_flow广播匹配)")
        # print(f"rt_flow.view后: {rt_flow.view(bs, 1, 1, 1, 4, 4).shape} (预期：[bs, 1, 1, 1, 4, 4])")

        # # -------------------------- 5. 网格变换后打印（解决之前维度错的核心） --------------------------
        # try:
        #     grid_transformed = rt_flow.view(bs, 1, 1, 1, 4, 4) @ grid_hom  # 矩阵乘法：[bs, h, w, z, 4, 1]
        #     print("\n5. 网格变换后形状（矩阵乘法成功！）：")
        #     print(f"grid_transformed: {grid_transformed.shape} (预期：[bs, h, w, z, 4, 1])")
        # except RuntimeError as e:
        #     print(f"\n5. 网格变换矩阵乘法报错！错误信息：{str(e)}")
        #     print(f"  - rt_flow.view形状: {rt_flow.view(bs, 1, 1, 1, 4, 4).shape}")
        #     print(f"  - grid_hom形状: {grid_hom.shape}")
        #     print("  提示：需确保grid_hom的第1-4维度与rt_flow.view的第2-5维度匹配（广播规则）")
        #     raise e  # 继续抛出错误，方便定位
        bs, mc, z, h, w = self.history_bev.shape
        n, c_, z, h, w = curr_bev.shape
        grid = rt_flow.view(n, 1, 1, 1, 4, 4) @ grid
        # -------------------------- 6. 采样网格生成后打印 --------------------------
        # 生成采样网格（归一化到[-1,1]，适配F.grid_sample）
        normalize_factor = torch.tensor([w - 1.0, h - 1.0, z - 1.0], dtype=curr_bev.dtype, device=device)
        # grid_sampler = grid_transformed[..., :3, 0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0  # [bs, h, w, z, 3]
        # # 调整采样网格维度（适配F.grid_sample输入：[bs, z, h, w, 3]）
        # grid_sampler_permuted = grid_sampler.permute(0, 3, 1, 2, 4)  # 交换z和h/w维度
        grid = grid[:, :, :, :, :3, 0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0   # grid order is x, y, z


        # print("\n6. 采样网格生成后形状：")
        # print(f"grid_sampler (归一化后): {grid_sampler.shape} (预期：[bs, h, w, z, 3])")
        # print(f"grid_sampler_permuted (适配采样): {grid_sampler_permuted.shape} (预期：[bs, z, h, w, 3])")

        # -------------------------- 7. 历史BEV采样后打印 --------------------------
        # 采样历史BEV特征
        sampled_history_bev = F.grid_sample(
            self.history_bev.reshape(bs, mc, z, h, w),  # 输入：[bs, mc, z, h, w]
            grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4),
            align_corners=True,
            mode='bilinear'
        )
        sampled_history_visibility = F.grid_sample(
            self.history_visibility.reshape(bs, self.history_num, z, h, w).half(),  # 输入：[bs, mc, z, h, w]
            grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4).half(),
            align_corners=True,
            mode='nearest'
        )
        # print("\n7. 历史BEV采样后形状：")
        # print(f"history_bev.reshape: {self.history_bev.reshape(bs, mc, z, h, w).shape} (预期：[bs, mc, z, h, w])")
        # print(f"sampled_history_bev: {sampled_history_bev.shape} (预期：[bs, mc, z, h, w])")

        # -------------------------- 8. 可见性计算后打印 --------------------------
        # 计算当前帧可见性
        # V_curr = self.compute_visibility(
        #     grid_3d, 
        #     cam_intrins=curr_cam_intrins,
        #     cam_extrins=curr_cam_extrins,
        #     img_shape=self.img_shape,
        #     img_feats=img_feats,
        #     spatial_shapes=spatial_shapes
        # )

        # print("V_curr (当前可见性).  "*3)
        # V_curr = slots
        # 计算历史帧可见性
        # prev_cam_intrins = self.history_cam_intrins[:, -1]
        # prev_cam_extrins = self.history_cam_extrins[:, -1]
        # V_prev = self.compute_visibility(
        #     grid_3d,
        #     cam_intrins=prev_cam_intrins,
        #     cam_extrins=prev_cam_extrins,
        #     img_shape=self.img_shape,
        #     img_feats=img_feats,
        #     spatial_shapes=spatial_shapes
        # )

        V_prev = sampled_history_visibility #bs,4,z,h,w

        # print("\n8. 可见性计算后形状：")
        # print(f"V_curr (当前可见性): {V_curr.shape} (预期：[bs, h, w, z])")
        # print(f"V_prev (历史可见性): {V_prev.shape} (预期：[bs, h, w, z])")

        # -------------------------- 9. 稀疏采样前展平变量打印 --------------------------
        # 展平变量（用于稀疏采样）
        curr_bev_flat = curr_bev.reshape(bs, c_, -1)  # [bs, c_, N], N=h*w*z
        history_bev_flat = sampled_history_bev.reshape(bs, mc, -1)  # [bs, mc, N]
        V_prev_flat = V_prev.reshape(bs,self.history_num, -1)  # [bs, 4,N]
        V_curr_flat = V_curr.reshape(bs, 1,-1)  # [bs, 1,N]
        nonempty_prob_flat = nonempty_prob.reshape(bs, -1)  # [bs, N]
        # print("print(nonempty_prob_flat.shape)")
        # print(nonempty_prob_flat.shape)
        total_voxels = nonempty_prob_flat.shape[1]

        # print("\n9. 稀疏采样前展平变量形状：")
        # print(f"curr_bev_flat: {curr_bev_flat.shape} (预期：[bs, c_, N], N={total_voxels})")
        # print(f"history_bev_flat: {history_bev_flat.shape} (预期：[bs, mc, N])")
        # print(f"nonempty_prob_flat: {nonempty_prob_flat.shape} (预期：[bs, N])")
        # print(f"total_voxels (h*w*z): {total_voxels} (预期：{h*w*z})")

        # -------------------------- 10. 前景/背景索引及融合后打印（可选，确认后续维度） --------------------------
        # 生成前景/背景索引
        fg_indices = torch.topk(nonempty_prob_flat, self.top_k, dim=1)[1]
        bg_indices = torch.topk(1 - nonempty_prob_flat, total_voxels - self.top_k, dim=1)[1]
        # 提取前景特征（示例，其他融合步骤类似）
        fg_history_feat = torch.gather(history_bev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, mc, 1))

        # print("\n10. 前景/背景索引及特征提取后形状：")
        # print(f"fg_indices (前景索引): {fg_indices.shape} (预期：[bs, top_k])")
        # print(f"bg_indices (背景索引): {bg_indices.shape} (预期：[bs, N-top_k])")
        # print(f"fg_history_feat (前景历史特征): {fg_history_feat.shape} (预期：[bs, mc, top_k])")
        # print("="*50)

        # -------------------------- 后续原有逻辑（略，维度已通过打印确认） --------------------------
        # 8. 前景融合（原有代码）
        # 9. 背景融合（原有代码）
        # 10. 更新当前BEV特征（原有代码）
        # curr_bev_flat = curr_bev.reshape(bs, c_, -1)  # [bs, c_, N] N=h*w*z
        # history_bev_flat = sampled_history_bev.reshape(bs, mc, -1)  # [bs, mc, N]
        # V_prev_flat = V_prev.reshape(bs, -1)  # [bs, N]
        # V_curr_flat = V_curr.reshape(bs, -1)  # [bs, N]
        # nonempty_prob_flat = nonempty_prob.reshape(bs, -1)  # [bs, N]
        # total_voxels = nonempty_prob_flat.shape[1]

        # fg_indices = torch.topk(nonempty_prob_flat, self.top_k, dim=1)[1]  # [bs, top_k]
        # bg_indices = torch.topk(1 - nonempty_prob_flat, total_voxels - self.top_k, dim=1)[1]  # [bs, N-top_k]

        # # 前景特征提取
        # fg_history_feat = torch.gather(history_bev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, mc, 1))  # [bs, mc, top_k]
        fg_curr_feat = torch.gather(curr_bev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, c_, 1))  # [bs, c_, top_k]

        # 历史特征时间聚合
        fg_history_feat_time = fg_history_feat.reshape(bs, self.history_num, c_, self.top_k)  # [bs, T, c_, K]
        #TODO 后续可以把time_weights也乘进去
        
        # time_weights = torch.exp(-0.5 * torch.arange(self.history_num, device=device)).view(1, self.history_num, 1, 1)

        fg_V_prev = torch.gather(V_prev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, self.history_num, 1))  # [bs, 4,K]
        fg_V_curr = torch.gather(V_curr_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, 1, 1))  # [bs, 1,K]
        # fg_time_vis_weights = fg_V_prev/(fg_V_prev.sum(dim=1).unsqueeze(1)+1e-10 ) # [bs, 4,K]
        fg_time_vis_weights = (fg_V_prev*(self.fg_scale.view(1,self.history_num,1))).softmax(dim=1)
        # print("print(fg_history_feat_time.shape)")
        # print(fg_history_feat_time.shape)
        # print("print(fg_time_vis_weights.shape)")
        # print(fg_time_vis_weights.shape)
        # fg_history_agg = (fg_history_feat_time * fg_time_vis_weights.unsqueeze(2)).sum(dim=1)  # [bs, c_, K]).sum(dim=1)  # [bs, c_, K]]
        fg_history_agg = (fg_history_feat_time * fg_time_vis_weights.unsqueeze(2)).sum(dim=1)  # [bs, c_, K]).sum(dim=1)  # [bs, c_, K]]


        # print("print(fg_history_agg.shape)")
        # print(fg_history_agg.shape)

        # 可见性聚合与门控
        
        # fg_V_prev_time = fg_V_prev.unsqueeze(1).repeat(1, self.history_num, 1)  # [bs, T, K]
        # fg_V_prev_agg = (fg_V_prev_time * time_weights.squeeze(-1)).sum(dim=1)  # [bs, K]
        fg_V_prev_agg = fg_V_prev.max(dim=1)[0]  # [bs, K]
        fg_w_hist, fg_w_curr = self.compute_gate_weights(fg_V_prev_agg, fg_V_curr.squeeze(1))  # [bs, K, 1]

        # 前景融合
        fg_history_agg_perm = fg_history_agg.permute(0, 2, 1)  # [bs, K, c_]
        fg_curr_perm = fg_curr_feat.permute(0, 2, 1)  # [bs, K, c_]
        fg_fused = fg_w_hist * fg_history_agg_perm + fg_w_curr * fg_curr_perm  # [bs, K, c_]

        # occupancy嵌入融合
        last_occ_reshaped = last_occ_pred.permute(0, 3, 2, 1, 4).reshape(bs, -1, last_occ_pred.shape[-1])  # [bs, N, num_classes]
        fg_occ_feat = torch.gather(last_occ_reshaped, dim=1, index=fg_indices.unsqueeze(-1).repeat(1, 1, last_occ_pred.shape[-1]))  # [bs, K, num_classes]
        fg_occ_embed = self.occ_embedding(fg_occ_feat).permute(0, 2, 1)  # [bs, occ_embedims, K]
        fg_fused = torch.cat([fg_fused, fg_occ_embed.permute(0, 2, 1)], dim=-1)  # [bs, K, c_+occ_embedims]
        fg_fused = self.history_fusion_linear(fg_fused).permute(0, 2, 1)  # [bs, c_, K]

        # 背景融合（原有代码）
        bg_history_feat = torch.gather(history_bev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, mc, 1))  # [bs, mc//2, bg_k]
        bg_curr_feat = torch.gather(curr_bev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, c_, 1))  # [bs, c_, bg_k]

        bg_history_feat_time = bg_history_feat.reshape(bs, self.history_num, c_, -1)  # [bs, T, c_//2, bg_k]
        # bg_history_agg = (bg_history_feat_time * time_weights).sum(dim=1)  # [bs, c_//2, bg_k]
        bg_V_prev = torch.gather(V_prev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, self.history_num, 1))  # [bs, bg_k]
        bg_V_curr = torch.gather(V_curr_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, 1, 1))  # [bs, bg_k]
        #TODO 这个10的超参数？ 调整成可学习？
        # bg_time_vis_weights = bg_V_prev/(bg_V_prev.sum(dim=1).unsqueeze(1) +1e-10) # [bs, 4,K]
        bg_time_vis_weights =(bg_V_prev*(self.bg_scale.view(1,self.history_num,1))).softmax(dim=1)
        bg_history_agg = (bg_history_feat_time * bg_time_vis_weights.unsqueeze(2)).sum(dim=1)  # [bs, c_, K]).sum(dim=1)  # [bs, c_, K]]



        # bg_history_agg_perm = F.pad(bg_history_agg.permute(0, 2, 1), (0, c_ - c_//2, 0, 0))  # [bs, bg_k, c_]

        bg_history_agg_perm = bg_history_agg.permute(0, 2, 1)  # [bs, bg_k, c_]
        # bg_V_prev_time = bg_V_prev.unsqueeze(1).repeat(1, self.history_num, 1)  # [bs, T, bg_k]
        # bg_V_prev_agg = (bg_V_prev_time * time_weights.squeeze(-1)).sum(dim=1)  # [bs, bg_k]
        bg_V_prev_agg = bg_V_prev.max(dim=1)[0]  # [bs, bg_k]
        bg_w_hist, bg_w_curr = self.compute_gate_weights(bg_V_prev_agg, bg_V_curr.squeeze(1))  # [bs, bg_k, 1]

        bg_curr_perm = bg_curr_feat.permute(0, 2, 1)  # [bs, bg_k, c_]
        # print("*"*50)
        # print("print(bg_w_hist.shape)")
        # print(bg_w_hist.shape)
        # print("print(bg_w_curr.shape)")
        # print(bg_w_curr.shape)
        # print("print(bg_history_agg_perm.shape)")
        # print(bg_history_agg_perm.shape)
        # print("print(bg_curr_perm.shape)")        
        # print(bg_curr_perm.shape)

        # # 断言批次大小一致
        # assert bg_w_hist.shape[0] == bg_history_agg_perm.shape[0] == bg_w_curr.shape[0] == bg_curr_perm.shape[0], \
        #     f"批次大小不匹配: {bg_w_hist.shape[0]}, {bg_history_agg_perm.shape[0]}, {bg_w_curr.shape[0]}, {bg_curr_perm.shape[0]}"

        # # 断言第二维度（bg_k）一致
        # assert bg_w_hist.shape[1] == bg_history_agg_perm.shape[1] == bg_w_curr.shape[1] == bg_curr_perm.shape[1], \
        #     f"bg_k维度不匹配: {bg_w_hist.shape[1]}, {bg_history_agg_perm.shape[1]}, {bg_w_curr.shape[1]}, {bg_curr_perm.shape[1]}"

        # # 断言第三维度（c_）匹配（bg_w_hist和bg_w_curr的第三维为1，不影响广播）
        # assert bg_history_agg_perm.shape[2] == bg_curr_perm.shape[2], \
        #     f"特征维度c_不匹配: {bg_history_agg_perm.shape[2]} vs {bg_curr_perm.shape[2]}"

        # print("bg_w_hist dtype:", bg_w_hist.dtype)
        # print("bg_history_agg_perm dtype:", bg_history_agg_perm.dtype)
        # print("bg_w_curr dtype:", bg_w_curr.dtype)
        # print("bg_curr_perm dtype:", bg_curr_perm.dtype)


        # print("bg_w_hist device:", bg_w_hist.device)
        # print("bg_history_agg_perm device:", bg_history_agg_perm.device)
        # print("bg_w_curr device:", bg_w_curr.device)
        # print("bg_curr_perm device:", bg_curr_perm.device)


        bg_fused = bg_w_hist * bg_history_agg_perm + bg_w_curr * bg_curr_perm  # [bs, bg_k, c_]
        # 先验证乘法是否正常
        # temp1 = bg_w_hist * bg_history_agg_perm
        # temp2 = bg_w_curr * bg_curr_perm
        # 再验证加法是否正常

        # bg_w_hist = bg_w_hist.contiguous()
        # bg_history_agg_perm = bg_history_agg_perm.contiguous()
        # bg_w_curr = bg_w_curr.contiguous()
        # bg_curr_perm = bg_curr_perm.contiguous()

        # # 重新计算
        # temp1 = bg_w_hist * bg_history_agg_perm
        # temp2 = bg_w_curr * bg_curr_perm
        # temp1 = temp1.contiguous()
        # temp2 = temp2.contiguous()
        # bg_fused = temp1 + temp2


        # 转移所有张量到CPU
        # bg_w_hist_cpu = bg_w_hist.cpu()
        # bg_history_agg_perm_cpu = bg_history_agg_perm.cpu()
        # bg_w_curr_cpu = bg_w_curr.cpu()
        # bg_curr_perm_cpu = bg_curr_perm.cpu()

        # # 分步执行运算
        # try:
        #     temp1_cpu = bg_w_hist_cpu * bg_history_agg_perm_cpu
        #     temp2_cpu = bg_w_curr_cpu * bg_curr_perm_cpu
        #     bg_fused_cpu = temp1_cpu + temp2_cpu
        #     print(bg_fused_cpu)
        #     print(bg_fused_cpu.shape)
        #     print("CPU运算成功，无明显错误")
        # except Exception as e:
        #     print(f"CPU运算报错：{e}")  # 此处会显示具体错误原因（如值异常）


        # bg_fused = temp1.clone() + temp2.clone()
        # 1/0
        bg_occ_feat = torch.gather(last_occ_reshaped, dim=1, index=bg_indices.unsqueeze(-1).repeat(1, 1, last_occ_pred.shape[-1]))  # [bs, bg_k, num_classes]
        bg_occ_embed = self.occ_embedding(bg_occ_feat).permute(0, 2, 1)  # [bs, occ_embedims, bg_k]
        bg_fused = torch.cat([bg_fused, bg_occ_embed.permute(0, 2, 1)], dim=-1)  # [bs, bg_k, c_+occ_embedims]
        bg_fused = self.history_fusion_bg_linear(bg_fused).permute(0, 2, 1)  # [bs, c_, bg_k]

        # 更新当前BEV
        curr_bev_updated = curr_bev_flat.clone()
        curr_bev_updated.scatter_add_(dim=2, index=fg_indices.unsqueeze(1).repeat(1, c_, 1), src=fg_fused)
        curr_bev_updated.scatter_add_(dim=2, index=bg_indices.unsqueeze(1).repeat(1, c_, 1), src=bg_fused)
        curr_bev_updated = curr_bev_updated.reshape(bs, c_, z, h, w)  # 恢复原形状

        # 更新历史缓存
        self.history_last_bev = curr_bev_updated.detach().clone()
        self.history_bev = torch.cat([curr_bev,sampled_history_bev[:, :-c_, ...]], dim=1).detach()
        # self.history_cam_intrins = torch.cat([curr_cam_intrins.unsqueeze(1),self.history_cam_intrins[:, :-1, ...]], dim=1).detach()
        # self.history_cam_extrins = torch.cat([curr_cam_extrins.unsqueeze(1),self.history_cam_extrins[:, 1-1:, ...]], dim=1).detach()
        self.history_visibility =torch.cat([V_curr, V_prev[:, :-1, ...]],dim=1).detach()
        self.history_forward_augs = forward_augs.clone()

        return curr_bev_updated

    def generate_grid(self, curr_bev):
        n, c_, z, h, w = curr_bev.shape
        # Generate grid
        xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w, 1).expand(h, w, z)
        ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1, 1).expand(h, w, z)
        zs = torch.linspace(0, z - 1, z, dtype=curr_bev.dtype, device=curr_bev.device).view(1, 1, z).expand(h, w, z)
        grid = torch.stack((xs, ys, zs, torch.ones_like(xs)), -1).view(1, h, w, z, 4).expand(n, h, w, z, 4).view(n, h,w, z, 4, 1)
        return grid

    def generate_feat2bev(self, grid, dx, bx):
        feat2bev = torch.zeros((4, 4), dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = dx[0]
        feat2bev[1, 1] = dx[1]
        feat2bev[2, 2] = dx[2]
        feat2bev[0, 3] = bx[0] - dx[0] / 2.
        feat2bev[1, 3] = bx[1] - dx[1] / 2.
        feat2bev[2, 3] = bx[2] - dx[2] / 2.
        feat2bev[3, 3] = 1
        feat2bev = feat2bev.view(1, 4, 4)
        return feat2bev





@HEADS.register_module()
class GatedTemporalFusion6_T_gate(BaseModule):
    def __init__(
        self,
        history_num=4,
        depth_sampler_embed_dims=256,
        depth_sampler_num_heads=8,
        depth_sampler_num_levels=4,
        depth_sampler_num_points=4,
        im2col_step=64,
        top_k=500,
        single_bev_num_channels=96,
        occ_embedims=32,
        num_classes=18,
        vis_theta=0.28,
        vis_beta=10.0,
        vis_gamma=0.4,
        vis_sigma=0.1,
        nonempty_thresh=0.1,
        max_step_ratio=1.2,
        **kwargs
    ):
        super(GatedTemporalFusion6_T_gate, self).__init__()

        # 基础参数
        self.history_num = history_num
        self.top_k = top_k
        self.single_bev_num_channels = single_bev_num_channels
        self.occ_embedims = occ_embedims
        # self.fg_scale = nn.Parameter(torch.full((history_num,), 10.0))
        # self.bg_scale = nn.Parameter(torch.full((history_num,), 10.0))
        # 可见性门控参数
        # self.vis_theta = vis_theta  # 可见性阈值
        # self.vis_beta = vis_beta    # 敏感度参数
        # self.vis_gamma = vis_gamma  # 场景4固定权重
        # self.vis_sigma = vis_sigma  # 软化参数
        # self.vis_theta = nn.Parameter(torch.tensor(vis_theta))   # 可见性阈值
        # self.vis_beta = nn.Parameter(torch.tensor(vis_beta))     # 敏感度参数
        # self.vis_gamma = nn.Parameter(torch.tensor(vis_gamma))   # 场景4固定权重
        # self.vis_sigma = nn.Parameter(torch.tensor(vis_sigma))  

        
        # 历史特征缓存（新增history_last_bev）
        self.history_bev = None  # 缓存历史多帧BEV特征
        self.history_last_bev = None  # 缓存上一帧最终融合后的BEV特征（关键新增）
        self.history_cam_intrins = None
        self.history_cam_extrins = None
        
        # 网络层（保持不变）
        self.occ_embedding = nn.Sequential(
            nn.Linear(num_classes, occ_embedims),
            nn.Softplus(),
            nn.Linear(occ_embedims, occ_embedims),
        )
        self.history_fusion_linear = nn.Sequential(
            nn.Linear(single_bev_num_channels + occ_embedims+1, single_bev_num_channels),
            nn.Softplus(),
            nn.Linear(single_bev_num_channels, single_bev_num_channels),
        )
        self.history_fusion_bg_linear = nn.Sequential(
            nn.Linear(single_bev_num_channels + occ_embedims+1, single_bev_num_channels),
            nn.Softplus(),
            nn.Linear(single_bev_num_channels, single_bev_num_channels),
        )
        self.history_fg_fusion_gate = nn.Sequential(
            nn.Linear((single_bev_num_channels+1) * (history_num + 1), single_bev_num_channels),
            nn.Softplus(),
            nn.Linear(single_bev_num_channels, history_num+1),
            nn.Sigmoid()
        )
        self.history_bg_fusion_gate = nn.Sequential(
            nn.Linear((single_bev_num_channels+1) * (history_num//2 + 1), single_bev_num_channels),
            nn.Softplus(),
            nn.Linear(single_bev_num_channels, history_num//2+1),
            nn.Sigmoid()
        )

        
        # 可见性计算组件（保持不变）
        self.rt_vis_calculator = EfficientRayTracingVisibility(
            nonempty_thresh=nonempty_thresh,
            max_step_ratio=max_step_ratio
        )
        # self.img_shape = (900, 1600)  # 默认图像尺寸

        # self.depth_sampler = DeformableDepthSampler(
        #     embed_dims=depth_sampler_embed_dims,
        #     num_heads=depth_sampler_num_heads,
        #     num_levels=depth_sampler_num_levels,
        #     num_points=depth_sampler_num_points
        # )
        self.history_forward_augs = None  # 用于缓存历史帧的变换矩阵（如BDAM矩阵）
        self.im2col_step = im2col_step
        # self.voxel_encoder = nn.Linear(single_bev_num_channels, depth_sampler_embed_dims)
        self.dbound = [1.0, 45.0, 0.5]
        self.pc_range = [-40, -40, -1.0, 40, 40, 5.4]
        self.final_dim = (256, 704)
    def compute_visibility(self, grid, cam_intrins, cam_extrins, img_shape, img_feats, spatial_shapes):
        """
        升级：结合可变形注意力采样的深度值优化可见性计算
        Args:
            img_feats: 多尺度图像特征 [bs, num_levels, c, h, w]
            spatial_shapes: 图像特征尺度 [num_levels, 2] (h, w)
            其他参数同原函数
        Returns:
            vis_prob: [bs, h, w, z] 优化后的可见性概率
        """
        bs, h, w, z, _ = grid.shape
        h_img, w_img = img_shape
        device = grid.device
        num_voxels = h * w * z  # 体素总数

        # 1. 原有相机投影逻辑（计算图像坐标和初始可见性）
        # 1.1 体素坐标→相机坐标→图像坐标
        grid_cam = grid.unsqueeze(1).expand(bs, self.num_cams, h, w, z, 3)  # [bs, num_cams, h, w, z, 3]
        grid_flat = grid_cam.reshape(-1, num_voxels, 3)  # [bs*num_cams, N, 3]
        grid_hom = torch.cat([grid_flat, torch.ones_like(grid_flat[..., :1])], dim=-1)  # [bs*num_cams, N, 4]

        cam_intrins_flat = cam_intrins.reshape(-1, 4, 4)  # [bs*num_cams, 4, 4]
        cam_extrins_flat = cam_extrins.reshape(-1, 4, 4)  # [bs*num_cams, 4, 4]
        extrins_inv = torch.inverse(cam_extrins_flat)
        cam_coords = torch.bmm(extrins_inv[:, :3, :4], grid_hom.permute(0, 2, 1)).permute(0, 2, 1)  # [bs*num_cams, N, 3]
        depth = cam_coords[..., 2:3] + 1e-8  # 相机坐标系下的深度

        # 1.2 计算初始可见性（原逻辑）
        img_coords = torch.bmm(cam_intrins_flat[:, :3, :3], cam_coords.permute(0, 2, 1)).permute(0, 2, 1)  # [bs*num_cams, N, 3]
        img_xy = img_coords[..., :2] / depth  # [bs*num_cams, N, 2] (u, v)
        depth_valid = (cam_coords[..., 2] > 0).float()  # [bs*num_cams, N]
        u_valid = (img_xy[..., 0] >= 0) & (img_xy[..., 0] < w_img)
        v_valid = (img_xy[..., 1] >= 0) & (img_xy[..., 1] < h_img)
        img_valid = (u_valid & v_valid).float()  # [bs*num_cams, N]
        initial_vis = depth_valid * img_valid  # [bs*num_cams, N]

        # 2. 可变形注意力深度采样
        # 2.1 准备输入：体素特征编码
        # voxel_feat = self.voxel_encoder(self.curr_bev_feat)  # [bs, c_, z, h, w] → [bs, z*h*w, embed_dims]（需提前展平体素特征）
        # voxel_feat = voxel_feat.reshape(bs, num_voxels, -1)  # [bs, N, embed_dims]

        # 2.2 生成参考点（归一化到[0,1]）
        norm_img_xy = img_xy / torch.tensor([w_img, h_img], device=device).view(1, 1, 2)  # [bs*num_cams, N, 2]
        # 取主相机（如第0个相机）的参考点作为采样基准
        ref_points = norm_img_xy.reshape(bs, self.num_cams, num_voxels, 2)[:, 0]  # [bs, N, 2]
        ref_points = ref_points.unsqueeze(2).repeat(1, 1, self.depth_sampler.num_levels, 1)  # [bs, N, L, 2]

        # 2.3 采样深度特征
        sampled_depth = self.depth_sampler(
            query=voxel_feat,
            value=img_feats,  # 多尺度图像特征 [bs, L, c, h, w]
            reference_points=ref_points,
            spatial_shapes=spatial_shapes
        )  # [bs, N]

        # 3. 结合采样深度优化可见性
        # 3.1 深度一致性校验：采样深度与相机投影深度的差异
        cam_depth = depth.reshape(bs, self.num_cams, num_voxels)[:, 0]  # 主相机的投影深度 [bs, N]
        depth_diff = torch.abs(sampled_depth - cam_depth) / (cam_depth + 1e-8)  # 相对深度差
        depth_consistent = (depth_diff < 0.3).float()  # 深度差小于30%则认为有效

        # 3.2 融合可见性：初始可见性 × 深度一致性
        initial_vis = initial_vis.reshape(bs, self.num_cams, num_voxels).max(dim=1)[0]  # [bs, N]（多相机取max）
        vis_prob = initial_vis * depth_consistent  # [bs, N]

        # 4. 还原形状
        return vis_prob.reshape(bs, h, w, z)  # [bs, h, w, z]

    def compute_alpha_unified(self, V_curr, V_prev):
        """统一计算当前帧融合权重α（覆盖四场景）"""
        eps = 1e-8
        # 1. 基础动态权重σ_base（场景1-3）
        ratio = V_curr / (V_curr + V_prev + eps)
        sigma_base = torch.sigmoid(self.vis_beta * (ratio - 0.5))
        
        # 2. 场景4软化掩码σ_both
        mask_curr = torch.sigmoid(-(V_curr - self.vis_theta) / self.vis_sigma)
        mask_prev = torch.sigmoid(-(V_prev - self.vis_theta) / self.vis_sigma)
        sigma_both = mask_curr * mask_prev
        
        # 3. 最终权重计算
        alpha = (1 - sigma_both) * sigma_base + sigma_both * self.vis_gamma
        return alpha.unsqueeze(-1)  # [bs, N, 1]

    def compute_gate_weights(self, V_prev_agg, V_curr):
        """计算历史和当前帧的门控权重"""
        alpha = self.compute_alpha_unified(V_curr, V_prev_agg)
        return 1 - alpha, alpha  # 历史权重，当前权重

    def generate_grid(self, curr_bev, voxel_min, voxel_max, voxel_size):
        """生成体素中心坐标网格（自车坐标系）"""
        bs, c_, z, h, w = curr_bev.shape
        device = curr_bev.device
        
        # 计算体素中心坐标
        x_coords = torch.linspace(
            voxel_min[0] + voxel_size[0]/2, 
            voxel_max[0] - voxel_size[0]/2, 
            w, device=device
        )
        y_coords = torch.linspace(
            voxel_min[1] + voxel_size[1]/2, 
            voxel_max[1] - voxel_size[1]/2, 
            h, device=device
        )
        z_coords = torch.linspace(
            voxel_min[2] + voxel_size[2]/2, 
            voxel_max[2] - voxel_size[2]/2, 
            z, device=device
        )
        
        # 生成网格并扩展维度 [bs, h, w, z, 3]
        x_grid, y_grid, z_grid = torch.meshgrid(x_coords, y_coords, z_coords, indexing='xy')
        grid = torch.stack([x_grid, y_grid, z_grid], dim=-1)  # [w, h, z, 3]
        grid = grid.permute(1, 0, 2, 3)  # [h, w, z, 3]
        return grid.unsqueeze(0).repeat(bs, 1, 1, 1, 1)  # [bs, h, w, z, 3]


    def get_reference_points(self, H, W, Z=None, num_points_in_pillar =4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
        torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    @force_fp32(apply_to=('reference_points', 'cam_params'))
    def point_sampling(self, reference_points, pc_range, img_metas, cam_params=None):
        # prepare for point sampling
        lidar2img = []
        ego2lidar = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])     # lidar2img update the post aug in the loading pipeline
            ego2lidar.append(img_meta['ego2lidar'])
        lidar2img = torch.stack(lidar2img, dim=0).to(reference_points.device)
        ego2lidar = torch.stack(ego2lidar, dim=0).to(reference_points.device)

        sensor2egos, ego2globals, intrins, post_augs, bda_mat = cam_params
        num_cam = sensor2egos.size(1)
        ogfH, ogfW = self.final_dim

        # reference_points defines in the bev space, [bs, D, hxw, 3]
        # change reference_points from bev-ego coordinate to ego coordinate
        reference_points = reference_points.clone()
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
                                     (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
                                     (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
                                     (pc_range[5] - pc_range[2]) + pc_range[2]

        # prepare for point sampling
        reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
        reference_points = reference_points.permute(1, 0, 2, 3)  # shape: (num_points_in_pillar,bs,h*w,4)
        D, B, num_query = reference_points.size()[:3]  # D=num_points_in_pillar , num_query=h*w
        reference_points = reference_points.view(D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)  # shape: (num_points_in_pillar,bs,num_cam,h*w,4)
        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)
        ego2lidar = ego2lidar.view(1, B, 1, 1, 4, 4).repeat(D, 1, num_cam, num_query, 1, 1)
        inverse_bda = bda_mat.view(1, B, 1, 1, 4, 4).repeat(D, 1, num_cam, num_query, 1, 1)

        # change reference_points from ego coordinate to img coordinate
        eps = 1e-5
        reference_points_cam = (lidar2img @ ego2lidar @ inverse_bda @ reference_points).squeeze(-1)   # [num_points_in_pillar, bs, num_cam, num_query=h*w, 4]
        reference_points_depth = reference_points_cam[..., 2:3]
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(reference_points_depth, torch.ones_like(reference_points_depth) * eps)

        # Bug!!
        # Correct normalize is
        # reference_points_cam[..., 0] /= ogfW
        # reference_points_cam[..., 1] /= ogfH
        # But for reproducing our results, we use the following normalization
        reference_points_cam[..., 0] /= ogfW
        reference_points_cam[..., 1] /= ogfH

        bev_mask = (reference_points_depth > eps)
        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)                  # shape: (num_cam, bs,h*w, num_points_in_pillar, 2)
        reference_points_depth = reference_points_depth.permute(2, 1, 3, 0, 4)              # shape: (num_cam, bs,h*w, num_points_in_pillar, 1)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)                        # shape: (num_cam, bs,h*w, num_points_in_pillar)

        return reference_points_cam, reference_points_depth, bev_mask


    @force_fp32()
    def forward(self, curr_bev, cam_params, history_fusion_params, dx, bx, history_last_bev=None, last_occ_pred=None, nonempty_prob=None, img_feats=None, spatial_shapes=None,pred_img_depth=None,**kwargs):
        """
        Args:
            curr_bev: [bs, c, z, h, w] 当前帧BEV特征
            cam_params: 相机参数列表，包含外参、内参等
            history_fusion_params: 历史融合参数（包含序列信息等）
            dx: 体素尺寸 (x, y, z)
            bx: 体素偏移
            nonempty_prob: [bs, z, h, w] 体素非空概率
            last_occ_pred: [bs, z, h, w, num_classes] 上一时刻 occupancy 预测
            img_feats: 多尺度图像特征 [bs, num_levels, c, h, w]（新增，用于深度采样）
            spatial_shapes: 图像特征尺度 [num_levels, 2] (h, w)（新增，用于深度采样）
        Returns:
            curr_bev_updated: [bs, c, z, h, w] 融合后BEV特征
        """
        # print(self.history_num)
        # print("print(self.fg_scale)")
        # print(self.fg_scale) 
        # print("print(self.bg_scale)")
        # print(self.bg_scale)
        # # 可见性门控参数
        # # self.vis_theta = vis_theta  # 可见性阈值
        # # self.vis_beta = vis_beta    # 敏感度参数
        # # self.vis_gamma = vis_gamma  # 场景4固定权重
        # # self.vis_sigma = vis_sigma  # 软化参数
        # print("print(self.vis_theta # 可见性阈值)")
        # print(self.vis_theta )  # 可见性阈值
        # print("print(self.vis_beta) # 敏感度参数")
        # print(self.vis_beta)      # 敏感度参数
        # print("print(self.vis_gamma) # 场景4固定权重")
        # print(self.vis_gamma)  # 场景4固定权重
        # print("print(self.vis_sigma) # 软化参数")
        # print(self.vis_sigma) 


        # if torch.rand(1).item() < 1/2000:
        #     print(self.history_num)
        #     print("print(self.fg_scale)")
        #     print(self.fg_scale) 
        #     print("print(self.bg_scale)")
        #     print(self.bg_scale)
        #     # 可见性门控参数
        #     print("print(self.vis_theta # 可见性阈值)")
        #     print(self.vis_theta)  # 可见性阈值
        #     print("print(self.vis_beta) # 敏感度参数")
        #     print(self.vis_beta)      # 敏感度参数
        #     print("print(self.vis_gamma) # 场景4固定权重")
        #     print(self.vis_gamma)  # 场景4固定权重
        #     print("print(self.vis_sigma) # 软化参数")
        #     print(self.vis_sigma)  

        # torch.cuda.empty_cache()

        # -------------------------- 1. 解析参数后打印核心形状 --------------------------
        # 解析相机参数
        curr_cam_extrins = cam_params[0]  # [bs, num_cams, 4, 4]
        curr_cam_intrins = cam_params[2]  # [bs, num_cams, 4, 4]
        forward_augs = cam_params[4]      # [bs, 4, 4] 前向变换矩阵
        self.num_cams = curr_cam_extrins.shape[1]  # 从外参中获取相机数量
        bs, c_, z, h, w = curr_bev.shape
        device = curr_bev.device
        mc = self.history_num * c_        # 历史特征总通道数
        # self.history_forward_augs = forward_augs.clone()


        ref_3d = self.get_reference_points(
            h, w, z, z, dim='3d', bs=bs, device=device, dtype=curr_bev.dtype) # torch.Size([3, 2, 625, 3]) #[bs,z,yx,3(x,y,z)]
        # ref_2d = self.get_reference_points(
        #     h, w, dim='2d', bs=bs, device=device, dtype=curr_bev.dtype) #torch.Size([3, 625, 1, 2])
        slots = torch.zeros(list([ref_3d.shape[0],ref_3d.shape[2],ref_3d.shape[1]])).to(ref_3d)
        reference_points_cam, reference_points_depth, bev_mask = self.point_sampling(ref_3d, self.pc_range, img_metas=kwargs['img_metas'], cam_params=cam_params)
        indexes = [[] for _ in range(bs)]
        spatial_shapes =[]
        spatial_shapes.append([16, 44])
        spatial_shapes = torch.tensor(spatial_shapes).to(device)
        pred_img_depth = pred_img_depth.view(bs * 6, -1, spatial_shapes[0][0], spatial_shapes[0][1])
        pred_img_depth = pred_img_depth.flatten(2).permute(0, 2, 1)  
        max_len = 0
        for j in range(bs):
            for i, mask_per_img in enumerate(bev_mask):
                index_query_per_img = mask_per_img[j].sum(-1).nonzero().squeeze(-1)
                if len(index_query_per_img) == 0:
                    index_query_per_img = bev_mask[i][j].sum(-1).nonzero().squeeze(-1)[0:1]
                indexes[j].append(index_query_per_img)
                # for batch operation, we need to pad the indexes to the same length
                max_len = max(max_len, len(index_query_per_img))
        reference_points_cam_rebatch = reference_points_cam.new_zeros([bs, self.num_cams, max_len, z, 2])
        reference_points_depth_rebatch = reference_points_depth.new_zeros([bs, self.num_cams, max_len, z, 1])

        for j in range(bs):
            for i, (reference_points_per_img, reference_points_depth_per_img) in enumerate(zip(reference_points_cam, reference_points_depth)):
                index_query_per_img = indexes[j][i]
                reference_points_cam_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]
                reference_points_depth_rebatch[j, i, :len(index_query_per_img)] = reference_points_depth_per_img[j, index_query_per_img]

        #use deformble attn
        depth_reference_points = reference_points_cam_rebatch.reshape(bs*6, max_len*z, 1, 1, 1, 2).contiguous()
        depth_attention_weights = torch.ones_like(depth_reference_points[..., 0]).contiguous()
        pred_img_depth = pred_img_depth.unsqueeze(2).contiguous()
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))



        bev_query_depth_rebatch = (reference_points_depth_rebatch- self.dbound[0])/ self.dbound[2]
        bev_query_depth_rebatch = torch.clip(torch.floor(bev_query_depth_rebatch), 0, 88-1).to(torch.long)
        bev_query_depth_rebatch = F.one_hot(bev_query_depth_rebatch.squeeze(-1),
                                   num_classes=88)

        depth_output = MultiScaleDeformableAttnFunction_fp32.apply(pred_img_depth, spatial_shapes,level_start_index,depth_reference_points,depth_attention_weights, self.im2col_step)
        depth_output = depth_output.reshape(bs,6, max_len,z, -1)   # [bs*num_cam, num_query, num_Z_anchors, C]
        # reference_points_depth_rebatch

        increment = torch.zeros_like(depth_output)
        # increment[..., 0] = 1e-9  # 非原地赋值（创建新张量）
        # depth_output = depth_output + increment
        depth_output = depth_output + torch.cat([(torch.zeros_like(depth_output[...,:1]) + 1e-9),torch.zeros_like(depth_output[...,1:])],dim=-1)

        # print("depth_output.sum(-1).shape")
        # print(depth_output.sum(-1).shape)
        # print("depth_output.sum(-1)")
        # print(depth_output.sum(-1))
        # print("print((depth_output.sum(-1)==0).sum())")
        # print((depth_output.sum(-1)==0).sum())
        depth_output =depth_output/depth_output.sum(-1)[...,None] #bs,xy,z,D

        # print("depth_output.sum(-1).shape")
        # print(depth_output.sum(-1).shape)
        # print("depth_output.sum(-1)")
        # print(depth_output.sum(-1))
        # print("print((depth_output.sum(-1)==1).sum())")
        # print((depth_output.sum(-1)>=0.99).sum())
        # print("depth_output")
        # print(depth_output)


        depth_output = (1-depth_output.cumsum(dim=-1))
        depth_output =torch.cat([torch.ones_like(depth_output[...,0:1]),depth_output[...,0:-1]],dim=-1)

        # print("print((depth_output[...,-1]==0).sum())")
        # print((depth_output[...,-1]==0).sum())
        depth_output = (bev_query_depth_rebatch*depth_output).sum(-1)

        #恢复depth_output的shape
        # fix_depth_output = depth_output.new_zeros([bs, 6, w*h, z, 88])
        # for j in range(bs):
        #     for i in range(6):
        #         index_query_per_img = indexes[j][i]
        #         fix_depth_output[j, i, index_query_per_img] = depth_output[j, i, :len(index_query_per_img)]

        for j in range(bs):
            for i in range(6):
                index_query_per_img = indexes[j][i]
                slots[j, index_query_per_img] = torch.max(slots[j, index_query_per_img],depth_output[j, i, :len(index_query_per_img)])

        # depth_sum = fix_depth_output.sum(dim=-1).view(3,6,25,25,2)
        # bs = depth_sum.shape[0]       # 3
        # num_cam = depth_sum.shape[1]  # 6
        # z_layers = depth_sum.shape[4] # 2
        # H, W = depth_sum.shape[2], depth_sum.shape[3]  # 25,25

        # # 颜色映射：0值用黑色，非0值用渐变色
        # cmap = plt.cm.viridis
        # cmap.set_bad(color='black')  # 0值标记为黑色

        # # 设置子图布局（不变）
        # fig, axes = plt.subplots(
        #     nrows=bs, ncols=num_cam * z_layers,
        #     figsize=(30, 8),
        #     squeeze=False
        # )
        # fig.suptitle('fix_depth_output.sum(-1) 可视化（黑色=0值，颜色=非0值）', fontsize=16, y=0.98)


        # # -------------------------- 2. 循环绘制每个子图（核心修改：添加 .detach()） --------------------------
        # for batch_idx in range(bs):
        #     for cam_idx in range(num_cam):
        #         for z_idx in range(z_layers):
        #             col_idx = cam_idx * z_layers + z_idx
        #             ax = axes[batch_idx, col_idx]
                    
        #             # 核心修改：先 detach() 切断计算图，再转 cpu 和 numpy
        #             data = depth_sum[batch_idx, cam_idx, :, :, z_idx].detach().cpu().numpy()
        #             data[data == 0] = np.nan  # 0值替换为NaN，显示为黑色
                    
        #             # 绘制热力图（修改 vmin/vmax：同样添加 .detach()）
        #             im = ax.imshow(
        #                 data, 
        #                 cmap=cmap, 
        #                 aspect='auto',
        #                 # 关键修改：depth_sum 先 detach 再转 numpy，确保不影响梯度
        #                 vmin=np.nanmin(depth_sum.detach().cpu().numpy()),
        #                 vmax=np.nanmax(depth_sum.detach().cpu().numpy())
        #             )
                    
        #             # 子图标题和坐标轴（不变）
        #             ax.set_title(
        #                 f'Batch{batch_idx+1}\nCam{cam_idx+1} Z{z_idx+1}',
        #                 fontsize=10, pad=5
        #             )
        #             ax.set_xticks([])
        #             ax.set_yticks([])


        # # -------------------------- 3. 添加颜色条（不变） --------------------------
        # cbar = fig.colorbar(
        #     im, 
        #     ax=axes.ravel().tolist(),
        #     shrink=0.8,
        #     pad=0.02
        # )
        # cbar.set_label('Sum of Depth Bins (D=88)', fontsize=12)


        # # -------------------------- 4. 调整布局并保存（不变） --------------------------
        # plt.tight_layout(rect=[0, 0, 0.98, 0.95])
        # plt.savefig('depth_sum_visualization.png', dpi=300, bbox_inches='tight')
        # plt.show()

        # output

        #计数更新，建立在纸上的假设成立的基础上
        # count = bev_mask.sum(-1) > 0
        # count = count.permute(1, 2, 0).sum(-1)
        # count = torch.clamp(count, min=1.0)
        # slots = slots / count[..., None]

        # print("slots.shape")
        # print(slots.shape)
        # print("slots")  
        # print(slots)
        # zzzzz=1/0

        V_curr =slots.view(bs, 1, h, w,z).permute(0, 1, 4, 2, 3)
        # slots[...,0]+=1e-9
        # print("slots.sum(-1).shape")
        # print(slots.sum(-1).shape)
        # print("slots.sum(-1)")
        # print(slots.sum(-1))
        # print("print((slots.sum(-1)==0).sum())")
        # print((slots.sum(-1)==0).sum())
        # slots =slots/slots.sum(-1)[...,None] #bs,xy,z,D

        # print("slots.sum(-1).shape")
        # print(slots.sum(-1).shape)
        # print("slots.sum(-1)")
        # print(slots.sum(-1))
        # print("print((slots.sum(-1)==1).sum())")
        # print("slots")
        # print(slots)


        # slots = (1-slots.cumsum(dim=-1))

        # print("print((slots[...,-1]==0).sum())")
        # print((slots[...,-1]==0).sum())

        # print("slots")
        # print(slots)
        # print("print((slots[...,-1]<0.01).sum())")
        # print((slots[...,-1]<0.01).sum())

        #TODO 这里的对于边界值的考虑，从0开始还是从1开始？




        # slots = self.output_proj(slots)
        #TODO 上面这里需要检查一下
        #这里相当于两次softmax，可能会导致分布变得不够尖锐，需要进一步确认

        #到这里slots就是可见性的概率分布了
        #这里先尝试使用期望值进行计算，使得可微分

        #然后再采用stc的原始离散计算方法，


        # 打印解析后关键变量形状
        # print("="*50)
        # print("1. 解析参数后核心变量形状：")
        # print(f"curr_bev: {curr_bev.shape} (预期：[bs, c, z, h, w])")
        # print(f"curr_cam_extrins: {curr_cam_extrins.shape} (预期：[bs, num_cams, 4, 4])")
        # print(f"curr_cam_intrins: {curr_cam_intrins.shape} (预期：[bs, num_cams, 4, 4])")
        # print(f"forward_augs: {forward_augs.shape} (预期：[bs, 4, 4])")
        # print(f"dx: {dx.shape if hasattr(dx, 'shape') else type(dx)} (预期：[3])")
        # print(f"bx: {bx.shape if hasattr(bx, 'shape') else type(bx)} (预期：[3])")
        # print(f"bs: {bs}, c_: {c_}, z: {z}, h: {h}, w: {w} (BEV特征维度)")
        # print("="*50)

        if type(history_fusion_params['sequence_group_idx']) is list:
            seq_ids = history_fusion_params['sequence_group_idx'][0]
        else:
            seq_ids = history_fusion_params['sequence_group_idx']
        if type(history_fusion_params['start_of_sequence']) is list:
            start_of_sequence = history_fusion_params['start_of_sequence'][0]
        else:
            start_of_sequence = history_fusion_params['start_of_sequence']
        if type(history_fusion_params['curr_to_prev_ego_rt']) is list:
            curr_to_prev_ego_rt = history_fusion_params['curr_to_prev_ego_rt'][0]
        else:
            curr_to_prev_ego_rt = history_fusion_params['curr_to_prev_ego_rt']
        forward_augs = cam_params[-1]  # bda

        # check seq_ids > 0
        assert (seq_ids >= 0).all()
        # -------------------------- 2. 初始化历史缓存后打印 --------------------------
        if self.history_bev is None:
            # self.history_bev = curr_bev.repeat(1, self.history_num, 1, 1, 1)  # [bs, mc, z, h, w]
            # self.history_cam_intrins = curr_cam_intrins.unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)  # [bs, T, num_cams, 4, 4]
            # self.history_cam_extrins = curr_cam_extrins.unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)  # [bs, T, num_cams, 4, 4]
            # self.history_bev = curr_bev.clone()
            self.history_forward_augs = forward_augs.clone()
            self.history_seq_ids = seq_ids.clone()
            self.history_bev = curr_bev.repeat(1, self.history_num, 1, 1, 1)
            self.history_sweep_time = curr_bev.new_zeros(curr_bev.shape[0], self.history_num)
            self.history_visibility = V_curr.repeat(1, self.history_num, 1, 1, 1).half()
        self.history_bev = self.history_bev.detach()
        self.history_visibility = self.history_visibility.detach().half()
        self.history_sweep_time += 1

        # 打印历史缓存形状
        # print("\n2. 历史缓存初始化后形状：")
        # print(f"history_bev: {self.history_bev.shape} (预期：[bs, mc, z, h, w]，mc={mc})")
        # print(f"history_cam_intrins: {self.history_cam_intrins.shape} (预期：[bs, history_num, num_cams, 4, 4])")
        # print(f"history_cam_extrins: {self.history_cam_extrins.shape} (预期：[bs, history_num, num_cams, 4, 4])")

        # -------------------------- 3. 生成网格和BEV变换矩阵后打印 --------------------------
        # 处理新序列（略，不影响维度）
        # start_of_sequence = history_fusion_params.get('start_of_sequence', torch.zeros(bs, dtype=torch.bool, device=device))
        if start_of_sequence.sum()>0:
            self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_num, 1, 1, 1)
            # self.history_cam_intrins[start_of_sequence] = curr_cam_intrins[start_of_sequence].unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)
            # self.history_cam_extrins[start_of_sequence] = curr_cam_extrins[start_of_sequence].unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)
            self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]
            self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
            self.history_sweep_time[start_of_sequence] = 0  # zero the new sequence timestep starts
            self.history_visibility[start_of_sequence] = V_curr[start_of_sequence].repeat(1, self.history_num, 1, 1, 1).half()

        # 生成体素网格和BEV变换矩阵
        grid = self.generate_grid(curr_bev) #[bs,y,x,z,4]
        grid_3d = grid
        feat2bev = self.generate_feat2bev(grid, dx, bx)

        # 打印网格和变换矩阵形状
        # print("\n3. 生成网格和BEV变换矩阵后形状：")
        # print(f"grid_3d (体素网格): {grid_3d.shape} (关键！预期：[bs, h, w, z, 3] 或 [bs, w, h, z, 3])")
        # print(f"feat2bev (BEV变换矩阵): {feat2bev.shape} (预期：[bs, 4, 4])")

        # -------------------------- 4. 运动补偿矩阵计算后打印 --------------------------
        # 获取帧间姿态变换
        # curr_to_prev_ego_rt = history_fusion_params.get('curr_to_prev_ego_rt', torch.eye(4, device=device).unsqueeze(0).repeat(bs, 1, 1))
        # 计算RT流（坐标变换矩阵）
        rt_flow = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_ego_rt @ torch.inverse(forward_augs) @ feat2bev)
        # 生成齐次网格
        # 在forward函数中，生成grid_hom的位置修正：
        # grid_3d = self.generate_grid(curr_bev)  # 现在形状：[3, 25, 25, 2, 3]（bs, h, w, z, 3）
        # # 生成齐次坐标（x,y,z,1），并添加最后一个维度（用于矩阵乘法）
        # grid_hom = torch.cat([
        #     grid_3d,  # [3,25,25,2,3]
        #     torch.ones_like(grid_3d[..., :1])  # [3,25,25,2,1]（补充1作为齐次坐标）
        # ], dim=-1).unsqueeze(-1)  # 最终形状：[3,25,25,2,4,1]（符合预期）
        # # 打印运动补偿相关形状（矩阵乘法前关键检查）
        # print("\n4. 运动补偿矩阵计算后形状（矩阵乘法前）：")
        # print(f"curr_to_prev_ego_rt (帧间姿态): {curr_to_prev_ego_rt.shape} (预期：[bs, 4, 4])")
        # print(f"rt_flow (变换流): {rt_flow.shape} (预期：[bs, 4, 4])")
        # print(f"grid_hom (齐次网格): {grid_hom.shape} (关键！预期：[bs, h, w, z, 4, 1]，需与rt_flow广播匹配)")
        # print(f"rt_flow.view后: {rt_flow.view(bs, 1, 1, 1, 4, 4).shape} (预期：[bs, 1, 1, 1, 4, 4])")

        # # -------------------------- 5. 网格变换后打印（解决之前维度错的核心） --------------------------
        # try:
        #     grid_transformed = rt_flow.view(bs, 1, 1, 1, 4, 4) @ grid_hom  # 矩阵乘法：[bs, h, w, z, 4, 1]
        #     print("\n5. 网格变换后形状（矩阵乘法成功！）：")
        #     print(f"grid_transformed: {grid_transformed.shape} (预期：[bs, h, w, z, 4, 1])")
        # except RuntimeError as e:
        #     print(f"\n5. 网格变换矩阵乘法报错！错误信息：{str(e)}")
        #     print(f"  - rt_flow.view形状: {rt_flow.view(bs, 1, 1, 1, 4, 4).shape}")
        #     print(f"  - grid_hom形状: {grid_hom.shape}")
        #     print("  提示：需确保grid_hom的第1-4维度与rt_flow.view的第2-5维度匹配（广播规则）")
        #     raise e  # 继续抛出错误，方便定位
        bs, mc, z, h, w = self.history_bev.shape
        n, c_, z, h, w = curr_bev.shape
        grid = rt_flow.view(n, 1, 1, 1, 4, 4) @ grid
        # -------------------------- 6. 采样网格生成后打印 --------------------------
        # 生成采样网格（归一化到[-1,1]，适配F.grid_sample）
        normalize_factor = torch.tensor([w - 1.0, h - 1.0, z - 1.0], dtype=curr_bev.dtype, device=device)
        # grid_sampler = grid_transformed[..., :3, 0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0  # [bs, h, w, z, 3]
        # # 调整采样网格维度（适配F.grid_sample输入：[bs, z, h, w, 3]）
        # grid_sampler_permuted = grid_sampler.permute(0, 3, 1, 2, 4)  # 交换z和h/w维度
        grid = grid[:, :, :, :, :3, 0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0   # grid order is x, y, z


        # print("\n6. 采样网格生成后形状：")
        # print(f"grid_sampler (归一化后): {grid_sampler.shape} (预期：[bs, h, w, z, 3])")
        # print(f"grid_sampler_permuted (适配采样): {grid_sampler_permuted.shape} (预期：[bs, z, h, w, 3])")

        # -------------------------- 7. 历史BEV采样后打印 --------------------------
        # 采样历史BEV特征
        sampled_history_bev = F.grid_sample(
            self.history_bev.reshape(bs, mc, z, h, w),  # 输入：[bs, mc, z, h, w]
            grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4),
            align_corners=True,
            mode='bilinear'
        )
        sampled_history_visibility = F.grid_sample(
            self.history_visibility.reshape(bs, self.history_num, z, h, w).half(),  # 输入：[bs, mc, z, h, w]
            grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4).half(),
            align_corners=True,
            mode='nearest'
        )
        # print("\n7. 历史BEV采样后形状：")
        # print(f"history_bev.reshape: {self.history_bev.reshape(bs, mc, z, h, w).shape} (预期：[bs, mc, z, h, w])")
        # print(f"sampled_history_bev: {sampled_history_bev.shape} (预期：[bs, mc, z, h, w])")

        # -------------------------- 8. 可见性计算后打印 --------------------------
        # 计算当前帧可见性
        # V_curr = self.compute_visibility(
        #     grid_3d, 
        #     cam_intrins=curr_cam_intrins,
        #     cam_extrins=curr_cam_extrins,
        #     img_shape=self.img_shape,
        #     img_feats=img_feats,
        #     spatial_shapes=spatial_shapes
        # )

        # print("V_curr (当前可见性).  "*3)
        # V_curr = slots
        # 计算历史帧可见性
        # prev_cam_intrins = self.history_cam_intrins[:, -1]
        # prev_cam_extrins = self.history_cam_extrins[:, -1]
        # V_prev = self.compute_visibility(
        #     grid_3d,
        #     cam_intrins=prev_cam_intrins,
        #     cam_extrins=prev_cam_extrins,
        #     img_shape=self.img_shape,
        #     img_feats=img_feats,
        #     spatial_shapes=spatial_shapes
        # )

        V_prev = sampled_history_visibility #bs,4,z,h,w

        # print("\n8. 可见性计算后形状：")
        # print(f"V_curr (当前可见性): {V_curr.shape} (预期：[bs, h, w, z])")
        # print(f"V_prev (历史可见性): {V_prev.shape} (预期：[bs, h, w, z])")

        # -------------------------- 9. 稀疏采样前展平变量打印 --------------------------
        # 展平变量（用于稀疏采样）
        curr_bev_flat = curr_bev.reshape(bs, c_, -1)  # [bs, c_, N], N=h*w*z
        history_bev_flat = sampled_history_bev.reshape(bs, mc, -1)  # [bs, mc, N]
        V_prev_flat = V_prev.reshape(bs,self.history_num, -1)  # [bs, 4,N]
        V_curr_flat = V_curr.reshape(bs, 1,-1)  # [bs, 1,N]
        nonempty_prob_flat = nonempty_prob.reshape(bs, -1)  # [bs, N]
        # print("print(nonempty_prob_flat.shape)")
        # print(nonempty_prob_flat.shape)
        total_voxels = nonempty_prob_flat.shape[1]

        # print("\n9. 稀疏采样前展平变量形状：")
        # print(f"curr_bev_flat: {curr_bev_flat.shape} (预期：[bs, c_, N], N={total_voxels})")
        # print(f"history_bev_flat: {history_bev_flat.shape} (预期：[bs, mc, N])")
        # print(f"nonempty_prob_flat: {nonempty_prob_flat.shape} (预期：[bs, N])")
        # print(f"total_voxels (h*w*z): {total_voxels} (预期：{h*w*z})")

        # -------------------------- 10. 前景/背景索引及融合后打印（可选，确认后续维度） --------------------------
        # 生成前景/背景索引
        fg_indices = torch.topk(nonempty_prob_flat, self.top_k, dim=1)[1]
        bg_indices = torch.topk(1 - nonempty_prob_flat, total_voxels - self.top_k, dim=1)[1]
        # 提取前景特征（示例，其他融合步骤类似）
        fg_history_feat = torch.gather(history_bev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, mc, 1))

        # print("\n10. 前景/背景索引及特征提取后形状：")
        # print(f"fg_indices (前景索引): {fg_indices.shape} (预期：[bs, top_k])")
        # print(f"bg_indices (背景索引): {bg_indices.shape} (预期：[bs, N-top_k])")
        # print(f"fg_history_feat (前景历史特征): {fg_history_feat.shape} (预期：[bs, mc, top_k])")
        # print("="*50)

        # -------------------------- 后续原有逻辑（略，维度已通过打印确认） --------------------------
        # 8. 前景融合（原有代码）
        # 9. 背景融合（原有代码）
        # 10. 更新当前BEV特征（原有代码）
        # curr_bev_flat = curr_bev.reshape(bs, c_, -1)  # [bs, c_, N] N=h*w*z
        # history_bev_flat = sampled_history_bev.reshape(bs, mc, -1)  # [bs, mc, N]
        # V_prev_flat = V_prev.reshape(bs, -1)  # [bs, N]
        # V_curr_flat = V_curr.reshape(bs, -1)  # [bs, N]
        # nonempty_prob_flat = nonempty_prob.reshape(bs, -1)  # [bs, N]
        # total_voxels = nonempty_prob_flat.shape[1]

        # fg_indices = torch.topk(nonempty_prob_flat, self.top_k, dim=1)[1]  # [bs, top_k]
        # bg_indices = torch.topk(1 - nonempty_prob_flat, total_voxels - self.top_k, dim=1)[1]  # [bs, N-top_k]

        # # 前景特征提取
        # fg_history_feat = torch.gather(history_bev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, mc, 1))  # [bs, mc, top_k]
        fg_curr_feat = torch.gather(curr_bev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, c_, 1))  # [bs, c_, top_k]

        # 历史特征时间聚合
        # fg_history_feat_time = fg_history_feat.reshape(bs, self.history_num, c_, self.top_k)  # [bs, T, c_, K]
        #TODO 后续可以把time_weights也乘进去


        # time_weights = torch.exp(-0.5 * torch.arange(self.history_num, device=device)).view(1, self.history_num, 1, 1)

        fg_V_prev = torch.gather(V_prev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, self.history_num, 1))  # [bs, 4,K]
        fg_V_curr = torch.gather(V_curr_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, 1, 1))  # [bs, 1,K]
        # fg_time_vis_weights = fg_V_prev/(fg_V_prev.sum(dim=1).unsqueeze(1)+1e-10 ) # [bs, 4,K]
        #下面进行替换，不用显示提取权重，不计算 softmax
        # fg_time_vis_weights = (fg_V_prev*(self.fg_scale.view(1,self.history_num,1))).softmax(dim=1)
        # fg_time_vis_weights = (fg_V_prev*(self.fg_scale.view(1,self.history_num,1)))
        # print("print(last_occ_pred.shape)")
        # print(last_occ_pred.shape) #bs,x,y,z,num_classes ->bs,z,y,x,num_classes
        last_occ_pred = last_occ_pred.permute(0, 3, 2, 1, 4).reshape(bs, -1, last_occ_pred.shape[-1])  # [bs, N, num_classes]

        fg_occ_feat = torch.gather(last_occ_pred, dim=1, index=fg_indices.unsqueeze(-1).repeat(1, 1, last_occ_pred.shape[-1]))  # [bs, K, num_classes]

        fg_occ_embed = self.occ_embedding(fg_occ_feat).permute(0, 2, 1)  # [bs, occ_embedims, K]

        # print("print(fg_V_curr.shape)")
        # print(fg_V_curr.shape) # bs,1,K
        # print("print(fg_curr_feat.shape)")
        # print(fg_curr_feat.shape) #torch.Size([2, 96, 500])  bs,c_,K
        # print("print(fg_V_prev.shape)") 
        # print(fg_V_prev.shape) #torch.Size([2, 4, 500])  bs,4,K
        # print("print(fg_history_feat.shape)")
        # print(fg_history_feat.shape) #torch.Size([2, 384, 500])  bs,mc,K
        # print("print(fg_occ_embed.shape)")
        # print(fg_occ_embed.shape) #torch.Size([2, 32, 500])  bs,occ_embedims,K
        
        # bs,(T+1,c_+1),N -> bs,T+1,N
        # bs,(T+1),c_+1,N * bs,T+1,1,N -> bs,(1,c_,32),N -> bs,c_,N
        fg_fused = torch.cat([ fg_V_curr * fg_curr_feat,fg_V_curr,torch.cat([fg_V_prev.unsqueeze(2) * fg_history_feat.view(bs, self.history_num, c_, self.top_k),fg_V_prev.unsqueeze(2)],dim=2).reshape(bs, self.history_num*(c_+1), self.top_k) ], dim=1).permute(0, 2, 1)
        fg_fused_gate =self.history_fg_fusion_gate(fg_fused).permute(0, 2, 1).unsqueeze(2)  # [bs, T+1,1, N]

        fg_fused = (fg_fused.permute(0,2,1).reshape(bs, self.history_num+1, c_+1, self.top_k) * fg_fused_gate).sum(dim=1).squeeze(1)  # [bs, T+1,c_, N]
        fg_fused = self.history_fusion_linear(torch.cat([fg_fused,fg_occ_embed],dim=1).permute(0,2,1)).permute(0, 2, 1)  # [bs, c_, K]


        bg_V_prev = torch.gather(V_prev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, self.history_num//2, 1))  # [bs, bg_k]
        bg_V_curr = torch.gather(V_curr_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, 1, 1))  # [bs, bg_k]
        bg_history_feat = torch.gather(history_bev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, mc//2, 1))  # [bs, mc//2, bg_k]
        bg_curr_feat = torch.gather(curr_bev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, c_, 1))  # [bs, c_, bg_k]
        bg_occ_feat = torch.gather(last_occ_pred, dim=1, index=bg_indices.unsqueeze(-1).repeat(1, 1, last_occ_pred.shape[-1]))  # [bs, bg_k, num_classes]
        bg_occ_embed = self.occ_embedding(bg_occ_feat).permute(0, 2, 1)  # [bs, occ_embedims, bg_k]

        # print(bg_V_curr.shape)
        # print("print(bg_V_curr.shape)")
        # print(bg_V_curr.shape)
        # print("print(bg_curr_feat.shape)")
        # print(bg_curr_feat.shape)
        # print("print(bg_V_prev.shape)")
        # print(bg_V_prev.shape)
        # print("print(bg_history_feat.shape)")
        # print(bg_history_feat.shape)
        # print("print(bg_occ_embed.shape)")
        # print(bg_occ_embed.shape)
        bg_fused = torch.cat([ bg_V_curr * bg_curr_feat,bg_V_curr,torch.cat([bg_V_prev.unsqueeze(2) * bg_history_feat.view(bs, self.history_num//2, c_, total_voxels - self.top_k),(bg_V_prev[:,0:self.history_num//2,:]).unsqueeze(2)],dim=2).reshape(bs, (self.history_num//2)*(c_+1), total_voxels - self.top_k)], dim=1).permute(0, 2, 1)
        # print("print(bg_fused.shape)")
        # print(bg_fused.shape)
        bg_fused_gate = self.history_bg_fusion_gate(bg_fused).permute(0, 2, 1).unsqueeze(2)  # [bs, T+1,1, N]
        bg_fused = (bg_fused.permute(0,2,1).reshape(bs, self.history_num//2+1, c_+1, total_voxels-self.top_k) * bg_fused_gate).sum(dim=1).squeeze(1)  # [bs, T+1,c_, N]
        
        # print("print(bg_fused.shape)")
        # print(bg_fused.shape)
        # print("print(bg_occ_embed.shape)")
        # print(bg_occ_embed.shape)
        # print(torch.cat([bg_fused,bg_occ_embed],dim=1).permute(0,2,1).shape)
        bg_fused = self.history_fusion_bg_linear(torch.cat([bg_fused,bg_occ_embed],dim=1).permute(0,2,1)).permute(0, 2, 1)  # [bs, c_, bg_k]

        # print()
        #TODO TODO 后续用senet实现一个通道注意力的版本  这个可能会更好？


        # # print("print(fg_history_feat_time.shape)")
        # # print(fg_history_feat_time.shape)
        # # print("print(fg_time_vis_weights.shape)")
        # # print(fg_time_vis_weights.shape)
        # # fg_history_agg = (fg_history_feat_time * fg_time_vis_weights.unsqueeze(2)).sum(dim=1)  # [bs, c_, K]).sum(dim=1)  # [bs, c_, K]]
        # fg_history_agg = (fg_history_feat_time * fg_time_vis_weights.unsqueeze(2)).sum(dim=1)  # [bs, c_, K]).sum(dim=1)  # [bs, c_, K]]


        # # print("print(fg_history_agg.shape)")
        # # print(fg_history_agg.shape)

        # # 可见性聚合与门控
        
        # # fg_V_prev_time = fg_V_prev.unsqueeze(1).repeat(1, self.history_num, 1)  # [bs, T, K]
        # # fg_V_prev_agg = (fg_V_prev_time * time_weights.squeeze(-1)).sum(dim=1)  # [bs, K]
        # fg_V_prev_agg = fg_V_prev.max(dim=1)[0]  # [bs, K]
        # fg_w_hist, fg_w_curr = self.compute_gate_weights(fg_V_prev_agg, fg_V_curr.squeeze(1))  # [bs, K, 1]

        # # 前景融合
        # fg_history_agg_perm = fg_history_agg.permute(0, 2, 1)  # [bs, K, c_]
        # fg_curr_perm = fg_curr_feat.permute(0, 2, 1)  # [bs, K, c_]
        # fg_fused = fg_w_hist * fg_history_agg_perm + fg_w_curr * fg_curr_perm  # [bs, K, c_]

        # # occupancy嵌入融合
        # last_occ_reshaped = last_occ_pred.permute(0, 3, 2, 1, 4).reshape(bs, -1, last_occ_pred.shape[-1])  # [bs, N, num_classes]
        # fg_occ_feat = torch.gather(last_occ_reshaped, dim=1, index=fg_indices.unsqueeze(-1).repeat(1, 1, last_occ_pred.shape[-1]))  # [bs, K, num_classes]
        # fg_occ_embed = self.occ_embedding(fg_occ_feat).permute(0, 2, 1)  # [bs, occ_embedims, K]
        # fg_fused = torch.cat([fg_fused, fg_occ_embed.permute(0, 2, 1)], dim=-1)  # [bs, K, c_+occ_embedims]
        # fg_fused = self.history_fusion_linear(fg_fused).permute(0, 2, 1)  # [bs, c_, K]

        # # 背景融合（原有代码）
        # bg_history_feat = torch.gather(history_bev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, mc, 1))  # [bs, mc//2, bg_k]
        # bg_curr_feat = torch.gather(curr_bev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, c_, 1))  # [bs, c_, bg_k]

        # bg_history_feat_time = bg_history_feat.reshape(bs, self.history_num, c_, -1)  # [bs, T, c_//2, bg_k]
        # # bg_history_agg = (bg_history_feat_time * time_weights).sum(dim=1)  # [bs, c_//2, bg_k]
        # bg_V_prev = torch.gather(V_prev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, self.history_num, 1))  # [bs, bg_k]
        # bg_V_curr = torch.gather(V_curr_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, 1, 1))  # [bs, bg_k]
        # #TODO 这个10的超参数？ 调整成可学习？
        # # bg_time_vis_weights = bg_V_prev/(bg_V_prev.sum(dim=1).unsqueeze(1) +1e-10) # [bs, 4,K]
        # bg_time_vis_weights =(bg_V_prev*(self.bg_scale.view(1,self.history_num,1))).softmax(dim=1)
        # bg_history_agg = (bg_history_feat_time * bg_time_vis_weights.unsqueeze(2)).sum(dim=1)  # [bs, c_, K]).sum(dim=1)  # [bs, c_, K]]



        # # bg_history_agg_perm = F.pad(bg_history_agg.permute(0, 2, 1), (0, c_ - c_//2, 0, 0))  # [bs, bg_k, c_]

        # bg_history_agg_perm = bg_history_agg.permute(0, 2, 1)  # [bs, bg_k, c_]
        # # bg_V_prev_time = bg_V_prev.unsqueeze(1).repeat(1, self.history_num, 1)  # [bs, T, bg_k]
        # # bg_V_prev_agg = (bg_V_prev_time * time_weights.squeeze(-1)).sum(dim=1)  # [bs, bg_k]
        # bg_V_prev_agg = bg_V_prev.max(dim=1)[0]  # [bs, bg_k]
        # bg_w_hist, bg_w_curr = self.compute_gate_weights(bg_V_prev_agg, bg_V_curr.squeeze(1))  # [bs, bg_k, 1]

        # bg_curr_perm = bg_curr_feat.permute(0, 2, 1)  # [bs, bg_k, c_]
        # # print("*"*50)
        # # print("print(bg_w_hist.shape)")
        # # print(bg_w_hist.shape)
        # # print("print(bg_w_curr.shape)")
        # # print(bg_w_curr.shape)
        # # print("print(bg_history_agg_perm.shape)")
        # # print(bg_history_agg_perm.shape)
        # # print("print(bg_curr_perm.shape)")        
        # # print(bg_curr_perm.shape)

        # # # 断言批次大小一致
        # # assert bg_w_hist.shape[0] == bg_history_agg_perm.shape[0] == bg_w_curr.shape[0] == bg_curr_perm.shape[0], \
        # #     f"批次大小不匹配: {bg_w_hist.shape[0]}, {bg_history_agg_perm.shape[0]}, {bg_w_curr.shape[0]}, {bg_curr_perm.shape[0]}"

        # # # 断言第二维度（bg_k）一致
        # # assert bg_w_hist.shape[1] == bg_history_agg_perm.shape[1] == bg_w_curr.shape[1] == bg_curr_perm.shape[1], \
        # #     f"bg_k维度不匹配: {bg_w_hist.shape[1]}, {bg_history_agg_perm.shape[1]}, {bg_w_curr.shape[1]}, {bg_curr_perm.shape[1]}"

        # # # 断言第三维度（c_）匹配（bg_w_hist和bg_w_curr的第三维为1，不影响广播）
        # # assert bg_history_agg_perm.shape[2] == bg_curr_perm.shape[2], \
        # #     f"特征维度c_不匹配: {bg_history_agg_perm.shape[2]} vs {bg_curr_perm.shape[2]}"

        # # print("bg_w_hist dtype:", bg_w_hist.dtype)
        # # print("bg_history_agg_perm dtype:", bg_history_agg_perm.dtype)
        # # print("bg_w_curr dtype:", bg_w_curr.dtype)
        # # print("bg_curr_perm dtype:", bg_curr_perm.dtype)


        # # print("bg_w_hist device:", bg_w_hist.device)
        # # print("bg_history_agg_perm device:", bg_history_agg_perm.device)
        # # print("bg_w_curr device:", bg_w_curr.device)
        # # print("bg_curr_perm device:", bg_curr_perm.device)


        # bg_fused = bg_w_hist * bg_history_agg_perm + bg_w_curr * bg_curr_perm  # [bs, bg_k, c_]
        # # 先验证乘法是否正常
        # # temp1 = bg_w_hist * bg_history_agg_perm
        # # temp2 = bg_w_curr * bg_curr_perm
        # # 再验证加法是否正常

        # # bg_w_hist = bg_w_hist.contiguous()
        # # bg_history_agg_perm = bg_history_agg_perm.contiguous()
        # # bg_w_curr = bg_w_curr.contiguous()
        # # bg_curr_perm = bg_curr_perm.contiguous()

        # # # 重新计算
        # # temp1 = bg_w_hist * bg_history_agg_perm
        # # temp2 = bg_w_curr * bg_curr_perm
        # # temp1 = temp1.contiguous()
        # # temp2 = temp2.contiguous()
        # # bg_fused = temp1 + temp2


        # # 转移所有张量到CPU
        # # bg_w_hist_cpu = bg_w_hist.cpu()
        # # bg_history_agg_perm_cpu = bg_history_agg_perm.cpu()
        # # bg_w_curr_cpu = bg_w_curr.cpu()
        # # bg_curr_perm_cpu = bg_curr_perm.cpu()

        # # # 分步执行运算
        # # try:
        # #     temp1_cpu = bg_w_hist_cpu * bg_history_agg_perm_cpu
        # #     temp2_cpu = bg_w_curr_cpu * bg_curr_perm_cpu
        # #     bg_fused_cpu = temp1_cpu + temp2_cpu
        # #     print(bg_fused_cpu)
        # #     print(bg_fused_cpu.shape)
        # #     print("CPU运算成功，无明显错误")
        # # except Exception as e:
        # #     print(f"CPU运算报错：{e}")  # 此处会显示具体错误原因（如值异常）


        # # bg_fused = temp1.clone() + temp2.clone()
        # # 1/0
        # bg_occ_feat = torch.gather(last_occ_reshaped, dim=1, index=bg_indices.unsqueeze(-1).repeat(1, 1, last_occ_pred.shape[-1]))  # [bs, bg_k, num_classes]
        # bg_occ_embed = self.occ_embedding(bg_occ_feat).permute(0, 2, 1)  # [bs, occ_embedims, bg_k]
        # bg_fused = torch.cat([bg_fused, bg_occ_embed.permute(0, 2, 1)], dim=-1)  # [bs, bg_k, c_+occ_embedims]
        # bg_fused = self.history_fusion_bg_linear(bg_fused).permute(0, 2, 1)  # [bs, c_, bg_k]

        # 更新当前BEV
        curr_bev_updated = curr_bev_flat.clone()
        # print("fg_fused")
        # print(fg_fused.shape)
        # print(fg_fused.mean())
        # print(fg_fused.max())
        # print(fg_fused.min())
        # print("bg_fused")
        # print(bg_fused.shape)
        # print(bg_fused.mean())
        # print(bg_fused.max())
        # print(bg_fused.min())
        curr_bev_updated.scatter_add_(dim=2, index=fg_indices.unsqueeze(1).repeat(1, c_, 1), src=fg_fused)
        curr_bev_updated.scatter_add_(dim=2, index=bg_indices.unsqueeze(1).repeat(1, c_, 1), src=bg_fused)
        curr_bev_updated = curr_bev_updated.reshape(bs, c_, z, h, w)  # 恢复原形状

        # 更新历史缓存
        self.history_last_bev = curr_bev_updated.detach().clone()
        self.history_bev = torch.cat([curr_bev,sampled_history_bev[:, :-c_, ...]], dim=1).detach()
        # self.history_cam_intrins = torch.cat([curr_cam_intrins.unsqueeze(1),self.history_cam_intrins[:, :-1, ...]], dim=1).detach()
        # self.history_cam_extrins = torch.cat([curr_cam_extrins.unsqueeze(1),self.history_cam_extrins[:, 1-1:, ...]], dim=1).detach()
        self.history_visibility =torch.cat([V_curr, V_prev[:, :-1, ...]],dim=1).detach()
        self.history_forward_augs = forward_augs.clone()
        torch.cuda.empty_cache()
        return curr_bev_updated

    def generate_grid(self, curr_bev):
        n, c_, z, h, w = curr_bev.shape
        # Generate grid
        xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w, 1).expand(h, w, z)
        ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1, 1).expand(h, w, z)
        zs = torch.linspace(0, z - 1, z, dtype=curr_bev.dtype, device=curr_bev.device).view(1, 1, z).expand(h, w, z)
        grid = torch.stack((xs, ys, zs, torch.ones_like(xs)), -1).view(1, h, w, z, 4).expand(n, h, w, z, 4).view(n, h,w, z, 4, 1)
        return grid

    def generate_feat2bev(self, grid, dx, bx):
        feat2bev = torch.zeros((4, 4), dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = dx[0]
        feat2bev[1, 1] = dx[1]
        feat2bev[2, 2] = dx[2]
        feat2bev[0, 3] = bx[0] - dx[0] / 2.
        feat2bev[1, 3] = bx[1] - dx[1] / 2.
        feat2bev[2, 3] = bx[2] - dx[2] / 2.
        feat2bev[3, 3] = 1
        feat2bev = feat2bev.view(1, 4, 4)
        return feat2bev
