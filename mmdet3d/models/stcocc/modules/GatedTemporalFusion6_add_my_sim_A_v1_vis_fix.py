import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import BaseModule, force_fp32
from mmcv.utils import TORCH_VERSION, digit_version

from mmdet3d.models.builder import HEADS
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32


def sharpen_depth_probs(depth_probs, temperature=0.1):
    """
    锐化深度bin的概率分布（沿深度bin维度做带温度的softmax）
    
    Args:
        depth_probs: 输入概率分布，shape=[2, 6, 156, 2, 88]
        temperature: 温度系数（越小越尖锐，建议0.01~1之间，默认0.1）
    
    Returns:
        sharpened_probs: 锐化后的概率分布，shape与输入一致
    """
    # 1. 确认输入维度（避免维度错误）
    assert depth_probs.dim() == 5, "输入必须是5维张量 [2,6,156,2,88]"
    assert depth_probs.shape[-1] == 88, "最后一维必须是深度bin数量88"
    
    # 2. 沿深度bin维度（最后一维，dim=-1）做温度缩放 + softmax
    # 温度缩放：将logits除以temperature（越小越集中）
    # 注：如果depth_probs已是概率，需先转logits（用softmax逆操作）
    if torch.all(depth_probs >= 0) and torch.allclose(depth_probs.sum(-1), torch.ones_like(depth_probs.sum(-1))):
        # 输入是概率分布，先转logits（加小epsilon避免log(0)）
        logits = torch.log(depth_probs + 1e-8)
    else:
        # 输入是logits，直接使用
        logits = depth_probs
    
    # 温度缩放 + softmax（核心锐化逻辑）
    sharpened_logits = logits / temperature
    sharpened_probs = torch.softmax(sharpened_logits, dim=-1)
    
    return sharpened_probs

@HEADS.register_module()
class GatedTemporalFusion6_add_sparsefusion_my_add_sim_A_v1_vis_fix(BaseModule):
    def __init__(
        self,
        history_num=4,
        depth_sampler_embed_dims=256,
        depth_score_mode='prob',
        visibility_depth_mode='linear',
        depth_sampler_num_heads=8,
        depth_sampler_num_levels=4,
        depth_sampler_num_points=4,
        im2col_step=64,
        top_k=500,
        single_bev_num_channels=96,
        occ_embedims=32,
        visibility_gamma=1.0,
        visibility_min=0.0,
        visibility_eps=1e-8,
        visibility_weight_mode='multiply',
        num_classes=18,
        vis_theta=0.28,
        vis_beta=10.0,
        vis_gamma=0.4,
        vis_sigma=0.1,
        nonempty_thresh=0.1,
        max_step_ratio=1.2,
        **kwargs
    ):
        super(GatedTemporalFusion6_add_sparsefusion_my_add_sim_A_v1_vis_fix, self).__init__()
        self.depth_score_mode = depth_score_mode
        self.visibility_depth_mode = visibility_depth_mode
        self.visibility_gamma = visibility_gamma
        self.visibility_min = visibility_min
        self.visibility_eps = visibility_eps
        self.visibility_weight_mode = visibility_weight_mode
        # 基础参数
        self.history_num = history_num
        self.top_k = top_k
        self.single_bev_num_channels = single_bev_num_channels
        self.occ_embedims = occ_embedims

        self.visible_embed = nn.Sequential(
                nn.Linear(6, 32),
                nn.Softplus(),
                nn.Linear(32, 1),
            )
        
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
        reference_points_cam = (lidar2img @ ego2lidar @ reference_points).squeeze(-1)   # [num_points_in_pillar, bs, num_cam, num_query=h*w, 4]
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
    def forward(self, curr_bev, cam_params, history_fusion_params, dx, bx, history_last_bev=None, last_occ_pred=None, nonempty_prob=None, img_feats=None, spatial_shapes=None,pred_img_depth=None,depth_bound=None,**kwargs):
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
        slots = torch.zeros(list([ref_3d.shape[0],6,ref_3d.shape[2],ref_3d.shape[1]])).to(ref_3d)
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

        depth_probs = MultiScaleDeformableAttnFunction_fp32.apply(pred_img_depth, spatial_shapes,level_start_index,depth_reference_points,depth_attention_weights, self.im2col_step)
        depth_probs = depth_probs.reshape(bs,6, max_len,z, -1)   # [bs*num_cam, num_query, num_Z_anchors, C]
        
        depth_probs = depth_probs.clamp(min=self.visibility_eps)
        depth_probs = depth_probs / depth_probs.sum(-1, keepdim=True)  # bs, num_query, num_Z_anchors, D_bins

        # Survival / transmittance per discrete depth bin:
        # V_k = 1 - sum_{j<k} P_j

        # depth_probs = sharpen_depth_probs(depth_probs, temperature=0.0001)
        visibility_bins = 1.0 - depth_probs.cumsum(dim=-1)
        visibility_bins = torch.cat(
            [torch.ones_like(visibility_bins[..., 0:1]), visibility_bins[..., 0:-1]],
            dim=-1,
        )
        visibility_bins = (bev_query_depth_rebatch * visibility_bins).sum(-1)

        # # Map continuous reference depth (meters) -> visibility scalar per anchor.
        # # Use depth_bound if provided; otherwise fallback to self.dbound.
        # dbound = depth_bound if depth_bound is not None else self.dbound
        # depth_start = reference_points_depth_rebatch.new_tensor(dbound[0])
        # depth_step = reference_points_depth_rebatch.new_tensor(dbound[2])
        # depth_bin = (reference_points_depth_rebatch - depth_start) / depth_step

        # num_depth_bins = visibility_bins.size(-1)
        # depth_bin = depth_bin.clamp(0.0, float(num_depth_bins - 1))

        # depth_bin_idx0 = depth_bin.floor().to(torch.long).clamp_(0, num_depth_bins - 1)
        # depth_bin_idx1 = (depth_bin_idx0 + 1).clamp_(max=num_depth_bins - 1)
        # frac = (depth_bin - depth_bin_idx0.to(depth_bin.dtype)).clamp(0.0, 1.0)

        # def _sample_bins(bins: torch.Tensor):
        #     val0 = bins.gather(-1, depth_bin_idx0)
        #     val1 = bins.gather(-1, depth_bin_idx1)
        #     val_linear = val0
        #     if num_depth_bins > 1:
        #         val_linear = val0 * (1.0 - frac) + val1 * frac
        #     if self.visibility_depth_mode == 'hard':
        #         val_used = val0
        #     elif self.visibility_depth_mode == 'linear':
        #         val_used = val_linear
        #     else:
        #         raise ValueError(f'Unsupported visibility_depth_mode={self.visibility_depth_mode!r}')
        #     return val0, val_linear, val_used

        # visibility_hard, visibility_linear, visibility_used = _sample_bins(visibility_bins)
        # prob_hard, prob_linear, prob_used = _sample_bins(depth_probs)
        # termination_hard, termination_linear, termination_used = _sample_bins(termination_bins)

        # visibility_hard = visibility_hard.clamp(0.0, 1.0)
        # visibility_linear = visibility_linear.clamp(0.0, 1.0)
        # visibility_used = visibility_used.clamp(0.0, 1.0)

        # prob_hard = prob_hard.clamp(0.0, 1.0)
        # prob_linear = prob_linear.clamp(0.0, 1.0)
        # prob_used = prob_used.clamp(0.0, 1.0)

        # termination_hard = termination_hard.clamp(0.0, 1.0)
        # termination_linear = termination_linear.clamp(0.0, 1.0)
        # termination_used = termination_used.clamp(0.0, 1.0)

        # if self.depth_score_mode == 'transmittance':
        #     depth_score = visibility_used
        # elif self.depth_score_mode == 'prob':
        #     depth_score = prob_used
        # elif self.depth_score_mode == 'termination':
        #     depth_score = termination_used
        # else:
        #     raise ValueError(f'Unsupported depth_score_mode={self.depth_score_mode!r}')

        # depth_score = depth_score.clamp(0.0, 1.0)
        # if self.visibility_gamma != 1.0:
        #     depth_score = depth_score.clamp(min=self.visibility_eps).pow(self.visibility_gamma)
        # if self.visibility_min > 0.0:
        #     depth_score = self.visibility_min + (1.0 - self.visibility_min) * depth_score

        # num_all_points = 1 * 1
        # num_anchors = depth_score.size(-1)
        # if num_all_points != num_anchors:
        #     if num_all_points % num_anchors != 0:
        #         raise ValueError(
        #             f'depth_score last dim ({num_anchors}) must divide attention_weights num_points '
        #             f'({num_all_points}). Consider setting num_points == num_Z_anchors.'
        #         )
        #     points_per_anchor = num_all_points // num_anchors
        #     depth_score_points = depth_score[:, :, None, :].expand(-1, -1, points_per_anchor, -1)
        #     depth_score_points = depth_score_points.reshape(bs, num_query, num_all_points)
        # else:
        #     depth_score_points = depth_score


        # if self.visibility_weight_mode == 'multiply':
        #     # attention_weights = depth_score_points[:, :, None, None, :]
        #     attention_weights = depth_score_points

        # elif self.visibility_weight_mode == 'multiply_renorm':
        #     attention_weights = depth_score_points[:, :, None, None, :]
        #     denom = attention_weights.sum(dim=(-1, -2), keepdim=True).clamp(min=self.visibility_eps)
        #     attention_weights = attention_weights / denom
        # else:
        #     raise ValueError(f'Unsupported visibility_weight_mode={self.visibility_weight_mode!r}')
       
        for j in range(bs):
            for i in range(6):
                index_query_per_img = indexes[j][i]
                # slots[j,i, index_query_per_img] = torch.max(slots[j, index_query_per_img],attention_weights[j, i, :len(index_query_per_img)])
                slots[j,i, index_query_per_img] = visibility_bins[j, i, :len(index_query_per_img)]


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
        slots[(bev_mask.permute(1, 0, 2, 3) == False)] = 0.0
        V_curr =slots.view(bs*6,1, h, w,z).permute(0, 1, 4, 2, 3).view(bs,6,1,z,h,w)
        

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
            
            self.history_forward_augs = forward_augs.clone()
            self.history_seq_ids = seq_ids.clone()
            self.history_bev = curr_bev.repeat(1, self.history_num, 1, 1, 1)
            self.history_sweep_time = curr_bev.new_zeros(curr_bev.shape[0], self.history_num)
            self.history_visibility = V_curr.repeat(1,1, self.history_num, 1, 1, 1).half()
        self.history_bev = self.history_bev.detach()
        self.history_visibility = self.history_visibility.detach().half()
        self.history_sweep_time += 1

        
        if start_of_sequence.sum()>0:
            self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_num, 1, 1, 1)
           
            self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]
            self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
            self.history_sweep_time[start_of_sequence] = 0  # zero the new sequence timestep starts
            self.history_visibility[start_of_sequence] = V_curr[start_of_sequence].repeat(1,1, self.history_num, 1, 1, 1).half()

        # 生成体素网格和BEV变换矩阵
        grid = self.generate_grid(curr_bev) #[bs,y,x,z,4]
        grid_3d = grid
        feat2bev = self.generate_feat2bev(grid, dx, bx)

        
        rt_flow = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_ego_rt @ torch.inverse(forward_augs) @ feat2bev)
       
        bs, mc, z, h, w = self.history_bev.shape
        n, c_, z, h, w = curr_bev.shape
        grid = rt_flow.view(n, 1, 1, 1, 4, 4) @ grid
        # -------------------------- 6. 采样网格生成后打印 --------------------------
        # 生成采样网格（归一化到[-1,1]，适配F.grid_sample）
        normalize_factor = torch.tensor([w - 1.0, h - 1.0, z - 1.0], dtype=curr_bev.dtype, device=device)

        grid = grid[:, :, :, :, :3, 0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0   # grid order is x, y, z


        sampled_history_bev = F.grid_sample(
            self.history_bev.reshape(bs, mc, z, h, w),  # 输入：[bs, mc, z, h, w]
            grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4),
            align_corners=True,
            mode='bilinear'
        )
        sampled_history_visibility = F.grid_sample(
            self.history_visibility.reshape(bs*6, self.history_num, z, h, w).half(),  # 输入：[bs, mc, z, h, w]
            grid[:,None].repeat(1,6,1,1,1,1).flatten(0,1).to(curr_bev.dtype).permute(0, 3, 1, 2, 4).half(),
            align_corners=True,
            mode='bilinear'
        )


        V_prev = sampled_history_visibility #bs,4,z,h,w


        curr_bev_flat = curr_bev.reshape(bs, c_, -1)  # [bs, c_, N], N=h*w*z
        history_bev_flat = sampled_history_bev.reshape(bs, mc, -1)  # [bs, mc, N]
        V_prev_flat = V_prev.reshape(bs*6,self.history_num, -1)  # [bs, 4,N]
        V_curr_flat = V_curr.reshape(bs*6, 1,-1)  # [bs, 1,N]
        nonempty_prob_flat = nonempty_prob.reshape(bs, -1)  # [bs, N]

        total_voxels = nonempty_prob_flat.shape[1]

        # print("\n9. 稀疏采样前展平变量形状：")

        # -------------------------- 10. 前景/背景索引及融合后打印（可选，确认后续维度） --------------------------
        # 生成前景/背景索引
        fg_indices = torch.topk(nonempty_prob_flat, self.top_k, dim=1)[1]
        bg_indices = torch.topk(1 - nonempty_prob_flat, total_voxels - self.top_k, dim=1)[1]
        # 提取前景特征（示例，其他融合步骤类似）
        fg_history_feat = torch.gather(history_bev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, mc, 1))

        # print("="*50)

        # -------------------------- 后续原有逻辑（略，维度已通过打印确认） --------------------------
        # 8. 前景融合（原有代码）
        # 9. 背景融合（原有代码）
        # 10. 更新当前BEV特征（原有代码）
       
        # # 前景特征提取
        # fg_history_feat = torch.gather(history_bev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, mc, 1))  # [bs, mc, top_k]
        fg_curr_feat = torch.gather(curr_bev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, c_, 1))  # [bs, c_, top_k]

        # 历史特征时间聚合
        # fg_history_feat_time = fg_history_feat.reshape(bs, self.history_num, c_, self.top_k)  # [bs, T, c_, K]
        #TODO 后续可以把time_weights也乘进去


        # time_weights = torch.exp(-0.5 * torch.arange(self.history_num, device=device)).view(1, self.history_num, 1, 1)

        fg_V_prev = torch.gather(V_prev_flat, dim=2, index=fg_indices[:,None].unsqueeze(1).repeat(1,6, self.history_num, 1).flatten(0,1))  # [bs, 4,K]
        fg_V_curr = torch.gather(V_curr_flat, dim=2, index=fg_indices[:,None].unsqueeze(1).repeat(1,6, 1, 1).flatten(0,1))  # [bs, 1,K]
    
        last_occ_pred = last_occ_pred.permute(0, 3, 2, 1, 4).reshape(bs, -1, last_occ_pred.shape[-1])  # [bs, N, num_classes]

        fg_occ_feat = torch.gather(last_occ_pred, dim=1, index=fg_indices.unsqueeze(-1).repeat(1, 1, last_occ_pred.shape[-1]))  # [bs, K, num_classes]

        fg_occ_embed = self.occ_embedding(fg_occ_feat).permute(0, 2, 1)  # [bs, occ_embedims, K]

    
        visible_feature = torch.cat([fg_V_curr.view(bs,6,-1,self.top_k),fg_V_prev.view(bs,6,-1,self.top_k)],dim=2).permute(0,2,3,1)
        # fg_fused = torch.cat([ fg_V_curr * fg_curr_feat,(fg_V_prev.unsqueeze(2) * fg_history_feat.view(bs, self.history_num, c_, self.top_k)).reshape(bs, self.history_num*c_, self.top_k) , fg_occ_embed], dim=1).permute(0, 2, 1)
        visible_embed=self.visible_embed(visible_feature).permute(0,1,3,2)
        fg_curr_feat = visible_embed[:,0,:,:]+fg_curr_feat
        fg_history_feat = (visible_embed[:,1:,:,:]+fg_history_feat.view(bs,-1,c_,self.top_k)).flatten(1,2)

        fg_fused = torch.cat([fg_curr_feat,fg_history_feat , fg_occ_embed], dim=1).permute(0, 2, 1)

        fg_fused = self.history_fusion_linear(fg_fused).permute(0, 2, 1)  # [bs, c_, K]


        # bg_V_prev = torch.gather(V_prev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, self.history_num//2, 1))  # [bs, bg_k]
        bg_V_prev = torch.gather(V_prev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1,6, self.history_num//2, 1).flatten(0,1))  # [bs, bg_k]

        # bg_V_curr = torch.gather(V_curr_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, 1, 1))  # [bs, bg_k]
        bg_V_curr = torch.gather(V_curr_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1,6, 1, 1).flatten(0,1))  # [bs, bg_k]

        bg_history_feat = torch.gather(history_bev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, mc//2, 1))  # [bs, mc//2, bg_k]
        bg_curr_feat = torch.gather(curr_bev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, c_, 1))  # [bs, c_, bg_k]
        bg_occ_feat = torch.gather(last_occ_pred, dim=1, index=bg_indices.unsqueeze(-1).repeat(1, 1, last_occ_pred.shape[-1]))  # [bs, bg_k, num_classes]
        bg_occ_embed = self.occ_embedding(bg_occ_feat).permute(0, 2, 1)  # [bs, occ_embedims, bg_k]
        
        visible_feature = torch.cat([bg_V_curr.view(bs,6,-1,total_voxels - self.top_k),bg_V_prev.view(bs,6,-1,total_voxels - self.top_k)],dim=2).permute(0,2,3,1)
        visible_embed=self.visible_embed(visible_feature).permute(0,1,3,2)
        bg_curr_feat = visible_embed[:,0,:,:]+bg_curr_feat
        bg_history_feat = (visible_embed[:,1:,:,:]+bg_history_feat.view(bs,-1,c_,total_voxels - self.top_k)).flatten(1,2)

        
        
        bg_fused = torch.cat([  bg_curr_feat, bg_history_feat, bg_occ_embed], dim=1).permute(0, 2, 1)
        
        bg_fused = self.history_fusion_bg_linear(bg_fused).permute(0, 2, 1)  # [bs, c_, bg_k]

        
        # 更新当前BEV
        curr_bev_updated = curr_bev_flat.clone()
        curr_bev_updated.scatter_add_(dim=2, index=fg_indices.unsqueeze(1).repeat(1, c_, 1), src=fg_fused)
        curr_bev_updated.scatter_add_(dim=2, index=bg_indices.unsqueeze(1).repeat(1, c_, 1), src=bg_fused)
        curr_bev_updated = curr_bev_updated.reshape(bs, c_, z, h, w)  # 恢复原形状

        # 更新历史缓存
        self.history_last_bev = curr_bev_updated.detach().clone()
        self.history_bev = torch.cat([curr_bev,sampled_history_bev[:, :-c_, ...]], dim=1).detach()
        self.history_visibility =torch.cat([V_curr, V_prev.view(bs,6,V_prev.size(1),V_prev.size(2),V_prev.size(3),V_prev.size(4))[:, :,:-1, ...]],dim=2).detach()
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