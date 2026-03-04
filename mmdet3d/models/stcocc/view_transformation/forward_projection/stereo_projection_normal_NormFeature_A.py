# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import NECKS
from .bevdet_utils.lss_transformation import LSSForwardProjection, LSSVStereoForwardPorjection
from mmdet3d.models.necks.depth_net_normal import DepthNetWithNormal
from mmdet3d.models.necks.depth_net_normal_improved import DepthNetWithNormalImproved
from mmdet3d.models.necks.depth_net_normal_isolated import DepthNetWithNormalIsolated
from torch.cuda.amp.autocast_mode import autocast
import time 


@NECKS.register_module()
class LSSForwardProjectionWithNormal_NormFeature_A(LSSForwardProjection):
    """
    扩展版的LSSForwardProjection，增加了对normal_comp相关bins的预测能力
    使用DepthNetWithNormal替代原始的DepthNet
    """
    
    def __init__(self,
                 loss_depth_weight=3.0,
                 loss_normal_weight=1.0,  # normal预测的损失权重
                 depthnet_cfg=dict(),
                 return_context=False,
                 normal_w_bins=8,        # normal_comp_W_bins的类别数
                 normal_h_bins=8,        # normal_comp_H_bins的类别数
                 normal_depth_bins=4,    # normal_comp_depth_bins的类别数
                 **kwargs):
        """
        初始化扩展的前向投影模块
        
        Args:
            loss_depth_weight: 深度损失的权重
            loss_normal_weight: normal损失的权重
            depthnet_cfg: 深度网络的配置
            return_context: 是否返回上下文特征
            normal_w_bins: normal_comp_W_bins的类别数
            normal_h_bins: normal_comp_H_bins的类别数
            normal_depth_bins: normal_comp_depth_bins的类别数
            **kwargs: 其他传递给LSSForwardProjection的参数
        """
        # 保存normal bins的配置信息
        self.normal_w_bins = normal_w_bins
        self.normal_h_bins = normal_h_bins
        self.normal_depth_bins = normal_depth_bins
        self.loss_normal_weight = loss_normal_weight
        self.return_context = return_context
        
        # 初始化父类，但不创建原始的depth_net
        super(LSSForwardProjection, self).__init__(**kwargs)  # 直接调用祖父类初始化
        
        # 使用扩展的DepthNetWithNormal替代原始的DepthNet
        self.loss_depth_weight = loss_depth_weight
        self.depth_net = DepthNetWithNormal(
            self.in_channels,
            self.in_channels,
            context_channels=self.out_channels,
            depth_channels=self.D,
            normal_w_bins=normal_w_bins,
            normal_h_bins=normal_h_bins,
            normal_depth_bins=normal_depth_bins,
            **depthnet_cfg
        )
    
    def forward(self, input, stereo_metas=None):
        """
        前向传播函数，扩展了原始LSSForwardProjection的功能
        
        Args:
            input: 输入数据，包含x, rots, trans, intrins, post_rot, post_aug, bda, mlp_input
            stereo_metas: 立体视觉相关的元数据
            
        Returns:
            tuple: (bev_feat, depth, tran_feat, normal_preds)
                bev_feat: BEV特征
                depth: 深度预测
                tran_feat: 转换特征
                normal_preds: 包含normal_comp_W_bins, normal_comp_H_bins, normal_comp_depth_bins预测的字典
        """
        x, rots, trans, intrins, post_rot, post_aug, bda, mlp_input = input[:8]
        1/0
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)

        # 使用扩展的深度网络获取预测结果
        combined_output = self.depth_net(x, mlp_input, stereo_metas)
        
        # 分割不同的预测结果
        depth_digit = combined_output[:, :self.D, ...]  # 原始深度预测
        normal_w_digit = combined_output[:, self.D:self.D+self.normal_w_bins, ...]  # normal_comp_W_bins预测
        normal_h_digit = combined_output[:, self.D+self.normal_w_bins:self.D+self.normal_w_bins+self.normal_h_bins, ...]  # normal_comp_H_bins预测
        normal_depth_digit = combined_output[:, self.D+self.normal_w_bins+self.normal_h_bins:self.D+self.normal_w_bins+self.normal_h_bins+self.normal_depth_bins, ...]  # normal_comp_depth_bins预测
        tran_feat = combined_output[:, self.D+self.normal_w_bins+self.normal_h_bins+self.normal_depth_bins:, ...]  # 上下文特征
        
        # 对深度预测进行softmax
        depth = depth_digit.softmax(dim=1)
        
        # 对normal bins预测进行softmax
        normal_w_prob = normal_w_digit.softmax(dim=1)
        normal_h_prob = normal_h_digit.softmax(dim=1)
        normal_depth_prob = normal_depth_digit.softmax(dim=1)
        
        # 构建normal预测结果字典
        normal_preds = {
            'normal_comp_W_bins': normal_w_prob,
            'normal_comp_H_bins': normal_h_prob,
            'normal_comp_depth_bins': normal_depth_prob
        }
        
        # 将normal概率从相机坐标系变换到ego坐标系
        # 注意：这里的rots是相机坐标系到ego坐标系的旋转矩阵
        normal_preds = self.transform_normal_to_ego(normal_preds, rots, trans)
        1/0
        # 如果需要在后续步骤中变换到ego坐标系，可以在使用时调用transform_normal_to_ego函数
        normal_preds = self.transform_normal_to_ego(normal_preds, rots, trans)
        # 进行视图变换
        bev_feat, depth = self.view_transform(input, depth, tran_feat)
        
        if self.return_context:
            return bev_feat, depth, tran_feat, normal_preds
        else:
            return bev_feat, depth, normal_preds
    
    @torch.no_grad()
    def get_downsampled_gt_normal(self, gt_normals, bins):
        """
        下采样真实的normal bins标签
        
        Args:
            gt_normals: 真实的normal bins标签 [B, N, H, W]
            bins: bins的数量
            
        Returns:
            torch.Tensor: 下采样后的one-hot编码normal标签 [B*N, H//downsample, W//downsample, bins]
        """
        B, N, H, W = gt_normals.shape
        
        # 下采样（使用最大值池化来保持bin索引）
        gt_normals = gt_normals.view(B * N, 1, H, W)
        gt_normals = F.max_pool2d(gt_normals, kernel_size=self.downsample, stride=self.downsample)
        gt_normals = gt_normals.squeeze(1).long()
        
        # 转换为one-hot编码
        gt_normals = F.one_hot(gt_normals, num_classes=bins)
        
        return gt_normals
    
    def transform_normal_to_ego(self, normal_preds, input):
        """
        将normal概率从相机坐标系变换到ego坐标系
        
        Args:
            normal_preds: 包含normal_comp_W_bins, normal_comp_H_bins, normal_comp_depth_bins预测的字典
            input: 输入数据，包含x, sensor2keyego, ego2global, intrins, post_rot, post_aug, bda, mlp_input
            
        Returns:
            dict: ego坐标系下的normal角度数值，包含ego_w_angle、ego_h_angle、ego_d_angle
        """
        import torch
        import torch.nn.functional as F
        
        # 从input中解包参数
        x, sensor2keyego, ego2global, intrins, post_rot, post_aug, bda, mlp_input = input[:8]
        
        # 获取概率分布
        normal_w_prob = normal_preds['normal_comp_W_bins']
        normal_h_prob = normal_preds['normal_comp_H_bins']
        normal_depth_prob = normal_preds['normal_comp_depth_bins']
        
        # 获取形状信息
        B, N, _, _ = sensor2keyego.shape
        H, W = normal_w_prob.shape[-2:]  # [B*N, num_bins, H, W]
        
        # 创建bin中心角度（弧度制）
        # W方向和H方向都是0-180度，D方向是0-90度
        num_w_bins = normal_w_prob.shape[1]
        num_h_bins = normal_h_prob.shape[1]
        num_d_bins = normal_depth_prob.shape[1]
        
        # 计算bin中心角度（弧度制）
        w_angles_rad = torch.linspace(0, torch.pi, num_w_bins, device=normal_w_prob.device)
        h_angles_rad = torch.linspace(0, torch.pi, num_h_bins, device=normal_w_prob.device)
        d_angles_rad = torch.linspace(0, torch.pi/2, num_d_bins, device=normal_w_prob.device)
        
        # 计算期望角度
        w_expected_angle = torch.sum(normal_w_prob * w_angles_rad.view(1, -1, 1, 1), dim=1)
        h_expected_angle = torch.sum(normal_h_prob * h_angles_rad.view(1, -1, 1, 1), dim=1)
        d_expected_angle = torch.sum(normal_depth_prob * d_angles_rad.view(1, -1, 1, 1), dim=1)
        
        # 从球面坐标转换为笛卡尔坐标（方向向量）
        # theta = h_expected_angle (极角，与z轴夹角)
        # phi = w_expected_angle (方位角，在xy平面上的投影与x轴夹角)
        # r = 1 (单位向量)
        sin_theta = torch.sin(h_expected_angle)
        cos_theta = torch.cos(h_expected_angle)
        sin_phi = torch.sin(w_expected_angle)
        cos_phi = torch.cos(w_expected_angle)
        
        # 计算方向向量的x, y, z分量
        # 注意：这里使用的是相机坐标系的约定
        x_dir = sin_theta * cos_phi
        y_dir = sin_theta * sin_phi
        z_dir = cos_theta
        
        # 归一化方向向量
        dir_vector = torch.stack([x_dir, y_dir, z_dir], dim=1)  # [B*N, 3, H, W]
        dir_vector = F.normalize(dir_vector, dim=1)
        
        # 重塑方向向量以适应后续变换
        # 重塑为 [B*N, 3, H*W] 以进行批量处理
        dir_vector_reshaped = dir_vector.view(B*N, 3, -1)  # [B*N, 3, H*W]
        
        # 处理后旋转（逆变换）
        # 重塑post_rot以匹配dir_vector_reshaped的形状
        post_rot_inv = torch.inverse(post_rot).view(B*N, 3, 3)  # [B*N, 3, 3]
        dir_vector = torch.bmm(post_rot_inv, dir_vector_reshaped)  # [B*N, 3, H*W]
        
        # 处理相机内参（逆变换）
        intrins_inv = torch.inverse(intrins)[:, :, :3, :3].view(B*N, 3, 3)  # [B*N, 3, 3]
        dir_vector = torch.bmm(intrins_inv, dir_vector)  # [B*N, 3, H*W]
        
        # 应用sensor2keyego变换（相机到ego坐标系）
        sensor2keyego_rot = sensor2keyego[:, :, :3, :3].view(B*N, 3, 3)  # [B*N, 3, 3]
        dir_vector = torch.bmm(sensor2keyego_rot, dir_vector)  # [B*N, 3, H*W]
        
        # 应用BDD变换
        if bda is not None:
            bda_rot = bda[:, :3, :3].view(B, 1, 3, 3).expand(B, N, 3, 3).contiguous().view(B*N, 3, 3)
            dir_vector = torch.bmm(bda_rot, dir_vector)  # [B*N, 3, H*W]
        
        # 重塑回原始形状
        ego_dir_vector = dir_vector.view(B*N, 3, H, W)
        
        # 归一化旋转后的方向向量
        ego_dir_vector = F.normalize(ego_dir_vector, dim=1)
        
        # 将旋转后的方向向量转换回角度表示
        # 计算极角theta (与z轴夹角)
        ego_z = ego_dir_vector[:, 2]
        ego_h_angle = torch.acos(torch.clamp(ego_z, -1.0, 1.0))  # [B*N, H, W]
        
        # 计算方位角phi (在xy平面上的投影与x轴夹角)
        ego_x = ego_dir_vector[:, 0]
        ego_y = ego_dir_vector[:, 1]
        ego_w_angle = torch.atan2(ego_y, ego_x)  # [B*N, H, W]
        
        # 归一化phi到0-2pi范围
        ego_w_angle = ego_w_angle % (2 * torch.pi)
        
        # 计算与y轴的夹角（D方向）
        ego_d_angle = torch.acos(torch.clamp(ego_y, -1.0, 1.0))
        
        # 构建ego坐标系下的normal角度结果
        ego_normal_results = {
            'ego_w_angle': ego_w_angle,  # W方向角度（弧度制）
            'ego_h_angle': ego_h_angle,  # H方向角度（弧度制）
            'ego_d_angle': ego_d_angle,  # D方向角度（弧度制）
            'gt_depth': normal_preds.get('gt_depth', None)
        }
        
        return ego_normal_results
    
    def get_normal_loss(self, normal_labels, normal_preds):
        """
        计算normal预测的损失
        
        Args:
            normal_labels: 真实的normal标签字典
            normal_preds: 预测的normal概率字典
            
        Returns:
            torch.Tensor: normal预测的总损失
        """
        total_loss = 0.0
        
        # 计算normal_comp_W_bins的损失
        if 'normal_comp_W_bins' in normal_labels and 'normal_comp_W_bins' in normal_preds:
            gt_w = self.get_downsampled_gt_normal(normal_labels['normal_comp_W_bins'], self.normal_w_bins)
            pred_w = normal_preds['normal_comp_W_bins'].permute(0, 2, 3, 1).contiguous()
            
            # 计算交叉熵损失
            w_loss = F.cross_entropy(
                pred_w.reshape(-1, self.normal_w_bins),
                gt_w.reshape(-1, self.normal_w_bins)
            )
            total_loss += w_loss
        
        # 计算normal_comp_H_bins的损失
        if 'normal_comp_H_bins' in normal_labels and 'normal_comp_H_bins' in normal_preds:
            gt_h = self.get_downsampled_gt_normal(normal_labels['normal_comp_H_bins'], self.normal_h_bins)
            pred_h = normal_preds['normal_comp_H_bins'].permute(0, 2, 3, 1).contiguous()
            
            # 计算交叉熵损失
            h_loss = F.cross_entropy(
                pred_h.reshape(-1, self.normal_h_bins),
                gt_h.reshape(-1, self.normal_h_bins)
            )
            total_loss += h_loss
        
        # 计算normal_comp_depth_bins的损失
        if 'normal_comp_depth_bins' in normal_labels and 'normal_comp_depth_bins' in normal_preds:
            gt_depth = self.get_downsampled_gt_normal(normal_labels['normal_comp_depth_bins'], self.normal_depth_bins)
            pred_depth = normal_preds['normal_comp_depth_bins'].permute(0, 2, 3, 1).contiguous()
            
            # 计算交叉熵损失
            depth_loss = F.cross_entropy(
                pred_depth.reshape(-1, self.normal_depth_bins),
                gt_depth.reshape(-1, self.normal_depth_bins)
            )
            total_loss += depth_loss
        
        # 应用权重
        return self.loss_normal_weight * total_loss


@NECKS.register_module()
class LSSVStereoForwardProjectionWithNormal_NormFeature_A(LSSVStereoForwardPorjection):
    """
    立体视觉版本的LSSForwardProjectionWithNormal
    继承了LSSVStereoForwardPorjection的所有功能，并添加了normal预测支持
    """
    
    def __init__(self, cv_downsample=4, loss_depth_weight=3.0, loss_normal_weight=1.0,
                 depthnet_cfg=dict(), return_context=False,use_ego_2D_3D_trans_fix=False,
                 normal_w_bins=8, normal_h_bins=8, normal_depth_bins=4, **kwargs):
        """
        初始化立体视觉版本的扩展前向投影模块
        
        Args:
            cv_downsample: CV下采样因子
            loss_depth_weight: 深度损失的权重
            loss_normal_weight: normal损失的权重
            depthnet_cfg: 深度网络的配置
            return_context: 是否返回上下文特征
            normal_w_bins: normal_comp_W_bins的类别数
            normal_h_bins: normal_comp_H_bins的类别数
            normal_depth_bins: normal_comp_depth_bins的类别数
            **kwargs: 其他传递给LSSVStereoForwardPorjection的参数
        """
        # 保存normal bins的配置信息
        self.normal_w_bins = normal_w_bins
        self.normal_h_bins = normal_h_bins
        self.normal_depth_bins = normal_depth_bins
        self.loss_normal_weight = loss_normal_weight
        self.return_context = return_context
        self.W_bins = 8
        self.H_bins = 8
        self.D_bins = 4
        self.loss_normal_W_weight = 1.0
        self.loss_normal_H_weight = 1.0
        self.loss_normal_D_weight = 1.0
        # 初始化父类
        self.use_ego_2D_3D_trans_fix = use_ego_2D_3D_trans_fix

        super().__init__(cv_downsample=cv_downsample, **kwargs)
        
        # 使用扩展的DepthNetWithNormal替代原始的depth_net
        self.loss_depth_weight = loss_depth_weight
        self.depth_net = DepthNetWithNormal(
            self.in_channels,
            self.in_channels,
            context_channels=self.out_channels,
            depth_channels=self.D,
            normal_w_bins=normal_w_bins,
            normal_h_bins=normal_h_bins,
            normal_depth_bins=normal_depth_bins,
            **depthnet_cfg
        )

    def transform_normal_to_ego(self, normal_preds, input):
        """
        将normal概率从相机坐标系变换到ego坐标系
        
        Args:
            normal_preds: 包含normal_comp_W_bins, normal_comp_H_bins, normal_comp_depth_bins预测的字典
            input: 输入数据，包含x, sensor2keyego, ego2global, intrins, post_rot, post_aug, bda, mlp_input
            
        Returns:
            dict: ego坐标系下的normal角度数值，包含ego_w_angle、ego_h_angle、ego_d_angle
        """
        # import torch
        # import torch.nn.functional as F
        
        # 从input中解包参数
        x, sensor2keyego, ego2global, intrins, post_rot, post_aug, bda, mlp_input = input[:8]
        
        # 获取概率分布
        normal_w_prob = normal_preds['normal_comp_W_bins']
        normal_h_prob = normal_preds['normal_comp_H_bins']
        normal_depth_prob = normal_preds['normal_comp_depth_bins']
        
        # 获取形状信息
        B, N, _, _ = sensor2keyego.shape
        H, W = normal_w_prob.shape[-2:]  # [B*N, num_bins, H, W]
        
        # 创建bin中心角度（弧度制）
        # W方向和H方向都是0-180度，D方向是0-90度
        num_w_bins = normal_w_prob.shape[1]
        num_h_bins = normal_h_prob.shape[1]
        num_d_bins = normal_depth_prob.shape[1]
        
        # 计算bin中心角度（弧度制）
        w_angles_rad = torch.linspace(0, torch.pi-torch.pi/num_w_bins, num_w_bins, device=normal_w_prob.device)
        h_angles_rad = torch.linspace(0, torch.pi-torch.pi/num_h_bins, num_h_bins, device=normal_w_prob.device)
        d_angles_rad = torch.linspace(0, torch.pi/2-torch.pi/2/num_d_bins, num_d_bins, device=normal_w_prob.device)
        
        # 计算期望角度
        w_expected_angle = torch.sum(normal_w_prob * w_angles_rad.view(1, -1, 1, 1), dim=1)
        h_expected_angle = torch.sum(normal_h_prob * h_angles_rad.view(1, -1, 1, 1), dim=1)
        d_expected_angle = torch.sum(normal_depth_prob * d_angles_rad.view(1, -1, 1, 1), dim=1)
        
        # 从球面坐标转换为笛卡尔坐标（方向向量）
        # theta = h_expected_angle (极角，与z轴夹角)
        # phi = w_expected_angle (方位角，在xy平面上的投影与x轴夹角)
        # r = 1 (单位向量)
        sin_theta = torch.sin(h_expected_angle)
        cos_theta = torch.cos(h_expected_angle)
        sin_phi = torch.sin(w_expected_angle)
        cos_phi = torch.cos(w_expected_angle)
        
        # 计算方向向量的x, y, z分量
        # 注意：这里使用的是相机坐标系的约定
        x_dir = sin_theta * cos_phi
        y_dir = sin_theta * sin_phi
        z_dir = cos_theta
        
        # 归一化方向向量
        dir_vector = torch.stack([x_dir, y_dir, z_dir], dim=1)  # [B*N, 3, H, W]
        dir_vector = F.normalize(dir_vector, dim=1)
        cam_dir_vector = dir_vector
        # 重塑方向向量以适应后续变换
        # 重塑为 [B*N, 3, H*W] 以进行批量处理
        dir_vector_reshaped = dir_vector.view(B*N, 3, -1)  # [B*N, 3, H*W]
        
        # 处理后旋转（逆变换）
        # 重塑post_rot以匹配dir_vector_reshaped的形状
        post_rot_inv = torch.inverse(post_rot).view(B*N, 3, 3)  # [B*N, 3, 3]
        dir_vector = torch.bmm(post_rot_inv, dir_vector_reshaped)  # [B*N, 3, H*W]
        
        # 处理相机内参（逆变换）
        intrins_inv = torch.inverse(intrins)[:, :, :3, :3].view(B*N, 3, 3)  # [B*N, 3, 3]
        dir_vector = torch.bmm(intrins_inv, dir_vector)  # [B*N, 3, H*W]
        
        # 应用sensor2keyego变换（相机到ego坐标系）
        sensor2keyego_rot = sensor2keyego[:, :, :3, :3].reshape(B*N, 3, 3)  # [B*N, 3, 3]
        dir_vector = torch.bmm(sensor2keyego_rot, dir_vector)  # [B*N, 3, H*W]
        
        # 应用BDD变换
        if bda is not None:
            bda_rot = bda[:, :3, :3].view(B, 1, 3, 3).expand(B, N, 3, 3).contiguous().view(B*N, 3, 3)
            dir_vector = torch.bmm(bda_rot, dir_vector)  # [B*N, 3, H*W]
        
        # 重塑回原始形状
        ego_dir_vector = dir_vector.view(B*N, 3, H, W)
        
        # 归一化旋转后的方向向量
        ego_dir_vector = F.normalize(ego_dir_vector, dim=1)
        
        # 将旋转后的方向向量转换回角度表示
        # 计算极角theta (与z轴夹角)
        ego_z = ego_dir_vector[:, 2]
        ego_h_angle = torch.acos(torch.clamp(ego_z, -1.0, 1.0))  # [B*N, H, W]
        
        # 计算方位角phi (在xy平面上的投影与x轴夹角)
        ego_x = ego_dir_vector[:, 0]
        ego_y = ego_dir_vector[:, 1]
        ego_w_angle = torch.atan2(ego_y, ego_x)  # [B*N, H, W]
        
        # 归一化phi到0-2pi范围
        ego_w_angle = ego_w_angle % (2 * torch.pi)
        
        # 计算与y轴的夹角（D方向）
        ego_d_angle = torch.acos(torch.clamp(ego_y, -1.0, 1.0))
        
        # 构建ego坐标系下的normal角度结果
        ego_normal_results = {
            'ego_w_angle': ego_w_angle,  # W方向角度（弧度制）
            'ego_h_angle': ego_h_angle,  # H方向角度（弧度制）
            'ego_d_angle': ego_d_angle,  # D方向角度（弧度制）
            'ego_dir_vector': ego_dir_vector,  # 归一化后的方向向量
            'gt_depth': normal_preds.get('gt_depth', None),
            'cam_dir_vector': cam_dir_vector
        }
        
        return ego_normal_results
    
    def transform_normal_to_ego_fix_remove_intrins_K(self, normal_preds, input):
        """
        将normal概率从相机坐标系变换到ego坐标系
        【核心修改】移除了多余的相机内参逆变换（因法向量来自LiDAR物理3D推导，无内参畸变）
        
        Args:
            normal_preds: 包含normal_comp_W_bins, normal_comp_H_bins, normal_comp_depth_bins预测的字典
            input: 输入数据，包含x, sensor2keyego, ego2global, intrins, post_rot, post_aug, bda, mlp_input
            
        Returns:
            dict: ego坐标系下的normal角度数值，包含ego_w_angle、ego_h_angle、ego_d_angle
        """
        # 从input中解包参数（intrins保留但不再使用，避免修改输入解包逻辑）
        x, sensor2keyego, ego2global, intrins, post_rot, post_aug, bda, mlp_input = input[:8]
        
        # 获取概率分布
        normal_w_prob = normal_preds['normal_comp_W_bins']
        normal_h_prob = normal_preds['normal_comp_H_bins']
        normal_depth_prob = normal_preds['normal_comp_depth_bins']
        
        # 获取形状信息
        B, N, _, _ = sensor2keyego.shape
        H, W = normal_w_prob.shape[-2:]  # [B*N, num_bins, H, W]
        
        # 创建bin中心角度（弧度制）
        # W方向和H方向都是0-180度，D方向是0-90度
        num_w_bins = normal_w_prob.shape[1]
        num_h_bins = normal_h_prob.shape[1]
        num_d_bins = normal_depth_prob.shape[1]
        
        # 计算bin中心角度（弧度制）
        w_angles_rad = torch.linspace(0, torch.pi-torch.pi/num_w_bins, num_w_bins, device=normal_w_prob.device)
        h_angles_rad = torch.linspace(0, torch.pi-torch.pi/num_h_bins, num_h_bins, device=normal_w_prob.device)
        d_angles_rad = torch.linspace(0, torch.pi/2-torch.pi/2/num_d_bins, num_d_bins, device=normal_w_prob.device)
        
        # 计算期望角度
        w_expected_angle = torch.sum(normal_w_prob * w_angles_rad.view(1, -1, 1, 1), dim=1)
        h_expected_angle = torch.sum(normal_h_prob * h_angles_rad.view(1, -1, 1, 1), dim=1)
        d_expected_angle = torch.sum(normal_depth_prob * d_angles_rad.view(1, -1, 1, 1), dim=1)
        
        # 从球面坐标转换为笛卡尔坐标（方向向量）
        # theta = h_expected_angle (极角，与z轴夹角)
        # phi = w_expected_angle (方位角，在xy平面上的投影与x轴夹角)
        # r = 1 (单位向量)
        sin_theta = torch.sin(h_expected_angle)
        cos_theta = torch.cos(h_expected_angle)
        sin_phi = torch.sin(w_expected_angle)
        cos_phi = torch.cos(w_expected_angle)
        
        # 计算方向向量的x, y, z分量（相机物理坐标系，无内参畸变）
        x_dir = sin_theta * cos_phi
        y_dir = sin_theta * sin_phi
        z_dir = cos_theta
        
        # 归一化方向向量
        dir_vector = torch.stack([x_dir, y_dir, z_dir], dim=1)  # [B*N, 3, H, W]
        dir_vector = F.normalize(dir_vector, dim=1)
        cam_dir_vector = dir_vector  # 保存相机坐标系原始方向向量
        
        # 重塑方向向量以适应后续变换：[B*N, 3, H*W]
        dir_vector_reshaped = dir_vector.view(B*N, 3, -1)
        
        # 1. 处理后旋转逆变换（必须保留：对齐图像增强前的相机轴）
        post_rot_inv = torch.inverse(post_rot).view(B*N, 3, 3)  # [B*N, 3, 3]
        dir_vector = torch.bmm(post_rot_inv, dir_vector_reshaped)  # [B*N, 3, H*W]
        
        # 【关键删除】移除相机内参逆变换（法向量来自LiDAR物理3D，无内参畸变，无需校正）
        # --- 原错误代码已删除 ---
        # intrins_inv = torch.inverse(intrins)[:, :, :3, :3].view(B*N, 3, 3)
        # dir_vector = torch.bmm(intrins_inv, dir_vector)
        
        # 2. 应用sensor2keyego变换（相机坐标系 → ego坐标系，必须保留）
        sensor2keyego_rot = sensor2keyego[:, :, :3, :3].reshape(B*N, 3, 3)  # [B*N, 3, 3]
        dir_vector = torch.bmm(sensor2keyego_rot, dir_vector)  # [B*N, 3, H*W]
        
        # 3. 应用BDD变换（可选，根据数据集需求保留）
        if bda is not None:
            bda_rot = bda[:, :3, :3].view(B, 1, 3, 3).expand(B, N, 3, 3).contiguous().view(B*N, 3, 3)
            dir_vector = torch.bmm(bda_rot, dir_vector)  # [B*N, 3, H*W]
        
        # 重塑回原始形状：[B*N, 3, H, W]
        ego_dir_vector = dir_vector.view(B*N, 3, H, W)
        
        # 归一化旋转后的方向向量（确保单位向量）
        ego_dir_vector = F.normalize(ego_dir_vector, dim=1)
        
        # 将ego坐标系方向向量转换回角度表示
        # 极角theta (与z轴夹角) → ego_h_angle
        ego_z = ego_dir_vector[:, 2]
        ego_h_angle = torch.acos(torch.clamp(ego_z, -1.0, 1.0))  # [B*N, H, W]
        
        # 方位角phi (xy平面投影与x轴夹角) → ego_w_angle
        ego_x = ego_dir_vector[:, 0]
        ego_y = ego_dir_vector[:, 1]
        ego_w_angle = torch.atan2(ego_y, ego_x)  # [B*N, H, W]
        ego_w_angle = ego_w_angle % (2 * torch.pi)  # 归一化到0-2pi
        
        # 与y轴的夹角 → ego_d_angle
        ego_d_angle = torch.acos(torch.clamp(ego_y, -1.0, 1.0))
        
        # 构建返回结果
        ego_normal_results = {
            'ego_w_angle': ego_w_angle,  # W方向角度（弧度制）
            'ego_h_angle': ego_h_angle,  # H方向角度（弧度制）
            'ego_d_angle': ego_d_angle,  # D方向角度（弧度制）
            'ego_dir_vector': ego_dir_vector,  # ego坐标系归一化方向向量
            'gt_depth': normal_preds.get('gt_depth', None),
            'cam_dir_vector': cam_dir_vector  # 相机坐标系原始方向向量
        }
        
        return ego_normal_results

    def forward(self, input, stereo_metas=None):
        """
        前向传播函数，扩展了原始LSSVStereoForwardPorjection的功能
        
        Args:
            input: 输入数据，包含x, rots, trans, intrins, post_rot, post_aug, bda, mlp_input
            stereo_metas: 立体视觉相关的元数据
            
        Returns:
            tuple: (bev_feat, depth, tran_feat, normal_preds)
                bev_feat: BEV特征
                depth: 深度预测
                tran_feat: 转换特征
                normal_preds: 包含normal_comp_W_bins, normal_comp_H_bins, normal_comp_depth_bins预测的字典
        """
        x, sensor2keyego, ego2global, intrins, post_rot, post_aug, bda, mlp_input = input[:8]

        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        start_time = time.time()
        
        # 使用扩展的深度网络获取预测结果
        combined_output = self.depth_net(x, mlp_input, stereo_metas)
        end_time = time.time()
        
        # 分割不同的预测结果
        depth_digit = combined_output[:, :self.D, ...]  # 原始深度预测
        normal_w_digit = combined_output[:, self.D:self.D+self.normal_w_bins, ...]  # normal_comp_W_bins预测
        normal_h_digit = combined_output[:, self.D+self.normal_w_bins:self.D+self.normal_w_bins+self.normal_h_bins, ...]  # normal_comp_H_bins预测
        normal_depth_digit = combined_output[:, self.D+self.normal_w_bins+self.normal_h_bins:self.D+self.normal_w_bins+self.normal_h_bins+self.normal_depth_bins, ...]  # normal_comp_depth_bins预测
        tran_feat = combined_output[:, self.D+self.normal_w_bins+self.normal_h_bins+self.normal_depth_bins:, ...]  # 上下文特征
        
        # 对深度预测进行softmax
        depth = depth_digit.softmax(dim=1)
        
        # 对normal bins预测进行softmax
        normal_w_prob = normal_w_digit.softmax(dim=1)
        normal_h_prob = normal_h_digit.softmax(dim=1)
        normal_depth_prob = normal_depth_digit.softmax(dim=1)
        
        # 构建normal预测结果字典
        normal_preds = {
            'normal_comp_W_bins': normal_w_prob,
            'normal_comp_H_bins': normal_h_prob,
            'normal_comp_depth_bins': normal_depth_prob
        }
        
        # 将normal概率从相机坐标系变换到ego坐标系
        # normal_preds.update(self.transform_normal_to_ego(normal_preds, input))
        # normal_preds.update(self.transform_normal_to_ego_fix_remove_intrins_K(normal_preds, input))
        if self.use_ego_2D_3D_trans_fix:
            normal_preds.update(self.transform_normal_to_ego_fix_remove_intrins_K(normal_preds, input))
        else:
        # 将normal概率从相机坐标系变换到ego坐标系
            normal_preds.update(self.transform_normal_to_ego(normal_preds, input))
        
        # 打印张量shape以进行调试
        # print(f"tran_feat shape: {tran_feat.shape}")
        # print(f"ego_dir_vector shape: {normal_preds['ego_dir_vector'].shape}")
        
        # 将ego_dir_vector与tran_feat进行拼接
        # 确保两者的空间维度匹配
        assert tran_feat.shape[2:] == normal_preds['ego_dir_vector'].shape[2:], "空间维度不匹配"
        
        # 在通道维度（dim=1）上拼接
        tran_feat_with_normal = torch.cat([tran_feat, normal_preds['ego_dir_vector'],normal_preds['normal_comp_W_bins'],normal_preds['normal_comp_H_bins'],normal_preds['normal_comp_depth_bins']], dim=1)
        # print(f"拼接后的tran_feat_with_normal shape: {tran_feat_with_normal.shape}")
        
        # 进行视图变换
        # 注意：这里使用拼接后的特征
        self.out_channels=83+20
        bev_feat, depth = self.view_transform(input, depth, tran_feat_with_normal)
        # print(f"view_transform time: {(end_time - start_time):.4f} s")
        bev_feat = bev_feat[:, :self.out_channels, ...]
        normal_preds['ego_dir_vector_3d'] = bev_feat[:, -3:, ...]
        normal_preds['tran_feat'] = tran_feat
        if self.return_context:
            return bev_feat, depth, tran_feat_with_normal, normal_preds
        else:
            return bev_feat, depth, normal_preds
    
    @torch.no_grad()
    def get_downsampled_gt_normal(self, gt_normals, bins):
        """
        下采样真实的normal bins标签
        
        Args:
            gt_normals: 真实的normal bins标签 [B, N, H, W]
            bins: bins的数量
            
        Returns:
            torch.Tensor: 下采样后的one-hot编码normal标签 [B*N, H//downsample, W//downsample, bins]
        """
        B, N, H, W = gt_normals.shape
        
        # 下采样（使用最大值池化来保持bin索引）
        gt_normals = gt_normals.view(B * N, 1, H, W)
        gt_normals = F.max_pool2d(gt_normals, kernel_size=self.downsample, stride=self.downsample)
        gt_normals = gt_normals.squeeze(1).long()
        
        # 转换为one-hot编码
        gt_normals = F.one_hot(gt_normals, num_classes=bins)
        
        return gt_normals
    
    def get_normal_loss(self, normal_labels, normal_preds):
        """
        计算normal预测的损失
        
        Args:
            normal_labels: 真实的normal标签字典
            normal_preds: 预测的normal概率字典
            
        Returns:
            torch.Tensor: normal预测的总损失
        """
        total_loss = 0.0
        
        # 计算normal_comp_W_bins的损失
        if 'normal_comp_W_bins' in normal_labels and 'normal_comp_W_bins' in normal_preds:
            gt_w = self.get_downsampled_gt_normal(normal_labels['normal_comp_W_bins'], self.normal_w_bins)
            pred_w = normal_preds['normal_comp_W_bins'].permute(0, 2, 3, 1).contiguous()
            
            # 计算交叉熵损失
            w_loss = F.cross_entropy(
                pred_w.reshape(-1, self.normal_w_bins),
                gt_w.reshape(-1, self.normal_w_bins)
            )
            total_loss += w_loss
        
        # 计算normal_comp_H_bins的损失
        if 'normal_comp_H_bins' in normal_labels and 'normal_comp_H_bins' in normal_preds:
            gt_h = self.get_downsampled_gt_normal(normal_labels['normal_comp_H_bins'], self.normal_h_bins)
            pred_h = normal_preds['normal_comp_H_bins'].permute(0, 2, 3, 1).contiguous()
            
            # 计算交叉熵损失
            h_loss = F.cross_entropy(
                pred_h.reshape(-1, self.normal_h_bins),
                gt_h.reshape(-1, self.normal_h_bins)
            )
            total_loss += h_loss
        
        # 计算normal_comp_depth_bins的损失
        if 'normal_comp_depth_bins' in normal_labels and 'normal_comp_depth_bins' in normal_preds:
            gt_depth = self.get_downsampled_gt_normal(normal_labels['normal_comp_depth_bins'], self.normal_depth_bins)
            pred_depth = normal_preds['normal_comp_depth_bins'].permute(0, 2, 3, 1).contiguous()
            
            # 计算交叉熵损失
            depth_loss = F.cross_entropy(
                pred_depth.reshape(-1, self.normal_depth_bins),
                gt_depth.reshape(-1, self.normal_depth_bins)
            )
            total_loss += depth_loss
        
        # 应用权重
        return self.loss_normal_weight * total_loss
        
    def get_downsampled_normal_W(self, normal_comp_W_bins, W_bins):
        """
        对W方向的离散化方向向量进行下采样，处理无效值0，输出独热编码（优化版）
        
        Input:
            normal_comp_W_bins: [B, N, H, W] 
                - B: 批量大小, N: 相机数量, H: 高度, W: 宽度
                - 每个元素是离散化的方向向量（整数），0表示无效值
        Output:
            downsampled_W: [B*N*h*w, W_bins]
                - h = H // self.downsample, w = W // self.downsample
                - W_bins: 方向向量的离散化类别数（对应self.W_bins）
        """
        # 1. 解析输入维度
        B, N, H, W = normal_comp_W_bins.shape
        s = self.downsample  # 下采样率（如2、4等）
        h = H // s  # 下采样后高度
        w = W // s  # 下采样后宽度
        num_blocks = B * N * h * w

        # 2. 拆分下采样块并展平 (与原版逻辑一致)
        # 形状变化：[B, N, H, W] -> [B*N, h, s, w, s] -> [B*N, h, w, s*s] -> [num_blocks, s*s]
        normal_bins = normal_comp_W_bins.view(B * N, h, s, w, s)
        normal_bins = normal_bins.permute(0, 1, 3, 2, 4).contiguous()
        normal_bins = normal_bins.view(num_blocks, s * s)
        
        # 3. 向量化计算众数
        # 3.1 将无效值0替换为一个超出类别范围的标记 (例如 -1)
        normal_bins_valid = torch.where(normal_bins != 0, normal_bins, torch.tensor(-1, device=normal_bins.device))
        
        # 3.2 生成所有可能的类别值 [1, 2, ..., W_bins]
        # 形状: [W_bins]
        classes = torch.arange(1, W_bins + 1, device=normal_bins.device)
        
        # 3.3 批量计算每个块内每个类别的出现次数
        # 使用广播机制: [num_blocks, s*s] != [1, W_bins] -> [num_blocks, s*s, W_bins]
        # 然后在 s*s 维度求和，得到每个块中每个类别的计数
        # 形状: [num_blocks, W_bins]
        counts = (normal_bins_valid.unsqueeze(-1) == classes).sum(dim=1)
        
        # 3.4 找到每个块中计数最多的类别索引
        # 形状: [num_blocks]
        mode_indices = counts.argmax(dim=1)
        
        # 3.5 将索引转换回对应的类别值
        # 形状: [num_blocks]
        downsampled = classes[mode_indices]
        
        # 3.6 处理全无效值的块 (所有计数都为0)
        # 计算每个块的有效像素数量
        valid_pixel_count = (normal_bins != 0).sum(dim=1)
        # 对于有效像素为0的块，将其值设为0
        downsampled[valid_pixel_count == 0] = 0

        # 4. 恢复下采样后的空间形状 (与原版逻辑一致)
        # 形状: [num_blocks] -> [B*N, h, w]
        downsampled = downsampled.view(B * N, h, w)
        
        # 5. 过滤异常值 (与原版逻辑一致)
        valid_range_mask = (downsampled >= 1) & (downsampled <= W_bins)
        downsampled = torch.where(valid_range_mask, downsampled, torch.tensor(0, device=downsampled.device))
        
        # 6. 转换为独热编码 (与原版逻辑一致)
        one_hot = F.one_hot(downsampled, num_classes=W_bins + 1)  # [B*N, h, w, W_bins+1]
        one_hot = one_hot.view(-1, W_bins + 1)[:, 1:]  # [B*N*h*w, W_bins]
        
        return one_hot.float()

    def get_normal_W_loss(self, normal_W_labels, normal_W_preds):
        """
        计算W方向离散化方向向量的损失（参考深度损失逻辑）
        
        Input:
            normal_W_labels: [B, N, H, W] 
                - 原始W方向标签（离散化整数，0为无效值）
            normal_W_preds: [B, W_bins, h, w] 或 [B*N, W_bins, h, w]
                - 模型预测的W方向向量概率（经过softmax前或后，根据模型输出定）
        Output:
            normal_W_loss: 标量损失值
        """
        # 1. 对标签进行下采样处理（使用之前实现的下采样函数）
        # 输出形状：[B*N*h*w, W_bins]（独热编码，排除无效值0的类别）
        normal_W_labels = self.get_downsampled_normal_W(normal_W_labels,8)
        
        # 2. 调整预测的形状，与标签匹配
        # 假设输入预测形状为 [B, N, W_bins, h, w] 或 [B*N, W_bins, h, w]
        # 转换为 [B*N*h*w, W_bins]（与标签维度一致）
        normal_W_preds = normal_W_preds.permute(0, 2, 3, 1).contiguous()  # [B*N, h, w, W_bins]
        normal_W_preds = normal_W_preds.view(-1, self.W_bins)  # [B*N*h*w, W_bins]
        
        # 3. 生成前景掩码（过滤无效值0对应的标签）
        # 标签是独热编码，最大值>0表示有效像素（非0）
        fg_mask = torch.max(normal_W_labels, dim=1).values > 0.0  # [B*N*h*w]
        
        # 4. 过滤无效的标签和预测
        normal_W_labels = normal_W_labels[fg_mask]  # [valid_num, W_bins]
        normal_W_preds = normal_W_preds[fg_mask]    # [valid_num, W_bins]
        
        # 5. 计算损失（禁用自动混合精度，确保计算精度）
        with autocast(enabled=False):
            # 方向向量是离散类别，用二元交叉熵（与深度损失一致，适配独热编码）
            normal_W_loss = F.binary_cross_entropy(
                normal_W_preds,
                normal_W_labels,
                reduction='none'  # 先不求和，后续按有效像素数平均
            ).sum() / max(1.0, fg_mask.sum())  # 除以有效像素数，避免样本不平衡
        
        # 6. 乘以损失权重并返回
        return self.loss_normal_W_weight * normal_W_loss
    
    def get_normal_H_loss(self, normal_W_labels, normal_W_preds):
        """
        计算W方向离散化方向向量的损失（参考深度损失逻辑）
        
        Input:
            normal_W_labels: [B, N, H, W] 
                - 原始W方向标签（离散化整数，0为无效值）
            normal_W_preds: [B, W_bins, h, w] 或 [B*N, W_bins, h, w]
                - 模型预测的W方向向量概率（经过softmax前或后，根据模型输出定）
        Output:
            normal_W_loss: 标量损失值
        """
        # 1. 对标签进行下采样处理（使用之前实现的下采样函数）
        # 输出形状：[B*N*h*w, W_bins]（独热编码，排除无效值0的类别）
        normal_W_labels = self.get_downsampled_normal_W(normal_W_labels,8)
        
        # 2. 调整预测的形状，与标签匹配
        # 假设输入预测形状为 [B, N, W_bins, h, w] 或 [B*N, W_bins, h, w]
        # 转换为 [B*N*h*w, W_bins]（与标签维度一致）
        normal_W_preds = normal_W_preds.permute(0, 2, 3, 1).contiguous()  # [B*N, h, w, W_bins]
        normal_W_preds = normal_W_preds.view(-1, self.H_bins)  # [B*N*h*w, W_bins]
        
        # 3. 生成前景掩码（过滤无效值0对应的标签）
        # 标签是独热编码，最大值>0表示有效像素（非0）
        fg_mask = torch.max(normal_W_labels, dim=1).values > 0.0  # [B*N*h*w]
        
        # 4. 过滤无效的标签和预测
        normal_W_labels = normal_W_labels[fg_mask]  # [valid_num, W_bins]
        normal_W_preds = normal_W_preds[fg_mask]    # [valid_num, W_bins]
        
        # 5. 计算损失（禁用自动混合精度，确保计算精度）
        # with autocast(enabled=False):
            # 方向向量是离散类别，用二元交叉熵（与深度损失一致，适配独热编码）
        normal_W_loss = F.binary_cross_entropy(
            normal_W_preds,
            normal_W_labels,
            reduction='none'  # 先不求和，后续按有效像素数平均
        ).sum() / max(1.0, fg_mask.sum())  # 除以有效像素数，避免样本不平衡
        
        # 6. 乘以损失权重并返回
        return self.loss_normal_H_weight * normal_W_loss

    def get_normal_D_loss(self, normal_W_labels, normal_W_preds):
        """
        计算W方向离散化方向向量的损失（参考深度损失逻辑）
        
        Input:
            normal_W_labels: [B, N, H, W] 
                - 原始W方向标签（离散化整数，0为无效值）
            normal_W_preds: [B, W_bins, h, w] 或 [B*N, W_bins, h, w]
                - 模型预测的W方向向量概率（经过softmax前或后，根据模型输出定）
        Output:
            normal_W_loss: 标量损失值
        """
        # 1. 对标签进行下采样处理（使用之前实现的下采样函数）
        # 输出形状：[B*N*h*w, W_bins]（独热编码，排除无效值0的类别）
        normal_W_labels = self.get_downsampled_normal_W(normal_W_labels,4)
        
        # 2. 调整预测的形状，与标签匹配
        # 假设输入预测形状为 [B, N, W_bins, h, w] 或 [B*N, W_bins, h, w]
        # 转换为 [B*N*h*w, W_bins]（与标签维度一致）
        normal_W_preds = normal_W_preds.permute(0, 2, 3, 1).contiguous()  # [B*N, h, w, W_bins]
        normal_W_preds = normal_W_preds.view(-1, self.D_bins)  # [B*N*h*w, W_bins]
        
        # 3. 生成前景掩码（过滤无效值0对应的标签）
        # 标签是独热编码，最大值>0表示有效像素（非0）
        fg_mask = torch.max(normal_W_labels, dim=1).values > 0.0  # [B*N*h*w]
        
        # 4. 过滤无效的标签和预测
        normal_W_labels = normal_W_labels[fg_mask]  # [valid_num, W_bins]
        normal_W_preds = normal_W_preds[fg_mask]    # [valid_num, W_bins]
        
        # 5. 计算损失（禁用自动混合精度，确保计算精度）
        # with autocast(enabled=False):
            # 方向向量是离散类别，用二元交叉熵（与深度损失一致，适配独热编码）
        normal_W_loss = F.binary_cross_entropy(
            normal_W_preds,
            normal_W_labels,
            reduction='none'  # 先不求和，后续按有效像素数平均
        ).sum() / max(1.0, fg_mask.sum())  # 除以有效像素数，避免样本不平衡
        
        # 6. 乘以损失权重并返回
        return self.loss_normal_D_weight * normal_W_loss





@NECKS.register_module()
class LSSVStereoForwardProjectionWithNormal_normalNet1_NormFeature_A(LSSVStereoForwardPorjection):
    """
    立体视觉版本的LSSForwardProjectionWithNormal
    继承了LSSVStereoForwardPorjection的所有功能，并添加了normal预测支持
    """
    
    def __init__(self, cv_downsample=4, loss_depth_weight=3.0, loss_normal_weight=1.0,
                 depthnet_cfg=dict(), return_context=False,use_ego_2D_3D_trans_fix=False,
                 normal_w_bins=8, normal_h_bins=8, normal_depth_bins=4, **kwargs):
        """
        初始化立体视觉版本的扩展前向投影模块
        
        Args:
            cv_downsample: CV下采样因子
            loss_depth_weight: 深度损失的权重
            loss_normal_weight: normal损失的权重
            depthnet_cfg: 深度网络的配置
            return_context: 是否返回上下文特征
            normal_w_bins: normal_comp_W_bins的类别数
            normal_h_bins: normal_comp_H_bins的类别数
            normal_depth_bins: normal_comp_depth_bins的类别数
            **kwargs: 其他传递给LSSVStereoForwardPorjection的参数
        """
        # 保存normal bins的配置信息
        self.normal_w_bins = normal_w_bins
        self.normal_h_bins = normal_h_bins
        self.normal_depth_bins = normal_depth_bins
        self.loss_normal_weight = loss_normal_weight
        self.return_context = return_context
        self.W_bins = 8
        self.H_bins = 8
        self.D_bins = 4
        self.loss_normal_W_weight = 1.0
        self.loss_normal_H_weight = 1.0
        self.loss_normal_D_weight = 1.0
        # 初始化父类
        self.use_ego_2D_3D_trans_fix = use_ego_2D_3D_trans_fix

        super().__init__(cv_downsample=cv_downsample, **kwargs)
        
        # 使用扩展的DepthNetWithNormal替代原始的depth_net
        self.loss_depth_weight = loss_depth_weight
        self.depth_net = DepthNetWithNormalImproved(
            self.in_channels,
            self.in_channels,
            context_channels=self.out_channels,
            depth_channels=self.D,
            normal_w_bins=normal_w_bins,
            normal_h_bins=normal_h_bins,
            normal_depth_bins=normal_depth_bins,
            **depthnet_cfg
        )

    def transform_normal_to_ego(self, normal_preds, input):
        """
        将normal概率从相机坐标系变换到ego坐标系
        
        Args:
            normal_preds: 包含normal_comp_W_bins, normal_comp_H_bins, normal_comp_depth_bins预测的字典
            input: 输入数据，包含x, sensor2keyego, ego2global, intrins, post_rot, post_aug, bda, mlp_input
            
        Returns:
            dict: ego坐标系下的normal角度数值，包含ego_w_angle、ego_h_angle、ego_d_angle
        """
        # import torch
        # import torch.nn.functional as F
        
        # 从input中解包参数
        x, sensor2keyego, ego2global, intrins, post_rot, post_aug, bda, mlp_input = input[:8]
        
        # 获取概率分布
        normal_w_prob = normal_preds['normal_comp_W_bins']
        normal_h_prob = normal_preds['normal_comp_H_bins']
        normal_depth_prob = normal_preds['normal_comp_depth_bins']
        
        # 获取形状信息
        B, N, _, _ = sensor2keyego.shape
        H, W = normal_w_prob.shape[-2:]  # [B*N, num_bins, H, W]
        
        # 创建bin中心角度（弧度制）
        # W方向和H方向都是0-180度，D方向是0-90度
        num_w_bins = normal_w_prob.shape[1]
        num_h_bins = normal_h_prob.shape[1]
        num_d_bins = normal_depth_prob.shape[1]
        
        # 计算bin中心角度（弧度制）
        w_angles_rad = torch.linspace(0, torch.pi-torch.pi/num_w_bins, num_w_bins, device=normal_w_prob.device)
        h_angles_rad = torch.linspace(0, torch.pi-torch.pi/num_h_bins, num_h_bins, device=normal_w_prob.device)
        d_angles_rad = torch.linspace(0, torch.pi/2-torch.pi/2/num_d_bins, num_d_bins, device=normal_w_prob.device)
        
        # 计算期望角度
        w_expected_angle = torch.sum(normal_w_prob * w_angles_rad.view(1, -1, 1, 1), dim=1)
        h_expected_angle = torch.sum(normal_h_prob * h_angles_rad.view(1, -1, 1, 1), dim=1)
        d_expected_angle = torch.sum(normal_depth_prob * d_angles_rad.view(1, -1, 1, 1), dim=1)
        
        # 从球面坐标转换为笛卡尔坐标（方向向量）
        # theta = h_expected_angle (极角，与z轴夹角)
        # phi = w_expected_angle (方位角，在xy平面上的投影与x轴夹角)
        # r = 1 (单位向量)
        sin_theta = torch.sin(h_expected_angle)
        cos_theta = torch.cos(h_expected_angle)
        sin_phi = torch.sin(w_expected_angle)
        cos_phi = torch.cos(w_expected_angle)
        
        # 计算方向向量的x, y, z分量
        # 注意：这里使用的是相机坐标系的约定
        x_dir = sin_theta * cos_phi
        y_dir = sin_theta * sin_phi
        z_dir = cos_theta
        
        # 归一化方向向量
        dir_vector = torch.stack([x_dir, y_dir, z_dir], dim=1)  # [B*N, 3, H, W]
        dir_vector = F.normalize(dir_vector, dim=1)
        cam_dir_vector = dir_vector
        # 重塑方向向量以适应后续变换
        # 重塑为 [B*N, 3, H*W] 以进行批量处理
        dir_vector_reshaped = dir_vector.view(B*N, 3, -1)  # [B*N, 3, H*W]
        
        # 处理后旋转（逆变换）
        # 重塑post_rot以匹配dir_vector_reshaped的形状
        post_rot_inv = torch.inverse(post_rot).view(B*N, 3, 3)  # [B*N, 3, 3]
        dir_vector = torch.bmm(post_rot_inv, dir_vector_reshaped)  # [B*N, 3, H*W]
        
        # 处理相机内参（逆变换）
        intrins_inv = torch.inverse(intrins)[:, :, :3, :3].view(B*N, 3, 3)  # [B*N, 3, 3]
        dir_vector = torch.bmm(intrins_inv, dir_vector)  # [B*N, 3, H*W]
        
        # 应用sensor2keyego变换（相机到ego坐标系）
        sensor2keyego_rot = sensor2keyego[:, :, :3, :3].reshape(B*N, 3, 3)  # [B*N, 3, 3]
        dir_vector = torch.bmm(sensor2keyego_rot, dir_vector)  # [B*N, 3, H*W]
        
        # 应用BDD变换
        if bda is not None:
            bda_rot = bda[:, :3, :3].view(B, 1, 3, 3).expand(B, N, 3, 3).contiguous().view(B*N, 3, 3)
            dir_vector = torch.bmm(bda_rot, dir_vector)  # [B*N, 3, H*W]
        
        # 重塑回原始形状
        ego_dir_vector = dir_vector.view(B*N, 3, H, W)
        
        # 归一化旋转后的方向向量
        ego_dir_vector = F.normalize(ego_dir_vector, dim=1)
        
        # 将旋转后的方向向量转换回角度表示
        # 计算极角theta (与z轴夹角)
        ego_z = ego_dir_vector[:, 2]
        ego_h_angle = torch.acos(torch.clamp(ego_z, -1.0, 1.0))  # [B*N, H, W]
        
        # 计算方位角phi (在xy平面上的投影与x轴夹角)
        ego_x = ego_dir_vector[:, 0]
        ego_y = ego_dir_vector[:, 1]
        ego_w_angle = torch.atan2(ego_y, ego_x)  # [B*N, H, W]
        
        # 归一化phi到0-2pi范围
        ego_w_angle = ego_w_angle % (2 * torch.pi)
        
        # 计算与y轴的夹角（D方向）
        ego_d_angle = torch.acos(torch.clamp(ego_y, -1.0, 1.0))
        
        # 构建ego坐标系下的normal角度结果
        ego_normal_results = {
            'ego_w_angle': ego_w_angle,  # W方向角度（弧度制）
            'ego_h_angle': ego_h_angle,  # H方向角度（弧度制）
            'ego_d_angle': ego_d_angle,  # D方向角度（弧度制）
            'ego_dir_vector': ego_dir_vector,  # 归一化后的方向向量
            'gt_depth': normal_preds.get('gt_depth', None),
            'cam_dir_vector': cam_dir_vector
        }
        
        return ego_normal_results
    
    def transform_normal_to_ego_fix_remove_intrins_K(self, normal_preds, input):
        """
        将normal概率从相机坐标系变换到ego坐标系
        【核心修改】移除了多余的相机内参逆变换（因法向量来自LiDAR物理3D推导，无内参畸变）
        
        Args:
            normal_preds: 包含normal_comp_W_bins, normal_comp_H_bins, normal_comp_depth_bins预测的字典
            input: 输入数据，包含x, sensor2keyego, ego2global, intrins, post_rot, post_aug, bda, mlp_input
            
        Returns:
            dict: ego坐标系下的normal角度数值，包含ego_w_angle、ego_h_angle、ego_d_angle
        """
        # 从input中解包参数（intrins保留但不再使用，避免修改输入解包逻辑）
        x, sensor2keyego, ego2global, intrins, post_rot, post_aug, bda, mlp_input = input[:8]
        
        # 获取概率分布
        normal_w_prob = normal_preds['normal_comp_W_bins']
        normal_h_prob = normal_preds['normal_comp_H_bins']
        normal_depth_prob = normal_preds['normal_comp_depth_bins']
        
        # 获取形状信息
        B, N, _, _ = sensor2keyego.shape
        H, W = normal_w_prob.shape[-2:]  # [B*N, num_bins, H, W]
        
        # 创建bin中心角度（弧度制）
        # W方向和H方向都是0-180度，D方向是0-90度
        num_w_bins = normal_w_prob.shape[1]
        num_h_bins = normal_h_prob.shape[1]
        num_d_bins = normal_depth_prob.shape[1]
        
        # 计算bin中心角度（弧度制）
        w_angles_rad = torch.linspace(0, torch.pi-torch.pi/num_w_bins, num_w_bins, device=normal_w_prob.device)
        h_angles_rad = torch.linspace(0, torch.pi-torch.pi/num_h_bins, num_h_bins, device=normal_w_prob.device)
        d_angles_rad = torch.linspace(0, torch.pi/2-torch.pi/2/num_d_bins, num_d_bins, device=normal_w_prob.device)
        
        # 计算期望角度
        w_expected_angle = torch.sum(normal_w_prob * w_angles_rad.view(1, -1, 1, 1), dim=1)
        h_expected_angle = torch.sum(normal_h_prob * h_angles_rad.view(1, -1, 1, 1), dim=1)
        d_expected_angle = torch.sum(normal_depth_prob * d_angles_rad.view(1, -1, 1, 1), dim=1)
        
        # 从球面坐标转换为笛卡尔坐标（方向向量）
        # theta = h_expected_angle (极角，与z轴夹角)
        # phi = w_expected_angle (方位角，在xy平面上的投影与x轴夹角)
        # r = 1 (单位向量)
        sin_theta = torch.sin(h_expected_angle)
        cos_theta = torch.cos(h_expected_angle)
        sin_phi = torch.sin(w_expected_angle)
        cos_phi = torch.cos(w_expected_angle)
        
        # 计算方向向量的x, y, z分量（相机物理坐标系，无内参畸变）
        x_dir = sin_theta * cos_phi
        y_dir = sin_theta * sin_phi
        z_dir = cos_theta
        
        # 归一化方向向量
        dir_vector = torch.stack([x_dir, y_dir, z_dir], dim=1)  # [B*N, 3, H, W]
        dir_vector = F.normalize(dir_vector, dim=1)
        cam_dir_vector = dir_vector  # 保存相机坐标系原始方向向量
        
        # 重塑方向向量以适应后续变换：[B*N, 3, H*W]
        dir_vector_reshaped = dir_vector.view(B*N, 3, -1)
        
        # 1. 处理后旋转逆变换（必须保留：对齐图像增强前的相机轴）
        post_rot_inv = torch.inverse(post_rot).view(B*N, 3, 3)  # [B*N, 3, 3]
        dir_vector = torch.bmm(post_rot_inv, dir_vector_reshaped)  # [B*N, 3, H*W]
        
        # 【关键删除】移除相机内参逆变换（法向量来自LiDAR物理3D，无内参畸变，无需校正）
        # --- 原错误代码已删除 ---
        # intrins_inv = torch.inverse(intrins)[:, :, :3, :3].view(B*N, 3, 3)
        # dir_vector = torch.bmm(intrins_inv, dir_vector)
        
        # 2. 应用sensor2keyego变换（相机坐标系 → ego坐标系，必须保留）
        sensor2keyego_rot = sensor2keyego[:, :, :3, :3].reshape(B*N, 3, 3)  # [B*N, 3, 3]
        dir_vector = torch.bmm(sensor2keyego_rot, dir_vector)  # [B*N, 3, H*W]
        
        # 3. 应用BDD变换（可选，根据数据集需求保留）
        if bda is not None:
            bda_rot = bda[:, :3, :3].view(B, 1, 3, 3).expand(B, N, 3, 3).contiguous().view(B*N, 3, 3)
            dir_vector = torch.bmm(bda_rot, dir_vector)  # [B*N, 3, H*W]
        
        # 重塑回原始形状：[B*N, 3, H, W]
        ego_dir_vector = dir_vector.view(B*N, 3, H, W)
        
        # 归一化旋转后的方向向量（确保单位向量）
        ego_dir_vector = F.normalize(ego_dir_vector, dim=1)
        
        # 将ego坐标系方向向量转换回角度表示
        # 极角theta (与z轴夹角) → ego_h_angle
        ego_z = ego_dir_vector[:, 2]
        ego_h_angle = torch.acos(torch.clamp(ego_z, -1.0, 1.0))  # [B*N, H, W]
        
        # 方位角phi (xy平面投影与x轴夹角) → ego_w_angle
        ego_x = ego_dir_vector[:, 0]
        ego_y = ego_dir_vector[:, 1]
        ego_w_angle = torch.atan2(ego_y, ego_x)  # [B*N, H, W]
        ego_w_angle = ego_w_angle % (2 * torch.pi)  # 归一化到0-2pi
        
        # 与y轴的夹角 → ego_d_angle
        ego_d_angle = torch.acos(torch.clamp(ego_y, -1.0, 1.0))
        
        # 构建返回结果
        ego_normal_results = {
            'ego_w_angle': ego_w_angle,  # W方向角度（弧度制）
            'ego_h_angle': ego_h_angle,  # H方向角度（弧度制）
            'ego_d_angle': ego_d_angle,  # D方向角度（弧度制）
            'ego_dir_vector': ego_dir_vector,  # ego坐标系归一化方向向量
            'gt_depth': normal_preds.get('gt_depth', None),
            'cam_dir_vector': cam_dir_vector  # 相机坐标系原始方向向量
        }
        
        return ego_normal_results

    def forward(self, input, stereo_metas=None):
        """
        前向传播函数，扩展了原始LSSVStereoForwardPorjection的功能
        
        Args:
            input: 输入数据，包含x, rots, trans, intrins, post_rot, post_aug, bda, mlp_input
            stereo_metas: 立体视觉相关的元数据
            
        Returns:
            tuple: (bev_feat, depth, tran_feat, normal_preds)
                bev_feat: BEV特征
                depth: 深度预测
                tran_feat: 转换特征
                normal_preds: 包含normal_comp_W_bins, normal_comp_H_bins, normal_comp_depth_bins预测的字典
        """
        x, sensor2keyego, ego2global, intrins, post_rot, post_aug, bda, mlp_input = input[:8]

        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        start_time = time.time()
        
        # 使用扩展的深度网络获取预测结果
        combined_output = self.depth_net(x, mlp_input, stereo_metas)
        end_time = time.time()
        
        # 分割不同的预测结果
        depth_digit = combined_output[:, :self.D, ...]  # 原始深度预测
        normal_w_digit = combined_output[:, self.D:self.D+self.normal_w_bins, ...]  # normal_comp_W_bins预测
        normal_h_digit = combined_output[:, self.D+self.normal_w_bins:self.D+self.normal_w_bins+self.normal_h_bins, ...]  # normal_comp_H_bins预测
        normal_depth_digit = combined_output[:, self.D+self.normal_w_bins+self.normal_h_bins:self.D+self.normal_w_bins+self.normal_h_bins+self.normal_depth_bins, ...]  # normal_comp_depth_bins预测
        tran_feat = combined_output[:, self.D+self.normal_w_bins+self.normal_h_bins+self.normal_depth_bins:, ...]  # 上下文特征
        
        # 对深度预测进行softmax
        depth = depth_digit.softmax(dim=1)
        
        # 对normal bins预测进行softmax
        normal_w_prob = normal_w_digit.softmax(dim=1)
        normal_h_prob = normal_h_digit.softmax(dim=1)
        normal_depth_prob = normal_depth_digit.softmax(dim=1)
        
        # 构建normal预测结果字典
        normal_preds = {
            'normal_comp_W_bins': normal_w_prob,
            'normal_comp_H_bins': normal_h_prob,
            'normal_comp_depth_bins': normal_depth_prob
        }
        
        # 将normal概率从相机坐标系变换到ego坐标系
        # normal_preds.update(self.transform_normal_to_ego(normal_preds, input))
        if self.use_ego_2D_3D_trans_fix:
            normal_preds.update(self.transform_normal_to_ego_fix_remove_intrins_K(normal_preds, input))
        else:
        # 将normal概率从相机坐标系变换到ego坐标系
            normal_preds.update(self.transform_normal_to_ego(normal_preds, input))

        # 打印张量shape以进行调试
        # print(f"tran_feat shape: {tran_feat.shape}")
        # print(f"ego_dir_vector shape: {normal_preds['ego_dir_vector'].shape}")
        
        # 将ego_dir_vector与tran_feat进行拼接
        # 确保两者的空间维度匹配
        assert tran_feat.shape[2:] == normal_preds['ego_dir_vector'].shape[2:], "空间维度不匹配"
        
        # 在通道维度（dim=1）上拼接
        # tran_feat_with_normal = torch.cat([tran_feat, normal_preds['ego_dir_vector']], dim=1)
        tran_feat_with_normal = torch.cat([tran_feat, normal_preds['ego_dir_vector'],normal_preds['normal_comp_W_bins'],normal_preds['normal_comp_H_bins'],normal_preds['normal_comp_depth_bins']], dim=1)

        # print(f"拼接后的tran_feat_with_normal shape: {tran_feat_with_normal.shape}")
        
        # 进行视图变换
        # 注意：这里使用拼接后的特征
        self.out_channels=83+20
        bev_feat, depth = self.view_transform(input, depth, tran_feat_with_normal)
        # print(f"view_transform time: {(end_time - start_time):.4f} s")
        bev_feat = bev_feat[:, :self.out_channels, ...]
        normal_preds['ego_dir_vector_3d'] = bev_feat[:, -3:, ...]
        normal_preds['tran_feat'] = tran_feat
        if self.return_context:
            return bev_feat, depth, tran_feat_with_normal, normal_preds
        else:
            return bev_feat, depth, normal_preds
    
    @torch.no_grad()
    def get_downsampled_gt_normal(self, gt_normals, bins):
        """
        下采样真实的normal bins标签
        
        Args:
            gt_normals: 真实的normal bins标签 [B, N, H, W]
            bins: bins的数量
            
        Returns:
            torch.Tensor: 下采样后的one-hot编码normal标签 [B*N, H//downsample, W//downsample, bins]
        """
        B, N, H, W = gt_normals.shape
        
        # 下采样（使用最大值池化来保持bin索引）
        gt_normals = gt_normals.view(B * N, 1, H, W)
        gt_normals = F.max_pool2d(gt_normals, kernel_size=self.downsample, stride=self.downsample)
        gt_normals = gt_normals.squeeze(1).long()
        
        # 转换为one-hot编码
        gt_normals = F.one_hot(gt_normals, num_classes=bins)
        
        return gt_normals
    
    def get_normal_loss(self, normal_labels, normal_preds):
        """
        计算normal预测的损失
        
        Args:
            normal_labels: 真实的normal标签字典
            normal_preds: 预测的normal概率字典
            
        Returns:
            torch.Tensor: normal预测的总损失
        """
        total_loss = 0.0
        
        # 计算normal_comp_W_bins的损失
        if 'normal_comp_W_bins' in normal_labels and 'normal_comp_W_bins' in normal_preds:
            gt_w = self.get_downsampled_gt_normal(normal_labels['normal_comp_W_bins'], self.normal_w_bins)
            pred_w = normal_preds['normal_comp_W_bins'].permute(0, 2, 3, 1).contiguous()
            
            # 计算交叉熵损失
            w_loss = F.cross_entropy(
                pred_w.reshape(-1, self.normal_w_bins),
                gt_w.reshape(-1, self.normal_w_bins)
            )
            total_loss += w_loss
        
        # 计算normal_comp_H_bins的损失
        if 'normal_comp_H_bins' in normal_labels and 'normal_comp_H_bins' in normal_preds:
            gt_h = self.get_downsampled_gt_normal(normal_labels['normal_comp_H_bins'], self.normal_h_bins)
            pred_h = normal_preds['normal_comp_H_bins'].permute(0, 2, 3, 1).contiguous()
            
            # 计算交叉熵损失
            h_loss = F.cross_entropy(
                pred_h.reshape(-1, self.normal_h_bins),
                gt_h.reshape(-1, self.normal_h_bins)
            )
            total_loss += h_loss
        
        # 计算normal_comp_depth_bins的损失
        if 'normal_comp_depth_bins' in normal_labels and 'normal_comp_depth_bins' in normal_preds:
            gt_depth = self.get_downsampled_gt_normal(normal_labels['normal_comp_depth_bins'], self.normal_depth_bins)
            pred_depth = normal_preds['normal_comp_depth_bins'].permute(0, 2, 3, 1).contiguous()
            
            # 计算交叉熵损失
            depth_loss = F.cross_entropy(
                pred_depth.reshape(-1, self.normal_depth_bins),
                gt_depth.reshape(-1, self.normal_depth_bins)
            )
            total_loss += depth_loss
        
        # 应用权重
        return self.loss_normal_weight * total_loss
        
    def get_downsampled_normal_W(self, normal_comp_W_bins, W_bins):
        """
        对W方向的离散化方向向量进行下采样，处理无效值0，输出独热编码（优化版）
        
        Input:
            normal_comp_W_bins: [B, N, H, W] 
                - B: 批量大小, N: 相机数量, H: 高度, W: 宽度
                - 每个元素是离散化的方向向量（整数），0表示无效值
        Output:
            downsampled_W: [B*N*h*w, W_bins]
                - h = H // self.downsample, w = W // self.downsample
                - W_bins: 方向向量的离散化类别数（对应self.W_bins）
        """
        # 1. 解析输入维度
        B, N, H, W = normal_comp_W_bins.shape
        s = self.downsample  # 下采样率（如2、4等）
        h = H // s  # 下采样后高度
        w = W // s  # 下采样后宽度
        num_blocks = B * N * h * w

        # 2. 拆分下采样块并展平 (与原版逻辑一致)
        # 形状变化：[B, N, H, W] -> [B*N, h, s, w, s] -> [B*N, h, w, s*s] -> [num_blocks, s*s]
        normal_bins = normal_comp_W_bins.view(B * N, h, s, w, s)
        normal_bins = normal_bins.permute(0, 1, 3, 2, 4).contiguous()
        normal_bins = normal_bins.view(num_blocks, s * s)
        
        # 3. 向量化计算众数
        # 3.1 将无效值0替换为一个超出类别范围的标记 (例如 -1)
        normal_bins_valid = torch.where(normal_bins != 0, normal_bins, torch.tensor(-1, device=normal_bins.device))
        
        # 3.2 生成所有可能的类别值 [1, 2, ..., W_bins]
        # 形状: [W_bins]
        classes = torch.arange(1, W_bins + 1, device=normal_bins.device)
        
        # 3.3 批量计算每个块内每个类别的出现次数
        # 使用广播机制: [num_blocks, s*s] != [1, W_bins] -> [num_blocks, s*s, W_bins]
        # 然后在 s*s 维度求和，得到每个块中每个类别的计数
        # 形状: [num_blocks, W_bins]
        counts = (normal_bins_valid.unsqueeze(-1) == classes).sum(dim=1)
        
        # 3.4 找到每个块中计数最多的类别索引
        # 形状: [num_blocks]
        mode_indices = counts.argmax(dim=1)
        
        # 3.5 将索引转换回对应的类别值
        # 形状: [num_blocks]
        downsampled = classes[mode_indices]
        
        # 3.6 处理全无效值的块 (所有计数都为0)
        # 计算每个块的有效像素数量
        valid_pixel_count = (normal_bins != 0).sum(dim=1)
        # 对于有效像素为0的块，将其值设为0
        downsampled[valid_pixel_count == 0] = 0

        # 4. 恢复下采样后的空间形状 (与原版逻辑一致)
        # 形状: [num_blocks] -> [B*N, h, w]
        downsampled = downsampled.view(B * N, h, w)
        
        # 5. 过滤异常值 (与原版逻辑一致)
        valid_range_mask = (downsampled >= 1) & (downsampled <= W_bins)
        downsampled = torch.where(valid_range_mask, downsampled, torch.tensor(0, device=downsampled.device))
        
        # 6. 转换为独热编码 (与原版逻辑一致)
        one_hot = F.one_hot(downsampled, num_classes=W_bins + 1)  # [B*N, h, w, W_bins+1]
        one_hot = one_hot.view(-1, W_bins + 1)[:, 1:]  # [B*N*h*w, W_bins]
        
        return one_hot.float()

    def get_normal_W_loss(self, normal_W_labels, normal_W_preds):
        """
        计算W方向离散化方向向量的损失（参考深度损失逻辑）
        
        Input:
            normal_W_labels: [B, N, H, W] 
                - 原始W方向标签（离散化整数，0为无效值）
            normal_W_preds: [B, W_bins, h, w] 或 [B*N, W_bins, h, w]
                - 模型预测的W方向向量概率（经过softmax前或后，根据模型输出定）
        Output:
            normal_W_loss: 标量损失值
        """
        # 1. 对标签进行下采样处理（使用之前实现的下采样函数）
        # 输出形状：[B*N*h*w, W_bins]（独热编码，排除无效值0的类别）
        normal_W_labels = self.get_downsampled_normal_W(normal_W_labels,8)
        
        # 2. 调整预测的形状，与标签匹配
        # 假设输入预测形状为 [B, N, W_bins, h, w] 或 [B*N, W_bins, h, w]
        # 转换为 [B*N*h*w, W_bins]（与标签维度一致）
        normal_W_preds = normal_W_preds.permute(0, 2, 3, 1).contiguous()  # [B*N, h, w, W_bins]
        normal_W_preds = normal_W_preds.view(-1, self.W_bins)  # [B*N*h*w, W_bins]
        
        # 3. 生成前景掩码（过滤无效值0对应的标签）
        # 标签是独热编码，最大值>0表示有效像素（非0）
        fg_mask = torch.max(normal_W_labels, dim=1).values > 0.0  # [B*N*h*w]
        
        # 4. 过滤无效的标签和预测
        normal_W_labels = normal_W_labels[fg_mask]  # [valid_num, W_bins]
        normal_W_preds = normal_W_preds[fg_mask]    # [valid_num, W_bins]
        
        # 5. 计算损失（禁用自动混合精度，确保计算精度）
        with autocast(enabled=False):
            # 方向向量是离散类别，用二元交叉熵（与深度损失一致，适配独热编码）
            normal_W_loss = F.binary_cross_entropy(
                normal_W_preds,
                normal_W_labels,
                reduction='none'  # 先不求和，后续按有效像素数平均
            ).sum() / max(1.0, fg_mask.sum())  # 除以有效像素数，避免样本不平衡
        
        # 6. 乘以损失权重并返回
        return self.loss_normal_W_weight * normal_W_loss
    
    def get_normal_H_loss(self, normal_W_labels, normal_W_preds):
        """
        计算W方向离散化方向向量的损失（参考深度损失逻辑）
        
        Input:
            normal_W_labels: [B, N, H, W] 
                - 原始W方向标签（离散化整数，0为无效值）
            normal_W_preds: [B, W_bins, h, w] 或 [B*N, W_bins, h, w]
                - 模型预测的W方向向量概率（经过softmax前或后，根据模型输出定）
        Output:
            normal_W_loss: 标量损失值
        """
        # 1. 对标签进行下采样处理（使用之前实现的下采样函数）
        # 输出形状：[B*N*h*w, W_bins]（独热编码，排除无效值0的类别）
        normal_W_labels = self.get_downsampled_normal_W(normal_W_labels,8)
        
        # 2. 调整预测的形状，与标签匹配
        # 假设输入预测形状为 [B, N, W_bins, h, w] 或 [B*N, W_bins, h, w]
        # 转换为 [B*N*h*w, W_bins]（与标签维度一致）
        normal_W_preds = normal_W_preds.permute(0, 2, 3, 1).contiguous()  # [B*N, h, w, W_bins]
        normal_W_preds = normal_W_preds.view(-1, self.H_bins)  # [B*N*h*w, W_bins]
        
        # 3. 生成前景掩码（过滤无效值0对应的标签）
        # 标签是独热编码，最大值>0表示有效像素（非0）
        fg_mask = torch.max(normal_W_labels, dim=1).values > 0.0  # [B*N*h*w]
        
        # 4. 过滤无效的标签和预测
        normal_W_labels = normal_W_labels[fg_mask]  # [valid_num, W_bins]
        normal_W_preds = normal_W_preds[fg_mask]    # [valid_num, W_bins]
        
        # 5. 计算损失（禁用自动混合精度，确保计算精度）
        # with autocast(enabled=False):
            # 方向向量是离散类别，用二元交叉熵（与深度损失一致，适配独热编码）
        normal_W_loss = F.binary_cross_entropy(
            normal_W_preds,
            normal_W_labels,
            reduction='none'  # 先不求和，后续按有效像素数平均
        ).sum() / max(1.0, fg_mask.sum())  # 除以有效像素数，避免样本不平衡
        
        # 6. 乘以损失权重并返回
        return self.loss_normal_H_weight * normal_W_loss

    def get_normal_D_loss(self, normal_W_labels, normal_W_preds):
        """
        计算W方向离散化方向向量的损失（参考深度损失逻辑）
        
        Input:
            normal_W_labels: [B, N, H, W] 
                - 原始W方向标签（离散化整数，0为无效值）
            normal_W_preds: [B, W_bins, h, w] 或 [B*N, W_bins, h, w]
                - 模型预测的W方向向量概率（经过softmax前或后，根据模型输出定）
        Output:
            normal_W_loss: 标量损失值
        """
        # 1. 对标签进行下采样处理（使用之前实现的下采样函数）
        # 输出形状：[B*N*h*w, W_bins]（独热编码，排除无效值0的类别）
        normal_W_labels = self.get_downsampled_normal_W(normal_W_labels,4)
        
        # 2. 调整预测的形状，与标签匹配
        # 假设输入预测形状为 [B, N, W_bins, h, w] 或 [B*N, W_bins, h, w]
        # 转换为 [B*N*h*w, W_bins]（与标签维度一致）
        normal_W_preds = normal_W_preds.permute(0, 2, 3, 1).contiguous()  # [B*N, h, w, W_bins]
        normal_W_preds = normal_W_preds.view(-1, self.D_bins)  # [B*N*h*w, W_bins]
        
        # 3. 生成前景掩码（过滤无效值0对应的标签）
        # 标签是独热编码，最大值>0表示有效像素（非0）
        fg_mask = torch.max(normal_W_labels, dim=1).values > 0.0  # [B*N*h*w]
        
        # 4. 过滤无效的标签和预测
        normal_W_labels = normal_W_labels[fg_mask]  # [valid_num, W_bins]
        normal_W_preds = normal_W_preds[fg_mask]    # [valid_num, W_bins]
        
        # 5. 计算损失（禁用自动混合精度，确保计算精度）
        # with autocast(enabled=False):
            # 方向向量是离散类别，用二元交叉熵（与深度损失一致，适配独热编码）
        normal_W_loss = F.binary_cross_entropy(
            normal_W_preds,
            normal_W_labels,
            reduction='none'  # 先不求和，后续按有效像素数平均
        ).sum() / max(1.0, fg_mask.sum())  # 除以有效像素数，避免样本不平衡
        
        # 6. 乘以损失权重并返回
        return self.loss_normal_D_weight * normal_W_loss


@NECKS.register_module()
class LSSVStereoForwardProjectionWithNormal_normalNet2_NormFeature_A(LSSVStereoForwardPorjection):
    """
    立体视觉版本的LSSForwardProjectionWithNormal
    继承了LSSVStereoForwardPorjection的所有功能，并添加了normal预测支持
    """
    
    def __init__(self, cv_downsample=4, loss_depth_weight=3.0, loss_normal_weight=1.0,
                 depthnet_cfg=dict(), return_context=False,use_ego_2D_3D_trans_fix=False,
                 normal_w_bins=8, normal_h_bins=8, normal_depth_bins=4, **kwargs):
        """
        初始化立体视觉版本的扩展前向投影模块
        
        Args:
            cv_downsample: CV下采样因子
            loss_depth_weight: 深度损失的权重
            loss_normal_weight: normal损失的权重
            depthnet_cfg: 深度网络的配置
            return_context: 是否返回上下文特征
            normal_w_bins: normal_comp_W_bins的类别数
            normal_h_bins: normal_comp_H_bins的类别数
            normal_depth_bins: normal_comp_depth_bins的类别数
            **kwargs: 其他传递给LSSVStereoForwardPorjection的参数
        """
        # 保存normal bins的配置信息
        self.normal_w_bins = normal_w_bins
        self.normal_h_bins = normal_h_bins
        self.normal_depth_bins = normal_depth_bins
        self.loss_normal_weight = loss_normal_weight
        self.return_context = return_context
        self.W_bins = 8
        self.H_bins = 8
        self.D_bins = 4
        self.loss_normal_W_weight = 1.0
        self.loss_normal_H_weight = 1.0
        self.loss_normal_D_weight = 1.0
        # 初始化父类
        self.use_ego_2D_3D_trans_fix = use_ego_2D_3D_trans_fix

        super().__init__(cv_downsample=cv_downsample, **kwargs)
        
        # 使用扩展的DepthNetWithNormal替代原始的depth_net
        self.loss_depth_weight = loss_depth_weight
        self.depth_net = DepthNetWithNormalIsolated(
            self.in_channels,
            self.in_channels,
            context_channels=self.out_channels,
            depth_channels=self.D,
            normal_w_bins=normal_w_bins,
            normal_h_bins=normal_h_bins,
            normal_depth_bins=normal_depth_bins,
            **depthnet_cfg
        )

    def transform_normal_to_ego(self, normal_preds, input):
        """
        将normal概率从相机坐标系变换到ego坐标系
        
        Args:
            normal_preds: 包含normal_comp_W_bins, normal_comp_H_bins, normal_comp_depth_bins预测的字典
            input: 输入数据，包含x, sensor2keyego, ego2global, intrins, post_rot, post_aug, bda, mlp_input
            
        Returns:
            dict: ego坐标系下的normal角度数值，包含ego_w_angle、ego_h_angle、ego_d_angle
        """
        # import torch
        # import torch.nn.functional as F
        
        # 从input中解包参数
        x, sensor2keyego, ego2global, intrins, post_rot, post_aug, bda, mlp_input = input[:8]
        
        # 获取概率分布
        normal_w_prob = normal_preds['normal_comp_W_bins']
        normal_h_prob = normal_preds['normal_comp_H_bins']
        normal_depth_prob = normal_preds['normal_comp_depth_bins']
        
        # 获取形状信息
        B, N, _, _ = sensor2keyego.shape
        H, W = normal_w_prob.shape[-2:]  # [B*N, num_bins, H, W]
        
        # 创建bin中心角度（弧度制）
        # W方向和H方向都是0-180度，D方向是0-90度
        num_w_bins = normal_w_prob.shape[1]
        num_h_bins = normal_h_prob.shape[1]
        num_d_bins = normal_depth_prob.shape[1]
        
        # 计算bin中心角度（弧度制）
        w_angles_rad = torch.linspace(0, torch.pi-torch.pi/num_w_bins, num_w_bins, device=normal_w_prob.device)
        h_angles_rad = torch.linspace(0, torch.pi-torch.pi/num_h_bins, num_h_bins, device=normal_w_prob.device)
        d_angles_rad = torch.linspace(0, torch.pi/2-torch.pi/2/num_d_bins, num_d_bins, device=normal_w_prob.device)
        
        # 计算期望角度
        w_expected_angle = torch.sum(normal_w_prob * w_angles_rad.view(1, -1, 1, 1), dim=1)
        h_expected_angle = torch.sum(normal_h_prob * h_angles_rad.view(1, -1, 1, 1), dim=1)
        d_expected_angle = torch.sum(normal_depth_prob * d_angles_rad.view(1, -1, 1, 1), dim=1)
        
        # 从球面坐标转换为笛卡尔坐标（方向向量）
        # theta = h_expected_angle (极角，与z轴夹角)
        # phi = w_expected_angle (方位角，在xy平面上的投影与x轴夹角)
        # r = 1 (单位向量)
        sin_theta = torch.sin(h_expected_angle)
        cos_theta = torch.cos(h_expected_angle)
        sin_phi = torch.sin(w_expected_angle)
        cos_phi = torch.cos(w_expected_angle)
        
        # 计算方向向量的x, y, z分量
        # 注意：这里使用的是相机坐标系的约定
        x_dir = sin_theta * cos_phi
        y_dir = sin_theta * sin_phi
        z_dir = cos_theta
        
        # 归一化方向向量
        dir_vector = torch.stack([x_dir, y_dir, z_dir], dim=1)  # [B*N, 3, H, W]
        dir_vector = F.normalize(dir_vector, dim=1)
        cam_dir_vector = dir_vector
        # 重塑方向向量以适应后续变换
        # 重塑为 [B*N, 3, H*W] 以进行批量处理
        dir_vector_reshaped = dir_vector.view(B*N, 3, -1)  # [B*N, 3, H*W]
        
        # 处理后旋转（逆变换）
        # 重塑post_rot以匹配dir_vector_reshaped的形状
        post_rot_inv = torch.inverse(post_rot).view(B*N, 3, 3)  # [B*N, 3, 3]
        dir_vector = torch.bmm(post_rot_inv, dir_vector_reshaped)  # [B*N, 3, H*W]
        
        # 处理相机内参（逆变换）
        intrins_inv = torch.inverse(intrins)[:, :, :3, :3].view(B*N, 3, 3)  # [B*N, 3, 3]
        dir_vector = torch.bmm(intrins_inv, dir_vector)  # [B*N, 3, H*W]
        
        # 应用sensor2keyego变换（相机到ego坐标系）
        sensor2keyego_rot = sensor2keyego[:, :, :3, :3].reshape(B*N, 3, 3)  # [B*N, 3, 3]
        dir_vector = torch.bmm(sensor2keyego_rot, dir_vector)  # [B*N, 3, H*W]
        
        # 应用BDD变换
        if bda is not None:
            bda_rot = bda[:, :3, :3].view(B, 1, 3, 3).expand(B, N, 3, 3).contiguous().view(B*N, 3, 3)
            dir_vector = torch.bmm(bda_rot, dir_vector)  # [B*N, 3, H*W]
        
        # 重塑回原始形状
        ego_dir_vector = dir_vector.view(B*N, 3, H, W)
        
        # 归一化旋转后的方向向量
        ego_dir_vector = F.normalize(ego_dir_vector, dim=1)
        
        # 将旋转后的方向向量转换回角度表示
        # 计算极角theta (与z轴夹角)
        ego_z = ego_dir_vector[:, 2]
        ego_h_angle = torch.acos(torch.clamp(ego_z, -1.0, 1.0))  # [B*N, H, W]
        
        # 计算方位角phi (在xy平面上的投影与x轴夹角)
        ego_x = ego_dir_vector[:, 0]
        ego_y = ego_dir_vector[:, 1]
        ego_w_angle = torch.atan2(ego_y, ego_x)  # [B*N, H, W]
        
        # 归一化phi到0-2pi范围
        ego_w_angle = ego_w_angle % (2 * torch.pi)
        
        # 计算与y轴的夹角（D方向）
        ego_d_angle = torch.acos(torch.clamp(ego_y, -1.0, 1.0))
        
        # 构建ego坐标系下的normal角度结果
        ego_normal_results = {
            'ego_w_angle': ego_w_angle,  # W方向角度（弧度制）
            'ego_h_angle': ego_h_angle,  # H方向角度（弧度制）
            'ego_d_angle': ego_d_angle,  # D方向角度（弧度制）
            'ego_dir_vector': ego_dir_vector,  # 归一化后的方向向量
            'gt_depth': normal_preds.get('gt_depth', None),
            'cam_dir_vector': cam_dir_vector
        }
        
        return ego_normal_results
    def transform_normal_to_ego_fix_remove_intrins_K(self, normal_preds, input):
        """
        将normal概率从相机坐标系变换到ego坐标系
        【核心修改】移除了多余的相机内参逆变换（因法向量来自LiDAR物理3D推导，无内参畸变）
        
        Args:
            normal_preds: 包含normal_comp_W_bins, normal_comp_H_bins, normal_comp_depth_bins预测的字典
            input: 输入数据，包含x, sensor2keyego, ego2global, intrins, post_rot, post_aug, bda, mlp_input
            
        Returns:
            dict: ego坐标系下的normal角度数值，包含ego_w_angle、ego_h_angle、ego_d_angle
        """
        # 从input中解包参数（intrins保留但不再使用，避免修改输入解包逻辑）
        x, sensor2keyego, ego2global, intrins, post_rot, post_aug, bda, mlp_input = input[:8]
        
        # 获取概率分布
        normal_w_prob = normal_preds['normal_comp_W_bins']
        normal_h_prob = normal_preds['normal_comp_H_bins']
        normal_depth_prob = normal_preds['normal_comp_depth_bins']
        
        # 获取形状信息
        B, N, _, _ = sensor2keyego.shape
        H, W = normal_w_prob.shape[-2:]  # [B*N, num_bins, H, W]
        
        # 创建bin中心角度（弧度制）
        # W方向和H方向都是0-180度，D方向是0-90度
        num_w_bins = normal_w_prob.shape[1]
        num_h_bins = normal_h_prob.shape[1]
        num_d_bins = normal_depth_prob.shape[1]
        
        # 计算bin中心角度（弧度制）
        w_angles_rad = torch.linspace(0, torch.pi-torch.pi/num_w_bins, num_w_bins, device=normal_w_prob.device)
        h_angles_rad = torch.linspace(0, torch.pi-torch.pi/num_h_bins, num_h_bins, device=normal_w_prob.device)
        d_angles_rad = torch.linspace(0, torch.pi/2-torch.pi/2/num_d_bins, num_d_bins, device=normal_w_prob.device)
        
        # 计算期望角度
        w_expected_angle = torch.sum(normal_w_prob * w_angles_rad.view(1, -1, 1, 1), dim=1)
        h_expected_angle = torch.sum(normal_h_prob * h_angles_rad.view(1, -1, 1, 1), dim=1)
        d_expected_angle = torch.sum(normal_depth_prob * d_angles_rad.view(1, -1, 1, 1), dim=1)
        
        # 从球面坐标转换为笛卡尔坐标（方向向量）
        # theta = h_expected_angle (极角，与z轴夹角)
        # phi = w_expected_angle (方位角，在xy平面上的投影与x轴夹角)
        # r = 1 (单位向量)
        sin_theta = torch.sin(h_expected_angle)
        cos_theta = torch.cos(h_expected_angle)
        sin_phi = torch.sin(w_expected_angle)
        cos_phi = torch.cos(w_expected_angle)
        
        # 计算方向向量的x, y, z分量（相机物理坐标系，无内参畸变）
        x_dir = sin_theta * cos_phi
        y_dir = sin_theta * sin_phi
        z_dir = cos_theta
        
        # 归一化方向向量
        dir_vector = torch.stack([x_dir, y_dir, z_dir], dim=1)  # [B*N, 3, H, W]
        dir_vector = F.normalize(dir_vector, dim=1)
        cam_dir_vector = dir_vector  # 保存相机坐标系原始方向向量
        
        # 重塑方向向量以适应后续变换：[B*N, 3, H*W]
        dir_vector_reshaped = dir_vector.view(B*N, 3, -1)
        
        # 1. 处理后旋转逆变换（必须保留：对齐图像增强前的相机轴）
        post_rot_inv = torch.inverse(post_rot).view(B*N, 3, 3)  # [B*N, 3, 3]
        dir_vector = torch.bmm(post_rot_inv, dir_vector_reshaped)  # [B*N, 3, H*W]
        
        # 【关键删除】移除相机内参逆变换（法向量来自LiDAR物理3D，无内参畸变，无需校正）
        # --- 原错误代码已删除 ---
        # intrins_inv = torch.inverse(intrins)[:, :, :3, :3].view(B*N, 3, 3)
        # dir_vector = torch.bmm(intrins_inv, dir_vector)
        
        # 2. 应用sensor2keyego变换（相机坐标系 → ego坐标系，必须保留）
        sensor2keyego_rot = sensor2keyego[:, :, :3, :3].reshape(B*N, 3, 3)  # [B*N, 3, 3]
        dir_vector = torch.bmm(sensor2keyego_rot, dir_vector)  # [B*N, 3, H*W]
        
        # 3. 应用BDD变换（可选，根据数据集需求保留）
        if bda is not None:
            bda_rot = bda[:, :3, :3].view(B, 1, 3, 3).expand(B, N, 3, 3).contiguous().view(B*N, 3, 3)
            dir_vector = torch.bmm(bda_rot, dir_vector)  # [B*N, 3, H*W]
        
        # 重塑回原始形状：[B*N, 3, H, W]
        ego_dir_vector = dir_vector.view(B*N, 3, H, W)
        
        # 归一化旋转后的方向向量（确保单位向量）
        ego_dir_vector = F.normalize(ego_dir_vector, dim=1)
        
        # 将ego坐标系方向向量转换回角度表示
        # 极角theta (与z轴夹角) → ego_h_angle
        ego_z = ego_dir_vector[:, 2]
        ego_h_angle = torch.acos(torch.clamp(ego_z, -1.0, 1.0))  # [B*N, H, W]
        
        # 方位角phi (xy平面投影与x轴夹角) → ego_w_angle
        ego_x = ego_dir_vector[:, 0]
        ego_y = ego_dir_vector[:, 1]
        ego_w_angle = torch.atan2(ego_y, ego_x)  # [B*N, H, W]
        ego_w_angle = ego_w_angle % (2 * torch.pi)  # 归一化到0-2pi
        
        # 与y轴的夹角 → ego_d_angle
        ego_d_angle = torch.acos(torch.clamp(ego_y, -1.0, 1.0))
        
        # 构建返回结果
        ego_normal_results = {
            'ego_w_angle': ego_w_angle,  # W方向角度（弧度制）
            'ego_h_angle': ego_h_angle,  # H方向角度（弧度制）
            'ego_d_angle': ego_d_angle,  # D方向角度（弧度制）
            'ego_dir_vector': ego_dir_vector,  # ego坐标系归一化方向向量
            'gt_depth': normal_preds.get('gt_depth', None),
            'cam_dir_vector': cam_dir_vector  # 相机坐标系原始方向向量
        }
        
        return ego_normal_results

    def forward(self, input, stereo_metas=None):
        """
        前向传播函数，扩展了原始LSSVStereoForwardPorjection的功能
        
        Args:
            input: 输入数据，包含x, rots, trans, intrins, post_rot, post_aug, bda, mlp_input
            stereo_metas: 立体视觉相关的元数据
            
        Returns:
            tuple: (bev_feat, depth, tran_feat, normal_preds)
                bev_feat: BEV特征
                depth: 深度预测
                tran_feat: 转换特征
                normal_preds: 包含normal_comp_W_bins, normal_comp_H_bins, normal_comp_depth_bins预测的字典
        """
        x, sensor2keyego, ego2global, intrins, post_rot, post_aug, bda, mlp_input = input[:8]

        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        start_time = time.time()
        
        # 使用扩展的深度网络获取预测结果
        combined_output = self.depth_net(x, mlp_input, stereo_metas)
        end_time = time.time()
        
        # 分割不同的预测结果
        depth_digit = combined_output[:, :self.D, ...]  # 原始深度预测
        normal_w_digit = combined_output[:, self.D:self.D+self.normal_w_bins, ...]  # normal_comp_W_bins预测
        normal_h_digit = combined_output[:, self.D+self.normal_w_bins:self.D+self.normal_w_bins+self.normal_h_bins, ...]  # normal_comp_H_bins预测
        normal_depth_digit = combined_output[:, self.D+self.normal_w_bins+self.normal_h_bins:self.D+self.normal_w_bins+self.normal_h_bins+self.normal_depth_bins, ...]  # normal_comp_depth_bins预测
        tran_feat = combined_output[:, self.D+self.normal_w_bins+self.normal_h_bins+self.normal_depth_bins:, ...]  # 上下文特征
        
        # 对深度预测进行softmax
        depth = depth_digit.softmax(dim=1)
        
        # 对normal bins预测进行softmax
        normal_w_prob = normal_w_digit.softmax(dim=1)
        normal_h_prob = normal_h_digit.softmax(dim=1)
        normal_depth_prob = normal_depth_digit.softmax(dim=1)
        
        # 构建normal预测结果字典
        normal_preds = {
            'normal_comp_W_bins': normal_w_prob,
            'normal_comp_H_bins': normal_h_prob,
            'normal_comp_depth_bins': normal_depth_prob
        }
        
        # 将normal概率从相机坐标系变换到ego坐标系
        # normal_preds.update(self.transform_normal_to_ego(normal_preds, input))
        if self.use_ego_2D_3D_trans_fix:
            normal_preds.update(self.transform_normal_to_ego_fix_remove_intrins_K(normal_preds, input))
        else:
        # 将normal概率从相机坐标系变换到ego坐标系
            normal_preds.update(self.transform_normal_to_ego(normal_preds, input))
        # 打印张量shape以进行调试
        # print(f"tran_feat shape: {tran_feat.shape}")
        # print(f"ego_dir_vector shape: {normal_preds['ego_dir_vector'].shape}")
        
        # 将ego_dir_vector与tran_feat进行拼接
        # 确保两者的空间维度匹配
        assert tran_feat.shape[2:] == normal_preds['ego_dir_vector'].shape[2:], "空间维度不匹配"
        
        # 在通道维度（dim=1）上拼接
        # tran_feat_with_normal = torch.cat([tran_feat, normal_preds['ego_dir_vector']], dim=1)
        tran_feat_with_normal = torch.cat([tran_feat, normal_preds['ego_dir_vector'],normal_preds['normal_comp_W_bins'],normal_preds['normal_comp_H_bins'],normal_preds['normal_comp_depth_bins']], dim=1)
        
        # print(f"拼接后的tran_feat_with_normal shape: {tran_feat_with_normal.shape}")
        
        # 进行视图变换
        # 注意：这里使用拼接后的特征
        self.out_channels=83+20
        bev_feat, depth = self.view_transform(input, depth, tran_feat_with_normal)
        # print(f"view_transform time: {(end_time - start_time):.4f} s")
        bev_feat = bev_feat[:, :self.out_channels, ...]
        normal_preds['ego_dir_vector_3d'] = bev_feat[:, -3:, ...]
        normal_preds['tran_feat'] = tran_feat
        if self.return_context:
            return bev_feat, depth, tran_feat_with_normal, normal_preds
        else:
            return bev_feat, depth, normal_preds
    
    @torch.no_grad()
    def get_downsampled_gt_normal(self, gt_normals, bins):
        """
        下采样真实的normal bins标签
        
        Args:
            gt_normals: 真实的normal bins标签 [B, N, H, W]
            bins: bins的数量
            
        Returns:
            torch.Tensor: 下采样后的one-hot编码normal标签 [B*N, H//downsample, W//downsample, bins]
        """
        B, N, H, W = gt_normals.shape
        
        # 下采样（使用最大值池化来保持bin索引）
        gt_normals = gt_normals.view(B * N, 1, H, W)
        gt_normals = F.max_pool2d(gt_normals, kernel_size=self.downsample, stride=self.downsample)
        gt_normals = gt_normals.squeeze(1).long()
        
        # 转换为one-hot编码
        gt_normals = F.one_hot(gt_normals, num_classes=bins)
        
        return gt_normals
    
    def get_normal_loss(self, normal_labels, normal_preds):
        """
        计算normal预测的损失
        
        Args:
            normal_labels: 真实的normal标签字典
            normal_preds: 预测的normal概率字典
            
        Returns:
            torch.Tensor: normal预测的总损失
        """
        total_loss = 0.0
        
        # 计算normal_comp_W_bins的损失
        if 'normal_comp_W_bins' in normal_labels and 'normal_comp_W_bins' in normal_preds:
            gt_w = self.get_downsampled_gt_normal(normal_labels['normal_comp_W_bins'], self.normal_w_bins)
            pred_w = normal_preds['normal_comp_W_bins'].permute(0, 2, 3, 1).contiguous()
            
            # 计算交叉熵损失
            w_loss = F.cross_entropy(
                pred_w.reshape(-1, self.normal_w_bins),
                gt_w.reshape(-1, self.normal_w_bins)
            )
            total_loss += w_loss
        
        # 计算normal_comp_H_bins的损失
        if 'normal_comp_H_bins' in normal_labels and 'normal_comp_H_bins' in normal_preds:
            gt_h = self.get_downsampled_gt_normal(normal_labels['normal_comp_H_bins'], self.normal_h_bins)
            pred_h = normal_preds['normal_comp_H_bins'].permute(0, 2, 3, 1).contiguous()
            
            # 计算交叉熵损失
            h_loss = F.cross_entropy(
                pred_h.reshape(-1, self.normal_h_bins),
                gt_h.reshape(-1, self.normal_h_bins)
            )
            total_loss += h_loss
        
        # 计算normal_comp_depth_bins的损失
        if 'normal_comp_depth_bins' in normal_labels and 'normal_comp_depth_bins' in normal_preds:
            gt_depth = self.get_downsampled_gt_normal(normal_labels['normal_comp_depth_bins'], self.normal_depth_bins)
            pred_depth = normal_preds['normal_comp_depth_bins'].permute(0, 2, 3, 1).contiguous()
            
            # 计算交叉熵损失
            depth_loss = F.cross_entropy(
                pred_depth.reshape(-1, self.normal_depth_bins),
                gt_depth.reshape(-1, self.normal_depth_bins)
            )
            total_loss += depth_loss
        
        # 应用权重
        return self.loss_normal_weight * total_loss
        
    def get_downsampled_normal_W(self, normal_comp_W_bins, W_bins):
        """
        对W方向的离散化方向向量进行下采样，处理无效值0，输出独热编码（优化版）
        
        Input:
            normal_comp_W_bins: [B, N, H, W] 
                - B: 批量大小, N: 相机数量, H: 高度, W: 宽度
                - 每个元素是离散化的方向向量（整数），0表示无效值
        Output:
            downsampled_W: [B*N*h*w, W_bins]
                - h = H // self.downsample, w = W // self.downsample
                - W_bins: 方向向量的离散化类别数（对应self.W_bins）
        """
        # 1. 解析输入维度
        B, N, H, W = normal_comp_W_bins.shape
        s = self.downsample  # 下采样率（如2、4等）
        h = H // s  # 下采样后高度
        w = W // s  # 下采样后宽度
        num_blocks = B * N * h * w

        # 2. 拆分下采样块并展平 (与原版逻辑一致)
        # 形状变化：[B, N, H, W] -> [B*N, h, s, w, s] -> [B*N, h, w, s*s] -> [num_blocks, s*s]
        normal_bins = normal_comp_W_bins.view(B * N, h, s, w, s)
        normal_bins = normal_bins.permute(0, 1, 3, 2, 4).contiguous()
        normal_bins = normal_bins.view(num_blocks, s * s)
        
        # 3. 向量化计算众数
        # 3.1 将无效值0替换为一个超出类别范围的标记 (例如 -1)
        normal_bins_valid = torch.where(normal_bins != 0, normal_bins, torch.tensor(-1, device=normal_bins.device))
        
        # 3.2 生成所有可能的类别值 [1, 2, ..., W_bins]
        # 形状: [W_bins]
        classes = torch.arange(1, W_bins + 1, device=normal_bins.device)
        
        # 3.3 批量计算每个块内每个类别的出现次数
        # 使用广播机制: [num_blocks, s*s] != [1, W_bins] -> [num_blocks, s*s, W_bins]
        # 然后在 s*s 维度求和，得到每个块中每个类别的计数
        # 形状: [num_blocks, W_bins]
        counts = (normal_bins_valid.unsqueeze(-1) == classes).sum(dim=1)
        
        # 3.4 找到每个块中计数最多的类别索引
        # 形状: [num_blocks]
        mode_indices = counts.argmax(dim=1)
        
        # 3.5 将索引转换回对应的类别值
        # 形状: [num_blocks]
        downsampled = classes[mode_indices]
        
        # 3.6 处理全无效值的块 (所有计数都为0)
        # 计算每个块的有效像素数量
        valid_pixel_count = (normal_bins != 0).sum(dim=1)
        # 对于有效像素为0的块，将其值设为0
        downsampled[valid_pixel_count == 0] = 0

        # 4. 恢复下采样后的空间形状 (与原版逻辑一致)
        # 形状: [num_blocks] -> [B*N, h, w]
        downsampled = downsampled.view(B * N, h, w)
        
        # 5. 过滤异常值 (与原版逻辑一致)
        valid_range_mask = (downsampled >= 1) & (downsampled <= W_bins)
        downsampled = torch.where(valid_range_mask, downsampled, torch.tensor(0, device=downsampled.device))
        
        # 6. 转换为独热编码 (与原版逻辑一致)
        one_hot = F.one_hot(downsampled, num_classes=W_bins + 1)  # [B*N, h, w, W_bins+1]
        one_hot = one_hot.view(-1, W_bins + 1)[:, 1:]  # [B*N*h*w, W_bins]
        
        return one_hot.float()

    def get_normal_W_loss(self, normal_W_labels, normal_W_preds):
        """
        计算W方向离散化方向向量的损失（参考深度损失逻辑）
        
        Input:
            normal_W_labels: [B, N, H, W] 
                - 原始W方向标签（离散化整数，0为无效值）
            normal_W_preds: [B, W_bins, h, w] 或 [B*N, W_bins, h, w]
                - 模型预测的W方向向量概率（经过softmax前或后，根据模型输出定）
        Output:
            normal_W_loss: 标量损失值
        """
        # 1. 对标签进行下采样处理（使用之前实现的下采样函数）
        # 输出形状：[B*N*h*w, W_bins]（独热编码，排除无效值0的类别）
        normal_W_labels = self.get_downsampled_normal_W(normal_W_labels,8)
        
        # 2. 调整预测的形状，与标签匹配
        # 假设输入预测形状为 [B, N, W_bins, h, w] 或 [B*N, W_bins, h, w]
        # 转换为 [B*N*h*w, W_bins]（与标签维度一致）
        normal_W_preds = normal_W_preds.permute(0, 2, 3, 1).contiguous()  # [B*N, h, w, W_bins]
        normal_W_preds = normal_W_preds.view(-1, self.W_bins)  # [B*N*h*w, W_bins]
        
        # 3. 生成前景掩码（过滤无效值0对应的标签）
        # 标签是独热编码，最大值>0表示有效像素（非0）
        fg_mask = torch.max(normal_W_labels, dim=1).values > 0.0  # [B*N*h*w]
        
        # 4. 过滤无效的标签和预测
        normal_W_labels = normal_W_labels[fg_mask]  # [valid_num, W_bins]
        normal_W_preds = normal_W_preds[fg_mask]    # [valid_num, W_bins]
        
        # 5. 计算损失（禁用自动混合精度，确保计算精度）
        with autocast(enabled=False):
            # 方向向量是离散类别，用二元交叉熵（与深度损失一致，适配独热编码）
            normal_W_loss = F.binary_cross_entropy(
                normal_W_preds,
                normal_W_labels,
                reduction='none'  # 先不求和，后续按有效像素数平均
            ).sum() / max(1.0, fg_mask.sum())  # 除以有效像素数，避免样本不平衡
        
        # 6. 乘以损失权重并返回
        return self.loss_normal_W_weight * normal_W_loss
    
    def get_normal_H_loss(self, normal_W_labels, normal_W_preds):
        """
        计算W方向离散化方向向量的损失（参考深度损失逻辑）
        
        Input:
            normal_W_labels: [B, N, H, W] 
                - 原始W方向标签（离散化整数，0为无效值）
            normal_W_preds: [B, W_bins, h, w] 或 [B*N, W_bins, h, w]
                - 模型预测的W方向向量概率（经过softmax前或后，根据模型输出定）
        Output:
            normal_W_loss: 标量损失值
        """
        # 1. 对标签进行下采样处理（使用之前实现的下采样函数）
        # 输出形状：[B*N*h*w, W_bins]（独热编码，排除无效值0的类别）
        normal_W_labels = self.get_downsampled_normal_W(normal_W_labels,8)
        
        # 2. 调整预测的形状，与标签匹配
        # 假设输入预测形状为 [B, N, W_bins, h, w] 或 [B*N, W_bins, h, w]
        # 转换为 [B*N*h*w, W_bins]（与标签维度一致）
        normal_W_preds = normal_W_preds.permute(0, 2, 3, 1).contiguous()  # [B*N, h, w, W_bins]
        normal_W_preds = normal_W_preds.view(-1, self.H_bins)  # [B*N*h*w, W_bins]
        
        # 3. 生成前景掩码（过滤无效值0对应的标签）
        # 标签是独热编码，最大值>0表示有效像素（非0）
        fg_mask = torch.max(normal_W_labels, dim=1).values > 0.0  # [B*N*h*w]
        
        # 4. 过滤无效的标签和预测
        normal_W_labels = normal_W_labels[fg_mask]  # [valid_num, W_bins]
        normal_W_preds = normal_W_preds[fg_mask]    # [valid_num, W_bins]
        
        # 5. 计算损失（禁用自动混合精度，确保计算精度）
        # with autocast(enabled=False):
            # 方向向量是离散类别，用二元交叉熵（与深度损失一致，适配独热编码）
        normal_W_loss = F.binary_cross_entropy(
            normal_W_preds,
            normal_W_labels,
            reduction='none'  # 先不求和，后续按有效像素数平均
        ).sum() / max(1.0, fg_mask.sum())  # 除以有效像素数，避免样本不平衡
        
        # 6. 乘以损失权重并返回
        return self.loss_normal_H_weight * normal_W_loss

    def get_normal_D_loss(self, normal_W_labels, normal_W_preds):
        """
        计算W方向离散化方向向量的损失（参考深度损失逻辑）
        
        Input:
            normal_W_labels: [B, N, H, W] 
                - 原始W方向标签（离散化整数，0为无效值）
            normal_W_preds: [B, W_bins, h, w] 或 [B*N, W_bins, h, w]
                - 模型预测的W方向向量概率（经过softmax前或后，根据模型输出定）
        Output:
            normal_W_loss: 标量损失值
        """
        # 1. 对标签进行下采样处理（使用之前实现的下采样函数）
        # 输出形状：[B*N*h*w, W_bins]（独热编码，排除无效值0的类别）
        normal_W_labels = self.get_downsampled_normal_W(normal_W_labels,4)
        
        # 2. 调整预测的形状，与标签匹配
        # 假设输入预测形状为 [B, N, W_bins, h, w] 或 [B*N, W_bins, h, w]
        # 转换为 [B*N*h*w, W_bins]（与标签维度一致）
        normal_W_preds = normal_W_preds.permute(0, 2, 3, 1).contiguous()  # [B*N, h, w, W_bins]
        normal_W_preds = normal_W_preds.view(-1, self.D_bins)  # [B*N*h*w, W_bins]
        
        # 3. 生成前景掩码（过滤无效值0对应的标签）
        # 标签是独热编码，最大值>0表示有效像素（非0）
        fg_mask = torch.max(normal_W_labels, dim=1).values > 0.0  # [B*N*h*w]
        
        # 4. 过滤无效的标签和预测
        normal_W_labels = normal_W_labels[fg_mask]  # [valid_num, W_bins]
        normal_W_preds = normal_W_preds[fg_mask]    # [valid_num, W_bins]
        
        # 5. 计算损失（禁用自动混合精度，确保计算精度）
        # with autocast(enabled=False):
            # 方向向量是离散类别，用二元交叉熵（与深度损失一致，适配独热编码）
        normal_W_loss = F.binary_cross_entropy(
            normal_W_preds,
            normal_W_labels,
            reduction='none'  # 先不求和，后续按有效像素数平均
        ).sum() / max(1.0, fg_mask.sum())  # 除以有效像素数，避免样本不平衡
        
        # 6. 乘以损失权重并返回
        return self.loss_normal_D_weight * normal_W_loss