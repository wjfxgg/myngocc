# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from .view_transformer import DepthNet

class DepthNetWithNormalImproved(DepthNet):
    """
    改进版的DepthNetWithNormal，将context和depth特征拼接后生成normal_feature
    - 保留原始的深度预测功能
    - 新增对normal_comp_W_bins (8类)、normal_comp_H_bins (8类)、normal_comp_depth_bins (4类)的预测
    - 改进点：将context和depth特征拼接后再生成normal_feature
    """
    
    def __init__(self, 
                 in_channels,
                 mid_channels,
                 context_channels,
                 depth_channels,
                 normal_w_bins=8,  # normal_comp_W_bins的类别数
                 normal_h_bins=8,  # normal_comp_H_bins的类别数
                 normal_depth_bins=4,  # normal_comp_depth_bins的类别数
                 **kwargs):
        """
        初始化改进的深度网络
        
        Args:
            in_channels: 输入特征通道数
            mid_channels: 中间特征通道数
            context_channels: 上下文特征通道数
            depth_channels: 深度预测通道数
            normal_w_bins: normal_comp_W_bins的类别数
            normal_h_bins: normal_comp_H_bins的类别数
            normal_depth_bins: normal_comp_depth_bins的类别数
            **kwargs: 其他传递给DepthNet的参数
        """
        super().__init__(in_channels, mid_channels, context_channels, depth_channels, **kwargs)
        
        # 保存normal bins的配置信息
        self.normal_w_bins = normal_w_bins
        self.normal_h_bins = normal_h_bins
        self.normal_depth_bins = normal_depth_bins
        
        # 为normal_comp_W_bins创建专门的预测头
        self.normal_w_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, normal_w_bins, kernel_size=1, stride=1, padding=0)
        )
        
        # 为normal_comp_H_bins创建专门的预测头
        self.normal_h_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, normal_h_bins, kernel_size=1, stride=1, padding=0)
        )
        
        # 为normal_comp_depth_bins创建专门的预测头
        self.normal_depth_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, normal_depth_bins, kernel_size=1, stride=1, padding=0)
        )
        
        # 改进：创建用于normal预测的特征提取器，输入为depth特征和context特征的拼接
        # 输入通道 = depth特征通道 + context特征通道
        # depth特征通道 = mid_channels + (depth_channels if hasattr(self, 'cost_volumn_net') else 0)
        # context特征通道 = context_channels
        input_channels = mid_channels + context_channels
        if hasattr(self, 'cost_volumn_net'):
            input_channels += depth_channels
        
        self.normal_feature_conv = nn.Sequential(
            nn.Conv2d(input_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, mlp_input, stereo_metas=None):
        """
        前向传播函数，改进了normal特征的生成方式
        
        Args:
            x: 输入特征 [B*N, C, H, W]
            mlp_input: MLP输入 [B*N, 27]
            stereo_metas: 立体视觉相关的元数据
            
        Returns:
            torch.Tensor: 包含深度预测和normal bins预测的拼接结果
                [B*N, depth_channels + normal_w_bins + normal_h_bins + normal_depth_bins + context_channels, H, W]
        """
        # 首先获取原始DepthNet的中间特征（在应用depth_conv之前）
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)
        
        # 计算上下文特征（与原始DepthNet相同）
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
        
        # 计算深度特征（与原始DepthNet相同）
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)
        
        # 如果启用了立体视觉，计算代价体积并与深度特征拼接
        if not stereo_metas is None:
            if stereo_metas['cv_feat_list'][0] is None:
                BN, _, H, W = x.shape
                scale_factor = float(stereo_metas['downsample'])/stereo_metas['cv_downsample']
                cost_volumn = \
                    torch.zeros((BN, self.depth_channels,
                                 int(H*scale_factor),
                                 int(W*scale_factor))).to(x)
            else:
                with torch.no_grad():
                    cost_volumn = self.calculate_cost_volumn(stereo_metas)
            cost_volumn = self.cost_volumn_net(cost_volumn)
            depth = torch.cat([depth, cost_volumn], dim=1)
        
        # 改进：将context和depth特征拼接后再生成normal_feature
        # 确保context和depth的空间尺寸一致
        if context.shape[2:] != depth.shape[2:]:
            context = F.interpolate(context, size=depth.shape[2:], mode='bilinear', align_corners=False)
        
        # 拼接context和depth特征
        combined_feature = torch.cat([depth, context], dim=1)
        
        # 创建专门用于normal预测的特征
        normal_feature = self.normal_feature_conv(combined_feature)
        
        # 预测深度（与原始DepthNet相同）
        if self.with_cp:
            depth_output = torch.utils.checkpoint.checkpoint(self.depth_conv, depth)
        else:
            depth_output = self.depth_conv(depth)
        
        # 预测normal相关的bins
        normal_w_output = self.normal_w_conv(normal_feature)
        normal_h_output = self.normal_h_conv(normal_feature)
        normal_depth_output = self.normal_depth_conv(normal_feature)
        
        # 拼接所有输出：深度预测 + normal预测 + 上下文特征
        return torch.cat([
            depth_output,            # 原始深度预测
            normal_w_output,         # normal_comp_W_bins预测
            normal_h_output,         # normal_comp_H_bins预测
            normal_depth_output,     # normal_comp_depth_bins预测
            context                  # 上下文特征
        ], dim=1)
