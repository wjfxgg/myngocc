# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from .view_transformer import DepthNet

class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class DepthNetWithNormalIsolated(DepthNet):
    """
    扩展版的DepthNet，增加了对normal_comp相关bins的预测能力
    - 保留原始的深度预测功能
    - 新增对normal_comp_W_bins (8类)、normal_comp_H_bins (8类)、normal_comp_depth_bins (4类)的预测
    - 改进：使用与depth和context完全隔离的方式计算normal_feature
        - 独立的normal_mlp用于计算SE参数
        - 独立的normal_se用于应用注意力机制
        - 独立的normal_context_conv用于处理normal特征
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
        初始化扩展的深度网络
        
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
        
        # 改进：使用与depth和context完全隔离的方式计算normal_feature
        # 1. 创建normal专用的MLP，用于生成SE参数
        self.normal_mlp = nn.Sequential(
            nn.Linear(27, 256),  # 与depth_mlp和context_mlp保持一致
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, mid_channels),
            nn.Sigmoid()
        )
        
        # 2. 创建normal专用的SE层，用于应用注意力机制
        # self.normal_se = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(mid_channels, mid_channels // 16, kernel_size=1, padding=0),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(mid_channels // 16, mid_channels, kernel_size=1, padding=0),
        #     nn.Sigmoid()
        # )
        self.normal_se = SELayer(mid_channels)  # NOTE: add camera-aware
        
        # 3. 创建normal专用的卷积层，用于处理normal特征
        self.normal_context_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
    
        self.normal_mlp = Mlp(27, mid_channels, mid_channels)

    def forward(self, x, mlp_input, stereo_metas=None):
        """
        前向传播函数，扩展了原始DepthNet的功能，使用完全隔离的方式计算normal_feature
        
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
        
        # 改进：使用与depth和context完全隔离的方式计算normal_feature
        # 1. 计算normal专用的SE参数
        normal_se = self.normal_mlp(mlp_input)[..., None, None]
        # 2. 应用normal专用的SE层
        normal = self.normal_se(x, normal_se)
        # 3. 应用normal专用的卷积层
        normal_feature = self.normal_context_conv(normal)
        
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
