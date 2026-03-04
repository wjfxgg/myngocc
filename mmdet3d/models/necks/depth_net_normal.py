# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from .view_transformer import DepthNet
import time

class DepthNetWithNormal(DepthNet):
    """
    扩展版的DepthNet，增加了对normal_comp相关bins的预测能力
    - 保留原始的深度预测功能
    - 新增对normal_comp_W_bins (8类)、normal_comp_H_bins (8类)、normal_comp_depth_bins (4类)的预测
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
        
        # 创建一个额外的卷积层来提取专门用于normal预测的特征
        # 输入通道需要考虑depth_se后的通道数加上可能的cost_volumn通道数
        self.normal_feature_conv = nn.Sequential(
            nn.Conv2d(mid_channels + (self.depth_channels if hasattr(self, 'cost_volumn_net') else 0), mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, mlp_input, stereo_metas=None):
        """
        前向传播函数，扩展了原始DepthNet的功能
        
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
        
        # 创建专门用于normal预测的特征
        normal_feature = self.normal_feature_conv(depth)
        
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
    import time

    def forward_new(self, x, mlp_input, stereo_metas=None):
        """
        前向传播函数，扩展了原始DepthNet的功能
        
        Args:
            x: 输入特征 [B*N, C, H, W]
            mlp_input: MLP输入 [B*N, 27]
            stereo_metas: 立体视觉相关的元数据
            
        Returns:
            torch.Tensor: 包含深度预测和normal bins预测的拼接结果
                [B*N, depth_channels + normal_w_bins + normal_h_bins + normal_depth_bins + context_channels, H, W]
        """
        # 初始化时间统计字典，方便后续打印
        time_stats = {}
        start_total = time.time()

        # 1. 基础预处理：bn + reduce_conv
        start = time.time()
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)
        time_stats['1. 基础预处理(bn+reduce_conv)'] = time.time() - start

        # 2. 上下文特征计算
        start = time.time()
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
        time_stats['2. 上下文特征计算(context_mlp+se+conv)'] = time.time() - start

        # 3. 深度特征计算（基础部分）
        start = time.time()
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)
        time_stats['3. 深度特征基础计算(depth_mlp+se)'] = time.time() - start

        # 4. 立体视觉相关处理（可选）
        stereo_time = 0.0
        # TODO 加快速度
        # stereo_metas = None
        if not stereo_metas is None:
            start_stereo = time.time()
            if True:
                # cost_volumn = depth.clone()
                pass
            elif stereo_metas['cv_feat_list'][0] is None:
                BN, _, H, W = x.shape
                scale_factor = float(stereo_metas['downsample'])/stereo_metas['cv_downsample']
                cost_volumn = \
                    torch.zeros((BN, self.depth_channels,
                                int(H*scale_factor),
                                int(W*scale_factor))).to(x)
            else:
                with torch.no_grad():
                    cost_volumn = self.calculate_cost_volumn(stereo_metas)
                    #TODO 加快速度
                    cost_volumn = depth.clone
            # 计算代价体积网络处理时间
            start_cv_net = time.time()
            cost_volumn = self.cost_volumn_net(stereo_metas['cv_feat_list'][0])
            cv_net_time = time.time() - start_cv_net
            # 拼接操作
            depth = torch.cat([depth, cost_volumn], dim=1)
            stereo_time = time.time() - start_stereo
            time_stats['4. 立体视觉处理(含cost_volumn计算)'] = stereo_time
            time_stats['  - 其中cost_volumn_net耗时'] = cv_net_time

        # 5. Normal特征提取
        start = time.time()
        normal_feature = self.normal_feature_conv(depth)
        time_stats['5. Normal特征提取(normal_feature_conv)'] = time.time() - start

        # 6. 深度预测（含checkpoint分支）
        start = time.time()
        if self.with_cp:
            depth_output = torch.utils.checkpoint.checkpoint(self.depth_conv, depth)
        else:
            depth_output = self.depth_conv(depth)
        time_stats['6. 深度预测(depth_conv)'] = time.time() - start

        # 7. Normal相关预测（三个分支）
        start = time.time()
        normal_w_output = self.normal_w_conv(normal_feature)
        normal_h_output = self.normal_h_conv(normal_feature)
        normal_depth_output = self.normal_depth_conv(normal_feature)
        time_stats['7. Normal三分支预测(normal_w/h/depth_conv)'] = time.time() - start

        # 8. 最终拼接
        start = time.time()
        output = torch.cat([
            depth_output,            # 原始深度预测
            normal_w_output,         # normal_comp_W_bins预测
            normal_h_output,         # normal_comp_H_bins预测
            normal_depth_output,     # normal_comp_depth_bins预测
            context                  # 上下文特征
        ], dim=1)
        time_stats['8. 最终特征拼接'] = time.time() - start

        # 总耗时
        total_time = time.time() - start_total
        time_stats['总耗时'] = total_time

        # 打印时间统计结果（格式化输出，保留6位小数）
        # print("="*80)
        # print("Forward函数关键组件执行时间统计（CPU）：")
        # print("="*80)
        # for step_name, duration in time_stats.items():
        #     print(f"{step_name:<40}: {duration:.6f} 秒")
        # print("="*80)
        # print()  # 空行分隔

        return output