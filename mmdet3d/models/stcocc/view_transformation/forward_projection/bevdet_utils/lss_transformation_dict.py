# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, force_fp32
from torch.cuda.amp.autocast_mode import autocast

from mmdet3d.models.necks.view_transformer import LSSViewTransformer
from mmdet3d.models.builder import NECKS


@NECKS.register_module()
class LSSForwardProjectionDict(LSSViewTransformer):
    """
    LSSForwardProjection的扩展版本，返回字典格式而不是元组。
    这是为了兼容STCOccWithoutPredictor类的调用方式。
    """

    def __init__(self,
                 loss_depth_weight=3.0,
                 depthnet_cfg=dict(),
                 return_context=False,
                 **kwargs):
        super(LSSForwardProjectionDict, self).__init__(**kwargs)
        self.loss_depth_weight = loss_depth_weight
        # 使用自定义的depth_net配置
        if depthnet_cfg:
            # 如果提供了自定义的depth_net配置，使用它
            # 这里可以根据需要扩展更复杂的初始化逻辑
            pass
        self.return_context = return_context

    def get_mlp_input(self, sensor2ego, ego2global, intrin, post_aug, bda):
        """
        生成MLP输入，这部分逻辑从原始LSSForwardProjection类复制
        """
        B, N, _, _ = sensor2ego.shape
        bda = bda.view(B, 1, 4, 4).repeat(1, N, 1, 1)
        post_rot = post_aug[:, :, :2, :2]
        post_tran = post_aug[:, :, :2, 2]
        mlp_input = torch.stack([
            # intrin info
            intrin[:, :, 0, 0],
            intrin[:, :, 1, 1],
            intrin[:, :, 0, 2],
            intrin[:, :, 1, 2],
            # post aug info
            post_rot[:, :, 0, 0],
            post_rot[:, :, 0, 1],
            post_tran[:, :, 0],
            post_rot[:, :, 1, 0],
            post_rot[:, :, 1, 1],
            post_tran[:, :, 1],
            # bda info
            bda[:, :, 0, 0],
            bda[:, :, 0, 1],
            bda[:, :, 1, 0],
            bda[:, :, 1, 1],
            bda[:, :, 2, 2],], dim=-1)
        sensor2ego = sensor2ego[:,:,:3,:].reshape(B, N, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        return mlp_input

    def get_downsampled_gt_depth(self, gt_depths):
        """
        下采样真实深度图，这部分逻辑从原始LSSForwardProjection类复制
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   self.downsample, W // self.downsample,
                                   self.downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   W // self.downsample)

        if not self.sid:
            gt_depths = (gt_depths - (self.grid_config['depth'][0] -
                                      self.grid_config['depth'][2])) / \
                        self.grid_config['depth'][2]
        else:
            gt_depths = torch.log(gt_depths) - torch.log(
                torch.tensor(self.grid_config['depth'][0]).float())
            gt_depths = gt_depths * (self.D - 1) / torch.log(
                torch.tensor(self.grid_config['depth'][1] - 1.).float() /
                self.grid_config['depth'][0])
            gt_depths = gt_depths + 1.

        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0), gt_depths, torch.zeros_like(gt_depths))

        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:, 1:]

        return gt_depths.float()

    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        """
        计算深度损失，这部分逻辑从原始LSSForwardProjection类复制
        """
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        return self.loss_depth_weight * depth_loss

    def forward(self, input, stereo_metas=None):
        """
        重写forward方法，返回字典格式而不是元组
        
        Args:
            input (list): 包含图像特征和各种变换矩阵的列表
            stereo_metas (dict, optional): 立体视觉相关的元数据
        
        Returns:
            dict: 包含以下键的字典：
                - voxel_feats: 体素特征列表
                - depth: 深度预测
                - tran_feats: 转换特征
                - cam_params: 相机参数列表
        """
        # 解包输入
        x, rots, trans, intrins, post_rot, post_aug, bda, mlp_input = input[:8]

        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)

        # 计算深度和转换特征
        x = self.depth_net(x)

        depth_digit = x[:, :self.D, ...]
        tran_feat = x[:, self.D:self.D + self.out_channels, ...]
        depth = depth_digit.softmax(dim=1)

        # 进行视图变换
        bev_feat, depth = self.view_transform(input, depth, tran_feat)

        # 返回字典格式以符合STCOccWithoutPredictor的期望
        return_dict = {
            'voxel_feats': [bev_feat],  # 包装在列表中
            'depth': depth,
            'tran_feats': tran_feat,
            'cam_params': [rots, trans, intrins, post_rot, post_aug, bda]  # 收集相机参数
        }
        
        # 如果需要返回上下文信息
        if self.return_context:
            return_dict['context'] = x[:, self.D + self.out_channels:, ...]
            
        return return_dict


@NECKS.register_module()
class LSSVStereoForwardProjectionDict(LSSForwardProjectionDict):
    """
    立体视觉版本的LSSForwardProjectionDict
    """

    def __init__(self, cv_downsample=4, **kwargs):
        super(LSSVStereoForwardProjectionDict, self).__init__(**kwargs)
        self.cv_frustum = self.create_frustum(kwargs['grid_config']['depth'],
                                              kwargs['input_size'],
                                              downsample=cv_downsample)