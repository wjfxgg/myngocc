# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn as nn
from mmdet.models import BACKBONES, NECKS, HEADS, LOSSES
from mmdet.models import NECKS

from .bevdet_stereo_projection import BEVDetStereoForwardProjection
from .stereo_projection_normal import LSSVStereoForwardProjectionWithNormal
from mmdet3d.models import builder


@NECKS.register_module()
class BEVDetStereoForwardProjectionWithNormal_implict(BEVDetStereoForwardProjection):
    """
    扩展版的BEVDetStereoForwardProjection，支持normal_comp相关bins的预测
    使用LSSVStereoForwardProjectionWithNormal替代原始的视图变换器
    """
    
    def __init__(self,
                 loss_depth_weight=3.0,
                 loss_normal_weight=1.0,  # normal预测的损失权重
                 normal_w_bins=8,         # normal_comp_W_bins的类别数
                 normal_h_bins=8,         # normal_comp_H_bins的类别数
                 normal_depth_bins=4,     # normal_comp_depth_bins的类别数
                 **kwargs):
        """
        初始化扩展的前向投影模块
        
        Args:
            loss_depth_weight: 深度损失的权重
            loss_normal_weight: normal损失的权重
            normal_w_bins: normal_comp_W_bins的类别数
            normal_h_bins: normal_comp_H_bins的类别数
            normal_depth_bins: normal_comp_depth_bins的类别数
            **kwargs: 其他传递给BEVDetStereoForwardProjection的参数
        """
        super().__init__(**kwargs)

        # 保存normal相关配置
        self.loss_normal_weight = loss_normal_weight
        self.normal_w_bins = normal_w_bins
        self.normal_h_bins = normal_h_bins
        self.normal_depth_bins = normal_depth_bins
        
        # 初始化父类，但不创建原始的img_view_transformer
        # self.img_backbone = builder.build_backbone(kwargs['img_backbone'])
        # self.img_neck = builder.build_neck(kwargs['img_neck'])
        
        # 构建使用LSSVStereoForwardProjectionWithNormal的img_view_transformer
        # 首先准备视图变换器的配置
        view_transformer_cfg = kwargs['img_view_transformer']
        # 确保使用我们的扩展类
        # view_transformer_cfg['type'] = 'LSSVStereoForwardProjectionWithNormal'
        view_transformer_cfg['loss_depth_weight'] = loss_depth_weight
        view_transformer_cfg['loss_normal_weight'] = loss_normal_weight
        view_transformer_cfg['normal_w_bins'] = normal_w_bins
        view_transformer_cfg['normal_h_bins'] = normal_h_bins
        view_transformer_cfg['normal_depth_bins'] = normal_depth_bins
        
        self.img_view_transformer = builder.build_neck(view_transformer_cfg)
        
        # 构建BEV编码器
        # self.img_bev_encoder_backbone = builder.build_backbone(kwargs['img_bev_encoder_backbone'])
        # self.img_bev_encoder_neck = builder.build_neck(kwargs['img_bev_encoder_neck'])
    
    def forward(self, img, stereo_metas=None, **kwargs):
        """
        前向传播函数
        
        Args:
            img: 输入图像 [B, N, 3, H, W]
            stereo_metas: 立体视觉相关的元数据
            **kwargs: 其他参数，可能包含rots, trans, intrins等
            
        Returns:
            tuple: (bev_feat, depth, normal_preds)
                bev_feat: BEV特征
                depth: 深度预测
                normal_preds: 包含normal_comp_W_bins, normal_comp_H_bins, normal_comp_depth_bins预测的字典
        """
        # 提取图像特征
        x = self.img_backbone(img)
        x = self.img_neck(x)
        
        # 准备视图变换所需的输入
        mlp_input = x[0]
        # 从kwargs中提取必要的变换参数
        rots = kwargs.get('rots')
        trans = kwargs.get('trans')
        intrins = kwargs.get('intrins')
        post_rot = kwargs.get('post_rot')
        post_aug = kwargs.get('post_aug')
        bda = kwargs.get('bda')
        
        # 准备视图变换的输入
        view_input = [x, rots, trans, intrins, post_rot, post_aug, bda, mlp_input]
        
        # 执行视图变换
        # 使用扩展版本的视图变换器，它会返回normal_preds
        if self.img_view_transformer.return_context:
            bev_feat, depth, tran_feat, normal_preds = self.img_view_transformer(view_input, stereo_metas)
        else:
            bev_feat, depth, normal_preds = self.img_view_transformer(view_input, stereo_metas)
        
        # 编码BEV特征
        bev_feat = self.img_bev_encoder_backbone(bev_feat)
        bev_feat = self.img_bev_encoder_neck(bev_feat)
        
        return bev_feat, depth, normal_preds
    
    def get_depth_loss(self, depth_pred, gt_depth):
        """
        计算深度损失
        
        Args:
            depth_pred: 深度预测 [B*N, D, H, W]
            gt_depth: 真实深度 [B, N, H, W]
            
        Returns:
            torch.Tensor: 深度损失
        """
        # 调用img_view_transformer中的get_depth_loss方法
        return self.img_view_transformer.get_depth_loss(depth_pred, gt_depth)
    
    def get_normal_loss(self, normal_preds, normal_labels):
        """
        计算normal预测的损失
        
        Args:
            normal_preds: 预测的normal概率字典
            normal_labels: 真实的normal标签字典
            
        Returns:
            torch.Tensor: normal预测的损失
        """
        # 调用img_view_transformer中的get_normal_loss方法
        return self.img_view_transformer.get_normal_loss(normal_labels, normal_preds)
    
    def extract_img_feat(self,
                         img,
                         img_metas,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        """
        从图像中提取特征，支持normal预测的处理
        
        Args:
            img: 输入图像数据
            img_metas: 图像元数据
            pred_prev: 是否预测前一帧
            sequential: 是否顺序处理
            **kwargs: 其他参数，可能包含normal相关的标签
            
        Returns:
            tuple: (voxel_feats, depth, tran_feats, ms_feats, cam_params)
                其中tran_feats可能是normal预测字典
        """
        if sequential:
            # Todo
            assert False

        # 准备输入
        imgs, sensor2keyegos, ego2globals, intrins, post_augs, bda, curr2adjsensor = self.prepare_inputs(img, stereo=True)

        # 提取图像特征
        bev_feat_list = []
        normal_feat_list = []
        depth_key_frame = None
        tran_feat_key_frame = None
        ms_feat_key_frame = None
        feat_prev_iv = None
        for fid in range(self.num_frame-1, -1, -1):
            # 检查是否为关键帧
            key_frame = fid == 0
            # 获取当前帧的输入
            img_curr, sensor2keyego, ego2global, intrin, post_aug = imgs[fid], sensor2keyegos[fid], ego2globals[fid], intrins[fid], post_augs[fid]

            extra_ref_frame = fid == self.num_frame-self.extra_ref_frames

            if key_frame or self.with_prev:
                # 获取相机参数和MLP输入
                mlp_input = self.img_view_transformer.get_mlp_input(sensor2keyegos[0], ego2globals[0], intrin, post_aug, bda)
                inputs_curr = (img_curr,
                               sensor2keyego, ego2global, intrin, post_aug, bda,
                               mlp_input,
                               feat_prev_iv, curr2adjsensor[fid],
                               extra_ref_frame)
                # 存储关键帧特征
                if key_frame:
                    # 调用父类的prepare_bev_feat方法
                    bev_feat, depth, feat_curr_iv, tran_feat, img_feats_reshape = self.prepare_bev_feat(*inputs_curr)
                    depth_key_frame = depth
                    bev_feat = bev_feat[:, :80, ...]
                    # 对于关键帧，tran_feat可能是normal预测字典
                    # 这里我们直接保留，不通过adjust_channel_conv
                    tran_feat_key_frame = tran_feat
                    ms_feat_key_frame = img_feats_reshape
                else:
                    with torch.no_grad():
                        bev_feat, depth, feat_curr_iv, tran_feat, img_feats_reshape = self.prepare_bev_feat(*inputs_curr)
                        bev_feat = bev_feat[:, :80, ...] if bev_feat is not None else None
                        #TODO 这个上面这个法向量特征似乎可以直接传输进网络里面 这里后面得做这个实验
                if not extra_ref_frame:
                    bev_feat_list.append(bev_feat)
                    normal_feat_list.append(tran_feat['ego_dir_vector_3d'])
                feat_prev_iv = feat_curr_iv

        # 使用bev_encoder融合多帧BEV特征
        bev_feat = torch.cat(bev_feat_list, dim=1)
        tran_feat_key_frame['ego_dir_vector_3d']=torch.cat(normal_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)

        cam_params_key_frame = [sensor2keyegos[0], ego2globals[0], intrins[0], post_augs[0], bda]

        # 注意：对于normal预测，我们不调整通道，直接返回字典
        # 只有当tran_feat_key_frame不是字典时，才考虑使用adjust_channel_conv
        if self.adjust_channel_conv is not None:
            tran_feat_key_frame['tran_feat'] = self.adjust_channel_conv(tran_feat_key_frame['tran_feat'])

        return x, depth_key_frame, tran_feat_key_frame, ms_feat_key_frame, cam_params_key_frame
    
    def extract_feat(self, points, img, img_metas, **kwargs):
        """
        从图像和点云中提取特征，支持normal预测的处理
        
        Args:
            points: 点云数据（这里可能未使用）
            img: 输入图像数据
            img_metas: 图像元数据
            **kwargs: 其他参数，可能包含normal相关的标签
            
        Returns:
            tuple: (voxel_feats, depth, tran_feats, ms_feats, cam_params)
                其中tran_feats可能是normal预测字典
        """
        voxel_feats, depth, tran_feats, ms_feat_key_frame, cam_params = self.extract_img_feat(img, img_metas, **kwargs)
        return voxel_feats, depth, tran_feats, ms_feat_key_frame, cam_params