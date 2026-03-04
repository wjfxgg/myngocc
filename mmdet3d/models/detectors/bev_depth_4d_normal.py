# Copyright (c) Phigent Robotics. All rights reserved.
import torch
from mmdet3d.models import DETECTORS, build_backbone
from .bev_det import BEVDepth4D


@DETECTORS.register_module()
class BEVDepth4DNormal(BEVDepth4D):
    """
    扩展版的BEVDepth4D模型，支持normal_comp相关bins的预测和损失计算
    继承自BEVDepth4D，并添加了normal预测相关的功能
    """
    
    def __init__(self, loss_normal_weight=1.0, **kwargs):
        """
        初始化扩展的BEVDepth4D模型
        
        Args:
            loss_normal_weight: normal损失的权重
            **kwargs: 其他传递给BEVDepth4D的参数
        """
        # 初始化父类
        super(BEVDepth4DNormal, self).__init__(**kwargs)
        self.loss_normal_weight = loss_normal_weight
    
    def forward_train(self, points=None, img_metas=None, img=None, **kwargs):
        """
        训练时的前向传播函数，处理normal预测结果并计算损失
        
        Args:
            points: 点云数据
            img_metas: 图像元数据
            img: 输入图像
            **kwargs: 其他参数，可能包含rots, trans, intrins等
            
        Returns:
            dict: 包含loss和log_vars的字典
        """
        # 执行前向投影
        # 注意：我们使用的扩展版forward_projection会返回normal_preds
        rots = kwargs.get('rots')
        trans = kwargs.get('trans')
        intrins = kwargs.get('intrins')
        post_rot = kwargs.get('post_rot')
        post_aug = kwargs.get('post_aug')
        bda = kwargs.get('bda')
        
        # 准备前向投影的输入
        forward_projection_input = {
            'rots': rots,
            'trans': trans,
            'intrins': intrins,
            'post_rot': post_rot,
            'post_aug': post_aug,
            'bda': bda
        }
        
        # 执行前向投影，获取BEV特征、深度预测和normal预测
        bev_feat, depth, normal_preds = self.forward_projection(img, **forward_projection_input)
        
        # 执行Transformer和占用预测
        x = self.transformer(bev_feat)
        x = self.temporal_fusion(x)
        
        # 初始化损失字典
        losses = dict()
        
        # 计算深度损失
        if 'gt_depth' in kwargs:
            gt_depth = kwargs['gt_depth']
            depth_loss = self.forward_projection.get_depth_loss(depth, gt_depth)
            losses['loss_depth'] = depth_loss
        
        # 计算normal预测的损失
        normal_labels = {
            'normal_comp_W_bins': kwargs.get('normal_comp_W_bins'),
            'normal_comp_H_bins': kwargs.get('normal_comp_H_bins'),
            'normal_comp_depth_bins': kwargs.get('normal_comp_depth_bins')
        }
        
        # 过滤掉None值
        valid_normal_labels = {k: v for k, v in normal_labels.items() if v is not None}
        
        if valid_normal_labels and normal_preds:
            normal_loss = self.forward_projection.get_normal_loss(normal_preds, valid_normal_labels)
            losses['loss_normal'] = normal_loss
        
        # 计算占用预测的损失
        if 'gt_occ' in kwargs:
            gt_occ = kwargs['gt_occ']
            occ_pred = self.occupancy_head(x)
            occ_loss = self.occupancy_head.loss(occ_pred, gt_occ, img_metas)
            losses.update(occ_loss)
        
        # 从损失字典中提取所有损失值并计算总损失
        loss_sum = sum(loss for loss in losses.values())
        losses['loss'] = loss_sum
        
        return losses
    
    def simple_test(self, points, img_metas, img=None, **kwargs):
        """
        测试时的前向传播函数
        
        Args:
            points: 点云数据
            img_metas: 图像元数据
            img: 输入图像
            **kwargs: 其他参数
            
        Returns:
            list: 预测结果列表
        """
        # 测试时不需要normal预测结果，直接使用父类的实现
        # 但需要修改前向投影部分以处理返回格式的变化
        rots = kwargs.get('rots')
        trans = kwargs.get('trans')
        intrins = kwargs.get('intrins')
        post_rot = kwargs.get('post_rot')
        post_aug = kwargs.get('post_aug')
        bda = kwargs.get('bda')
        
        # 准备前向投影的输入
        forward_projection_input = {
            'rots': rots,
            'trans': trans,
            'intrins': intrins,
            'post_rot': post_rot,
            'post_aug': post_aug,
            'bda': bda
        }
        
        # 执行前向投影，获取BEV特征
        bev_feat, depth, normal_preds = self.forward_projection(img, **forward_projection_input)
        
        # 执行Transformer和占用预测
        x = self.transformer(bev_feat)
        x = self.temporal_fusion(x)
        
        # 使用占用头生成预测结果
        occ_pred = self.occupancy_head(x)
        
        # 处理预测结果
        results = self.occupancy_head.get_occ(occ_pred, img_metas)
        
        return results