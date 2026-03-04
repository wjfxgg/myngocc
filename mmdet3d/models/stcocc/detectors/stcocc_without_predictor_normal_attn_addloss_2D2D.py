import os
import copy
import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.nn as nn

from mmdet.models import DETECTORS

from mmdet3d.models.detectors.centerpoint import CenterPoint
from mmdet3d.models import builder

from mmdet3d.models.stcocc.losses.semkitti import geo_scal_loss, sem_scal_loss
from mmdet3d.models.stcocc.losses.lovasz_softmax import lovasz_softmax

@DETECTORS.register_module()
class STCOccWithoutPredictor_normal_attn_addloss_2D2D(CenterPoint):
    """STCOcc without predictor implementation for ablation study."""

    def __init__(self,
                 num_stage,
                 bev_h,
                 bev_w,
                 bev_z,
                 train_top_k=None,
                 val_top_k=None,
                 class_weights=None,
                 history_frame_num=None,
                 backward_num_layer=None,
                 empty_idx=17,
                 foreground_idx=[],
                 background_idx=[],
                 train_flow=False,
                 intermediate_pred_loss_weight=None,
                 class_weights_group=None,
                 forward_projection=None,
                 backward_projection=None,
                 temporal_fusion=None,
                 occupancy_head=None,
                 flow_head=None,
                 use_ms_feats=False,
                 save_results=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        self.num_stage = num_stage
        self.train_top_k = train_top_k if train_top_k is not None else [12500, 2500, 500]
        self.val_top_k = val_top_k if val_top_k is not None else [12500, 2500, 500]
        self.intermediate_pred_loss_weight = intermediate_pred_loss_weight
        self.class_weights_group = class_weights_group
        self.history_frame_num = history_frame_num
        self.foreground_idx = foreground_idx
        self.background_idx = background_idx
        self.train_flow = train_flow
        self.use_ms_feats = use_ms_feats
        self.save_results = save_results
        self.empty_idx = empty_idx
        self.scene_can_bus_info = dict()
        self.scene_loss = dict()
        # ---------------------- init loss ------------------------------
        self.class_weights = torch.tensor(np.array(class_weights), dtype=torch.float32, device='cuda')
        self.flow_loss = builder.build_loss(dict(type='L1Loss', loss_weight=1.0))
        self.focal_loss_dict = self._build_focal_loss(dict(type='CustomFocalLoss', bev_h=200, bev_w=200), num_stage)
        # ---------------------- build components ------------------------------
        # BEVDet-Series
        self.forward_projection = builder.build_neck(forward_projection)
        # BEVFormer-Series
        self._build_backward_projection(backward_projection, num_stage)
        # Temporal-Fsuion
        self._build_temporal_fusion(temporal_fusion, num_stage) if temporal_fusion else None
        # Simple Occupancy Head
        self.occupancy_head = builder.build_head(occupancy_head)
        # flow head
        self.flow_head = builder.build_head(flow_head) if flow_head else None

    def _build_focal_loss(self, focal_loss_config, num_stage):
        loss_dict = dict()
        focal_loss = builder.build_loss(focal_loss_config)
        loss_dict['num_stage_1_1'] = focal_loss
        for i in range(0, num_stage):
            focal_loss_config = copy.deepcopy(focal_loss_config)
            focal_loss_config['bev_h'] = int(focal_loss_config['bev_h'] / 2)
            focal_loss_config['bev_w'] = int(focal_loss_config['bev_w'] / 2)
            focal_loss = builder.build_loss(focal_loss_config)
            loss_dict['num_stage_1_{}'.format(2**(i+1))] = focal_loss
        return loss_dict

    def _build_backward_projection(self, backward_projection_config, num_stage):
        self.backward_projection_list = nn.ModuleList()
        backward_projection_config_dict = dict()
        
        # 使用默认值或配置的backward_num_layer
        if hasattr(self, 'backward_num_layer') and self.backward_num_layer is not None:
            backward_projection_config['transformer']['encoder']['num_layers'] = self.backward_num_layer[-1]
        else:
            # 使用默认值2
            backward_projection_config['transformer']['encoder']['num_layers'] = 2
            
        backward_projection_config_dict['num_stage_0'] = copy.deepcopy(backward_projection_config)

        for index in range(num_stage):
            # first stage:
            if index == num_stage - 1:
                backward_projection_config_dict['num_stage_{}'.format(index)]['transformer']['encoder']['first_stage'] = True

            backward_projection = builder.build_neck(backward_projection_config_dict['num_stage_{}'.format(index)])
            self.backward_projection_list.append(backward_projection)
            if index != num_stage-1:
                # different stage, adjust backward params, copy avoid in-place operation
                backward_projection_config_dict['num_stage_{}'.format(index+1)] = copy.deepcopy(backward_projection_config_dict['num_stage_{}'.format(index)])

                # num_layer adjust - 使用默认值2
                if hasattr(self, 'backward_num_layer') and self.backward_num_layer is not None and len(self.backward_num_layer) > num_stage-index-2:
                    backward_projection_config_dict['num_stage_{}'.format(index+1)]['transformer']['encoder']['num_layers'] = self.backward_num_layer[num_stage-index-2]
                else:
                    backward_projection_config_dict['num_stage_{}'.format(index+1)]['transformer']['encoder']['num_layers'] = 2

                # bev shape adjust or voxel shape adjust
                backward_projection_config_dict['num_stage_{}'.format(index + 1)]['bev_h'] = int(backward_projection_config_dict['num_stage_{}'.format(index + 1)]['bev_h'] / 2)
                backward_projection_config_dict['num_stage_{}'.format(index + 1)]['bev_w'] = int(backward_projection_config_dict['num_stage_{}'.format(index + 1)]['bev_w'] / 2)
                if 'bev_z' in backward_projection_config_dict['num_stage_{}'.format(index + 1)]:
                    backward_projection_config_dict['num_stage_{}'.format(index + 1)]['bev_z'] = int(backward_projection_config_dict['num_stage_{}'.format(index + 1)]['bev_z'] / 2)

                # grid config bev shape adjust
                backward_projection_config_dict['num_stage_{}'.format(index+1)]['transformer']['encoder']['grid_config']['x'][2] = \
                    backward_projection_config_dict['num_stage_{}'.format(index+1)]['transformer']['encoder']['grid_config']['x'][2] * 2
                backward_projection_config_dict['num_stage_{}'.format(index+1)]['transformer']['encoder']['grid_config']['y'][2] = \
                    backward_projection_config_dict['num_stage_{}'.format(index+1)]['transformer']['encoder']['grid_config']['y'][2] * 2
                backward_projection_config_dict['num_stage_{}'.format(index+1)]['transformer']['encoder']['grid_config']['z'][2] = \
                    backward_projection_config_dict['num_stage_{}'.format(index+1)]['transformer']['encoder']['grid_config']['z'][2] * 2

                # transformerlayers bev shape adjust
                backward_projection_config_dict['num_stage_{}'.format(index+1)]['transformer']['encoder']['transformerlayers']['attn_cfgs'][1][
                    'deformable_attention']['num_points'] = \
                    int(backward_projection_config_dict['num_stage_{}'.format(index+1)]['transformer']['encoder']['transformerlayers']['attn_cfgs'][1][
                        'deformable_attention']['num_points'] / 2)
                backward_projection_config_dict['num_stage_{}'.format(index+1)]['transformer']['encoder']['transformerlayers']['attn_cfgs'][1]['num_points'] = \
                    int(backward_projection_config_dict['num_stage_{}'.format(index+1)]['transformer']['encoder']['transformerlayers']['attn_cfgs'][1]['num_points'] / 2)
                backward_projection_config_dict['num_stage_{}'.format(index + 1)]['transformer']['encoder'][
                    'transformerlayers']['attn_cfgs'][0]['num_points'] = \
                    int(backward_projection_config_dict['num_stage_{}'.format(index + 1)]['transformer']['encoder'][
                            'transformerlayers']['attn_cfgs'][0]['num_points'] / 2)

                if 'num_Z_anchors' in backward_projection_config_dict['num_stage_{}'.format(index + 1)]['transformer']['encoder'][
                    'transformerlayers']['attn_cfgs'][1]['deformable_attention']:
                    backward_projection_config_dict['num_stage_{}'.format(index + 1)]['transformer']['encoder'][
                        'transformerlayers']['attn_cfgs'][1]['deformable_attention']['num_Z_anchors'] = \
                        int(backward_projection_config_dict['num_stage_{}'.format(index + 1)]['transformer']['encoder'][
                                'transformerlayers']['attn_cfgs'][1]['deformable_attention']['num_Z_anchors'] / 2)

                # positional_encoding bev shape adjust
                backward_projection_config_dict['num_stage_{}'.format(index+1)]['positional_encoding']['row_num_embed'] = \
                    int(backward_projection_config_dict['num_stage_{}'.format(index+1)]['positional_encoding']['row_num_embed'] / 2)
                backward_projection_config_dict['num_stage_{}'.format(index+1)]['positional_encoding']['col_num_embed'] = \
                    int(backward_projection_config_dict['num_stage_{}'.format(index+1)]['positional_encoding']['col_num_embed'] / 2)

    def _build_temporal_fusion(self, temporal_fusion_config, num_stage):
        self.temporal_fusion_list = nn.ModuleList()
        self.dx_list, self.bx_list, self.nx_list = [], [], []
        x_config = self.forward_projection.img_view_transformer.grid_config['x']
        y_config = self.forward_projection.img_view_transformer.grid_config['y']
        z_config = self.forward_projection.img_view_transformer.grid_config['z']
        for i in range(num_stage):
            temporal_fusion_config['history_num'] = self.history_frame_num[i]
            # 根据不同的stage调整参数
            if i > 0:
                temporal_fusion_config['bev_h'] = int(temporal_fusion_config['bev_h'] / 2)
                temporal_fusion_config['bev_w'] = int(temporal_fusion_config['bev_w'] / 2)
                temporal_fusion_config['bev_z'] = int(temporal_fusion_config['bev_z'] / 2)
            # 计算坐标变换参数
            dx = (x_config[1] - x_config[0]) / (temporal_fusion_config['bev_w'] - 1)
            bx = x_config[0]
            nx = temporal_fusion_config['bev_w']
            self.dx_list.append(dx)
            self.bx_list.append(bx)
            self.nx_list.append(nx)
            # 构建时间融合模块
            temporal_fusion = builder.build_neck(temporal_fusion_config)
            self.temporal_fusion_list.append(temporal_fusion)

    def create_multi_scale_features_fix(self,edg_feat_bev, scales=None):
        multi_scale_feats = [edg_feat_bev]
        multi_scale_feats.append(F.interpolate(
                edg_feat_bev.float(),  # 转换为浮点类型
                scale_factor=(0.5, 0.5, 0.5),  # Z/X/Y各维度缩小2倍
                mode='area'  # 3D最近邻插值
            ))
        multi_scale_feats.append(F.interpolate(
                edg_feat_bev.float(),  # 转换为浮点类型
                scale_factor=(0.25, 0.25, 0.25),  # Z/X/Y各维度缩小2倍
                mode='area'  # 3D最近邻插值
            ))
        return multi_scale_feats

    def get_voxel_loss(self,
                       pred_voxel_semantic,
                       target_voxel_semantic,
                       loss_weight,
                       focal_loss=None,
                       tag='c_0',
                       ):
        # change pred_voxel_semantic from [bs, w, h, z, c] -> [bs, c, w, h, z]  !!!
        pred_voxel_semantic = pred_voxel_semantic.permute(0, 4, 1, 2, 3)
        loss_dict = {}

        loss_dict['loss_voxel_ce_{}'.format(tag)] = loss_weight * focal_loss(
            pred_voxel_semantic,
            target_voxel_semantic,
            self.class_weights,
            ignore_index=255)
        loss_dict['loss_voxel_sem_scal_{}'.format(tag)] = loss_weight * sem_scal_loss(
            pred_voxel_semantic,
            target_voxel_semantic,
            ignore_index=255)
        loss_dict['loss_voxel_geo_scal_{}'.format(tag)] = loss_weight * geo_scal_loss(
            pred_voxel_semantic,
            target_voxel_semantic,
            ignore_index=255,
            empty_idx=self.empty_idx)
        loss_dict['loss_voxel_lovasz_{}'.format(tag)] = loss_weight * lovasz_softmax(
            torch.softmax(pred_voxel_semantic, dim=1),
            target_voxel_semantic,
            ignore=255,
        )

        return loss_dict


    def forward_train(self, points=None, img_metas=None, img_inputs=None, voxel_semantics=None, voxel_semantics_1_2=None, voxel_semantics_1_4=None, voxel_semantics_1_8=None, **kwargs):
        # 初始化时间记录字典
        time_dict = {}
        start_total = time.time()
        # 打印kwargs中的变量，查看图像法向量相关信息
        # print("kwargs keys:", kwargs.keys())
        # # 打印每个键的值类型和形状（如果是张量）
        # for key, value in kwargs.items():
        #     if isinstance(value, torch.Tensor):
        #         print(f"{key}: type={type(value)}, shape={value.shape}")
        #     else:
        #         print(f"{key}: type={type(value)}")
        #         # 如果是列表且元素是张量，打印第一个元素的形状
        #         if isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
        #             print(f"  First element shape: {value[0].shape}")
        # ---------------------- obtain feats from images -----------------------------
        
        # forward projection: generate voxel_feats and depth
        # 按照stcocc.py的实现，应该调用extract_feat方法
        
        img = img_inputs

        if self.with_specific_component('temporal_fusion_list'):
            use_temporal = True
            sequence_group_idx = torch.stack(
                [torch.tensor(img_meta['sequence_group_idx'], device=img[0].device) for img_meta in img_metas])
            start_of_sequence = torch.stack(
                [torch.tensor(img_meta['start_of_sequence'], device=img[0].device) for img_meta in img_metas])
            curr_to_prev_ego_rt = torch.stack(
                [torch.tensor(np.array(img_meta['curr_to_prev_ego_rt']), device=img[0].device) for img_meta in img_metas])
            history_fusion_params = {
                'sequence_group_idx': sequence_group_idx,
                'start_of_sequence': start_of_sequence,
                'curr_to_prev_ego_rt': curr_to_prev_ego_rt
            }
            # process can_bus info
            if 'can_bus' in img_metas[0]:
                for index, start in enumerate(start_of_sequence):
                    if start:
                        can_bus = copy.deepcopy(img_metas[index]['can_bus'])
                        temp_pose = copy.deepcopy(can_bus[:3])
                        temp_angle = copy.deepcopy(can_bus[-1])
                        can_bus[:3] = 0
                        can_bus[-1] = 0
                        self.scene_can_bus_info[sequence_group_idx[index].item()] = {
                            'prev_pose':temp_pose,
                            'prev_angle':temp_angle
                        }
                        img_metas[index]['can_bus'] = can_bus
                    else:
                        can_bus = copy.deepcopy(img_metas[index]['can_bus'])
                        temp_pose = copy.deepcopy(can_bus[:3])
                        temp_angle = copy.deepcopy(can_bus[-1])
                        can_bus[:3] = can_bus[:3] - self.scene_can_bus_info[sequence_group_idx[index].item()]['prev_pose']
                        can_bus[-1] = can_bus[-1] - self.scene_can_bus_info[sequence_group_idx[index].item()]['prev_angle']
                        self.scene_can_bus_info[sequence_group_idx[index].item()] = {
                            'prev_pose': temp_pose,
                            'prev_angle': temp_angle
                        }
                        img_metas[index]['can_bus'] = can_bus

        else:
            use_temporal = False
            history_fusion_params = None
        
        
        # 测量前向投影时间
        start_forward_proj = time.time()
        voxel_feats, depth, tran_feats, ms_feats, cam_params = self.forward_projection.extract_feat(points, img=img_inputs, img_metas=img_metas, **kwargs)
        time_dict['forward_projection'] = time.time() - start_forward_proj
        normal_comp_W_bins_pred,normal_comp_H_bins_pred,normal_comp_depth_bins_pred,ego_w_angle,ego_h_angle,ego_d_angle,ego_dir_vector,_,cam_dir_vector,ego_dir_vector_3d,tran_feats = tran_feats.values()
        
        # 预处理多尺度特征
        start_prep = time.time()
        ego_dir_vector_3d_xyz=self.create_multi_scale_features_fix(ego_dir_vector_3d)
        time_dict['prep_multi_scale'] = time.time() - start_prep
        # history_fusion_params = None
        # ---------------------- temporal fusion -----------------------------        
        # use_temporal = self.with_specific_component('temporal_fusion_list')
        # if use_temporal:
        #     history_fusion_params = {}
        #     for i in range(len(self.temporal_fusion_list)):
        #         history_fusion_params[i] = self.temporal_fusion_list[i].prepare_history_params(img_metas)
            # history_fusion_params = self.temporal_fusion_list[0].prepare_history_params(img_metas)
                
        # ---------------------- backward projection -----------------------------        
        last_voxel_feat = None
        last_occ_pred = None
        voxel_feats_index = len(voxel_feats) - 1
        intermediate_occ_pred_dict = {}
        intermediate_voxel_feat = []
        ego_dir_vector_3d_xyz=self.create_multi_scale_features_fix(ego_dir_vector_3d)

        # 测量反向投影总时间
        start_backward_proj_total = time.time()
        for i in range(len(self.backward_projection_list)-1, -1, -1):
            voxel_feat = voxel_feats[voxel_feats_index]

            if last_voxel_feat is not None:
                voxel_feat = last_voxel_feat + voxel_feat

            # 测量单个反向投影时间
            start_backward_proj = time.time()
            # voxel_feats shape: [bs, c, z, h, w]
            # 按照stcocc.py的实现，backward_projection返回两个值：bev_feat和occ_pred
            bev_feat, occ_pred = self.backward_projection_list[i](
                mlvl_feats=ms_feats if self.use_ms_feats else [tran_feats],
                img_metas=img_metas,
                voxel_feats=voxel_feat,  # mean in the z direction
                cam_params=cam_params,
                pred_img_depth=depth,
                last_occ_pred=last_occ_pred,
                prev_bev=self.temporal_fusion_list[i].history_last_bev if use_temporal else None,
                prev_bev_aug=self.temporal_fusion_list[i].history_forward_augs,
                history_fusion_params=history_fusion_params,
                ego_dir_vector_3d_xyz_i = ego_dir_vector_3d_xyz[i],
                cam_dir_vector = cam_dir_vector,
                ego_dir_vector = ego_dir_vector,
            )
            time_dict[f'backward_projection_stage_{i}'] = time.time() - start_backward_proj
            # 更新last_occ_pred
            last_occ_pred = occ_pred.clone().detach()
            # 将occ_pred添加到intermediate_occ_pred_dict中
            # intermediate_occ_pred_dict[i] = occ_pred
            intermediate_occ_pred_dict['pred_voxel_semantic_1_{}'.format(2 ** (i + 1))] = occ_pred
            # bev_feats shape: [bs, c, h, w], recover to occupancy without occ weight
            if bev_feat.dim() == 4:
                bs, c, z, h, w = voxel_feat.shape
                bev_feat = bev_feat.unsqueeze(2).repeat(1, 1, z, 1, 1)
                last_voxel_feat = voxel_feat + bev_feat
            else:
                last_voxel_feat = voxel_feat + bev_feat

            # Option: temporal fusion
            if self.with_specific_component('temporal_fusion_list'):
                # 测量时间融合时间
                start_temporal_fusion = time.time()
                # 使用last_occ_pred进行时间融合
                last_voxel_feat = self.temporal_fusion_list[i](
                    last_voxel_feat, cam_params, history_fusion_params, dx=self.dx_list[i], bx=self.bx_list[i],
                    history_last_bev=self.temporal_fusion_list[i+1].history_bev if i+1 < len(self.temporal_fusion_list)-1 else None
                )
                # 更新历史记录
                self.temporal_fusion_list[i].update_history(voxel_feat, last_occ_pred, img_metas)
                time_dict[f'temporal_fusion_stage_{i}'] = time.time() - start_temporal_fusion

            # output stage don't need to upsample
        
            if i != 0:
                last_voxel_feat = F.interpolate(last_voxel_feat, scale_factor=2, align_corners=False, mode='trilinear')

            voxel_feats_index = voxel_feats_index - 1
            intermediate_voxel_feat.append(last_voxel_feat)
            # 确保intermediate_occ_pred_dict是一个列表格式
        intermediate_occ_pred_list = [intermediate_occ_pred_dict[k] for k in sorted(intermediate_occ_pred_dict.keys())]

        time_dict['backward_projection_total'] = time.time() - start_backward_proj_total



        # 添加打印信息来帮助定位问题
        # print("\n=== Debug Info ===")
        # print(f"last_voxel_feat type: {type(last_voxel_feat)}, shape: {last_voxel_feat.shape if hasattr(last_voxel_feat, 'shape') else 'No shape'}")
        # print(f"last_occ_pred type: {type(last_occ_pred)}, shape: {last_occ_pred.shape if hasattr(last_occ_pred, 'shape') else 'No shape'}")
        # print(f"depth type: {type(depth)}, shape: {depth.shape if hasattr(depth, 'shape') else 'No shape'}")
        # print(f"tran_feats type: {type(tran_feats)}, shape: {tran_feats.shape if hasattr(tran_feats, 'shape') else 'No shape'}")
        # print(f"intermediate_occ_pred_dict type: {type(intermediate_occ_pred_dict)}")
        # if isinstance(intermediate_occ_pred_dict, dict):
        #     for key, pred in intermediate_occ_pred_dict.items():
        #         print(f"  intermediate_occ_pred_dict[{key}] type: {type(pred)}, shape: {pred.shape if hasattr(pred, 'shape') else 'No shape'}")
        # print("===============\n")

        
        





        return_dict = dict(
            voxel_feats=last_voxel_feat,
            last_occ_pred=last_occ_pred,
            depth=depth,
            tran_feats=tran_feats,
            cam_params=cam_params,
            intermediate_occ_pred_dict=intermediate_occ_pred_dict,
            history_fusion_params=history_fusion_params,
        )



        voxel_feats = return_dict['voxel_feats']    # shape: [bs, c, z, h, w]
        last_occ_pred = return_dict['last_occ_pred']
        depth = return_dict['depth']
        intermediate_occ_pred_dict = return_dict['intermediate_occ_pred_dict']
        history_fusion_params = return_dict['history_fusion_params']

        # ---------------------- forward ------------------------------
        # 测量占用率预测头时间
        start_occupancy_head = time.time()
        pred_voxel_semantic, pred_voxel_feats = self.occupancy_head(voxel_feats, last_occ_pred=last_occ_pred)
        time_dict['occupancy_head'] = time.time() - start_occupancy_head
        intermediate_occ_pred_dict['pred_voxel_semantic_1_1'] = pred_voxel_semantic

        if self.with_specific_component('flow_head'):
            pred_voxel_flows, foreground_masks = self.flow_head(voxel_feats, pred_voxel_semantic)

        # ---------------------- calc loss ------------------------------
        # 测量损失计算时间
        start_loss_calc = time.time()
        losses = dict()

        gt_semantic_voxel_dict = dict()
        gt_semantic_voxel_dict['gt_semantic_voxel_1_1'] = voxel_semantics
        # kwargs['voxel_semantics_1_1'] = voxel_semantics
        kwargs['voxel_semantics_1_2'] = voxel_semantics_1_2
        kwargs['voxel_semantics_1_4'] = voxel_semantics_1_4
        kwargs['voxel_semantics_1_8'] = voxel_semantics_1_8

        num_stage = self.num_stage
        for index in range(num_stage):
            gt_semantic_voxel_dict['gt_semantic_voxel_1_{}'.format(2**(index+1))] = kwargs['voxel_semantics_1_{}'.format(2**(index+1))]

        # calc forward-projection depth-loss
        start_depth_loss = time.time()
        loss_depth = self.forward_projection.img_view_transformer.get_depth_loss(kwargs['gt_depth'], depth)
        time_dict['loss_depth_calculation'] = time.time() - start_depth_loss
        losses['loss_depth'] = loss_depth
        
        start_normal_W_loss = time.time()
        loss_normal_W = self.forward_projection.img_view_transformer.get_normal_W_loss(kwargs['normal_comp_W_bins'],normal_comp_W_bins_pred)
        time_dict['loss_normal_W_calculation'] = time.time() - start_normal_W_loss
        losses['loss_normal_W'] = loss_normal_W
        
        start_normal_H_loss = time.time()
        loss_normal_H = self.forward_projection.img_view_transformer.get_normal_H_loss(kwargs['normal_comp_H_bins'],normal_comp_H_bins_pred)
        time_dict['loss_normal_H_calculation'] = time.time() - start_normal_H_loss
        losses['loss_normal_H'] = loss_normal_H
        
        start_normal_D_loss = time.time()
        loss_normal_D = self.forward_projection.img_view_transformer.get_normal_D_loss(kwargs['normal_comp_depth_bins'],normal_comp_depth_bins_pred)
        time_dict['loss_normal_D_calculation'] = time.time() - start_normal_D_loss
        losses['loss_normal_D'] = loss_normal_D



        # calc voxel loss
        for index in range(num_stage+1):
            loss_occ = self.get_voxel_loss(
                intermediate_occ_pred_dict['pred_voxel_semantic_1_{}'.format(2**index)],
                gt_semantic_voxel_dict['gt_semantic_voxel_1_{}'.format(2 **index)],
                self.intermediate_pred_loss_weight[index],
                focal_loss=self.focal_loss_dict['num_stage_1_{}'.format(2 **index)],
                tag='c_1_{}'.format(2**index),
            )
            losses.update(loss_occ)

        if self.with_specific_component('flow_head'):
            losses.update(
                self.get_flow_loss(
                    pred_voxel_flows,
                    kwargs['voxel_flows'],
                    gt_semantic_voxel_dict['gt_semantic_voxel_1_1'],
                    loss_weight=0.8
                )
            )
        
        time_dict['loss_calculation'] = time.time() - start_loss_calc
        time_dict['total_forward_train'] = time.time() - start_total
        
        # 打印执行时间
        # print("\n--- 执行时间统计 ---")
        # for key, value in time_dict.items():
        #     print(f"{key}: {value:.4f}秒")
        # print("-------------------\n")

        return losses



        


        # 直接返回空的损失字典，因为没有预测器
        # 这是一个临时解决方案，实际训练时可能需要实现适当的损失计算
        losses = dict()
        # 添加一个虚拟损失来避免错误
        losses['loss_dummy'] = torch.tensor(0.0, requires_grad=True, device=last_voxel_feat.device)
        return losses

    def with_specific_component(self, component_name):
        """Whether the detector has a specific component.

        Args:
            component_name (str): The name of the component.

        Returns:
            bool: Whether the detector has the component.
        """
        return hasattr(self, component_name) and \
               getattr(self, component_name) is not None

    def simple_test(self, points, img_metas, img=None, img_inputs=None, **kwargs):
        # ---------------------- obtain feats from images -----------------------------
        
        # forward projection: generate voxel_feats and depth
        # 按照stcocc.py的实现，应该调用extract_feat方法
        img = img_inputs[0]

        if self.with_specific_component('temporal_fusion_list'):
            use_temporal = True
            sequence_group_idx = torch.stack(
                [torch.tensor(img_meta['sequence_group_idx'], device=img[0].device) for img_meta in img_metas])
            start_of_sequence = torch.stack(
                [torch.tensor(img_meta['start_of_sequence'], device=img[0].device) for img_meta in img_metas])
            curr_to_prev_ego_rt = torch.stack(
                [torch.tensor(np.array(img_meta['curr_to_prev_ego_rt']), device=img[0].device) for img_meta in img_metas])
            history_fusion_params = {
                'sequence_group_idx': sequence_group_idx,
                'start_of_sequence': start_of_sequence,
                'curr_to_prev_ego_rt': curr_to_prev_ego_rt
            }
            # print(f"img_metas[0]: {img_metas[0]}")
            # process can_bus info
            if 'can_bus' in img_metas[0]:
                # print("right！"*5)
                for index, start in enumerate(start_of_sequence):
                    if start:
                        can_bus = copy.deepcopy(img_metas[index]['can_bus'])
                        temp_pose = copy.deepcopy(can_bus[:3])
                        temp_angle = copy.deepcopy(can_bus[-1])
                        can_bus[:3] = 0
                        can_bus[-1] = 0
                        self.scene_can_bus_info[sequence_group_idx[index].item()] = {
                            'prev_pose':temp_pose,
                            'prev_angle':temp_angle
                        }
                        img_metas[index]['can_bus'] = can_bus
                    else:
                        can_bus = copy.deepcopy(img_metas[index]['can_bus'])
                        temp_pose = copy.deepcopy(can_bus[:3])
                        temp_angle = copy.deepcopy(can_bus[-1])
                        can_bus[:3] = can_bus[:3] - self.scene_can_bus_info[sequence_group_idx[index].item()]['prev_pose']
                        can_bus[-1] = can_bus[-1] - self.scene_can_bus_info[sequence_group_idx[index].item()]['prev_angle']
                        self.scene_can_bus_info[sequence_group_idx[index].item()] = {
                            'prev_pose': temp_pose,
                            'prev_angle': temp_angle
                        }
                        img_metas[index]['can_bus'] = can_bus

        else:
            use_temporal = False
            history_fusion_params = None
        
        
        voxel_feats, depth, tran_feats, ms_feats, cam_params = self.forward_projection.extract_feat(points, img=img_inputs[0], img_metas=img_metas, **kwargs)
        normal_comp_W_bins_pred,normal_comp_H_bins_pred,normal_comp_depth_bins_pred,ego_w_angle,ego_h_angle,ego_d_angle,ego_dir_vector,_,cam_dir_vector,ego_dir_vector_3d,tran_feats = tran_feats.values()
        
        # history_fusion_params = None
        # ---------------------- temporal fusion -----------------------------        
        # use_temporal = self.with_specific_component('temporal_fusion_list')
        # if use_temporal:
        #     history_fusion_params = {}
        #     for i in range(len(self.temporal_fusion_list)):
        #         history_fusion_params[i] = self.temporal_fusion_list[i].prepare_history_params(img_metas)
            # history_fusion_params = self.temporal_fusion_list[0].prepare_history_params(img_metas)
                
        # ---------------------- backward projection -----------------------------        
        last_voxel_feat = None
        last_occ_pred = None
        voxel_feats_index = len(voxel_feats) - 1
        intermediate_occ_pred_dict = {}
        intermediate_voxel_feat = []
        ego_dir_vector_3d_xyz=self.create_multi_scale_features_fix(ego_dir_vector_3d)

        for i in range(len(self.backward_projection_list)-1, -1, -1):
            voxel_feat = voxel_feats[voxel_feats_index]

            if last_voxel_feat is not None:
                voxel_feat = last_voxel_feat + voxel_feat

            # voxel_feats shape: [bs, c, z, h, w]
            # 按照stcocc.py的实现，backward_projection返回两个值：bev_feat和occ_pred
            bev_feat, occ_pred = self.backward_projection_list[i](
                mlvl_feats=ms_feats if self.use_ms_feats else [tran_feats],
                img_metas=img_metas,
                voxel_feats=voxel_feat,  # mean in the z direction
                cam_params=cam_params,
                pred_img_depth=depth,
                last_occ_pred=last_occ_pred,
                prev_bev=self.temporal_fusion_list[i].history_last_bev if use_temporal else None,
                prev_bev_aug=self.temporal_fusion_list[i].history_forward_augs,
                history_fusion_params=history_fusion_params,
                ego_dir_vector_3d_xyz_i = ego_dir_vector_3d_xyz[i],
                cam_dir_vector = cam_dir_vector

            )
            # 更新last_occ_pred
            last_occ_pred = occ_pred.clone().detach()
            # 将occ_pred添加到intermediate_occ_pred_dict中
            # intermediate_occ_pred_dict[i] = occ_pred
            intermediate_occ_pred_dict['pred_voxel_semantic_1_{}'.format(2 ** (i + 1))] = occ_pred
            # bev_feats shape: [bs, c, h, w], recover to occupancy without occ weight
            if bev_feat.dim() == 4:
                bs, c, z, h, w = voxel_feat.shape
                bev_feat = bev_feat.unsqueeze(2).repeat(1, 1, z, 1, 1)
                last_voxel_feat = voxel_feat + bev_feat
            else:
                last_voxel_feat = voxel_feat + bev_feat

            # Option: temporal fusion
            if self.with_specific_component('temporal_fusion_list'):
                # 使用last_occ_pred进行时间融合
                last_voxel_feat = self.temporal_fusion_list[i](
                    last_voxel_feat, cam_params, history_fusion_params, dx=self.dx_list[i], bx=self.bx_list[i],
                    history_last_bev=self.temporal_fusion_list[i+1].history_bev if i+1 < len(self.temporal_fusion_list)-1 else None
                )
                # 更新历史记录
                self.temporal_fusion_list[i].update_history(voxel_feat, last_occ_pred, img_metas)

            # output stage don't need to upsample
            if i != 0:
                last_voxel_feat = F.interpolate(last_voxel_feat, scale_factor=2, align_corners=False, mode='trilinear')

            voxel_feats_index = voxel_feats_index - 1
            intermediate_voxel_feat.append(last_voxel_feat)
            # 确保intermediate_occ_pred_dict是一个列表格式
        intermediate_occ_pred_list = [intermediate_occ_pred_dict[k] for k in sorted(intermediate_occ_pred_dict.keys())]




        # 添加打印信息来帮助定位问题
        # print("\n=== Debug Info ===")
        # print(f"last_voxel_feat type: {type(last_voxel_feat)}, shape: {last_voxel_feat.shape if hasattr(last_voxel_feat, 'shape') else 'No shape'}")
        # print(f"last_occ_pred type: {type(last_occ_pred)}, shape: {last_occ_pred.shape if hasattr(last_occ_pred, 'shape') else 'No shape'}")
        # print(f"depth type: {type(depth)}, shape: {depth.shape if hasattr(depth, 'shape') else 'No shape'}")
        # print(f"tran_feats type: {type(tran_feats)}, shape: {tran_feats.shape if hasattr(tran_feats, 'shape') else 'No shape'}")
        # print(f"intermediate_occ_pred_dict type: {type(intermediate_occ_pred_dict)}")
        # if isinstance(intermediate_occ_pred_dict, dict):
        #     for key, pred in intermediate_occ_pred_dict.items():
        #         print(f"  intermediate_occ_pred_dict[{key}] type: {type(pred)}, shape: {pred.shape if hasattr(pred, 'shape') else 'No shape'}")
        # print("===============\n")

        
        





        return_dict = dict(
            voxel_feats=last_voxel_feat,
            last_occ_pred=last_occ_pred,
            depth=depth,
            tran_feats=tran_feats,
            cam_params=cam_params,
            intermediate_occ_pred_dict=intermediate_occ_pred_dict,
            history_fusion_params=history_fusion_params,
        )



        voxel_feat = return_dict['voxel_feats']
        last_occ_pred = return_dict['last_occ_pred']

        # ---------------------- forward ------------------------------
        pred_voxel_semantic, pred_voxel_feats = self.occupancy_head(voxel_feat, last_occ_pred=last_occ_pred)
        pred_voxel_semantic_cls = pred_voxel_semantic.softmax(-1).argmax(-1)
        if self.with_specific_component('flow_head'):
            pred_voxel_flows, foreground_mask = self.flow_head(voxel_feat, pred_voxel_semantic)
            return_pred_voxel_flows = torch.zeros_like(pred_voxel_flows)
            return_pred_voxel_flows[foreground_mask != 0] = pred_voxel_flows[foreground_mask != 0]
        else:
            return_pred_voxel_flows = torch.zeros(size=(pred_voxel_semantic_cls.shape + (2,)), device=pred_voxel_semantic_cls.device)

        return_dict = dict()
        return_dict['occ_results'] = pred_voxel_semantic_cls.cpu().numpy().astype(np.uint8)
        return_dict['flow_results'] = return_pred_voxel_flows.cpu().numpy().astype(np.float16)
        return_dict['index'] = [img_meta['index'] for img_meta in img_metas]
        if self.save_results:
            sample_idx = [img_meta['sample_idx'] for img_meta in img_metas]
            scene_name = [img_meta['scene_name'] for img_meta in img_metas]
            # check save_dir
            for name in scene_name:
                if not os.path.exists('results/{}'.format(name)):
                    os.makedirs('results/{}'.format(name))
            for i, idx in enumerate(sample_idx):
                np.savez('results/{}/{}.npz'.format(scene_name[i], idx),semantics=return_dict['occ_results'][i], flow=return_dict['flow_results'][i])
        return [return_dict]