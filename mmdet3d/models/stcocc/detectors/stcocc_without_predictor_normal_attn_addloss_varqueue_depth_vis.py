import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import copy
import numpy as np
from collections import OrderedDict

from mmcv.runner import BaseModule, force_fp32
# from mmdet3d.models.detectors import CenterPoint
from mmdet3d.models.stcocc.detectors.stcocc_without_predictor_normal_attn_addloss import STCOccWithoutPredictor_normal_attn_addloss
from mmdet3d.models.builder import DETECTORS


@DETECTORS.register_module()
class STCOccWithoutPredictor_normal_attn_addloss_VarQueueDepthVis(STCOccWithoutPredictor_normal_attn_addloss):
    """
    Detector with variable queue temporal fusion and depth-based visibility.
    Inherits from STCOccWithoutPredictor_normal_attn_addloss and modifies
    the temporal fusion call to include pred_img_depth and img_metas parameters.
    """
    
    def forward_train(
        self,
        points=None,
        img_metas=None,
        img_inputs=None,
        voxel_semantics=None,
        voxel_semantics_1_2=None,
        voxel_semantics_1_4=None,
        voxel_semantics_1_8=None,
        **kwargs,
    ):
        """
        Override forward_train to pass pred_img_depth and img_metas to temporal_fusion.
        """
        time_dict = {}

        if points is not None:
            points = points[0]

        # 直接使用参数传递的img_inputs

        # ---------------------- history fusion -----------------------------        
        use_temporal = self.with_specific_component('temporal_fusion_list')
        if use_temporal:
            sequence_group_idx = torch.stack(
                [torch.tensor(img_meta['sequence_group_idx'], device=img_inputs[0].device) for img_meta in img_metas])
            start_of_sequence = torch.stack(
                [torch.tensor(img_meta['start_of_sequence'], device=img_inputs[0].device) for img_meta in img_metas])
            curr_to_prev_ego_rt = torch.stack(
                [torch.tensor(np.array(img_meta['curr_to_prev_ego_rt']), device=img_inputs[0].device) for img_meta in img_metas])
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
                cam_dir_vector = cam_dir_vector
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
                    history_last_bev=self.temporal_fusion_list[i+1].history_bev if i+1 < len(self.temporal_fusion_list)-1 else None,
                    last_occ_pred=last_occ_pred,
                    pred_img_depth=depth,
                    img_metas=img_metas
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

        time_dict['loss_calc'] = time.time() - start_loss_calc
        total_time = sum(time_dict.values())
        time_dict['total'] = total_time
        time_dict['fps'] = 1.0 / total_time if total_time > 0 else 0

        # 打印各部分时间
        if self.training and self.iter_count % 50 == 0:
            print(f"Iter {self.iter_count} Time Breakdown:")
            for key, value in time_dict.items():
                print(f"  {key}: {value:.4f}s")
            print(f"Total FPS: {time_dict['fps']:.2f}")

        self.iter_count += 1

        return losses