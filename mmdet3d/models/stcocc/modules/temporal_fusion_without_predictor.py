import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import BaseModule, force_fp32

from mmdet3d.models.builder import HEADS


@HEADS.register_module()
class SparseFusionWithoutPredictor(BaseModule):
    """Sparse Fusion module without predictor dependency for ablation study."""

    def __init__(
            self,
            top_k=None,
            history_num=8,
            single_bev_num_channels=None,
            foreground_idx=None,
            **kwargs
    ):
        super(SparseFusionWithoutPredictor, self).__init__()
        self.single_bev_num_channels = single_bev_num_channels
        self.history_bev = None
        self.history_last_bev = None
        self.history_forward_augs = None
        self.history_num = history_num
        self.history_seq_ids = None
        self.history_sweep_time = None
        self.history_cam_sweep_freq = 0.5       # seconds between each frame
        self.top_k = top_k                      # top_k sampling
        self.foreground_idx = foreground_idx    # Set the foreground index

        # 修改：移除occ_embedding相关层
        # 修改：调整fusion层的输入通道数，不再包含occ_embedims
        self.history_fusion_linear = nn.Sequential(
            nn.Linear(single_bev_num_channels * (history_num + 1), single_bev_num_channels),
            nn.Softplus(),
            nn.Linear(single_bev_num_channels, single_bev_num_channels),
        )
        # self.history_fusion_bg_linear = nn.Sequential(
        #     nn.Linear(single_bev_num_channels * (history_num//2 + 1), single_bev_num_channels),
        #     nn.Softplus(),
        #     nn.Linear(single_bev_num_channels, single_bev_num_channels),
        # )

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
        
        # 处理dx和bx可能是浮点数的情况
        if isinstance(dx, (int, float)):
            # 如果dx是单个数值，假设三个维度都使用相同的值
            feat2bev[0, 0] = dx
            feat2bev[1, 1] = dx
            feat2bev[2, 2] = dx
        else:
            # 否则假设dx是可索引的对象（列表、元组或张量）
            feat2bev[0, 0] = dx[0]
            feat2bev[1, 1] = dx[1]
            feat2bev[2, 2] = dx[2]
        
        if isinstance(bx, (int, float)):
            # 如果bx是单个数值，假设三个维度都使用相同的值
            bx_value = bx
            if isinstance(dx, (int, float)):
                dx_value = dx
            else:
                dx_value = dx[0]  # 随便选一个维度的dx值
            feat2bev[0, 3] = bx_value - dx_value / 2.
            feat2bev[1, 3] = bx_value - dx_value / 2.
            feat2bev[2, 3] = bx_value - dx_value / 2.
        else:
            # 否则假设bx是可索引的对象
            if isinstance(dx, (int, float)):
                dx_value = dx
                feat2bev[0, 3] = bx[0] - dx_value / 2.
                feat2bev[1, 3] = bx[1] - dx_value / 2.
                feat2bev[2, 3] = bx[2] - dx_value / 2.
            else:
                feat2bev[0, 3] = bx[0] - dx[0] / 2.
                feat2bev[1, 3] = bx[1] - dx[1] / 2.
                feat2bev[2, 3] = bx[2] - dx[2] / 2.
        
        feat2bev[3, 3] = 1
        feat2bev = feat2bev.view(1, 4, 4)
        return feat2bev

    @force_fp32()
    def forward(self, curr_bev, cam_params, history_fusion_params, dx, bx, history_last_bev=None):
        """Forward pass without predictor dependency.
        
        Args:
            curr_bev: [bs, c, z, h, w]
            cam_params: dict, contain bda_mat
            history_fusion_params: dict, contain sequence information
            dx: coordinate transformation parameters
            bx: coordinate transformation parameters
            history_last_bev: optional, previous history bev features
        """
        # curr_bev: [bs, c, z, h, w]
        # cam_params: dict, contain bda_mat

        # 0、check process voxel or bev features
        voxel_feat = True if len(curr_bev.shape) == 5 else False

        # 1、Get some history fusion information
        # Process test situation with fallback values
        # 添加默认值处理，防止KeyError
        seq_ids = history_fusion_params.get('sequence_group_idx', torch.zeros(curr_bev.shape[0], dtype=torch.long, device=curr_bev.device))
        if isinstance(seq_ids, list):
            seq_ids = seq_ids[0]
        
        start_of_sequence = history_fusion_params.get('start_of_sequence', torch.ones(curr_bev.shape[0], dtype=torch.bool, device=curr_bev.device))
        if isinstance(start_of_sequence, list):
            start_of_sequence = start_of_sequence[0]
        
        curr_to_prev_ego_rt = history_fusion_params.get('curr_to_prev_ego_rt', torch.eye(4, dtype=curr_bev.dtype, device=curr_bev.device).unsqueeze(0))
        if isinstance(curr_to_prev_ego_rt, list):
            curr_to_prev_ego_rt = curr_to_prev_ego_rt[0]
        forward_augs = cam_params[-1]  # bda

        # check seq_ids > 0
        assert (seq_ids >= 0).all()

        # 2、Deal with first batch
        if self.history_bev is None:
            self.history_bev = curr_bev.clone()
            self.history_forward_augs = forward_augs.clone()
            self.history_seq_ids = seq_ids.clone()
            self.history_bev = curr_bev.repeat(1, self.history_num, 1, 1, 1)
            self.history_sweep_time = curr_bev.new_zeros(curr_bev.shape[0], self.history_num)

        self.history_bev = self.history_bev.detach()
        assert self.history_bev.dtype == torch.float32

        # 3、 Deal with the new sequences
        # Replace all the new sequences' positions in history with the curr_bev information
        assert (self.history_seq_ids != seq_ids)[~start_of_sequence].sum() == 0, "{}, {}, {}".format(self.history_seq_ids, seq_ids, start_of_sequence)
        self.history_sweep_time += 1
        if start_of_sequence.sum() > 0:
            self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_num, 1, 1, 1)
            self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]
            self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
            self.history_sweep_time[start_of_sequence] = 0  # zero the new sequence timestep starts

        # 4、Deal with the history fusion
        tmp_bev = self.history_bev
        bs, mc, z, h, w = tmp_bev.shape
        n, c_, z, h, w = curr_bev.shape

        # obtain the history fusion information
        grid = self.generate_grid(curr_bev)
        feat2bev = self.generate_feat2bev(grid, dx, bx)

        rt_flow = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_ego_rt @ torch.inverse(forward_augs) @ feat2bev)
        grid = rt_flow.view(n, 1, 1, 1, 4, 4) @ grid

        normalize_factor = torch.tensor([w - 1.0, h - 1.0, z - 1.0], dtype=curr_bev.dtype, device=curr_bev.device)
        grid = grid[:, :, :, :, :3, 0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0   # grid order is x, y, z

        # sample the history bev
        tmp_bev = tmp_bev.reshape(bs, mc, z, h, w)
        sampled_history_bev = F.grid_sample(tmp_bev, grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4),  align_corners=True, mode='bilinear')

        # 修改：由于没有predictor，我们使用特征的范数或其他方式来选择重要区域
        # 或者如果top_k为None，则使用所有体素
        if self.top_k is None:
            # 如果没有指定top_k，直接进行简单的特征融合
            curr_bev = curr_bev.reshape(n, c_, -1)
            sampled_history_bev = sampled_history_bev.reshape(n, mc, -1)
            
            # 简单的特征融合
            # 对于前景，使用所有历史帧特征
            sampled_history = sampled_history_bev
            # sampled_current = curr_bev.unsqueeze(1).repeat(1, self.history_num, 1, 1).reshape(n, mc, -1)
            
            # 融合特征
            sampled_fusion = torch.cat([curr_bev,sampled_history], dim=1).permute(0, 2, 1)
            sampled_fusion = self.history_fusion_linear(sampled_fusion)
            sampled_fusion = sampled_fusion.permute(0, 2, 1)
            
            # 添加融合特征到当前bev
            curr_bev = curr_bev + sampled_fusion
            
            curr_bev = curr_bev.reshape(n, c_, z, h, w)
        else:
            # 如果指定了top_k，使用特征的L2范数作为重要性指标
            # 计算特征的重要性（使用特征范数）
            curr_bev_flat = curr_bev.reshape(bs, c_, -1)
            feature_importance = curr_bev_flat.norm(dim=1)
            total_number = feature_importance.shape[1]
            
            # 确保top_k不超过特征维度的大小
            safe_top_k = min(self.top_k, total_number)
            if safe_top_k < self.top_k:
                print(f"Warning: top_k {self.top_k} exceeds feature dimension {total_number}, using {safe_top_k} instead")
            
            # 选择最重要的top_k体素作为前景
            indices = torch.topk(feature_importance, safe_top_k, dim=1)[1]
            # 选择剩余体素作为背景
            bg_indices = torch.topk(-feature_importance, total_number - safe_top_k, dim=1)[1]
            
            sampled_history, sampled_current = [], []
            sampled_bg_history, sampled_bg_current = [], []
            
            for i in range(bs):
                sampled_history_feature = sampled_history_bev[i].reshape(mc, -1)[:, indices[i]]
                sampled_history_bg_feature = sampled_history_bev[i, :mc//2].reshape(mc//2, -1)[:, bg_indices[i]]
                sampled_current_feature = curr_bev[i].reshape(c_, -1)[:, indices[i]]
                sampled_current_bg_feature = curr_bev[i].reshape(c_, -1)[:, bg_indices[i]]

                sampled_history.append(sampled_history_feature)
                sampled_current.append(sampled_current_feature)
                sampled_bg_history.append(sampled_history_bg_feature)
                sampled_bg_current.append(sampled_current_bg_feature)

            sampled_history = torch.stack(sampled_history, dim=0)
            sampled_current = torch.stack(sampled_current, dim=0)
            sampled_bg_history = torch.stack(sampled_bg_history, dim=0)
            sampled_bg_current = torch.stack(sampled_bg_current, dim=0)

            # 修改：不再包含occ_embed
            sampled_fusion = torch.cat([sampled_history, sampled_current], dim=1).permute(0, 2, 1)
            sampled_bg_fusion = torch.cat([sampled_bg_history, sampled_bg_current], dim=1).permute(0, 2, 1)
            sampled_fusion = self.history_fusion_linear(sampled_fusion)
            sampled_bg_fusion = self.history_fusion_bg_linear(sampled_bg_fusion)
            sampled_fusion = sampled_fusion.permute(0, 2, 1)
            sampled_bg_fusion = sampled_bg_fusion.permute(0, 2, 1)

            # add the sampled fusion to the current bev
            curr_bev = curr_bev.reshape(n, c_, -1)
            for i in range(bs):
                curr_bev[i, :, indices[i]] += sampled_fusion[i]
                curr_bev[i, :, bg_indices[i]] += sampled_bg_fusion[i]

            curr_bev = curr_bev.reshape(n, c_, z, h, w)

        sampled_history_bev = sampled_history_bev.reshape(n, mc, z, h, w)

        feats_cat = torch.cat([curr_bev, sampled_history_bev], dim=1)  # B x (1 + T) * C x H x W or B x (1 + T) * C x Z x H x W
        feats_cat = feats_cat.reshape(feats_cat.shape[0], self.history_num + 1, self.single_bev_num_channels, *feats_cat.shape[2:])  # B x (1 + T) x C x H x W

        # feats_to_return: shape [bs, 1 + T, c, z, h, w]
        feats_cat = feats_cat.reshape(feats_cat.shape[0], -1, *feats_cat.shape[3:])
        feats_to_return = curr_bev

        # update history information
        self.history_bev = feats_cat[:, :-self.single_bev_num_channels, ...].detach().clone()
        self.history_last_bev = feats_to_return.detach().clone()
        self.history_forward_augs = forward_augs.clone()

        return feats_to_return.clone()
        
    def update_history(self, voxel_feat, occ_pred, img_metas):
        """Update history information (placeholder method).
        
        This method is required by the stcocc_without_predictor.py but doesn't do anything
        in this simplified implementation without predictor.
        """
        # 由于没有predictor，这个方法只是一个占位符
        # 在有predictor的版本中，这个方法可能会更新历史预测信息
        pass

    def prepare_history_params(self, img_metas):
        """Prepare history parameters for temporal fusion.
        
        Args:
            img_metas: List of image metas
            
        Returns:
            dict: History fusion parameters including sequence_group_idx, start_of_sequence, 
                  and curr_to_prev_ego_rt
        """
        # Extract sequence information from img_metas
        sequence_group_idx = []
        start_of_sequence = []
        curr_to_prev_ego_rt = []
        
        for img_meta in img_metas:
            # Handle sequence indices
            if 'sequence_group_idx' in img_meta:
                sequence_group_idx.append(img_meta['sequence_group_idx'])
            else:
                sequence_group_idx.append(0)  # Default to 0 if not present
                
            # Check if this is the start of a new sequence
            if 'start_of_sequence' in img_meta:
                start_of_sequence.append(img_meta['start_of_sequence'])
            else:
                start_of_sequence.append(False)
                
            # Get ego motion information
            if 'curr_to_prev_ego_rt' in img_meta:
                curr_to_prev_ego_rt.append(img_meta['curr_to_prev_ego_rt'])
            else:
                # Default to identity matrix if not present
                curr_to_prev_ego_rt.append(torch.eye(4, device='cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Convert to tensors
        sequence_group_idx = torch.tensor(sequence_group_idx, device='cuda' if torch.cuda.is_available() else 'cpu')
        start_of_sequence = torch.tensor(start_of_sequence, device='cuda' if torch.cuda.is_available() else 'cpu')
        curr_to_prev_ego_rt = torch.stack(curr_to_prev_ego_rt)
        
        return {
            'sequence_group_idx': sequence_group_idx,
            'start_of_sequence': start_of_sequence,
            'curr_to_prev_ego_rt': curr_to_prev_ego_rt
        }