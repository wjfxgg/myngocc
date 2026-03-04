import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import BaseModule, force_fp32
from mmdet.models import HEADS

from .gated_temporal_fusion import GatedTemporalFusion6_cat
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32


@HEADS.register_module()
class ObservableQueueFusionDepthVis_fix(GatedTemporalFusion6_cat):
    """Variable-length temporal queues with observability from depth (GatedTemporalFusion6_cat style).

    - Use depth-based visibility computation (same pipeline as GatedTemporalFusion6_cat).
    - Rank voxels by visibility, split into 4 groups, fuse with windows (num_T/8, num_T/4, num_T/2, num_T).
    - Aggregate history by mean per group and add back to current BEV features.
    """

    def __init__(
        self,
        single_bev_num_channels=96,
        num_T=16,
        queue_windows=None,
        group_ratios=None,
        **kwargs,
    ):
        # Use num_T as history_num, override any inherited history_num from config
        kwargs['history_num'] = num_T
        super().__init__(single_bev_num_channels=single_bev_num_channels, **kwargs)
        self.occ_embedding = None
        self.history_fusion_linear = None
        self.history_fusion_bg_linear = None
        if queue_windows is None:
            queue_windows = (
                max(num_T // 8, 1),
                max(num_T // 4, 1),
                max(num_T // 2, 1),
                num_T,
            )
        self.queue_windows = queue_windows
        self.num_T = num_T
        
        # 组比例参数，用于控制各组体素数量的分配比例
        if group_ratios is None:
            group_ratios = (1, 1, 1, 1)  # 默认等比例分配
        self.group_ratios = group_ratios
        # override buffers for queue-based fusion
        self.history_bev = None
        self.history_forward_augs = None
        self.history_seq_ids = None
        self.history_sweep_time = None
        self.num_cams = 6
        
        # 创建多个MLP层用于多级沙漏融合
        num_groups = len(queue_windows)
        
        # 原有的前景融合MLP层
        self.history_fusion_linears = nn.ModuleList()
        
        # 为每个分组创建一个独立的MLP层
        for i in range(num_groups):
            # 每个分组的历史窗口大小
            win_size = queue_windows[i]
            # 输入通道数：当前特征 + 历史特征（根据窗口大小）
            input_channels = single_bev_num_channels + (win_size * single_bev_num_channels)
            
            # 创建与原有MLP结构相同的融合层
            fusion_linear = nn.Sequential(
                nn.Linear(input_channels, single_bev_num_channels),
                nn.Softplus(beta=1, threshold=20),
                nn.Linear(single_bev_num_channels, single_bev_num_channels),
            )
            self.history_fusion_linears.append(fusion_linear)

    def _compute_visibility_from_depth(self, curr_bev, cam_params, dx, bx, pred_img_depth, img_metas):
        """Compute visibility map V_curr using depth deformable attention (mirrors GatedTemporalFusion6_cat)."""
        bs, _, z, h, w = curr_bev.shape
        device = curr_bev.device

        ref_3d = self.get_reference_points(h, w, z, z, dim='3d', bs=bs, device=device, dtype=curr_bev.dtype)
        slots = torch.zeros(list([ref_3d.shape[0], ref_3d.shape[2], ref_3d.shape[1]])).to(ref_3d)

        reference_points_cam, reference_points_depth, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, img_metas=img_metas, cam_params=cam_params
        )

        spatial_shapes = torch.tensor([[pred_img_depth.shape[-2], pred_img_depth.shape[-1]]], device=device)
        pred_img_depth = pred_img_depth.flatten(2).permute(0, 2, 1)  # [bs*cam, hw, C]

        # pad indices per image
        max_len = 0
        indexes = [[] for _ in range(bs)]
        for j in range(bs):
            for i, mask_per_img in enumerate(bev_mask):
                index_query_per_img = mask_per_img[j].sum(-1).nonzero().squeeze(-1)
                if len(index_query_per_img) == 0:
                    index_query_per_img = bev_mask[i][j].sum(-1).nonzero().squeeze(-1)[0:1]
                indexes[j].append(index_query_per_img)
                max_len = max(max_len, len(index_query_per_img))

        reference_points_cam_rebatch = reference_points_cam.new_zeros([bs, self.num_cams, max_len, z, 2])
        reference_points_depth_rebatch = reference_points_depth.new_zeros([bs, self.num_cams, max_len, z, 1])
        for j in range(bs):
            for i, (reference_points_per_img, reference_points_depth_per_img) in enumerate(
                zip(reference_points_cam, reference_points_depth)
            ):
                index_query_per_img = indexes[j][i]
                reference_points_cam_rebatch[j, i, : len(index_query_per_img)] = reference_points_per_img[
                    j, index_query_per_img
                ]
                reference_points_depth_rebatch[j, i, : len(index_query_per_img)] = reference_points_depth_per_img[
                    j, index_query_per_img
                ]

        depth_reference_points = reference_points_cam_rebatch.reshape(bs * self.num_cams, max_len * z, 1, 1, 1, 2).contiguous()
        depth_attention_weights = torch.ones_like(depth_reference_points[..., 0]).contiguous()
        pred_img_depth = pred_img_depth.unsqueeze(2).contiguous()
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        bev_query_depth_rebatch = (reference_points_depth_rebatch - self.dbound[0]) / self.dbound[2]
        bev_query_depth_rebatch = torch.clip(torch.floor(bev_query_depth_rebatch), 0, 88 - 1).to(torch.long)
        bev_query_depth_rebatch = F.one_hot(bev_query_depth_rebatch.squeeze(-1), num_classes=88)

        depth_output = MultiScaleDeformableAttnFunction_fp32.apply(
            pred_img_depth, spatial_shapes, level_start_index, depth_reference_points, depth_attention_weights, self.im2col_step
        )
        depth_output = depth_output.reshape(bs, self.num_cams, max_len, z, -1)
        depth_output = depth_output + torch.cat(
            [(torch.zeros_like(depth_output[..., :1]) + 1e-9), torch.zeros_like(depth_output[..., 1:])], dim=-1
        )
        depth_output = depth_output / depth_output.sum(-1)[..., None]
        depth_output = (1 - depth_output.cumsum(dim=-1))
        depth_output = torch.cat([torch.ones_like(depth_output[..., 0:1]), depth_output[..., 0:-1]], dim=-1)
        depth_output = (bev_query_depth_rebatch * depth_output).sum(-1)

        for j in range(bs):
            for i in range(self.num_cams):
                index_query_per_img = indexes[j][i]
                slots[j, index_query_per_img] = torch.max(slots[j, index_query_per_img], depth_output[j, i, : len(index_query_per_img)])

        V_curr = slots.view(bs, 1, h, w, z).permute(0, 1, 4, 2, 3).contiguous()  # [bs,1,z,h,w]
        return V_curr

    def _split_indices(self, V_curr, V_prev):
        """
        实现多级沙漏算法的索引分组逻辑
        
        Args:
            V_curr: 当前时间的可观测性 [bs, 1, z, h, w]
            V_prev: 历史时间的可观测性 [bs, history_num, z, h, w]
                   按时间由近到远排列: t-1, t-2, t-3...
        
        Returns:
            groups: 分组后的索引列表
        """
        b, _, z, h, w = V_curr.shape
        num_voxels = z * h * w
        
        # 将所有可观测性展平
        V_curr_flat = V_curr.reshape(b, num_voxels)
        V_prev_flat = V_prev.reshape(b, self.num_T, num_voxels)
        
        # 计算每个体素在所有时间步的最大可观测性评分
        all_visibility = torch.cat([V_curr_flat.unsqueeze(1), V_prev_flat], dim=1)  # [bs, num_T+1, num_voxels]
        max_scores, _ = all_visibility.max(dim=1)  # [bs, num_voxels]
        
        # 对体素按最大可观测性评分降序排序
        sorted_indices = torch.argsort(max_scores, dim=1, descending=True)
        
        # 计算分组数量
        num_groups = len(self.queue_windows)
        groups = []
        remaining_mask = torch.ones(b, num_voxels, dtype=torch.bool, device=V_curr.device)
        
        # 计算总比例数
        total_ratio = sum(self.group_ratios)
        
        # 计算各组体素数量，确保总和等于所有体素数
        group_sizes = []
        remaining_total = num_voxels
        
        # 先计算前num_groups-1组的大小
        for i in range(num_groups - 1):
            size = int(num_voxels * self.group_ratios[i] / total_ratio)
            group_sizes.append(size)
            remaining_total -= size
        
        # 最后一组使用剩余的所有体素，确保总和等于num_voxels
        group_sizes.append(remaining_total)
        
        for group_idx in range(num_groups):
            # 根据当前组索引确定使用的历史窗口大小
            win_size = self.queue_windows[group_idx] if group_idx < len(self.queue_windows) else self.queue_windows[-1]
            win_size = min(win_size, self.num_T)
            
            # 获取当前组的大小
            group_size = group_sizes[group_idx]
            
            # 计算每个剩余体素在指定时间窗口内的最大可观测性
            remaining_mask_expanded = remaining_mask.unsqueeze(1).expand(-1, self.num_T+1, -1)
            window_visibility = all_visibility[:, :win_size+1, :]  # 包含当前时间t和前win_size个历史时间
            window_visibility_masked = window_visibility * remaining_mask_expanded[:, :win_size+1, :].float()
            window_max_scores, _ = window_visibility_masked.max(dim=1)  # [bs, num_voxels]
            
            # 在剩余体素中找到评分最高的group_size个体素
            window_max_scores[~remaining_mask] = -1e9  # 将非剩余体素的评分设为最小值
            _, top_indices = torch.topk(window_max_scores, k=group_size, dim=1, largest=True)
            group_indices = top_indices
                
            # 将选中的体素从剩余列表中移除
            for batch_idx in range(b):
                selected_voxels = group_indices[batch_idx]
                remaining_mask[batch_idx, selected_voxels] = False
            
            groups.append(group_indices)
        
        return groups

    def _init_history(self, curr_bev, forward_augs, seq_ids):
        b, c, z, h, w = curr_bev.shape
        self.history_bev = curr_bev.clone()

        # self.history_bev = curr_bev.new_zeros(b, self.num_T, c, z, h, w)
        # self.history_bev[:, 0] = curr_bev
        self.history_bev = curr_bev.repeat(1, self.history_num, 1, 1, 1)

        self.history_forward_augs = forward_augs.clone()
        self.history_seq_ids = seq_ids.clone()
        self.history_sweep_time = curr_bev.new_zeros(b, self.num_T)

    def _roll_history(self):
        self.history_bev = torch.roll(self.history_bev, shifts=1, dims=1)
        self.history_bev[:, 0] = 0
        self.history_sweep_time = torch.roll(self.history_sweep_time, shifts=1, dims=1)
        self.history_sweep_time[:, 0] = 0

    @force_fp32()
    def update_history(self, voxel_feat, last_occ_pred, img_metas):
        """Update history information (called by detector)."""
        # This method is called by the detector, but the actual history update
        # is already handled in the forward method, so this is a no-op
        pass

    def forward(
        self,
        curr_bev,
        cam_params,
        history_fusion_params,
        dx,
        bx,
        history_last_bev=None,
        last_occ_pred=None,
        nonempty_prob=None,
        img_feats=None,
        spatial_shapes=None,
        pred_img_depth=None,
        **kwargs,
    ):
        curr_cam_extrins = cam_params[0]  # [bs, num_cams, 4, 4]
        curr_cam_intrins = cam_params[2]  # [bs, num_cams, 4, 4]
        forward_augs = cam_params[4]      # [bs, 4, 4] 前向变换矩阵
        self.num_cams = curr_cam_extrins.shape[1]  # 从外参中获取相机数量
        bs, c_, z, h, w = curr_bev.shape
        device = curr_bev.device
        mc = self.history_num * c_        # 历史特征总通道数
        # self.history_forward_augs = forward_augs.clone()


        ref_3d = self.get_reference_points(
            h, w, z, z, dim='3d', bs=bs, device=device, dtype=curr_bev.dtype) # torch.Size([3, 2, 625, 3]) #[bs,z,yx,3(x,y,z)]
        # ref_2d = self.get_reference_points(
        #     h, w, dim='2d', bs=bs, device=device, dtype=curr_bev.dtype) #torch.Size([3, 625, 1, 2])
        slots = torch.zeros(list([ref_3d.shape[0],ref_3d.shape[2],ref_3d.shape[1]])).to(ref_3d)
        reference_points_cam, reference_points_depth, bev_mask = self.point_sampling(ref_3d, self.pc_range, img_metas=kwargs['img_metas'], cam_params=cam_params)
        indexes = [[] for _ in range(bs)]
        spatial_shapes =[]
        spatial_shapes.append([16, 44])
        spatial_shapes = torch.tensor(spatial_shapes).to(device)
        pred_img_depth = pred_img_depth.view(bs * 6, -1, spatial_shapes[0][0], spatial_shapes[0][1])
        pred_img_depth = pred_img_depth.flatten(2).permute(0, 2, 1)  
        max_len = 0
        for j in range(bs):
            for i, mask_per_img in enumerate(bev_mask):
                index_query_per_img = mask_per_img[j].sum(-1).nonzero().squeeze(-1)
                if len(index_query_per_img) == 0:
                    index_query_per_img = bev_mask[i][j].sum(-1).nonzero().squeeze(-1)[0:1]
                indexes[j].append(index_query_per_img)
                # for batch operation, we need to pad the indexes to the same length
                max_len = max(max_len, len(index_query_per_img))
        reference_points_cam_rebatch = reference_points_cam.new_zeros([bs, self.num_cams, max_len, z, 2])
        reference_points_depth_rebatch = reference_points_depth.new_zeros([bs, self.num_cams, max_len, z, 1])

        for j in range(bs):
            for i, (reference_points_per_img, reference_points_depth_per_img) in enumerate(zip(reference_points_cam, reference_points_depth)):
                index_query_per_img = indexes[j][i]
                reference_points_cam_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]
                reference_points_depth_rebatch[j, i, :len(index_query_per_img)] = reference_points_depth_per_img[j, index_query_per_img]

        #use deformble attn
        depth_reference_points = reference_points_cam_rebatch.reshape(bs*6, max_len*z, 1, 1, 1, 2).contiguous()
        depth_attention_weights = torch.ones_like(depth_reference_points[..., 0]).contiguous()
        pred_img_depth = pred_img_depth.unsqueeze(2).contiguous()
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))



        bev_query_depth_rebatch = (reference_points_depth_rebatch- self.dbound[0])/ self.dbound[2]
        bev_query_depth_rebatch = torch.clip(torch.floor(bev_query_depth_rebatch), 0, 88-1).to(torch.long)
        bev_query_depth_rebatch = F.one_hot(bev_query_depth_rebatch.squeeze(-1),
                                   num_classes=88)

        depth_output = MultiScaleDeformableAttnFunction_fp32.apply(pred_img_depth, spatial_shapes,level_start_index,depth_reference_points,depth_attention_weights, self.im2col_step)
        depth_output = depth_output.reshape(bs,6, max_len,z, -1)   # [bs*num_cam, num_query, num_Z_anchors, C]
        # reference_points_depth_rebatch

        increment = torch.zeros_like(depth_output)

        depth_output = depth_output + torch.cat([(torch.zeros_like(depth_output[...,:1]) + 1e-9),torch.zeros_like(depth_output[...,1:])],dim=-1)


        depth_output =depth_output/depth_output.sum(-1)[...,None] #bs,xy,z,D




        depth_output = (1-depth_output.cumsum(dim=-1))
        depth_output =torch.cat([torch.ones_like(depth_output[...,0:1]),depth_output[...,0:-1]],dim=-1)

        # print("print((depth_output[...,-1]==0).sum())")
        # print((depth_output[...,-1]==0).sum())
        depth_output = (bev_query_depth_rebatch*depth_output).sum(-1)



        for j in range(bs):
            for i in range(6):
                index_query_per_img = indexes[j][i]
                slots[j, index_query_per_img] = torch.max(slots[j, index_query_per_img],depth_output[j, i, :len(index_query_per_img)])

        V_curr =slots.view(bs, 1, h, w,z).permute(0, 1, 4, 2, 3)
        b, c, z, h, w = curr_bev.shape

        if type(history_fusion_params['sequence_group_idx']) is list:
            seq_ids = history_fusion_params['sequence_group_idx'][0]
        else:
            seq_ids = history_fusion_params['sequence_group_idx']
        if type(history_fusion_params['start_of_sequence']) is list:
            start_of_sequence = history_fusion_params['start_of_sequence'][0]
        else:
            start_of_sequence = history_fusion_params['start_of_sequence']
        if type(history_fusion_params['curr_to_prev_ego_rt']) is list:
            curr_to_prev_ego_rt = history_fusion_params['curr_to_prev_ego_rt'][0]
        else:
            curr_to_prev_ego_rt = history_fusion_params['curr_to_prev_ego_rt']
        forward_augs = cam_params[-1]  # bda

        # compute visibility using depth (GatedTemporalFusion6_cat style)
        # V_curr_new = self._compute_visibility_from_depth(curr_bev, cam_params, dx, bx, pred_img_depth, kwargs.get('img_metas', None))

        # init / reset history
        assert (seq_ids >= 0).all()

        # 2、Deal with first batch
        if self.history_bev is None:
            self.history_bev = curr_bev.clone()
            self.history_forward_augs = forward_augs.clone()
            self.history_seq_ids = seq_ids.clone()
            self.history_bev = curr_bev.repeat(1, self.history_num, 1, 1, 1)
            self.history_sweep_time = curr_bev.new_zeros(curr_bev.shape[0], self.history_num)
            self.history_visibility = V_curr.repeat(1, self.history_num, 1, 1, 1).half()

        self.history_bev = self.history_bev.detach()
        self.history_visibility = self.history_visibility.detach().half()
        
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
            self.history_visibility[start_of_sequence] = V_curr[start_of_sequence].repeat(1, self.history_num, 1, 1, 1).half()
        # 生成体素网格和BEV变换矩阵
        grid = self.generate_grid(curr_bev) #[bs,y,x,z,4]
        grid_3d = grid
        feat2bev = self.generate_feat2bev(grid, dx, bx)

        
        rt_flow = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_ego_rt @ torch.inverse(forward_augs) @ feat2bev)
        
        bs, mc, z, h, w = self.history_bev.shape
        n, c_, z, h, w = curr_bev.shape
        grid = rt_flow.view(n, 1, 1, 1, 4, 4) @ grid
        # -------------------------- 6. 采样网格生成后打印 --------------------------
        # 生成采样网格（归一化到[-1,1]，适配F.grid_sample）
        normalize_factor = torch.tensor([w - 1.0, h - 1.0, z - 1.0], dtype=curr_bev.dtype, device=device)
        # grid_sampler = grid_transformed[..., :3, 0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0  # [bs, h, w, z, 3]
        # # 调整采样网格维度（适配F.grid_sample输入：[bs, z, h, w, 3]）
        # grid_sampler_permuted = grid_sampler.permute(0, 3, 1, 2, 4)  # 交换z和h/w维度
        grid = grid[:, :, :, :, :3, 0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0   # grid order is x, y, z


        # print("\n6. 采样网格生成后形状：")
        # print(f"grid_sampler (归一化后): {grid_sampler.shape} (预期：[bs, h, w, z, 3])")
        # print(f"grid_sampler_permuted (适配采样): {grid_sampler_permuted.shape} (预期：[bs, z, h, w, 3])")

        # -------------------------- 7. 历史BEV采样后打印 --------------------------
        # 采样历史BEV特征
        sampled_history_bev = F.grid_sample(
            self.history_bev.reshape(bs, mc, z, h, w),  # 输入：[bs, mc, z, h, w]
            grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4),
            align_corners=True,
            mode='bilinear'
        )
        sampled_history_visibility = F.grid_sample(
            self.history_visibility.reshape(bs, self.history_num, z, h, w).half(),  # 输入：[bs, mc, z, h, w]
            grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4).half(),
            align_corners=True,
            mode='nearest'
        )
        V_prev = sampled_history_visibility #bs,4,z,h,w

        # # push current into queue
        # # self._roll_history()

        # # self.history_bev[:, 0] = curr_bev
        # # self.history_forward_augs = forward_augs
        # # self.history_seq_ids = seq_ids
        # tmp_bev = self.history_bev
        # bs, mc, z, h, w = tmp_bev.shape
        # n, c_, z, h, w = curr_bev.shape

        # grid = self.generate_grid(curr_bev)
        # feat2bev = self.generate_feat2bev(grid, dx, bx)
        # rt_flow = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_ego_rt @ torch.inverse(forward_augs) @ feat2bev)
        # grid = rt_flow.view(n, 1, 1, 1, 4, 4) @ grid

        # normalize_factor = torch.tensor([w - 1.0, h - 1.0, z - 1.0], dtype=curr_bev.dtype, device=curr_bev.device)
        # grid = grid[:, :, :, :, :3, 0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0   # grid order is x, y, z

        # # sample the history bev
        # tmp_bev = tmp_bev.reshape(bs, mc, z, h, w)
        # sampled_history_bev = F.grid_sample(tmp_bev, grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4),  align_corners=True, mode='bilinear')

        # 使用多级沙漏融合算法
        b, _, z, h, w = curr_bev.shape
        num_voxels = z * h * w
        n, c_, z, h, w = curr_bev.shape
        
        # 获取分组索引
        groups = self._split_indices(V_curr, V_prev)
        
        # 将特征展平
        curr_bev_flat = curr_bev.reshape(b, c_, num_voxels)
        history_bev_flat = sampled_history_bev.reshape(b, self.num_T, c_, num_voxels)
        
        # 初始化融合后的特征
        fused = curr_bev_flat.clone()
        
        # 对每个分组进行独立的融合
        for group_idx, group_indices in enumerate(groups):
            if group_indices.numel() == 0:
                continue
            
            # 获取当前分组的历史窗口大小
            win_size = self.queue_windows[group_idx] if group_idx < len(self.queue_windows) else self.queue_windows[-1]
            win_size = min(win_size, self.num_T)
            
            # 获取当前分组的大小
            group_size = group_indices.shape[1]
            if group_size == 0:
                continue
            
            # 批量处理所有batch
            # 1. 获取当前体素的特征 [b, c, group_size]
            curr_voxels = torch.gather(curr_bev_flat, dim=2, index=group_indices.unsqueeze(1).expand(b, c_, group_size))
            
            # 2. 获取历史特征 [b, win_size, c, group_size]
            history_voxels = torch.gather(history_bev_flat[:, :win_size], dim=3, index=group_indices.unsqueeze(1).unsqueeze(1).expand(b, win_size, c_, group_size))
            
            # 3. 调整历史特征形状 [b, group_size, c*win_size]
            history_voxels = history_voxels.permute(0, 3, 2, 1).reshape(b, group_size, -1)
            
            # 4. 调整当前特征形状 [b, group_size, c]
            curr_voxels = curr_voxels.permute(0, 2, 1)
            
            # 5. 融合当前特征和历史特征 [b, group_size, c + c*win_size]
            combined = torch.cat([curr_voxels, history_voxels], dim=2)
            
            # 6. 使用当前分组对应的MLP层进行融合 [b, group_size, c]
            fused_features = self.history_fusion_linears[group_idx](combined)
            
            # 7. 调整融合特征形状 [b, c, group_size]
            fused_features = fused_features.permute(0, 2, 1)
            
            # 8. 将融合后的特征添加到当前特征中
            fused = fused.scatter_add(2, group_indices.unsqueeze(1).expand(b, c_, group_size), fused_features)
        
        # 调整形状并返回
        fused = fused.view(b, c_, z, h, w)
        return fused
