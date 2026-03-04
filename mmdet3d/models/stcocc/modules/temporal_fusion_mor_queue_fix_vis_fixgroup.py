import math
import os
from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import BaseModule, force_fp32

from mmdet3d.models.builder import HEADS


def _as_tensor_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _clamp_windows(windows: Sequence[int], history_num: int) -> Tuple[int, ...]:
    clamped = []
    for w in windows:
        if w is None:
            continue
        w_int = int(w)
        if w_int <= 0:
            continue
        clamped.append(min(w_int, int(history_num)))
    if not clamped:
        return (int(history_num),)
    clamped = sorted(set(clamped))
    return tuple(clamped)


def _default_windows(history_num: int) -> Tuple[int, ...]:
    # Similar to variable queues commonly used in this repo: (T/8, T/4, T/2, T).
    candidates = (
        max(int(history_num) // 8, 1),
        max(int(history_num) // 4, 1),
        max(int(history_num) // 2, 1),
        int(history_num),
    )
    return _clamp_windows(candidates, history_num)


def _default_capacity_factors(num_steps: int) -> Tuple[float, ...]:
    # MoR expert-choice default: N_r/N_r, (N_r-1)/N_r, ..., 1/N_r
    # This yields equal-sized depth buckets in expectation.
    num_steps = max(int(num_steps), 1)
    return tuple((num_steps - i) / num_steps for i in range(num_steps))


def _compute_occ_hardness(
    occ_logits: torch.Tensor,
    free_index: Optional[int],
    eps: float,
    mode: str,
) -> torch.Tensor:
    """Compute per-voxel difficulty score from occupancy logits.

    Args:
        occ_logits: [B, H, W, Z, C] logits (as produced by BEVFormerEncoderWithoutPredictor).
        free_index: index of free/empty class (optional). When provided, down-weight near-free voxels.
        eps: numerical epsilon.
        mode: 'entropy' | 'margin' | '1-max'

    Returns:
        score: [B, Z, H, W] in [0, 1] (approximately).
    """
    if occ_logits.dim() != 5:
        raise ValueError(f'Expected occ_logits dim=5, got shape={tuple(occ_logits.shape)}')

    # [B, H, W, Z, C] -> [B, C, Z, H, W]
    logits = occ_logits.permute(0, 4, 3, 1, 2).contiguous()
    b, c, z, h, w = logits.shape
    probs = logits.softmax(dim=1)

    free_prob = None
    if free_index is not None:
        free_index = int(free_index) % c
        free_prob = probs[:, free_index]  # [B, Z, H, W]

    if mode == 'entropy':
        entropy = -(probs * (probs.clamp(min=eps)).log()).sum(dim=1)  # [B, Z, H, W]
        entropy = entropy / math.log(float(c))
        score = entropy
    elif mode == '1-max':
        score = 1.0 - probs.max(dim=1).values
    elif mode == 'margin':
        top2 = probs.topk(k=2, dim=1).values  # [B, 2, Z, H, W]
        margin = top2[:, 0] - top2[:, 1]
        score = 1.0 - margin
    else:
        raise ValueError(f'Unsupported occ hardness mode={mode!r}')

    score = score.clamp(0.0, 1.0)
    if free_prob is not None:
        score = score * (1.0 - free_prob.clamp(0.0, 1.0))
    return score


@HEADS.register_module()
class MoRQueueFusionWithoutPredictor_fix_vis_fixgroup(BaseModule):
    """MoR-style expert-choice routing for variable temporal windows.

    This replaces unstable observability-based routing with a more task-aligned notion of "difficulty":
    voxels that are harder (less discriminative / more uncertain) are fused with longer temporal windows and
    more recursion steps, while easy voxels exit early (short window), reducing compute.

    Key ideas adapted from MoR (expert-choice routing):
    - At each recursion step, keep only top-k tokens (voxels) to continue.
    - Active set shrinks with depth according to a fixed capacity schedule (perfect load balancing).
    - Parameter sharing: use the same small fusion MLP across steps (optional step embedding).
    """

    def __init__(
        self,
        single_bev_num_channels: int = 96,
        history_num: int = 16,
        top_k: Optional[int] = None,
        queue_windows: Optional[Sequence[int]] = None,
        capacity_factors: Optional[Sequence[float]] = None,
        capacity_warmup_steps: int = 0,
        score_source: str = 'occ',  # 'occ' | 'feat'
        occ_score_mode: str = 'entropy',  # 'entropy' | 'margin' | '1-max'
        free_index: Optional[int] = -1,
        score_eps: float = 1e-6,
        num_classes=18,
        occ_embedims=32,
        fusion_hidden_channels: Optional[int] = None,
        share_fusion: bool = True,
        use_step_embedding: bool = True,
        debug_dump: bool = False,
        debug_dump_dir: Optional[str] = None,
        debug_dump_interval: int = 2000,
        debug_dump_max: int = 20,
        group_ratios=None,
        **kwargs,
    ):
        super().__init__()
        self.occ_embedims = occ_embedims

        self.single_bev_num_channels = int(single_bev_num_channels)
        self.history_num = int(history_num)
        self.top_k = None if top_k is None else int(top_k)
        self.num_T = self.history_num
        if queue_windows is None:
            queue_windows = _default_windows(self.history_num)
        self.queue_windows = _clamp_windows(queue_windows, self.history_num)
        self.time_fusion_n_channel=[self.queue_windows[0]]+ [self.queue_windows[x+1]-self.queue_windows[x] for x in range(len(self.queue_windows)-1)]
        self.time_fusion_in_channel = [self.queue_windows[0]*self.single_bev_num_channels]+ [(self.queue_windows[x+1]-self.queue_windows[x])*self.single_bev_num_channels for x in range(len(self.queue_windows)-1)]
        self.num_steps = len(self.queue_windows)
        self.occ_embedding = nn.Sequential(
            nn.Linear(num_classes, occ_embedims),
            nn.Softplus(),
            nn.Linear(occ_embedims, occ_embedims),
        )
        if capacity_factors is None:
            capacity_factors = _default_capacity_factors(self.num_steps)
        capacity_factors = [float(x) for x in _as_tensor_list(capacity_factors)]
        if len(capacity_factors) != self.num_steps:
            raise ValueError(
                f'capacity_factors length ({len(capacity_factors)}) must match number of steps ({self.num_steps})'
            )
        if any(x <= 0.0 or x > 1.0 for x in capacity_factors):
            raise ValueError(f'capacity_factors must be in (0,1], got {capacity_factors}')
        for i in range(1, len(capacity_factors)):
            if capacity_factors[i] > capacity_factors[i - 1]:
                raise ValueError(f'capacity_factors must be non-increasing, got {capacity_factors}')
        self.capacity_factors = tuple(capacity_factors)
        self.capacity_warmup_steps = int(capacity_warmup_steps)
        self._train_forward_calls = 0

        self.score_source = score_source
        self.occ_score_mode = occ_score_mode
        self.free_index = free_index
        self.score_eps = float(score_eps)

        hidden = int(fusion_hidden_channels) if fusion_hidden_channels is not None else self.single_bev_num_channels
        fusion_in = self.single_bev_num_channels * 2
        if share_fusion:
            self.fusion_mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(fusion_in+self.single_bev_num_channels+1, hidden),
                    nn.SiLU(inplace=True),
                    nn.Linear(hidden, self.single_bev_num_channels),
                )
            ])
        else:
            self.fusion_mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(in_channel+self.single_bev_num_channels+self.occ_embedims, hidden),
                    nn.SiLU(inplace=True),
                    nn.Linear(hidden, self.single_bev_num_channels),
                )
                for in_channel in self.time_fusion_in_channel
            ])
        self.share_fusion = bool(share_fusion)

        self.use_step_embedding = bool(use_step_embedding)
        # self.step_embedding = (
        #     nn.Embedding(self.num_steps, self.single_bev_num_channels) if self.use_step_embedding else None
        # )

        # history buffers (kept in "previous frame coordinate", then aligned each forward via grid_sample)
        self.history_bev: Optional[torch.Tensor] = None  # [B, T, C, Z, H, W]
        self.history_last_bev: Optional[torch.Tensor] = None  # [B, C, Z, H, W]
        self.history_forward_augs: Optional[torch.Tensor] = None  # [B, 4, 4]
        self.history_seq_ids: Optional[torch.Tensor] = None  # [B]

        # debug dumping
        self.debug_dump = bool(debug_dump)
        self.debug_dump_dir = debug_dump_dir
        self.debug_dump_interval = int(debug_dump_interval)
        self.debug_dump_max = int(debug_dump_max)
        self._debug_forward_calls = 0
        self._debug_dump_count = 0
        if group_ratios is None:
            group_ratios = (8, 4, 2, 1)  # 默认等比例分配
        self.group_ratios = group_ratios

    def update_history(self, voxel_feat, last_occ_pred, img_metas):
        # History is managed in forward(); this method exists for detector compatibility.
        return

    @staticmethod
    def generate_grid(curr_bev: torch.Tensor) -> torch.Tensor:
        """Generate homogeneous grid in BEV index space.

        Returns:
            grid: [B, H, W, Z, 4, 1], with (x, y, z, 1) coordinates.
        """
        b, _, z, h, w = curr_bev.shape
        xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w, 1).expand(h, w, z)
        ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1, 1).expand(h, w, z)
        zs = torch.linspace(0, z - 1, z, dtype=curr_bev.dtype, device=curr_bev.device).view(1, 1, z).expand(h, w, z)
        grid = torch.stack((xs, ys, zs, torch.ones_like(xs)), -1)
        grid = grid.view(1, h, w, z, 4).expand(b, h, w, z, 4).view(b, h, w, z, 4, 1)
        return grid

    @staticmethod
    def generate_feat2bev(grid: torch.Tensor, dx, bx) -> torch.Tensor:
        feat2bev = torch.zeros((4, 4), dtype=grid.dtype, device=grid.device)
        if isinstance(dx, (int, float)):
            feat2bev[0, 0] = dx
            feat2bev[1, 1] = dx
            feat2bev[2, 2] = dx
            dx0 = float(dx)
        else:
            feat2bev[0, 0] = dx[0]
            feat2bev[1, 1] = dx[1]
            feat2bev[2, 2] = dx[2]
            dx0 = float(dx[0])

        if isinstance(bx, (int, float)):
            bx0 = float(bx)
            feat2bev[0, 3] = bx0 - dx0 / 2.0
            feat2bev[1, 3] = bx0 - dx0 / 2.0
            feat2bev[2, 3] = bx0 - dx0 / 2.0
        else:
            feat2bev[0, 3] = bx[0] - dx[0] / 2.0 if not isinstance(dx, (int, float)) else bx[0] - dx0 / 2.0
            feat2bev[1, 3] = bx[1] - dx[1] / 2.0 if not isinstance(dx, (int, float)) else bx[1] - dx0 / 2.0
            feat2bev[2, 3] = bx[2] - dx[2] / 2.0 if not isinstance(dx, (int, float)) else bx[2] - dx0 / 2.0

        feat2bev[3, 3] = 1.0
        return feat2bev.view(1, 4, 4)

    def _compute_score(self, curr_bev: torch.Tensor, last_occ_pred: Optional[torch.Tensor]) -> torch.Tensor:
        b, c, z, h, w = curr_bev.shape
        if self.score_source == 'occ' and last_occ_pred is not None:
            # score: [B, Z, H, W]
            return _compute_occ_hardness(
                occ_logits=last_occ_pred,
                free_index=self.free_index,
                eps=self.score_eps,
                mode=self.occ_score_mode,
            )
        if self.score_source == 'feat':
            # feature-norm difficulty (fallback)
            score = curr_bev.reshape(b, c, -1).norm(dim=1).view(b, z, h, w)
            score = (score - score.amin(dim=(1, 2, 3), keepdim=True)) / (
                score.amax(dim=(1, 2, 3), keepdim=True) - score.amin(dim=(1, 2, 3), keepdim=True) + self.score_eps
            )
            return score
        raise ValueError(f'Unsupported score_source={self.score_source!r} or missing last_occ_pred')

    def _maybe_dump_debug(
        self,
        score_zhw: torch.Tensor,
        depth_zhw: torch.Tensor,
        meta: dict,
    ) -> None:
        self._debug_forward_calls += 1
        do_dump = (
            self.debug_dump
            and self.debug_dump_dir is not None
            and self._debug_dump_count < self.debug_dump_max
            and self.debug_dump_interval > 0
            and (self._debug_forward_calls % self.debug_dump_interval == 0)
        )
        if not do_dump:
            return

        is_main = True
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            is_main = torch.distributed.get_rank() == 0
        if not is_main:
            return

        os.makedirs(self.debug_dump_dir, exist_ok=True)
        out_path = os.path.join(self.debug_dump_dir, f'mor_queue_debug_step{self._debug_forward_calls:06d}.pt')
        payload = {
            **meta,
            'queue_windows': self.queue_windows,
            'capacity_factors': self.capacity_factors,
            'score_source': self.score_source,
            'occ_score_mode': self.occ_score_mode,
            'score_zhw': score_zhw.detach().cpu(),
            'depth_zhw': depth_zhw.detach().cpu(),
        }
        torch.save(payload, out_path)
        self._debug_dump_count += 1

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
        b, z, h, w = V_curr.shape
        num_voxels = z * h * w
        
        # 将所有可观测性展平
        V_curr_flat = V_curr.reshape(b, num_voxels)
        V_prev_flat = V_prev.reshape(b, self.num_T, num_voxels)
        
        # 计算每个体素在所有时间步的最大可观测性评分
        all_visibility = torch.cat([V_curr_flat.unsqueeze(1), V_prev_flat], dim=1)  # [bs, num_T+1, num_voxels]
        max_scores, _ = all_visibility.max(dim=1)  # [bs, num_voxels]
        
        # 对体素按最大可观测性评分降序排序
        sorted_indices = torch.argsort(max_scores, dim=1, descending=False)
        
        # 计算分组数量
        num_groups = len(self.queue_windows)
        groups = []
        remaining_mask = torch.ones(b, num_voxels, dtype=torch.bool, device=V_curr.device)
        
        # 计算总比例数
        total_ratio = sum(self.group_ratios[:num_groups])
        
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
            window_min_scores, _ = window_visibility_masked.min(dim=1)  # [bs, num_voxels]
            
            # 在剩余体素中找到评分最高的group_size个体素
            window_min_scores[~remaining_mask] = 1e9  # 将非剩余体素的评分设为最小值
            _, top_indices = torch.topk(window_min_scores, k=group_size, dim=1, largest=False)
            group_indices = top_indices
                
            # 将选中的体素从剩余列表中移除
            for batch_idx in range(b):
                selected_voxels = group_indices[batch_idx]
                remaining_mask[batch_idx, selected_voxels] = False
            
            groups.append(group_indices)
        
        return groups

    @force_fp32()
    def forward(
        self,
        curr_bev: torch.Tensor,
        cam_params,
        history_fusion_params: dict,
        dx,
        bx,
        history_last_bev=None,
        last_occ_pred: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Parse history fusion params with safe fallbacks.
        b, c, z, h, w = curr_bev.shape
        c_=c
        device = curr_bev.device
        n = z * h * w
        score_cur = self._compute_score(curr_bev, last_occ_pred)  # [B, Z, H, W]
        # score_cur = score_zhw.reshape(b, n)
        seq_ids = history_fusion_params.get('sequence_group_idx', torch.zeros(b, dtype=torch.long, device=device))
        if isinstance(seq_ids, list):
            seq_ids = seq_ids[0]
        start_of_sequence = history_fusion_params.get(
            'start_of_sequence', torch.ones(b, dtype=torch.bool, device=device)
        )
        if isinstance(start_of_sequence, list):
            start_of_sequence = start_of_sequence[0]
        curr_to_prev_ego_rt = history_fusion_params.get(
            'curr_to_prev_ego_rt', torch.eye(4, dtype=curr_bev.dtype, device=device).unsqueeze(0)
        )
        if isinstance(curr_to_prev_ego_rt, list):
            curr_to_prev_ego_rt = curr_to_prev_ego_rt[0]

        forward_augs = cam_params[-1]  # bda matrix, [B, 4, 4]

        # Init history on first call.
        # if self.history_bev is None:
        #     self.history_bev = curr_bev.repeat(1, self.history_num, 1, 1, 1)
        #     self.history_forward_augs = forward_augs.clone()
        #     self.history_seq_ids = seq_ids.clone()
        
        # self.history_bev = self.history_bev.detach()
        if self.history_bev is None:
            
            self.history_forward_augs = forward_augs.clone()
            self.history_seq_ids = seq_ids.clone()
            self.history_bev = curr_bev.repeat(1, self.history_num, 1, 1, 1)
            # self.history_sweep_time = curr_bev.new_zeros(curr_bev.shape[0], self.history_num)
            # self.history_visibility = V_curr.repeat(1,1, self.history_num, 1, 1, 1).half()
            self.history_score = score_cur.unsqueeze(1).repeat(1, self.history_num, 1, 1, 1).half()
        self.history_bev = self.history_bev.detach()

        # Reset new sequences.
        # if start_of_sequence.any():
        #     self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(
        #         1, self.history_num, 1, 1, 1
        #     )
        #     self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]
        #     self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
        if start_of_sequence.sum()>0:
            self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_num, 1, 1, 1)
           
            self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]
            self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
            # self.history_sweep_time[start_of_sequence] = 0  # zero the new sequence timestep starts
            self.history_score[start_of_sequence] = score_cur[start_of_sequence].unsqueeze(1).repeat(1,self.history_num, 1, 1, 1).half()
        torch.cuda.empty_cache()

        # Sanity: sequence ids should only change at start_of_sequence.
        mismatch = (self.history_seq_ids != seq_ids) & (~start_of_sequence)
        if mismatch.any():
            raise AssertionError(f'Unexpected sequence id change at indices: {mismatch.nonzero().view(-1).tolist()}')

        # Align history BEV to current coordinate.
        grid = self.generate_grid(curr_bev)
        feat2bev = self.generate_feat2bev(grid, dx, bx)
        rt_flow = (
            torch.inverse(feat2bev)
            @ self.history_forward_augs
            @ curr_to_prev_ego_rt
            @ torch.inverse(forward_augs)
            @ feat2bev
        )
        grid = rt_flow.view(b, 1, 1, 1, 4, 4) @ grid
        normalize_factor = torch.tensor([w - 1.0, h - 1.0, z - 1.0], dtype=curr_bev.dtype, device=device)
        grid = grid[:, :, :, :, :3, 0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0

        # hist_in = self.history_bev.reshape(b, self.history_num * c, z, h, w)
        sampled_history_bev = F.grid_sample(
            self.history_bev.reshape(b, self.history_num * c, z, h, w),
            grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4),
            align_corners=True,
            mode='bilinear',
        )
        sampled_history_score = F.grid_sample(
            self.history_score.reshape(b, self.history_num, z, h, w).half(),  # 输入：[bs, mc, z, h, w]
            grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4).half(),
            align_corners=True,
            mode='bilinear'
        )
        score_prev = sampled_history_score #bs,4,z,h,w

        # hist_aligned = hist_aligned.view(b, self.history_num, c, z, h, w)

        # Flatten tokens.
        n = z * h * w
        num_voxels = n
        curr_bev_flat = curr_bev.reshape(b, c, num_voxels)
        history_bev_flat = sampled_history_bev.reshape(b, self.num_T, c, num_voxels)
        fused = curr_bev_flat.clone()

        # x = curr_bev.reshape(b, c, n).permute(0, 2, 1).contiguous()  # [B, N, C]
        # hist = hist_aligned.view(b, self.history_num, c, n).contiguous()  # [B, T, C, N]

        # Compute difficulty score (higher => harder).
        # fg_occ_feat = torch.gather(last_occ_pred, dim=1, index=fg_indices.unsqueeze(-1).repeat(1, 1, last_occ_pred.shape[-1]))  # [bs, K, num_classes]
        # occ_feat = self.occ_embedding(last_occ_pred.permute(0, 3, 1, 2, 4).contiguous()).reshape(b, n, self.occ_embedims).permute(0, 2, 1)  # [bs, K, occ_embedims]
        occ_feat = self.occ_embedding(last_occ_pred.permute(0, 3, 1, 2, 4)).reshape(b, n, self.occ_embedims).permute(0, 2, 1)  # [bs, K, occ_embedims]
        
        self.top_k = curr_bev.shape[2]*curr_bev.shape[3]*curr_bev.shape[4]
        # Candidate sparsification (optional).
        groups = self._split_indices(score_cur, score_prev)
        # cur_and_history_bev_flat= torch.cat([curr_bev_flat, history_bev_flat.reshape(b, self.num_T*c, num_voxels)], dim=1)
        for group_idx, group_indices in enumerate(groups):
            if group_indices.numel() == 0:
                continue
            
            # 获取当前分组的历史窗口大小
            win_size = self.queue_windows[group_idx] if group_idx < len(self.queue_windows) else self.queue_windows[-1]
            win_size = min(win_size, self.num_T)
            
            # 获取当前分组的大小
            # group_size = group_indices.shape[1]
            group_size = sum([  x.shape[1] for x in  groups[group_idx:]])
            if group_size == 0:
                continue
            
            # 批量处理所有batch
            # 1. 获取当前体素的特征 [b, c, group_size]
            curr_voxels = torch.gather(curr_bev_flat, dim=2, index=torch.cat(groups[group_idx:],dim=1).unsqueeze(1).expand(b, c, group_size))
            curr_occ_feat = torch.gather(occ_feat, dim=2, index=torch.cat(groups[group_idx:],dim=1).unsqueeze(1).expand(b, self.occ_embedims, group_size))
            
            # 2. 获取历史特征 [b, win_size, c, group_size]
            history_voxels = torch.gather(history_bev_flat[:, sum(self.time_fusion_n_channel[:group_idx]):sum(self.time_fusion_n_channel[:group_idx+1])], dim=3, index=torch.cat(groups[group_idx:],dim=1).unsqueeze(1).unsqueeze(1).expand(b, self.time_fusion_n_channel[group_idx], c, group_size))
            
            # 3. 调整历史特征形状 [b, group_size, c*win_size]
            history_voxels = history_voxels.permute(0, 3, 2, 1).reshape(b, group_size, -1)
            
            # 4. 调整当前特征形状 [b, group_size, c]
            curr_voxels = curr_voxels.permute(0, 2, 1)
            curr_occ_feat = curr_occ_feat.permute(0, 2, 1)
            # 5. 融合当前特征和历史特征 [b, group_size, c + c*win_size + occ_embedims]
            # combined = torch.cat([curr_voxels, history_voxels, curr_occ_feat], dim=2).contiguous()
            combined = torch.cat([curr_voxels, history_voxels, curr_occ_feat], dim=2)

            
            # 6. 使用当前分组对应的MLP层进行融合 [b, group_size, c]
            fused_features = self.fusion_mlps[group_idx](combined)
            
            # 7. 调整融合特征形状 [b, c, group_size]
            fused_features = fused_features.permute(0, 2, 1)
            
            # 8. 将融合后的特征添加到当前特征中
            fused = fused.scatter_add(2, torch.cat(groups[group_idx:],dim=1).unsqueeze(1).expand(b, c, group_size), fused_features)
        
        # 调整形状并返回
        fused = fused.view(b, c, z, h, w)
        # 更新历史缓存
        self.history_last_bev = fused.detach().clone()
        self.history_bev = torch.cat([curr_bev,sampled_history_bev[:, :-c, ...]], dim=1).detach()
        # self.history_cam_intrins = torch.cat([curr_cam_intrins.unsqueeze(1),self.history_cam_intrins[:, :-1, ...]], dim=1).detach()
        # self.history_cam_extrins = torch.cat([curr_cam_extrins.unsqueeze(1),self.history_cam_extrins[:, 1-1:, ...]], dim=1).detach()
        self.history_score =torch.cat([score_cur.unsqueeze(1), score_prev[:, :-1, ...]],dim=1).detach().half()
        self.history_forward_augs = forward_augs.clone()
        # Debug dump: score and depth map (0 means not selected / not in candidates).
        # if self.debug_dump and self.debug_dump_dir is not None:
        #     # rank within candidates
        #     rank = torch.empty_like(sorted_pos)
        #     rank.scatter_(1, sorted_pos, torch.arange(k, device=device).view(1, k).expand(b, -1))
        #     depth_cand = torch.zeros_like(rank)
        #     for ks in k_steps:
        #         depth_cand += (rank < ks).to(depth_cand.dtype)

        #     depth_full = torch.zeros(b, n, dtype=torch.int16, device=device)
        #     depth_full.scatter_(1, cand_idx, depth_cand.to(depth_full.dtype))
        #     depth_zhw = depth_full.view(b, z, h, w)
        #     meta = {
        #         'num_voxels': int(n),
        #         'candidate_k': int(k),
        #         'k_steps': [int(x) for x in k_steps],
        #         'capacity_warmup_steps': int(self.capacity_warmup_steps),
        #         'capacity_warmup_ratio': float(warmup_ratio),
        #         'capacity_factors_used': [float(x) for x in cap],
        #         'score_min': float(score.min().item()),
        #         'score_max': float(score.max().item()),
        #     }
        #     self._maybe_dump_debug(score_zhw=score_zhw, depth_zhw=depth_zhw, meta=meta) 
        
        
        if self.debug_dump and self.debug_dump_dir is not None:
            self._debug_forward_calls += 1
            do_dump = (
                self._debug_dump_count < self.debug_dump_max
                and self.debug_dump_interval > 0
                and (self._debug_forward_calls % self.debug_dump_interval == 0)
            )
            
            if do_dump:
                # 只有主进程保存
                is_main = True
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    is_main = torch.distributed.get_rank() == 0
                
                if is_main:
                    os.makedirs(self.debug_dump_dir, exist_ok=True)
                    
                    # 1. 构造 Group Map (可视化核心：每个体素被分配到了哪个组)
                    b, c, z, h, w = curr_bev.shape
                    n = z * h * w
                    group_zhw = torch.zeros((b, z, h, w), dtype=torch.int, device=curr_bev.device)
                    
                    # 遍历 groups，填充 group ID
                    # 注意：你的代码逻辑是 group_idx 越大，窗口越长。我们将其可视化出来。
                    for group_idx, indices in enumerate(groups):
                        if indices.numel() == 0:
                            continue
                        # indices shape: [b, k]
                        # 我们需要把这些索引对应的位置填上 group_idx
                        for batch_idx in range(b):
                            # 创建一个临时的一维向量
                            tmp = torch.zeros(n, dtype=torch.int, device=curr_bev.device)
                            tmp[indices[batch_idx]] = group_idx + 1  # +1 是为了让 0 代表未分配（虽然理论上全部分配了）
                            # 放回 3D 空间
                            group_zhw[batch_idx] += tmp.view(z, h, w)

                    # 2. 保存文件
                    out_path = os.path.join(self.debug_dump_dir, f'mor_vis_step_{self._debug_forward_calls:06d}.pt')
                    
                    # 收集元数据
                    meta = {
                        'queue_windows': self.queue_windows,
                        'group_ratios': self.group_ratios,
                        'step': self._debug_forward_calls,
                    }
                    
                    # 保存张量 ( detach 并转到 cpu )
                    payload = {
                        'score_zhw': score_cur.detach().cpu(),       # [B, Z, H, W]
                        'group_zhw': group_zhw.detach().cpu(),       # [B, Z, H, W]
                        'meta': meta
                    }
                    
                    torch.save(payload, out_path)
                    print(f"[Debug] Saved visualization data to {out_path}")
                    self._debug_dump_count += 1
        # =========================================================================================
        return fused
        
        if self.top_k is not None:
            k = max(1, min(int(self.top_k), int(n)))
            cand_idx = torch.topk(score, k, dim=1, largest=True, sorted=False)[1]  # [B, k]
        else:
            k = n
            cand_idx = torch.arange(n, device=device).view(1, n).repeat(b, 1)

        cand_idx_exp = cand_idx.unsqueeze(-1).expand(-1, -1, c)
        x_cand = torch.gather(x, dim=1, index=cand_idx_exp)  # [B, k, C]
        score_cand = torch.gather(score, dim=1, index=cand_idx)  # [B, k]
        hist_cand = torch.gather(hist, dim=3, index=cand_idx.view(b, 1, 1, k).expand(-1, self.history_num, c, -1))

        # Sort candidates by difficulty once (nested top-k across steps).
        sorted_pos = torch.argsort(score_cand, dim=1, descending=True)

        # Pre-compute per-step token counts and depth assignment for debugging.
        k_steps = [min(k, max(1, int(round(self.capacity_factors[s] * k)))) for s in range(self.num_steps)]
        cap = self.capacity_factors
        warmup_ratio = 1.0
        if self.training and self.capacity_warmup_steps > 0:
            # Linearly warm up from no dropping (all ones) to the target capacity schedule.
            warmup_ratio = min(1.0, float(self._train_forward_calls) / float(self.capacity_warmup_steps))
            cap = tuple(1.0 - (1.0 - float(x)) * warmup_ratio for x in cap)
        if self.training:
            self._train_forward_calls += 1

        k_steps = [min(k, max(1, int(round(cap[s] * k)))) for s in range(self.num_steps)]
        k_steps[0] = k  # ensure step-0 touches all candidates

        # Recursion steps: each step fuses an incremental chunk of history.
        prev_w = 0
        for step_idx, win in enumerate(self.queue_windows):
            start = prev_w
            end = int(win)
            prev_w = end
            if end <= start:
                continue
            if start >= self.history_num:
                break

            # Aggregate history chunk: [B, C, k] -> [B, k, C]
            chunk = hist_cand[:, start:end].mean(dim=1).permute(0, 2, 1).contiguous()

            # Select top-k tokens for this step (expert-choice).
            ks = k_steps[step_idx]
            pos = sorted_pos[:, :ks]
            pos_exp = pos.unsqueeze(-1).expand(-1, -1, c)

            x_sel = torch.gather(x_cand, dim=1, index=pos_exp)
            chunk_sel = torch.gather(chunk, dim=1, index=pos_exp)

            if self.step_embedding is not None:
                x_sel = x_sel + self.step_embedding.weight[step_idx].view(1, 1, -1)

            fusion_in = torch.cat([x_sel, chunk_sel], dim=-1)
            mlp = self.fusion_mlps[0] if self.share_fusion else self.fusion_mlps[step_idx]
            
            #下面这两行会导致梯度爆炸，先注释掉
            # delta = mlp(fusion_in)

            # x_cand = x_cand.scatter_add(dim=1, index=pos_exp, src=delta)

            #不会梯度爆炸的修复版本
            # 使用残差连接而不是累加，避免梯度爆炸
            fused_features = mlp(fusion_in)
            # 添加梯度裁剪以增强稳定性
            fused_features = torch.clamp(fused_features, min=-1.0, max=1.0)
            # 使用scatter替换scatter_add，避免多次累加导致的梯度爆炸
            x_sel_updated = x_sel + fused_features
            x_cand = x_cand.scatter(dim=1, index=pos_exp, src=x_sel_updated)
        # Scatter updated candidates back to full BEV.
        x_out = x.scatter(dim=1, index=cand_idx_exp, src=x_cand)
        fused = x_out.permute(0, 2, 1).contiguous().view(b, c, z, h, w)

        # Update history in CURRENT coordinate: [t, (t-1..t-(T-1))]
        new_hist = torch.empty_like(hist_aligned)
        new_hist[:, 0] = fused.detach()
        new_hist[:, 1:] = hist_aligned[:, : self.history_num - 1].detach()
        self.history_bev = new_hist
        self.history_last_bev = fused.detach()
        self.history_forward_augs = forward_augs.clone()
        self.history_seq_ids = seq_ids.clone()

        

        return fused

