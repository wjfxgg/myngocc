import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import BaseModule, force_fp32
from mmdet3d.models.builder import HEADS


@HEADS.register_module()
class ObservableQueueFusion(BaseModule):
    """Temporal fusion with observability-based variable queues.

    - Flatten voxel queries (z*h*w) and rank by observability score.
    - Split into 4 groups (high->low); use shorter windows for high observability,
      longer windows for low observability to save compute.
    - Window lengths default to (2, 4, 8, 16) for num_T=16.
    - Fused features are added back to the current BEV feature at the corresponding voxels.
    """

    def __init__(
        self,
        single_bev_num_channels,
        num_T=16,
        queue_windows=(2, 4, 8, 16),
        score_from='norm',  # 'norm' or 'nonempty'
        **kwargs,
    ):
        super().__init__()
        self.single_bev_num_channels = single_bev_num_channels
        self.num_T = num_T
        self.queue_windows = queue_windows
        self.score_from = score_from

        self.history_bev = None
        self.history_forward_augs = None
        self.history_seq_ids = None
        self.history_sweep_time = None
        
    @property
    def history_last_bev(self):
        """Return the most recent BEV feature for compatibility."""
        return self.history_bev[:, 0] if self.history_bev is not None else None

    def update_history(self, voxel_feat, occ_pred, img_metas):
        """Update history information (placeholder method for compatibility)."""
        # This method is required by the detector interface but doesn't need to do anything
        # as history is already managed within the forward method
        pass

    def init_history(self, curr_bev, forward_augs, seq_ids):
        b, c, z, h, w = curr_bev.shape
        self.history_bev = curr_bev.new_zeros(b, self.num_T, c, z, h, w)
        self.history_bev[:, 0] = curr_bev.detach()
        self.history_forward_augs = forward_augs.clone()
        self.history_seq_ids = seq_ids.clone()
        self.history_sweep_time = curr_bev.new_zeros(b, self.num_T)

    def roll_history(self):
        # shift right, drop the last
        self.history_bev = torch.roll(self.history_bev, shifts=1, dims=1).detach()
        self.history_bev[:, 0] = 0
        self.history_sweep_time = torch.roll(self.history_sweep_time, shifts=1, dims=1).detach()
        self.history_sweep_time[:, 0] = 0

    @staticmethod
    def _compute_score(curr_bev, nonempty_prob=None, method='norm'):
        b, c, z, h, w = curr_bev.shape
        if method == 'nonempty' and nonempty_prob is not None:
            score = nonempty_prob.reshape(b, -1)
        else:
            score = curr_bev.reshape(b, c, -1).norm(dim=1)
        return score

    def _split_indices(self, score):
        b, n = score.shape
        sorted_idx = torch.argsort(score, dim=1, descending=True)
        chunk = max(n // 4, 1)
        groups = [
            sorted_idx[:, i * chunk : (i + 1) * chunk] if i < 3 else sorted_idx[:, i * chunk :]
            for i in range(4)
        ]
        return groups

    @force_fp32()
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
    ):
        b, c, z, h, w = curr_bev.shape

        # seq ids and flags
        seq_ids = history_fusion_params.get('sequence_group_idx')
        if isinstance(seq_ids, list):
            seq_ids = seq_ids[0]
        start_of_sequence = history_fusion_params.get('start_of_sequence', torch.zeros(b, device=curr_bev.device, dtype=torch.bool))
        if isinstance(start_of_sequence, list):
            start_of_sequence = start_of_sequence[0]
        forward_augs = cam_params[-1]

        # init history
        if self.history_bev is None:
            self.init_history(curr_bev, forward_augs, seq_ids)
        # reset new sequences
        if start_of_sequence.any():
            self.history_bev[start_of_sequence] = 0
            self.history_bev[start_of_sequence, 0] = curr_bev[start_of_sequence]
            self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]
            self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
            self.history_sweep_time[start_of_sequence] = 0
            # ensure time dim is zeroed
            self.history_sweep_time[start_of_sequence, 1:] = 0

        # update history
        self.roll_history()
        self.history_bev[:, 0] = curr_bev.detach()
        self.history_forward_augs = forward_augs
        self.history_seq_ids = seq_ids

        # compute observability score and groups
        score = self._compute_score(curr_bev, nonempty_prob, self.score_from)
        groups = self._split_indices(score)

        # precompute fused features per window
        history_flat = self.history_bev.reshape(b, self.num_T, c, -1)  # [b, T, c, N]
        fused = curr_bev.reshape(b, c, -1)

        for g_idx, idx in enumerate(groups):
            if idx.numel() == 0:
                continue
            win = self.queue_windows[g_idx] if g_idx < len(self.queue_windows) else self.queue_windows[-1]
            win = min(win, self.num_T)
            hist_slice = history_flat[:, :win]  # [b, win, c, N]
            agg = hist_slice.mean(dim=1)  # [b, c, N]
            # gather-add on selected positions
            gather_idx = idx.unsqueeze(1).expand(-1, c, -1)
            fused.scatter_add_(2, gather_idx, torch.gather(agg, 2, gather_idx))

        fused = fused.view(b, c, z, h, w)
        return fused
