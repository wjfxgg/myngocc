import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models.stcocc.view_transformation.backward_projection.bevformer_utils.multi_scale_deformable_attn_function import \
    MultiScaleDeformableAttnFunction_fp32


class DeformableAttention3D(nn.Module):
    """Lightweight deformable attention over a 3D block (z, y, x merged into 2D grid)."""

    def __init__(self, embed_dims, num_heads=8, num_points=4, im2col_step=64, dropout=0.0):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_points = num_points
        self.im2col_step = im2col_step
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, Z, H, W]
        b, c, z, h, w = x.shape
        hw = h * z  # merge z and y to fit 2D deformable attention kernel
        # [B, Z, H, W, C]
        x_perm = x.permute(0, 2, 3, 4, 1).contiguous()
        x_flat = x_perm.view(b, hw * w, c)

        value = self.value_proj(x_flat).view(b, hw * w, self.num_heads, c // self.num_heads)

        offsets = self.sampling_offsets(x_flat).view(b, hw * w, self.num_heads, self.num_points, 2)
        attn = self.attention_weights(x_flat).view(b, hw * w, self.num_heads, self.num_points)
        attn = F.softmax(attn, dim=-1)

        # reference points on merged (hz, w) grid, normalized to [0,1]
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, hw - 0.5, hw, device=x.device, dtype=x.dtype),
            torch.linspace(0.5, w - 0.5, w, device=x.device, dtype=x.dtype),
            indexing='ij'
        )
        ref = torch.stack((ref_x, ref_y), dim=-1).view(hw * w, 2)  # (L, 2) with order (x, y)
        ref = ref[None].repeat(b, 1, 1)  # (B, L, 2)
        ref = ref / ref.new_tensor([w, hw])[None, None, :]

        sampling_locations = ref[:, :, None, None, :].repeat(1, 1, self.num_heads, self.num_points, 1)
        # normalize offsets
        sampling_locations = sampling_locations + offsets / offsets.new_tensor([w, hw])[None, None, None, None, :]

        spatial_shapes = x.new_tensor([[hw, w]], dtype=torch.long)
        level_start_index = spatial_shapes.new_tensor([0])

        out = MultiScaleDeformableAttnFunction_fp32.apply(
            value, spatial_shapes, level_start_index, sampling_locations, attn, self.im2col_step
        )

        out = self.output_proj(out)
        out = out.view(b, z, h, w, c).permute(0, 4, 1, 2, 3).contiguous()
        return self.dropout(out)


class MultiScaleBlockTPVDeformable(nn.Module):
    """Block-based TPV with intra/inter block deformable attention for memory efficiency."""

    def __init__(
        self,
        embed_dims,
        num_heads=8,
        num_points=4,
        splits=((1, 1, 1), (2, 2, 2), (4, 4, 4), (8, 8, 8)),
        dropout=0.0,
        ffn_ratio=2.0,
        im2col_step=64,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.splits = splits
        self.intra_attn = nn.ModuleList(
            [DeformableAttention3D(embed_dims, num_heads, num_points, im2col_step, dropout) for _ in splits]
        )
        self.inter_attn = nn.ModuleList(
            [DeformableAttention3D(embed_dims, num_heads, num_points, im2col_step, dropout) for _ in splits]
        )
        ffn_channels = int(embed_dims * ffn_ratio)
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(embed_dims, ffn_channels, kernel_size=1),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv3d(ffn_channels, embed_dims, kernel_size=1),
                    nn.Dropout(dropout),
                )
                for _ in splits
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv3d(embed_dims * len(splits), embed_dims, kernel_size=1, bias=False),
            nn.BatchNorm3d(embed_dims),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            return x
        outputs = [self._forward_single_scale(x, idx, split) for idx, split in enumerate(self.splits)]
        fused = torch.cat(outputs, dim=1)
        return self.fuse(fused)

    def _forward_single_scale(self, x: torch.Tensor, idx: int, split_cfg):
        b, c, z, h, w = x.shape
        x_residual = x
        split_x, split_y, split_z = split_cfg
        block_x = math.ceil(w / split_x)
        block_y = math.ceil(h / split_y)
        block_z = math.ceil(z / split_z)

        pad_x = block_x * split_x - w
        pad_y = block_y * split_y - h
        pad_z = block_z * split_z - z
        if pad_x > 0 or pad_y > 0 or pad_z > 0:
            x = F.pad(x, (0, pad_x, 0, pad_y, 0, pad_z))
            z_pad, h_pad, w_pad = z + pad_z, h + pad_y, w + pad_x
        else:
            z_pad, h_pad, w_pad = z, h, w

        # reshape to blocks
        x_block_view = x.view(
            b, c, split_z, block_z, split_y, block_y, split_x, block_x
        ).permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()  # [B, sz, sy, sx, bz, by, bx, C]
        num_blocks = split_z * split_y * split_x
        blocks = x_block_view.view(b * num_blocks, block_z, block_y, block_x, c).permute(0, 4, 1, 2, 3)

        # intra-block deformable attention
        blocks = blocks + self.intra_attn[idx](blocks)
        blocks = blocks + self.ffns[idx](blocks)

        # restore spatial order
        blocks = blocks.permute(0, 2, 3, 4, 1).contiguous().view(
            b, split_z, split_y, split_x, block_z, block_y, block_x, c
        )
        x_intra = blocks.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous().view(b, c, z_pad, h_pad, w_pad)

        # inter-block: pooled block tokens
        block_tokens = blocks.mean(dim=(4, 5, 6)).permute(0, 4, 1, 2, 3)  # [B, C, sz, sy, sx]
        block_tokens = block_tokens + self.inter_attn[idx](block_tokens)
        block_tokens = block_tokens + self.ffns[idx](block_tokens)

        # broadcast block features back to full grid
        x_inter = block_tokens.repeat_interleave(block_z, dim=2).repeat_interleave(block_y, dim=3).repeat_interleave(
            block_x, dim=4
        )
        x_inter = x_inter[:, :, :z_pad, :h_pad, :w_pad]

        x_out = (x_intra + x_inter)[:, :, :z, :h, :w]
        return x_out + x_residual
