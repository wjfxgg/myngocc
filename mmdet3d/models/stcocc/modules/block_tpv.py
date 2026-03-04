import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleBlockTPV(nn.Module):
    """Block-based TPV aggregation with multi-scale partitioning."""

    def __init__(
        self,
        embed_dims,
        num_heads=8,
        splits=((1, 1, 1), (2, 2, 2), (4, 4, 4), (8, 8, 8)),
        dropout=0.0,
        ffn_ratio=2.0,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.splits = splits
        self.intra_attn = nn.ModuleList(
            [nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout) for _ in splits]
        )
        self.inter_attn = nn.ModuleList(
            [nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout) for _ in splits]
        )
        self.intra_norm = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in splits])
        self.inter_norm = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in splits])
        ffn_channels = int(embed_dims * ffn_ratio)
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dims, ffn_channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ffn_channels, embed_dims),
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
        """Apply intra-block self-attention and inter-block cross-attention at one scale."""
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

        x_perm = x.permute(0, 2, 3, 4, 1).contiguous()
        x_blocks = (
            x_perm.view(
                b,
                split_z,
                block_z,
                split_y,
                block_y,
                split_x,
                block_x,
                c,
            )
            .permute(0, 1, 3, 5, 2, 4, 6, 7)
            .contiguous()
        )
        num_blocks = split_x * split_y * split_z
        block_size = block_x * block_y * block_z
        tokens = x_blocks.view(b, num_blocks, block_size, c)
        tokens = tokens.view(b * num_blocks, block_size, c).transpose(0, 1)  # (block_size, b*num_blocks, c)

        intra_out, _ = self.intra_attn[idx](tokens, tokens, tokens)
        tokens = self.intra_norm[idx](tokens + intra_out)

        tokens_block = tokens.transpose(0, 1)  # (b*num_blocks, block_size, c)
        block_repr = tokens_block.mean(1).view(b, num_blocks, c).transpose(0, 1)  # (num_blocks, b, c)
        inter_out, _ = self.inter_attn[idx](block_repr, block_repr, block_repr)
        block_repr = self.inter_norm[idx](block_repr + inter_out)

        block_repr = block_repr.transpose(0, 1).contiguous().view(b, num_blocks, 1, c)
        block_repr = block_repr.repeat(1, 1, block_size, 1)
        tokens_enhanced = tokens_block + block_repr
        tokens_enhanced = tokens_enhanced + self.ffns[idx](tokens_enhanced)

        tokens_enhanced = tokens_enhanced.view(b, num_blocks, block_size, c)
        x_out = tokens_enhanced.view(
            b, split_z, split_y, split_x, block_z, block_y, block_x, c
        ).permute(0, 1, 4, 2, 5, 3, 6, 7)
        x_out = x_out.contiguous().view(b, z_pad, h_pad, w_pad, c)
        x_out = x_out[:, :z, :h, :w, :].permute(0, 4, 1, 2, 3).contiguous()
        return x_out + x_residual
