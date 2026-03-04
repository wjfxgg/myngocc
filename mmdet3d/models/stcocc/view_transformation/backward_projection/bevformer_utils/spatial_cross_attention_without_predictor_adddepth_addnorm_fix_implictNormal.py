import warnings
import math

import torch
import torch.nn as nn

from mmcv.utils import ext_loader
from mmcv.runner import force_fp32
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.runner.base_module import BaseModule
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.transformer import build_attention
from timm.models.layers import DropPath
import torch.utils.checkpoint as cp
import numpy as np
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])

def custom_build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()

    raise NotImplementedError(f'build_act_layer does not support {act_layer}')

def position_decode(pts_xyz, pc_range=None):
    pts_xyz = pts_xyz.clone()
    pts_xyz[..., 0:1] = pts_xyz[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
    pts_xyz[..., 1:2] = pts_xyz[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
    pts_xyz[..., 2:3] = pts_xyz[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]
    return pts_xyz

def custom_build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)

class MLPLayer(nn.Module):
    r""" MLP layer of InternImage
    Args:
        in_features (int): number of input features
        hidden_features (int): number of hidden features
        out_features (int): number of output features
        act_layer (str): activation layer
        drop (float): dropout rate
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer='GELU',
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = custom_build_act_layer(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
@ATTENTION.register_module()
class OA_SpatialCrossAttentionWithoutPredictor_adddepth_addnorm_fix_A_implict(BaseModule):
    """An attention module used in BEVFormer without predictor.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 deformable_attention=dict(
                     type='OA_MSDeformableAttention3DWithoutPredictor',
                     embed_dims=256,
                     num_levels=4),
                 layer_scale=1.0,
                 
                 dbound=None,
                 num_points=8,
                 grid_range_scale=1.0,
                 offset_single_level=True,
                 mlp_ratio=4.,
                 act_layer='GELU',
                 drop=0.,
                 drop_path=0.1,
                 with_cp=False,
                 sigma=2,
                 
                 num_heads = 8,
                 need_center_grid =False,
                 num_levels = 1,
                 **kwargs
                 ):
        super(OA_SpatialCrossAttentionWithoutPredictor_adddepth_addnorm_fix_A_implict, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        self.init_cfg = init_cfg
        self.pc_range = pc_range
        self.offset_single_level = offset_single_level
        self.dropout = nn.Dropout(dropout)

        self.fp16_enabled = False
        # self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.dbound = dbound

        self.kernel_size = 3
        self.dilation = 1
        self.sigma = sigma
        
        self.num_points_in_pillar = num_points
        self.im2col_step = 64
        self.embed_dims = embed_dims
        self.num_levels = 1
        self.num_heads = num_heads
        self.num_cams = 6
        self.attn_norm_sigmoid = True
        self.with_ffn = True
        self.point_dim = 3
        self.need_center_grid = (self.kernel_size % 2 == 0) & (self.dilation % 2 ==0) & need_center_grid
        self.per_ref_points = num_points
        self.grid_range_scale = nn.Parameter(grid_range_scale * torch.ones(1), requires_grad=True)
        per_query_points = self.per_ref_points

        if self.offset_single_level:
            self.offset = nn.Linear(
                embed_dims, num_heads * 1 * per_query_points * self.point_dim)
        else:
            self.offset = nn.Linear(
                embed_dims, num_heads * num_levels * per_query_points * self.point_dim)
        self.weight = nn.Linear(embed_dims,
            num_heads * num_cams * num_levels * per_query_points)

        self.dir_vector_mlp = nn.Sequential(
            nn.Linear(20*2*self.per_ref_points, embed_dims),  # 20是bin的数量
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims)
        )
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.norm1 = custom_build_norm_layer(embed_dims, 'LN')

        if self.with_ffn:
            self.mlp = MLPLayer(in_features=embed_dims,
                                hidden_features=int(embed_dims * mlp_ratio),
                                act_layer=act_layer,
                                drop=drop)
            self.norm2 = custom_build_norm_layer(embed_dims, 'LN')

        self.gamma1 = nn.Parameter(layer_scale * torch.ones(embed_dims),
                                    requires_grad=True)
        if self.with_ffn:
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(embed_dims),
                                    requires_grad=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()

        self.batch_first = batch_first
        self.use_checkpoint = with_cp

        self.batch_first = batch_first

        self.init_weight()

        

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        """Default initialization for Parameters of Module."""

        # similar init concept as in deformabel-detr
        grid = self.generate_dilation_grids(self.kernel_size, self.kernel_size, self.dilation, self.dilation, 'cpu') # head 2  9 2
        assert (grid.size(0) == self.num_heads) & (self.embed_dims % self.num_heads == 0)

        # grid = grid.unsqueeze(1).repeat(1, self.per_ref_points, 1)# # 9 2 -> head  ref_num  2 

        # for i in range(self.per_ref_points):
        #     grid[:, i, ...] *= (i + 1)  # 这里在执行的时候是1 这里感觉有点像是要多尺度 或者说是用来表示z轴
        # 为了适配viewformer 所以这样
        grid = grid.unsqueeze(1)# # 9 2 -> head  1  2 

        # for i in range(1):
        #     grid[:, i, ...] *= (i + 1)  # 这里在执行的时候是1 这里感觉有点像是要多尺度 或者说是用来表示z轴
        # grid /= self.per_ref_points # 进行归一化
        self.grid = grid 

        if self.point_dim == 3:
            self.grid = torch.cat([torch.zeros_like(self.grid[..., :1]), self.grid], dim=-1) #head ref_num 3 (0,y,z)

        constant_init(self.offset, 0., 0.)
        constant_init(self.weight, 0., 0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def generate_dilation_grids(self, kernel_h, kernel_w, dilation_w, dilation_h, device):
        x, y = torch.meshgrid(
            torch.linspace(
                -((dilation_w * (kernel_w - 1)) // 2),  # -(1*(3-1)//2) -1
                -((dilation_w * (kernel_w - 1)) // 2) + # -1 +2*1   1
                (kernel_w - 1) * dilation_w, kernel_w,     # 3
                dtype=torch.float32,
                device=device),
            torch.linspace(
                -((dilation_h * (kernel_h - 1)) // 2),
                -((dilation_h * (kernel_h - 1)) // 2) +
                (kernel_h - 1) * dilation_h, kernel_h,
                dtype=torch.float32,
                device=device)) # [3,3]
        grid = torch.stack([x, y], -1).reshape(-1, 2) # [3,3,2] ->[9,2] head 2

        if self.need_center_grid:
            grid = torch.cat([grid, torch.zeros_like(grid[0:1, :])], dim=0)

        if not self.need_center_grid:
            grid = torch.cat([grid[:4],grid[-4:]])
        return grid



    @force_fp32()
    def point_sampling(self, reference_points,  img_metas):
        ego2lidar = []
        lidar2img = []
        for img_meta in img_metas:
            ego2lidar.append(img_meta['ego2lidar'])
            lidar2img.append(img_meta['lidar2img'])
        # ego2lidar = np.asarray(ego2lidar)
        ego2lidar = torch.cat(ego2lidar).to(reference_points) # (B, 4, 4)
        # lidar2img = np.asarray(lidar2img)
        lidar2img = torch.stack(lidar2img).to(reference_points) # (B, N, 4, 4)
        ego2img = lidar2img @ ego2lidar[:, None, :, :]

        num_cam = ego2img.size(1)
        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1)
        reference_points = reference_points.unsqueeze(1).repeat(1, num_cam, 1, 1, 1, 1, 1)

        reference_points_cam = torch.matmul(ego2img.to(torch.float32)[:, :, None, None, None, None, :], reference_points.to(torch.float32).unsqueeze(-1)).squeeze(-1)

        eps = 1e-5
        proj_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        # reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        # reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
        # TODO 这里暂时使用硬编码
        reference_points_cam[..., 0] /= 704
        reference_points_cam[..., 1] /= 256

        proj_mask = (proj_mask & (reference_points_cam[..., 1:2] > 0.0)
                     & (reference_points_cam[..., 1:2] < 1.0)
                     & (reference_points_cam[..., 0:1] < 1.0)
                     & (reference_points_cam[..., 0:1] > 0.0))

        return reference_points_cam, proj_mask.squeeze(-1)

    # 我现在是图像坐标系下面的法向量的已经有了，然后ego坐标系下面的法向量坐标也有了，现在就是需要考虑应该是在什么坐标系下面考虑问题
    def tangent_plane_sampling(self, reference_points, offset, img_metas): # [bs zyx 3]  [bs zyx head num_L 1 3]
        reference_points = reference_points.flatten(1,2)  # bs zyx 3
        offset =  offset.permute(0,4,1,2,3,5).flatten(1,2).unsqueeze(-2) # bs zyx head unm_L 1 3
        grid = self.grid.type_as(reference_points) * self.grid_range_scale # trans to pc_range 缩放系数 这里实际上是1.0 [9,1,3] <head num_ref 3> # 8 2 3

        local_pts = grid[None, None, :, None, ...] + offset  # bs zyx head num_L 1 3
        if self.point_dim == 2:
            local_pts = torch.cat([torch.zeros_like(local_pts[..., :1]), local_pts], dim=-1)

        reference_points = position_decode(reference_points, self.pc_range) #把归一化的数据变换到真实的空间
        # the query-specific view angle mentioned in paper
        azimuth_angle = torch.atan2(reference_points[..., 1], reference_points[..., 0]) #这个是y x 根据坐标位置变换到角度  #bs zyx 1

        rot_sin, rot_cos = torch.sin(azimuth_angle), torch.cos(azimuth_angle) #旋转角度
        zeros, ones = torch.zeros_like(azimuth_angle), torch.ones_like(azimuth_angle) #凑成几个0向量 和1向量
        rot_matrix = torch.stack([rot_cos, -rot_sin, zeros,
                                  rot_sin,  rot_cos, zeros,
                                  zeros,    zeros,   ones], dim=-1).reshape(*reference_points.shape[:-1], 3, 3) # bs zyx 3 3

        local_pts = torch.matmul(rot_matrix[:, :, None, None, None, ...], local_pts.unsqueeze(-1)).squeeze(-1) #这个相当于是局部小区域的偏移
        reference_points = reference_points[:, :, None, None, None, :] + local_pts #最终的参照点=原始参照点+局部小偏移

        reference_points_cam, proj_mask = self.point_sampling(reference_points, img_metas) #然后再根据这个去图像里面采样  这样确实合理 而且这个才是我想要的

        if self.offset_single_level:
            reference_points_cam = reference_points_cam.repeat(1, 1, 1, 1, self.num_levels, 1, 1)  #num_levels 这个是多尺度的维度
            proj_mask = proj_mask.repeat(1, 1, 1, 1, self.num_levels, 1)
        bs,n,zhw, head, num_L, ref_point, _ =reference_points_cam.shape
        return reference_points_cam.view(bs, n, self.per_ref_points,-1, head, num_L, ref_point, 2).permute(0,1,3,4,5,2,6,7).squeeze(-2), proj_mask.view(bs, n, self.per_ref_points,-1, head, num_L, ref_point).permute(0,1,3,4,5,2,6).squeeze(-1),local_pts  # bs n zhw head num_L ref_point 2; 

    # 我现在是图像坐标系下面的法向量的已经有了，然后ego坐标系下面的法向量坐标也有了，现在就是需要考虑应该是在什么坐标系下面考虑问题
    def tangent_plane_sampling_originA(self, reference_points, offset, img_metas): # [bs zyx 3]  [bs zyx head num_L 1 3]
        reference_points = reference_points.flatten(1,2)  # bs zyx 3
        offset =  offset.permute(0,4,1,2,3,5).flatten(1,2).unsqueeze(-2) # bs zyx head unm_L 1 3
        grid = self.grid.type_as(reference_points) * self.grid_range_scale # trans to pc_range 缩放系数 这里实际上是1.0 [9,1,3] <head num_ref 3> # 8 2 3

        local_pts = grid[None, None, :, None, ...] + offset  # bs zyx head num_L 1 3
        if self.point_dim == 2:
            local_pts = torch.cat([torch.zeros_like(local_pts[..., :1]), local_pts], dim=-1)

        reference_points = position_decode(reference_points, self.pc_range) #把归一化的数据变换到真实的空间
        # the query-specific view angle mentioned in paper
        azimuth_angle = torch.atan2(reference_points[..., 1], reference_points[..., 0]) #这个是y x 根据坐标位置变换到角度  #bs zyx 1

        rot_sin, rot_cos = torch.sin(azimuth_angle), torch.cos(azimuth_angle) #旋转角度
        zeros, ones = torch.zeros_like(azimuth_angle), torch.ones_like(azimuth_angle) #凑成几个0向量 和1向量
        rot_matrix = torch.stack([rot_cos, -rot_sin, zeros,
                                  rot_sin,  rot_cos, zeros,
                                  zeros,    zeros,   ones], dim=-1).reshape(*reference_points.shape[:-1], 3, 3) # bs zyx 3 3

        local_pts = torch.matmul(rot_matrix[:, :, None, None, None, ...], local_pts.unsqueeze(-1)).squeeze(-1) #这个相当于是局部小区域的偏移
        reference_points = reference_points[:, :, None, None, None, :] + local_pts #最终的参照点=原始参照点+局部小偏移

        reference_points_cam, proj_mask = self.point_sampling(reference_points, img_metas) #然后再根据这个去图像里面采样  这样确实合理 而且这个才是我想要的

        if self.offset_single_level:
            reference_points_cam = reference_points_cam.repeat(1, 1, 1, 1, self.num_levels, 1, 1)  #num_levels 这个是多尺度的维度
            proj_mask = proj_mask.repeat(1, 1, 1, 1, self.num_levels, 1)

        return reference_points_cam, proj_mask

    def attention(self,
                  query, # bs zyx c
                  value, # bs N (h1w1+h2w2+...) num_head c
                  reference_points_cam, # [bs n zyx head_num_level per_query_point 2]
                  weights, # bs zyx head N num_levels ref_num
                  proj_mask, # bs zyx head N num_levels ref_num
                  spatial_shapes,
                  level_start_index,
                  ):
        num_query = query.size(1)
        bs, num_cam, num_value = value.shape[:3]
        num_all_points = weights.size(-1)

        slots = torch.zeros_like(query)

        # (bs, num_query, num_head, num_cam, num_level, num_p) 
        # --> (bs, num_cam, num_query, num_head, num_level, num_p)
        weights = weights.permute(0, 3, 1, 2, 4, 5).contiguous()

        # save memory trick, similar as bevformer_occ
        indexes = [[] for _ in range(bs)]
        max_len = 0
        for i in range(bs):
            for j in range(num_cam):
                index_query_per_img = proj_mask[i, j].flatten(1).sum(-1).nonzero().squeeze(-1)
                indexes[i].append(index_query_per_img)
                max_len = max(max_len, index_query_per_img.numel())

        queries_rebatch = query.new_zeros(
            [bs, self.num_cams, max_len, self.embed_dims])
        reference_points_cam_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, self.num_heads, self.num_levels, num_all_points, 2])
        weights_rebatch = weights.new_zeros(
            [bs, self.num_cams, max_len, self.num_heads, self.num_levels, num_all_points])

        for i in range(bs):
            for j in range(num_cam):  
                index_query_per_img = indexes[i][j]
                curr_numel = index_query_per_img.numel()
                queries_rebatch[i, j, :curr_numel] = query[i, index_query_per_img]
                reference_points_cam_rebatch[i, j, :curr_numel] = reference_points_cam[i, j, index_query_per_img]
                weights_rebatch[i, j, :curr_numel] = weights[i, j, index_query_per_img]

        value = value.view(bs*num_cam, num_value, self.num_heads, -1)
        sampling_locations = reference_points_cam_rebatch.view(bs*num_cam, max_len, self.num_heads, self.num_levels, num_all_points, 2)
        attention_weights = weights_rebatch.reshape(bs*num_cam, max_len, self.num_heads, self.num_levels, num_all_points)

        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction_fp32.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = output.view(bs, num_cam, max_len, -1)
        for i in range(bs):
            for j in range(num_cam):
                index_query_per_img = indexes[i][j]
                slots[i, index_query_per_img] += output[i, j, :len(index_query_per_img)]
        return slots

    @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def forward(self,
                query, #bs yx c
                key,
                value,
                residual=None,
                query_pos=None,
                spatial_shapes=None,
                reference_points=None, #bs,z,yx,3
                reference_points_cam=None,
                reference_points_depth=None,
                level_start_index=None,
                bev_query_depth=None,
                pred_img_depth=None,
                depth_bound=None,
                nonempty_voxel_logits=None,
                bev_mask=None,
                **kwargs):
            if self.use_checkpoint and self.training:
                query = cp.checkpoint(self.inner_forward, query, key, value, residual, query_pos, spatial_shapes, reference_points, reference_points_cam, reference_points_depth, level_start_index, bev_query_depth, pred_img_depth, depth_bound, nonempty_voxel_logits, bev_mask, **kwargs)
            else:
                query = self.inner_forward(query, key, value, residual, query_pos, spatial_shapes, reference_points, reference_points_cam, reference_points_depth, level_start_index, bev_query_depth, pred_img_depth, depth_bound, nonempty_voxel_logits, bev_mask, **kwargs)
            return query
    
    @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def inner_forward(self,
                query, #bs yx c
                key,
                value,
                residual=None,
                query_pos=None,
                spatial_shapes=None,
                reference_points=None, #bs,z,yx,3
                reference_points_cam=None,
                reference_points_depth=None,
                level_start_index=None,
                bev_query_depth=None,
                pred_img_depth=None,
                depth_bound=None,
                nonempty_voxel_logits=None,
                bev_mask=None,
                **kwargs):
            if residual is None:
                residual = query
            output = self.forward_layer(query, key, value, residual, query_pos, spatial_shapes, reference_points, reference_points_cam, reference_points_depth, level_start_index, bev_query_depth, pred_img_depth, depth_bound, nonempty_voxel_logits, bev_mask, **kwargs) 
            query = residual +self.drop_path(self.gamma1*self.norm1(output))
            if self.with_ffn:
                query = query +self.drop_path(self.gamma2*self.norm2(self.mlp(query)))
            return query
    
    @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def forward_layer(self,
                query, #bs yx c
                key,
                value, # N (hw) bs C
                residual=None,
                query_pos=None,
                spatial_shapes=None,
                reference_points=None, #bs,z,yx,3
                reference_points_cam=None,
                reference_points_depth=None,
                level_start_index=None,
                bev_query_depth=None,
                pred_img_depth=None,
                depth_bound=None,
                nonempty_voxel_logits=None,
                bev_mask=None,
                img_metas=None,
                ego_dir_vector_3d_xyz_i=None,
                **kwargs):
        """Forward Function of Detr3DCrossAtten without predictor.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for  `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        origin_reference_points_cam = reference_points_cam # n bs yx z 2
        
        # input check
        if key is None:
            key = query
        if value is None:
            value = key
        if residual is None:
            # inp_residual = query
            slots = torch.zeros_like(query)
        slots = torch.zeros_like(query)
        
        if query_pos is not None:
            query = query + query_pos

        # preprocess
        bs, num_query, _ = query.size()
        num_cam, D = reference_points_cam.size(0), reference_points_cam.size(3)         # reference_points_cam: [num_cam, bs, num_query, D, 2]
        
        # 不再使用nonempty_voxel_logits
        # if nonempty_voxel_logits is None:
        #     nonempty_voxel_logits = torch.ones(bs, num_query, D, 1, device=query.device)
        # else:
        #     nonempty_voxel_logits = nonempty_voxel_logits.reshape(bs, num_query, D, 1)
        

        #在这里修改 加入viewformer的对应的函数
        per_query_points = self.per_ref_points

        value =self.value_proj(value)  # N hw bs c
        # per_query_point = self.per_ref_points
        sampling_offsets = self.offset(query).view(
            bs, num_query, self.num_heads, -1, per_query_points, self.point_dim)  # 输入：bs yx c 输出L:bs yx head(9) num_L per_query_point(2 z) 3

        reference_points_cam, proj_mask, ref_points_3d = self.tangent_plane_sampling(reference_points, sampling_offsets, img_metas) # reference_points_cam.shape [bs,n,yx,head,num_level,per_query_point,2]  proj_mask.shape[bs,n,yx,head,num_level,per_query_point]
        
        # A = torch.cat([sampling_offsets,torch.zeros_like(sampling_offsets[...,:1])],dim=-1) # bs,num_query,num_heads,num_bev_queue,num_levels,num_points,3(x,y,z)
        # B = ego_dir_vector_3d_xyz_i.flatten(2).permute(0,2,1)[:,:,None,None,None,:] #bs,num_query,num_points,3(x,y,z)
        # unit_A = ref_points_3d /  torch.linalg.vector_norm(ref_points_3d, dim=-1)[...,None]
        # sin_squard = (1-(unit_A*B).sum(-1)**2).view(bs,per_query_points,-1,self.num_heads,1,1).squeeze(-1).permute(0,2,3,4,1) # 2,625,8,1,1,2
        
        pred_img_depth = pred_img_depth.view(bs * num_cam, -1, spatial_shapes[0][0], spatial_shapes[0][1]) # (bs n)  D h w
        pred_img_depth = pred_img_depth.flatten(2).permute(0, 2, 1)                     # [(bs n), h*w, C]
        
        weights = self.weight(query).view(bs, num_query, self.num_heads, self.num_cams, self.num_levels, per_query_points)
        weights = weights.masked_fill(~proj_mask.permute(0, 2, 3, 1, 4, 5), float("-inf"))

        if self.attn_norm_sigmoid:
            weights = weights.view(bs, num_query, self.num_heads, -1).sigmoid()
        else:
            weights = weights.view(bs, num_query, self.num_heads, -1).softmax(-1)
        weights = torch.nan_to_num(weights)
        weights = weights.view(bs, num_query,self.num_heads, self.num_cams, self.num_levels, -1)

        num_all_points = weights.size(-1)

        # (bs, num_query, num_head, num_cam, num_level, num_p) 
        # --> (bs, num_cam, num_query, num_head, num_level, num_p)
        weights = weights.permute(0, 3, 1, 2, 4, 5).contiguous()


        # 不再使用sampling_rate和occlusion_mask，直接处理所有点
        # 初始化bev_mask为全True
        if bev_mask is None:
            bev_mask = torch.ones(num_cam, bs, num_query, D, device=query.device, dtype=torch.bool)


        # origin 使用bev mask
        # Use bev_mask to check if the reference_points_cam is valid
        # bev_mask: [num_cam, bs, h*w, num_points_in_pillar]
        # indexes = [[] for _ in range(bs)]
        # max_len = 0
        # for j in range(bs):
        #     for i, mask_per_img in enumerate(bev_mask):
        #         index_query_per_img = mask_per_img[j].sum(-1).nonzero().squeeze(-1)
        #         if len(index_query_per_img) == 0:
        #             index_query_per_img = torch.arange(num_query, device=query.device)[:1]  # 确保至少有一个点
        #         indexes[j].append(index_query_per_img)
        #         # for batch operation, we need to pad the indexes to the same length
        #         max_len = max(max_len, len(index_query_per_img))


        #TODO 这里还有一些技巧 可以再节省很大一部分显存 等会来实现这里  这里也算是两种分支和情况


        # save memory trick, similar as bevformer_occ
        indexes = [[] for _ in range(bs)]
        max_len = 0
        for i in range(bs):
            for j in range(num_cam):
                index_query_per_img = proj_mask[i, j].flatten(1).sum(-1).nonzero().squeeze(-1)
                if len(index_query_per_img) == 0:
                    index_query_per_img = torch.arange(num_query, device=query.device)[:1]  # 确保至少有一个点
                indexes[i].append(index_query_per_img)
                max_len = max(max_len, index_query_per_img.numel())

        # each camera only interacts with its corresponding BEV queries. This step can greatly save GPU memory.
        queries_rebatch = query.new_zeros([bs, self.num_cams, max_len, self.embed_dims])
        origin_reference_points_cam_rebatch = origin_reference_points_cam.new_zeros([bs, self.num_cams, max_len, D, 2])
        reference_points_cam_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, self.num_heads, self.num_levels, num_all_points, 2])
        reference_points_depth_rebatch = reference_points_depth.new_zeros([bs, self.num_cams, max_len, D, 1])
        weights_rebatch = weights.new_zeros(
            [bs, self.num_cams, max_len, self.num_heads, self.num_levels, num_all_points])
        # sin_squard_rebatch = sin_squard.new_zeros([bs, self.num_cams, max_len, self.num_heads, self.num_levels, num_all_points])
        

        
        # reference_points_occlusion_rebatch = nonempty_voxel_logits.new_zeros([bs, max_len, D, 1])
        # reference_points_3d_rebatch = reference_points.new_zeros([bs, max_len, D, 3])
        
        # TODO origin版本的实现
        # # get non mask query and reference points
        # for j in range(bs):
        #     for i, (reference_points_per_img, reference_points_depth_per_img) in enumerate(zip(reference_points_cam, reference_points_depth)):
        #         index_query_per_img = indexes[j][i]
        #         queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
        #         reference_points_cam_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]
        #         reference_points_depth_rebatch[j, i, :len(index_query_per_img)] = reference_points_depth_per_img[j, index_query_per_img]
        #         # reference_points_occlusion_rebatch[j, :len(index_query_per_img)] = nonempty_voxel_logits[j, index_query_per_img]
        # # repeat along the num_cams
        # # reference_points_occlusion_rebatch = reference_points_occlusion_rebatch.unsqueeze(1).repeat(1, self.num_cams, 1, 1, 1)

        for i in range(bs):
            for j in range(num_cam):  
                index_query_per_img = indexes[i][j]
                curr_numel = index_query_per_img.numel()
                queries_rebatch[i, j, :curr_numel] = query[i, index_query_per_img]
                reference_points_cam_rebatch[i, j, :curr_numel] = reference_points_cam[i, j, index_query_per_img]
                origin_reference_points_cam_rebatch[i, j, :curr_numel] = origin_reference_points_cam[j, i, index_query_per_img]
                weights_rebatch[i, j, :curr_numel] = weights[i, j, index_query_per_img]
                reference_points_depth_rebatch[i, j, :curr_numel] = reference_points_depth[j, i, index_query_per_img]
                # sin_squard_rebatch[i, j, :curr_numel] = sin_squard[i, index_query_per_img]
        
        num_cams, l, bs, embed_dims = key.shape
        n,num_value,bs,c = value.shape
        value = value.permute(2,0,1,3).contiguous().view(bs*num_cam, num_value, self.num_heads, -1) #n hw bs c 

        sampling_locations = reference_points_cam_rebatch.view(bs*num_cam, max_len, self.num_heads, self.num_levels, num_all_points, 2)
        attention_weights = weights_rebatch.reshape(bs*num_cam, max_len, self.num_heads, self.num_levels, num_all_points)

        



        reference_points_depth_rebatch = reference_points_depth_rebatch.view(bs * self.num_cams, max_len, D)
        # 改造下面的原来的对应的代码
        depth_reference_points = origin_reference_points_cam_rebatch.reshape(bs*num_cam, max_len*D, 1, 1, 1, 2).contiguous() # (bs n) len_max head num_L ref_point 2
        depth_attention_weights = torch.ones_like(depth_reference_points[..., 0]).contiguous() # (bs n) len_max head num_L ref_point
        pred_img_depth = pred_img_depth.unsqueeze(2).contiguous()
        
        depth_output = MultiScaleDeformableAttnFunction_fp32.apply(pred_img_depth, spatial_shapes,level_start_index,depth_reference_points,depth_attention_weights, self.im2col_step)
        depth_output = depth_output.reshape(bs*n, max_len, D, -1)   # [bs*num_cam, num_query, num_Z_anchors, C]
        depth_output = depth_output.softmax(-1).argmax(-1) * depth_bound[2] + depth_bound[0]
        depth_weights = torch.exp(
            -torch.square(
                torch.min(torch.abs(reference_points_depth_rebatch - (depth_output - depth_bound[2])), torch.abs(reference_points_depth_rebatch - (depth_output + depth_bound[2])))
            ) / 2 * (self.sigma ** 2)
        )
        # with torch.no_grad():
        # cam_dir_vector = cam_dir_vector.flatten(2).permute(0,2,1).unsqueeze(2).contiguous()
        # print("#"*50)
        # print(f"cam_dir_vector {cam_dir_vector.shape} sampling_locations {sampling_locations.shape}")
        # print("#"*50)
        # cam_dir_sampling_locations = MultiScaleDeformableAttnFunction_fp32.apply(
        #     cam_dir_vector, 
        #     spatial_shapes, 
        #     level_start_index, 
        #     sampling_locations.flatten(1,2).unsqueeze(3).clone(), 
        #     torch.ones_like(attention_weights).flatten(1,-1).unsqueeze(2).unsqueeze(3), 
        #     self.im2col_step
        # ).view(*list(attention_weights.shape),3)  # cam_dir_sampling_locations shape: torch.Size([18, 171, 8, 1, 2, 3])
        # print(f"cam_dir_sampling_locations {cam_dir_sampling_locations.shape} ")
        # cam_dir_depth_reference_point = MultiScaleDeformableAttnFunction_fp32.apply(
        #     cam_dir_vector, 
        #     spatial_shapes, 
        #     level_start_index, 
        #     depth_reference_points, 
        #     depth_attention_weights, 
        #     self.im2col_step
        # ).view(*list(depth_weights.shape),3)
        # print(f"cam_dir_depth_reference_point {cam_dir_depth_reference_point.shape} ")
        #     # norm_weight = (cam_dir_sampling_locations*cam_dir_depth_reference_point[:,:,None,None]).sum(-1)
        # # torch.cuda.empty_cache()
        # attention_weights = attention_weights * depth_weights[:, :, None, None]
        # # attention_weights = attention_weights * depth_weights[:, :, None, None]*norm_weight

        # # 直接使用attention_weights进行计算
        # output = MultiScaleDeformableAttnFunction_fp32.apply(
        #     value, 
        #     spatial_shapes, 
        #     level_start_index, 
        #     sampling_locations, 
        #     attention_weights, 
        #     self.im2col_step
        # )

        # if not self.batch_first:
        #     output = output.permute(1, 0, 2)
        # return output

        # TODO 开启或者关闭depth_weight
        # attention_weights = attention_weights * depth_weights[:, :, None, None]*(sin_squard_rebatch.flatten(0,1))
        
        
        #origin
        # attention_weights = attention_weights *(sin_squard_rebatch.flatten(0,1))

        # 直接使用attention_weights进行计算
        queries = MultiScaleDeformableAttnFunction_fp32.apply(
            value, 
            spatial_shapes, 
            level_start_index, 
            sampling_locations, 
            attention_weights, 
            self.im2col_step
        )

        if not self.batch_first:
            queries = queries.permute(1, 0, 2)
        # return output








        # queries = self.deformable_attention(
        #     query=queries_rebatch.view(bs * self.num_cams, max_len, self.embed_dims),
        #     key=key.permute(2, 0, 1, 3).reshape(bs * self.num_cams, l, self.embed_dims),
        #     value=value.permute(2, 0, 1, 3).reshape(bs * self.num_cams, l, self.embed_dims),
        #     reference_points=reference_points_cam_rebatch.view(bs * self.num_cams, max_len, D, 2),
        #     reference_points_depth=reference_points_depth_rebatch.view(bs * self.num_cams, max_len, D),
        #     # reference_points_occlusion=reference_points_occlusion_rebatch.view(bs * self.num_cams, max_len, D),
        #     # params
        #     spatial_shapes=spatial_shapes,
        #     level_start_index=level_start_index,
        #     pred_img_depth=pred_img_depth,
        #     depth_bound=depth_bound,
        #     **kwargs
        # )
        queries = queries.view(bs, self.num_cams, max_len, self.embed_dims)

        # aug the query
        for j in range(bs):
            for i in range(num_cams):
                index_query_per_img = indexes[j][i]
                slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]

        # output
        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        # 处理3D方向特征
        # if ego_dir_vector_3d_xyz_i is not None:
        #     print(f"ego_dir_vector_3d_xyz_i shape: {ego_dir_vector_3d_xyz_i.shape}")
        #     # 动态处理不同维度的情况
        #     if len(ego_dir_vector_3d_xyz_i.shape) == 5:
        #         # 形状为[bs, 20, z, h, w]，进行平均池化
        #         bs, num_bins, z, h, w = ego_dir_vector_3d_xyz_i.shape
        #         ego_dir_vector_3d_xyz_i = ego_dir_vector_3d_xyz_i.mean(dim=[2, 3, 4])  # [bs, 20]
        #     elif len(ego_dir_vector_3d_xyz_i.shape) == 2:
        #         # 形状已经是[bs, 20]
        #         pass
        #     else:
        #         # 其他形状情况，直接使用
        #         ego_dir_vector_3d_xyz_i = ego_dir_vector_3d_xyz_i.mean(dim=1)  # 对bin维度求平均
            
        #     # 通过MLP转换方向特征
        #     dir_features = self.dir_vector_mlp(ego_dir_vector_3d_xyz_i)
            
        #     # 扩展维度以匹配slots的形状
        #     dir_features = dir_features.unsqueeze(1).unsqueeze(1)  # [bs, 1, 1, embed_dims]
            
        #     # 将方向特征与slots相乘
        #     slots = slots * dir_features

        if ego_dir_vector_3d_xyz_i is not None:
            # 打印形状以调试
            # print(f"ego_dir_vector_3d_xyz_i shape: {ego_dir_vector_3d_xyz_i.shape}")
            
            # 根据实际形状进行处理
            # if ego_dir_vector_3d_xyz_i.dim() == 5:
            #     # 形状: [bs, num_bins, z, h, w]
            #     bs = ego_dir_vector_3d_xyz_i.shape[0]
            #     # 对z, h, w维度进行平均池化，得到每个query的方向特征
            #     dir_features = ego_dir_vector_3d_xyz_i.mean(dim=[2, 3, 4])  # [bs, num_bins]
            # else:
            #     # 其他形状，直接使用
            #     dir_features = ego_dir_vector_3d_xyz_i
            dir_features=ego_dir_vector_3d_xyz_i.flatten(3).permute(0,3,1,2).flatten(2)
            # 使用MLP进行特征变换
            dir_features = self.dir_vector_mlp(dir_features)  # [bs, embed_dims]
            
            # 扩展维度以匹配output形状
            # dir_features = dir_features.unsqueeze(1)  # [bs, 1, embed_dims]
            
            # 与output相乘
            slots = slots * dir_features

        # return self.dropout(slots) + inp_residual
        return self.dropout(slots)

@ATTENTION.register_module()
class OA_MSDeformableAttention3DWithoutPredictor_adddepth_addnorm_fix_A_implict(BaseModule):
    """An attention module used in BEVFormer based on Deformable-Detr without predictor.
    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=8,
                 im2col_step=64,
                 dropout=0.1,
                 sigma=2,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sigma = sigma

        # 移除reference_points_occlusion相关的维度
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        if self.output_proj is not None:
            xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                pred_img_depth=None,
                reference_points_depth=None,
                depth_bound=None,
                reference_points_occlusion=None,
                cam_dir_vector=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention without predictor.
        Args:
            query (Tensor): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key,  embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key,  embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        # 不再使用reference_points_occlusion
        sampling_offsets = self.sampling_offsets(query).view(bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)

        if reference_points.shape[-1] == 2:
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """
            bs, num_query, num_Z_anchors, xy = reference_points.shape

            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            reference_points = reference_points[:, :, None, None, None, :, :]
            sampling_offsets = sampling_offsets / offset_normalizer[None, None, None, :, None, :]

            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)

            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape

            assert num_all_points == num_points * num_Z_anchors

            sampling_locations = sampling_locations.view(bs, num_query, num_heads, num_levels, num_all_points, xy)

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(f'Last dim of reference_points must be 'f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        # obtain the attention weights for deformable attention
        attention_weights = self.attention_weights(query).view(bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(bs, num_query, self.num_heads, self.num_levels, self.num_points)

        # 不再使用depth和occlusion weights

        # obtain the attention weights for Depth deformable attention
        depth_reference_points = reference_points.reshape(bs, num_query * num_Z_anchors, 1, 1, 1, 2).contiguous()
        depth_attention_weights = torch.ones_like(depth_reference_points[..., 0]).contiguous()
        pred_img_depth = pred_img_depth.unsqueeze(2).contiguous()

        depth_output = MultiScaleDeformableAttnFunction_fp32.apply(pred_img_depth, spatial_shapes,level_start_index,depth_reference_points,depth_attention_weights, self.im2col_step)
        depth_output = depth_output.reshape(bs, num_query, num_Z_anchors, -1)   # [bs*num_cam, num_query, num_Z_anchors, C]
        depth_output = depth_output.softmax(-1).argmax(-1) * depth_bound[2] + depth_bound[0]
        depth_weights = torch.exp(
            -torch.square(
                torch.min(torch.abs(reference_points_depth - (depth_output - depth_bound[2])), torch.abs(reference_points_depth - (depth_output + depth_bound[2])))
            ) / 2 * (self.sigma ** 2)
        )
        # with torch.no_grad():
        cam_dir_vector = cam_dir_vector.flatten(2).permute(0,2,1).unsqueeze(2).contiguous()
        print("#"*50)
        print(f"cam_dir_vector {cam_dir_vector.shape} sampling_locations {sampling_locations.shape}")
        print("#"*50)
        cam_dir_sampling_locations = MultiScaleDeformableAttnFunction_fp32.apply(
            cam_dir_vector, 
            spatial_shapes, 
            level_start_index, 
            sampling_locations.flatten(1,2).unsqueeze(3).clone(), 
            torch.ones_like(attention_weights).flatten(1,-1).unsqueeze(2).unsqueeze(3), 
            self.im2col_step
        ).view(*list(attention_weights.shape),3)  # cam_dir_sampling_locations shape: torch.Size([18, 171, 8, 1, 2, 3])
        print(f"cam_dir_sampling_locations {cam_dir_sampling_locations.shape} ")
        cam_dir_depth_reference_point = MultiScaleDeformableAttnFunction_fp32.apply(
            cam_dir_vector, 
            spatial_shapes, 
            level_start_index, 
            depth_reference_points, 
            depth_attention_weights, 
            self.im2col_step
        ).view(*list(depth_weights.shape),3)
        print(f"cam_dir_depth_reference_point {cam_dir_depth_reference_point.shape} ")
            # norm_weight = (cam_dir_sampling_locations*cam_dir_depth_reference_point[:,:,None,None]).sum(-1)
        # torch.cuda.empty_cache()
        attention_weights = attention_weights * depth_weights[:, :, None, None]
        # attention_weights = attention_weights * depth_weights[:, :, None, None]*norm_weight

        # 直接使用attention_weights进行计算
        output = MultiScaleDeformableAttnFunction_fp32.apply(
            value, 
            spatial_shapes, 
            level_start_index, 
            sampling_locations, 
            attention_weights, 
            self.im2col_step
        )

        if not self.batch_first:
            output = output.permute(1, 0, 2)
        return output
