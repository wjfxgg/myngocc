import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import BaseModule, force_fp32
from mmcv.utils import TORCH_VERSION, digit_version

from mmdet3d.models.builder import HEADS
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32


@HEADS.register_module()
class GatedTemporalFusion6_add_sparsefusion_my_add(BaseModule):
    def __init__(
        self,
        history_num=4,
        depth_sampler_embed_dims=256,
        depth_score_mode='prob',
        visibility_depth_mode='linear',
        depth_sampler_num_heads=8,
        depth_sampler_num_levels=4,
        depth_sampler_num_points=4,
        im2col_step=64,
        top_k=500,
        single_bev_num_channels=96,
        occ_embedims=32,
        visibility_gamma=1.0,
        visibility_min=0.0,
        visibility_eps=1e-6,
        visibility_weight_mode='multiply',
        num_classes=18,
        vis_theta=0.28,
        vis_beta=10.0,
        vis_gamma=0.4,
        vis_sigma=0.1,
        nonempty_thresh=0.1,
        max_step_ratio=1.2,
        **kwargs
    ):
        super(GatedTemporalFusion6_add_sparsefusion_my_add, self).__init__()
        self.depth_score_mode = depth_score_mode
        self.visibility_depth_mode = visibility_depth_mode
        self.visibility_gamma = visibility_gamma
        self.visibility_min = visibility_min
        self.visibility_eps = visibility_eps
        self.visibility_weight_mode = visibility_weight_mode
        # 基础参数
        self.history_num = history_num
        self.top_k = top_k
        self.single_bev_num_channels = single_bev_num_channels
        self.occ_embedims = occ_embedims
        # self.fg_scale = nn.Parameter(torch.full((history_num,), 10.0))
        # self.bg_scale = nn.Parameter(torch.full((history_num,), 10.0))
        # 可见性门控参数
        # self.vis_theta = vis_theta  # 可见性阈值
        # self.vis_beta = vis_beta    # 敏感度参数
        # self.vis_gamma = vis_gamma  # 场景4固定权重
        # self.vis_sigma = vis_sigma  # 软化参数
        # self.vis_theta = nn.Parameter(torch.tensor(vis_theta))   # 可见性阈值
        # self.vis_beta = nn.Parameter(torch.tensor(vis_beta))     # 敏感度参数
        # self.vis_gamma = nn.Parameter(torch.tensor(vis_gamma))   # 场景4固定权重
        # self.vis_sigma = nn.Parameter(torch.tensor(vis_sigma))  
        self.visible_embed = nn.Sequential(
                nn.Linear(6, single_bev_num_channels*2),
                nn.Softplus(),
                nn.Linear(single_bev_num_channels*2, single_bev_num_channels),
            )
        
        # 历史特征缓存（新增history_last_bev）
        self.history_bev = None  # 缓存历史多帧BEV特征
        self.history_last_bev = None  # 缓存上一帧最终融合后的BEV特征（关键新增）
        self.history_cam_intrins = None
        self.history_cam_extrins = None
        
        # 网络层（保持不变）
        self.occ_embedding = nn.Sequential(
            nn.Linear(num_classes, occ_embedims),
            nn.Softplus(),
            nn.Linear(occ_embedims, occ_embedims),
        )
        self.history_fusion_linear = nn.Sequential(
            nn.Linear(single_bev_num_channels * (history_num + 1) + occ_embedims, single_bev_num_channels),
            nn.Softplus(),
            nn.Linear(single_bev_num_channels, single_bev_num_channels),
        )
        self.history_fusion_bg_linear = nn.Sequential(
            nn.Linear(single_bev_num_channels * (history_num//2 + 1) + occ_embedims, single_bev_num_channels),
            nn.Softplus(),
            nn.Linear(single_bev_num_channels, single_bev_num_channels),
        )

        
        # 可见性计算组件（保持不变）
        # self.rt_vis_calculator = EfficientRayTracingVisibility(
        #     nonempty_thresh=nonempty_thresh,
        #     max_step_ratio=max_step_ratio
        # )
        # self.img_shape = (900, 1600)  # 默认图像尺寸

        # self.depth_sampler = DeformableDepthSampler(
        #     embed_dims=depth_sampler_embed_dims,
        #     num_heads=depth_sampler_num_heads,
        #     num_levels=depth_sampler_num_levels,
        #     num_points=depth_sampler_num_points
        # )
        self.history_forward_augs = None  # 用于缓存历史帧的变换矩阵（如BDAM矩阵）
        self.im2col_step = im2col_step
        # self.voxel_encoder = nn.Linear(single_bev_num_channels, depth_sampler_embed_dims)
        self.dbound = [1.0, 45.0, 0.5]
        self.pc_range = [-40, -40, -1.0, 40, 40, 5.4]
        self.final_dim = (256, 704)
    def compute_visibility(self, grid, cam_intrins, cam_extrins, img_shape, img_feats, spatial_shapes):
        """
        升级：结合可变形注意力采样的深度值优化可见性计算
        Args:
            img_feats: 多尺度图像特征 [bs, num_levels, c, h, w]
            spatial_shapes: 图像特征尺度 [num_levels, 2] (h, w)
            其他参数同原函数
        Returns:
            vis_prob: [bs, h, w, z] 优化后的可见性概率
        """
        bs, h, w, z, _ = grid.shape
        h_img, w_img = img_shape
        device = grid.device
        num_voxels = h * w * z  # 体素总数

        # 1. 原有相机投影逻辑（计算图像坐标和初始可见性）
        # 1.1 体素坐标→相机坐标→图像坐标
        grid_cam = grid.unsqueeze(1).expand(bs, self.num_cams, h, w, z, 3)  # [bs, num_cams, h, w, z, 3]
        grid_flat = grid_cam.reshape(-1, num_voxels, 3)  # [bs*num_cams, N, 3]
        grid_hom = torch.cat([grid_flat, torch.ones_like(grid_flat[..., :1])], dim=-1)  # [bs*num_cams, N, 4]

        cam_intrins_flat = cam_intrins.reshape(-1, 4, 4)  # [bs*num_cams, 4, 4]
        cam_extrins_flat = cam_extrins.reshape(-1, 4, 4)  # [bs*num_cams, 4, 4]
        extrins_inv = torch.inverse(cam_extrins_flat)
        cam_coords = torch.bmm(extrins_inv[:, :3, :4], grid_hom.permute(0, 2, 1)).permute(0, 2, 1)  # [bs*num_cams, N, 3]
        depth = cam_coords[..., 2:3] + 1e-8  # 相机坐标系下的深度

        # 1.2 计算初始可见性（原逻辑）
        img_coords = torch.bmm(cam_intrins_flat[:, :3, :3], cam_coords.permute(0, 2, 1)).permute(0, 2, 1)  # [bs*num_cams, N, 3]
        img_xy = img_coords[..., :2] / depth  # [bs*num_cams, N, 2] (u, v)
        depth_valid = (cam_coords[..., 2] > 0).float()  # [bs*num_cams, N]
        u_valid = (img_xy[..., 0] >= 0) & (img_xy[..., 0] < w_img)
        v_valid = (img_xy[..., 1] >= 0) & (img_xy[..., 1] < h_img)
        img_valid = (u_valid & v_valid).float()  # [bs*num_cams, N]
        initial_vis = depth_valid * img_valid  # [bs*num_cams, N]

        # 2. 可变形注意力深度采样
        # 2.1 准备输入：体素特征编码
        # voxel_feat = self.voxel_encoder(self.curr_bev_feat)  # [bs, c_, z, h, w] → [bs, z*h*w, embed_dims]（需提前展平体素特征）
        # voxel_feat = voxel_feat.reshape(bs, num_voxels, -1)  # [bs, N, embed_dims]

        # 2.2 生成参考点（归一化到[0,1]）
        norm_img_xy = img_xy / torch.tensor([w_img, h_img], device=device).view(1, 1, 2)  # [bs*num_cams, N, 2]
        # 取主相机（如第0个相机）的参考点作为采样基准
        ref_points = norm_img_xy.reshape(bs, self.num_cams, num_voxels, 2)[:, 0]  # [bs, N, 2]
        ref_points = ref_points.unsqueeze(2).repeat(1, 1, self.depth_sampler.num_levels, 1)  # [bs, N, L, 2]

        # 2.3 采样深度特征
        sampled_depth = self.depth_sampler(
            query=voxel_feat,
            value=img_feats,  # 多尺度图像特征 [bs, L, c, h, w]
            reference_points=ref_points,
            spatial_shapes=spatial_shapes
        )  # [bs, N]

        # 3. 结合采样深度优化可见性
        # 3.1 深度一致性校验：采样深度与相机投影深度的差异
        cam_depth = depth.reshape(bs, self.num_cams, num_voxels)[:, 0]  # 主相机的投影深度 [bs, N]
        depth_diff = torch.abs(sampled_depth - cam_depth) / (cam_depth + 1e-8)  # 相对深度差
        depth_consistent = (depth_diff < 0.3).float()  # 深度差小于30%则认为有效

        # 3.2 融合可见性：初始可见性 × 深度一致性
        initial_vis = initial_vis.reshape(bs, self.num_cams, num_voxels).max(dim=1)[0]  # [bs, N]（多相机取max）
        vis_prob = initial_vis * depth_consistent  # [bs, N]

        # 4. 还原形状
        return vis_prob.reshape(bs, h, w, z)  # [bs, h, w, z]

    def compute_alpha_unified(self, V_curr, V_prev):
        """统一计算当前帧融合权重α（覆盖四场景）"""
        eps = 1e-8
        # 1. 基础动态权重σ_base（场景1-3）
        ratio = V_curr / (V_curr + V_prev + eps)
        sigma_base = torch.sigmoid(self.vis_beta * (ratio - 0.5))
        
        # 2. 场景4软化掩码σ_both
        mask_curr = torch.sigmoid(-(V_curr - self.vis_theta) / self.vis_sigma)
        mask_prev = torch.sigmoid(-(V_prev - self.vis_theta) / self.vis_sigma)
        sigma_both = mask_curr * mask_prev
        
        # 3. 最终权重计算
        alpha = (1 - sigma_both) * sigma_base + sigma_both * self.vis_gamma
        return alpha.unsqueeze(-1)  # [bs, N, 1]

    def compute_gate_weights(self, V_prev_agg, V_curr):
        """计算历史和当前帧的门控权重"""
        alpha = self.compute_alpha_unified(V_curr, V_prev_agg)
        return 1 - alpha, alpha  # 历史权重，当前权重

    def generate_grid(self, curr_bev, voxel_min, voxel_max, voxel_size):
        """生成体素中心坐标网格（自车坐标系）"""
        bs, c_, z, h, w = curr_bev.shape
        device = curr_bev.device
        
        # 计算体素中心坐标
        x_coords = torch.linspace(
            voxel_min[0] + voxel_size[0]/2, 
            voxel_max[0] - voxel_size[0]/2, 
            w, device=device
        )
        y_coords = torch.linspace(
            voxel_min[1] + voxel_size[1]/2, 
            voxel_max[1] - voxel_size[1]/2, 
            h, device=device
        )
        z_coords = torch.linspace(
            voxel_min[2] + voxel_size[2]/2, 
            voxel_max[2] - voxel_size[2]/2, 
            z, device=device
        )
        
        # 生成网格并扩展维度 [bs, h, w, z, 3]
        x_grid, y_grid, z_grid = torch.meshgrid(x_coords, y_coords, z_coords, indexing='xy')
        grid = torch.stack([x_grid, y_grid, z_grid], dim=-1)  # [w, h, z, 3]
        grid = grid.permute(1, 0, 2, 3)  # [h, w, z, 3]
        return grid.unsqueeze(0).repeat(bs, 1, 1, 1, 1)  # [bs, h, w, z, 3]


    def get_reference_points(self, H, W, Z=None, num_points_in_pillar =4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
        torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    @force_fp32(apply_to=('reference_points', 'cam_params'))
    def point_sampling(self, reference_points, pc_range, img_metas, cam_params=None):
        # prepare for point sampling
        lidar2img = []
        ego2lidar = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])     # lidar2img update the post aug in the loading pipeline
            ego2lidar.append(img_meta['ego2lidar'])
        lidar2img = torch.stack(lidar2img, dim=0).to(reference_points.device)
        ego2lidar = torch.stack(ego2lidar, dim=0).to(reference_points.device)

        sensor2egos, ego2globals, intrins, post_augs, bda_mat = cam_params
        num_cam = sensor2egos.size(1)
        ogfH, ogfW = self.final_dim

        # reference_points defines in the bev space, [bs, D, hxw, 3]
        # change reference_points from bev-ego coordinate to ego coordinate
        reference_points = reference_points.clone()
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
                                     (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
                                     (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
                                     (pc_range[5] - pc_range[2]) + pc_range[2]

        # prepare for point sampling
        reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
        reference_points = reference_points.permute(1, 0, 2, 3)  # shape: (num_points_in_pillar,bs,h*w,4)
        D, B, num_query = reference_points.size()[:3]  # D=num_points_in_pillar , num_query=h*w
        reference_points = reference_points.view(D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)  # shape: (num_points_in_pillar,bs,num_cam,h*w,4)
        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)
        ego2lidar = ego2lidar.view(1, B, 1, 1, 4, 4).repeat(D, 1, num_cam, num_query, 1, 1)
        inverse_bda = bda_mat.view(1, B, 1, 1, 4, 4).repeat(D, 1, num_cam, num_query, 1, 1)

        # change reference_points from ego coordinate to img coordinate
        eps = 1e-5
        reference_points_cam = (lidar2img @ ego2lidar @ inverse_bda @ reference_points).squeeze(-1)   # [num_points_in_pillar, bs, num_cam, num_query=h*w, 4]
        reference_points_depth = reference_points_cam[..., 2:3]
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(reference_points_depth, torch.ones_like(reference_points_depth) * eps)

        # Bug!!
        # Correct normalize is
        # reference_points_cam[..., 0] /= ogfW
        # reference_points_cam[..., 1] /= ogfH
        # But for reproducing our results, we use the following normalization
        reference_points_cam[..., 0] /= ogfW
        reference_points_cam[..., 1] /= ogfH

        bev_mask = (reference_points_depth > eps)
        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)                  # shape: (num_cam, bs,h*w, num_points_in_pillar, 2)
        reference_points_depth = reference_points_depth.permute(2, 1, 3, 0, 4)              # shape: (num_cam, bs,h*w, num_points_in_pillar, 1)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)                        # shape: (num_cam, bs,h*w, num_points_in_pillar)

        return reference_points_cam, reference_points_depth, bev_mask


    @force_fp32()
    def forward(self, curr_bev, cam_params, history_fusion_params, dx, bx, history_last_bev=None, last_occ_pred=None, nonempty_prob=None, img_feats=None, spatial_shapes=None,pred_img_depth=None,depth_bound=None,**kwargs):
        """
        Args:
            curr_bev: [bs, c, z, h, w] 当前帧BEV特征
            cam_params: 相机参数列表，包含外参、内参等
            history_fusion_params: 历史融合参数（包含序列信息等）
            dx: 体素尺寸 (x, y, z)
            bx: 体素偏移
            nonempty_prob: [bs, z, h, w] 体素非空概率
            last_occ_pred: [bs, z, h, w, num_classes] 上一时刻 occupancy 预测
            img_feats: 多尺度图像特征 [bs, num_levels, c, h, w]（新增，用于深度采样）
            spatial_shapes: 图像特征尺度 [num_levels, 2] (h, w)（新增，用于深度采样）
        Returns:
            curr_bev_updated: [bs, c, z, h, w] 融合后BEV特征
        """
        # print(self.history_num)
        # print("print(self.fg_scale)")
        # print(self.fg_scale) 
        # print("print(self.bg_scale)")
        # print(self.bg_scale)
        # # 可见性门控参数
        # # self.vis_theta = vis_theta  # 可见性阈值
        # # self.vis_beta = vis_beta    # 敏感度参数
        # # self.vis_gamma = vis_gamma  # 场景4固定权重
        # # self.vis_sigma = vis_sigma  # 软化参数
        # print("print(self.vis_theta # 可见性阈值)")
        # print(self.vis_theta )  # 可见性阈值
        # print("print(self.vis_beta) # 敏感度参数")
        # print(self.vis_beta)      # 敏感度参数
        # print("print(self.vis_gamma) # 场景4固定权重")
        # print(self.vis_gamma)  # 场景4固定权重
        # print("print(self.vis_sigma) # 软化参数")
        # print(self.vis_sigma) 


        # if torch.rand(1).item() < 1/2000:
        #     print(self.history_num)
        #     print("print(self.fg_scale)")
        #     print(self.fg_scale) 
        #     print("print(self.bg_scale)")
        #     print(self.bg_scale)
        #     # 可见性门控参数
        #     print("print(self.vis_theta # 可见性阈值)")
        #     print(self.vis_theta)  # 可见性阈值
        #     print("print(self.vis_beta) # 敏感度参数")
        #     print(self.vis_beta)      # 敏感度参数
        #     print("print(self.vis_gamma) # 场景4固定权重")
        #     print(self.vis_gamma)  # 场景4固定权重
        #     print("print(self.vis_sigma) # 软化参数")
        #     print(self.vis_sigma)  


        # -------------------------- 1. 解析参数后打印核心形状 --------------------------
        # 解析相机参数
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
        slots = torch.zeros(list([ref_3d.shape[0],6,ref_3d.shape[2],ref_3d.shape[1],1])).to(ref_3d)
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

        depth_probs = MultiScaleDeformableAttnFunction_fp32.apply(pred_img_depth, spatial_shapes,level_start_index,depth_reference_points,depth_attention_weights, self.im2col_step)
        depth_probs = depth_probs.reshape(bs,6, max_len,z, -1)   # [bs*num_cam, num_query, num_Z_anchors, C]
        
        depth_probs = depth_probs.clamp(min=self.visibility_eps)
        depth_probs = depth_probs / depth_probs.sum(-1, keepdim=True)  # bs, num_query, num_Z_anchors, D_bins

        # Survival / transmittance per discrete depth bin:
        # V_k = 1 - sum_{j<k} P_j
        visibility_bins = 1.0 - depth_probs.cumsum(dim=-1)
        visibility_bins = torch.cat(
            [torch.ones_like(visibility_bins[..., 0:1]), visibility_bins[..., 0:-1]],
            dim=-1,
        )
        termination_bins = visibility_bins * depth_probs

        # Map continuous reference depth (meters) -> visibility scalar per anchor.
        # Use depth_bound if provided; otherwise fallback to self.dbound.
        dbound = depth_bound if depth_bound is not None else self.dbound
        depth_start = reference_points_depth_rebatch.new_tensor(dbound[0])
        depth_step = reference_points_depth_rebatch.new_tensor(dbound[2])
        depth_bin = (reference_points_depth_rebatch - depth_start) / depth_step

        num_depth_bins = visibility_bins.size(-1)
        depth_bin = depth_bin.clamp(0.0, float(num_depth_bins - 1))

        depth_bin_idx0 = depth_bin.floor().to(torch.long).clamp_(0, num_depth_bins - 1)
        depth_bin_idx1 = (depth_bin_idx0 + 1).clamp_(max=num_depth_bins - 1)
        frac = (depth_bin - depth_bin_idx0.to(depth_bin.dtype)).clamp(0.0, 1.0)

        def _sample_bins(bins: torch.Tensor):
            val0 = bins.gather(-1, depth_bin_idx0)
            val1 = bins.gather(-1, depth_bin_idx1)
            val_linear = val0
            if num_depth_bins > 1:
                val_linear = val0 * (1.0 - frac) + val1 * frac
            if self.visibility_depth_mode == 'hard':
                val_used = val0
            elif self.visibility_depth_mode == 'linear':
                val_used = val_linear
            else:
                raise ValueError(f'Unsupported visibility_depth_mode={self.visibility_depth_mode!r}')
            return val0, val_linear, val_used

        visibility_hard, visibility_linear, visibility_used = _sample_bins(visibility_bins)
        prob_hard, prob_linear, prob_used = _sample_bins(depth_probs)
        termination_hard, termination_linear, termination_used = _sample_bins(termination_bins)

        visibility_hard = visibility_hard.clamp(0.0, 1.0)
        visibility_linear = visibility_linear.clamp(0.0, 1.0)
        visibility_used = visibility_used.clamp(0.0, 1.0)

        prob_hard = prob_hard.clamp(0.0, 1.0)
        prob_linear = prob_linear.clamp(0.0, 1.0)
        prob_used = prob_used.clamp(0.0, 1.0)

        termination_hard = termination_hard.clamp(0.0, 1.0)
        termination_linear = termination_linear.clamp(0.0, 1.0)
        termination_used = termination_used.clamp(0.0, 1.0)

        if self.depth_score_mode == 'transmittance':
            depth_score = visibility_used
        elif self.depth_score_mode == 'prob':
            depth_score = prob_used
        elif self.depth_score_mode == 'termination':
            depth_score = termination_used
        else:
            raise ValueError(f'Unsupported depth_score_mode={self.depth_score_mode!r}')

        depth_score = depth_score.clamp(0.0, 1.0)
        if self.visibility_gamma != 1.0:
            depth_score = depth_score.clamp(min=self.visibility_eps).pow(self.visibility_gamma)
        if self.visibility_min > 0.0:
            depth_score = self.visibility_min + (1.0 - self.visibility_min) * depth_score

        num_all_points = 1 * 1
        num_anchors = depth_score.size(-1)
        if num_all_points != num_anchors:
            if num_all_points % num_anchors != 0:
                raise ValueError(
                    f'depth_score last dim ({num_anchors}) must divide attention_weights num_points '
                    f'({num_all_points}). Consider setting num_points == num_Z_anchors.'
                )
            points_per_anchor = num_all_points // num_anchors
            depth_score_points = depth_score[:, :, None, :].expand(-1, -1, points_per_anchor, -1)
            depth_score_points = depth_score_points.reshape(bs, num_query, num_all_points)
        else:
            depth_score_points = depth_score


        if self.visibility_weight_mode == 'multiply':
            # attention_weights = depth_score_points[:, :, None, None, :]
            attention_weights = depth_score_points

        elif self.visibility_weight_mode == 'multiply_renorm':
            attention_weights = depth_score_points[:, :, None, None, :]
            denom = attention_weights.sum(dim=(-1, -2), keepdim=True).clamp(min=self.visibility_eps)
            attention_weights = attention_weights / denom
        else:
            raise ValueError(f'Unsupported visibility_weight_mode={self.visibility_weight_mode!r}')
        # 直接使用attention_weights进行计算
        
        # depth_output = depth_output + torch.cat([(torch.zeros_like(depth_output[...,:1]) + 1e-9),torch.zeros_like(depth_output[...,1:])],dim=-1)
        # depth_output =depth_output/depth_output.sum(-1)[...,None] #bs,xy,z,D

        
        # reference_points_depth_rebatch

        # increment = torch.zeros_like(depth_output)
        # increment[..., 0] = 1e-9  # 非原地赋值（创建新张量）
        # depth_output = depth_output + increment
        
        # print("depth_output.sum(-1).shape")
        # print(depth_output.sum(-1).shape)
        # print("depth_output.sum(-1)")
        # print(depth_output.sum(-1))
        # print("print((depth_output.sum(-1)==0).sum())")
        # print((depth_output.sum(-1)==0).sum())

        # print("depth_output.sum(-1).shape")
        # print(depth_output.sum(-1).shape)
        # print("depth_output.sum(-1)")
        # print(depth_output.sum(-1))
        # print("print((depth_output.sum(-1)==1).sum())")
        # print((depth_output.sum(-1)>=0.99).sum())
        # print("depth_output")
        # print(depth_output)


        # depth_output = (1-depth_output.cumsum(dim=-1))
        # depth_output =torch.cat([torch.ones_like(depth_output[...,0:1]),depth_output[...,0:-1]],dim=-1)

        # # print("print((depth_output[...,-1]==0).sum())")
        # # print((depth_output[...,-1]==0).sum())
        # depth_output = (bev_query_depth_rebatch*depth_output).sum(-1)

        #恢复depth_output的shape
        # fix_depth_output = depth_output.new_zeros([bs, 6, w*h, z, 88])
        # for j in range(bs):
        #     for i in range(6):
        #         index_query_per_img = indexes[j][i]
        #         fix_depth_output[j, i, index_query_per_img] = depth_output[j, i, :len(index_query_per_img)]

        for j in range(bs):
            for i in range(6):
                index_query_per_img = indexes[j][i]
                # slots[j,i, index_query_per_img] = torch.max(slots[j, index_query_per_img],attention_weights[j, i, :len(index_query_per_img)])
                slots[j,i, index_query_per_img] = attention_weights[j, i, :len(index_query_per_img)]


        # depth_sum = fix_depth_output.sum(dim=-1).view(3,6,25,25,2)
        # bs = depth_sum.shape[0]       # 3
        # num_cam = depth_sum.shape[1]  # 6
        # z_layers = depth_sum.shape[4] # 2
        # H, W = depth_sum.shape[2], depth_sum.shape[3]  # 25,25

        # # 颜色映射：0值用黑色，非0值用渐变色
        # cmap = plt.cm.viridis
        # cmap.set_bad(color='black')  # 0值标记为黑色

        # # 设置子图布局（不变）
        # fig, axes = plt.subplots(
        #     nrows=bs, ncols=num_cam * z_layers,
        #     figsize=(30, 8),
        #     squeeze=False
        # )
        # fig.suptitle('fix_depth_output.sum(-1) 可视化（黑色=0值，颜色=非0值）', fontsize=16, y=0.98)


        # # -------------------------- 2. 循环绘制每个子图（核心修改：添加 .detach()） --------------------------
        # for batch_idx in range(bs):
        #     for cam_idx in range(num_cam):
        #         for z_idx in range(z_layers):
        #             col_idx = cam_idx * z_layers + z_idx
        #             ax = axes[batch_idx, col_idx]
                    
        #             # 核心修改：先 detach() 切断计算图，再转 cpu 和 numpy
        #             data = depth_sum[batch_idx, cam_idx, :, :, z_idx].detach().cpu().numpy()
        #             data[data == 0] = np.nan  # 0值替换为NaN，显示为黑色
                    
        #             # 绘制热力图（修改 vmin/vmax：同样添加 .detach()）
        #             im = ax.imshow(
        #                 data, 
        #                 cmap=cmap, 
        #                 aspect='auto',
        #                 # 关键修改：depth_sum 先 detach 再转 numpy，确保不影响梯度
        #                 vmin=np.nanmin(depth_sum.detach().cpu().numpy()),
        #                 vmax=np.nanmax(depth_sum.detach().cpu().numpy())
        #             )
                    
        #             # 子图标题和坐标轴（不变）
        #             ax.set_title(
        #                 f'Batch{batch_idx+1}\nCam{cam_idx+1} Z{z_idx+1}',
        #                 fontsize=10, pad=5
        #             )
        #             ax.set_xticks([])
        #             ax.set_yticks([])


        # # -------------------------- 3. 添加颜色条（不变） --------------------------
        # cbar = fig.colorbar(
        #     im, 
        #     ax=axes.ravel().tolist(),
        #     shrink=0.8,
        #     pad=0.02
        # )
        # cbar.set_label('Sum of Depth Bins (D=88)', fontsize=12)


        # # -------------------------- 4. 调整布局并保存（不变） --------------------------
        # plt.tight_layout(rect=[0, 0, 0.98, 0.95])
        # plt.savefig('depth_sum_visualization.png', dpi=300, bbox_inches='tight')
        # plt.show()

        # output

        #计数更新，建立在纸上的假设成立的基础上
        # count = bev_mask.sum(-1) > 0
        # count = count.permute(1, 2, 0).sum(-1)
        # count = torch.clamp(count, min=1.0)
        # slots = slots / count[..., None]

        # print("slots.shape")
        # print(slots.shape)
        # print("slots")  
        # print(slots)
        # zzzzz=1/0

        V_curr =slots.view(bs*6,1, h, w,z).permute(0, 1, 4, 2, 3).view(bs,6,1,z,h,w)
        # slots[...,0]+=1e-9
        # print("slots.sum(-1).shape")
        # print(slots.sum(-1).shape)
        # print("slots.sum(-1)")
        # print(slots.sum(-1))
        # print("print((slots.sum(-1)==0).sum())")
        # print((slots.sum(-1)==0).sum())
        # slots =slots/slots.sum(-1)[...,None] #bs,xy,z,D

        # print("slots.sum(-1).shape")
        # print(slots.sum(-1).shape)
        # print("slots.sum(-1)")
        # print(slots.sum(-1))
        # print("print((slots.sum(-1)==1).sum())")
        # print("slots")
        # print(slots)


        # slots = (1-slots.cumsum(dim=-1))

        # print("print((slots[...,-1]==0).sum())")
        # print((slots[...,-1]==0).sum())

        # print("slots")
        # print(slots)
        # print("print((slots[...,-1]<0.01).sum())")
        # print((slots[...,-1]<0.01).sum())

        #TODO 这里的对于边界值的考虑，从0开始还是从1开始？




        # slots = self.output_proj(slots)
        #TODO 上面这里需要检查一下
        #这里相当于两次softmax，可能会导致分布变得不够尖锐，需要进一步确认

        #到这里slots就是可见性的概率分布了
        #这里先尝试使用期望值进行计算，使得可微分

        #然后再采用stc的原始离散计算方法，


        # 打印解析后关键变量形状
        # print("="*50)
        # print("1. 解析参数后核心变量形状：")
        # print(f"curr_bev: {curr_bev.shape} (预期：[bs, c, z, h, w])")
        # print(f"curr_cam_extrins: {curr_cam_extrins.shape} (预期：[bs, num_cams, 4, 4])")
        # print(f"curr_cam_intrins: {curr_cam_intrins.shape} (预期：[bs, num_cams, 4, 4])")
        # print(f"forward_augs: {forward_augs.shape} (预期：[bs, 4, 4])")
        # print(f"dx: {dx.shape if hasattr(dx, 'shape') else type(dx)} (预期：[3])")
        # print(f"bx: {bx.shape if hasattr(bx, 'shape') else type(bx)} (预期：[3])")
        # print(f"bs: {bs}, c_: {c_}, z: {z}, h: {h}, w: {w} (BEV特征维度)")
        # print("="*50)

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

        # check seq_ids > 0
        assert (seq_ids >= 0).all()
        # -------------------------- 2. 初始化历史缓存后打印 --------------------------
        if self.history_bev is None:
            # self.history_bev = curr_bev.repeat(1, self.history_num, 1, 1, 1)  # [bs, mc, z, h, w]
            # self.history_cam_intrins = curr_cam_intrins.unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)  # [bs, T, num_cams, 4, 4]
            # self.history_cam_extrins = curr_cam_extrins.unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)  # [bs, T, num_cams, 4, 4]
            # self.history_bev = curr_bev.clone()
            self.history_forward_augs = forward_augs.clone()
            self.history_seq_ids = seq_ids.clone()
            self.history_bev = curr_bev.repeat(1, self.history_num, 1, 1, 1)
            self.history_sweep_time = curr_bev.new_zeros(curr_bev.shape[0], self.history_num)
            self.history_visibility = V_curr.repeat(1,1, self.history_num, 1, 1, 1).half()
        self.history_bev = self.history_bev.detach()
        self.history_visibility = self.history_visibility.detach().half()
        self.history_sweep_time += 1

        # 打印历史缓存形状
        # print("\n2. 历史缓存初始化后形状：")
        # print(f"history_bev: {self.history_bev.shape} (预期：[bs, mc, z, h, w]，mc={mc})")
        # print(f"history_cam_intrins: {self.history_cam_intrins.shape} (预期：[bs, history_num, num_cams, 4, 4])")
        # print(f"history_cam_extrins: {self.history_cam_extrins.shape} (预期：[bs, history_num, num_cams, 4, 4])")

        # -------------------------- 3. 生成网格和BEV变换矩阵后打印 --------------------------
        # 处理新序列（略，不影响维度）
        # start_of_sequence = history_fusion_params.get('start_of_sequence', torch.zeros(bs, dtype=torch.bool, device=device))
        if start_of_sequence.sum()>0:
            self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_num, 1, 1, 1)
            # self.history_cam_intrins[start_of_sequence] = curr_cam_intrins[start_of_sequence].unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)
            # self.history_cam_extrins[start_of_sequence] = curr_cam_extrins[start_of_sequence].unsqueeze(1).repeat(1, self.history_num, 1, 1, 1)
            self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]
            self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
            self.history_sweep_time[start_of_sequence] = 0  # zero the new sequence timestep starts
            self.history_visibility[start_of_sequence] = V_curr[start_of_sequence].repeat(1,1, self.history_num, 1, 1, 1).half()

        # 生成体素网格和BEV变换矩阵
        grid = self.generate_grid(curr_bev) #[bs,y,x,z,4]
        grid_3d = grid
        feat2bev = self.generate_feat2bev(grid, dx, bx)

        # 打印网格和变换矩阵形状
        # print("\n3. 生成网格和BEV变换矩阵后形状：")
        # print(f"grid_3d (体素网格): {grid_3d.shape} (关键！预期：[bs, h, w, z, 3] 或 [bs, w, h, z, 3])")
        # print(f"feat2bev (BEV变换矩阵): {feat2bev.shape} (预期：[bs, 4, 4])")

        # -------------------------- 4. 运动补偿矩阵计算后打印 --------------------------
        # 获取帧间姿态变换
        # curr_to_prev_ego_rt = history_fusion_params.get('curr_to_prev_ego_rt', torch.eye(4, device=device).unsqueeze(0).repeat(bs, 1, 1))
        # 计算RT流（坐标变换矩阵）
        rt_flow = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_ego_rt @ torch.inverse(forward_augs) @ feat2bev)
        # 生成齐次网格
        # 在forward函数中，生成grid_hom的位置修正：
        # grid_3d = self.generate_grid(curr_bev)  # 现在形状：[3, 25, 25, 2, 3]（bs, h, w, z, 3）
        # # 生成齐次坐标（x,y,z,1），并添加最后一个维度（用于矩阵乘法）
        # grid_hom = torch.cat([
        #     grid_3d,  # [3,25,25,2,3]
        #     torch.ones_like(grid_3d[..., :1])  # [3,25,25,2,1]（补充1作为齐次坐标）
        # ], dim=-1).unsqueeze(-1)  # 最终形状：[3,25,25,2,4,1]（符合预期）
        # # 打印运动补偿相关形状（矩阵乘法前关键检查）
        # print("\n4. 运动补偿矩阵计算后形状（矩阵乘法前）：")
        # print(f"curr_to_prev_ego_rt (帧间姿态): {curr_to_prev_ego_rt.shape} (预期：[bs, 4, 4])")
        # print(f"rt_flow (变换流): {rt_flow.shape} (预期：[bs, 4, 4])")
        # print(f"grid_hom (齐次网格): {grid_hom.shape} (关键！预期：[bs, h, w, z, 4, 1]，需与rt_flow广播匹配)")
        # print(f"rt_flow.view后: {rt_flow.view(bs, 1, 1, 1, 4, 4).shape} (预期：[bs, 1, 1, 1, 4, 4])")

        # # -------------------------- 5. 网格变换后打印（解决之前维度错的核心） --------------------------
        # try:
        #     grid_transformed = rt_flow.view(bs, 1, 1, 1, 4, 4) @ grid_hom  # 矩阵乘法：[bs, h, w, z, 4, 1]
        #     print("\n5. 网格变换后形状（矩阵乘法成功！）：")
        #     print(f"grid_transformed: {grid_transformed.shape} (预期：[bs, h, w, z, 4, 1])")
        # except RuntimeError as e:
        #     print(f"\n5. 网格变换矩阵乘法报错！错误信息：{str(e)}")
        #     print(f"  - rt_flow.view形状: {rt_flow.view(bs, 1, 1, 1, 4, 4).shape}")
        #     print(f"  - grid_hom形状: {grid_hom.shape}")
        #     print("  提示：需确保grid_hom的第1-4维度与rt_flow.view的第2-5维度匹配（广播规则）")
        #     raise e  # 继续抛出错误，方便定位
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
            self.history_visibility.reshape(bs*6, self.history_num, z, h, w).half(),  # 输入：[bs, mc, z, h, w]
            grid[:,None].repeat(1,6,1,1,1,1).flatten(0,1).to(curr_bev.dtype).permute(0, 3, 1, 2, 4).half(),
            align_corners=True,
            mode='bilinear'
        )
        # print("\n7. 历史BEV采样后形状：")
        # print(f"history_bev.reshape: {self.history_bev.reshape(bs, mc, z, h, w).shape} (预期：[bs, mc, z, h, w])")
        # print(f"sampled_history_bev: {sampled_history_bev.shape} (预期：[bs, mc, z, h, w])")

        # -------------------------- 8. 可见性计算后打印 --------------------------
        # 计算当前帧可见性
        # V_curr = self.compute_visibility(
        #     grid_3d, 
        #     cam_intrins=curr_cam_intrins,
        #     cam_extrins=curr_cam_extrins,
        #     img_shape=self.img_shape,
        #     img_feats=img_feats,
        #     spatial_shapes=spatial_shapes
        # )

        # print("V_curr (当前可见性).  "*3)
        # V_curr = slots
        # 计算历史帧可见性
        # prev_cam_intrins = self.history_cam_intrins[:, -1]
        # prev_cam_extrins = self.history_cam_extrins[:, -1]
        # V_prev = self.compute_visibility(
        #     grid_3d,
        #     cam_intrins=prev_cam_intrins,
        #     cam_extrins=prev_cam_extrins,
        #     img_shape=self.img_shape,
        #     img_feats=img_feats,
        #     spatial_shapes=spatial_shapes
        # )

        V_prev = sampled_history_visibility #bs,4,z,h,w

        # print("\n8. 可见性计算后形状：")
        # print(f"V_curr (当前可见性): {V_curr.shape} (预期：[bs, h, w, z])")
        # print(f"V_prev (历史可见性): {V_prev.shape} (预期：[bs, h, w, z])")

        # -------------------------- 9. 稀疏采样前展平变量打印 --------------------------
        # 展平变量（用于稀疏采样）
        curr_bev_flat = curr_bev.reshape(bs, c_, -1)  # [bs, c_, N], N=h*w*z
        history_bev_flat = sampled_history_bev.reshape(bs, mc, -1)  # [bs, mc, N]
        V_prev_flat = V_prev.reshape(bs*6,self.history_num, -1)  # [bs, 4,N]
        V_curr_flat = V_curr.reshape(bs*6, 1,-1)  # [bs, 1,N]
        nonempty_prob_flat = nonempty_prob.reshape(bs, -1)  # [bs, N]
        # print("print(nonempty_prob_flat.shape)")
        # print(nonempty_prob_flat.shape)
        total_voxels = nonempty_prob_flat.shape[1]

        # print("\n9. 稀疏采样前展平变量形状：")
        # print(f"curr_bev_flat: {curr_bev_flat.shape} (预期：[bs, c_, N], N={total_voxels})")
        # print(f"history_bev_flat: {history_bev_flat.shape} (预期：[bs, mc, N])")
        # print(f"nonempty_prob_flat: {nonempty_prob_flat.shape} (预期：[bs, N])")
        # print(f"total_voxels (h*w*z): {total_voxels} (预期：{h*w*z})")

        # -------------------------- 10. 前景/背景索引及融合后打印（可选，确认后续维度） --------------------------
        # 生成前景/背景索引
        fg_indices = torch.topk(nonempty_prob_flat, self.top_k, dim=1)[1]
        bg_indices = torch.topk(1 - nonempty_prob_flat, total_voxels - self.top_k, dim=1)[1]
        # 提取前景特征（示例，其他融合步骤类似）
        fg_history_feat = torch.gather(history_bev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, mc, 1))

        # print("\n10. 前景/背景索引及特征提取后形状：")
        # print(f"fg_indices (前景索引): {fg_indices.shape} (预期：[bs, top_k])")
        # print(f"bg_indices (背景索引): {bg_indices.shape} (预期：[bs, N-top_k])")
        # print(f"fg_history_feat (前景历史特征): {fg_history_feat.shape} (预期：[bs, mc, top_k])")
        # print("="*50)

        # -------------------------- 后续原有逻辑（略，维度已通过打印确认） --------------------------
        # 8. 前景融合（原有代码）
        # 9. 背景融合（原有代码）
        # 10. 更新当前BEV特征（原有代码）
        # curr_bev_flat = curr_bev.reshape(bs, c_, -1)  # [bs, c_, N] N=h*w*z
        # history_bev_flat = sampled_history_bev.reshape(bs, mc, -1)  # [bs, mc, N]
        # V_prev_flat = V_prev.reshape(bs, -1)  # [bs, N]
        # V_curr_flat = V_curr.reshape(bs, -1)  # [bs, N]
        # nonempty_prob_flat = nonempty_prob.reshape(bs, -1)  # [bs, N]
        # total_voxels = nonempty_prob_flat.shape[1]

        # fg_indices = torch.topk(nonempty_prob_flat, self.top_k, dim=1)[1]  # [bs, top_k]
        # bg_indices = torch.topk(1 - nonempty_prob_flat, total_voxels - self.top_k, dim=1)[1]  # [bs, N-top_k]

        # # 前景特征提取
        # fg_history_feat = torch.gather(history_bev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, mc, 1))  # [bs, mc, top_k]
        fg_curr_feat = torch.gather(curr_bev_flat, dim=2, index=fg_indices.unsqueeze(1).repeat(1, c_, 1))  # [bs, c_, top_k]

        # 历史特征时间聚合
        # fg_history_feat_time = fg_history_feat.reshape(bs, self.history_num, c_, self.top_k)  # [bs, T, c_, K]
        #TODO 后续可以把time_weights也乘进去


        # time_weights = torch.exp(-0.5 * torch.arange(self.history_num, device=device)).view(1, self.history_num, 1, 1)

        fg_V_prev = torch.gather(V_prev_flat, dim=2, index=fg_indices[:,None].unsqueeze(1).repeat(1,6, self.history_num, 1).flatten(0,1))  # [bs, 4,K]
        fg_V_curr = torch.gather(V_curr_flat, dim=2, index=fg_indices[:,None].unsqueeze(1).repeat(1,6, 1, 1).flatten(0,1))  # [bs, 1,K]
        # fg_time_vis_weights = fg_V_prev/(fg_V_prev.sum(dim=1).unsqueeze(1)+1e-10 ) # [bs, 4,K]
        #下面进行替换，不用显示提取权重，不计算 softmax
        # fg_time_vis_weights = (fg_V_prev*(self.fg_scale.view(1,self.history_num,1))).softmax(dim=1)
        # fg_time_vis_weights = (fg_V_prev*(self.fg_scale.view(1,self.history_num,1)))
        # print("print(last_occ_pred.shape)")
        # print(last_occ_pred.shape) #bs,x,y,z,num_classes ->bs,z,y,x,num_classes
        last_occ_pred = last_occ_pred.permute(0, 3, 2, 1, 4).reshape(bs, -1, last_occ_pred.shape[-1])  # [bs, N, num_classes]

        fg_occ_feat = torch.gather(last_occ_pred, dim=1, index=fg_indices.unsqueeze(-1).repeat(1, 1, last_occ_pred.shape[-1]))  # [bs, K, num_classes]

        fg_occ_embed = self.occ_embedding(fg_occ_feat).permute(0, 2, 1)  # [bs, occ_embedims, K]

        # print("print(fg_V_curr.shape)")
        # print(fg_V_curr.shape) # bs,1,K
        # print("print(fg_curr_feat.shape)")
        # print(fg_curr_feat.shape) #torch.Size([2, 96, 500])  bs,c_,K
        # print("print(fg_V_prev.shape)") 
        # print(fg_V_prev.shape) #torch.Size([2, 4, 500])  bs,4,K
        # print("print(fg_history_feat.shape)")
        # print(fg_history_feat.shape) #torch.Size([2, 384, 500])  bs,mc,K
        # print("print(fg_occ_embed.shape)")
        # print(fg_occ_embed.shape) #torch.Size([2, 32, 500])  bs,occ_embedims,K
        
        visible_feature = torch.cat([fg_V_curr.view(bs,6,-1,self.top_k),fg_V_prev.view(bs,6,-1,self.top_k)],dim=2).permute(0,2,3,1)
        # fg_fused = torch.cat([ fg_V_curr * fg_curr_feat,(fg_V_prev.unsqueeze(2) * fg_history_feat.view(bs, self.history_num, c_, self.top_k)).reshape(bs, self.history_num*c_, self.top_k) , fg_occ_embed], dim=1).permute(0, 2, 1)
        visible_embed=self.visible_embed(visible_feature).permute(0,1,3,2)
        fg_curr_feat = visible_embed[:,0,:,:]+fg_curr_feat
        fg_history_feat = visible_embed[:,1:,:,:].flatten(1,2)+fg_history_feat

        fg_fused = torch.cat([fg_curr_feat,fg_history_feat , fg_occ_embed], dim=1).permute(0, 2, 1)

        fg_fused = self.history_fusion_linear(fg_fused).permute(0, 2, 1)  # [bs, c_, K]


        # bg_V_prev = torch.gather(V_prev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, self.history_num//2, 1))  # [bs, bg_k]
        bg_V_prev = torch.gather(V_prev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1,6, self.history_num//2, 1).flatten(0,1))  # [bs, bg_k]

        # bg_V_curr = torch.gather(V_curr_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, 1, 1))  # [bs, bg_k]
        bg_V_curr = torch.gather(V_curr_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1,6, 1, 1).flatten(0,1))  # [bs, bg_k]

        bg_history_feat = torch.gather(history_bev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, mc//2, 1))  # [bs, mc//2, bg_k]
        bg_curr_feat = torch.gather(curr_bev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, c_, 1))  # [bs, c_, bg_k]
        bg_occ_feat = torch.gather(last_occ_pred, dim=1, index=bg_indices.unsqueeze(-1).repeat(1, 1, last_occ_pred.shape[-1]))  # [bs, bg_k, num_classes]
        bg_occ_embed = self.occ_embedding(bg_occ_feat).permute(0, 2, 1)  # [bs, occ_embedims, bg_k]

        # print(bg_V_curr.shape)
        # print("print(bg_V_curr.shape)")
        # print(bg_V_curr.shape)
        # print("print(bg_curr_feat.shape)")
        # print(bg_curr_feat.shape)
        # print("print(bg_V_prev.shape)")
        # print(bg_V_prev.shape)
        # print("print(bg_history_feat.shape)")
        # print(bg_history_feat.shape)
        # print("print(bg_occ_embed.shape)")
        # print(bg_occ_embed.shape)
        # bg_fused = torch.cat([ bg_V_curr * bg_curr_feat,(bg_V_prev.unsqueeze(2) * bg_history_feat.view(bs, self.history_num//2, c_, total_voxels - self.top_k)).reshape(bs, self.history_num*c_//2, total_voxels - self.top_k), bg_occ_embed], dim=1).permute(0, 2, 1)
        
        visible_feature = torch.cat([bg_V_curr.view(bs,6,-1,total_voxels - self.top_k),bg_V_prev.view(bs,6,-1,total_voxels - self.top_k)],dim=2).permute(0,2,3,1)
        # fg_fused = torch.cat([ fg_V_curr * fg_curr_feat,(fg_V_prev.unsqueeze(2) * fg_history_feat.view(bs, self.history_num, c_, self.top_k)).reshape(bs, self.history_num*c_, self.top_k) , fg_occ_embed], dim=1).permute(0, 2, 1)
        visible_embed=self.visible_embed(visible_feature).permute(0,1,3,2)
        bg_curr_feat = visible_embed[:,0,:,:]+bg_curr_feat
        bg_history_feat = visible_embed[:,1:,:,:].flatten(1,2)+bg_history_feat

        # fg_fused = torch.cat([fg_curr_feat,fg_history_feat , fg_occ_embed], dim=1).permute(0, 2, 1)
        
        
        bg_fused = torch.cat([  bg_curr_feat, bg_history_feat, bg_occ_embed], dim=1).permute(0, 2, 1)
        
        # print("print(bg_fused.shape)")
        # print(bg_fused.shape)
        

        
        bg_fused = self.history_fusion_bg_linear(bg_fused).permute(0, 2, 1)  # [bs, c_, bg_k]

        # print()
        #TODO TODO 后续用senet实现一个通道注意力的版本  这个可能会更好？


        # # print("print(fg_history_feat_time.shape)")
        # # print(fg_history_feat_time.shape)
        # # print("print(fg_time_vis_weights.shape)")
        # # print(fg_time_vis_weights.shape)
        # # fg_history_agg = (fg_history_feat_time * fg_time_vis_weights.unsqueeze(2)).sum(dim=1)  # [bs, c_, K]).sum(dim=1)  # [bs, c_, K]]
        # fg_history_agg = (fg_history_feat_time * fg_time_vis_weights.unsqueeze(2)).sum(dim=1)  # [bs, c_, K]).sum(dim=1)  # [bs, c_, K]]


        # # print("print(fg_history_agg.shape)")
        # # print(fg_history_agg.shape)

        # # 可见性聚合与门控
        
        # # fg_V_prev_time = fg_V_prev.unsqueeze(1).repeat(1, self.history_num, 1)  # [bs, T, K]
        # # fg_V_prev_agg = (fg_V_prev_time * time_weights.squeeze(-1)).sum(dim=1)  # [bs, K]
        # fg_V_prev_agg = fg_V_prev.max(dim=1)[0]  # [bs, K]
        # fg_w_hist, fg_w_curr = self.compute_gate_weights(fg_V_prev_agg, fg_V_curr.squeeze(1))  # [bs, K, 1]

        # # 前景融合
        # fg_history_agg_perm = fg_history_agg.permute(0, 2, 1)  # [bs, K, c_]
        # fg_curr_perm = fg_curr_feat.permute(0, 2, 1)  # [bs, K, c_]
        # fg_fused = fg_w_hist * fg_history_agg_perm + fg_w_curr * fg_curr_perm  # [bs, K, c_]

        # # occupancy嵌入融合
        # last_occ_reshaped = last_occ_pred.permute(0, 3, 2, 1, 4).reshape(bs, -1, last_occ_pred.shape[-1])  # [bs, N, num_classes]
        # fg_occ_feat = torch.gather(last_occ_reshaped, dim=1, index=fg_indices.unsqueeze(-1).repeat(1, 1, last_occ_pred.shape[-1]))  # [bs, K, num_classes]
        # fg_occ_embed = self.occ_embedding(fg_occ_feat).permute(0, 2, 1)  # [bs, occ_embedims, K]
        # fg_fused = torch.cat([fg_fused, fg_occ_embed.permute(0, 2, 1)], dim=-1)  # [bs, K, c_+occ_embedims]
        # fg_fused = self.history_fusion_linear(fg_fused).permute(0, 2, 1)  # [bs, c_, K]

        # # 背景融合（原有代码）
        # bg_history_feat = torch.gather(history_bev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, mc, 1))  # [bs, mc//2, bg_k]
        # bg_curr_feat = torch.gather(curr_bev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, c_, 1))  # [bs, c_, bg_k]

        # bg_history_feat_time = bg_history_feat.reshape(bs, self.history_num, c_, -1)  # [bs, T, c_//2, bg_k]
        # # bg_history_agg = (bg_history_feat_time * time_weights).sum(dim=1)  # [bs, c_//2, bg_k]
        # bg_V_prev = torch.gather(V_prev_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, self.history_num, 1))  # [bs, bg_k]
        # bg_V_curr = torch.gather(V_curr_flat, dim=2, index=bg_indices.unsqueeze(1).repeat(1, 1, 1))  # [bs, bg_k]
        # #TODO 这个10的超参数？ 调整成可学习？
        # # bg_time_vis_weights = bg_V_prev/(bg_V_prev.sum(dim=1).unsqueeze(1) +1e-10) # [bs, 4,K]
        # bg_time_vis_weights =(bg_V_prev*(self.bg_scale.view(1,self.history_num,1))).softmax(dim=1)
        # bg_history_agg = (bg_history_feat_time * bg_time_vis_weights.unsqueeze(2)).sum(dim=1)  # [bs, c_, K]).sum(dim=1)  # [bs, c_, K]]



        # # bg_history_agg_perm = F.pad(bg_history_agg.permute(0, 2, 1), (0, c_ - c_//2, 0, 0))  # [bs, bg_k, c_]

        # bg_history_agg_perm = bg_history_agg.permute(0, 2, 1)  # [bs, bg_k, c_]
        # # bg_V_prev_time = bg_V_prev.unsqueeze(1).repeat(1, self.history_num, 1)  # [bs, T, bg_k]
        # # bg_V_prev_agg = (bg_V_prev_time * time_weights.squeeze(-1)).sum(dim=1)  # [bs, bg_k]
        # bg_V_prev_agg = bg_V_prev.max(dim=1)[0]  # [bs, bg_k]
        # bg_w_hist, bg_w_curr = self.compute_gate_weights(bg_V_prev_agg, bg_V_curr.squeeze(1))  # [bs, bg_k, 1]

        # bg_curr_perm = bg_curr_feat.permute(0, 2, 1)  # [bs, bg_k, c_]
        # # print("*"*50)
        # # print("print(bg_w_hist.shape)")
        # # print(bg_w_hist.shape)
        # # print("print(bg_w_curr.shape)")
        # # print(bg_w_curr.shape)
        # # print("print(bg_history_agg_perm.shape)")
        # # print(bg_history_agg_perm.shape)
        # # print("print(bg_curr_perm.shape)")        
        # # print(bg_curr_perm.shape)

        # # # 断言批次大小一致
        # # assert bg_w_hist.shape[0] == bg_history_agg_perm.shape[0] == bg_w_curr.shape[0] == bg_curr_perm.shape[0], \
        # #     f"批次大小不匹配: {bg_w_hist.shape[0]}, {bg_history_agg_perm.shape[0]}, {bg_w_curr.shape[0]}, {bg_curr_perm.shape[0]}"

        # # # 断言第二维度（bg_k）一致
        # # assert bg_w_hist.shape[1] == bg_history_agg_perm.shape[1] == bg_w_curr.shape[1] == bg_curr_perm.shape[1], \
        # #     f"bg_k维度不匹配: {bg_w_hist.shape[1]}, {bg_history_agg_perm.shape[1]}, {bg_w_curr.shape[1]}, {bg_curr_perm.shape[1]}"

        # # # 断言第三维度（c_）匹配（bg_w_hist和bg_w_curr的第三维为1，不影响广播）
        # # assert bg_history_agg_perm.shape[2] == bg_curr_perm.shape[2], \
        # #     f"特征维度c_不匹配: {bg_history_agg_perm.shape[2]} vs {bg_curr_perm.shape[2]}"

        # # print("bg_w_hist dtype:", bg_w_hist.dtype)
        # # print("bg_history_agg_perm dtype:", bg_history_agg_perm.dtype)
        # # print("bg_w_curr dtype:", bg_w_curr.dtype)
        # # print("bg_curr_perm dtype:", bg_curr_perm.dtype)


        # # print("bg_w_hist device:", bg_w_hist.device)
        # # print("bg_history_agg_perm device:", bg_history_agg_perm.device)
        # # print("bg_w_curr device:", bg_w_curr.device)
        # # print("bg_curr_perm device:", bg_curr_perm.device)


        # bg_fused = bg_w_hist * bg_history_agg_perm + bg_w_curr * bg_curr_perm  # [bs, bg_k, c_]
        # # 先验证乘法是否正常
        # # temp1 = bg_w_hist * bg_history_agg_perm
        # # temp2 = bg_w_curr * bg_curr_perm
        # # 再验证加法是否正常

        # # bg_w_hist = bg_w_hist.contiguous()
        # # bg_history_agg_perm = bg_history_agg_perm.contiguous()
        # # bg_w_curr = bg_w_curr.contiguous()
        # # bg_curr_perm = bg_curr_perm.contiguous()

        # # # 重新计算
        # # temp1 = bg_w_hist * bg_history_agg_perm
        # # temp2 = bg_w_curr * bg_curr_perm
        # # temp1 = temp1.contiguous()
        # # temp2 = temp2.contiguous()
        # # bg_fused = temp1 + temp2


        # # 转移所有张量到CPU
        # # bg_w_hist_cpu = bg_w_hist.cpu()
        # # bg_history_agg_perm_cpu = bg_history_agg_perm.cpu()
        # # bg_w_curr_cpu = bg_w_curr.cpu()
        # # bg_curr_perm_cpu = bg_curr_perm.cpu()

        # # # 分步执行运算
        # # try:
        # #     temp1_cpu = bg_w_hist_cpu * bg_history_agg_perm_cpu
        # #     temp2_cpu = bg_w_curr_cpu * bg_curr_perm_cpu
        # #     bg_fused_cpu = temp1_cpu + temp2_cpu
        # #     print(bg_fused_cpu)
        # #     print(bg_fused_cpu.shape)
        # #     print("CPU运算成功，无明显错误")
        # # except Exception as e:
        # #     print(f"CPU运算报错：{e}")  # 此处会显示具体错误原因（如值异常）


        # # bg_fused = temp1.clone() + temp2.clone()
        # # 1/0
        # bg_occ_feat = torch.gather(last_occ_reshaped, dim=1, index=bg_indices.unsqueeze(-1).repeat(1, 1, last_occ_pred.shape[-1]))  # [bs, bg_k, num_classes]
        # bg_occ_embed = self.occ_embedding(bg_occ_feat).permute(0, 2, 1)  # [bs, occ_embedims, bg_k]
        # bg_fused = torch.cat([bg_fused, bg_occ_embed.permute(0, 2, 1)], dim=-1)  # [bs, bg_k, c_+occ_embedims]
        # bg_fused = self.history_fusion_bg_linear(bg_fused).permute(0, 2, 1)  # [bs, c_, bg_k]

        # 更新当前BEV
        curr_bev_updated = curr_bev_flat.clone()
        curr_bev_updated.scatter_add_(dim=2, index=fg_indices.unsqueeze(1).repeat(1, c_, 1), src=fg_fused)
        curr_bev_updated.scatter_add_(dim=2, index=bg_indices.unsqueeze(1).repeat(1, c_, 1), src=bg_fused)
        curr_bev_updated = curr_bev_updated.reshape(bs, c_, z, h, w)  # 恢复原形状

        # 更新历史缓存
        self.history_last_bev = curr_bev_updated.detach().clone()
        self.history_bev = torch.cat([curr_bev,sampled_history_bev[:, :-c_, ...]], dim=1).detach()
        # self.history_cam_intrins = torch.cat([curr_cam_intrins.unsqueeze(1),self.history_cam_intrins[:, :-1, ...]], dim=1).detach()
        # self.history_cam_extrins = torch.cat([curr_cam_extrins.unsqueeze(1),self.history_cam_extrins[:, 1-1:, ...]], dim=1).detach()
        self.history_visibility =torch.cat([V_curr, V_prev.view(bs,6,V_prev.size(1),V_prev.size(2),V_prev.size(3),V_prev.size(4))[:, :,:-1, ...]],dim=2).detach()
        self.history_forward_augs = forward_augs.clone()

        return curr_bev_updated



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
        feat2bev[0, 0] = dx[0]
        feat2bev[1, 1] = dx[1]
        feat2bev[2, 2] = dx[2]
        feat2bev[0, 3] = bx[0] - dx[0] / 2.
        feat2bev[1, 3] = bx[1] - dx[1] / 2.
        feat2bev[2, 3] = bx[2] - dx[2] / 2.
        feat2bev[3, 3] = 1
        feat2bev = feat2bev.view(1, 4, 4)
        return feat2bev