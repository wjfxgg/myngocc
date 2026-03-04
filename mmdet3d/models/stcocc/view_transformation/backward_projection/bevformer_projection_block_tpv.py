import copy
from mmcv.runner import force_fp32

from mmdet.models import HEADS

from .bevformer_projection_norm import BEVFormerBackwardProjection_Norm
from mmdet3d.models.stcocc.modules.block_tpv import MultiScaleBlockTPV


@HEADS.register_module()
class BEVFormerBackwardProjectionBlockTPV(BEVFormerBackwardProjection_Norm):
    """BEVFormer backward projection augmented with block TPV aggregation."""

    def __init__(self, block_tpv_cfg=None, **kwargs):
        super().__init__(**kwargs)
        default_cfg = dict(
            embed_dims=self.embed_dims,
            num_heads=8,
            splits=((1, 1, 1), (2, 2, 2), (4, 4, 4), (8, 8, 8)),
            dropout=0.0,
            ffn_ratio=2.0,
        )
        if block_tpv_cfg is None:
            block_tpv_cfg = default_cfg
        else:
            block_tpv_cfg = copy.deepcopy(block_tpv_cfg)
            block_tpv_cfg.setdefault('embed_dims', self.embed_dims)
            block_tpv_cfg.setdefault('num_heads', 8)
            block_tpv_cfg.setdefault('splits', ((1, 1, 1), (2, 2, 2), (4, 4, 4), (8, 8, 8)))
            block_tpv_cfg.setdefault('dropout', 0.0)
            block_tpv_cfg.setdefault('ffn_ratio', 2.0)
        self.block_tpv = MultiScaleBlockTPV(**block_tpv_cfg)

    @force_fp32(apply_to=('mlvl_feats',))
    def forward(
        self,
        mlvl_feats,
        img_metas,
        voxel_feats=None,
        cam_params=None,
        pred_img_depth=None,
        bev_mask=None,
        last_occ_pred=None,
        prev_bev=None,
        prev_bev_aug=None,
        history_fusion_params=None,
        **kwargs
    ):
        voxel_feats = self.block_tpv(voxel_feats) if voxel_feats is not None else voxel_feats
        return super().forward(
            mlvl_feats=mlvl_feats,
            img_metas=img_metas,
            voxel_feats=voxel_feats,
            cam_params=cam_params,
            pred_img_depth=pred_img_depth,
            bev_mask=bev_mask,
            last_occ_pred=last_occ_pred,
            prev_bev=prev_bev,
            prev_bev_aug=prev_bev_aug,
            history_fusion_params=history_fusion_params,
            **kwargs
        )
