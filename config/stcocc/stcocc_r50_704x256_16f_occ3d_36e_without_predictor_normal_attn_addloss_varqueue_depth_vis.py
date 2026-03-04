_base_ = ['./stcocc_r50_704x256_16f_occ3d_36e_without_predictor_normal_attn_addloss_SA_adddepth_fix_normalNet1_.py']

custom_imports = dict(
    imports=[
        'mmdet3d.models.stcocc.modules.temporal_fusion_variable_queue_depth_vis',
        'mmdet3d.models.stcocc.view_transformation.backward_projection.bevformer_projection_block_tpv_deformable',
        'mmdet3d.models.stcocc.modules.block_tpv_deformable',
        'mmdet3d.models.stcocc.detectors.stcocc_without_predictor_normal_attn_addloss_varqueue_depth_vis',
    ],
    allow_failed_imports=False,
)

backward_numC_Trans = 96

model = dict(
    type='STCOccWithoutPredictor_normal_attn_addloss_VarQueueDepthVis',
    temporal_fusion=dict(
        type='ObservableQueueFusionDepthVis',
        single_bev_num_channels=backward_numC_Trans,
        num_T=16,
        queue_windows=(2, 4, 8, 16),
    )
)
