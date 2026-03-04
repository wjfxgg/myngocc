_base_ = ['./stcocc_r50_704x256_16f_occ3d_36e_without_predictor_normal_attn_addloss_block_tpv_deformable.py']

custom_imports = dict(
    imports=[
        'mmdet3d.models.stcocc.modules.temporal_fusion_variable_queue',
        'mmdet3d.models.stcocc.view_transformation.backward_projection.bevformer_projection_block_tpv_deformable',
        'mmdet3d.models.stcocc.modules.block_tpv_deformable',
    ],
    allow_failed_imports=False,
)

# backward_numC_Trans = 96

model = dict(
    temporal_fusion=dict(
        type='ObservableQueueFusion',
        single_bev_num_channels=backward_numC_Trans,
        num_T=16,
        queue_windows=(2, 4, 8, 16),
        score_from='norm',
    )
)
