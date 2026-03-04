_base_ = ['./stcocc_r50_704x256_16f_occ3d_36e_without_predictor_normal_attn_addloss_SA_adddepth_fix_normalNet1_.py']

custom_imports = dict(
    imports=[
        'mmdet3d.models.stcocc.view_transformation.backward_projection.bevformer_projection_block_tpv',
        'mmdet3d.models.stcocc.modules.block_tpv',
    ],
    allow_failed_imports=False,
)
num_gpus = 1
samples_per_gpu = 1
workers_per_gpu = 1
total_epoch = 36
# Channel settings follow the parent config for backward projection.
backward_numC_Trans = 96

model = dict(
    backward_projection=dict(
        type='BEVFormerBackwardProjectionBlockTPV',
        block_tpv_cfg=dict(
            embed_dims=backward_numC_Trans,
            num_heads=8,
            splits=((1, 1, 1), (2, 2, 2), (4, 4, 4), (8, 8, 8)),
            dropout=0.0,
            ffn_ratio=2.0,
        ),
    )
)
