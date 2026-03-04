# STCOcc config without predictor and with modified forward_train
_base_ = ['stcocc_r50_704x256_16f_occ3d_36e.py']

model = dict(
    type='STCOccWithoutPredictor',  # 使用新创建的不带predictor的检测器类
    occupancy_head=dict(
        type='OccHead',
        in_channels=256,
        hidden_channels=128,
        out_channels=20,
        kernel_size=3,
        num_blocks=2,
        dropout=0.1,
        num_classes=20,
        weight=1.0,
        loss_kwargs=dict(
            type='LovaszSoftmaxLoss',
            ignore_index=20,
            reduction='none',
            per_image=True),
        train_cfg=dict(),
        test_cfg=dict()))

# 添加temporal_fusion配置，使用不带predictor的SparseFusionWithoutPredictor
model['temporal_fusion'] = dict(
    type='SparseFusionWithoutPredictor',
    top_k=12500,
    history_num=3,
    single_bev_num_channels=256,
    bev_h=100,
    bev_w=100,
    bev_z=8
)

# 修改backward_projection配置，使用不带predictor的BEVFormerEncoder
model['backward_projection'] = dict(
    type='BackwardProjection',
    bev_h=100,
    bev_w=100,
    bev_z=8,
    backbone_cfg=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    transformer=dict(
        type='BEVFormer',
        encoder=dict(
            type='BEVFormerEncoderWithoutPredictor',  # 使用不带predictor的encoder
            num_layers=3,
            point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            grid_config=dict(
                x=[-51.2, 51.2, 0.8],
                y=[-51.2, 51.2, 0.8],
                z=[-5.0, 3.0, 0.8],
                depth=[1.0, 46.0, 0.5]),
            point_sampler=dict(
                type='PointSampler',
                num_points=[1, 2, 4, 8],
                alpha=[1.0, 0.5, 0.25, 0.125],
                beta=[0.2, 0.2, 0.2, 0.2],
                gamma=[1.0, 1.0, 1.0, 1.0]),
            transformerlayers=dict(
                type='BEVFormerEncoderLayerWithoutPredictor',  # 使用不带predictor的encoder layer
                attn_cfgs=[
                    dict(
                        type='OA_TemporalAttentionWithoutPredictor',
                        embed_dims=256,
                        num_levels=1),
                    dict(
                        type='OA_SpatialCrossAttentionWithoutPredictor',
                        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                        deformable_attention=dict(
                            type='OA_MSDeformableAttention3DWithoutPredictor',
                            embed_dims=256,
                            num_points=8,
                            num_levels=4,
                            num_Z_anchors=8),
                        batch_first=True)],
                feedforward_channels=512,
                ffn_dropout=0.1,
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm')),  # 移除'conv'操作
            positional_encoding=dict(
                type='LearnedPositionalEncoding',
                row_num_embed=100,
                col_num_embed=100,
                dim=256)),
        max_img_queue_len=3,
        return_intermediate=True),
    num_stage=3,
    bev_h_list=[100, 50, 25],
    bev_w_list=[100, 50, 25],
    bev_z_list=[8, 4, 2],
    intermediate_loss=False)

# 调整其他参数以适应不带predictor的实现
train_cfg = dict(
    train_backward_projection=True,
    train_forward_projection=False,
    train_occupancy_head=True,
    train_flow=False)

test_cfg = dict(
    output_dir='./outputs',
    vis=False,
    format_only=False,
    merge_class=True)

# 优化器配置
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.),
            relative_position_bias_table=dict(decay_mult=0.),
            norm=dict(decay_mult=0.),
            backbone=dict(lr_mult=0.1))))

# 学习率配置
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# 学习策略
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 1e-4),
    cyclic_times=1,
    step_ratio_up=0.4)

# 运行设置
total_epochs = 36
find_unused_parameters = True
runner = dict(type='EpochBasedRunner', max_epochs=36)

# 数据增强
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='SemanticKITTIDataset',
        data_root='data/semantickitti/',
        ann_file='data/semantickitti/semantickitti_infos_train.pkl',
        split='train',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='PhotoMetricDistortionMultiViewImage'),
            dict(type='LoadAnnotations3D', with_bbox_3d=False, with_label_3d=False, with_seg_3d=True),
            dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
            dict(type='ObjectRangeFilter', point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
            dict(type='ObjectNameFilter', classes=['Car', 'Pedestrian', 'Cyclist']),
            dict(type='NormalizeMultiviewImage', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(type='DefaultFormatBundle3D', class_names=['Car', 'Pedestrian', 'Cyclist']),
            dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d', 'voxel_semantics'])
        ]),
    val=dict(
        type='SemanticKITTIDataset',
        data_root='data/semantickitti/',
        ann_file='data/semantickitti/semantickitti_infos_val.pkl',
        split='val',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='LoadAnnotations3D', with_bbox_3d=False, with_label_3d=False, with_seg_3d=True),
            dict(type='NormalizeMultiviewImage', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(type='DefaultFormatBundle3D', class_names=['Car', 'Pedestrian', 'Cyclist']),
            dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d', 'voxel_semantics'])
        ]),
    test=dict(
        type='SemanticKITTIDataset',
        data_root='data/semantickitti/',
        ann_file='data/semantickitti/semantickitti_infos_test.pkl',
        split='test',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='NormalizeMultiviewImage', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(type='DefaultFormatBundle3D', class_names=['Car', 'Pedestrian', 'Cyclist']),
            dict(type='Collect3D', keys=['img'])
        ]))

# 钩子配置
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'))

# 评估配置
eval_config = dict(
    interval=1,
    metric='segm')