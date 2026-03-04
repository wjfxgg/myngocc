# Copyright (c) Phigent Robotics. All rights reserved.
_base_ = ['./stcocc_r50_704x256_16f_occ3d_36e_without_predictor_normal_attn.py']

# 修改模型类型为我们的扩展类
model = dict(
    type='BEVDepth4DNormal',  # 使用我们新创建的支持normal预测的模型类
    loss_normal_weight=1.0,   # normal预测的损失权重
    forward_projection=dict(
        type='BEVDetStereoForwardProjectionWithNormal',  # 使用扩展的前向投影模块
        loss_normal_weight=1.0,   # normal预测的损失权重
        normal_w_bins=8,          # normal_comp_W_bins的类别数
        normal_h_bins=8,          # normal_comp_H_bins的类别数
        normal_depth_bins=4,      # normal_comp_depth_bins的类别数
        img_backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
        img_neck=dict(
            type='CustomFPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=1,
            start_level=0,
            out_ids=[0]),
        img_view_transformer=dict(
            type='LSSVStereoForwardProjectionWithNormal',  # 使用扩展的视图变换器
            loss_normal_weight=1.0,   # normal预测的损失权重
            normal_w_bins=8,          # normal_comp_W_bins的类别数
            normal_h_bins=8,          # normal_comp_H_bins的类别数
            normal_depth_bins=4,      # normal_comp_depth_bins的类别数
            grid_config=dict(
                x=(-40.0, 40.0, 0.4),
                y=(-20.0, 20.0, 0.4),
                z=(-1.0, 3.0, 4.0),
                depth=(1.0, 60.0, 0.5)),
            input_size=(704, 256),
            downsample=4,
            output_channel=80,
            cv_downsample=4,
            depthnet_cfg=dict(
                use_dcn=False,
                aspp_mid_channels=96,
                stereo=True,
                bias=False,
                dcn_config=dict(
                    type='DCN',
                    deform_groups=8,
                    fallback_on_stride=False),
                # 注意：depthnet会自动扩展以支持normal预测，不需要修改这里的depth_channels
            )),
        img_bev_encoder_backbone=dict(
            type='CustomResNet3D',
            numC_input=80,
            num_layer=[1, 2, 4],
            num_channels=[80, 160, 320],
            stride=[1, 2, 4],
            backbone_output_ids=[0, 1, 2]),
        img_bev_encoder_neck=dict(
            type='SECONDFPN',
            in_channels=[80, 160, 320],
            upsample_strides=[1, 2, 4],
            out_channels=[80, 80, 80]))
)

# 由于我们添加了新的损失项，需要在optimizer_config中进行相应的调整
# 同时保持原始配置中的其他参数不变