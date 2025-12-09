lib = True
lib_dir = 'lib'

voxel_size = [0.16, 0.16, 4]
region_shape = (24, 24, 1)
point_cloud_range = [0, -40.32, -3, 80.64, 40.32, 1]
sparse_shape = (504, 504, 1)

drop_info_training = {
    0: {'max_tokens': 30, 'drop_range': (0, 30)},
    1: {'max_tokens': 60, 'drop_range': (30, 60)},
    2: {'max_tokens': 100, 'drop_range': (60, 100000)},
}
drop_info_test = {
    0: {'max_tokens': 30, 'drop_range': (0, 30)},
    1: {'max_tokens': 60, 'drop_range': (30, 60)},
    2: {'max_tokens': 100, 'drop_range': (60, 100)},
    3: {'max_tokens': 144, 'drop_range': (100, 100000)},
}
drop_info = (drop_info_training, drop_info_test)

model = dict(
    type='DeLiVoTr',
    pts_voxel_layer=dict(
        voxel_size=voxel_size,
        max_num_points=-1,
        point_cloud_range=point_cloud_range,
        max_voxels=(-1, -1)
    ),
    pts_voxel_encoder=dict(
        type='DynamicVFECustom',
        in_channels=4,
        feat_channels=[64, 128],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1dCustom', eps=1e-3, momentum=0.01),
        with_centroid_aware_vox=True,
        centroid_to_point_pos_emb_dims=32,
    ),
    pts_middle_encoder=dict(
        type='DeLiVoTrInputLayer',
        drop_info=drop_info,
        region_shape=region_shape,
        sparse_shape=sparse_shape,
        shuffle_voxels=True,
        debug=True,
        normalize_pos=False,
        pos_temperature=10000,
    ),
    pts_backbone=dict(
        type='DeLiVoTrEncoder',
        d_model=128,
        enc_num_layers=6,
        checkpoint_layers=[],
        region_shape=region_shape,
        sparse_shape=sparse_shape,
        deli_cfg=dict(
            enc_min_depth=4,
            enc_max_depth=8,
            enc_width_mult=2.0,
            dextra_dropout=0.1,
            dextra_proj=2,
            attn_dropout=0.1,
            dropout=0.1,
            act_dropout=0.0,
            ffn_dropout=0.1,
            enc_ffn_red=2,
            norm_type='ln',
            act_type='relu',
            normalize_before=True,
        ),
        num_attached_conv=4,
        conv_kwargs=[
            dict(kernel_size=3, dilation=1, padding=1, stride=1),
            dict(kernel_size=3, dilation=1, padding=1, stride=1),
            dict(kernel_size=3, dilation=1, padding=1, stride=1),
            dict(kernel_size=3, dilation=2, padding=2, stride=1),
        ],
        conv_in_channel=128,
        conv_out_channel=128,
        conv_shortcut=True,
    ),
    pts_neck=dict(
        type='SECONDFPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        in_channels=[128, ],
        upsample_strides=[1, ],
        out_channels=[128, ]
    ),
    bbox_head=dict(
        type='CenterHead',
        in_channels=128,
        tasks=[
            dict(num_class=3, class_names=['pedestrian', 'bicycle', 'car']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2),
        ),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[0, -40, -5, 72, 40, 5],
            max_num=500,
            score_threshold=0.01,  # Lower threshold for domain shift
            out_size_factor=1,
            voxel_size=voxel_size[:2],
            pc_range=point_cloud_range[:2],
            code_size=7
        ),
        separate_head=dict(
            type='DCNSeparateHead',
            init_bias=-2.19,
            final_kernel=3,
            dcn_config=dict(
                type='DCN',
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                groups=4,
                bias=False
            ),
            norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        ),
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=2),
        norm_bbox=True
    ),
    train_cfg=dict(
        grid_size=sparse_shape,
        voxel_size=voxel_size,
        out_size_factor=1,
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=2,
        point_cloud_range=point_cloud_range,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    ),
    test_cfg=dict(
        post_center_limit_range=[0, -50, -5, 80, 50, 5],
        max_per_img=500,
        max_pool_nms=False,
        min_radius=[4, 0.85, 12],
        score_threshold=0.05,  # Lowered to capture domain-shifted predictions
        pc_range=point_cloud_range[:2],
        out_size_factor=1,
        voxel_size=voxel_size[:2],
        nms_type='rotate',
        pre_max_size=4096,
        post_max_size=500,
        nms_thr=0.2  # Standard NMS threshold
    )
)

dataset_type = 'LyftDataset'
data_root = 'data/lyft/'
file_client_args = dict(backend='disk')
class_names = ['pedestrian', 'bicycle', 'car']
input_modality = dict(use_lidar=True, use_camera=False)

# SIMPLE test pipeline WITHOUT MultiScaleFlipAug3D
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        file_client_args=file_client_args
    ),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False
    ),
    dict(type='Collect3D', keys=['points'])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'lyft_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        file_client_args=file_client_args
    )
)

evaluation = dict(interval=2)
dist_params = dict(backend='nccl')
log_level = 'INFO'
find_unused_parameters = True
