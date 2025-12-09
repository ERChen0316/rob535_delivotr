lib = True
lib_dir = 'lib'

voxel_size = [0.1, 0.1, 0.2]  # nuScenes uses finer voxels
region_shape = (24, 24, 1)
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]  # 360 deg range
sparse_shape = (40, 1024, 1024)  # derived from range and voxel size

drop_info_training = {
    0: {'max_tokens': 30, 'drop_range': (0, 30)},
    1: {'max_tokens': 60, 'drop_range': (30, 60)},
    2: {'max_tokens': 100, 'drop_range': (60, 100000)},
}
drop_info_test = {
    0: {'max_tokens': 30, 'drop_range': (0, 30)},
    1: {'max_tokens': 60, 'drop_range': (30, 60)},
    2: {'max_tokens': 100, 'drop_range': (60, 100)},
}
drop_info = dict(train=drop_info_training, test=drop_info_test)

model = dict(
    type='DeLiVoTr',
    pts_voxel_layer=dict(
        voxel_size=voxel_size,  # updated geometry
        max_num_points=-1,
        point_cloud_range=point_cloud_range,  # updated range
        max_voxels=(-1, -1)
    ),
    pts_voxel_encoder=dict(
        type='DynamicVFECustom',
        in_channels=5,  # nuScenes points have 5 dims
        feat_channels=[64, 128],
        with_distance=False,
        voxel_size=voxel_size,  # updated geometry
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,  # updated range
        norm_cfg=dict(type='naiveSyncBN1dCustom', eps=1e-3, momentum=0.01),
        with_centroid_aware_vox=True,
        centroid_to_point_pos_emb_dims=32,
    ),
    pts_middle_encoder=dict(
        type='DeLiVoTrInputLayer',
        drop_info=drop_info,
        region_shape=region_shape,
        sparse_shape=sparse_shape,  # updated grid size
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
        sparse_shape=sparse_shape,  # updated grid size
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
            dict(
                channels=128,
                kernel_size=3,
                padding=1,
                stride=1,
                norm_cfg=dict(
                    type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
            ) for _ in range(4)
        ],
        conv_scales=(1, 2, 4, 8),
    ),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128],
        out_channels=[128],
        upsample_strides=[1],
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True,
    ),
    bbox_head=dict(
        type='CenterHead',
        in_channels=128,
        tasks=[
            dict(num_class=10, class_names=[  # 10-class nuScenes set
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
            ]),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2),
        ),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],  # 360 deg box
            max_num=500,  # more objects than KITTI
            score_threshold=0.1,
            out_size_factor=1,
            voxel_size=voxel_size[:2],  # updated geometry
            pc_range=point_cloud_range[:2],  # symmetric range
            code_size=7),
        separate_head=dict(
            type='DCNSeparateHead', init_bias=-2.19, final_kernel=3,
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
        grid_size=sparse_shape,  # updated grid
        voxel_size=voxel_size,  # updated geometry
        out_size_factor=1,
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=500,  # nuScenes has more objects
        min_radius=2,
        point_cloud_range=point_cloud_range,  # updated range
        code_weights=[1.0] * 8  # extended list to match head dims
    ),
    test_cfg=dict(
        post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],  # 360 deg
        max_per_img=500,
        max_pool_nms=False,
        min_radius=[4, 12, 10, 1, 0.85, 0.175],
        score_threshold=0.1,
        pc_range=point_cloud_range[:2],  # symmetric range
        out_size_factor=1,
        voxel_size=voxel_size[:2],  # updated geometry
        nms_type='rotate',
        pre_max_size=4096,
        post_max_size=500,
        nms_thr=0.7
    )
)

dataset_type = 'NuScenesDataset'  # dataset swap
data_root = 'data/nuscenes/'  # nuScenes root
file_client_args = dict(backend='disk')

class_names = [  # nuScenes class list
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]  # repeated for dataloader

input_modality = dict(use_lidar=True, use_camera=False)

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'nuscenes_dbinfos_train.pkl',  # nuScenes dbinfo
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(  # 10-class requirements
            car=5, truck=5, construction_vehicle=5, bus=5,
            trailer=5, barrier=5, motorcycle=5, bicycle=5,
            pedestrian=5, traffic_cone=5,
        )),
    classes=class_names,
    sample_groups=dict(
        car=15, truck=10, construction_vehicle=10, bus=10,
        trailer=10, barrier=8, motorcycle=8, bicycle=8,
        pedestrian=15, traffic_cone=8,
    ),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5, use_dim=5,  # 5-dim lidar
        file_client_args=file_client_args),
    file_client_args=file_client_args)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5, use_dim=5,  # 5-dim points
        file_client_args=file_client_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        file_client_args=file_client_args),
    # ObjectSample kept disabled as in KITTI
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),  # updated range
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),  # updated range
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),  # 10 classes
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5, use_dim=5,  # 5-dim points
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),  # updated range
            dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5, use_dim=5,  # 5-dim points
        file_client_args=file_client_args),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='Collect3D', keys=['points'])
]

data = dict(
    samples_per_gpu=2,  # nuScenes needs lower batch size
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,  # nuScenes dataset
            data_root=data_root,  # nuScenes root
            ann_file=data_root + 'nuscenes_infos_train.pkl',  # nuScenes infos
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            box_type_3d='LiDAR',
            file_client_args=file_client_args)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',  # nuScenes infos
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        file_client_args=file_client_args),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',  # nuScenes infos
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        file_client_args=file_client_args),
)

evaluation = dict(interval=1, pipeline=eval_pipeline)

lr = 1e-5
optimizer = dict(
    type='AdamW',
    lr=lr,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(custom_keys={'norm': dict(decay_mult=0.)}),
)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(100, 1e-3),
    cyclic_times=1,
    step_ratio_up=0.1,
)
momentum_config = None
runner = dict(type='EpochBasedRunner', max_epochs=40)

checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]

find_unused_parameters = True
opencv_num_threads = 0
mp_start_method = 'fork'
