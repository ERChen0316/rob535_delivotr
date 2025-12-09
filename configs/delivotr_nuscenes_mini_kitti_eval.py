# Config for evaluating KITTI model on nuScenes with proper class mapping
# Uses full 10 nuScenes classes but only evaluates the 3 overlapping ones

lib = True
lib_dir = 'lib'

# Using KITTI's voxel configuration (not used for eval-only, but needed)
voxel_size = [0.16, 0.16, 4]
point_cloud_range = [0, -40.32, -3, 80.64, 40.32, 1]

# Dataset settings
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes-v1.0-mini/'

# Use ALL 10 nuScenes classes (default order)
# This matches the remapped prediction indices
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

input_modality = dict(use_lidar=True, use_camera=False)

# Simple test pipeline
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
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
        ann_file=data_root + 'nuscenes_infos_val_kitti_range.pkl',  # Filtered GT
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,  # All 10 classes
        test_mode=True,
        box_type_3d='LiDAR',
        with_velocity=False
    )
)

# Pipeline for visualization
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
    ),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False
    ),
    dict(type='Collect3D', keys=['points'])
]

evaluation = dict(interval=2)
dist_params = dict(backend='nccl')
log_level = 'INFO'
find_unused_parameters = True
