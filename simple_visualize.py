"""
Simple BEV visualizations for zero-shot generalization study.
Works with nuScenes and Lyft datasets.

Usage:
    python simple_visualize.py --dataset nuscenes --predictions delivotr_nuscenes_remapped.pkl
    python simple_visualize.py --dataset lyft --predictions predictions_lyft_remapped.pkl
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mmcv import Config
from mmdet3d.datasets import build_dataset
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Simple BEV visualizations')
    parser.add_argument('--dataset', type=str, required=True, choices=['nuscenes', 'lyft', 'kitti'],
                        help='Dataset to visualize')
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to remapped predictions pickle file')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of samples to visualize (default: 5)')
    parser.add_argument('--score-thresh', type=float, default=0.1,
                        help='Score threshold for predictions (default: 0.1)')
    parser.add_argument('--output-dir', type=str, default='simple_visualizations',
                        help='Output directory for visualizations (default: simple_visualizations)')
    return parser.parse_args()


def get_config(dataset_name):
    """Load appropriate config based on dataset."""
    if dataset_name == 'nuscenes':
        cfg = Config.fromfile('configs/delivotr_nuscenes_mini_kitti_compatible.py')
        cfg.data.test.ann_file = 'data/nuscenes-v1.0-mini/nuscenes_infos_val_kitti_range.pkl'
        cfg.data.test.classes = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']
    elif dataset_name == 'lyft':
        cfg = Config.fromfile('configs/delivotr_lyft_kitti_compatible.py')
        cfg.data.test.ann_file = 'data/lyft/lyft_infos_val_kitti_range.pkl'
        cfg.data.test.classes = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']
    elif dataset_name == 'kitti':
        cfg = Config.fromfile('configs/delivotr_kitti.py')
        cfg.data.test.ann_file = 'data/kitti/kitti_infos_val.pkl'
        cfg.data.test.classes = ['Pedestrian', 'Cyclist', 'Car']

    return cfg


args = parse_args()

print(f"=== Generating {args.dataset.upper()} Visualizations ===\n")

# Load config and dataset
cfg = get_config(args.dataset)
dataset = build_dataset(cfg.data.test)

# Load predictions
with open(args.predictions, 'rb') as f:
    predictions = pickle.load(f)

# Create output directory
output_dir = f"{args.output_dir}_{args.dataset}"
os.makedirs(output_dir, exist_ok=True)

print(f"Dataset: {args.dataset}")
print(f"Samples in dataset: {len(dataset)}")
print(f"Predictions loaded: {len(predictions)}")
print(f"Visualizing: {min(args.num_samples, len(dataset))} samples")
print(f"Score threshold: {args.score_thresh}")
print(f"Output directory: {output_dir}\n")

print("Generating visualizations...")

# Visualize samples
for sample_idx in range(min(args.num_samples, len(dataset))):
    pred = predictions[sample_idx]

    # Handle different dataset formats for GT
    if args.dataset == 'kitti':
        # KITTI: Use get_ann_info to get GT in LiDAR coordinates
        ann_info = dataset.get_ann_info(sample_idx)
        gt_bboxes_3d = ann_info['gt_bboxes_3d']
        gt_names = ann_info['gt_names']
        gt_labels = ann_info['gt_labels_3d']

        # Filter out objects not in target classes (label == -1)
        valid_mask = gt_labels >= 0
        gt_boxes = gt_bboxes_3d.tensor.cpu().numpy()[valid_mask]
        gt_names = gt_names[valid_mask]

        # Load point cloud for KITTI
        gt_info = dataset.data_infos[sample_idx]
        lidar_path = gt_info['point_cloud']['velodyne_path']
        # Prepend data root if path is relative
        if not os.path.isabs(lidar_path):
            lidar_path = os.path.join('data/kitti', lidar_path)
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    else:
        # NuScenes/Lyft format
        gt_info = dataset.data_infos[sample_idx]
        lidar_path = gt_info['lidar_path']
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
        gt_boxes = gt_info['gt_boxes']
        gt_names = gt_info['gt_names']

    # Get pred boxes
    pred_boxes = pred['boxes_3d'].tensor.cpu().numpy() if hasattr(pred['boxes_3d'], 'tensor') else pred['boxes_3d']
    pred_labels = pred['labels_3d'].cpu().numpy() if hasattr(pred['labels_3d'], 'cpu') else pred['labels_3d']
    pred_scores = pred['scores_3d'].cpu().numpy() if hasattr(pred['scores_3d'], 'cpu') else pred['scores_3d']

    # Filter predictions by score
    valid_mask = pred_scores > args.score_thresh
    pred_boxes = pred_boxes[valid_mask]
    pred_labels = pred_labels[valid_mask]
    pred_scores = pred_scores[valid_mask]

    # Create figure
    fig = plt.figure(figsize=(16, 8))

    # BEV (Bird's Eye View)
    ax1 = fig.add_subplot(121)
    ax1.scatter(points[:, 0], points[:, 1], s=0.1, c='gray', alpha=0.3)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_xlim([-5, 80])
    ax1.set_ylim([-40, 40])
    ax1.set_aspect('equal', adjustable='box')

    # Plot GT boxes in green
    for i, (box, name) in enumerate(zip(gt_boxes, gt_names)):
        x, y, z, l, w, h, yaw = box
        # Draw box corners
        corners_x = [x + l/2 * np.cos(yaw) - w/2 * np.sin(yaw),
                     x + l/2 * np.cos(yaw) + w/2 * np.sin(yaw),
                     x - l/2 * np.cos(yaw) + w/2 * np.sin(yaw),
                     x - l/2 * np.cos(yaw) - w/2 * np.sin(yaw),
                     x + l/2 * np.cos(yaw) - w/2 * np.sin(yaw)]
        corners_y = [y + l/2 * np.sin(yaw) + w/2 * np.cos(yaw),
                     y + l/2 * np.sin(yaw) - w/2 * np.cos(yaw),
                     y - l/2 * np.sin(yaw) - w/2 * np.cos(yaw),
                     y - l/2 * np.sin(yaw) + w/2 * np.cos(yaw),
                     y + l/2 * np.sin(yaw) + w/2 * np.cos(yaw)]
        ax1.plot(corners_x, corners_y, 'g-', linewidth=2, label=f'GT {name}' if i == 0 else '')

    # Plot pred boxes with class-specific colors
    class_colors = {
        'car': 'red',
        'pedestrian': 'blue',
        'bicycle': 'orange',
        'truck': 'purple',
        'bus': 'brown',
        'motorcycle': 'cyan',
        # KITTI class names (capitalized)
        'Car': 'red',
        'Pedestrian': 'blue',
        'Cyclist': 'orange'
    }

    for i, (box, label, score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
        x, y, z, l, w, h, yaw = box
        cls_name = dataset.CLASSES[int(label)]
        color = class_colors.get(cls_name, 'red')

        corners_x = [x + l/2 * np.cos(yaw) - w/2 * np.sin(yaw),
                     x + l/2 * np.cos(yaw) + w/2 * np.sin(yaw),
                     x - l/2 * np.cos(yaw) + w/2 * np.sin(yaw),
                     x - l/2 * np.cos(yaw) - w/2 * np.sin(yaw),
                     x + l/2 * np.cos(yaw) - w/2 * np.sin(yaw)]
        corners_y = [y + l/2 * np.sin(yaw) + w/2 * np.cos(yaw),
                     y + l/2 * np.sin(yaw) - w/2 * np.cos(yaw),
                     y - l/2 * np.sin(yaw) - w/2 * np.cos(yaw),
                     y - l/2 * np.sin(yaw) + w/2 * np.cos(yaw),
                     y + l/2 * np.sin(yaw) + w/2 * np.cos(yaw)]
        ax1.plot(corners_x, corners_y, '--', color=color, linewidth=1.5, alpha=0.7)

        # Add text label on the box
        ax1.text(x, y, f'{cls_name}\n{score:.2f}',
                color=color, fontsize=8, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=color))

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', label='Ground Truth')]
    for cls_name, color in class_colors.items():
        if any(dataset.CLASSES[int(label)] == cls_name for label in pred_labels):
            legend_elements.append(Patch(facecolor=color, label=f'Pred: {cls_name}'))
    ax1.legend(handles=legend_elements, loc='upper right')

    # Statistics
    ax2 = fig.add_subplot(122)
    ax2.axis('off')

    # Build prediction class distribution
    pred_class_counts = {}
    for label in pred_labels:
        cls_name = dataset.CLASSES[int(label)]
        pred_class_counts[cls_name] = pred_class_counts.get(cls_name, 0) + 1

    stats_text = f"""
Sample {sample_idx} Statistics:

Ground Truth:
  Total: {len(gt_boxes)}
  Classes: {dict((name, list(gt_names).count(name)) for name in set(gt_names))}

Predictions (score > {args.score_thresh}):
  Total: {len(pred_boxes)}
  Classes: {pred_class_counts}

Prediction Details:
"""
    for cls_name, count in sorted(pred_class_counts.items(), key=lambda x: -x[1]):
        stats_text += f"  {cls_name}: {count}\n"

    # Add average scores per class
    stats_text += "\nAvg Scores by Class:\n"
    for cls_idx in range(len(dataset.CLASSES)):
        cls_scores = [score for label, score in zip(pred_labels, pred_scores) if label == cls_idx]
        if cls_scores:
            avg_score = np.mean(cls_scores)
            stats_text += f"  {dataset.CLASSES[cls_idx]}: {avg_score:.3f}\n"

    ax2.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
            verticalalignment='center')

    # Save figure
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sample_{sample_idx:03d}.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved visualization for sample {sample_idx}")

print(f"\nVisualizations saved to {output_dir}/")
print(f"Generated {min(args.num_samples, len(dataset))} images")
