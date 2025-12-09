"""
Simple distance-based evaluation for zero-shot generalization study.
Works with nuScenes and Lyft datasets.

Usage:
    python simple_eval_multi.py --dataset nuscenes --predictions delivotr_nuscenes_remapped.pkl
    python simple_eval_multi.py --dataset lyft --predictions delivotr_lyft_remapped.pkl
"""

import pickle
import numpy as np
import argparse
from mmcv import Config
from mmdet3d.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Simple distance-based evaluation')
    parser.add_argument('--dataset', type=str, required=True, choices=['nuscenes', 'lyft', 'kitti'],
                        help='Dataset to evaluate on')
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to remapped predictions pickle file')
    parser.add_argument('--score-thresh', type=float, default=0.1,
                        help='Score threshold for predictions')
    return parser.parse_args()


def get_config(dataset_name):
    """Load appropriate config based on dataset."""
    if dataset_name == 'nuscenes':
        cfg = Config.fromfile('configs/delivotr_nuscenes_mini_kitti_compatible.py')
        cfg.data.test.ann_file = 'data/nuscenes-v1.0-mini/nuscenes_infos_val_kitti_range.pkl'
        dataset_type = 'NuScenesDataset'
        # Use all 10 classes for proper evaluation
        cfg.data.test.classes = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']
    elif dataset_name == 'lyft':
        cfg = Config.fromfile('configs/delivotr_lyft_kitti_compatible.py')
        # Use filtered GT (only objects in KITTI range)
        cfg.data.test.ann_file = 'data/lyft/lyft_infos_val_kitti_range.pkl'
        dataset_type = 'LyftDataset'
        # Use all 10 classes for proper evaluation
        cfg.data.test.classes = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']
    elif dataset_name == 'kitti':
        cfg = Config.fromfile('configs/delivotr_kitti.py')
        cfg.data.test.ann_file = 'data/kitti/kitti_infos_val.pkl'
        dataset_type = 'KittiDataset'
        # KITTI uses capitalized class names
        cfg.data.test.classes = ['Pedestrian', 'Cyclist', 'Car']

    return cfg, dataset_type


def main():
    args = parse_args()

    print(f"=== Simple Distance-Based Evaluation ({args.dataset.upper()}) ===\n")

    # Load config and dataset
    cfg, dataset_type = get_config(args.dataset)
    dataset = build_dataset(cfg.data.test)

    # Load remapped predictions
    with open(args.predictions, 'rb') as f:
        predictions = pickle.load(f)

    # Use relaxed distance thresholds
    distance_thresholds = [2.0, 3.0, 5.0, 10.0]

    # Map dataset-specific class names to common evaluation names
    if args.dataset == 'kitti':
        # KITTI uses capitalized names, map to lowercase for evaluation
        class_mapping = {'Car': 'car', 'Pedestrian': 'pedestrian', 'Cyclist': 'bicycle'}
        target_classes = ['car', 'pedestrian', 'bicycle']
        dataset_classes = ['Car', 'Pedestrian', 'Cyclist']
    else:
        # NuScenes and Lyft use lowercase
        class_mapping = {'car': 'car', 'pedestrian': 'pedestrian', 'bicycle': 'bicycle'}
        target_classes = ['car', 'pedestrian', 'bicycle']
        dataset_classes = target_classes

    print(f"Dataset: {dataset_type}")
    print(f"Total samples: {len(dataset)}")
    print(f"Predictions: {len(predictions)}")
    print(f"Score threshold: {args.score_thresh}")
    print(f"Target classes: {target_classes}")
    if args.dataset == 'kitti':
        print(f"Dataset classes: {dataset_classes}")
        print(f"Class mapping: {class_mapping}\n")
    else:
        print()

    results = {cls: {thresh: {'tp': 0, 'fp': 0, 'fn': 0} for thresh in distance_thresholds}
               for cls in target_classes}
    total_gt = {cls: 0 for cls in target_classes}
    total_pred = {cls: 0 for cls in target_classes}

    # Evaluate each sample
    for sample_idx in range(len(dataset)):
        pred = predictions[sample_idx]

        # Handle different dataset formats
        if args.dataset == 'kitti':
            # KITTI: Use get_ann_info to get GT in LiDAR coordinates
            if 'boxes_3d' not in pred:
                continue
            ann_info = dataset.get_ann_info(sample_idx)

            # Get GT boxes and names
            gt_bboxes_3d = ann_info['gt_bboxes_3d']
            gt_names = ann_info['gt_names']
            gt_labels = ann_info['gt_labels_3d']

            # Filter out objects not in our target classes (label == -1)
            valid_mask = gt_labels >= 0
            gt_boxes = gt_bboxes_3d.tensor.cpu().numpy()[valid_mask]
            gt_names = gt_names[valid_mask]
        else:
            # NuScenes/Lyft format
            gt_info = dataset.data_infos[sample_idx]
            if 'gt_boxes' not in gt_info or 'boxes_3d' not in pred:
                continue
            gt_boxes = gt_info['gt_boxes']
            gt_names = gt_info['gt_names']

        # Extract prediction data
        boxes_3d = pred['boxes_3d']
        if hasattr(boxes_3d, 'tensor'):
            pred_boxes = boxes_3d.tensor.cpu().numpy()
        elif hasattr(boxes_3d, 'numpy'):
            pred_boxes = boxes_3d.numpy()
        else:
            pred_boxes = np.array(boxes_3d)

        pred_labels = pred['labels_3d'].cpu().numpy() if hasattr(pred['labels_3d'], 'cpu') else np.array(pred['labels_3d'])
        pred_scores = pred['scores_3d'].cpu().numpy() if hasattr(pred['scores_3d'], 'cpu') else np.array(pred['scores_3d'])

        # Filter predictions by score threshold
        valid_pred = pred_scores >= args.score_thresh
        pred_boxes = pred_boxes[valid_pred]
        pred_labels = pred_labels[valid_pred]

        for dataset_cls in dataset_classes:
            # Map to evaluation class name
            eval_cls = class_mapping[dataset_cls]
            cls_idx = dataset.CLASSES.index(dataset_cls)

            # Get GT boxes for this class
            gt_cls_mask = np.array([name == dataset_cls for name in gt_names])
            gt_cls_boxes = gt_boxes[gt_cls_mask]
            total_gt[eval_cls] += len(gt_cls_boxes)

            # Get pred boxes for this class
            pred_cls_mask = pred_labels == cls_idx
            pred_cls_boxes = pred_boxes[pred_cls_mask]
            total_pred[eval_cls] += len(pred_cls_boxes)

            # For each distance threshold
            for thresh in distance_thresholds:
                matched_gt = set()
                matched_pred = set()

                # Match predictions to GT
                for pred_idx, pred_box in enumerate(pred_cls_boxes):
                    best_match = None
                    best_dist = thresh

                    for gt_idx, gt_box in enumerate(gt_cls_boxes):
                        if gt_idx in matched_gt:
                            continue

                        # 2D distance
                        dist = np.linalg.norm(pred_box[:2] - gt_box[:2])
                        if dist < best_dist:
                            best_dist = dist
                            best_match = gt_idx

                    if best_match is not None:
                        matched_gt.add(best_match)
                        matched_pred.add(pred_idx)
                        results[eval_cls][thresh]['tp'] += 1
                    else:
                        results[eval_cls][thresh]['fp'] += 1

                # Unmatched GT are false negatives
                results[eval_cls][thresh]['fn'] += len(gt_cls_boxes) - len(matched_gt)

    # Print results
    print("\nResults by class and distance threshold:\n")
    print(f"{'Class':<15} {'Thresh':<8} {'TP':<6} {'FP':<6} {'FN':<6} {'Prec':<8} {'Recall':<8} {'F1':<8}")
    print("-" * 80)

    for cls in target_classes:
        for thresh in distance_thresholds:
            tp = results[cls][thresh]['tp']
            fp = results[cls][thresh]['fp']
            fn = results[cls][thresh]['fn']

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            print(f"{cls:<15} {thresh:<8.1f} {tp:<6} {fp:<6} {fn:<6} {precision:<8.3f} {recall:<8.3f} {f1:<8.3f}")

    print(f"\n{'='*80}")
    print(f"Total GT: {total_gt}")
    print(f"Total Predictions: {total_pred}")
    print(f"\nNote: Using score threshold >= {args.score_thresh}")

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY - {args.dataset.upper()} Zero-Shot Evaluation")
    print(f"{'='*80}")
    for cls in target_classes:
        tp_10m = results[cls][10.0]['tp']
        fn_10m = results[cls][10.0]['fn']
        fp_10m = results[cls][10.0]['fp']
        recall_10m = tp_10m / (tp_10m + fn_10m) if (tp_10m + fn_10m) > 0 else 0
        precision_10m = tp_10m / (tp_10m + fp_10m) if (tp_10m + fp_10m) > 0 else 0
        print(f"{cls.capitalize():<12}: Recall@10m = {recall_10m*100:5.1f}%, Precision = {precision_10m*100:5.1f}%")


if __name__ == '__main__':
    main()
