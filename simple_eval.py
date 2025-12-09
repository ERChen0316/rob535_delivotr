import pickle
import numpy as np
from mmcv import Config
from mmdet3d.datasets import build_dataset

# Load config and dataset (using full config with model definition)
cfg = Config.fromfile('configs/delivotr_nuscenes_mini_kitti_compatible.py')
# Update to use filtered GT
cfg.data.test.ann_file = 'data/nuscenes-v1.0-mini/nuscenes_infos_val_kitti_range.pkl'
# Use all 10 nuScenes classes for proper evaluation
cfg.data.test.classes = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                          'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']
dataset = build_dataset(cfg.data.test)

# Load remapped predictions
with open('delivotr_nuscenes_remapped.pkl', 'rb') as f:
    predictions = pickle.load(f)

print("=== Simple Distance-Based Evaluation ===\n")

# Use relaxed distance thresholds
distance_thresholds = [2.0, 3.0, 5.0, 10.0]
target_classes = ['car', 'pedestrian', 'bicycle']

results = {cls: {thresh: {'tp': 0, 'fp': 0, 'fn': 0} for thresh in distance_thresholds} for cls in target_classes}
total_gt = {cls: 0 for cls in target_classes}
total_pred = {cls: 0 for cls in target_classes}

for sample_idx in range(len(dataset)):
    gt_info = dataset.data_infos[sample_idx]
    pred = predictions[sample_idx]

    if 'gt_boxes' not in gt_info or 'boxes_3d' not in pred:
        continue

    gt_boxes = gt_info['gt_boxes']
    gt_names = gt_info['gt_names']

    pred_boxes = pred['boxes_3d'].tensor.cpu().numpy() if hasattr(pred['boxes_3d'], 'tensor') else pred['boxes_3d']
    pred_labels = pred['labels_3d'].cpu().numpy() if hasattr(pred['labels_3d'], 'cpu') else pred['labels_3d']
    pred_scores = pred['scores_3d'].cpu().numpy() if hasattr(pred['scores_3d'], 'cpu') else pred['scores_3d']

    # Filter predictions by score threshold
    score_thresh = 0.1
    valid_pred = pred_scores >= score_thresh
    pred_boxes = pred_boxes[valid_pred]
    pred_labels = pred_labels[valid_pred]

    for cls in target_classes:
        cls_idx = dataset.CLASSES.index(cls)

        # Get GT boxes for this class
        gt_cls_mask = np.array([name == cls for name in gt_names])
        gt_cls_boxes = gt_boxes[gt_cls_mask]
        total_gt[cls] += len(gt_cls_boxes)

        # Get pred boxes for this class
        pred_cls_mask = pred_labels == cls_idx
        pred_cls_boxes = pred_boxes[pred_cls_mask]
        total_pred[cls] += len(pred_cls_boxes)

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
                    results[cls][thresh]['tp'] += 1
                else:
                    results[cls][thresh]['fp'] += 1

            # Unmatched GT are false negatives
            results[cls][thresh]['fn'] += len(gt_cls_boxes) - len(matched_gt)

# Compute precision, recall, F1
print("Results by class and distance threshold:\n")
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
print(f"\nNote: Using score threshold >= {score_thresh}")
