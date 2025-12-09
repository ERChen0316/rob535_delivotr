"""
Remap KITTI class indices to nuScenes class indices.

KITTI model outputs: 0=pedestrian, 1=cyclist, 2=car
nuScenes expects: 0=car, 5=bicycle, 7=pedestrian

Usage:
    python remap_predictions.py <input.pkl> <output.pkl>
"""

import pickle
import numpy as np
import sys
import copy

def remap_predictions(input_file, output_file):
    # Load predictions
    print(f"Loading predictions from {input_file}...")
    with open(input_file, 'rb') as f:
        predictions = pickle.load(f)

    # KITTI model class order
    KITTI_CLASSES = ['pedestrian', 'bicycle', 'car']  # indices 0, 1, 2

    # nuScenes class order
    NUSCENES_CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                        'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier')

    # Create mapping
    kitti_to_nuscenes = {}
    for kitti_idx, kitti_class in enumerate(KITTI_CLASSES):
        if kitti_class in NUSCENES_CLASSES:
            nuscenes_idx = NUSCENES_CLASSES.index(kitti_class)
            kitti_to_nuscenes[kitti_idx] = nuscenes_idx
            print(f"  KITTI class {kitti_idx} ({kitti_class}) → nuScenes class {nuscenes_idx}")

    # Remap predictions
    print(f"\nRemapping {len(predictions)} samples...")
    remapped_predictions = []
    total_boxes = 0

    for pred in predictions:
        remapped_pred = copy.deepcopy(pred)

        if 'labels_3d' in pred:
            labels = pred['labels_3d'].cpu().numpy() if hasattr(pred['labels_3d'], 'cpu') else pred['labels_3d']
            remapped_labels = np.array([kitti_to_nuscenes.get(int(label), -1) for label in labels])

            # Filter out invalid mappings
            valid_mask = remapped_labels != -1

            if hasattr(pred['labels_3d'], 'cpu'):
                # Keep as tensor
                import torch
                remapped_pred['labels_3d'] = torch.from_numpy(remapped_labels[valid_mask]).long()
                if 'boxes_3d' in pred:
                    remapped_pred['boxes_3d'] = pred['boxes_3d'][torch.from_numpy(valid_mask)]
                if 'scores_3d' in pred:
                    remapped_pred['scores_3d'] = pred['scores_3d'][torch.from_numpy(valid_mask)]
            else:
                # Keep as numpy
                remapped_pred['labels_3d'] = remapped_labels[valid_mask]
                if 'boxes_3d' in pred:
                    remapped_pred['boxes_3d'] = pred['boxes_3d'][valid_mask]
                if 'scores_3d' in pred:
                    remapped_pred['scores_3d'] = pred['scores_3d'][valid_mask]

            total_boxes += len(remapped_labels[valid_mask])

        remapped_predictions.append(remapped_pred)

    # Save remapped predictions
    print(f"\nSaving {total_boxes} remapped boxes to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(remapped_predictions, f)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python remap_predictions.py <input.pkl> <output.pkl>")
        print("Example: python remap_predictions.py results.pkl results_remapped.pkl")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    remap_predictions(input_file, output_file)
