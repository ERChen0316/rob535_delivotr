"""
Filter Lyft ground truth to KITTI detection range.
Similar to what was done for nuScenes.

Usage:
    python filter_lyft_gt.py
"""

import pickle
import numpy as np
import copy

# KITTI's detection range
KITTI_RANGE = [0, -40.32, -3, 80.64, 40.32, 1]

print("=== Filtering Lyft GT to KITTI Range ===\n")
print(f"KITTI range: {KITTI_RANGE}\n")

# Load original Lyft annotations
input_file = 'data/lyft/lyft_infos_val.pkl'
output_file = 'data/lyft/lyft_infos_val_kitti_range.pkl'

try:
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
except FileNotFoundError:
    print(f"ERROR: {input_file} not found!")
    print("Make sure Lyft dataset is prepared according to MMDetection3D instructions.")
    exit(1)

print(f"Loaded data from: {input_file}")
print(f"Data keys: {list(data.keys())}")

# Get the infos list
if 'infos' in data:
    infos = data['infos']
elif 'data_list' in data:
    infos = data['data_list']
else:
    print("ERROR: Could not find 'infos' or 'data_list' in pickle file")
    exit(1)

print(f"Total samples: {len(infos)}")

# Filter annotations
filtered_infos = []
total_original_boxes = 0
total_filtered_boxes = 0
class_counts_before = {}
class_counts_after = {}

for sample in infos:
    filtered_sample = copy.deepcopy(sample)

    if 'gt_boxes' in sample and 'gt_names' in sample:
        gt_boxes = sample['gt_boxes']
        gt_names = sample['gt_names']

        # Count original boxes
        total_original_boxes += len(gt_boxes)
        for name in gt_names:
            class_counts_before[name] = class_counts_before.get(name, 0) + 1

        # Filter boxes within KITTI range
        mask = (
            (gt_boxes[:, 0] >= KITTI_RANGE[0]) &  # x_min
            (gt_boxes[:, 0] <= KITTI_RANGE[3]) &  # x_max
            (gt_boxes[:, 1] >= KITTI_RANGE[1]) &  # y_min
            (gt_boxes[:, 1] <= KITTI_RANGE[4]) &  # y_max
            (gt_boxes[:, 2] >= KITTI_RANGE[2]) &  # z_min
            (gt_boxes[:, 2] <= KITTI_RANGE[5])    # z_max
        )

        filtered_sample['gt_boxes'] = gt_boxes[mask]
        filtered_sample['gt_names'] = gt_names[mask]

        # Also filter velocity if present
        if 'gt_velocity' in sample:
            filtered_sample['gt_velocity'] = sample['gt_velocity'][mask]

        # Count filtered boxes
        total_filtered_boxes += len(filtered_sample['gt_boxes'])
        for name in filtered_sample['gt_names']:
            class_counts_after[name] = class_counts_after.get(name, 0) + 1

    filtered_infos.append(filtered_sample)

# Create new filtered dataset
filtered_data = copy.deepcopy(data)
if 'infos' in data:
    filtered_data['infos'] = filtered_infos
elif 'data_list' in data:
    filtered_data['data_list'] = filtered_infos

# Save filtered annotations
with open(output_file, 'wb') as f:
    pickle.dump(filtered_data, f)

print(f"\n=== Filtering Results ===")
print(f"Total original boxes: {total_original_boxes}")
print(f"Total filtered boxes: {total_filtered_boxes}")
print(f"Kept: {100 * total_filtered_boxes / total_original_boxes:.1f}%")

print(f"\nClass distribution BEFORE filtering:")
for cls, count in sorted(class_counts_before.items(), key=lambda x: -x[1]):
    print(f"  {cls}: {count}")

print(f"\nClass distribution AFTER filtering:")
for cls, count in sorted(class_counts_after.items(), key=lambda x: -x[1]):
    kept_pct = 100 * count / class_counts_before[cls] if cls in class_counts_before else 0
    print(f"  {cls}: {count} ({kept_pct:.1f}% kept)")

print(f"\n✓ Filtered annotations saved to: {output_file}")
print(f"\nNow you can run evaluation with this filtered GT file.")
