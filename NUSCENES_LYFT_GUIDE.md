# DeLiVoTr: nuScenes & Lyft Evaluation Guide

Reference guide for running the KITTI-trained DeLiVoTr model on nuScenes and Lyft datasets.

---

## Prerequisites

```bash
# Activate environment after creating and installing dependencies based on DeliVoTr
source delivotr_env/bin/activate

# Set environment variables (may differ based on system)
export PYTHONPATH=$(pwd):$PYTHONPATH
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```
## Data Preparation
Set up data through mmdetection3d pipeline referenced by original DeliVoTr paper

---

## nuScenes Evaluation

### Complete Workflow

```bash
# Step 1: Filter GT to KITTI range (one-time)
python3 create_filtered_gt.py

# Step 2: Generate predictions
python3 tools/test.py \
  configs/delivotr_nuscenes_mini_kitti_compatible.py \
  delivotr_kitti.pth \
  --out predictions_nuscenes.pkl

# Step 3: Remap class indices
python3 remap_predictions.py \
  predictions_nuscenes.pkl \
  predictions_nuscenes_remapped.pkl

# Step 4: Evaluate
python3 simple_eval_multi.py \
  --dataset nuscenes \
  --predictions predictions_nuscenes_remapped.pkl \
  --score-thresh 0.1

# Step 5: Visualize (optional)
python3 simple_visualize.py \
  --dataset nuscenes \
  --predictions predictions_nuscenes_remapped.pkl \
  --num-samples 5 \
  --score-thresh 0.1
```

### Required Files
- Input: `data/nuscenes-v1.0-mini/nuscenes_infos_val.pkl`
- Checkpoint: `delivotr_kitti.pth`
- Config: `configs/delivotr_nuscenes_mini_kitti_compatible.py`

### Generated Files
- `data/nuscenes-v1.0-mini/nuscenes_infos_val_kitti_range.pkl`
- `predictions_nuscenes.pkl`
- `predictions_nuscenes_remapped.pkl`
- `simple_visualizations_nuscenes/*.png`

---

## Lyft Evaluation

### Complete Workflow

```bash
# Step 1: Filter GT to KITTI range (one-time)
python3 filter_lyft_gt.py

# Step 2: Generate predictions
python3 tools/test.py \
  configs/delivotr_lyft_kitti_compatible.py \
  delivotr_kitti.pth \
  --out predictions_lyft.pkl

# Step 3: Remap class indices
python3 remap_predictions.py \
  predictions_lyft.pkl \
  predictions_lyft_remapped.pkl

# Step 4: Evaluate
python3 simple_eval_multi.py \
  --dataset lyft \
  --predictions predictions_lyft_remapped.pkl \
  --score-thresh 0.1

# Step 5: Visualize (optional)
python3 simple_visualize.py \
  --dataset lyft \
  --predictions predictions_lyft_remapped.pkl \
  --num-samples 5 \
  --score-thresh 0.1
```

### Required Files
- Input: `data/lyft/lyft_infos_val.pkl`
- Checkpoint: `delivotr_kitti.pth`
- Config: `configs/delivotr_lyft_kitti_compatible.py`

### Generated Files
- `data/lyft/lyft_infos_val_kitti_range.pkl`
- `predictions_lyft.pkl`
- `predictions_lyft_remapped.pkl`
- `simple_visualizations_lyft/*.png`

---

## Script Options

### simple_eval_multi.py
```bash
python3 simple_eval_multi.py \
  --dataset <nuscenes|lyft|kitti> \
  --predictions <path-to-predictions.pkl> \
  --score-thresh <threshold>  # default: 0.1
```

### simple_visualize.py
```bash
python3 simple_visualize.py \
  --dataset <nuscenes|lyft|kitti> \
  --predictions <path-to-predictions.pkl> \
  --num-samples <number>      # default: 5
  --score-thresh <threshold>  # default: 0.1
  --output-dir <directory>    # default: simple_visualizations
```

---