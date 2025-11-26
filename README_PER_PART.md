# Per-Part Contact Maps Implementation

This directory contains a modified version of ContactGen that outputs **per-part contact maps** instead of a single combined contact map.

## What Changed

- **Original**: Outputs single contact map `[B, N, 1]` for all hand parts combined
- **Per-Part**: Outputs separate contact maps `[B, N, 16]` - one channel for each of the 16 hand parts

## Files Created

### Core Model Files:
- `contactgen/model_per_part.py` - `ContactGenModelPerPart` class
- `contactgen/hand_object_per_part.py` - `HandObjectPerPart` class  
- `contactgen/trainer_per_part.py` - `TrainerPerPart` class

### Scripts:
- `train_per_part.py` - Training script
- `demo_per_part.py` - Single object inference
- `eval_per_part.py` - Batch evaluation on test set

### Optimizer:
- `contactgen/contact/contact_optimizer_per_part.py` - Modified optimizer

## Usage

### 1. Train the Model

```bash
python train_per_part.py --work-dir ./exp_per_part --batch-size 128 --lr 8e-4
```

This will:
- Train a new model from scratch with per-part contact map architecture
- Save checkpoints to `./exp_per_part/checkpoints/`
- Save final checkpoint to `./exp_per_part/checkpoint.pt`

### 2. Run Inference (Single Object)

```bash
python demo_per_part.py \
    --obj_path assets/toothpaste.ply \
    --n_samples 5 \
    --save_root exp/demo_results_per_part \
    --checkpoint exp_per_part/checkpoint.pt
```

**Output:**
- `contact_maps.npy` - Shape: `[n_samples, 2048, 16]` (16 channels, one per hand part)
- `contact_map_0.npy`, etc. - Individual per-sample maps: `[2048, 16]`
- `grasp_*.obj` - Hand mesh files

### 3. Run Evaluation (Test Set)

```bash
python eval_per_part.py \
    --checkpoint exp_per_part/checkpoint.pt \
    --n_samples 10 \
    --save_root exp/results_per_part
```

## Output Format

The contact maps now have shape `[n_samples, 2048, 16]` where:
- `n_samples`: Number of generated grasps
- `2048`: Number of sampled object points
- `16`: One channel for each hand part (0=palm, 1-3=thumb, 4-6=index, 7-9=middle, 10-12=ring, 13-15=pinky)

To access contact map for a specific part:
```python
contact_maps = np.load('contact_maps.npy')  # [n_samples, 2048, 16]
thumb_contact = contact_maps[:, :, 1]  # Get thumb contact map (part 1)
```

## Important Notes

1. **Retraining Required**: The architecture change requires training from scratch. The original checkpoint won't work.

2. **Original Files Preserved**: All original files (`model.py`, `trainer.py`, `demo.py`, etc.) remain unchanged.

3. **Ground Truth**: During training, per-part ground truth is automatically computed from the grasp pose - each point's contact value is assigned to the channel corresponding to its nearest hand part.

## Next Steps

1. Train the model: `python train_per_part.py`
2. Wait for training to complete (checkpoints saved periodically)
3. Test inference: `python demo_per_part.py --obj_path assets/toothpaste.ply`
4. Verify output shape: Check that `contact_maps.npy` has shape `[n_samples, 2048, 16]`

