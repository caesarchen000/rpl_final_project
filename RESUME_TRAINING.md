# Resuming Training

Yes, you can stop and resume training! The trainer saves checkpoints that include:
- Model weights (`state_dict`)
- Epoch number
- Optimizer state
- Learning rate scheduler state

## How to Resume

### Option 1: Automatic Resume (Recommended)
The trainer automatically looks for `checkpoint.pt` in the work directory. If you stop training and restart with the same `--work-dir`, it will automatically resume:

```bash
# Start training
python train_per_part.py --work-dir ./exp_per_part

# Stop training (Ctrl+C or kill process)

# Resume training (same command, automatically resumes)
python train_per_part.py --work-dir ./exp_per_part
```

### Option 2: Explicit Resume
Specify the checkpoint path explicitly:

```bash
python train_per_part.py \
    --work-dir ./exp_per_part \
    --resume ./exp_per_part/checkpoint.pt
```

### Option 3: Resume from Specific Snapshot
Resume from one of the periodic snapshots:

```bash
python train_per_part.py \
    --work-dir ./exp_per_part \
    --resume ./exp_per_part/checkpoints/E1000_net.pt
```

## What Gets Saved

Every epoch, `checkpoint.pt` is saved with:
- `epoch`: Current epoch number (starts from 1)
- `state_dict`: Model weights
- `optimizer`: Optimizer state (for proper learning rate continuation)
- `scheduler`: Learning rate scheduler state

## Example Workflow

```bash
# Day 1: Start training
python train_per_part.py --work-dir ./exp_per_part
# ... training runs to epoch 500 ...
# Stop it (Ctrl+C)

# Day 2: Resume from where you left off
python train_per_part.py --work-dir ./exp_per_part
# Training continues from epoch 501 to 3000
```

## Important Notes

1. **Same work-dir**: Use the same `--work-dir` when resuming, or explicitly specify `--resume` path
2. **Checkpoint location**: The checkpoint is saved to `{work_dir}/checkpoint.pt`
3. **Epoch numbering**: Training resumes from the saved epoch number
4. **Learning rate**: The learning rate schedule continues from where it left off

## Verify Resume

When resuming, you should see in the log:
```
Resuming from epoch 500
Loaded optimizer state
Loaded scheduler state
Load model from ./exp_per_part/checkpoint.pt
--- starting Epoch # 500
```

This confirms training is resuming correctly!


