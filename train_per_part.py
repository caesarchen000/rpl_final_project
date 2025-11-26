import argparse
import os
from contactgen.utils.cfg_parser import Config
from contactgen.trainer_per_part import TrainerPerPart


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ContactGen Training - Per-Part Contact Maps')

    parser.add_argument('--work-dir', default='./exp_per_part', type=str, help='exp dir')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='Training batch size')
    parser.add_argument('--lr', default=8e-4, type=float,
                        help='Training learning rate')
    parser.add_argument("--config_path", type=str, default="contactgen/configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None, help='Path to checkpoint.pt to resume from')
    parser.add_argument('--n-epochs', default=1000, type=int,
                        help='Number of training epochs (default: 1000, recommended: 500-1000)')
    
    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)

    cwd = os.getcwd()
    cfg_path = args.config_path

    # Auto-detect checkpoint if not specified and one exists
    checkpoint_path = args.resume
    if checkpoint_path is None:
        auto_checkpoint = os.path.join(args.work_dir, 'checkpoint.pt')
        if os.path.exists(auto_checkpoint):
            checkpoint_path = auto_checkpoint
            print(f"Found existing checkpoint: {checkpoint_path}")
            print("Training will resume from this checkpoint.")
            print("To start fresh, delete it or use --resume with a different path.")

    cfg = {
        'batch_size': args.batch_size,
        'base_lr': args.lr,
        'base_dir': cwd,
        'work_dir': args.work_dir,
        'checkpoint': checkpoint_path,  # Use resume checkpoint if provided or auto-detected
        'n_epochs': args.n_epochs,  # Override default epochs
    }

    cfg = Config(default_cfg_path=cfg_path, **cfg)
    cf_trainer = TrainerPerPart(cfg=cfg)
    cf_trainer.fit()

