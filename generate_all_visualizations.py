#!/usr/bin/env python3
"""
Complete pipeline: Generate contact/partition maps and create all visualizations.

This script combines:
1. Generate maps using partial_demo.py (contact map, partition map, sample points)
2. Visualize contact heatmap
3. Visualize partition map
4. Visualize partition × contact combined

Usage:
    python generate_all_visualizations.py \
      --obj_path <path_to_obj_file> \
      --output_dir <output_directory>
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print("\n" + "="*70)
    print(f"{description}")
    print("="*70)
    print(f"Running: {' '.join(cmd)}")
    print("="*70)
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n✗ Error: {description} failed!")
        sys.exit(1)
    
    print(f"\n✓ {description} completed successfully!")
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Generate contact/partition maps and all visualizations from OBJ file'
    )
    parser.add_argument('--obj_path', type=str, required=True,
                       help='Path to input object mesh file (.ply or .obj)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for all results')
    parser.add_argument('--checkpoint', type=str, default='checkpoint/checkpoint.pt',
                       help='Path to ContactGen model checkpoint (default: checkpoint/checkpoint.pt)')
    parser.add_argument('--n_samples', type=int, default=1,
                       help='Number of samples to generate (default: 1)')
    parser.add_argument('--w_contact', type=float, default=0.1,
                       help='Contact loss weight (default: 0.1)')
    parser.add_argument('--w_pene', type=float, default=3.0,
                       help='Penetration loss weight (default: 3.0)')
    parser.add_argument('--w_uv', type=float, default=0.01,
                       help='UV loss weight (default: 0.01)')
    parser.add_argument('--brightness_scale', type=float, default=1.0,
                       help='Brightness scale for contact modulation (default: 1.0)')
    parser.add_argument('--min_brightness', type=float, default=0.0,
                       help='Minimum brightness for contact modulation (default: 0.0)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.obj_path):
        print(f"✗ Error: Object file not found: {args.obj_path}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get object name for output files
    obj_name = Path(args.obj_path).stem
    
    print("\n" + "="*70)
    print("Complete Visualization Pipeline")
    print("="*70)
    print(f"Input object: {args.obj_path}")
    print(f"Output directory: {args.output_dir}")
    print("="*70)
    
    # Step 1: Generate maps using partial_demo.py
    print("\n[Step 1/4] Generating contact and partition maps...")
    cmd_step1 = [
        'python', 'partial_demo.py',
        '--obj_path', args.obj_path,
        '--checkpoint', args.checkpoint,
        '--save_root', args.output_dir,
        '--n_samples', str(args.n_samples),
        '--w_contact', str(args.w_contact),
        '--w_pene', str(args.w_pene),
        '--w_uv', str(args.w_uv)
    ]
    run_command(cmd_step1, "Step 1: Generate maps")
    
    # Verify required files were created
    contact_map_path = os.path.join(args.output_dir, 'contact_map.npy')
    partition_hard_path = os.path.join(args.output_dir, 'part_hard.npy')
    sample_points_path = os.path.join(args.output_dir, 'sample_points.npy')
    
    required_files = [contact_map_path, partition_hard_path, sample_points_path]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"\n✗ Error: Required files not found after Step 1:")
        for f in missing_files:
            print(f"  - {f}")
        sys.exit(1)
    
    print(f"\n✓ All required files generated:")
    print(f"  - {contact_map_path}")
    print(f"  - {partition_hard_path}")
    print(f"  - {sample_points_path}")
    
    # Step 2: Visualize contact heatmap
    print("\n[Step 2/4] Visualizing contact heatmap...")
    heatmap_output = os.path.join(args.output_dir, 'heatmap.obj')
    cmd_step2 = [
        'python', 'visualize_contact_map.py',
        '--obj_path', args.obj_path,
        '--contact_map', contact_map_path,
        '--sample_points', sample_points_path,
        '--output', heatmap_output
    ]
    run_command(cmd_step2, "Step 2: Visualize contact heatmap")
    
    # Step 3: Visualize partition map
    print("\n[Step 3/4] Visualizing partition map...")
    partition_output = os.path.join(args.output_dir, 'partition.obj')
    cmd_step3 = [
        'python', 'visualize_partition_map.py',
        '--obj_path', args.obj_path,
        '--partition_hard', partition_hard_path,
        '--sample_points', sample_points_path,
        '--output', partition_output
    ]
    run_command(cmd_step3, "Step 3: Visualize partition map")
    
    # Step 4: Visualize partition × contact combined
    print("\n[Step 4/4] Visualizing partition × contact combined...")
    combined_output = os.path.join(args.output_dir, 'partition_contact.obj')
    cmd_step4 = [
        'python', 'visualize_partition_contact_multiply.py',
        '--obj_path', args.obj_path,
        '--partition_hard', partition_hard_path,
        '--contact_map', contact_map_path,
        '--sample_points', sample_points_path,
        '--output', combined_output,
        '--brightness_scale', str(args.brightness_scale),
        '--min_brightness', str(args.min_brightness)
    ]
    run_command(cmd_step4, "Step 4: Visualize partition × contact combined")
    
    # Summary
    print("\n" + "="*70)
    print("Pipeline Complete! All visualizations generated.")
    print("="*70)
    print(f"\nOutput files in: {args.output_dir}")
    print("\nGenerated visualizations:")
    print(f"  1. Contact heatmap:     {heatmap_output}")
    print(f"  2. Partition map:       {partition_output}")
    print(f"  3. Partition × Contact: {combined_output}")
    print("\nGenerated data files:")
    print(f"  - contact_map.npy       (contact probabilities)")
    print(f"  - part_hard.npy         (partition assignments)")
    print(f"  - sample_points.npy      (sample point coordinates)")
    print(f"  - part_logits.npy       (raw partition logits)")
    print(f"  - part_probs.npy        (partition probabilities)")
    print(f"  - grasp_0.obj           (optimized hand mesh)")
    print("="*70)


if __name__ == '__main__':
    main()

