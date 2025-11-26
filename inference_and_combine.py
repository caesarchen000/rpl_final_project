#!/usr/bin/env python3
"""
Run inference and automatically combine hand + object into mesh files
Usage: 
  python inference_and_combine.py --obj_path <object.ply> [options]
  python inference_and_combine.py --eval (for test set)
"""
import os
import sys
import subprocess
import argparse
import glob


def run_inference(obj_path=None, eval_mode=False, n_samples=10, save_root='exp/results', **kwargs):
    """Run inference using demo.py or eval.py"""
    if eval_mode:
        print("Running inference on test set...")
        cmd = ['python', 'eval.py', '--n_samples', str(n_samples), '--save_root', save_root]
        for key, val in kwargs.items():
            if val is not None:
                cmd.extend([f'--{key}', str(val)])
    else:
        if obj_path is None:
            print("Error: --obj_path required for single object inference")
            return False
        print(f"Running inference on: {obj_path}")
        cmd = ['python', 'demo.py', '--obj_path', obj_path, '--n_samples', str(n_samples), '--save_root', save_root]
        for key, val in kwargs.items():
            if val is not None:
                cmd.extend([f'--{key}', str(val)])
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.getcwd())
    return result.returncode == 0


def combine_grasps(save_root, output_dir='meshes'):
    """Combine all generated grasps with their objects"""
    import trimesh
    
    os.makedirs(output_dir, exist_ok=True)
    combined_count = 0
    
    # Find all grasp files
    grasp_files = glob.glob(os.path.join(save_root, '**', 'grasp_*.obj'), recursive=True)
    grasp_files.sort()
    
    if not grasp_files:
        print(f"No grasp files found in {save_root}")
        return 0
    
    print(f"\nFound {len(grasp_files)} grasp files")
    
    # Group grasps by object
    obj_grasps = {}
    for grasp_file in grasp_files:
        dir_path = os.path.dirname(grasp_file)
        obj_name = os.path.basename(dir_path)
        
        # Find object mesh in the same directory
        obj_file = None
        for ext in ['.ply', '.obj']:
            obj_candidates = glob.glob(os.path.join(dir_path, f'*{ext}'))
            # Exclude grasp files
            obj_candidates = [f for f in obj_candidates if 'grasp' not in os.path.basename(f).lower()]
            if obj_candidates:
                obj_file = obj_candidates[0]
                break
        
        if obj_file is None:
            # Try parent directory or look for object in grab_data
            obj_base = obj_name
            if os.path.exists(f'grab_data/obj_meshes/{obj_base}.ply'):
                obj_file = f'grab_data/obj_meshes/{obj_base}.ply'
            else:
                print(f"Warning: Could not find object mesh for {grasp_file}, skipping...")
                continue
        
        if obj_name not in obj_grasps:
            obj_grasps[obj_name] = {'obj_file': obj_file, 'grasps': []}
        obj_grasps[obj_name]['grasps'].append(grasp_file)
    
    # Combine each grasp with its object
    for obj_name, data in obj_grasps.items():
        obj_file = data['obj_file']
        print(f"\nProcessing {obj_name} ({len(data['grasps'])} grasps)...")
        
        for grasp_file in data['grasps']:
            grasp_num = os.path.basename(grasp_file).replace('grasp_', '').replace('.obj', '')
            output_file = os.path.join(output_dir, f'{obj_name}_grasp_{grasp_num}.obj')
            
            try:
                # Load and combine
                hand_mesh = trimesh.load(grasp_file)
                obj_mesh = trimesh.load(obj_file)
                
                # Ensure coordinate alignment - use object from same directory (centered version)
                # If object is from grab_data, it needs to be centered
                if 'grab_data' in obj_file:
                    # Center object to match inference coordinate space
                    offset = obj_mesh.vertices.mean(axis=0, keepdims=True)
                    obj_mesh.vertices = obj_mesh.vertices - offset
                
                combined = trimesh.util.concatenate([hand_mesh, obj_mesh])
                combined.export(output_file)
                combined_count += 1
                print(f"  ✓ Combined: {output_file}")
            except Exception as e:
                print(f"  ✗ Failed to combine {grasp_file}: {e}")
    
    return combined_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference and combine hand+object meshes')
    
    # Inference options
    parser.add_argument('--obj_path', type=str, default=None, help='Object mesh path (for single object inference)')
    parser.add_argument('--eval', action='store_true', help='Run inference on test set instead of single object')
    parser.add_argument('--n_samples', type=int, default=10, help='Number of samples per object')
    parser.add_argument('--save_root', type=str, default='exp/results', help='Directory to save inference results')
    parser.add_argument('--checkpoint', type=str, default='checkpoint/checkpoint.pt', help='Model checkpoint path')
    
    # Contact solver options
    parser.add_argument('--w_contact', type=float, default=1e-1, help='Contact weight')
    parser.add_argument('--w_pene', type=float, default=3.0, help='Penetration weight')
    parser.add_argument('--w_uv', type=float, default=1e-2, help='UV weight')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    # Combine options
    parser.add_argument('--output_dir', type=str, default='meshes', help='Directory to save combined meshes')
    parser.add_argument('--skip_inference', action='store_true', help='Skip inference, only combine existing results')
    
    args = parser.parse_args()
    
    # Set environment variables for CUDA
    env = os.environ.copy()
    if 'CUDA_HOME' not in env:
        # Try to detect from conda
        conda_prefix = os.environ.get('CONDA_PREFIX', '')
        if conda_prefix:
            env['CUDA_HOME'] = conda_prefix
            env['PATH'] = f"{conda_prefix}/bin:{env.get('PATH', '')}"
            env['LD_LIBRARY_PATH'] = f"{conda_prefix}/lib:{conda_prefix}/lib64:{env.get('LD_LIBRARY_PATH', '')}"
    
    # Run inference
    if not args.skip_inference:
        kwargs = {
            'checkpoint': args.checkpoint,
            'w_contact': args.w_contact,
            'w_pene': args.w_pene,
            'w_uv': args.w_uv,
            'seed': args.seed
        }
        success = run_inference(args.obj_path, args.eval, args.n_samples, args.save_root, **kwargs)
        if not success:
            print("Inference failed!")
            sys.exit(1)
        print("\n✓ Inference completed successfully")
    else:
        print("Skipping inference (using existing results)")
    
    # Combine meshes
    print(f"\nCombining meshes and saving to: {args.output_dir}")
    count = combine_grasps(args.save_root, args.output_dir)
    print(f"\n✓ Complete! Created {count} combined mesh files in {args.output_dir}/")

