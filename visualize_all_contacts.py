#!/usr/bin/env python3
"""
Automatically visualize contact maps for all generated grasps
Usage: python visualize_all_contacts.py [--results_dir exp/demo_results] [options]
"""
import argparse
import os
import glob
import numpy as np
from visualize_contact_map import visualize_contact_map


def find_object_mesh(result_dir, obj_name=None):
    """Find the object mesh in the results directory"""
    # Look for .ply or .obj files (excluding grasp files)
    for ext in ['.ply', '.obj']:
        candidates = glob.glob(os.path.join(result_dir, f'*{ext}'))
        candidates = [f for f in candidates if 'grasp' not in os.path.basename(f).lower()]
        if candidates:
            return candidates[0]
    
    # If not found, try to construct from obj_name
    if obj_name:
        for ext in ['.ply', '.obj']:
            candidate = os.path.join(result_dir, f'{obj_name}{ext}')
            if os.path.exists(candidate):
                return candidate
    
    return None


def visualize_all_contacts(results_dir='exp/demo_results', output_dir=None, colormap='hot', threshold=None, obj_mesh_path=None):
    """
    Visualize contact maps for all grasps in results directory
    """
    if output_dir is None:
        output_dir = os.path.join(results_dir, 'contact_visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all contact map files
    contact_map_files = glob.glob(os.path.join(results_dir, 'contact_map_*.npy'))
    contact_map_files.sort()
    
    if not contact_map_files:
        # Try to find contact_maps.npy and extract individual maps
        contact_maps_file = os.path.join(results_dir, 'contact_maps.npy')
        if os.path.exists(contact_maps_file):
            print(f"Found contact_maps.npy, extracting individual maps...")
            contact_maps = np.load(contact_maps_file)  # (n_samples, 2048)
            for i in range(len(contact_maps)):
                contact_map_file = os.path.join(results_dir, f'contact_map_{i}.npy')
                np.save(contact_map_file, contact_maps[i])
            contact_map_files = glob.glob(os.path.join(results_dir, 'contact_map_*.npy'))
            contact_map_files.sort()
    
    if not contact_map_files:
        print(f"No contact map files found in {results_dir}")
        print(f"Expected: contact_map_*.npy or contact_maps.npy")
        print(f"\nNote: Contact maps are only saved when running inference with the updated demo.py/eval.py")
        print(f"Please run inference again to generate contact maps.")
        return
    
    print(f"Found {len(contact_map_files)} contact map files")
    
    # Find object mesh
    if obj_mesh_path and os.path.exists(obj_mesh_path):
        obj_mesh = obj_mesh_path
    else:
        obj_mesh = find_object_mesh(results_dir)
        if obj_mesh is None:
            print(f"Error: Could not find object mesh in {results_dir}")
            print(f"Please provide --obj_mesh path")
            print(f"\nExample:")
            print(f"  python visualize_all_contacts.py --results_dir {results_dir} --obj_mesh exp/demo_results/O02_0015_00026.obj")
            return
    
    print(f"Using object mesh: {obj_mesh}")
    
    # Visualize each contact map
    for contact_map_file in contact_map_files:
        grasp_num = os.path.basename(contact_map_file).replace('contact_map_', '').replace('.npy', '')
        output_file = os.path.join(output_dir, f'contact_map_{grasp_num}.obj')
        
        try:
            print(f"\nVisualizing contact map {grasp_num}...")
            visualize_contact_map(
                obj_mesh,
                contact_map_file,
                output_file,
                colormap=colormap,
                threshold=threshold
            )
        except Exception as e:
            print(f"  ✗ Failed to visualize {contact_map_file}: {e}")
    
    print(f"\n✓ Visualized {len(contact_map_files)} contact maps")
    print(f"  Output directory: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize contact maps for all generated grasps')
    parser.add_argument('--results_dir', type=str, default='exp/demo_results', 
                        help='Directory containing inference results')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for visualizations (default: results_dir/contact_visualizations)')
    parser.add_argument('--obj_mesh', type=str, default=None,
                        help='Object mesh path (auto-detected if not provided)')
    parser.add_argument('--colormap', type=str, default='hot',
                        choices=['hot', 'viridis', 'plasma', 'inferno', 'coolwarm', 'red'],
                        help='Colormap to use')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Threshold contact values (only show above threshold)')
    
    args = parser.parse_args()
    
    visualize_all_contacts(
        args.results_dir,
        args.output_dir,
        colormap=args.colormap,
        threshold=args.threshold,
        obj_mesh_path=args.obj_mesh
    )

