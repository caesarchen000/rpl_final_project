"""
Extract partition map (hand part assignment) from the original ContactGen model.

The partition map indicates which hand part (0-15) each point on the object belongs to.
This script extracts both:
1. Raw partition logits: [B, N, 16] - probability distribution over 16 hand parts
2. Hard partition assignment: [B, N] - which part each point belongs to (argmax)
"""
import os
import argparse
import numpy as np
import torch
import trimesh
from contactgen.utils.cfg_parser import Config
from contactgen.model import ContactGenModel


def extract_partition_map(obj_path, checkpoint_path, n_samples=5, save_root='exp/partition_results', n_points=None):
    """
    Extract partition map from ContactGen model.
    
    Args:
        obj_path: Path to object mesh file
        checkpoint_path: Path to model checkpoint
        n_samples: Number of samples to generate
        save_root: Directory to save results
        n_points: Number of points to sample (None = use all vertices, or specify number)
    """
    os.makedirs(save_root, exist_ok=True)
    
    # Load model
    cfg = Config(default_cfg_path="contactgen/configs/default.yaml")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ContactGenModel(cfg).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
    model.eval()
    
    # Load object
    print(f"Loading object from {obj_path}")
    obj_mesh = trimesh.load(obj_path)
    offset = obj_mesh.vertices.mean(axis=0, keepdims=True)
    obj_verts = obj_mesh.vertices - offset
    obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_mesh.faces)
    
    # Determine number of points to use
    n_vertices = len(obj_mesh.vertices)
    if n_points is None or n_points == 0:
        # Use all vertices for dense visualization
        print(f"Using all {n_vertices} mesh vertices for dense partition map")
        n_points = n_vertices  # For display purposes
        obj_verts = obj_mesh.vertices.astype(np.float32)
        # Compute vertex normals (trimesh will compute if not present)
        if obj_mesh.vertex_normals is None or len(obj_mesh.vertex_normals) == 0:
            obj_mesh.vertex_normals  # Trigger computation
        obj_vn = obj_mesh.vertex_normals.astype(np.float32)
        sample_points = obj_verts.copy()
        use_all_vertices = True
    else:
        # Sample specified number of points
        print(f"Sampling {n_points} points from mesh (has {n_vertices} vertices)")
        sample = trimesh.sample.sample_surface(obj_mesh, n_points)
        obj_verts = sample[0].astype(np.float32)
        obj_vn = obj_mesh.face_normals[sample[1]].astype(np.float32)
        sample_points = obj_verts
        use_all_vertices = False
    
    obj_verts = torch.from_numpy(obj_verts).unsqueeze(dim=0).float().to(device).repeat(n_samples, 1, 1)
    obj_vn = torch.from_numpy(obj_vn).unsqueeze(dim=0).float().to(device).repeat(n_samples, 1, 1)
    
    # Run model
    print("Running ContactGen model to extract partition map...")
    with torch.no_grad():
        sample_results = model.sample(obj_verts, obj_vn)
    
    contacts_object, partition_object, uv_object = sample_results
    # partition_object: [B, N, 16] - logits for each of 16 hand parts
    
    n_points_actual = partition_object.shape[1]
    print(f"\nPartition map shapes:")
    print(f"  partition_object (logits): {partition_object.shape} [B={n_samples}, N={n_points_actual}, 16 parts]")
    
    # Get hard assignment (argmax)
    partition_hard = partition_object.argmax(dim=-1)  # [B, N] - values 0-15
    print(f"  partition_hard (argmax): {partition_hard.shape} [B={n_samples}, N={n_points_actual}]")
    
    # Get softmax probabilities
    partition_probs = torch.softmax(partition_object, dim=-1)  # [B, N, 16]
    print(f"  partition_probs (softmax): {partition_probs.shape} [B={n_samples}, N={n_points_actual}, 16 parts]")
    
    # Convert to numpy
    partition_logits_np = partition_object.detach().cpu().numpy()  # [B, N, 16]
    partition_hard_np = partition_hard.detach().cpu().numpy()  # [B, N]
    partition_probs_np = partition_probs.detach().cpu().numpy()  # [B, N, 16]
    
    # Hand part names for reference
    part_names = [
        "palm",           # 0
        "thumb_mcp",      # 1
        "thumb_pip",      # 2
        "thumb_tip",      # 3
        "index_mcp",      # 4
        "index_pip",      # 5
        "index_tip",      # 6
        "middle_mcp",     # 7
        "middle_pip",     # 8
        "middle_tip",     # 9
        "ring_mcp",       # 10
        "ring_pip",       # 11
        "ring_tip",       # 12
        "pinky_mcp",      # 13
        "pinky_pip",      # 14
        "pinky_tip"       # 15
    ]
    
    # Print statistics
    print(f"\nPartition statistics (sample 0):")
    for part_id in range(16):
        count = (partition_hard_np[0] == part_id).sum()
        percentage = count / len(partition_hard_np[0]) * 100
        print(f"  Part {part_id:2d} ({part_names[part_id]:12s}): {count:4d} points ({percentage:5.2f}%)")
    
    # Save results
    print(f"\nSaving results to {save_root}...")
    
    # Save raw logits
    np.save(os.path.join(save_root, 'partition_logits.npy'), partition_logits_np)
    
    # Save hard assignment
    np.save(os.path.join(save_root, 'partition_hard.npy'), partition_hard_np)
    
    # Save probabilities
    np.save(os.path.join(save_root, 'partition_probs.npy'), partition_probs_np)
    
    # Save sample points
    np.save(os.path.join(save_root, 'sample_points.npy'), sample_points)
    
    # Save individual samples
    for i in range(n_samples):
        np.save(os.path.join(save_root, f'partition_logits_{i}.npy'), partition_logits_np[i])  # [2048, 16]
        np.save(os.path.join(save_root, f'partition_hard_{i}.npy'), partition_hard_np[i])  # [2048]
        np.save(os.path.join(save_root, f'partition_probs_{i}.npy'), partition_probs_np[i])  # [2048, 16]
    
    # Save part names for reference
    with open(os.path.join(save_root, 'part_names.txt'), 'w') as f:
        for part_id, part_name in enumerate(part_names):
            f.write(f"{part_id}: {part_name}\n")
    
    print(f"✓ Saved partition maps:")
    print(f"  - partition_logits.npy: Raw logits [B, N, 16]")
    print(f"  - partition_hard.npy: Hard assignment (argmax) [B, N] - values 0-15")
    print(f"  - partition_probs.npy: Softmax probabilities [B, N, 16]")
    print(f"  - sample_points.npy: Sampled points [N, 3]")
    print(f"  - part_names.txt: Hand part names for reference")
    
    return partition_logits_np, partition_hard_np, partition_probs_np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract partition map from ContactGen model')
    parser.add_argument('--checkpoint', default='checkpoint/checkpoint.pt', type=str,
                       help='Path to model checkpoint')
    parser.add_argument('--obj_path', type=str, required=True,
                       help='Path to object mesh file')
    parser.add_argument('--n_samples', default=5, type=int,
                       help='Number of samples to generate')
    parser.add_argument('--save_root', default='exp/partition_results', type=str,
                       help='Directory to save results')
    parser.add_argument('--n_points', default=2048, type=int,
                       help='Number of points to sample. Use --n_points 0 to use ALL vertices for dense visualization (default: 2048)')
    
    args = parser.parse_args()
    
    # Use None if 0 is specified (means use all vertices)
    n_points_arg = None if args.n_points == 0 else args.n_points
    
    extract_partition_map(
        args.obj_path,
        args.checkpoint,
        args.n_samples,
        args.save_root,
        n_points_arg
    )
    
    print("\nDone! Partition maps extracted successfully.")
    print("\nUsage:")
    print("  partition_logits: Raw model output (logits) - use for soft assignments")
    print("  partition_hard: Hard assignment (argmax) - use for visualization")
    print("  partition_probs: Softmax probabilities - use for uncertainty analysis")

