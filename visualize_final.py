#!/usr/bin/env python3
"""
Visualize final.npy file with hand part colors.

The final.npy file has shape [N, 16] where each row has at most one non-zero value
representing the brightness at the winning hand part.
"""
import os
import argparse
import numpy as np
import trimesh
from scipy.spatial import cKDTree


# Hand part names and colors (same as visualize_partition_contact_multiply.py)
PART_NAMES = [
    "palm",        # 0
    "index_0",     # 1
    "index_1",     # 2
    "index_2",     # 3
    "middle_0",    # 4
    "middle_1",    # 5
    "middle_2",    # 6
    "ring_0",      # 7
    "ring_1",     # 8
    "ring_2",      # 9
    "pinky_0",     # 10
    "pinky_1",     # 11
    "pinky_2",     # 12
    "thumb_0",     # 13
    "thumb_1",     # 14
    "thumb_2"      # 15
]

PART_COLORS = np.array([
    [0.60, 0.35, 0.15],   # 0: palm     - brown
    [1.00, 0.90, 0.30],   # 1: index_0  - light yellow
    [1.00, 0.80, 0.05],   # 2: index_1  - medium yellow
    [0.95, 0.70, 0.00],   # 3: index_2  - dark yellow / ochre
    [0.20, 0.90, 0.20],   # 4: middle_0 - bright green
    [0.00, 0.80, 0.40],   # 5: middle_1 - teal green
    [0.00, 0.60, 0.00],   # 6: middle_2 - dark green
    [0.30, 0.60, 1.00],   # 7: ring_0   - sky blue
    [0.10, 0.35, 0.95],   # 8: ring_1   - medium blue
    [0.00, 0.10, 0.80],   # 9: ring_2   - deep blue
    [0.80, 0.50, 1.00],   # 10: pinky_0 - light purple
    [0.65, 0.30, 0.95],   # 11: pinky_1 - medium purple
    [0.50, 0.15, 0.80],   # 12: pinky_2 - deep purple
    [1.00, 0.35, 0.35],   # 13: thumb_0 - light red
    [0.90, 0.15, 0.15],   # 14: thumb_1 - medium red
    [0.70, 0.05, 0.05],   # 15: thumb_2 - deep red
])


def visualize_final(
    obj_path,
    final_npy_path,
    sample_points_path,
    output_path,
    sample_idx=0,
):
    """
    Visualize final.npy file with hand part colors.
    
    Args:
        obj_path: Path to object mesh (.ply or .obj)
        final_npy_path: Path to final.npy [N, 16] - one-hot encoded contact probabilities
        sample_points_path: Path to sample_points.npy [N, 3] - 3D points where maps are defined
        output_path: Output file path (.obj or .ply)
        sample_idx: Which sample to use if batch dimension exists (default: 0)
    """
    print("="*70)
    print("Visualizing final.npy with Hand Part Colors")
    print("="*70)
    
    # Load object mesh
    print(f"\n[1/4] Loading object mesh: {obj_path}")
    obj_mesh = trimesh.load(obj_path)
    if isinstance(obj_mesh, trimesh.Scene):
        obj_mesh = obj_mesh.dump(concatenate=True)
    mesh_vertices = np.array(obj_mesh.vertices)
    n_vertices = len(mesh_vertices)
    print(f"  Object: {n_vertices} vertices, {len(obj_mesh.faces)} faces")
    
    # Center the mesh (same as in visualize_partition_contact_multiply.py)
    mesh_center = mesh_vertices.mean(axis=0)
    mesh_vertices = mesh_vertices - mesh_center
    print(f"  Centered mesh (offset: {mesh_center})")
    
    # Load final.npy
    print(f"\n[2/4] Loading final.npy: {final_npy_path}")
    final_data = np.load(final_npy_path)
    print(f"  Final shape: {final_data.shape}")
    
    # Handle batch dimension if present
    if len(final_data.shape) == 3:
        final_data = final_data[sample_idx]
    elif len(final_data.shape) != 2:
        raise ValueError(f"Unexpected final.npy shape: {final_data.shape}, expected [N, 16] or [B, N, 16]")
    
    if final_data.shape[1] != 16:
        raise ValueError(f"Final data must have 16 columns (hand parts), got {final_data.shape[1]}")
    
    n_points = final_data.shape[0]
    print(f"  Processed shape: {final_data.shape}")
    
    # Extract hand part assignments and brightness values
    # For each point, find which index has the non-zero value (should be at most one)
    part_assignments = np.full(n_points, -1, dtype=np.int32)  # -1 means no part assigned (black)
    brightness_values = np.zeros(n_points, dtype=np.float32)
    
    for i in range(n_points):
        row = final_data[i]
        nonzero_indices = np.where(row > 0)[0]
        
        if len(nonzero_indices) == 0:
            # All zeros - will be black
            part_assignments[i] = -1
            brightness_values[i] = 0.0
        elif len(nonzero_indices) == 1:
            # Exactly one non-zero value - use its index as hand part
            part_idx = nonzero_indices[0]
            part_assignments[i] = part_idx
            brightness_values[i] = row[part_idx]
        else:
            # Multiple non-zero values (shouldn't happen, but handle it)
            # Use the one with maximum value
            part_idx = np.argmax(row)
            part_assignments[i] = part_idx
            brightness_values[i] = row[part_idx]
            print(f"  ⚠ Warning: Point {i} has {len(nonzero_indices)} non-zero values, using max")
    
    # Points with part_assignments == -1 (all zeros in row) will be black
    valid_points_mask = part_assignments >= 0
    
    print(f"  Points with assigned hand part: {np.sum(valid_points_mask)} / {n_points}")
    print(f"  Points with no hand part (black): {np.sum(~valid_points_mask)} / {n_points}")
    if np.any(valid_points_mask):
        print(f"  Brightness range: [{brightness_values[valid_points_mask].min():.4f}, {brightness_values[valid_points_mask].max():.4f}]")
    
    # Load sample points
    print(f"\n[3/4] Loading sample points: {sample_points_path}")
    sample_points = np.load(sample_points_path)
    print(f"  Sample points shape: {sample_points.shape}")
    
    # Handle batch dimension if present
    if len(sample_points.shape) == 3:
        sample_points = sample_points[sample_idx]
    elif len(sample_points.shape) != 2:
        raise ValueError(f"Unexpected sample_points shape: {sample_points.shape}")
    
    if sample_points.shape[1] != 3:
        raise ValueError(f"Sample points must have 3 columns (x, y, z), got {sample_points.shape[1]}")
    
    # Center sample points to match mesh
    sample_points = sample_points - sample_points.mean(axis=0)
    
    if len(sample_points) != n_points:
        raise ValueError(f"Sample points ({len(sample_points)}) and final data ({n_points}) must have same length!")
    
    # Map from sample points to mesh vertices
    print(f"\n[4/4] Mapping data from {len(sample_points)} sample points to {n_vertices} mesh vertices...")
    
    if len(sample_points) == n_vertices:
        max_distance = np.max(np.linalg.norm(sample_points - mesh_vertices, axis=1))
        if max_distance < 1e-4:
            print(f"  Sample points match mesh vertices (max diff: {max_distance:.2e}) - using direct assignment")
            vertex_parts = part_assignments.copy()
            vertex_brightness = brightness_values.copy()
        else:
            print(f"  Same number but positions differ (max diff: {max_distance:.2e}) - using nearest neighbor")
            tree_samples = cKDTree(sample_points)
            distances, indices = tree_samples.query(mesh_vertices, k=1)
            vertex_parts = part_assignments[indices]
            vertex_brightness = brightness_values[indices]
    else:
        print(f"  Different number of points - using nearest neighbor mapping")
        tree_samples = cKDTree(sample_points)
        distances, indices = tree_samples.query(mesh_vertices, k=1)
        vertex_parts = part_assignments[indices]
        vertex_brightness = brightness_values[indices]
    
    # Apply colors
    print(f"\nApplying colors...")
    vertex_colors = np.zeros((n_vertices, 3), dtype=np.float32)
    
    # For vertices with assigned hand part (part_assignments != -1), use hand part color modulated by brightness
    # Vertices with part_assignments == -1 remain black (already zeros)
    valid_mask = vertex_parts >= 0  # Valid hand part assigned (not -1)
    if np.any(valid_mask):
        # Get base color for the assigned hand part
        vertex_colors[valid_mask] = PART_COLORS[vertex_parts[valid_mask]]
        # Modulate by brightness
        vertex_colors[valid_mask] *= vertex_brightness[valid_mask, np.newaxis]
        vertex_colors[valid_mask] = np.clip(vertex_colors[valid_mask], 0.0, 1.0)
    
    # Vertices with part_assignments == -1 remain black (already zeros)
    
    print(f"  Colored vertices (with assigned hand part): {np.sum(valid_mask)} / {n_vertices}")
    print(f"  Black vertices (no hand part assigned): {np.sum(~valid_mask)} / {n_vertices}")
    print(f"  Color range: [{vertex_colors.min():.3f}, {vertex_colors.max():.3f}]")
    
    # Print statistics
    print(f"\nPart statistics:")
    unique_parts, counts = np.unique(vertex_parts[valid_mask], return_counts=True) if np.any(valid_mask) else ([], [])
    for part_id, count in zip(unique_parts, counts):
        percentage = count / np.sum(valid_mask) * 100 if np.any(valid_mask) else 0
        part_mask = (vertex_parts == part_id) & valid_mask
        avg_brightness = vertex_brightness[part_mask].mean() if np.any(part_mask) else 0.0
        print(f"  Part {part_id:2d} ({PART_NAMES[part_id]:12s}): "
              f"{count:5d} vertices ({percentage:5.2f}%), avg brightness: {avg_brightness:.3f}")
    
    # Create colored mesh
    print(f"\nSaving colored mesh to {output_path}...")
    colored_mesh = trimesh.Trimesh(
        vertices=mesh_vertices,
        faces=obj_mesh.faces,
        vertex_colors=vertex_colors
    )
    colored_mesh.export(output_path)
    print(f"  ✓ Saved colored mesh with {n_vertices} vertices")
    
    return colored_mesh


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize final.npy file with hand part colors',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python visualize_final.py \\
      --obj_path assets/toothpaste.ply \\
      --final_npy tmp/tmp/final.npy \\
      --sample_points tmp/tmp/sample_points.npy \\
      --output tmp/tmp/final_visualization.obj
        """
    )
    parser.add_argument('--obj_path', type=str, required=True,
                       help='Path to object mesh (.ply or .obj)')
    parser.add_argument('--final_npy', type=str, required=True,
                       help='Path to final.npy (one-hot encoded contact probabilities [N, 16])')
    parser.add_argument('--sample_points', type=str, required=True,
                       help='Path to sample_points.npy (3D points where maps are defined)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output file path (.obj or .ply)')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Which sample to use if batch dimension exists (default: 0)')
    
    args = parser.parse_args()
    
    visualize_final(
        args.obj_path,
        args.final_npy,
        args.sample_points,
        args.output,
        args.sample_idx,
    )

