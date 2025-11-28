"""
Visualize contact map as a heatmap on the object mesh.
Shows contact probability (0-1) as a color gradient from blue (low) to red (high).
"""
import os
import argparse
import numpy as np
import trimesh
from scipy.spatial import cKDTree
import matplotlib.cm as cm


def visualize_contact_map(obj_path, contact_map_path, sample_points_path, output_path, 
                        sample_idx=0, colormap='hot'):
    """
    Visualize contact map as a heatmap on object mesh.
    
    Args:
        obj_path: Path to object mesh (.ply or .obj)
        contact_map_path: Path to contact_map.npy [B, N] or [N] - contact probabilities (0-1)
        sample_points_path: Path to sample_points.npy [N, 3] - 3D points where map is defined
        output_path: Output file path (.obj or .ply)
        sample_idx: Which sample to use if batch dimension exists (default: 0)
        colormap: Matplotlib colormap name (default: 'hot')
    """
    print("="*70)
    print("Contact Map Visualization (Heatmap)")
    print("="*70)
    print(f"Colormap: {colormap} (blue=low contact, red=high contact)")
    print("="*70)
    
    # Load object mesh
    print(f"\n[1/4] Loading object mesh: {obj_path}")
    obj_mesh = trimesh.load(obj_path, process=False)
    mesh_vertices = np.array(obj_mesh.vertices)
    n_vertices = len(mesh_vertices)
    print(f"  Object: {n_vertices} vertices, {len(obj_mesh.faces)} faces")
    
    # Load contact map
    print(f"\n[2/4] Loading contact map: {contact_map_path}")
    contact_map = np.load(contact_map_path)
    original_contact_shape = contact_map.shape
    
    # Handle different shapes: [N], [N, 1], [B, N], [B, N, 1]
    if len(contact_map.shape) == 3:
        # [B, N, 1] -> select sample and squeeze
        contact_map = contact_map[sample_idx].squeeze()
    elif len(contact_map.shape) == 2:
        # Could be [B, N] or [N, 1]
        if contact_map.shape[1] == 1:
            # [N, 1] -> squeeze to [N]
            contact_map = contact_map.squeeze()
        else:
            # [B, N] -> select sample
            contact_map = contact_map[sample_idx]
    elif len(contact_map.shape) == 1:
        # [N] -> use as is
        pass
    else:
        raise ValueError(f"Unexpected contact map shape: {contact_map.shape}")
    
    contact_map = contact_map.flatten()
    print(f"  Contact shape: {original_contact_shape} -> {contact_map.shape}")
    print(f"  Contact range: [{contact_map.min():.4f}, {contact_map.max():.4f}] (should be 0-1)")
    
    # Load sample points
    print(f"\n[3/4] Loading sample points: {sample_points_path}")
    sample_points = np.load(sample_points_path)
    if len(sample_points.shape) == 3:
        sample_points = sample_points[sample_idx]
    print(f"  Sample points shape: {sample_points.shape}")
    
    # Verify alignment
    assert len(contact_map) == len(sample_points), \
        f"Contact map ({len(contact_map)}) and sample points ({len(sample_points)}) must have same length!"
    print(f"  ✓ Maps are aligned: {len(contact_map)} points")
    
    # Map contact values from sample points to mesh vertices
    print(f"\n[4/4] Mapping contact values from {len(sample_points)} sample points to {n_vertices} mesh vertices...")
    tree = cKDTree(sample_points)
    _, nearest_indices = tree.query(mesh_vertices, k=1)
    vertex_contact = contact_map[nearest_indices]
    
    print(f"  Contact range on mesh: [{vertex_contact.min():.4f}, {vertex_contact.max():.4f}]")
    
    # Normalize contact values to [0, 1] for colormap
    contact_normalized = (vertex_contact - vertex_contact.min()) / (vertex_contact.max() - vertex_contact.min() + 1e-8)
    
    # Apply colormap
    try:
        cmap = cm.colormaps[colormap]
    except (KeyError, AttributeError):
        cmap = cm.get_cmap(colormap)
    vertex_colors = cmap(contact_normalized)[:, :3]  # RGB only, ignore alpha
    
    # Apply colors to mesh
    obj_mesh.visual.vertex_colors = (vertex_colors * 255).astype(np.uint8)
    
    # Save
    print(f"\nSaving contact map visualization to {output_path}...")
    obj_mesh.export(output_path)
    print(f"✓ Saved to {output_path}")
    
    # Statistics
    print(f"\nContact Map Statistics:")
    print(f"  Min contact: {vertex_contact.min():.4f}")
    print(f"  Max contact: {vertex_contact.max():.4f}")
    print(f"  Mean contact: {vertex_contact.mean():.4f}")
    print(f"  Median contact: {np.median(vertex_contact):.4f}")
    
    print("\n" + "="*70)
    print("Visualization complete!")
    print("  - Blue areas: Low contact probability")
    print("  - Red areas: High contact probability")
    print("="*70)
    
    return obj_mesh


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize contact map as heatmap')
    parser.add_argument('--obj_path', type=str, required=True,
                        help='Path to object mesh (.ply or .obj)')
    parser.add_argument('--contact_map', type=str, required=True,
                        help='Path to contact_map.npy')
    parser.add_argument('--sample_points', type=str, required=True,
                        help='Path to sample_points.npy')
    parser.add_argument('--output', type=str, required=True,
                       help='Output file path (.obj or .ply)')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Sample index if batch dimension exists (default: 0)')
    parser.add_argument('--colormap', type=str, default='hot',
                        help='Matplotlib colormap name (default: hot)')
    
    args = parser.parse_args()
    
    visualize_contact_map(
        args.obj_path,
        args.contact_map,
        args.sample_points,
        args.output,
        args.sample_idx,
        args.colormap
    )

