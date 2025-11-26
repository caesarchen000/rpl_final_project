#!/usr/bin/env python3
"""
Visualize contact maps on object meshes
Usage: python visualize_contact_map.py --obj_mesh <object.obj> --contact_map <contact.npy> [options]
"""
import argparse
import numpy as np
import trimesh
import os


def map_contact_to_colors(contact_values, colormap='hot', vmin=None, vmax=None):
    """
    Map contact values (0-1) to RGB colors
    Args:
        contact_values: array of contact values (N,)
        colormap: 'hot', 'viridis', 'plasma', 'inferno', 'coolwarm', or 'red'
        vmin, vmax: value range for normalization (auto-detect if None)
    Returns:
        colors: RGB colors array (N, 3) in range [0, 255]
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    # Auto-detect range if not provided, but use actual min/max for better contrast
    if vmin is None:
        vmin = contact_values.min()
    if vmax is None:
        vmax = contact_values.max()
    
    # Normalize values - use percentile-based scaling for better contrast
    # For contact maps, we want to emphasize the high-contact regions
    # Use a more aggressive scaling that stretches the upper range
    p2 = np.percentile(contact_values, 2)  # Bottom 2%
    p98 = np.percentile(contact_values, 98)  # Top 98%
    
    # Map values: anything below p2 -> 0, p2 to p98 -> 0 to 0.8, above p98 -> 0.8 to 1.0
    # This stretches the high-contact regions for better visibility
    contact_clipped = np.clip(contact_values, p2, p98)
    contact_norm = (contact_clipped - p2) / (p98 - p2 + 1e-6)
    
    # Apply gamma correction to emphasize high values
    contact_norm = np.power(contact_norm, 0.7)  # Gamma < 1 makes bright regions brighter
    
    # Get colormap
    if colormap == 'hot':
        cmap = cm.hot
    elif colormap == 'viridis':
        cmap = cm.viridis
    elif colormap == 'plasma':
        cmap = cm.plasma
    elif colormap == 'inferno':
        cmap = cm.inferno
    elif colormap == 'coolwarm':
        cmap = cm.coolwarm
    elif colormap == 'red':
        # Simple red gradient (white to red)
        colors = np.zeros((len(contact_norm), 3))
        colors[:, 0] = 255  # Red channel
        colors[:, 1] = (1 - contact_norm) * 255  # Green (less contact = more green)
        colors[:, 2] = (1 - contact_norm) * 255  # Blue (less contact = more blue)
        return colors.astype(np.uint8)
    else:
        cmap = cm.hot
    
    # Map to colors
    colors = cmap(contact_norm)[:, :3]  # (N, 3) RGB in [0, 1]
    colors = (colors * 255).astype(np.uint8)
    
    return colors


def visualize_contact_map(obj_mesh_path, contact_map_path, output_path=None, colormap='hot', threshold=None):
    """
    Visualize contact map on object mesh
    Args:
        obj_mesh_path: path to object mesh (.obj or .ply)
        contact_map_path: path to contact map (.npy file with shape (N,) or (N, 1))
        output_path: where to save colored mesh (if None, auto-generate)
        colormap: colormap name
        threshold: if provided, only show contact above this threshold
    """
    # Check if files exist
    if not os.path.exists(obj_mesh_path):
        raise FileNotFoundError(f"Object mesh not found: {obj_mesh_path}")
    if not os.path.exists(contact_map_path):
        raise FileNotFoundError(f"Contact map not found: {contact_map_path}\n"
                              f"Note: Contact maps are only saved when running inference with the updated demo.py/eval.py")
    
    print(f"Loading object mesh: {obj_mesh_path}")
    obj_mesh = trimesh.load(obj_mesh_path)
    print(f"  Object: {len(obj_mesh.vertices)} vertices, {len(obj_mesh.faces)} faces")
    
    print(f"Loading contact map: {contact_map_path}")
    contact_map = np.load(contact_map_path)
    
    # Handle different shapes
    if contact_map.ndim == 2:
        if contact_map.shape[1] == 1:
            contact_map = contact_map.squeeze()
        else:
            raise ValueError(f"Unexpected contact map shape: {contact_map.shape}")
    
    print(f"  Contact map: {contact_map.shape} values, range [{contact_map.min():.3f}, {contact_map.max():.3f}]")
    
    # Check if contact map matches mesh vertices
    if len(contact_map) != len(obj_mesh.vertices):
        print(f"  Warning: Contact map size ({len(contact_map)}) != mesh vertices ({len(obj_mesh.vertices)})")
        print(f"  This is a point cloud contact map. Mapping to mesh vertices...")
        
        # Interpolate contact values from point cloud to mesh vertices using nearest neighbors
        try:
            from scipy.spatial import cKDTree
            
            # Try to load sample points if available
            # Look in the same directory as contact_map
            contact_dir = os.path.dirname(contact_map_path)
            sample_points_path = os.path.join(contact_dir, 'sample_points.npy')
            if os.path.exists(sample_points_path):
                print(f"  Loading sample points from: {sample_points_path}")
                sample_points = np.load(sample_points_path)
                # Build KDTree from sample points
                tree = cKDTree(sample_points)
                # Find nearest sample point for each mesh vertex
                distances, indices = tree.query(obj_mesh.vertices, k=1)
                # Map contact values to mesh vertices
                contact_map = contact_map[indices]
                print(f"  Mapped contact map from {len(sample_points)} points to {len(obj_mesh.vertices)} vertices")
            else:
                # Fallback: use distance-weighted interpolation
                print(f"  Sample points not found, using distance-weighted interpolation...")
                # This is a simplified approach - for better results, provide sample_points.npy
                tree = cKDTree(obj_mesh.vertices[:len(contact_map)])  # Use first N vertices as reference
                distances, indices = tree.query(obj_mesh.vertices, k=min(3, len(contact_map)))
                if distances.ndim == 1:
                    # Single nearest neighbor
                    contact_map = contact_map[indices]
                else:
                    # Weighted average of k nearest neighbors
                    weights = 1.0 / (distances + 1e-6)
                    weights = weights / weights.sum(axis=1, keepdims=True)
                    contact_map = (contact_map[indices] * weights).sum(axis=1)
                print(f"  Interpolated contact map to mesh vertices")
        except ImportError:
            print(f"  Warning: scipy not available, using simple mapping...")
            # Simple fallback: repeat or sample
            if len(contact_map) < len(obj_mesh.vertices):
                n_repeats = len(obj_mesh.vertices) // len(contact_map) + 1
                contact_map = np.tile(contact_map, n_repeats)[:len(obj_mesh.vertices)]
            else:
                contact_map = contact_map[:len(obj_mesh.vertices)]
    
    # Apply threshold if specified
    if threshold is not None:
        contact_map = np.where(contact_map >= threshold, contact_map, 0.0)
        print(f"  Applied threshold: {threshold}")
    
    # Map to colors
    colors = map_contact_to_colors(contact_map, colormap=colormap)
    
    # Apply colors to mesh - use both vertex and face colors for better viewer compatibility
    obj_mesh.visual.vertex_colors = colors
    
    # Also set face colors (average of vertex colors for each face) - many viewers prefer this
    if len(obj_mesh.faces) > 0:
        # Get face colors by averaging vertex colors of each face
        face_colors = colors[obj_mesh.faces].mean(axis=1).astype(np.uint8)
        obj_mesh.visual.face_colors = face_colors
    
    # Generate output path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(obj_mesh_path))[0]
        output_dir = os.path.dirname(obj_mesh_path) or '.'
        output_path = os.path.join(output_dir, f'{base_name}_contact_map.obj')
    
    # Save colored mesh
    obj_mesh.export(output_path)
    print(f"✓ Saved colored mesh to: {output_path}")
    
    # Also save as PLY (better color support)
    ply_path = output_path.replace('.obj', '.ply')
    obj_mesh.export(ply_path)
    print(f"✓ Saved colored mesh to: {ply_path}")
    
    return obj_mesh, contact_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize contact maps on object meshes')
    parser.add_argument('--obj_mesh', type=str, required=True, help='Object mesh path (.obj or .ply)')
    parser.add_argument('--contact_map', type=str, required=True, help='Contact map file (.npy)')
    parser.add_argument('--output', type=str, default=None, help='Output path for colored mesh (auto-generated if not provided)')
    parser.add_argument('--colormap', type=str, default='hot', 
                        choices=['hot', 'viridis', 'plasma', 'inferno', 'coolwarm', 'red'],
                        help='Colormap to use')
    parser.add_argument('--threshold', type=float, default=None, 
                        help='Threshold contact values (only show above threshold)')
    
    args = parser.parse_args()
    
    visualize_contact_map(
        args.obj_mesh, 
        args.contact_map, 
        args.output, 
        colormap=args.colormap,
        threshold=args.threshold
    )

