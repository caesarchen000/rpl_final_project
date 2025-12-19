"""
Visualize partition map (which hand part) with contact map (contact probability) combined.

This shows:
- Colors = which hand part (from partition map)
- Brightness = contact probability (from contact map)
"""
import os
import argparse
import numpy as np
import trimesh
from scipy.spatial import cKDTree


# Hand part names and colors
# Mapping (following your convention):
#   0      : palm
#   1 ~ 3  : index finger (base, middle, tip)
#   4 ~ 6  : middle finger (base, middle, tip)
#   7 ~ 9  : ring finger (base, middle, tip)
#   10 ~ 12: pinky (base, middle, tip)
#   13 ~ 15: thumb (base, middle, tip)
PART_NAMES = [
    "palm",        # 0
    "index_0",     # 1
    "index_1",     # 2
    "index_2",     # 3
    "middle_0",    # 4
    "middle_1",    # 5
    "middle_2",    # 6
    "ring_0",      # 7
    "ring_1",      # 8
    "ring_2",      # 9
    "pinky_0",     # 10
    "pinky_1",     # 11
    "pinky_2",     # 12
    "thumb_0",     # 13
    "thumb_1",     # 14
    "thumb_2"      # 15
]

# Palette tuned for clarity:
#   palm  -> browns
#   index -> yellows
#   middle-> greens
#   ring  -> blues
#   pinky -> purples
#   thumb -> reds
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


def visualize_partition_with_contact(obj_mesh_path, partition_hard_path, contact_map_path,
                                     sample_points_path, output_path, sample_idx=0,
                                     brightness_scale=1.0, min_brightness=0.2):
    """
    Visualize partition map (hand part colors) modulated by contact probability (brightness).
    
    Args:
        obj_mesh_path: Path to object mesh
        partition_hard_path: Path to partition_hard.npy [B, N] or [N]
        contact_map_path: Path to contact_map.npy [B, N] or [N]
        sample_points_path: Path to sample_points.npy [N, 3]
        output_path: Output OBJ file path
        sample_idx: Which sample to visualize
        brightness_scale: Scale factor for brightness (default: 1.0)
        min_brightness: Minimum brightness (default: 0.2)
    """
    
    # Load object mesh
    print(f"Loading object mesh from {obj_mesh_path}")
    obj_mesh = trimesh.load(obj_mesh_path)
    mesh_vertices = obj_mesh.vertices
    n_vertices = len(mesh_vertices)
    
    # Load partition map
    print(f"Loading partition map from {partition_hard_path}")
    partition_hard = np.load(partition_hard_path)
    if len(partition_hard.shape) == 2:
        partition_hard = partition_hard[sample_idx]
    print(f"Partition shape: {partition_hard.shape}")
    
    # Load contact map
    print(f"Loading contact map from {contact_map_path}")
    contact_map = np.load(contact_map_path)
    if len(contact_map.shape) == 2:
        contact_map = contact_map[sample_idx]
    print(f"Contact shape: {contact_map.shape}")
    
    # Load sample points
    print(f"Loading sample points from {sample_points_path}")
    sample_points = np.load(sample_points_path)
    print(f"Sample points shape: {sample_points.shape}")
    
    # Map from sample_points to mesh_vertices
    if len(sample_points) == n_vertices and np.allclose(sample_points, mesh_vertices, atol=1e-6):
        vertex_partitions = partition_hard
        vertex_contacts = contact_map
    else:
        print(f"Mapping {len(sample_points)} sample points to {n_vertices} mesh vertices...")
        tree = cKDTree(sample_points)
        distances, indices = tree.query(mesh_vertices, k=1)
        vertex_partitions = partition_hard[indices]
        vertex_contacts = contact_map[indices]
    
    # Get base colors for each hand part
    colors = PART_COLORS[vertex_partitions]  # [N, 3]
    
    # Modulate brightness by contact probability
    contact_normalized = np.clip(vertex_contacts * brightness_scale, min_brightness, 1.0)
    colors = colors * contact_normalized[:, np.newaxis]
    colors = np.clip(colors, 0.0, 1.0)
    
    # Create colored mesh
    colored_mesh = trimesh.Trimesh(
        vertices=mesh_vertices,
        faces=obj_mesh.faces,
        vertex_colors=colors
    )
    
    # Save
    colored_mesh.export(output_path)
    print(f"✓ Saved to {output_path}")
    
    # Print statistics
    print(f"\nVisualization summary:")
    print(f"  Colors = Hand part assignment (partition map)")
    print(f"  Brightness = Contact probability (contact map)")
    print(f"  Formula: final_color = part_color * contact_probability")
    print(f"\nPartition statistics:")
    for part_id in range(16):
        mask = vertex_partitions == part_id
        if mask.sum() > 0:
            avg_contact = vertex_contacts[mask].mean()
            print(f"  Part {part_id:2d} ({PART_NAMES[part_id]:12s}): "
                  f"{mask.sum():5d} vertices, avg contact: {avg_contact:.3f}")
    
    return colored_mesh


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize partition map (hand part) with contact map (brightness)'
    )
    parser.add_argument('--obj_path', type=str, required=True)
    parser.add_argument('--partition_hard', type=str, required=True)
    parser.add_argument('--contact_map', type=str, required=True)
    parser.add_argument('--sample_points', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--sample_idx', type=int, default=0)
    parser.add_argument('--brightness_scale', type=float, default=1.0)
    parser.add_argument('--min_brightness', type=float, default=0.2)
    
    args = parser.parse_args()
    
    visualize_partition_with_contact(
        args.obj_path,
        args.partition_hard,
        args.contact_map,
        args.sample_points,
        args.output,
        args.sample_idx,
        args.brightness_scale,
        args.min_brightness
    )

