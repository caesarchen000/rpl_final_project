"""
Visualize partition map (hand part colors) multiplied by contact map (contact probability).

This visualization shows:
- Colors = Hand part assignment (from partition map, 16 different colors)
- Brightness = Contact probability (from contact map, values 0-1)
- Final color = partition_color * contact_probability

The result: Each hand part has its own color, and the brightness/intensity
is modulated by how likely that region is to be in contact with the hand.
"""
import os
import argparse
import numpy as np
import trimesh
from scipy.spatial import cKDTree


# Hand part names and colors (16 distinct colors for each hand part)
PART_NAMES = [
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

# Highly saturated, distinct colors for maximum visual separation
PART_COLORS = np.array([
    [0.2, 1.0, 0.6],      # 0: palm - bright turquoise/lime green
    [1.0, 0.0, 0.0],      # 1: thumb_mcp - pure red
    [1.0, 0.5, 0.7],      # 2: thumb_pip - bright coral/salmon
    [1.0, 1.0, 0.0],      # 3: thumb_tip - pure yellow
    [0.0, 0.8, 0.0],      # 4: index_mcp - green
    [0.0, 0.8, 0.8],      # 5: index_pip - cyan
    [0.0, 0.4, 1.0],      # 6: index_tip - bright blue
    [0.0, 0.0, 1.0],      # 7: middle_mcp - pure blue
    [0.6, 0.0, 1.0],      # 8: middle_pip - violet
    [0.8, 0.0, 0.8],      # 9: middle_tip - magenta
    [1.0, 0.0, 0.6],      # 10: ring_mcp - hot pink
    [1.0, 0.4, 0.0],      # 11: ring_pip - red-orange
    [0.8, 0.8, 0.0],      # 12: ring_tip - olive yellow
    [0.0, 0.6, 0.4],      # 13: pinky_mcp - teal green
    [0.4, 0.0, 0.8],      # 14: pinky_pip - dark purple
    [0.8, 0.4, 0.0],      # 15: pinky_tip - brown-orange
])


def visualize_partition_times_contact(obj_path, partition_hard_path, contact_map_path,
                                      sample_points_path, output_path, sample_idx=0,
                                      brightness_scale=1.0, min_brightness=0.0):
    """
    Visualize partition map multiplied by contact map.
    
    Formula: final_color = partition_color * contact_probability
    
    Args:
        obj_path: Path to object mesh (.ply or .obj)
        partition_hard_path: Path to partition_hard.npy [B, N] or [N] - hand part assignments (0-15)
        contact_map_path: Path to contact_map.npy [B, N] or [N] - contact probabilities (0-1)
        sample_points_path: Path to sample_points.npy [N, 3] - 3D points where maps are defined
        output_path: Output file path (.obj or .ply)
        sample_idx: Which sample to use if batch dimension exists (default: 0)
        brightness_scale: Scale factor for contact values (default: 1.0)
        min_brightness: Minimum brightness to ensure colors are visible (default: 0.0)
    
    Returns:
        Colored mesh object
    """
    print("="*70)
    print("Partition Map × Contact Map Visualization")
    print("="*70)
    print("Formula: final_color = partition_color * contact_probability")
    print("  - Colors come from partition map (which hand part)")
    print("  - Brightness comes from contact map (contact probability 0-1)")
    print("="*70)
    
    # Load object mesh
    print(f"\n[1/5] Loading object mesh: {obj_path}")
    obj_mesh = trimesh.load(obj_path, process=False)
    mesh_vertices = np.array(obj_mesh.vertices)
    n_vertices = len(mesh_vertices)
    print(f"  Object: {n_vertices} vertices, {len(obj_mesh.faces)} faces")
    
    # Load partition map (hand part assignments)
    print(f"\n[2/5] Loading partition map: {partition_hard_path}")
    partition_hard = np.load(partition_hard_path)
    original_partition_shape = partition_hard.shape
    if len(partition_hard.shape) == 2:
        partition_hard = partition_hard[sample_idx]
    elif len(partition_hard.shape) == 3:
        partition_hard = partition_hard[sample_idx].squeeze()
    print(f"  Partition shape: {original_partition_shape} -> {partition_hard.shape}")
    print(f"  Part range: [{partition_hard.min()}, {partition_hard.max()}] (should be 0-15)")
    
    # Load contact map (contact probabilities)
    print(f"\n[3/5] Loading contact map: {contact_map_path}")
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
    
    print(f"  Contact shape: {original_contact_shape} -> {contact_map.shape}")
    print(f"  Contact range: [{contact_map.min():.4f}, {contact_map.max():.4f}] (should be 0-1)")
    
    # Load sample points
    print(f"\n[4/5] Loading sample points: {sample_points_path}")
    sample_points = np.load(sample_points_path)
    print(f"  Sample points shape: {sample_points.shape}")
    
    # CRITICAL: Verify alignment - partition map and contact map must be aligned
    # They should have the same length and correspond to the same sample points
    if len(partition_hard) != len(sample_points):
        raise ValueError(f"Partition map ({len(partition_hard)}) and sample_points ({len(sample_points)}) must have same length!")
    if len(contact_map) != len(sample_points):
        raise ValueError(f"Contact map ({len(contact_map)}) and sample_points ({len(sample_points)}) must have same length!")
    if len(partition_hard) != len(contact_map):
        raise ValueError(f"Partition map ({len(partition_hard)}) and contact map ({len(contact_map)}) must have same length!")
    print(f"  ✓ All maps are aligned: {len(partition_hard)} points")
    
    # CRITICAL: Ensure partition and contact maps are aligned
    # They should have the same length and correspond to the same sample points
    assert len(partition_hard) == len(contact_map) == len(sample_points), \
        f"Alignment error: partition={len(partition_hard)}, contact={len(contact_map)}, points={len(sample_points)}"
    print(f"  ✓ Maps are aligned: {len(partition_hard)} points")
    
    # Map partition and contact from sample_points to mesh vertices
    print(f"\n[5/5] Mapping data from {len(sample_points)} sample points to {n_vertices} mesh vertices...")
    
    if len(sample_points) == n_vertices:
        max_distance = np.max(np.linalg.norm(sample_points - mesh_vertices, axis=1))
        if max_distance < 1e-4:
            print(f"  Sample points match mesh vertices (max diff: {max_distance:.2e}) - using direct assignment")
            vertex_partition = partition_hard.copy()
            vertex_contact = contact_map.copy()
        else:
            print(f"  Same number but positions differ (max diff: {max_distance:.2e}) - using nearest neighbor")
            tree_samples = cKDTree(sample_points)
            distances, indices = tree_samples.query(mesh_vertices, k=1)
            if np.max(distances) > 0.01:
                print(f"  ⚠ Warning: Some points far from mesh (max distance: {np.max(distances):.4f})")
            vertex_partition = partition_hard[indices]
            vertex_contact = contact_map[indices]
    else:
        # Different number of points - use voting for partition, nearest neighbor for contact
        print(f"  Different number of points - using voting for partition, nearest neighbor for contact")
        tree_mesh = cKDTree(mesh_vertices)
        tree_samples = cKDTree(sample_points)
        distances, indices = tree_mesh.query(sample_points, k=1)
        
        # Create vertex partition assignments using voting (same as visualize_partition_map.py)
        vertex_partition = np.zeros(n_vertices, dtype=np.int32)
        vertex_count = np.zeros(n_vertices)
        vertex_votes = np.zeros((n_vertices, 16), dtype=np.int32)
        
        for i, (vertex_idx, part_id) in enumerate(zip(indices, partition_hard)):
            vertex_votes[vertex_idx, part_id] += 1
            vertex_count[vertex_idx] += 1
        
        # Assign each vertex to the part with most votes
        vertex_partition = np.argmax(vertex_votes, axis=1)
        
        # For vertices without any sample, assign to nearest sampled vertex's part
        unassigned = vertex_count == 0
        if unassigned.sum() > 0:
            print(f"  Assigning {unassigned.sum()} unassigned vertices to nearest part...")
            unassigned_verts = mesh_vertices[unassigned]
            _, nearest_sample_indices = tree_samples.query(unassigned_verts, k=1)
            vertex_partition[unassigned] = partition_hard[nearest_sample_indices]
        
        # Map contact map using nearest neighbor
        _, indices = tree_samples.query(mesh_vertices, k=1)
        vertex_contact = contact_map[indices]
    
    # Apply colors and modulate brightness by contact probability
    print(f"\nApplying colors and contact modulation...")
    print(f"  Formula: final_color = partition_color * contact_probability")
    print(f"  - Each hand part keeps its distinct color (hue)")
    print(f"  - Only brightness/intensity changes based on contact probability")
    
    # Get base colors for each hand part (these define the color/hue)
    vertex_colors = PART_COLORS[vertex_partition].copy()  # [N, 3] - each vertex gets its part's color
    
    # Prepare contact-based brightness modulation
    # The contact probability (0-1) will scale the brightness while preserving color ratios
    vertex_contact_clipped = np.clip(vertex_contact, 0.0, 1.0)
    vertex_contact_scaled = vertex_contact_clipped * brightness_scale
    vertex_contact_scaled = np.clip(vertex_contact_scaled, min_brightness, 1.0)
    
    # Multiply each RGB channel by contact probability
    # This preserves the color ratios (hue) while scaling brightness
    # Example: Red [1.0, 0.0, 0.0] * 0.5 = [0.5, 0.0, 0.0] (still red, but dimmer)
    vertex_colors = vertex_colors * vertex_contact_scaled[:, np.newaxis]
    vertex_colors = np.clip(vertex_colors, 0.0, 1.0)
    
    print(f"  Contact range: [{vertex_contact.min():.4f}, {vertex_contact.max():.4f}]")
    print(f"  Contact scaled range: [{vertex_contact_scaled.min():.4f}, {vertex_contact_scaled.max():.4f}]")
    print(f"  Final colors range: [{vertex_colors.min():.3f}, {vertex_colors.max():.3f}]")
    print(f"  ✓ Color hue preserved, brightness modulated by contact probability")
    
    # Print statistics
    print(f"\nPartition statistics:")
    unique_parts, counts = np.unique(vertex_partition, return_counts=True)
    for part_id, count in zip(unique_parts, counts):
        percentage = count / n_vertices * 100
        mask = vertex_partition == part_id
        avg_contact = vertex_contact[mask].mean() if mask.sum() > 0 else 0.0
        print(f"  Part {part_id:2d} ({PART_NAMES[part_id]:12s}): "
              f"{count:5d} vertices ({percentage:5.2f}%), avg contact: {avg_contact:.3f}")
    
    # Create colored mesh
    print(f"\nSaving colored mesh to {output_path}...")
    colored_mesh = trimesh.Trimesh(
        vertices=mesh_vertices,
        faces=obj_mesh.faces,
        vertex_colors=vertex_colors
    )
    
    # Save
    colored_mesh.export(output_path)
    print(f"✓ Saved to {output_path}")
    
    # Print color legend
    print(f"\n" + "="*70)
    print("Color Legend (16 Hand Parts)")
    print("="*70)
    for part_id, (name, color) in enumerate(zip(PART_NAMES, PART_COLORS)):
        print(f"  Part {part_id:2d} ({name:12s}): RGB({color[0]:.1f}, {color[1]:.1f}, {color[2]:.1f})")
    print("="*70)
    print("\nVisualization complete!")
    print("  - Each hand part has its own color")
    print("  - Brightness is modulated by contact probability (0-1)")
    print("  - Formula: final_color = part_color * contact_probability")
    
    return colored_mesh


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize partition map (hand part colors) multiplied by contact map (contact probability)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python visualize_partition_contact_multiply.py \\
    --obj_path grab_data/obj_meshes/toothpaste.ply \\
    --partition_hard exp/my_results/partition_hard.npy \\
    --contact_map exp/my_results/contact_map.npy \\
    --sample_points exp/my_results/sample_points.npy \\
    --output exp/partition_times_contact.obj \\
    --brightness_scale 1.0 \\
    --min_brightness 0.0

Formula: final_color = partition_color * contact_probability
  - partition_color: Color assigned to each hand part (16 colors)
  - contact_probability: Contact probability value (0-1)
  - Result: Hand part colors with brightness modulated by contact probability
        """
    )
    parser.add_argument('--obj_path', type=str, required=True,
                       help='Path to object mesh (.ply or .obj)')
    parser.add_argument('--partition_hard', type=str, required=True,
                       help='Path to partition_hard.npy (hand part assignments 0-15)')
    parser.add_argument('--contact_map', type=str, required=True,
                       help='Path to contact_map.npy (contact probabilities 0-1)')
    parser.add_argument('--sample_points', type=str, required=True,
                       help='Path to sample_points.npy (3D points where maps are defined)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output file path (.obj or .ply)')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Which sample to use if batch dimension exists (default: 0)')
    parser.add_argument('--brightness_scale', type=float, default=1.0,
                       help='Scale factor for contact values (default: 1.0)')
    parser.add_argument('--min_brightness', type=float, default=0.0,
                       help='Minimum brightness to ensure colors are visible (default: 0.0)')
    
    args = parser.parse_args()
    
    visualize_partition_times_contact(
        args.obj_path,
        args.partition_hard,
        args.contact_map,
        args.sample_points,
        args.output,
        args.sample_idx,
        args.brightness_scale,
        args.min_brightness
    )

