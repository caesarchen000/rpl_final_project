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

# Highly saturated, distinct colors for maximum visual separation
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

def visualize_partition_times_contact(
    obj_path,
    partition_hard_path,
    contact_map_path,
    part_logits_path,
    sample_points_path,
    output_path,
    sample_idx=0,
    brightness_scale=1.0,
    min_brightness=0.0,
):
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
    print("\nFile paths:")
    print(f"  Object mesh:        {obj_path}")
    print(f"  Partition hard:     {partition_hard_path}")
    print(f"  Contact map:        {contact_map_path}")
    print(f"  Sample points:      {sample_points_path}")
    print(f"  Part logits:        {part_logits_path}")
    print(f"  Output:             {output_path}")
    print("="*70)
    
    # Load object mesh
    print(f"\n[1/6] Loading object mesh: {obj_path}")
    obj_mesh = trimesh.load(obj_path, process=False)
    mesh_vertices = np.array(obj_mesh.vertices)
    n_vertices = len(mesh_vertices)
    print(f"  Object: {n_vertices} vertices, {len(obj_mesh.faces)} faces")
    
    # Load partition map (hand part assignments)
    print(f"\n[2/6] Loading partition map: {partition_hard_path}")
    partition_hard = np.load(partition_hard_path)
    original_partition_shape = partition_hard.shape
    if len(partition_hard.shape) == 2:
        partition_hard = partition_hard[sample_idx]
    elif len(partition_hard.shape) == 3:
        partition_hard = partition_hard[sample_idx].squeeze()
    print(f"  Partition shape: {original_partition_shape} -> {partition_hard.shape}")
    print(f"  Part range: [{partition_hard.min()}, {partition_hard.max()}] (should be 0-15)")
    
    # Load contact map (contact probabilities)
    print(f"\n[3/6] Loading contact map: {contact_map_path}")
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
    print(f"\n[4/6] Loading sample points: {sample_points_path}")
    sample_points = np.load(sample_points_path)
    print(f"  Sample points shape: {sample_points.shape}")
    
    # Detect if wrong file was loaded (e.g., part_logits.npy instead of sample_points.npy)
    if len(sample_points.shape) == 2 and sample_points.shape[1] == 16:
        raise ValueError(
            f"ERROR: Wrong file loaded! Expected sample_points.npy with shape [N, 3], "
            f"but got shape {sample_points.shape} which looks like part_logits.npy. "
            f"Please check that --sample_points points to the correct file."
        )
    
    # Handle different shapes: [N, 3], [B, N, 3]
    if len(sample_points.shape) == 3:
        # [B, N, 3] -> select sample
        if sample_points.shape[2] != 3:
            raise ValueError(f"Sample points must have shape [B, N, 3], got {sample_points.shape}")
        sample_points = sample_points[sample_idx]
    elif len(sample_points.shape) == 2:
        # [N, 3] -> use as is
        if sample_points.shape[1] != 3:
            raise ValueError(
                f"Sample points must have shape [N, 3] or [B, N, 3], got {sample_points.shape}. "
                f"Did you accidentally load the wrong file (e.g., part_logits.npy)?"
            )
    else:
        raise ValueError(f"Unexpected sample points shape: {sample_points.shape}")
    
    print(f"  Sample points shape after processing: {sample_points.shape}")
    
    # CRITICAL: Verify alignment - partition map and contact map must be aligned
    # They should have the same length and correspond to the same sample points
    if len(partition_hard) != len(sample_points):
        raise ValueError(f"Partition map ({len(partition_hard)}) and sample_points ({len(sample_points)}) must have same length!")
    if len(contact_map) != len(sample_points):
        raise ValueError(f"Contact map ({len(contact_map)}) and sample_points ({len(sample_points)}) must have same length!")
    if len(partition_hard) != len(contact_map):
        raise ValueError(f"Partition map ({len(partition_hard)}) and contact_map ({len(contact_map)}) must have same length!")
    print(f"  ✓ All maps are aligned: {len(partition_hard)} points")

    # ------------------------------------------------------------------
    # NEW: build per-point [N, 16] partition×contact matrix from logits
    # Each row i has exactly one non-zero entry:
    #   - find argmax over 16 logits at point i
    #   - set that index to brightness value (contact probability scaled by brightness_scale)
    # Result saved as <output_dir>/final.npy with shape [N, 16]
    # ------------------------------------------------------------------
    point_pc = None
    if part_logits_path is not None and part_logits_path != "":
        print(f"\n[5/6] Loading partition logits for per-part contact matrix: {part_logits_path}")
        if not os.path.exists(part_logits_path):
            print(f"  ⚠ Warning: part_logits file not found, skipping final.npy: {part_logits_path}")
        else:
            part_logits = np.load(part_logits_path)
            logits_shape = part_logits.shape
            # Expected shapes: [B, N, 16] or [N, 16]
            if len(part_logits.shape) == 3:
                part_logits = part_logits[sample_idx]  # [N, 16]
            elif len(part_logits.shape) == 2:
                # [N, 16] -> use as is
                pass
            else:
                print(f"  ⚠ Warning: unexpected logits shape {logits_shape}, skipping final.npy")
                part_logits = None

            if part_logits is not None:
                if part_logits.shape[0] != len(sample_points) or part_logits.shape[1] != 16:
                    print(f"  ⚠ Warning: logits shape {part_logits.shape} incompatible with N={len(sample_points)}; skipping final.npy")
                else:
                    # For each point: find the hand part with highest logit, assign brightness to that part, others = 0
                    # Each point has 16 logits (one per hand part), we choose argmax to get the winning part
                    best_part = np.argmax(part_logits, axis=1).astype(np.int64)  # [N] - part ID with highest logit per point
                    n_points = part_logits.shape[0]
                    point_pc = np.zeros((n_points, 16), dtype=np.float32)  # Initialize all to 0
                    # Calculate brightness from contact map (heatmap value scaled by brightness_scale, clipped to [min_brightness, 1.0])
                    contact_clipped = np.clip(contact_map, 0.0, 1.0)
                    brightness = contact_clipped * brightness_scale
                    brightness = np.clip(brightness, min_brightness, 1.0)
                    # Assign brightness to the part with highest logit, all other 15 parts remain 0
                    point_indices = np.arange(n_points, dtype=np.int64)
                    point_pc[point_indices, best_part] = brightness.astype(np.float32)
                    out_dir = os.path.dirname(output_path)
                    pc_path = os.path.join(out_dir, "final.npy")
                    np.save(pc_path, point_pc)
                    print(f"  ✓ Saved final matrix to {pc_path} with shape {point_pc.shape}")

    # Map partition and contact from sample_points to mesh vertices
    print(f"\n[6/6] Mapping data from {len(sample_points)} sample points to {n_vertices} mesh vertices...")
    
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
        # Validate sample_points shape before use
        if len(sample_points.shape) != 2 or sample_points.shape[1] != 3:
            raise ValueError(f"sample_points must have shape [N, 3], got {sample_points.shape}")
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

    # ---------------------------------------------------------------------
    # Also save a NumPy file for the combined partition × contact values.
    #
    # We save per-sample-point colors (before mapping onto the mesh),
    # aligned with:
    #   - partition_hard (after sample selection)
    #   - contact_map   (after sample selection)
    #   - sample_points
    #
    # Shape: [N, 3], values in [0, 1]
    # Path:  <same directory as output OBJ>/partition_contact.npy
    # ---------------------------------------------------------------------
    try:
        # Per-point base colors and contact (aligned with sample_points)
        point_base_colors = PART_COLORS[partition_hard].copy()          # [N, 3]
        point_contact_clipped = np.clip(contact_map, 0.0, 1.0)          # [N]
        point_contact_scaled = point_contact_clipped * brightness_scale
        point_contact_scaled = np.clip(point_contact_scaled, min_brightness, 1.0)
        partition_contact_points = point_base_colors * point_contact_scaled[:, np.newaxis]
        partition_contact_points = np.clip(partition_contact_points, 0.0, 1.0)

        output_dir = os.path.dirname(output_path)
        partition_contact_npy_path = os.path.join(output_dir, "partition_contact.npy")
        np.save(partition_contact_npy_path, partition_contact_points)
        print(f"\nSaved partition × contact NumPy array to:")
        print(f"  {partition_contact_npy_path}")
        print(f"  Shape: {partition_contact_points.shape}  (N, 3)")
    except Exception as e:
        print("\n⚠ Warning: Failed to save partition_contact.npy:")
        print(f"  {e}")
    
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
    parser.add_argument('--part_logits', type=str, default=None,
                       help='Path to part_logits.npy (for saving final.npy)')
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
        args.part_logits,
        args.sample_points,
        args.output,
        args.sample_idx,
        args.brightness_scale,
        args.min_brightness
    )

