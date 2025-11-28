"""
Visualize partition map (hand part assignment) on object mesh.

Colors each vertex based on which hand part it's assigned to.
Optionally modulates brightness by contact probability.
"""
import os
import argparse
import numpy as np
import trimesh
from scipy.spatial import cKDTree


# Hand part names and colors (distinct colors for each part)
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
    [0.4, 0.4, 0.4],      # 0: palm - dark grey
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


def visualize_partition_map(obj_mesh_path, partition_hard_path, sample_points_path,
                            output_path, sample_idx=0, contact_map_path=None, 
                            brightness_scale=1.0):
    """
    Visualize partition map on object mesh.
    
    Args:
        obj_mesh_path: Path to original object mesh
        partition_hard_path: Path to partition_hard.npy [B, N] or [N]
        sample_points_path: Path to sample_points.npy [N, 3]
        output_path: Output OBJ file path
        sample_idx: Which sample to visualize (if partition is [B, N])
        contact_map_path: Optional path to contact_map.npy for brightness modulation
        brightness_scale: Scale factor for brightness modulation (default: 1.0)
    """
    
    # Load object mesh
    print(f"Loading object mesh from {obj_mesh_path}")
    obj_mesh = trimesh.load(obj_mesh_path)
    
    # Load partition map
    print(f"Loading partition map from {partition_hard_path}")
    partition_hard = np.load(partition_hard_path)
    print(f"Partition map shape: {partition_hard.shape}")
    
    # Handle different shapes
    if len(partition_hard.shape) == 2:
        partition_hard = partition_hard[sample_idx]  # [N]
    elif len(partition_hard.shape) == 1:
        pass  # Already [N]
    else:
        raise ValueError(f"Unexpected partition map shape: {partition_hard.shape}")
    
    print(f"Using partition map shape: {partition_hard.shape}")
    print(f"Part values range: [{partition_hard.min()}, {partition_hard.max()}]")
    
    # Load sample points
    if sample_points_path and os.path.exists(sample_points_path):
        print(f"Loading sample points from {sample_points_path}")
        sample_points = np.load(sample_points_path)
        print(f"Sample points shape: {sample_points.shape}")
    else:
        print(f"Sample points file not found. Generating from mesh...")
        sample = trimesh.sample.sample_surface(obj_mesh, 2048)
        sample_points = sample[0].astype(np.float32)
        print(f"Generated sample points shape: {sample_points.shape}")
    
    # Map partition assignments to mesh vertices
    mesh_vertices = np.array(obj_mesh.vertices)
    
    # Check if sample points match mesh vertices (dense sampling case)
    if len(sample_points) == len(mesh_vertices):
        max_distance = np.max(np.linalg.norm(sample_points - mesh_vertices, axis=1))
        if max_distance < 1e-4:
            print(f"Sample points match mesh vertices (max diff: {max_distance:.2e}) - using direct assignment")
            vertex_partition = partition_hard.copy()
        else:
            print(f"Same number of points but positions differ (max diff: {max_distance:.2e}) - using nearest neighbor")
            tree_samples = cKDTree(sample_points)
            distances, indices = tree_samples.query(mesh_vertices, k=1)
            if np.max(distances) > 0.01:
                print(f"Warning: Some sample points are far from mesh vertices (max distance: {np.max(distances):.4f})")
            vertex_partition = partition_hard[indices]
    else:
        # Different number of points - use nearest neighbor mapping
        print(f"Mapping {len(sample_points)} sample points to {len(mesh_vertices)} mesh vertices...")
        tree_mesh = cKDTree(mesh_vertices)
        tree_samples = cKDTree(sample_points)
        distances, indices = tree_mesh.query(sample_points, k=1)
        
        # Create vertex partition assignments (use mode/voting for vertices with multiple samples)
        vertex_partition = np.zeros(len(mesh_vertices), dtype=np.int32)
        vertex_count = np.zeros(len(mesh_vertices))
        vertex_votes = np.zeros((len(mesh_vertices), 16), dtype=np.int32)
        
        for i, (vertex_idx, part_id) in enumerate(zip(indices, partition_hard)):
            vertex_votes[vertex_idx, part_id] += 1
            vertex_count[vertex_idx] += 1
        
        # Assign each vertex to the part with most votes
        vertex_partition = np.argmax(vertex_votes, axis=1)
        
        # For vertices without any sample, assign to nearest sampled vertex's part
        unassigned = vertex_count == 0
        if unassigned.sum() > 0:
            print(f"Assigning {unassigned.sum()} unassigned vertices to nearest part...")
            unassigned_verts = mesh_vertices[unassigned]
            _, nearest_sample_indices = tree_samples.query(unassigned_verts, k=1)
            vertex_partition[unassigned] = partition_hard[nearest_sample_indices]
    
    # Create vertex colors based on partition assignment
    vertex_colors = PART_COLORS[vertex_partition].copy()
    
    # If contact map is provided, use it to modulate brightness
    if contact_map_path and os.path.exists(contact_map_path):
        print(f"Loading contact map from {contact_map_path} for brightness modulation")
        contact_map = np.load(contact_map_path)
        
        # Handle different contact map shapes
        original_shape = contact_map.shape
        if len(contact_map.shape) == 3:
            contact_map = contact_map[sample_idx]
        elif len(contact_map.shape) == 2:
            if contact_map.shape[0] <= 10:  # Likely batch dimension
                contact_map = contact_map[sample_idx]
        
        # Squeeze out singleton dimensions
        if len(contact_map.shape) == 2 and contact_map.shape[1] == 1:
            contact_map = contact_map.squeeze()
        elif len(contact_map.shape) == 1:
            pass
        
        print(f"Contact map shape: {original_shape} -> {contact_map.shape}")
        
        # Check if contact map has same number of points as sample_points
        if len(contact_map) != len(sample_points):
            print(f"Warning: Contact map has {len(contact_map)} points but sample_points has {len(sample_points)}")
            print("  Will interpolate contact values to match sample_points")
            contact_on_samples = contact_map[:len(sample_points)] if len(contact_map) > len(sample_points) else contact_map
        else:
            contact_on_samples = contact_map
        
        # Map from sample_points to mesh vertices
        if len(sample_points) == len(mesh_vertices):
            max_distance = np.max(np.linalg.norm(sample_points - mesh_vertices, axis=1))
            if max_distance < 1e-4:
                vertex_contact = contact_on_samples.copy()
            else:
                tree_samples = cKDTree(sample_points)
                _, indices = tree_samples.query(mesh_vertices, k=1)
                vertex_contact = contact_on_samples[indices]
        else:
            tree_samples = cKDTree(sample_points)
            _, indices = tree_samples.query(mesh_vertices, k=1)
            vertex_contact = contact_on_samples[indices]
        
        # Multiply partition colors by contact probability directly
        # Formula: final_color = partition_color * contact_probability
        vertex_contact_clipped = np.clip(vertex_contact, 0.0, 1.0)
        vertex_contact_scaled = vertex_contact_clipped * brightness_scale
        vertex_contact_scaled = np.clip(vertex_contact_scaled, 0.0, 1.0)
        
        # Multiply partition colors by contact probability
        vertex_colors = vertex_colors * vertex_contact_scaled[:, np.newaxis]
        
        print(f"Contact-based brightness modulation:")
        print(f"  Contact range: [{vertex_contact.min():.4f}, {vertex_contact.max():.4f}]")
        print(f"  Contact scaled range: [{vertex_contact_scaled.min():.4f}, {vertex_contact_scaled.max():.4f}]")
        print(f"  Formula: final_color = partition_color * contact_probability")
    else:
        print("No contact map provided - using full brightness partition colors")
    
    # Print statistics
    print(f"\nPartition statistics:")
    unique_parts, counts = np.unique(vertex_partition, return_counts=True)
    for part_id, count in zip(unique_parts, counts):
        percentage = count / len(vertex_partition) * 100
        print(f"  Part {part_id:2d} ({PART_NAMES[part_id]:12s}): {count:5d} vertices ({percentage:5.2f}%)")
    
    # Write OBJ file with vertex colors
    print(f"\nWriting OBJ file to {output_path}")
    
    with open(output_path, 'w') as f:
        f.write("# Partition map visualization\n")
        f.write("# Each vertex is colored based on which hand part it's assigned to\n")
        if contact_map_path:
            f.write("# Brightness modulated by contact probability\n")
        f.write("# Format: v x y z r g b\n")
        
        # Write vertices with colors
        for v, c in zip(mesh_vertices, vertex_colors):
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f} {c[0]:.8f} {c[1]:.8f} {c[2]:.8f}\n")
        
        # Write faces
        mesh_faces = np.array(obj_mesh.faces)
        for face in mesh_faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    print(f"✓ Created: {output_path}")
    print("\nColor legend:")
    for part_id, (name, color) in enumerate(zip(PART_NAMES, PART_COLORS)):
        print(f"  Part {part_id:2d} ({name:12s}): RGB({color[0]:.1f}, {color[1]:.1f}, {color[2]:.1f})")
    
    return vertex_colors, vertex_partition


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize partition map on object mesh')
    parser.add_argument('--obj_path', type=str, required=True,
                       help='Path to original object mesh')
    parser.add_argument('--partition_hard', type=str, required=True,
                       help='Path to partition_hard.npy file')
    parser.add_argument('--sample_points', type=str, required=True,
                       help='Path to sample_points.npy file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output OBJ file path')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Which sample to visualize (if partition is [B, N])')
    parser.add_argument('--contact_map', type=str, default=None,
                       help='Optional: Path to contact_map.npy to modulate brightness')
    parser.add_argument('--brightness_scale', type=float, default=1.0,
                       help='Scale factor for brightness modulation (default: 1.0)')
    
    args = parser.parse_args()
    
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.obj_path))[0]
        args.output = f'{base_name}_partition_map.obj'
    
    visualize_partition_map(
        args.obj_path,
        args.partition_hard,
        args.sample_points,
        args.output,
        args.sample_idx,
        args.contact_map,
        args.brightness_scale
    )
    
    print("\nOpen the OBJ file in your 3D viewer to see the partition map!")
    if args.contact_map:
        print("Brightness shows contact probability: brighter = higher contact")
