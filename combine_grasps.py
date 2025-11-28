#!/usr/bin/env python3
"""
Combine hand and object meshes into a single mesh file
Usage: python combine_grasp.py --hand_path <hand.obj> --obj_path <object.ply> --output <grasp.obj>
"""
import argparse
import os
import trimesh
import numpy as np


def combine_meshes(hand_path, obj_path, output_path, hand_color=None, obj_color=None, auto_center=False):
    """
    Combine hand and object meshes into a single mesh file
    
    Args:
        hand_path: Path to hand mesh file
        obj_path: Path to object mesh file  
        output_path: Output path for combined mesh
        hand_color: Optional RGB color for hand (0-255 or 0-1)
        obj_color: Optional RGB color for object (0-255 or 0-1)
        auto_center: If True, center the object mesh to match inference coordinate space
    """
    # Load meshes
    print(f"Loading hand mesh from: {hand_path}")
    hand_mesh = trimesh.load(hand_path)
    print(f"  Hand: {len(hand_mesh.vertices)} vertices, {len(hand_mesh.faces)} faces")
    
    print(f"Loading object mesh from: {obj_path}")
    obj_mesh = trimesh.load(obj_path)
    print(f"  Object: {len(obj_mesh.vertices)} vertices, {len(obj_mesh.faces)} faces")
    
    # Auto-center object if requested (to match inference coordinate space)
    if auto_center:
        print("  Centering object to match inference coordinate space...")
        offset = obj_mesh.vertices.mean(axis=0, keepdims=True)
        obj_mesh.vertices = obj_mesh.vertices - offset
    
    # Apply colors if provided
    if hand_color is not None:
        hand_color = np.array(hand_color)
        if hand_color.max() > 1.0:
            hand_color = hand_color / 255.0
        hand_mesh.visual.vertex_colors = hand_color
    
    if obj_color is not None:
        obj_color = np.array(obj_color)
        if obj_color.max() > 1.0:
            obj_color = obj_color / 255.0
        obj_mesh.visual.vertex_colors = obj_color
    
    # Combine meshes while preserving colors
    print("Combining meshes...")
    hand_vertices = hand_mesh.vertices
    obj_vertices = obj_mesh.vertices
    combined_vertices = np.vstack([hand_vertices, obj_vertices])

    hand_faces = hand_mesh.faces
    obj_faces = obj_mesh.faces + len(hand_vertices)
    combined_faces = np.vstack([hand_faces, obj_faces])

    combined = trimesh.Trimesh(vertices=combined_vertices, faces=combined_faces, process=False)

    # Preserve vertex colors
    def _get_vertex_colors(mesh, default_color=None):
        if hasattr(mesh.visual, 'vertex_colors') and len(mesh.visual.vertex_colors) == len(mesh.vertices):
            return mesh.visual.vertex_colors.copy()
        if default_color is None:
            default_color = np.array([200, 200, 200, 255], dtype=np.uint8)
        if default_color.max() <= 1.0:
            default_color = (default_color * 255).astype(np.uint8)
        return np.tile(default_color, (len(mesh.vertices), 1))

    hand_colors = _get_vertex_colors(hand_mesh, hand_color)
    obj_colors = _get_vertex_colors(obj_mesh, obj_color)
    combined.visual.vertex_colors = np.vstack([hand_colors, obj_colors])

    # Preserve face colors if available
    def _get_face_colors(mesh, vertex_colors):
        if hasattr(mesh.visual, 'face_colors') and len(mesh.visual.face_colors) == len(mesh.faces):
            return mesh.visual.face_colors.copy()
        # Fallback: average vertex colors for each face
        return vertex_colors[mesh.faces].mean(axis=1).astype(np.uint8)

    try:
        hand_face_colors = _get_face_colors(hand_mesh, hand_colors)
        obj_face_colors = _get_face_colors(obj_mesh, obj_colors)
        combined.visual.face_colors = np.vstack([hand_face_colors, obj_face_colors])
    except Exception:
        pass

    # Save as mesh
    output_ext = os.path.splitext(output_path)[1].lower()
    print(f"Saving combined mesh to: {output_path}")
    
    if output_ext == '.obj':
        combined.export(output_path)
    elif output_ext == '.ply':
        combined.export(output_path)
    else:
        # Default to .obj if extension not recognized
        if not output_path.endswith('.obj'):
            output_path += '.obj'
        combined.export(output_path)
    
    print(f"✓ Successfully created combined mesh:")
    print(f"  Total vertices: {len(combined.vertices)}")
    print(f"  Total faces: {len(combined.faces)}")
    print(f"  Output: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine hand and object meshes into a single file')
    parser.add_argument('--hand_path', type=str, required=True, help='Path to hand mesh file (.obj or .ply)')
    parser.add_argument('--obj_path', type=str, required=True, help='Path to object mesh file (.obj or .ply)')
    parser.add_argument('--output', type=str, default=None, help='Output path for combined mesh (default: auto-generated)')
    parser.add_argument('--format', type=str, choices=['obj', 'ply'], default='obj', help='Output format (default: obj)')
    parser.add_argument('--auto_center', action='store_true', help='Auto-center object mesh to match inference coordinate space')
    
    args = parser.parse_args()
    
    # Check input files exist
    if not os.path.exists(args.hand_path):
        print(f"Error: Hand mesh not found: {args.hand_path}")
        exit(1)
    if not os.path.exists(args.obj_path):
        print(f"Error: Object mesh not found: {args.obj_path}")
        exit(1)
    
    # Generate output path if not provided
    if args.output is None:
        hand_base = os.path.splitext(os.path.basename(args.hand_path))[0]
        obj_base = os.path.splitext(os.path.basename(args.obj_path))[0]
        # Save to meshes directory by default
        output_dir = "meshes"
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, f"{obj_base}_grasped_by_{hand_base}.{args.format}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    # Auto-detect object mesh from same directory (centered version from inference)
    user_provided_obj = args.obj_path and os.path.exists(args.obj_path)
    hand_dir = os.path.dirname(args.hand_path)
    if hand_dir and not user_provided_obj:
        # Look for object mesh files in the same directory (excluding grasp files)
        import glob
        for ext in ['.ply', '.obj']:
            obj_candidates = glob.glob(os.path.join(hand_dir, f'*{ext}'))
            obj_candidates = [f for f in obj_candidates 
                            if 'grasp' not in os.path.basename(f).lower() 
                            and os.path.basename(f) != os.path.basename(args.hand_path)]
            if obj_candidates:
                # Found object in same directory - use it (already centered from inference)
                args.obj_path = obj_candidates[0]
                args.auto_center = False
                print(f"Auto-detected object mesh from inference: {args.obj_path}")
                user_provided_obj = True
                break
    
    # If using original object path, suggest auto-centering
    if args.obj_path and not args.auto_center and 'grab_data' not in args.obj_path and hand_dir and hand_dir not in args.obj_path:
        print("Note: Using original object mesh. If hand position is wrong, try --auto_center")
    
    # Combine meshes
    combine_meshes(args.hand_path, args.obj_path, args.output, auto_center=args.auto_center)
