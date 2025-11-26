import argparse
import os
import sys
import trimesh

# Check if we're in a headless environment (no display)
import os
HEADLESS = not os.environ.get('DISPLAY') or os.environ.get('DISPLAY') == ''

# Try to import open3d, but skip if headless to avoid segfaults
OPEN3D_AVAILABLE = False
if not HEADLESS:
    try:
        import open3d
        OPEN3D_AVAILABLE = True
    except ImportError:
        OPEN3D_AVAILABLE = False

colors = {'light_red': [0.85882353, 0.74117647, 0.65098039],
          'light_blue': [145/255, 191/255, 219/255]}


def ho_plot_headless(hand_mesh, obj_mesh, save_path):
    """Fallback headless visualization using matplotlib"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Sample points for faster rendering
    if len(hand_mesh.vertices) > 5000:
        hand_sample = hand_mesh.vertices[::len(hand_mesh.vertices)//5000]
    else:
        hand_sample = hand_mesh.vertices
    
    if len(obj_mesh.vertices) > 5000:
        obj_sample = obj_mesh.vertices[::len(obj_mesh.vertices)//5000]
    else:
        obj_sample = obj_mesh.vertices
    
    # Plot hand in red
    ax.scatter(hand_sample[:, 0], hand_sample[:, 1], hand_sample[:, 2], 
               c='red', alpha=0.6, s=2, label='Hand')
    
    # Plot object in blue
    ax.scatter(obj_sample[:, 0], obj_sample[:, 1], obj_sample[:, 2], 
               c='blue', alpha=0.4, s=2, label='Object')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Hand-Object Grasp Visualization')
    ax.legend()
    
    # Set equal aspect ratio
    all_verts = np.concatenate([hand_mesh.vertices, obj_mesh.vertices])
    max_range = np.array([all_verts[:, 0].max() - all_verts[:, 0].min(),
                          all_verts[:, 1].max() - all_verts[:, 1].min(),
                          all_verts[:, 2].max() - all_verts[:, 2].min()]).max() / 2.0
    mid_x = (all_verts[:, 0].max() + all_verts[:, 0].min()) * 0.5
    mid_y = (all_verts[:, 1].max() + all_verts[:, 1].min()) * 0.5
    mid_z = (all_verts[:, 2].max() + all_verts[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.view_init(elev=20, azim=45)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def ho_plot_open3d(hand_mesh, obj_mesh, save_path, viewpoint_path="assets/viewpoint.json"):
    """Open3D visualization (requires display)"""
    vis = open3d.visualization.Visualizer()
    
    # Try to create window, fall back to headless if it fails
    try:
        vis.create_window(visible=False)
    except Exception as e:
        print(f"Warning: Open3D GUI not available ({e}), falling back to headless visualization")
        ho_plot_headless(hand_mesh, obj_mesh, save_path)
        return
    
    vis_hand = open3d.geometry.TriangleMesh()
    vis_hand.vertices = open3d.utility.Vector3dVector(hand_mesh.vertices)
    vis_hand.triangles = open3d.utility.Vector3iVector(hand_mesh.faces)
    vis_hand.paint_uniform_color(colors['light_red'])
    vis_hand.compute_vertex_normals()
    vis.add_geometry(vis_hand)
    vis_obj = open3d.geometry.TriangleMesh()
    vis_obj.vertices = open3d.utility.Vector3dVector(obj_mesh.vertices)
    vis_obj.triangles = open3d.utility.Vector3iVector(obj_mesh.faces)
    vis_obj.paint_uniform_color(colors['light_blue'])        
    vis_obj.compute_vertex_normals()
    vis.add_geometry(vis_obj)
    
    # Try to load and apply viewpoint if available
    try:
        if os.path.exists(viewpoint_path):
            ctr = vis.get_view_control()
            param = open3d.io.read_pinhole_camera_parameters(viewpoint_path)
            ctr.convert_from_pinhole_camera_parameters(param)
    except:
        # If viewpoint file doesn't exist or fails, use default view
        pass
    
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(save_path)
    vis.destroy_window()


def ho_plot(hand_mesh, obj_mesh, save_path, viewpoint_path="assets/viewpoint.json"):
    """Main visualization function - tries Open3D first, falls back to matplotlib"""
    # Always use headless if no display, or if Open3D not available
    if HEADLESS or not OPEN3D_AVAILABLE:
        ho_plot_headless(hand_mesh, obj_mesh, save_path)
        print(f"✓ Saved visualization to {save_path} (headless mode)")
        return
    
    # Try Open3D if we have a display
    try:
        ho_plot_open3d(hand_mesh, obj_mesh, save_path, viewpoint_path)
        print(f"✓ Saved visualization to {save_path}")
    except Exception as e:
        print(f"Warning: Open3D visualization failed ({e}), using headless fallback")
        ho_plot_headless(hand_mesh, obj_mesh, save_path)
        print(f"✓ Saved visualization to {save_path} (headless mode)")

if __name__ == '__main__':
    parse = argparse.ArgumentParser("Visualize grasp")
    parse.add_argument("--hand_path", type=str, default="exp/demo_results/grasp_0.obj", help="hand mesh path")
    parse.add_argument("--obj_path", type=str, default="assets/toothpaste.ply", help="object mesh path")
    parse.add_argument("--save_path", type=str, default="exp/demo_results/grasp_0.png", help="save path")
    args = parse.parse_args()
    
    if not os.path.exists(args.hand_path):
        print(f"Error: Hand mesh not found: {args.hand_path}")
        sys.exit(1)
    if not os.path.exists(args.obj_path):
        print(f"Error: Object mesh not found: {args.obj_path}")
        sys.exit(1)
    
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    hand_mesh = trimesh.load(args.hand_path)
    obj_mesh = trimesh.load(args.obj_path)
    ho_plot(hand_mesh, obj_mesh, args.save_path)
