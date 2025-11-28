#!/usr/bin/env python3
"""
Use partial_demo.py pipeline but with existing contact/partition maps.
This follows the exact same optimization process as partial_demo.py.
"""
import os
import random
import argparse
import pickle
import numpy as np
import trimesh
import torch
from torch.nn import functional as F
from omegaconf import OmegaConf
from manopth.manolayer import ManoLayer
from contactgen.hand_sdf.hand_model import ArtiHand
from contactgen.contact.contact_optimizer_per_part import optimize_pose

# Patch for chumpy compatibility
import inspect
import collections
if not hasattr(inspect, 'ArgSpec'):
    ArgSpec = collections.namedtuple('ArgSpec', 'args varargs keywords defaults')
    inspect.ArgSpec = ArgSpec

def patched_getargspec(func):
    try:
        sig = inspect.signature(func)
        args = []
        varargs = None
        keywords = None
        defaults = None
        for name, param in sig.parameters.items():
            if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                args.append(name)
                if param.default != inspect.Parameter.empty:
                    if defaults is None:
                        defaults = []
                    defaults.append(param.default)
            elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                varargs = name
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                keywords = name
        return inspect.ArgSpec(args=args, varargs=varargs, keywords=keywords, 
                              defaults=tuple(defaults) if defaults else None)
    except Exception:
        return inspect.ArgSpec(args=[], varargs=None, keywords=None, defaults=None)

inspect.getargspec = patched_getargspec

# Patch numpy for chumpy
import numpy as np
if not hasattr(np, 'int'):
    np.int = np.int64
    np.float = np.float64
    np.complex = np.complex128
    np.bool = np.bool_
    np.object = object
    np.unicode = str
    np.str = str


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ContactGen Demo with Existing Maps')
    parser.add_argument('--obj_path', type=str, required=True, help='object mesh path')
    parser.add_argument('--contact_map', type=str, required=True, help='contact map .npy file')
    parser.add_argument('--partition_hard', type=str, required=True, help='partition hard .npy file')
    parser.add_argument('--sample_points', type=str, required=True, help='sample points .npy file')
    parser.add_argument('--save_root', type=str, required=True, help='result save root')
    parser.add_argument('--n_samples', default=1, type=int, help='number of samples per object')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    
    # contact solver options
    parser.add_argument('--w_contact', default=1e-1, type=float, help='contact weight')
    parser.add_argument('--w_pene', default=3.0, type=float, help='penetration weight')
    parser.add_argument('--w_uv', default=1e-2, type=float, help='uv weight')
    
    args = parser.parse_args()
    os.makedirs(args.save_root, exist_ok=True)
    
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    device = torch.device('cuda')
    
    # Load hand model (same as partial_demo.py)
    config_file = "contactgen/hand_sdf/config.yaml"
    config = OmegaConf.load(config_file)
    hand_model = ArtiHand(config['model_params'], pose_size=config['pose_size'])
    checkpoint = torch.load("contactgen/hand_sdf/hand_model.pt", map_location=device)
    hand_model.load_state_dict(checkpoint['state_dict'], strict=True)
    hand_model.eval()
    hand_model.to(device)

    mano_layer = ManoLayer(mano_root='mano/models', use_pca=True, ncomps=26, side='right', flat_hand_mean=False)
    mano_layer.to(device)
    with open("assets/closed_mano_faces.pkl", 'rb') as f:
        hand_face = pickle.load(f)
    
    # Load object mesh (same as partial_demo.py)
    obj_mesh = trimesh.load(args.obj_path)
    offset = obj_mesh.vertices.mean(axis=0, keepdims=True)
    obj_verts_original = obj_mesh.vertices - offset
    obj_mesh = trimesh.Trimesh(vertices=obj_verts_original, faces=obj_mesh.faces)
    
    # Load existing maps
    print("Loading existing contact and partition maps...")
    contact_map = np.load(args.contact_map)
    if len(contact_map.shape) == 2:
        contact_map = contact_map[0]
    
    partition_hard = np.load(args.partition_hard)
    if len(partition_hard.shape) == 2:
        partition_hard = partition_hard[0]
    
    sample_points = np.load(args.sample_points)
    if len(sample_points.shape) == 3:
        sample_points = sample_points[0]
    
    # Get normals for sample points
    sample = trimesh.sample.sample_surface(obj_mesh, len(sample_points))
    obj_verts = sample_points.astype(np.float32)
    obj_vn = obj_mesh.face_normals[sample[1]].astype(np.float32)
    
    # Convert to torch (same format as partial_demo.py)
    obj_verts = torch.from_numpy(obj_verts).unsqueeze(dim=0).float().to(device).repeat(args.n_samples, 1, 1)
    obj_vn = torch.from_numpy(obj_vn).unsqueeze(dim=0).float().to(device).repeat(args.n_samples, 1, 1)
    
    # Convert maps to torch format (same as partial_demo.py)
    contacts_object = torch.from_numpy(contact_map).unsqueeze(0).float().to(device)
    if contacts_object.dim() == 1:
        contacts_object = contacts_object.unsqueeze(0)  # [N] -> [1, N]
    contacts_object = contacts_object.repeat(args.n_samples, 1)  # [1, N] -> [B, N]
    
    obj_partition = torch.from_numpy(partition_hard).unsqueeze(0).long().to(device)
    obj_partition = obj_partition.repeat(args.n_samples, 1)  # [1, N] -> [B, N]
    
    # Generate UV (simplified - using zeros, optimizer will handle it)
    uv_object = torch.zeros(args.n_samples, len(sample_points), 3).float().to(device)
    
    # Run grasp optimization (EXACTLY same as partial_demo.py)
    print("Optimizing hand pose based on contact map...")
    global_pose, mano_pose, mano_shape, mano_trans = optimize_pose(
        hand_model, mano_layer, obj_verts,
        contacts_object, obj_partition, uv_object,
        w_contact=args.w_contact,
        w_pene=args.w_pene,
        w_uv=args.w_uv
    )

    hand_verts, hand_frames = mano_layer(
        torch.cat((global_pose, mano_pose), dim=1),
        th_betas=mano_shape,
        th_trans=mano_trans
    )
    hand_verts = hand_verts.detach().cpu().numpy()

    # Save results (same as partial_demo.py)
    for i in range(len(hand_verts)):
        obj_mesh.export(os.path.join(args.save_root, args.obj_path.split('/')[-1]))
        hand_mesh = trimesh.Trimesh(vertices=hand_verts[i], faces=hand_face)
        hand_mesh.export(os.path.join(args.save_root, f'grasp_{i}.obj'))

    print("all done")


