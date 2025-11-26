import os
import random
import argparse
import pickle
import numpy as np
import trimesh
import torch
from torch.nn import functional as F
from tqdm import tqdm
from omegaconf import OmegaConf
from manopth.manolayer import ManoLayer
from contactgen.utils.cfg_parser import Config
from contactgen.model_per_part import ContactGenModelPerPart
from contactgen.hand_sdf.hand_model import ArtiHand
from contactgen.contact.contact_optimizer_per_part import optimize_pose
from contactgen.datasets.eval_dataset import TestSet


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ContactGen Eval - Per-Part Contact Maps')
    parser.add_argument('--checkpoint', default='checkpoint/checkpoint.pt', type=str, help='exp dir')
    parser.add_argument('--n_samples', default=10, type=int, help='number of samples per object')
    parser.add_argument('--save_root', default='exp/results', type=str, help='result save root')
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
    
    cfg_path = "contactgen/configs/default.yaml"
    model_path = "checkpoint/checkpoint.pt"

    cfg = Config(default_cfg_path=cfg_path)
    device = torch.device('cuda')
    model = ContactGenModelPerPart(cfg).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    
    dataset = TestSet()
    config_file = "contactgen/hand_sdf/config.yaml"
    config = OmegaConf.load(config_file)
    hand_model = ArtiHand(config['model_params'], pose_size=config['pose_size'])
    checkpoint = torch.load("contactgen/hand_sdf/hand_model.pt")
    hand_model.load_state_dict(checkpoint['state_dict'], strict=True)
    hand_model.eval()
    hand_model.to(device)

    mano_layer = ManoLayer(mano_root='mano/models', use_pca=True, ncomps=26, side='right', flat_hand_mean=False)
    mano_layer.to(device)

    with open("assets/closed_mano_faces.pkl", 'rb') as f:
        hand_face = pickle.load(f)

    for idx, input in tqdm(enumerate(dataset)):
        obj_name = input['obj_name']
        obj_verts = torch.from_numpy(input['obj_verts']).unsqueeze(dim=0).float().to(device).repeat(args.n_samples, 1, 1)
        obj_vn = torch.from_numpy(input['obj_vn']).unsqueeze(dim=0).float().to(device).repeat(args.n_samples, 1, 1)
        with torch.no_grad():
            sample_results = model.sample(obj_verts, obj_vn)
        contacts_object, partition_object, uv_object = sample_results
        # contacts_object is now [B, N, 16] - per-part contact maps
        partition_object = partition_object.argmax(dim=-1)  # [B, N] - part IDs
            
        global_pose, mano_pose, mano_shape, mano_trans = optimize_pose(hand_model, mano_layer, obj_verts, contacts_object, partition_object, uv_object,
                                                                       w_contact=args.w_contact, w_pene=args.w_pene, w_uv=args.w_uv) 
        hand_verts, hand_frames = mano_layer(torch.cat((global_pose, mano_pose), dim=1), th_betas=mano_shape, th_trans=mano_trans)
        hand_verts = hand_verts.detach()
        hand_verts = hand_verts.cpu().numpy()
        
        save_dir = os.path.join(args.save_root, obj_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save contact maps
        contacts_object_np = contacts_object.detach().cpu().numpy()  # (n_samples, 2048, 16) - per-part contact maps
        np.save(os.path.join(save_dir, 'contact_maps.npy'), contacts_object_np)
        print(f"Saved contact maps for {obj_name}: {contacts_object_np.shape} (n_samples, n_points, n_parts=16)")
        
        for i in range(len(hand_verts)):
            hand_mesh = trimesh.Trimesh(vertices=hand_verts[i], faces=hand_face)
            hand_mesh.export(os.path.join(save_dir, 'grasp_{}.obj'.format(i)))
 
    print("all done")

