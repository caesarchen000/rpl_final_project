"""
Simple script to test if the per-part model is producing reasonable outputs
without needing the full demo pipeline.
"""
import os
import torch
import numpy as np
from contactgen.utils.cfg_parser import Config
from contactgen.model_per_part import ContactGenModelPerPart

def test_model_output(checkpoint_path='exp_per_part/checkpoint.pt'):
    """Test the model with dummy data to see if outputs are reasonable."""
    
    print("=" * 70)
    print("TESTING PER-PART MODEL OUTPUT")
    print("=" * 70)
    
    # Load config
    cfg_path = "contactgen/configs/default.yaml"
    cfg = Config(default_cfg_path=cfg_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {checkpoint_path}")
    model = ContactGenModelPerPart(cfg).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint, strict=True)
        print("Loaded checkpoint (no epoch info)")
    
    model.eval()
    
    # Create dummy object data (batch_size=2, n_points=2048)
    batch_size = 2
    n_points = 2048
    print(f"\nTesting with dummy data: batch_size={batch_size}, n_points={n_points}")
    
    obj_verts = torch.randn(batch_size, n_points, 3).to(device) * 0.1  # Small object
    obj_vn = torch.randn(batch_size, n_points, 3).to(device)
    obj_vn = obj_vn / (obj_vn.norm(dim=-1, keepdim=True) + 1e-8)  # Normalize normals
    
    print("\nRunning model inference...")
    with torch.no_grad():
        sample_results = model.sample(obj_verts, obj_vn)
    
    contacts_object, partition_object, uv_object = sample_results
    
    # Analyze outputs
    print("\n" + "=" * 70)
    print("OUTPUT ANALYSIS")
    print("=" * 70)
    
    print(f"\nContact maps shape: {contacts_object.shape}")
    print(f"  Expected: [batch_size={batch_size}, n_points={n_points}, n_parts=16]")
    print(f"  Actual: {list(contacts_object.shape)}")
    
    # Check contact map values (should be in [0, 1] after sigmoid)
    contact_min = contacts_object.min().item()
    contact_max = contacts_object.max().item()
    contact_mean = contacts_object.mean().item()
    print(f"\nContact map statistics:")
    print(f"  Min: {contact_min:.4f} (should be >= 0)")
    print(f"  Max: {contact_max:.4f} (should be <= 1)")
    print(f"  Mean: {contact_mean:.4f}")
    
    # Check how many points have contact per part
    contact_threshold = 0.5
    active_contacts = (contacts_object > contact_threshold).sum(dim=1)  # [B, 16]
    print(f"\nPoints with contact > {contact_threshold} per part:")
    for b in range(batch_size):
        print(f"  Sample {b}: {active_contacts[b].cpu().numpy()}")
    
    # Check partition
    partition_object_argmax = partition_object.argmax(dim=-1)  # [B, N]
    print(f"\nPartition shape: {partition_object_argmax.shape}")
    print(f"Partition value range: [{partition_object_argmax.min().item()}, {partition_object_argmax.max().item()}]")
    print(f"  (should be 0-15 for 16 hand parts)")
    
    # Count points per part
    for b in range(batch_size):
        unique, counts = torch.unique(partition_object_argmax[b], return_counts=True)
        print(f"  Sample {b}: {len(unique)} unique parts assigned")
        print(f"    Part distribution: {dict(zip(unique.cpu().numpy(), counts.cpu().numpy()))}")
    
    # Check UV
    print(f"\nUV shape: {uv_object.shape}")
    print(f"  Expected: [batch_size={batch_size}, n_points={n_points}, 3]")
    uv_norm = uv_object.norm(dim=-1)  # Should be ~1.0 (normalized)
    print(f"UV norm statistics:")
    print(f"  Min: {uv_norm.min().item():.4f}")
    print(f"  Max: {uv_norm.max().item():.4f}")
    print(f"  Mean: {uv_norm.mean().item():.4f} (should be ~1.0)")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    # Basic sanity checks
    issues = []
    if contact_min < 0 or contact_max > 1:
        issues.append("⚠️  Contact values outside [0, 1] range")
    else:
        print("✓ Contact values in valid range [0, 1]")
    
    if partition_object_argmax.min() < 0 or partition_object_argmax.max() >= 16:
        issues.append("⚠️  Partition values outside [0, 15] range")
    else:
        print("✓ Partition values in valid range [0, 15]")
    
    if uv_norm.mean() < 0.9 or uv_norm.mean() > 1.1:
        issues.append("⚠️  UV vectors not properly normalized")
    else:
        print("✓ UV vectors properly normalized")
    
    if len(issues) == 0:
        print("\n✅ Model outputs look reasonable!")
        print("   The model appears to be working correctly.")
        print("   You can now test with real objects using demo_per_part.py")
    else:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"   {issue}")
    
    return contacts_object, partition_object, uv_object

if __name__ == '__main__':
    import sys
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else 'exp_per_part/checkpoint.pt'
    test_model_output(checkpoint)

