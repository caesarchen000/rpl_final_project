# Working Scripts - Commands Reference

## 1. `extract_partition_map.py`
**Purpose**: Extract partition map, contact map, and other outputs from ContactGen model

**Command**:
```bash
python extract_partition_map.py \
  --obj_path grab_data/obj_meshes/mug.ply \
  --checkpoint checkpoint/checkpoint.pt \
  --save_root exp/extract_results \
  --n_samples 1 \
  --n_points 2048
```

**Options**:
- `--n_points 0`: Use ALL mesh vertices for dense visualization
- `--n_samples 5`: Generate 5 different samples (default)

**Output**: Creates `contact_map.npy`, `partition_hard.npy`, `sample_points.npy`, `partition_logits.npy`, `partition_probs.npy` in `save_root`

---

## 2. `partial_demo.py`
**Purpose**: Full pipeline - extract maps + optimize hand grasp pose

**Command**:
```bash
python partial_demo.py \
  --obj_path grab_data/obj_meshes/mug.ply \
  --checkpoint checkpoint/checkpoint.pt \
  --save_root exp/demo_results \
  --n_samples 1 \
  --w_contact 0.1 \
  --w_pene 3.0 \
  --w_uv 0.01
```

**Options**:
- `--w_contact`: Contact loss weight (default: 0.1, increase for better alignment)
- `--w_pene`: Penetration loss weight (default: 3.0)
- `--w_uv`: UV/direction loss weight (default: 0.01)

**Output**: Creates `grasp_0.obj` (hand mesh), `contact_map.npy`, `partition_hard.npy`, `sample_points.npy`, etc.

---

## 3. `visualize_partition_map.py`
**Purpose**: Visualize partition map (hand part assignments) on object mesh

**Command (partition only)**:
```bash
python visualize_partition_map.py \
  --obj_path grab_data/obj_meshes/mug.ply \
  --partition_hard exp/my_visualizations/mug_complete/partition_hard.npy \
  --sample_points exp/my_visualizations/mug_complete/sample_points.npy \
  --output exp/partition_visualization.obj \
  --sample_idx 0
```

**Command (with contact brightness)**:
```bash
python visualize_partition_map.py \
  --obj_path grab_data/obj_meshes/mug.ply \
  --partition_hard exp/my_visualizations/mug_complete/partition_hard.npy \
  --sample_points exp/my_visualizations/mug_complete/sample_points.npy \
  --contact_map exp/my_visualizations/mug_complete/contact_map.npy \
  --output exp/partition_with_contact.obj \
  --sample_idx 0 \
  --brightness_scale 1.0
```

**Options**:
- `--contact_map`: Optional - modulates brightness by contact probability
- `--brightness_scale`: Scale factor for brightness (default: 1.0)

**Output**: Creates OBJ file with vertex colors showing hand part assignments

---

## 4. `visualize_partition_with_contact.py`
**Purpose**: Visualize partition map with contact-based brightness modulation

**Command**:
```bash
python visualize_partition_with_contact.py \
  --obj_path grab_data/obj_meshes/mug.ply \
  --partition_hard exp/my_visualizations/mug_complete/partition_hard.npy \
  --contact_map exp/my_visualizations/mug_complete/contact_map.npy \
  --sample_points exp/my_visualizations/mug_complete/sample_points.npy \
  --output exp/partition_contact_combined.obj \
  --sample_idx 0 \
  --brightness_scale 1.0 \
  --min_brightness 0.2
```

**Options**:
- `--brightness_scale`: Scale factor for brightness (default: 1.0)
- `--min_brightness`: Minimum brightness to ensure colors are visible (default: 0.2)

**Output**: Creates OBJ file with:
- **Colors** = Hand part assignment (partition map)
- **Brightness** = Contact probability (contact map)

---

## Typical Workflow

### Step 1: Extract maps from model
```bash
python extract_partition_map.py \
  --obj_path grab_data/obj_meshes/mug.ply \
  --save_root exp/my_results \
  --n_samples 1
```

### Step 2: Generate grasp (optional)
```bash
python partial_demo.py \
  --obj_path grab_data/obj_meshes/mug.ply \
  --save_root exp/my_results \
  --n_samples 1
```

### Step 3: Visualize partition map
```bash
python visualize_partition_map.py \
  --obj_path grab_data/obj_meshes/mug.ply \
  --partition_hard exp/my_results/partition_hard.npy \
  --sample_points exp/my_results/sample_points.npy \
  --output exp/my_results/partition.obj
```

### Step 4: Visualize with contact brightness
```bash
python visualize_partition_with_contact.py \
  --obj_path grab_data/obj_meshes/mug.ply \
  --partition_hard exp/my_results/partition_hard.npy \
  --contact_map exp/my_results/contact_map.npy \
  --sample_points exp/my_results/sample_points.npy \
  --output exp/my_results/partition_with_contact.obj
```

