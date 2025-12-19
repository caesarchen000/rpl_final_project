# Hand Part Color Legend

## Partition Map Colors (16 Hand Parts)

**ID layout (your convention):**
- 0: palm  
- 1–3: index finger (base → tip)  
- 4–6: middle finger (base → tip)  
- 7–9: ring finger (base → tip)  
- 10–12: pinky (base → tip)  
- 13–15: thumb (base → tip)  

| Part ID | Hand Part Name | Color | RGB Values |
|---------|----------------|-------|------------|
| 0 | palm | Brown | RGB(0.60, 0.35, 0.15) |
| 1 | index_0 | Light Yellow | RGB(1.00, 0.90, 0.30) |
| 2 | index_1 | Medium Yellow | RGB(1.00, 0.80, 0.05) |
| 3 | index_2 | Dark Yellow / Ochre | RGB(0.95, 0.70, 0.00) |
| 4 | middle_0 | Bright Green | RGB(0.20, 0.90, 0.20) |
| 5 | middle_1 | Teal Green | RGB(0.00, 0.80, 0.40) |
| 6 | middle_2 | Dark Green | RGB(0.00, 0.60, 0.00) |
| 7 | ring_0 | Sky Blue | RGB(0.30, 0.60, 1.00) |
| 8 | ring_1 | Medium Blue | RGB(0.10, 0.35, 0.95) |
| 9 | ring_2 | Deep Blue | RGB(0.00, 0.10, 0.80) |
| 10 | pinky_0 | Light Purple | RGB(0.80, 0.50, 1.00) |
| 11 | pinky_1 | Medium Purple | RGB(0.65, 0.30, 0.95) |
| 12 | pinky_2 | Deep Purple | RGB(0.50, 0.15, 0.80) |
| 13 | thumb_0 | Light Red | RGB(1.00, 0.35, 0.35) |
| 14 | thumb_1 | Medium Red | RGB(0.90, 0.15, 0.15) |
| 15 | thumb_2 | Deep Red | RGB(0.70, 0.05, 0.05) |

**Note**: When contact map is used for brightness modulation:
- Colors = Hand part assignment (from partition map)
- Brightness = Contact probability (from contact map)
- Formula: `final_color = part_color * contact_probability`

**⚠️ Important**: Always use files generated together from the same `extract_partition_map.py` run! Files from different runs may have different sample points, causing misalignment and incorrect visualizations.

---

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

**Output**: Creates in `save_root`:
- `contact_map.npy`: Batch contact map [B, N]
- `partition_hard.npy`: Batch partition assignments [B, N]
- `sample_points.npy`: Sampled points [N, 3]
- `partition_logits.npy`: Raw logits [B, N, 16]
- `partition_probs.npy`: Softmax probabilities [B, N, 16]
- Individual sample files: `contact_map_{i}.npy`, `partition_hard_{i}.npy`, etc. for each sample i

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

## 4. `visualize_partition_contact_multiply.py`
**Purpose**: Visualize partition map multiplied by contact map (recommended method)

**Command**:
```bash
python visualize_partition_contact_multiply.py \
  --obj_path grab_data/obj_meshes/toothpaste.ply \
  --partition_hard exp/partition_results/partition_hard_0.npy \
  --contact_map exp/partition_results/contact_map_0.npy \
  --sample_points exp/partition_results/sample_points.npy \
  --output exp/partition_times_contact_toothpaste.obj \
  --brightness_scale 1.0 \
  --min_brightness 0.0
```

**Important**: Use files generated together from the same `extract_partition_map.py` run to ensure alignment!

**Options**:
- `--brightness_scale`: Scale factor for brightness (default: 1.0)
- `--min_brightness`: Minimum brightness to ensure colors are visible (default: 0.0)

**Output**: Creates OBJ file with:
- **Colors** = Hand part assignment (partition map)
- **Brightness** = Contact probability (contact map)
- **Formula**: `final_color = partition_color * contact_probability`

---

## 5. `visualize_partition_with_contact.py`
**Purpose**: Alternative visualization script for partition map with contact-based brightness modulation

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

### Step 4: Visualize with contact brightness (recommended)
```bash
python visualize_partition_contact_multiply.py \
  --obj_path grab_data/obj_meshes/mug.ply \
  --partition_hard exp/my_results/partition_hard_0.npy \
  --contact_map exp/my_results/contact_map_0.npy \
  --sample_points exp/my_results/sample_points.npy \
  --output exp/my_results/partition_times_contact.obj
```

**Note**: `extract_partition_map.py` automatically creates individual sample files (`contact_map_0.npy`, `partition_hard_0.npy`, etc.). If you need to extract from the batch file manually:
```bash
python -c "import numpy as np; np.save('exp/my_results/contact_map_0.npy', np.load('exp/my_results/contact_map.npy')[0])"
```
