# Workflow: Generate All Visualizations

## Quick Start

Generate contact maps, partition maps, and all visualizations from a single OBJ file:

```bash
python generate_all_visualizations.py \
  --obj_path <path_to_obj_file> \
  --output_dir <output_directory>
```

## Example

```bash
python generate_all_visualizations.py \
  --obj_path grab_data/obj_meshes/bottle.ply \
  --output_dir tmp/bottle
```

## What It Does

The script automatically runs 4 steps:

1. **Generate Maps** - Creates contact map, partition map, and sample points
2. **Visualize Heatmap** - Creates colored mesh showing contact probability
3. **Visualize Partition** - Creates colored mesh showing hand part assignments
4. **Visualize Combined** - Creates colored mesh showing partition × contact

## Output Files

All files are saved in the `--output_dir` directory:

### Visualizations (.obj files)
- `heatmap.obj` - Contact probability heatmap (red = high contact, black = low)
- `partition.obj` - Partition map showing which hand part touches each region
- `partition_contact.obj` - Combined visualization (partition colors × contact brightness)
- `grasp_0.obj` - Optimized hand mesh grasping the object
- `<object_name>.obj` - Copy of input object mesh

### Data Files (.npy files)
- `contact_map.npy` - Contact probabilities [N] (values 0-1)
- `part_hard.npy` - Partition assignments [N] (values 0-15, hand part IDs)
- `sample_points.npy` - Sample point coordinates [N, 3]
- `part_logits.npy` - Raw partition logits [N, 16]
- `part_probs.npy` - Partition probabilities [N, 16]

## Optional Parameters

```bash
python generate_all_visualizations.py \
  --obj_path <obj_file> \
  --output_dir <output_dir> \
  --checkpoint checkpoint/checkpoint.pt \    # Model checkpoint
  --n_samples 1 \                              # Number of samples
  --w_contact 0.1 \                            # Contact loss weight
  --w_pene 3.0 \                               # Penetration loss weight
  --w_uv 0.01 \                                # UV loss weight
  --brightness_scale 1.0 \                     # Brightness scale
  --min_brightness 0.0                          # Minimum brightness
```

## Requirements

- ✅ ContactGen model checkpoint (`checkpoint/checkpoint.pt`)
- ✅ `pointnet2_cuda` CUDA extension (for map generation)
- ✅ Python environment with all dependencies

## Notes

- All `.npy` files are generated together and aligned
- Use the same `sample_points.npy` for all visualizations
- The script will create the output directory if it doesn't exist
- Visualization files can be opened in any 3D viewer (MeshLab, Blender, etc.)

