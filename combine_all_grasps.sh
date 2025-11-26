#!/bin/bash
# Combine all demo grasps with the object
# Usage: ./combine_all_grasps.sh

OBJ_PATH="exp/my_results/O02_0015_00026.ply"
OUTPUT_DIR="meshes"

mkdir -p "$OUTPUT_DIR"

for i in {0..9}; do
    HAND_PATH="exp/my_results/grasp_${i}.obj"
    if [ -f "$HAND_PATH" ]; then
        OUTPUT="${OUTPUT_DIR}/O02_0015_0002_grasp_${i}.obj"
        echo "Combining grasp $i..."
        python combine_grasp.py --hand_path "$HAND_PATH" --obj_path "$OBJ_PATH" --output "$OUTPUT"
    fi
done

echo ""
echo "✓ All grasps combined! Files saved in: $OUTPUT_DIR"

