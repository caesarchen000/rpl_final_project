#!/usr/bin/env python3
"""
Print detailed information about final.npy file including shape, size, and contents.
"""
import numpy as np
import os
import sys

def print_final_info(npy_path):
    """Print comprehensive information about final.npy file."""
    if not os.path.exists(npy_path):
        print(f"Error: File not found: {npy_path}")
        sys.exit(1)
    
    data = np.load(npy_path)
    file_size_mb = os.path.getsize(npy_path) / (1024 * 1024)
    array_size_mb = data.nbytes / (1024 * 1024)
    
    print("=" * 70)
    print(f"File: {npy_path}")
    print("=" * 70)
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    print(f"File size: {file_size_mb:.6f} MB")
    print(f"Array size in memory: {array_size_mb:.6f} MB")
    print(f"Total elements: {data.size:,}")
    
    # Statistics
    nonzero_rows = np.where(np.any(data > 0, axis=1))[0]
    nonzero_count = len(nonzero_rows)
    total_rows = data.shape[0]
    
    print(f"\n{'=' * 70}")
    print("Statistics:")
    print(f"{'=' * 70}")
    print(f"Rows with non-zero values: {nonzero_count} out of {total_rows} ({100*nonzero_count/total_rows:.2f}%)")
    print(f"Rows with all zeros: {total_rows - nonzero_count} ({100*(total_rows-nonzero_count)/total_rows:.2f}%)")
    
    if nonzero_count > 0:
        nonzero_values = data[data > 0]
        print(f"\nNon-zero value statistics:")
        print(f"  Min: {np.min(nonzero_values):.6f}")
        print(f"  Max: {np.max(nonzero_values):.6f}")
        print(f"  Mean: {np.mean(nonzero_values):.6f}")
        print(f"  Median: {np.median(nonzero_values):.6f}")
        print(f"  Std: {np.std(nonzero_values):.6f}")
    
    # Part distribution
    if nonzero_count > 0:
        part_counts = {}
        for i in nonzero_rows:
            nonzero_col = np.where(data[i] > 0)[0]
            if len(nonzero_col) > 0:
                part_id = nonzero_col[0]
                part_counts[part_id] = part_counts.get(part_id, 0) + 1
        
        print(f"\n{'=' * 70}")
        print("Part distribution (which hand parts are assigned):")
        print(f"{'=' * 70}")
        for part_id in sorted(part_counts.keys()):
            count = part_counts[part_id]
            print(f"  Part {part_id:2d}: {count:4d} points ({100*count/nonzero_count:.2f}%)")
    
    # Sample of non-zero entries
    print(f"\n{'=' * 70}")
    print("Sample of non-zero entries (first 20):")
    print(f"{'=' * 70}")
    sample_size = min(20, nonzero_count)
    for idx, i in enumerate(nonzero_rows[:sample_size]):
        nonzero_col = np.where(data[i] > 0)[0]
        if len(nonzero_col) > 0:
            col = nonzero_col[0]
            print(f"Row {i:4d}: Part {col:2d} = {data[i, col]:.6f}")
    
    if nonzero_count > 20:
        print(f"... and {nonzero_count - 20} more rows with non-zero values")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Print detailed information about final.npy file")
    parser.add_argument("--npy_path", type=str, default="tmp/tmp/final.npy",
                        help="Path to final.npy file (default: tmp/tmp/final.npy)")
    args = parser.parse_args()
    
    print_final_info(args.npy_path)


