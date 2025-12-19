#!/usr/bin/env python3
import numpy as np

data = np.load('tmp/tmp/final.npy')
print(f'Shape: {data.shape}')
print(f'\nFull array:')
print(data)
print(f'\n\nNon-zero entries summary:')
nonzero_rows = np.where(np.any(data > 0, axis=1))[0]
print(f'Rows with non-zero values: {len(nonzero_rows)} out of {data.shape[0]}')
print(f'\nAll rows with non-zero values:')
for i in nonzero_rows:
    nonzero_col = np.where(data[i] > 0)[0]
    if len(nonzero_col) > 0:
        col = nonzero_col[0]
        print(f'Row {i:4d}: Part {col:2d} = {data[i, col]:.6f}')

