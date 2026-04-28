"""Extract projections from _proj.h5 for a given bin level.

Usage:
    python extract_proj.py <main_h5> --bin 2 [options]

Options:
    --bin N          bin level to extract (default: 0)
    --step N         keep every N-th angle (default: 1 = all)
    --ids i j ...    explicit angle indices to extract (overrides --step)
    --out <file>     output file (default: <main_h5 stem>_proj_bin<N>_extracted.h5)

Writes:
  /proj   (n_angles, nobj_bin, nobj_bin)  float32
  /ids    (n_angles,)                     int32 — original angle indices
"""

import sys
import os
import argparse
import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('main_h5', help='Path to main .h5 file (e.g. y350a_80um_4k_08nm.h5)')
    parser.add_argument('--bin',  type=int, default=0, help='Bin level (default: 0)')
    parser.add_argument('--step', type=int, default=1, help='Keep every N-th angle (default: 1)')
    parser.add_argument('--ids',  type=int, nargs='+', default=None,
                        help='Explicit angle indices to extract')
    parser.add_argument('--out',  default=None, help='Output file path')
    args = parser.parse_args()

    proj_path = args.main_h5.replace('.h5', '_proj.h5')
    if not os.path.exists(proj_path):
        print(f'Error: {proj_path} not found')
        sys.exit(1)

    key = f'/exchange/proj_bin{args.bin}'

    with h5py.File(proj_path, 'r') as f:
        if key not in f:
            available = [k for k in f.get('/exchange', {}).keys()]
            print(f'Error: {key} not found. Available: {available}')
            sys.exit(1)
        ds = f[key]
        ntheta, ny, nx = ds.shape
        print(f'Dataset {key}: shape=({ntheta}, {ny}, {nx})  dtype={ds.dtype}')

        if args.ids is not None:
            ids = np.array(args.ids, dtype='int32')
        else:
            ids = np.arange(0, ntheta, args.step, dtype='int32')

        print(f'Extracting {len(ids)} angles ...')
        out_data = np.empty((len(ids), ny, nx), dtype='float32')
        for i, idx in enumerate(ids):
            out_data[i] = ds[idx]
            if (i + 1) % 100 == 0 or i == len(ids) - 1:
                print(f'  {i+1}/{len(ids)}', end='\r')
        print()

    if args.out is None:
        stem = args.main_h5.replace('.h5', '')
        out_path = f'{stem}_proj_bin{args.bin}_extracted.h5'
    else:
        out_path = args.out

    with h5py.File(out_path, 'w') as dst:
        dst.attrs['source'] = proj_path
        dst.attrs['bin']    = args.bin
        dst.create_dataset('proj', data=out_data)
        dst.create_dataset('ids',  data=ids)

    print(f'Saved → {out_path}  shape={out_data.shape}')


if __name__ == '__main__':
    main()
