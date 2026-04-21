"""Extract middle slices of obj_re from step6 checkpoint files into one file.

Usage:
    python extract_slices.py <path_out> [options]

Options:
    --iter N          extract only checkpoint N (default: all)
    --init <file>     path to _obj.h5 containing the Paganin initial guess
    --paganin <val>   paganin tag used in the initial guess dataset name (default: 60)
    --bin <val>       bin level used in the initial guess dataset name (default: 0)

Writes <path_out>/slices.h5 with:
  obj_re_xy  (n_ckpt, nobj,  nobj)  — horizontal, z = nzobj//2
  obj_re_xz  (n_ckpt, nzobj, nobj)  — vertical,   y = nobj//2
  obj_re_yz  (n_ckpt, nzobj, nobj)  — vertical,   x = nobj//2
  iters      (n_ckpt,)              — iteration numbers (-1 = initial guess)

If --init is given, the initial guess slice is prepended as the first entry.
"""

import sys
import os
import glob
import argparse
import h5py
import numpy as np


def read_init_slices(init_file, paganin, bin_level, nz_out, nobj_out):
    """Read middle slices from the Paganin initial guess, cropped to (nz_out, nobj_out)."""
    tag = int(paganin) if paganin == int(paganin) else paganin
    key = f'/exchange/obj_init_re{tag}_{bin_level}'
    with h5py.File(init_file, 'r') as f:
        if key not in f:
            raise KeyError(f'{key} not found in {init_file}')
        ds = f[key]
        nz0, ny0, nx0 = ds.shape
        stz = nz0 // 2 - nz_out // 2
        stx = nx0 // 2 - nobj_out // 2
        endx = nx0 // 2 + nobj_out // 2
        iz = nz0 // 2
        iy = ny0 // 2
        ix = nx0 // 2
        xy = ds[iz, stx:endx, stx:endx]
        xz = ds[stz:stz + nz_out, iy, stx:endx]
        yz = ds[stz:stz + nz_out, stx:endx, ix]
    return xy, xz, yz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path_out', help='Directory containing checkpoint_XXXX.h5 files')
    parser.add_argument('--iter',    type=int,   default=None, help='Extract a single iteration')
    parser.add_argument('--init',    default=None, help='Path to _obj.h5 with Paganin initial guess')
    parser.add_argument('--paganin', type=float, default=60,   help='Paganin tag (default: 60)')
    parser.add_argument('--bin',     type=int,   default=0,    help='Bin level (default: 0)')
    args = parser.parse_args()

    if args.iter is not None:
        paths = [os.path.join(args.path_out, f'checkpoint_{args.iter:04}.h5')]
    else:
        paths = sorted(glob.glob(os.path.join(args.path_out, 'checkpoint_????.h5')))

    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        print(f'No checkpoints found in {args.path_out}')
        sys.exit(1)

    print(f'Found {len(paths)} checkpoint(s)')

    with h5py.File(paths[0], 'r') as f:
        nz, ny, nx = f['obj_re'].shape

    has_init = args.init is not None
    n = len(paths) + (1 if has_init else 0)
    offset = 1 if has_init else 0

    xy_all = np.empty((n, ny, nx),  dtype='float32')
    xz_all = np.empty((n, nz, nx),  dtype='float32')
    yz_all = np.empty((n, nz, ny),  dtype='float32')
    iters  = np.empty(n, dtype='int32')

    if has_init:
        print(f'Reading initial guess from {args.init}')
        xy_all[0], xz_all[0], yz_all[0] = read_init_slices(
            args.init, args.paganin, args.bin, nz, nobj_out=nx)
        iters[0] = -1
        print(f'  initial guess  shape=({nz},{ny},{nx})')

    for i, p in enumerate(paths):
        with h5py.File(p, 'r') as f:
            vol = f['obj_re']
            xy_all[offset + i] = vol[nz // 2, :, :]
            xz_all[offset + i] = vol[:, ny // 2, :]
            yz_all[offset + i] = vol[:, :, nx // 2]
            iters[offset + i]  = f.attrs.get('iter', -1)
        print(f'  [{i+1}/{len(paths)}] iter={iters[offset+i]:4d}  {os.path.basename(p)}')

    out_path = os.path.join(args.path_out, 'slices.h5')
    with h5py.File(out_path, 'w') as dst:
        dst.create_dataset('obj_re_xy', data=xy_all)
        dst.create_dataset('obj_re_xz', data=xz_all)
        dst.create_dataset('obj_re_yz', data=yz_all)
        dst.create_dataset('iters',     data=iters)

    print(f'Saved → {out_path}')


if __name__ == '__main__':
    main()
