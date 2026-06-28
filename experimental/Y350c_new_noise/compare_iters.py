#!/usr/bin/env python
"""Extract reconstruction at two iterations, save the difference, plot mid-slices.

Reads:
    {path_out}/checkpoints/checkpoint_{iter:04d}.h5      (/obj_re, /obj_im)

Writes:
    {path_out}/diff_iter{i2:04d}_minus_{i1:04d}.h5
        /obj_re    (nzobj, nobj, nobj)   float32  — obj2.re - obj1.re   (delta)
        /obj_im    (nzobj, nobj, nobj)   float32  — obj2.im - obj1.im   (beta)
        attributes: iter1, iter2, source_path

    {path_out}/diff_iter{i2:04d}_minus_{i1:04d}.png
        2 rows (delta = real, beta = imag) × 3 cols (iter1, iter2, diff)
        each panel shows the middle z-slice; per-column colorbars.
"""

import os
import sys
import argparse
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_checkpoint(path):
    """Return (obj_re, obj_im) as (nzobj, nobj, nobj) float32 arrays."""
    with h5py.File(path, 'r') as f:
        re = f['/obj_re'][:].astype('float32')
        im = f['/obj_im'][:].astype('float32') if '/obj_im' in f else np.zeros_like(re)
    return re, im


def load_convergence(path_out):
    """Read {path_out}/conv.csv if present; return (iters, err) or (None, None)."""
    csv_path = os.path.join(path_out, 'conv.csv')
    if not os.path.exists(csv_path):
        return None, None
    df = pd.read_csv(csv_path)
    # Skip initial-state row (iter == -1) if present
    df = df[df['iter'] >= 0]
    return df['iter'].to_numpy(), df['err'].to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path_out',
                    help='step6 output dir (containing checkpoints/checkpoint_*.h5)')
    ap.add_argument('--i1', type=int, default=128, help='first iteration (default 128)')
    ap.add_argument('--i2', type=int, default=512, help='second iteration (default 512)')
    ap.add_argument('--out', default=None,
                    help='diff HDF5 file (default: {path_out}/diff_iter{i2}_minus_{i1}.h5)')
    ap.add_argument('--png', default=None,
                    help='figure file (default: same stem as --out, .png)')
    args = ap.parse_args()

    ckpt_dir = os.path.join(args.path_out, 'checkpoints')
    p1 = os.path.join(ckpt_dir, f'checkpoint_{args.i1:04d}.h5')
    p2 = os.path.join(ckpt_dir, f'checkpoint_{args.i2:04d}.h5')

    print(f'reading {p1}')
    re1, im1 = load_checkpoint(p1)
    print(f'reading {p2}')
    re2, im2 = load_checkpoint(p2)
    assert re1.shape == re2.shape, f'shape mismatch: {re1.shape} vs {re2.shape}'

    diff_re = re2 - re1
    diff_im = im2 - im1

    # ----------------- save diff h5 ---------------------------------------
    out_h5 = args.out or os.path.join(args.path_out,
                                      f'diff_iter{args.i2:04d}_minus_{args.i1:04d}.h5')
    print(f'writing {out_h5}')
    with h5py.File(out_h5, 'w') as f:
        f.create_dataset('/obj_re', data=diff_re)
        f.create_dataset('/obj_im', data=diff_im)
        f.attrs['iter1']       = args.i1
        f.attrs['iter2']       = args.i2
        f.attrs['source_path'] = args.path_out

    # ----------------- middle-slice figure --------------------------------
    nz = re1.shape[0]
    mid = nz // 2
    print(f'plotting mid-slice z={mid}')

    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)

    def panel(ax, img, title, cmap='gray', sym=False):
        if sym:
            vlim = float(np.percentile(np.abs(img), 99))
            im = ax.imshow(img, cmap=cmap, vmin=-vlim, vmax=+vlim)
        else:
            vmin = float(np.percentile(img, 1))
            vmax = float(np.percentile(img, 99))
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        return im

    # Row 0: delta (real part)
    im00 = panel(axes[0, 0], re1[mid], f'delta @ iter {args.i1}')
    im01 = panel(axes[0, 1], re2[mid], f'delta @ iter {args.i2}')
    im02 = panel(axes[0, 2], diff_re[mid],
                 f'delta diff ({args.i2} - {args.i1})', cmap='RdBu_r', sym=True)
    fig.colorbar(im00, ax=axes[0, 0], shrink=0.85)
    fig.colorbar(im01, ax=axes[0, 1], shrink=0.85)
    fig.colorbar(im02, ax=axes[0, 2], shrink=0.85)

    # Row 1: beta (imag part)
    im10 = panel(axes[1, 0], im1[mid], f'beta  @ iter {args.i1}')
    im11 = panel(axes[1, 1], im2[mid], f'beta  @ iter {args.i2}')
    im12 = panel(axes[1, 2], diff_im[mid],
                 f'beta  diff ({args.i2} - {args.i1})', cmap='RdBu_r', sym=True)
    fig.colorbar(im10, ax=axes[1, 0], shrink=0.85)
    fig.colorbar(im11, ax=axes[1, 1], shrink=0.85)
    fig.colorbar(im12, ax=axes[1, 2], shrink=0.85)

    fig.suptitle(f'{os.path.basename(args.path_out)}  —  middle z-slice (z={mid})',
                 fontsize=12)

    out_png = args.png or out_h5.replace('.h5', '.png')
    plt.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f'saved figure: {out_png}')

    # ----------------- numeric summary ------------------------------------
    print()
    print(f'shape (nzobj, nobj, nobj) = {re1.shape}')
    for name, arr in [('delta diff', diff_re), ('beta  diff', diff_im)]:
        print(f'{name}: min={arr.min():+.4e}  max={arr.max():+.4e}  '
              f'rms={np.sqrt((arr.astype(np.float64)**2).mean()):.4e}')


if __name__ == '__main__':
    main()
