#!/usr/bin/env python3
"""
Extract the middle z-slice of obj_re from checkpoint HDF5 files across
iterations and save as a multi-page TIFF stack.

Usage:
    python extract_tiff.py <path_out> [out.tiff]

    path_out  — directory containing checkpoint_NNNN.h5 files
    out.tiff  — output file (default: <path_out>/mid_slice_stack.tiff)
"""

import sys
import glob
import os
import h5py
import numpy as np
import tifffile


def extract_tiff_stack(path_out, out_file=None):
    checkpoints = sorted(glob.glob(os.path.join(path_out, "checkpoint_*.h5")))
    print(checkpoints)
    if not checkpoints:
        print(f"No checkpoints found in {path_out}")
        sys.exit(1)

    if out_file is None:
        out_file = os.path.join(path_out, "mid_slice_stack.tiff")

    # Get middle slice index from the first checkpoint
    with h5py.File(checkpoints[0], 'r') as f:
        nzobj = f['obj_re'].shape[0]
        mid   = nzobj // 2

    print(f"Found {len(checkpoints)} checkpoints, nzobj={nzobj}, middle slice z={mid}")

    slices = []
    iters  = []
    for ckpt in checkpoints[:100]:
        print(ckpt)
        with h5py.File(ckpt, 'r') as f:
            slices.append(f['obj_re'][mid].astype('float32'))
            iters.append(int(f.attrs.get('iter', -1)))

    stack = np.stack(slices, axis=0)   # [n_iter, nobj, nobj]

    # Write multi-page TIFF; iteration numbers go into the ImageDescription
    descriptions = [f"iter={it}" for it in iters]
    tifffile.imwrite(
        out_file,
        stack,
        imagej=True,
        metadata={'Labels': descriptions},
    )
    print(f"Saved stack {stack.shape} (float32) → {out_file}")
    print(f"Iterations: {iters}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path_out = sys.argv[1]
    out_file = sys.argv[2] if len(sys.argv) > 2 else None
    extract_tiff_stack(path_out, out_file)
