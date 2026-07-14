"""
Extract obj_re from a specific checkpoint of four reconstructions,
crop the central 2048^3, save each as a TIFF, and plot the central
z-slice of each plus (p0 - p1) differences on a single figure.

Usage:
    python extract_and_compare.py <iter>
e.g. `python extract_and_compare.py 512` uses checkpoint_0512.h5 in every run.
"""

import os
import sys
import h5py
import numpy as np
import tifffile
import matplotlib.pyplot as plt

BASE = "/eagle/APS_IRI/vnikitin/20240515"

RUNS = {
    "cubic_p0": f"{BASE}/Y350c_rec_new_cubic_p0",
    "cubic_p1": f"{BASE}/Y350c_rec_new_cubic_p1",
    "fft_p0":   f"{BASE}/Y350c_rec_new_fft_p0",
    "fft_p1":   f"{BASE}/Y350c_rec_new_fft_p1",
}

OUT_DIR   = BASE
CROP_SIZE = 2048
Z_CHUNK   = 32          # z-slices streamed per h5→tiff write (~ny*nx*4*Z_CHUNK bytes)


def checkpoint_path(path_out, it):
    p = os.path.join(path_out, "checkpoints", f"checkpoint_{it:04d}.h5")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p


def crop_bounds(full, crop):
    """Return (start, stop) indices for a centred crop of length min(crop, full)."""
    c = min(crop, full)
    s = (full - c) // 2
    return s, s + c


def extract_and_save(name, path_out, it):
    """Stream the central crop from h5 to a BigTIFF in z-slabs; return middle slice."""
    ckpt = checkpoint_path(path_out, it)
    print(f"[{name}] using checkpoint: {ckpt}")

    tiff_path = os.path.join(OUT_DIR, f"{name}.tiff")

    with h5py.File(ckpt, "r") as f:
        ds = f["obj_re"]
        nz, ny, nx = ds.shape
        z0, z1 = crop_bounds(nz, CROP_SIZE)
        y0, y1 = crop_bounds(ny, CROP_SIZE)
        x0, x1 = crop_bounds(nx, CROP_SIZE)
        cz, cy, cx = z1 - z0, y1 - y0, x1 - x0
        print(f"[{name}] src {ds.shape}, crop → ({cz},{cy},{cx}); "
              f"target size {cz*cy*cx*4/1e9:.1f} GB")

        mid_z_src   = z0 + cz // 2                # index in the source volume
        mid_slice   = None

        # Write each z-slab as its own TIFF page(s) so the whole volume never
        # lives in RAM. Peak buffer per iteration: Z_CHUNK * cy * cx * 4 bytes.
        with tifffile.TiffWriter(tiff_path, bigtiff=True) as tw:
            for zs in range(z0, z1, Z_CHUNK):
                ze = min(zs + Z_CHUNK, z1)
                slab = ds[zs:ze, y0:y1, x0:x1].astype(np.float32, copy=False)
                for k in range(slab.shape[0]):
                    tw.write(slab[k], contiguous=True)
                    if (zs + k) == mid_z_src:
                        mid_slice = slab[k].copy()
                print(f"[{name}]  wrote z=[{zs-z0}:{ze-z0}] / {cz}")

    if mid_slice is None:
        raise RuntimeError(f"[{name}] failed to capture middle slice")
    print(f"[{name}] wrote {tiff_path}")
    return mid_slice


def add_panel(ax, img, title, symmetric=False):
    if symmetric:
        v = float(np.max(np.abs(img)))
        im = ax.imshow(img, cmap="gray", vmin=-v, vmax=v)
    else:
        lo, hi = np.percentile(img, [1, 99])
        im = ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python extract_and_compare.py <iter>")
    it = int(sys.argv[1])
    slices = {name: extract_and_save(name, path, it) for name, path in RUNS.items()}
    fig_path = f"{BASE}/compare_middle_slice_{it:04d}.png"

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    add_panel(axes[0, 0], slices["cubic_p0"], "cubic p0")
    add_panel(axes[0, 1], slices["cubic_p1"], "cubic p1")
    add_panel(axes[0, 2], slices["cubic_p0"] - slices["cubic_p1"],
              "cubic p0 - p1", symmetric=True)

    add_panel(axes[1, 0], slices["fft_p0"], "fft p0")
    add_panel(axes[1, 1], slices["fft_p1"], "fft p1")
    add_panel(axes[1, 2], slices["fft_p0"] - slices["fft_p1"],
              "fft p0 - p1", symmetric=True)

    fig.suptitle(f"obj_re central z-slice (iter {it}, cropped {CROP_SIZE}^3)", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    print(f"figure saved → {fig_path}")


if __name__ == "__main__":
    main()
