"""
Extract obj_re from the latest checkpoint of four reconstructions,
crop the central 2048^3, save each as a TIFF, and plot the central
z-slice of each plus (p0 - p1) differences on a single figure.
"""

import os
import glob
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
FIG_PATH  = f"{BASE}/compare_middle_slice.png"
CROP_SIZE = 2048


def latest_checkpoint(path_out):
    files = sorted(glob.glob(os.path.join(path_out, "checkpoints", "checkpoint_*.h5")))
    if not files:
        raise FileNotFoundError(f"No checkpoints in {path_out}/checkpoints")
    return files[-1]


def crop_bounds(full, crop):
    """Return (start, stop) indices for a centred crop of length min(crop, full)."""
    c = min(crop, full)
    s = (full - c) // 2
    return s, s + c


def extract_and_save(name, path_out):
    ckpt = latest_checkpoint(path_out)
    print(f"[{name}] using checkpoint: {ckpt}")

    with h5py.File(ckpt, "r") as f:
        ds = f["obj_re"]
        nz, ny, nx = ds.shape
        z0, z1 = crop_bounds(nz, CROP_SIZE)
        y0, y1 = crop_bounds(ny, CROP_SIZE)
        x0, x1 = crop_bounds(nx, CROP_SIZE)
        print(f"[{name}] shape {ds.shape}, crop z[{z0}:{z1}] y[{y0}:{y1}] x[{x0}:{x1}]")
        vol = ds[z0:z1, y0:y1, x0:x1].astype(np.float32, copy=False)

    tiff_path = os.path.join(OUT_DIR, f"{name}.tiff")
    tifffile.imwrite(tiff_path, vol, bigtiff=True)
    print(f"[{name}] wrote {tiff_path}  ({vol.shape}, {vol.nbytes/1e9:.1f} GB)")

    mid_slice = vol[vol.shape[0] // 2].copy()
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
    slices = {name: extract_and_save(name, path) for name, path in RUNS.items()}

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    add_panel(axes[0, 0], slices["cubic_p0"], "cubic p0")
    add_panel(axes[0, 1], slices["cubic_p1"], "cubic p1")
    add_panel(axes[0, 2], slices["cubic_p0"] - slices["cubic_p1"],
              "cubic p0 - p1", symmetric=True)

    add_panel(axes[1, 0], slices["fft_p0"], "fft p0")
    add_panel(axes[1, 1], slices["fft_p1"], "fft p1")
    add_panel(axes[1, 2], slices["fft_p0"] - slices["fft_p1"],
              "fft p0 - p1", symmetric=True)

    fig.suptitle(f"obj_re central z-slice (cropped {CROP_SIZE}^3)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    print(f"figure saved → {FIG_PATH}")


if __name__ == "__main__":
    main()
