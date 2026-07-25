#!/usr/bin/env python
"""Build a 6-frame GIF from 2 slices in ~/cltx.tif.

Sequence (1 s per frame):
    1. full   — swapped[0]   label "original"
    2. full   — swapped[1]   label "joint"
    3. mid 2048² — swapped[0]  label "original"
    4. mid 2048² — swapped[1]  label "joint"
    5. mid-of-mid 1024² — swapped[0]  label "original"
    6. mid-of-mid 1024² — swapped[1]  label "joint"

Colorbar clamped to [-5, 0.1]. Order of the input pair is swapped before use.
"""
import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

SRC = os.path.expanduser('~/cltx.tif')
DST = os.path.expanduser('~/ctxl.gif')
VMIN, VMAX = -5.0, -1.0
FRAME_SEC = 3.0
LABELS = ['original', 'joint']

# --- Load and swap --------------------------------------------------------
img = tifffile.imread(SRC)                       # shape (2, H, W) or list
if img.ndim == 2:                                # single-page fallback
    img = np.stack([img, img])
elif img.ndim != 3 or img.shape[0] != 2:
    raise ValueError(f'{SRC}: expected shape (2, H, W); got {img.shape}')

slices = img[::-1].astype('float32')             # SWAP ORDER
H, W = slices.shape[1:]
assert H == W, f'expected square slices, got {H}×{W}'

# --- Build the 3 crops ---------------------------------------------------
def crop_center(a, size):
    s = (a.shape[-1] - size) // 2
    return a[..., s:s + size, s:s + size]

def crop_around(a, cy, cx, size):
    y0 = cy - size // 2
    x0 = cx - size // 2
    return a[..., y0:y0 + size, x0:x0 + size]

crops = [
    (slices,                              'full'),
    (crop_center(slices, 2048),           'mid 2048²'),
    (crop_center(slices, 1024),           'mid-of-mid 1024²'),
    (crop_around(slices, 2800, 2400, 1024), 'ROI 1024² @ (y=2800, x=2400)'),
]

# --- Render frames --------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_axis_off()
im = ax.imshow(np.zeros((2, 2), dtype='float32'), cmap='gray',
               vmin=VMIN, vmax=VMAX, interpolation='nearest')
label_txt = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                    color='white', fontsize=14, va='top', ha='left',
                    bbox=dict(facecolor='black', alpha=0.6, pad=4, edgecolor='none'))
crop_txt = ax.text(0.98, 0.98, '', transform=ax.transAxes,
                   color='white', fontsize=11, va='top', ha='right',
                   bbox=dict(facecolor='black', alpha=0.4, pad=3, edgecolor='none'))

writer = PillowWriter(fps=1 / FRAME_SEC)
with writer.saving(fig, DST, dpi=120):
    for arr, crop_name in crops:
        for k in range(2):
            im.set_data(arr[k])
            im.set_extent((0, arr.shape[-1], arr.shape[-2], 0))
            ax.set_xlim(0, arr.shape[-1])
            ax.set_ylim(arr.shape[-2], 0)
            label_txt.set_text(LABELS[k])
            crop_txt.set_text(crop_name)
            writer.grab_frame()
print(f'wrote {DST}  ({len(crops) * 2} frames, {FRAME_SEC:g}s each)')
