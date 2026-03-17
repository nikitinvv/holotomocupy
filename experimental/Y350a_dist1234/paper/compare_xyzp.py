import json
import os
import numpy as np
from scipy.ndimage import laplace, gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib as mpl

mpl.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size":        14,
})

# Grid: 4 rows x 3 cols
# rows = (mrec0, prec0, mrec1, prec1), cols = (z, y, x)
# mrec = Proposed, prec = Conventional; proposed comes first

def _metrics_from_png(img):
    """Fallback: compute metrics from PNG pixel values."""
    gray = img[:, :, :3].mean(axis=2).astype('float32') if img.ndim == 3 else img.astype('float32')
    sharpness = float(np.std(laplace(gray)))
    flat = gray.ravel()
    thresh = np.median(flat)
    low, high = flat[flat < thresh], flat[flat > thresh]
    pooled_std = np.sqrt((np.std(low)**2 + np.std(high)**2) / 2)
    hs = float(abs(np.mean(high) - np.mean(low)) / pooled_std) if pooled_std > 0 else float('nan')
    return {'sharpness': sharpness, 'hs': hs}

metric_path = "figs/metrics.json"
metrics = json.load(open(metric_path)) if os.path.exists(metric_path) else {}

directions = ['z', 'y', 'x']

# (rec, tag) for each of the 4 rows
rows = [('mrec', '0'), ('prec', '0'), ('mrec', '1'), ('prec', '1')]

col_labels = ['z-slice', 'y-slice', 'x-slice']
row_labels = ['Proposed\n(patch 0)', 'Conventional\n(patch 0)',
              'Proposed\n(patch 1)', 'Conventional\n(patch 1)']

# 5-row GridSpec: rows 0,1 = patch 0; row 2 = spacer; rows 3,4 = patch 1
fig = plt.figure(figsize=(10.5, 14.5))
gs  = gridspec.GridSpec(5, 3, height_ratios=[1, 1, 0.06, 1, 1],
                        hspace=0.04, wspace=0.04)
gs_rows = [0, 1, 3, 4]  # skip spacer row 2
axes = [[fig.add_subplot(gs[gs_rows[ri], ci]) for ci in range(3)] for ri in range(4)]

for ri, (rec, tag) in enumerate(rows):
    for ci, direction in enumerate(directions):
        fname = f"figs/{rec}{direction}p{tag}.png"
        img   = mpimg.imread(fname)
        ax    = axes[ri][ci]
        ax.imshow(img)
        ax.axis('off')
        key = f"{rec}{direction}p{tag}"
        m = metrics.get(key, _metrics_from_png(img))
        label = f"S={m['sharpness']:.3f}\nHS={m['hs']:.3f}"
        ax.text(0.96, 0.04, label, transform=ax.transAxes,
                va='bottom', ha='right', fontsize=16, color='white',
                bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.5))


fig.savefig("figs/compare_xyzp.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
print("Saved figs/compare_xyzp.png")
