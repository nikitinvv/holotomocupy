import os
import json
import numpy as np
from scipy.ndimage import laplace
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
    "font.size":        16,
})

# Data dimensions (from plots2.py)
nx, ny, nz = 3216, 3216, 2048

# Extents in data pixels [xmin, xmax, ymax, ymin]
extents = {
    'mrecz': [0, nx, ny, 0],  'precz': [0, nx, ny, 0],
    'mrecy': [0, nx, nz, 0],  'precy': [0, nx, nz, 0],
}
ylabels = {'mrecz': 'y (px)', 'precz': 'y (px)',
           'mrecy': 'z (px)', 'precy': 'z (px)'}

def crop_colorbar(img):
    gray = img[:, :, :3].mean(axis=2)
    row_white = (gray > 0.99).mean(axis=1)
    col_white = (gray > 0.99).mean(axis=0)
    top   = next(i for i in range(len(row_white)) if row_white[i] < 0.99)
    left  = next(j for j in range(len(col_white)) if col_white[j] < 0.99)
    right = next(j for j in range(len(col_white)-1, -1, -1) if col_white[j] < 0.99) + 1
    mid   = gray.shape[0] // 2
    sep   = next(i for i in range(mid, len(row_white)) if row_white[i] > 0.99)
    return img[top:sep, left:right]

def _metrics_from_png(img):
    gray = img[:, :, :3].mean(axis=2).astype('float32') if img.ndim == 3 else img.astype('float32')
    sharpness = float(np.std(laplace(gray)))
    flat = gray.ravel()
    thresh = np.median(flat)
    low, high = flat[flat < thresh], flat[flat > thresh]
    pooled_std = np.sqrt((np.std(low)**2 + np.std(high)**2) / 2)
    hs = float(abs(np.mean(high) - np.mean(low)) / pooled_std) if pooled_std > 0 else float('nan')
    # PNG-based fallback histogram (no float data available)
    counts, edges = np.histogram(1 - gray.ravel(), bins=80, range=(0, 1), density=True)
    return {'sharpness': sharpness, 'hs': hs,
            'hist_counts': counts.tolist(), 'hist_edges': edges.tolist()}

metric_path = "figs/metrics.json"
metrics = json.load(open(metric_path)) if os.path.exists(metric_path) else {}

def load(name):
    raw = f'figs/{name}_raw.png'
    if os.path.exists(raw):
        return mpimg.imread(raw), True
    return crop_colorbar(mpimg.imread(f'figs/{name}.png')), False

imgs   = {}
is_raw = {}
for name in ['mrecz', 'precz', 'mrecy', 'precy']:
    imgs[name], is_raw[name] = load(name)

h_z = imgs['mrecz'].shape[0]
h_y = imgs['mrecy'].shape[0]

fig = plt.figure(figsize=(14, 14 * (h_z + h_y) / (imgs['mrecz'].shape[1] * 2)))
gs  = gridspec.GridSpec(2, 2, height_ratios=[h_z, h_y], hspace=0.18, wspace=0.08)

layout     = [('mrecz', 'precz'), ('mrecy', 'precy')]
col_labels = ['Proposed', 'Conventional']

for ri, (left_key, right_key) in enumerate(layout):
    for ci, key in enumerate([left_key, right_key]):
        ax = fig.add_subplot(gs[ri, ci])
        ax.imshow(imgs[key], extent=extents[key] if is_raw[key] else None)
        if is_raw[key]:
            ax.tick_params(labelsize=18)
        else:
            ax.axis('off')

        # ── Inscribed histogram (top-right, top row only) ─────────────────────
        m = metrics.get(key, {})
        if 'hist_counts' not in m:
            m = _metrics_from_png(imgs[key])
        if ri == 0:
            counts  = np.array(m['hist_counts'])[1:-1]   # drop min/max bins
            edges   = np.array(m['hist_edges'])[1:-1]
            centers = (edges[:-1] + edges[1:]) / 2
            axh = ax.inset_axes([0.52, 0.72, 0.46, 0.26])
            axh.bar(centers, counts, width=edges[1]-edges[0], color='steelblue', linewidth=0)
            axh.grid(axis='both', linewidth=0.5, color='gray', alpha=0.6)
            axh.set_axisbelow(True)
            axh.set_xlim(edges[0], edges[-1])
            axh.set_ylim(0, counts.max() * 1.05)
            axh.set_xticks([]); axh.set_yticks([])
            for spine in axh.spines.values():
                spine.set_visible(False)
            axh.patch.set_facecolor('white')
            axh.patch.set_alpha(0.85)
            axh.patch.set_visible(True)

        # ── Metrics (bottom-right) ────────────────────────────────────────────
        label = f"S={m['sharpness']:.3f}\nHS={m['hs']:.3f}"
        ax.text(0.96, 0.04, label, transform=ax.transAxes,
                va='bottom', ha='right', fontsize=22, color='white',
                bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.5))

fig.savefig('figs/compare_full.png', dpi=200, bbox_inches='tight', pad_inches=0.05)
print('Saved figs/compare_full.png')
