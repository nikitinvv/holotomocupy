"""
Visualisation: interpolation from a 48×48 extended object grid to a 32×32
detector grid, scaled by norm_magnifications.

4 panels (one per distance).  Each shows the 48×48 object grid (grey) with
the 32×32 output grid projected back onto it (coloured).  The output grid
cell size in object pixels is 1/norm_magnifications[j], and it covers a
region of width  w_j = n_det / norm_magnifications[j]  centred in the object.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl

mpl.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size':        22,
})

# ── same parameters as s.py ───────────────────────────────────────────────────
focustodetectordistance = 1.217
sx0 = -3.135e-3
z1_all = np.array([5.110, 5.464, 6.879, 9.817,
                   10.372, 11.146, 12.594, 17.209]) * 1e-3 - sx0
z1_ids = np.array([0, 1, 2, 3])
z1 = z1_all[z1_ids]
magnifications      = focustodetectordistance / z1
norm_magnifications = magnifications / magnifications[0]

n_obj = 28   # extended object grid
n_det = 16   # detector / output grid

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# ── helper: draw a pixel grid ─────────────────────────────────────────────────
def draw_grid(ax, x0, y0, ncells, cell, color, lw, alpha, label=None):
    """Draw an ncells×ncells grid starting at (x0,y0) with cell size `cell`."""
    size = ncells * cell
    for i in range(ncells + 1):
        kw = dict(color=color, lw=lw, alpha=alpha)
        ax.plot([x0, x0 + size], [y0 + i*cell, y0 + i*cell], **kw)
        ax.plot([x0 + i*cell, x0 + i*cell], [y0, y0 + size], **kw)
    if label is not None:
        ax.text(x0 + size + 0.3, y0 + size / 2, label,
                va='center', ha='left', fontsize=8,
                color=color, fontweight='bold')

# ─────────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(7)

fig, axes = plt.subplots(1, 4, figsize=(14, 5.5))
fig.subplots_adjust(wspace=0.05)

for j, ax in enumerate(axes):
    nm  = norm_magnifications[j]
    cell = 1.0 / nm          # output cell size in object pixels
    w    = n_det * cell       # total output region size in object pixels
    # shift in big-grid (N_u) coordinates; zero for reference distance
    if j == 0:
        shift_obj = np.zeros(2)
    else:
        max_shift_obj = min(2.0, (n_obj - w) / 2)
        shift_obj = rng.uniform(-max_shift_obj, max_shift_obj, size=2)
    x0 = (n_obj - w) / 2 + shift_obj[0]
    y0 = (n_obj - w) / 2 + shift_obj[1]

    ax.set_xlim(-0.5, n_obj + 0.5)
    ax.set_ylim(-0.5, n_obj + 0.5)
    ax.set_aspect('equal')
    m_val = 1.0 / nm
    ax.set_title(
        f'distance $z_{j}$\n'
        f'magnification $m = {m_val:.3f}$\n'
        f'$r = ({shift_obj[0]:+.2f},\\ {shift_obj[1]:+.2f})$',
        fontsize=18)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)

    # background: N_u x N_u object grid (1-pixel cells)
    draw_grid(ax, 0, 0, n_obj, 1.0, color='#aaaaaa', lw=0.4, alpha=0.6)

    # overlay: N x N output grid projected back into object space
    draw_grid(ax, x0, y0, n_det, cell, color=colors[j], lw=1.2, alpha=0.9)

    # shade the output region
    ax.add_patch(mpatches.Rectangle(
        (x0, y0), w, w,
        facecolor=colors[j], alpha=0.08, linewidth=0))


    # outer border with N_u label
    ax.add_patch(mpatches.Rectangle(
        (0, 0), n_obj, n_obj,
        linewidth=1.5, edgecolor='black', facecolor='none'))
    # ax.text(n_obj / 2, n_obj + 0.8, r'$N_u\times N_u$',
    #         ha='center', va='bottom', fontsize=9, color='black')

    # # N x N label inside the output region (first panel only, to avoid clutter)
    # if j == 0:
    #     ax.text(n_obj / 2, x0 + w / 2, r'$N\times N$',
    #             ha='center', va='center', fontsize=9,
    #             color=colors[j], fontweight='bold')

fig.tight_layout(rect=[0, 0, 1, 1])
plt.savefig('figs/interp_grids.pdf', bbox_inches='tight')
plt.savefig('figs/interp_grids.png', dpi=150, bbox_inches='tight')
print('Saved figs/interp_grids.pdf  and  figs/interp_grids.png')
plt.show()
