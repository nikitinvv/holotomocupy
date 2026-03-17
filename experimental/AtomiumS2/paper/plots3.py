import os
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib as mpl

os.makedirs('figs', exist_ok=True)

ckpt_path = '/data2/vnikitin/alcf/atomium/checkpoint_1344.h5'

# ── Matplotlib style ──────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

def mshow(a, **kwargs):
    fig, axs = plt.subplots(1, 1, figsize=(6, 6))
    im = axs.imshow(a, cmap="gray", **kwargs)
    cbar = fig.colorbar(im, fraction=0.046, pad=0.1,
                        orientation="horizontal", location="bottom")
    scalebar = ScaleBar(7e-3, "um", length_fraction=0.25,
                        font_properties={"family": "serif"},
                        location="lower right")
    cbar.ax.tick_params(labelsize=20)
    axs.add_artist(scalebar)

# ── Probe ─────────────────────────────────────────────────────────────────────
mpl.rcParams['font.size']          = 24
mpl.rcParams["xtick.labelsize"]    = 16
mpl.rcParams["ytick.labelsize"]    = 16

print("Loading probe ...")
with h5py.File(ckpt_path, 'r') as f:
    prb_abs   = f['prb_abs'][:]    # (4, 2048, 2048)
    prb_phase = f['prb_phase'][:]  # (4, 2048, 2048)

mshow(prb_abs[0],   vmax=1.5, vmin=0.5);  plt.savefig("figs/prbabs0.png",   dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow(prb_phase[0], vmax=1.2, vmin=-1.2); plt.savefig("figs/prbangle0.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow(prb_abs[3],   vmax=1.5, vmin=0.5);  plt.savefig("figs/prbabs3.png",   dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow(prb_phase[3], vmax=2.4, vmin=-2.4); plt.savefig("figs/prbangle3.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()

# ── Position errors ───────────────────────────────────────────────────────────
mpl.rcParams['font.size']          = 28
mpl.rcParams["xtick.labelsize"]    = 28
mpl.rcParams["ytick.labelsize"]    = 28

print("Loading positions ...")
with h5py.File('/data2/vnikitin/atomium_rec/20240924/AtomiumS2/data.h5', 'r') as f:
    pos_init = f['/exchange/cshifts_final'][:]  # (1800, 4, 2)
with h5py.File(ckpt_path, 'r') as f:
    pos = f['pos'][:]  # (1800, 4, 2)
pos_init[..., 1] += -27.00227
pos_init[..., 0] += 400
pos = pos - pos_init

labels = ["0", "1", "2", "3"]

fig, ax = plt.subplots(1, 1, figsize=(10, 4))
l = ax.plot(pos[:, :, 1], ".")
ax.set_xlabel('angle')
ax.set_ylabel('horizontal error')
leg = ax.legend(l, labels, title="distance", ncol=len(labels), loc='upper right',
                columnspacing=1, handletextpad=0.1, handlelength=0.6,
                fontsize=26, markerscale=4.0)
leg._legend_box.align = "left"
ax.grid()
plt.savefig("figs/errx.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.close()

fig, ax = plt.subplots(1, 1, figsize=(10, 4))
l = ax.plot(pos[:, :, 0], ".")
ax.set_xlabel('angle')
ax.set_ylabel('vertical error')
leg = ax.legend(l, labels, title="distance", ncol=len(labels), loc='upper right',
                columnspacing=1, handletextpad=0.1, handlelength=0.6,
                fontsize=26, markerscale=4.0)
leg._legend_box.align = "left"
ax.grid()
plt.savefig("figs/erry.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.close()

print("Done — figures saved to figs/")
