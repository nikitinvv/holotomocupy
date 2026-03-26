"""Standalone horizontal colorbar for brain electron density figures."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

os.makedirs('figs', exist_ok=True)

mpl.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size":        18,
    "xtick.labelsize":  16,
    "ytick.labelsize":  16,
})

wb     = LinearSegmentedColormap.from_list("white_black", ["white", "black"])
vmin   = 0.38   # e⁻/Å³  (near-empty embedding)
vmax   = 0.58   # e⁻/Å³  (dense myelin)

fig, ax = plt.subplots(figsize=(8, 0.8))
fig.subplots_adjust(bottom=0.35, top=0.65, left=0.05, right=0.95)

sm = plt.cm.ScalarMappable(cmap=wb, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])

cbar = fig.colorbar(sm, cax=ax, orientation='horizontal')
cbar.set_ticks([vmin, vmax])
cbar.set_label(r'Electron density (e$^-$/Å$^3$)', fontsize=26, labelpad=-26)
cbar.ax.tick_params(labelsize=26)

plt.savefig('figs/colorbar_brain.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.close()
print("Saved: figs/colorbar_brain.png")
