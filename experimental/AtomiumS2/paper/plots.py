import h5py
import dxchange
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib_scalebar.scalebar import ScaleBar
from holotomocupy.utils import *

## Parameters
n = 2048
ntheta = 1800
detector_pixelsize = 3.03751328912064e-6
energy = 33.35
wavelength = 1.24e-09 / energy
focustodetectordistance = 1.28

sx0 = 1.286e-3
z1 = np.array([4.236, 4.3625, 4.8685, 5.9195]) * 1e-3 - sx0
ndist = len(z1)
z2 = focustodetectordistance - z1

distances = (z1 * z2) / focustodetectordistance
magnifications = focustodetectordistance / z1
norm_magnifications = magnifications / magnifications[0]
voxelsizes = np.abs(detector_pixelsize / magnifications)
voxelsize = voxelsizes[0]

path = '/data2/vnikitin/atomium/20240924/AtomiumS2/'
pfile = 'AtomiumS2_HT_007nm'
path_out = '/data2/vnikitin/atomium_rec/20240924/AtomiumS2/'
file_out = f'data.h5'

print(voxelsize)
#1581.972670

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})
mpl.rcParams["xtick.labelsize"] = 22
mpl.rcParams["ytick.labelsize"] = 22

def mshow(a, show=False, **args):
    if show:
        if isinstance(a, cp.ndarray):
            a = a.get()
        fig, axs = plt.subplots(1, 1, figsize=(6,6))
        im = axs.imshow(a, cmap="gray", **args)
        cbar = fig.colorbar(im, fraction=0.046, pad=0.04)
        scalebar = ScaleBar(7e-3, "um", length_fraction=0.25, font_properties={
            "family": "serif","size": 25,
        }, location="lower right")
        cbar.ax.tick_params(labelsize=28)
        axs.add_artist(scalebar)

## Parse files and save everything to h5
os.makedirs(path_out, exist_ok=True)
for k in range(0, ndist):
    dname = f'{path}/{pfile}_{k+1}_'

    [n0, n1] = dxchange.read_edf(f'{dname}/ref{0:04}_0000.edf')[0].shape
    sty, endy = n0//2 - n//2, n0//2 + n//2
    stx, endx = n1//2 - n//2, n1//2 + n//2
    for id in range(20):
        data_white0 = dxchange.read_edf(f'{dname}/ref{id:04}_0000.edf')[0][sty:endy, stx:endx]

    for id in range(1):
        fname = f'{dname}/{pfile}_{k+1}_{id:04}.edf'

        data = dxchange.read_edf(fname)[0][sty:endy, stx:endx]

        mshow(data, True, vmax=12000, vmin=1000)
        plt.savefig(f"figs/data{k}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)

        mshow(data/data_white0, True, vmax=1.2, vmin=0.5)
        plt.savefig(f"figs/rdata{k}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
