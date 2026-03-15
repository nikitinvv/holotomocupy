#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import h5py 
import dxchange
import os
import numpy as np
from holotomocupy.utils import *


# ## Parameters

# In[ ]:


n = 2048
ntheta = 4500
detector_pixelsize = 1.4760147601476e-6 * 2
energy = 17.1
wavelength = 1.24e-09 / energy
focustodetectordistance = 1.217

sx0 = -3.135e-3
z1 = np.array([5.110, 5.464, 6.879, 9.817, 10.372, 11.146, 12.594, 17.209]) * 1e-3 - sx0
z1_ids = np.array([0, 1, 2, 3]) ### note using index starting from 0
str_z1_ids = ''.join(map(str, z1_ids + 1)) 
z1 = z1[z1_ids]
ndist = len(z1)
z2 = focustodetectordistance - z1

distances = (z1 * z2) / focustodetectordistance
magnifications = focustodetectordistance / z1
norm_magnifications = magnifications / magnifications[0]
voxelsizes = np.abs(detector_pixelsize / magnifications)
voxelsize = voxelsizes[0]

path = '/data2/vnikitin/brain/20251115/'
pfile = 'Y350a_HT_20nm_8dist'
path_out = '/data2/vnikitin/brain_rec/20251115/Y350a'
file_out = f'data{str_z1_ids}.h5'


# In[ ]:


import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],  # fallbacks
})
# mpl.rcParams['font.size'] = 28  # optional

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",   # Times-like math
    # optional:
    # "mathtext.default": "regular",
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
        scalebar = ScaleBar(20e-3, "um", length_fraction=0.25, font_properties={
            "family": "serif",
        },  # For more information, see the cell below                
        location="lower right")
        cbar.ax.tick_params(labelsize=28)
        axs.add_artist(scalebar)
        # plt.show()


# ### Parse files and save everything to h5

# In[ ]:


os.makedirs(path_out, exist_ok=True)
for k in range(0,ndist,3):
    dname = f'{path}/{pfile}_{z1_ids[k] + 1}_'

    [n0,n1] = dxchange.read_edf(f'{dname}/ref{0:04}_0000.edf')[0].shape
    sty,endy = n0//2 - n//2,n0//2 + n//2
    stx,endx = n1//2 - n//2,n1//2 + n//2
    for id in range(20):
        data_white0 = dxchange.read_edf(f'{dname}/ref{id:04}_0000.edf')[0][sty:endy,stx:endx]

    for id in range(1):
        fname = f'{dname}/{pfile}_{z1_ids[k] + 1}_{id:04}.edf'

        data = dxchange.read_edf(fname)[0][sty:endy,stx:endx]

        mshow(data,True,vmax=7000,vmin=1000)
        plt.savefig(f"figs/data{k}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)


        mshow(data/data_white0,True,vmax=1.3,vmin=0.5)
        plt.savefig(f"figs/rdata{k}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)

