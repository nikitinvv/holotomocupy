#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cupy as cp
import h5py
import matplotlib.pyplot as plt
import cupyx.scipy.ndimage as ndimage
# Use managed memory
import h5py
import sys
import warnings
warnings.filterwarnings("ignore", message=f".*peer.*")

sys.path.insert(0, '..')
from utils import *


# # Init data sizes and parametes of the PXM of ID16A

# In[2]:


step = 1


# In[3]:


path = f'/data/vnikitin/ESRF/ID16A/brain/20250604/Y350a/'
pfile = f'Y350a_HT_nobin_020nm'
path_out = f'/data/vnikitin/ESRF/ID16A/brain_rec/20250604/Y350a/'
with h5py.File(f'{path_out}/{pfile}.h5') as fid:
    detector_pixelsize = fid['/exchange/detector_pixelsize'][0]    
    focusToDetectorDistance = fid['/exchange/focusdetectordistance'][0]    
    z1 = fid['/exchange/z1'][:]        
    theta = fid['/exchange/theta'][::step]
    shifts = fid['/exchange/shifts'][::step]
    attrs = fid['/exchange/attrs'][::step]
    pos_shifts = fid['/exchange/pos_shifts'][::step]*1e-6
    shape = fid['/exchange/data0'][::step].shape
    shape_ref = fid['/exchange/data_white_start0'].shape
    shape_dark = fid['/exchange/data_dark0'].shape
    #pos_shifts-=pos_shifts[0]


# In[4]:


ndist=4
ntheta,n = shape[:2]
ndark = shape_dark[0]
nref = shape_ref[0]


energy = 33.35  # [keV] xray energy
wavelength = 1.2398419840550367e-09/energy  # [m] wave length
z2 = focusToDetectorDistance-z1
distances = (z1*z2)/focusToDetectorDistance
magnifications = focusToDetectorDistance/z1
norm_magnifications = magnifications/magnifications[0]
show = True


shifts_random = np.zeros([ntheta,ndist,2],dtype='float32')
for k in range(ndist):
    shifts_random[:,k,0] = shifts[:,k,1]/norm_magnifications[k]    #+(1024-(2048+0-0)/2)*(1/norm_magnifications[k]-1)#/norm_magnifications[k]
    shifts_random[:,k,1] = shifts[:,k,0]/norm_magnifications[k]    #+(1024-(2048+0-0)/2)*(1/norm_magnifications[k]-1)#/norm_magnifications[k]

import scipy
pshifts = scipy.io.loadmat('/data/vnikitin/ESRF/ID16A/brain/20250604/Y350a/Y350a_HT_nobin_020nm_/rhapp_fixed.mat')['pshifts'][0,0][0]
pshifts=-pshifts.swapaxes(0,2)[:4000:step]

shifts = cp.array(pshifts)+cp.array(shifts_random)

# s = np.loadtxt('/data/vnikitin/ESRF/ID16A/brain/20250604/Y350a/Y350a_HT_nobin_020nm_/correct_correct3D.txt')[:4000:step][:,::-1]

shifts_final = shifts.get()#+s[:,np.newaxis]

with h5py.File(f'{path_out}/{pfile}_corr.h5','a') as fid:
    try:
        del fid[f'/exchange/cshifts_final']
    except:
        pass
    fid.create_dataset(f'/exchange/cshifts_final',data = shifts_final)

