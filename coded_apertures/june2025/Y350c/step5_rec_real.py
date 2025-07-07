#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cupy as cp
import h5py
import matplotlib.pyplot as plt
from types import SimpleNamespace
import pandas as pd
import h5py
import sys
import warnings
warnings.filterwarnings("ignore", message=f".*peer.*")

sys.path.insert(0, '..')
from utils import *
from rec import Rec
import os
import psutil
process = psutil.Process(os.getpid())


# # Init data sizes and parametes of the PXM of ID16A

# In[2]:

ngpus = int(sys.argv[1])
ndist= int(sys.argv[2])
bin = int(sys.argv[3])
nchunk = int(sys.argv[4])
step = int(sys.argv[5])
ntheta = int(sys.argv[6])
st = int(sys.argv[7])


os.system('hostname')

print(f'{ngpus=},{ndist=},{bin=},{nchunk=},{step=},{ntheta=},{st=}',flush=True)
print(f'{st*ntheta*step} - {st*ntheta*step+ntheta*step}',flush=True)
# In[3]:


pfile = f'Y350c_HT_015nm'
path_out = f'/data/vnikitin/ESRF/ID16A/brain_rec/20240515/Y350c2'
with h5py.File(f'{path_out}/{pfile}.h5') as fid:
    detector_pixelsize = fid['/exchange/detector_pixelsize'][0]    
    focusToDetectorDistance = fid['/exchange/focusdetectordistance'][0]    
    z1 = fid['/exchange/z1'][:ndist]        
    shape = fid['/exchange/data0'].shape
    shape_ref = fid['/exchange/data_white_start0'].shape
    shape_dark = fid['/exchange/data_dark0'].shape
    


# In[4]:


n = shape[-1]
n//=2**bin


# In[5]:


energy = 33.35  # [keV] xray energy
wavelength = 1.2398419840550367e-09/energy  # [m] wave length
z2 = focusToDetectorDistance-z1

magnifications = focusToDetectorDistance/z1
norm_magnifications = magnifications/magnifications[0]
distances = (z1*z2)/focusToDetectorDistance*norm_magnifications**2#!!!!
voxelsize = detector_pixelsize/magnifications[0]*2048/n  # object voxel size
show = False


# In[6]:


npsi = int(np.ceil(2048/norm_magnifications[-1]/16))*16  # make multiple of 8
npsi//=(2048//n)


# In[7]:


def _downsample(data, binning):
    res = data.copy()
    for k in range(binning):
        res = 0.5*(res[..., ::2,:]+res[..., 1::2,:])
        res = 0.5*(res[..., :, ::2]+res[..., :, 1::2])
    return res


# In[8]:


with h5py.File(f'{path_out}/{pfile}_corr.h5') as fid:
    r = (fid[f'/exchange/cshifts_final'][st*ntheta*step:st*ntheta*step+ntheta*step:step,:ndist]).astype('float32')
    psi_abs = (fid[f'/exchange/psi_init_abs'][st*ntheta*step:st*ntheta*step+ntheta*step:step,:]).astype('float32')
    psi_angle = (fid[f'/exchange/psi_init_angle'][st*ntheta*step:st*ntheta*step+ntheta*step:step,:]).astype('float32')
    psi = psi_abs*np.exp(1j*psi_angle)
    psi = _downsample(psi,bin)

    data = np.empty([ntheta,ndist,n,n],dtype='float32')
    for k in range(ndist):
        data[:,k] = np.sqrt(_downsample(fid[f'/exchange/data{k}'][st*ntheta*step:st*ntheta*step+ntheta*step:step],bin))                    
    ref = fid[f'/exchange/ref'][:ndist]    
    ref=_downsample(ref,bin)




# In[ ]:


args = SimpleNamespace()
args.ngpus = ngpus

args.n = n
args.npsi = npsi
args.ndist = ndist
args.ntheta = ntheta
args.pad = 0

args.nq = n + 2 * 0
args.nchunk = nchunk

args.voxelsize = voxelsize
args.wavelength = wavelength
args.distance = distances
args.norm_magnifications = norm_magnifications

args.niter = 257
args.vis_step = 32
args.err_step = 32
args.lam = 0

args.show = False


args.rho = [1,0.25,0.25]
args.path_out = f"{path_out}/{pfile}/rec_psi_{args.rho[0]}_{args.rho[1]}_{args.rho[2]}_{ndist}_{ntheta}_{st}"
cl_rec = Rec(args)    
q = np.empty([ndist,args.nq,args.nq],dtype='complex64')        
for j in range(ndist):
    q[j] = cl_rec.DT(np.sqrt(ref[j:j+1]),j)[0]
vars={}
vars["q"] = cp.array(q.copy())
vars["r"] = r
vars["r_init"] = r.copy()
vars["psi"] = psi
# data=cp.array(data)
vars["table"] = pd.DataFrame(columns=["iter", "err", "time"])    
vars = cl_rec.BH(data, vars)  
