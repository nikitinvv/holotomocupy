#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cupy as cp
import h5py
import matplotlib.pyplot as plt
import cupyx.scipy.ndimage as ndimage
from types import SimpleNamespace
import pandas as pd

# Use managed memory
import h5py
import sys
import warnings
warnings.filterwarnings("ignore", message=f".*peer.*")

sys.path.insert(0, '..')
from utils import *
from rec import Rec


# # Init data sizes and parametes of the PXM of ID16A

# In[2]:



bin = int(sys.argv[1])
ndist = int(sys.argv[2])
step = int(sys.argv[3])
qrec_type = int(sys.argv[4])
rhoq = int(sys.argv[5])
rhor = int(sys.argv[6])
rotation_axis = float(sys.argv[7])
lam = float(sys.argv[8])
ngpus = int(sys.argv[9])


# In[3]:


with h5py.File(f'/data/vnikitin/ESRF/ID16A/20240924_rec/AtomiumS2/AtomiumS2_HT_007nm.h5') as fid:
    detector_pixelsize = fid['/exchange/detector_pixelsize'][0]    
    focusToDetectorDistance = fid['/exchange/focusdetectordistance'][0]    
    z1 = fid['/exchange/z1'][:]        
    theta = fid['/exchange/theta'][::step,0]
    shape = fid['/exchange/data0'][::step].shape
    


# In[4]:


theta = theta/180*np.pi


# In[5]:


ntheta,n = shape[:2]
n//=2**bin


# In[6]:


energy = 33.35  # [keV] xray energy
wavelength = 1.2398419840550367e-09/energy  # [m] wave length
z2 = focusToDetectorDistance-z1

magnifications = focusToDetectorDistance/z1
norm_magnifications = magnifications/magnifications[0]
distances = (z1*z2)/focusToDetectorDistance*norm_magnifications**2
voxelsize = detector_pixelsize/magnifications[0]*2048/n  # object voxel size
show = False


# In[7]:


# ndist = 4
pad = 0
npsi = int(np.ceil((2048+2*pad)/norm_magnifications[-1]/16))*16  # make multiple of 8
npsi//=(2048//n)


# In[ ]:


args = SimpleNamespace()
args.ngpus = ngpus

args.n = n
args.ndist = ndist
args.ntheta = ntheta
args.pad = pad

args.nq = n + 2 * pad
args.nchunk = 8

args.voxelsize = voxelsize
args.wavelength = wavelength
args.distance = distances
args.eps = 1e-8
args.rho = [1, 5, 3]
args.lam = lam
args.path_out = f"/data/vnikitin/ESRF/ID16A/20240924_rec/AtomiumS2/AtomiumS2_HT_007nm/sfinal_{bin}_{ndist}_{step}_{rhoq}_{rhor}_{rotation_axis}_{args.lam}"
args.show = False

args.niter=1
args.vis_step=1
args.err_step=1
args.method = "BH-CG"


args.rotation_axis=(rotation_axis)*n/1024#397.5*2#499.75*n//1024+npsi//2-n//2
args.npsi = npsi
args.theta = theta
args.norm_magnifications = norm_magnifications


# In[9]:


data = np.zeros([ntheta,ndist,n,n],dtype='float32')
with h5py.File(f'/data/vnikitin/ESRF/ID16A/20240924_rec/AtomiumS2/AtomiumS2_HT_007nm_corr.h5') as fid:
    for k in range(ndist):
        tmp = fid[f'/exchange/data{k}'][::step].copy()
        
        for j in range(bin):
            tmp = 0.5*(tmp[:,:,::2]+tmp[:,:,1::2])
            tmp = 0.5*(tmp[:,::2,:]+tmp[:,1::2,:])        
        data[:,k]=tmp.copy()
    tmp = fid[f'/exchange/ref'][:ndist]
    for j in range(bin):
        tmp = 0.5*(tmp[...,::2]+tmp[...,1::2])
        tmp = 0.5*(tmp[...,::2,:]+tmp[...,1::2,:])
    ref=tmp
    


# In[10]:


import scipy as sp

def _downsample(data, binning):
    res = data.copy()
    for k in range(binning):
        res = 0.5*(res[..., ::2,:]+res[..., 1::2,:])
        res = 0.5*(res[..., :, ::2]+res[..., :, 1::2])
    return res

def _downsample3d(data, binning):
    res = data.copy()
    for k in range(binning):
        res = 0.5*(res[..., ::2,:,:]+res[..., 1::2,:, :])
        res = 0.5*(res[..., ::2,:]+res[..., 1::2,:])
        res = 0.5*(res[..., :, ::2]+res[..., :, 1::2])
    return res

def _fftupsample(f, dims):
    paddim = np.zeros([np.ndim(f), 2], dtype='int32')
    dims = np.asarray(dims).astype('int32')
    paddim[dims, 0] = np.asarray(f.shape)[dims]//2
    paddim[dims, 1] = np.asarray(f.shape)[dims]//2
    fsize = f.size
    f = sp.fft.ifftshift(sp.fft.fftn(sp.fft.fftshift(
        f, dims), axes=dims, workers=-1), dims)
    f = np.pad(f, paddim)
    f = sp.fft.fftshift(f, dims)
    f = sp.fft.ifftn(f, axes=dims, workers=-1)
    f = sp.fft.ifftshift(f, dims)
    return f.astype('complex64')*(f.size/fsize)


# In[ ]:
nlevels = 4-bin
# iters = np.array([1025,384,129*4])
iters = np.array([1025,385,1000])
vis_steps = [16,16,16]
err_steps = [16,16,16]
chunks = [64,32,8]

for level in range(nlevels):    
    print(f'{level=}')    

    args.n = n//2**(nlevels-level-1)
    args.npsi = npsi//2**(nlevels-level-1)
    args.nq = (n + args.pad)//2**(nlevels-level-1)    
    args.voxelsize = voxelsize*2**(nlevels-level-1)    

    args.nchunk = chunks[level]
    args.niter=iters[level]
    args.vis_step=vis_steps[level]
    args.err_step=err_steps[level]
    args.rotation_axis=(rotation_axis)*n/1024/2**(nlevels-level-1)
    args.rho = [1,rhoq,rhor]

    if level==0:
        vars = {}                
        with h5py.File(f'/data/vnikitin/ESRF/ID16A/20240924_rec/AtomiumS2/AtomiumS2_HT_007nm_corr.h5') as fid:
            u_init = fid['/exchange/u_init_re'][:]+1j*fid['/exchange/u_init_imag'][:]                
            r_init = fid[f'/exchange/cshifts_final'][::step,:ndist]*args.n/2048
        if qrec_type==0:
            cl_rec = Rec(args)    
            q_init = np.ones([ndist,args.nq,args.nq],dtype='complex64')        
            ref_bin = _downsample(ref,nlevels-1)    
            for j in range(ndist):
                q_init[j] = cl_rec.DT(np.sqrt(ref_bin[j:j+1]),j)[0]
            q = q_init.copy()
        u = u_init.copy()        
        r = r_init.copy()
    else:
        u = _fftupsample(0.5*vars['u'],[0])
        u = _fftupsample(u,[1])
        u = _fftupsample(u,[2])        
        if qrec_type==0:
            q = _fftupsample(vars['q'].get(),[1])
            q = _fftupsample(q,[2])        
        r = vars['r']*2

    data_bin = _downsample(data,nlevels-level-1)        
    cl_rec = Rec(args)    
    if qrec_type==1:
        q_init = np.ones([ndist,args.nq,args.nq],dtype='complex64')        
        ref_bin = _downsample(ref,nlevels-level-1)    
        for j in range(ndist):
            q_init[j] = cl_rec.DT(np.sqrt(ref_bin[j:j+1]),j)[0]
        q = q_init.copy()
    vars["u"] = u
    vars["q"] = cp.array(q)
    vars["r"] = r
    with h5py.File(f'/data/vnikitin/ESRF/ID16A/20240924_rec/AtomiumS2/AtomiumS2_HT_007nm_corr.h5') as fid:
        vars["r_init"] = fid[f'/exchange/cshifts_final'][::step,:ndist]*args.n/2048#/2**(nlevels-1)
    vars["Ru"] = cl_rec.R(vars['u'])
    vars["psi"] = cl_rec.expR(vars['Ru'])        
    vars["table"] = pd.DataFrame(columns=["iter", "err", "time"])    
    vars = cl_rec.BH(data_bin, vars)  
    
    
    

