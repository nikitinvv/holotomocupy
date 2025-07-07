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
import scipy.ndimage as ndimage
warnings.filterwarnings("ignore", message=f".*peer.*")

sys.path.insert(0, '..')
from utils import *
from rec import Rec
import os
import psutil
process = psutil.Process(os.getpid())


# # Init data sizes and parametes of the PXM of ID16A

# In[2]:


rhor = int(sys.argv[1])
lam= int(sys.argv[2])
paganin= int(sys.argv[3])
step= int(sys.argv[4])

ngpus = cp.cuda.runtime.getDeviceCount()
bin = 0
ndist = 4


# In[3]:


pfile = f'Y350a_HT_nobin_020nm'
path_out = f'/data/vnikitin/ESRF/ID16A/brain_rec/20250604/Y350a'
with h5py.File(f'{path_out}/{pfile}.h5') as fid:
    detector_pixelsize = fid['/exchange/detector_pixelsize'][0]    
    focusToDetectorDistance = fid['/exchange/focusdetectordistance'][0]    
    z1 = fid['/exchange/z1'][:ndist]        
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
distances = (z1*z2)/focusToDetectorDistance*norm_magnifications**2#!!!!
z1p = z1[0]  # positions of the probe for reconstruction
z2p = z1-np.tile(z1p, len(z1))
# magnification when propagating from the probe plane to the detector
magnifications2 = (z1p+z2p)/z1p
# propagation distances after switching from the point source wave to plane wave,
distances2 = (z1p*z2p)/(z1p+z2p)
norm_magnifications2 = magnifications2/(z1p/z1[0])  # normalized magnifications
# scaled propagation distances due to magnified probes
distances2 = distances2*norm_magnifications2**2
distances2 = distances2*(z1p/z1)**2

voxelsize = detector_pixelsize/magnifications[0]*4096/n  # object voxel size
show = False


# In[7]:


print(f'{voxelsize=}')
print(f'{distances/distances[0]=}')


# In[8]:


pad = 0
npsi = int(np.ceil((4096+2*pad)/norm_magnifications[-1]/32))*32  # make multiple of 8
# npsi+=64
rotation_axis=npsi//2-11.791#(879-(1616-npsi//2)//2+2.5)*n/1024#n/2#(796.25+2)*n/1024#397.5*2#499.75*n//1024+npsi//2-n//2

print(rotation_axis)
npsi//=(4096//n)
rotation_axis/=(4096//n)


# In[9]:


args = SimpleNamespace()
args.ngpus = ngpus

args.n = n
args.ndist = ndist
args.ntheta = ntheta
args.pad = pad

args.nq = n + 2 * pad

args.voxelsize = voxelsize
args.wavelength = wavelength
args.distance = distances
args.distancep = distances2
args.path_out = f"{path_out}/{pfile}/rec_full_{rhor}_{lam}_{paganin}_{step}"
args.show = False

args.rotation_axis=rotation_axis
args.npsi = npsi
args.theta = theta
args.norm_magnifications = norm_magnifications


# In[10]:


with h5py.File(f'{path_out}/{pfile}_corr.h5') as fid:
    tmp = fid[f'/exchange/ref'][:ndist]
    for j in range(bin):
        tmp = 0.5*(tmp[...,::2]+tmp[...,1::2])
        tmp = 0.5*(tmp[...,::2,:]+tmp[...,1::2,:])
    ref=tmp
    


# In[11]:


import scipy as sp

def _downsample(data, binning):
    res = data.copy()
    for k in range(binning):
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

def upsample(f,zoom):
    f = ndimage.zoom(f,zoom,order=0)
    
    return f#f.astype('complex64')*(f.size/fsize)


# In[12]:


nlevels = 5
iters = np.array([1025,257,129,65*6])
vis_steps = [32,32,32,32][:nlevels]
err_steps = [32,32,32,32][:nlevels]
chunks = [64,32,8,1][:nlevels]

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
    args.rotation_axis=rotation_axis/2**(nlevels-level-1)
    
    args.rho = [1,25,rhor]
    if level==0:
        vars = {}                
        with h5py.File(f'{path_out}/{pfile}_corr.h5') as fid:
            u = fid[f'/exchange/u_init_re{paganin}'][:]+1j*fid[f'/exchange/u_init_imag{paganin}'][:]                
            r = (fid[f'/exchange/cshifts_final'][::step,:ndist]*args.n/4096).astype('float32')
            s = np.loadtxt('/data/vnikitin/ESRF/ID16A/brain/20240515/Y350c/Y350c_HT_015nm_/correct_correct3D.txt')[:3000:step][:,::-1]
            
        cl_rec = Rec(args)    
        q = np.ones([ndist,args.nq,args.nq],dtype='complex64')        
        ref = _downsample(ref,nlevels-1)    
        for j in range(ndist):
            q[j] = cl_rec.DT(np.sqrt(ref[j:j+1]),j)
        cl_rec=[]
    else:
        u = upsample(u,[2,2,2])
        q = upsample(q,[1,2,2])
        r = vars['r']*2

    data = np.zeros([ntheta,ndist,args.n,args.n],dtype='float32')
    with h5py.File(f'{path_out}/{pfile}_corr.h5') as fid:
        for j in range(0,ntheta):
            print(j)
            for k in range(ndist):
                data[j,k] = np.sqrt(_downsample(fid[f'/exchange/data{k}'][j*step],nlevels-level-1))  
                
    args.lam = lam
    args.path_out = f"{path_out}/{pfile}/rec_full_{rhor}_{lam}_{paganin}_{step}"
    cl_rec = Rec(args)    
    vars["u"] = u
    vars["q"] = cp.array(q)
    vars["r"] = r
    with h5py.File(f'{path_out}/{pfile}_corr.h5') as fid:
        vars["r_init"] = (fid[f'/exchange/cshifts_final'][::step,:ndist]*args.n/4096).astype('float32')#/2**(nlevels-1)        
    vars["psi"] = cl_rec.R(vars['u'])        
    vars["psi"][:] = cl_rec.expR(vars["psi"])        
    vars["table"] = pd.DataFrame(columns=["iter", "err", "time"])    
    
    vars = cl_rec.BH(data, vars)  

    u = vars['u']
    r = vars['r']
    q = vars['q'].get()

# In[ ]:

