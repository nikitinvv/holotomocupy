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
lam=float(sys.argv[1])
ngpus=int(sys.argv[2])
rhoq = float(sys.argv[3])
rhor = float(sys.argv[4])

step = 1
bin = 0
ndist = 4


# In[3]:


pfile = f'Y350c_HT_015nm'
path_out = f'/data/vnikitin/ESRF/ID16A/brain_rec/20240515/Y350c'
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
voxelsize = detector_pixelsize/magnifications[0]*2048/n  # object voxel size
show = True


# In[ ]:


print(f'{voxelsize=}')
print(f'{distances/distances[0]=}')


# In[8]:


pad = 0
npsi = int(np.ceil((2048+2*pad)/norm_magnifications[-1]/16))*16  # make multiple of 8
npsi=3232
rotation_axis=(879-(1616-npsi//2)//2+2.5)*n/1024#n/2#(796.25+2)*n/1024#397.5*2#499.75*n//1024+npsi//2-n//2
npsi//=(2048//n)


# In[9]:


args = SimpleNamespace()
args.ngpus = ngpus

args.n = n
args.ndist = ndist
args.ntheta = ntheta
args.pad = pad

args.nq = n + 2 * pad
args.nchunk = 4

args.voxelsize = voxelsize
args.wavelength = wavelength
args.distance = distances
args.eps = 1e-12
args.rho = [1, 5, 3]

args.path_out = f"{path_out}/{pfile}/rec_final4_{ndist}_{rotation_axis}_{lam}_{rhoq}_{rhor}"
args.show = False

args.niter=1
args.vis_step=1
args.err_step=1


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


# In[ ]:


nlevels = 4
iters = np.array([513,127,65,3300])
# iters = np.array([3,3,3,3])[:nlevels]
vis_steps = [16,16,16,16][:nlevels]
err_steps = [16,16,16,16][:nlevels]
chunks = [32,16,4,1][:nlevels]

# iters = np.array([0,0,0,3])
# # iters = np.array([3,3,3,3])[:nlevels]
# vis_steps = [1,-1,-1,-1][:nlevels]
# err_steps = [-1,-1,-1,-1][:nlevels]
# chunks = [32,16,4,1][:nlevels]

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
    
    args.rho = [1,rhoq,rhor]
    if level==0:
        vars = {}                
        with h5py.File(f'{path_out}/{pfile}_corr.h5') as fid:
            u = fid['/exchange/u_init_re'][:]+1j*fid['/exchange/u_init_imag'][:]                
            r = (fid[f'/exchange/cshifts_final'][::step,:ndist]*args.n/2048).astype('float32')
        cl_rec = Rec(args)    
        q = np.ones([ndist,args.nq,args.nq],dtype='complex64')        
        ref = _downsample(ref,nlevels-1)    
        for j in range(ndist):
            q[j] = cl_rec.DT(np.sqrt(ref[j:j+1]),j)[0]
        cl_rec=[]
    else:
        u = _fftupsample(0.5*vars['u'],[0])
        u = _fftupsample(u,[1])
        u = _fftupsample(u,[2])        
        q = _fftupsample(vars['q'].get(),[1])
        q = _fftupsample(q,[2])        
        r = vars['r']*2

    data = np.zeros([ntheta,ndist,n,n],dtype='float32')
    with h5py.File(f'{path_out}/{pfile}_corr.h5') as fid:
        for k in range(ndist):
            data[:,k] = fid[f'/exchange/data{k}'][::step].copy()                    
    data = np.sqrt(_downsample(data,nlevels-level-1))  

    # data = np.ones([ntheta,ndist,n//2**(nlevels-level-1),n//2**(nlevels-level-1)],dtype='float32')
    

    args.lam = lam
    cl_rec = Rec(args)    
    vars["u"] = u
    vars["q"] = cp.array(q)
    vars["r"] = r
    with h5py.File(f'{path_out}/{pfile}_corr.h5') as fid:
        vars["r_init"] = (fid[f'/exchange/cshifts_final'][::step,:ndist]*args.n/2048).astype('float32')#/2**(nlevels-1)
    vars["psi"] = cl_rec.R(vars['u'])        
    vars["psi"][:] = cl_rec.expR(vars["psi"])        
    vars["table"] = pd.DataFrame(columns=["iter", "err", "time"])    

    vars = cl_rec.BH(data, vars)  

    
    


# In[14]:


# data = np.zeros([ntheta,ndist,n,n],dtype='float32')
# with h5py.File(f'{path_out}/{pfile}_corr.h5') as fid:
#     for k in range(ndist):
#         data[:,k] = fid[f'/exchange/data{k}'][::step].copy()                    


# In[15]:


# nlevels = 4
# # iters = np.array([1025,384,129*4])
# iters = np.array([17,33,9,4])
# vis_steps = [-1,8,8,8]
# err_steps = [4,8,8.8]
# chunks = [64,32,8,2]
# data_bin = np.sqrt(_downsample(data,nlevels-1))  
# for level in range(1):    
#     print(f'{level=}')    

#     args.n = n//2**(nlevels-level-1)
#     args.npsi = npsi//2**(nlevels-level-1)
#     args.nq = (n + args.pad)//2**(nlevels-level-1)    
#     args.voxelsize = voxelsize*2**(nlevels-level-1)    

#     args.nchunk = chunks[level]
#     args.niter=iters[level]
#     args.vis_step=vis_steps[level]
#     args.err_step=err_steps[level]
#     args.rotation_axis=(879)*n/1024/2**(nlevels-level-1)
    
#     for rr in [5,10,20,40]:
#         print(rr)
#         args.rho = [1,25,rr]
#         if level==0:
#             vars = {}                
#             with h5py.File(f'{path_out}/{pfile}_corr.h5') as fid:
#                 u_init = fid['/exchange/u_init_re'][:]+1j*fid['/exchange/u_init_imag'][:]                
#                 r_init = (fid[f'/exchange/cshifts_final'][::step]*args.n/2048).astype('float32')
#             cl_rec = Rec(args)    
#             q_init = np.ones([ndist,args.nq,args.nq],dtype='complex64')        
#             ref_bin = _downsample(ref,nlevels-1)    
#             for j in range(ndist):
#                 q_init[j] = cl_rec.DT(np.sqrt(ref_bin[j:j+1]),j)[0]
#             q = q_init.copy()
#             u = u_init.copy()        
#             r = r_init.copy()
#         else:
#             u = _fftupsample(0.5*vars['u'],[0])
#             u = _fftupsample(u,[1])
#             u = _fftupsample(u,[2])        
#             q = _fftupsample(vars['q'].get(),[1])
#             q = _fftupsample(q,[2])        
#             r = vars['r']*2
        
#         cl_rec = Rec(args)    
#         vars["u"] = u
#         vars["q"] = cp.array(q)
#         vars["r"] = r
#         with h5py.File(f'{path_out}/{pfile}_corr.h5') as fid:
#             vars["r_init"] = (fid[f'/exchange/cshifts_final'][::step]*args.n/2048).astype('float32')#/2**(nlevels-1)
#         vars["psi"] = cl_rec.R(vars['u'])        
#         vars["psi"][:] = cl_rec.expR(vars["psi"])      
#         vars["table"] = pd.DataFrame(columns=["iter", "err", "time"])    
#         vars = cl_rec.BH(data_bin, vars)  
#         err = vars["table"]["err"]
#         plt.plot(err,label=rr)
# plt.legend()
# plt.yscale('log')
# plt.show()
    
    


# In[ ]:




