#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cupy as cp
import matplotlib.pyplot as plt
import pandas as pd
import sys
import h5py
from types import SimpleNamespace
import warnings
warnings.filterwarnings("ignore", message=f".*peer.*")

sys.path.insert(0, '..')
from utils import *
from rec import Rec


cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)


# ## Sizes and propagation settings

# In[2]:


npos = 324


# In[3]:


path = f'/data/vnikitin/ESRF/ID16A/20240924_rec_ca/data/'
with  h5py.File(f'{path}/data_ca.h5','r') as fid:
    data = fid[f'/exchange/pdata'][:npos].astype('float32')
    ref = fid[f'/exchange/pref'][:].astype('float32')
    shifts = fid[f'/exchange/shifts'][:npos].astype('float32')    
    psi_init = fid[f'/exchange/psi_init'][:]
    
    z1 = fid['/exchange/z1'][0]
    detector_pixelsize = fid['/exchange/detector_pixelsize'][0]
    focusdetectordistance = fid['/exchange/focusdetectordistance'][0]
    energy = fid['/exchange/energy'][0]


# In[4]:


wavelength = 1.24e-09/energy  
focusToDetectorDistance = 1.28  
z2 = focusToDetectorDistance-z1
distance = (z1*z2)/focusToDetectorDistance
magnification = focusToDetectorDistance/z1
voxelsize = np.abs(detector_pixelsize/magnification)  


# In[5]:


args = SimpleNamespace()

args.ngpus = 2
args.lam = 0.0

args.n = 2048
args.npsi = 8192
args.pad = 0
args.nq = args.n + 2 * args.pad
args.ex = 0
args.npatch = args.nq + 2 * args.ex
args.npos = npos
args.nchunk = 8

args.voxelsize = voxelsize
args.wavelength = wavelength
args.distance = distance
args.eps = 1e-8
args.rho = [1, 2, 0.1]
args.path_out = f"/data/vnikitin/ESRF/ID16A/20240924_rec_ca/rec_ca/r_{args.lam}_{args.pad}"

args.niter = 4096*2
args.err_step = 32
args.vis_step = 32
args.show = False

# create class
cl_rec = Rec(args)


# # init probe

# In[6]:


q_init = cp.array(cl_rec.DT(np.sqrt(ref[np.newaxis]))[0])

ppad = 3 * args.pad // 2
q_init = np.pad(
    q_init[ppad : args.nq - ppad, ppad : args.nq - ppad],
    ((ppad, ppad), (ppad, ppad)),
    "symmetric",
)
v = cp.ones(args.nq, dtype="float32")
vv = cp.sin(cp.linspace(0, cp.pi / 2, ppad))
v[:ppad] = vv
v[args.nq - ppad :] = vv[::-1]
v = cp.outer(v, v)
q_init = cp.abs(q_init * v) * cp.exp(1j * cp.angle(q_init) * v)

mshow_polar(q_init,args.show)


# In[ ]:


# variables
vars = {}
vars["psi"] = cp.array(psi_init)
vars["q"] = cp.array(q_init)
vars["ri"] = np.floor(shifts).astype("int32")
vars["r"] = np.array(shifts - vars["ri"]).astype("float32")
vars["table"] = pd.DataFrame(columns=["iter", "err", "time"])
# reconstruction
vars = cl_rec.BH(data, ref, vars)


# In[ ]:


# results
erra = vars["table"]["err"].values
plt.plot(erra)
plt.yscale("log")
plt.grid()
mshow_polar(vars["psi"],args.show)
mshow_polar(vars["q"],args.show)
pos_rec = vars["ri"] + vars["r"]
if args.show:
    plt.plot((shifts[:, 1] - pos_rec[:, 1]), ".", label="x difference")
    plt.plot((shifts[:, 0] - pos_rec[:, 0]), ".", label="y difference")
    plt.legend()
    plt.grid()
    plt.plot()

