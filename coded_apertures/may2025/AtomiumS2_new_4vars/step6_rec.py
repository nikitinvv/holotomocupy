#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cupy as cp
import sys
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from types import SimpleNamespace
import h5py
import dxchange
import warnings
import pandas as pd
warnings.filterwarnings("ignore", message=f".*peer.*")

sys.path.insert(0, '..')
from utils import *
from rec import Rec


# In[2]:


bin = int(sys.argv[1])
ntheta = int(sys.argv[2])
ngpus = int(sys.argv[3])
path = f'/data/vnikitin/ESRF/ID16A/20240924_rec_ca/data/'
ind = np.arange(0,7200,7200/ntheta).astype('int32')[:ntheta]
with  h5py.File(f'{path}/data_atomium.h5','r') as fid:
    data = np.sqrt(fid[f'/exchange/pdata'][ind].astype('float32'))
    print(data.shape)
    # u_init = fid[f'/exchange/u_init'][:].astype('complex64')
    ref0 = fid[f'/exchange/pref0'][:].astype('float32')
    # ref1 = fid[f'/exchange/pref1'][:].astype('float32')
    theta = fid[f'/exchange/theta'][ind].astype('float32')
    code = fid[f'/exchange/code'][:].astype('complex64')
    q = fid[f'/exchange/prb'][:].astype('complex64')
    
    shifts_cor = fid[f'/exchange/shifts_cor'][ind].astype('float32')
    shifts_code = fid[f'/exchange/shifts_code'][ind].astype('float32')
    # cdata = fid['/exchange/cdata'][ind]
    
    z1 = fid['/exchange/z1'][0]
    detector_pixelsize = fid['/exchange/detector_pixelsize'][0]
    focusdetectordistance = fid['/exchange/focusdetectordistance'][0]
    energy = fid['/exchange/energy'][0]
with  h5py.File(f'{path}/data_ca.h5','r') as fid:
    z1c = fid['/exchange/z1'][0] 

for k in range(bin):
    data = 0.5*(data[:,::2]+data[:,1::2])
    data = 0.5*(data[:,:,::2]+data[:,:,1::2])
    # u_init = 0.5*(u_init[::2]+u_init[1::2])
    # u_init = 0.5*(u_init[:,::2]+u_init[:,1::2])
    # u_init = 0.5*(u_init[:,:,::2]+u_init[:,:,1::2])    
    ref0 = 0.5*(ref0[::2]+ref0[1::2])
    ref0 = 0.5*(ref0[:,::2]+ref0[:,1::2])
    # cdata = 0.5*(cdata[::2]+cdata[1::2])
    # cdata = 0.5*(cdata[:,::2]+cdata[:,1::2])
    q = 0.5*(q[::2]+q[1::2])
    q = 0.5*(q[:,::2]+q[:,1::2])
    code = 0.5*(code[::2]+code[1::2])
    code = 0.5*(code[:,::2]+code[:,1::2])
shifts_cor/=2**bin
shifts_code/=2**bin


# In[3]:


wavelength = 1.24e-09/energy  # [m] wave length
focusToDetectorDistance = 1.28  # [m]
z2 = focusToDetectorDistance-z1
distance = (z1*z2)/focusToDetectorDistance
magnification = focusToDetectorDistance/z1
voxelsize = np.abs(detector_pixelsize/magnification)  # object voxel size
magnifications2 = z1/z1c
distancec = (z1-z1c)/(z1c/z1)
show = False


# In[4]:


args = SimpleNamespace()
args.ngpus = ngpus

args.n = 2048//2**bin
voxelsize *= 2048/args.n

args.ntheta = ntheta
args.ncode = 8192*args.n//2048
args.pad = 0
args.npsi = args.n + 2 * args.pad
args.nq = args.n + 2 * args.pad
args.ex = 8
args.npatch = args.nq + 2 * args.ex
args.nchunk = 4

args.voxelsize = voxelsize
args.wavelength = wavelength
args.distance = distance
args.distancec = distancec

args.rotation_axis=756*args.n/1024-args.n//4+args.pad
args.theta = theta*np.pi/180
# create class
cl_rec = Rec(args)


# In[ ]:


vars = {}
vars["code"] = code
vars["u"] = np.zeros([args.npsi,args.npsi,args.npsi],dtype='complex64')
vars["q"] = cp.array(q)
vars["ri"] = shifts_code.astype("int32")
vars["r_init"] = shifts_code - vars["ri"].astype("int32")
vars["r"] = vars["r_init"].copy()
vars["rpsi_init"] = shifts_cor.astype("float32")
vars["rpsi"] = vars["rpsi_init"].copy()
# vars["Ru"] = cl_rec.Spsi(cl_rec.R(vars['u']),vars["rpsi"])
vars["psi"] = cl_rec.expR(cl_rec.Spsi(cl_rec.R(vars['u']),vars["rpsi"]))
vars["table"] = pd.DataFrame(columns=["iter", "err", "time"])

cl_rec.rho = [1,30,10,int(sys.argv[4])]
cl_rec.lam = float(sys.argv[5])
cl_rec.vis_step=32
cl_rec.err_step=4
cl_rec.eps=0
cl_rec.niter=1025*10
cl_rec.eps = 0
cl_rec.path_out = f"/data/vnikitin/ESRF/ID16A/20240924_rec_ca/rec_atomium_4vars/r_{cl_rec.n}_{cl_rec.ntheta}_{cl_rec.rho[3]}_{cl_rec.lam}"
cl_rec.show = show
vars = cl_rec.BH(data, vars)
err = vars["table"]["err"]
# plt.plot(err,label=rr)

