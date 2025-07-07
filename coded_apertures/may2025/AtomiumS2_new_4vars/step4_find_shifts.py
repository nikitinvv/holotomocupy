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

import warnings
warnings.filterwarnings("ignore", message=f".*peer.*")

sys.path.insert(0, '..')
from utils import *
from rec import Rec


# In[2]:


ntheta = 7200
path = f'/data/vnikitin/ESRF/ID16A/20240924_rec_ca/data/'
with  h5py.File(f'{path}/data_atomium.h5','r') as fid:
    data = fid[f'/exchange/pdata'][::7200//ntheta].astype('float32')
    ref0 = fid[f'/exchange/pref0'][:].astype('float32')
    ref1 = fid[f'/exchange/pref1'][:].astype('float32')
    theta = fid[f'/exchange/theta'][::7200//ntheta].astype('float32')
    
    shifts = fid[f'/exchange/shifts'][::7200//ntheta].astype('float32')
    
    z1 = fid['/exchange/z1'][0]
    detector_pixelsize = fid['/exchange/detector_pixelsize'][0]
    focusdetectordistance = fid['/exchange/focusdetectordistance'][0]
    energy = fid['/exchange/energy'][0]


with  h5py.File(f'{path}/data_ca.h5','r') as fid:
    z1c = fid['/exchange/z1'][0]    


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


# In[ ]:


import dxchange
iter = int(sys.argv[1])
path_code = '/data/vnikitin/ESRF/ID16A/20240924_rec_ca/rec_ca/r_0.0_0/'
code_angle = dxchange.read_tiff(f'{path_code}/rec_psi_angle/{iter:04}.tiff')
code_abs = dxchange.read_tiff(f'{path_code}/rec_psi_abs/{iter:04}.tiff')
code = code_abs*np.exp(1j*code_angle)
mshow_polar(code,show)

q_angle = read_tiff(f'{path_code}/rec_prb_angle/{iter:04}.tiff')
q_abs = read_tiff(f'{path_code}/rec_prb_abs/{iter:04}.tiff')
q = q_abs*np.exp(1j*q_angle)
mshow_polar(q,show)


# In[5]:


args = SimpleNamespace()

args.ngpus = int(sys.argv[2])
args.n = 8192
args.ncode = 8192
args.npsi = 8192
args.pad = 0
args.nq = 8192
args.ex = 0
args.npatch = 8192
args.npos = 1
args.nchunk = 1
args.ntheta=1
args.theta=np.array([0])
args.rotation_axis=0

args.voxelsize = voxelsize
args.wavelength = wavelength
args.distance = distance
args.distancec = distancec

# doesnt matter
args.lam = 0
args.eps = 1e-8
args.rho = [1, 0.01, 0.1]
args.crop = 0
args.path_out = ""
args.niter = 2049
args.err_step = -1
args.vis_step = -1

args.show = True

# create class
cl_rec = Rec(args)


# In[ ]:


code =cp.array(code)
cdata = np.abs(cl_rec.D(cl_rec.Dc(code[np.newaxis]))[0])**2
mshow(cdata,show)
mshow(cdata[2000:4048,2000:4048],show)


# In[7]:


args.ngpus = int(sys.argv[2])
args.n = 2048
args.pad = 0#args.n // 8
args.npsi = args.n + 2 * args.pad
args.nq = args.n + 2 * args.pad
args.ex = 32
args.npatch = args.nq + 2 * args.ex
args.nchunk = 4
args.ntheta = len(theta)
args.theta = theta
cl_rec = Rec(args)


# In[8]:


ref = np.abs(cl_rec.D(cl_rec.Dc(q[np.newaxis]))[0])**2
rdata = data/(ref+1e-6)
rref0 = ref0/(ref+1e-6)
mshow(ref,show,vmax=3)
mshow(rdata[0],show,vmax=3)
mshow(rref0,show,vmax=3)


# In[9]:


def my_phase_corr(d1, d2):
    image_product = np.fft.fft2(d1) * np.fft.fft2(d2).conj()
    cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
    ind = np.unravel_index(np.argmax(cc_image.real, axis=None), cc_image.real.shape)
    shifts = cp.zeros(2,'float32')
    shifts[0] = ind[0]
    shifts[1] = ind[1]
    shifts -= d1.shape[-1]//2
    return shifts.get()

shifts_code = np.zeros([args.ntheta,2],dtype='float32')
a = cp.array(cdata)
nn = cdata.shape[-1]
rrdata=rdata.copy()
for k in range(rdata.shape[0]):        
    b = cp.pad(cp.array(rdata[k]),((nn//2-args.n//2,nn//2-args.n//2),(nn//2-args.n//2,nn//2-args.n//2)),'constant',constant_values=1)
    shift = -my_phase_corr(a,b)
    # mshow_complex(a+1j*b,show,vmax=2)
    shifts_code[k] = shift
    print(shift)
    aa = a[nn//2-shift[0]-args.n//2:nn//2-shift[0]+args.n//2,
           nn//2-shift[1]-args.n//2:nn//2-shift[1]+args.n//2]
    bb = cp.array(rdata[k])
    rrdata[k] = (bb/aa).get()
mshow_complex(bb+1j*aa,show,vmax=2)
mshow(rrdata[-1],show,vmin=0.5,vmax=1.5)
print(shifts_code)
np.save('shifts_code',shifts_code)


# In[10]:


shifts[:5]
shifts_cor = shifts.copy()
v=np.arange(-ntheta//2,ntheta//2)/(ntheta//2)
cc = 16*(v)**2
shifts_cor[:,0]+=cc
# plt.plot(cc)
# plt.show()


# In[11]:


def S(psi, p):
    """Apply shift for all projections."""
    res=psi.copy()
    for k in range(p.shape[0]):
        psi0 = cp.array(psi[k:k+1])
        p0 = cp.array(p[k:k+1])
        tmp = cp.pad(psi0,((0,0),(args.n//2,args.n//2),(args.n//2,args.n//2)), 'symmetric')
        [x, y] = cp.meshgrid(cp.fft.rfftfreq(2*args.n),
                            cp.fft.fftfreq(2*args.n))
        shift = cp.exp(-2*cp.pi*1j *
                    (x*p0[:, 1, None, None]+y*p0[:, 0, None, None]))
        res0 = cp.fft.irfft2(shift*cp.fft.rfft2(tmp))
        res[k] = res0[:, args.n//2:3*args.n//2, args.n//2:3*args.n//2].get()
    return res

srrdata = S(rrdata,-shifts_cor)
# dxchange.write_tiff_stack(srrdata,'/data/tmp/test_shift/t',overwrite=True)


# In[12]:


def multiPaganin(data, distances, wavelength, voxelsize, delta_beta,  alpha):    
    
    fx = cp.fft.fftfreq(data.shape[-1], d=voxelsize).astype('float32')
    [fx, fy] = cp.meshgrid(fx, fx)
    numerator = 0
    denominator = 0
    rad_freq = cp.fft.fft2(data)
    taylorExp = 1 + wavelength * distances * cp.pi * (delta_beta) * (fx**2+fy**2)
    numerator = numerator + taylorExp * (rad_freq)
    denominator = denominator + taylorExp**2

    denominator = (denominator) + alpha

    phase = cp.log(cp.real(cp.fft.ifft2(numerator / denominator)))
    phase = (delta_beta) * 0.5 * phase

    return phase

def CTFPurePhase(data, distances, wavelength, voxelsize, alpha):   

    fx = cp.fft.fftfreq(data.shape[-1], d=voxelsize).astype('float32')
    [fx, fy] = cp.meshgrid(fx, fx)
    numerator = 0
    denominator = 0
    rad_freq = cp.fft.fft2(data)
    taylorExp = cp.sin(cp.pi*wavelength*distances*(fx**2+fy**2))
    numerator = numerator + taylorExp * (rad_freq)
    denominator = denominator + 2*taylorExp**2
    denominator = (denominator) + alpha
    phase = cp.real(cp.fft.ifft2(numerator / denominator))
    phase = 0.5 * phase
    return phase

def rec_init(rdata):
    recMultiPaganin = np.zeros([args.ntheta,args.nq, args.nq], dtype="float32")
    for j in range(0, rdata.shape[0]):
        print(j)
        r = cp.array(rdata[j])
        distances_pag = (distance)
        r = multiPaganin(r, distances_pag,wavelength, voxelsize,1,1e-3)             
        recMultiPaganin[j] = r.get()           
        # recMultiPaganin[j]-=np.mean(recMultiPaganin[j,:32,:32])
    recMultiPaganin = np.exp(1j * recMultiPaganin)
    return recMultiPaganin

# psi_init = rec_init(srrdata[:])
# mpad = args.npsi//2-args.nq//2
# psi_init = np.pad(psi_init,((0,0),(mpad,mpad),(mpad,mpad)),'edge')
# mshow_polar(psi_init[0],args.show)
# mshow_polar(psi_init[1],args.show)

# dxchange.write_tiff_stack(np.angle(psi_init),'/data/tmp/test_shift/st',overwrite=True)


# In[13]:


ri = np.round(shifts_code).astype('int32')
r = shifts_code-ri
cdata = np.abs(cl_rec.D(cl_rec.Dc(cl_rec.S(ri,r,code)*q)))**2


# In[14]:


code=code.get()


# In[15]:


path = f'/data/vnikitin/ESRF/ID16A/20240924_rec_ca/data/'
with  h5py.File(f'{path}/data_atomium.h5','a') as fid:
    # try:
    #     # del fid['/exchange/cdata']
    #     # del fid['/exchange/ref']
    #     del fid['/exchange/prb']
    #     del fid['/exchange/code']
    #     del fid['/exchange/shifts_cor']
    #     del fid['/exchange/shifts_code']
    #     # del fid['/exchange/psi_init']
    # except:
    #     pass
    fid.create_dataset(f'/exchange/cdata{iter}',data=cdata)
    # fid.create_dataset('/exchange/ref',data=ref)
    fid.create_dataset(f'/exchange/prb{iter}',data=q)
    fid.create_dataset(f'/exchange/code{iter}',data=code)
    fid.create_dataset(f'/exchange/shifts_cor{iter}',data=shifts_cor)
    fid.create_dataset(f'/exchange/shifts_code{iter}',data=shifts_code)    
    # fid.create_dataset('/exchange/psi_init',data=psi_init)    

