#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cupy as cp
import h5py
import matplotlib.pyplot as plt
import cupyx.scipy.ndimage as ndimage
from types import SimpleNamespace

# Use managed memory
import h5py
import sys
import warnings
warnings.filterwarnings("ignore", message=f".*peer.*")

sys.path.insert(0, '..')
from utils import *
from rec import Rec
ngpus = cp.cuda.runtime.getDeviceCount()


# 

# # Init data sizes and parametes of the PXM of ID16A

# In[2]:


step = 4
bin = 3
ndist = 4
paganin = int(sys.argv[1])



# In[3]:


pfile = f'Y350c_HT_015nm'
path_out = f'/data/vnikitin/ESRF/ID16A/brain_rec/20240515/Y350c'
with h5py.File(f'{path_out}/{pfile}.h5') as fid:
    detector_pixelsize = fid['/exchange/detector_pixelsize'][0]    
    focusToDetectorDistance = fid['/exchange/focusdetectordistance'][0]    
    z1 = fid['/exchange/z1'][:ndist]        
    theta = fid['/exchange/theta'][::step,0]
    shifts = fid['/exchange/shifts'][::step,:ndist]
    attrs = fid['/exchange/attrs'][::step,:ndist]
    pos_shifts = fid['/exchange/pos_shifts'][::step,:ndist]*1e-6
    shape = fid['/exchange/data0'][::step].shape
    shape_ref = fid['/exchange/data_white_start0'].shape
    shape_dark = fid['/exchange/data_dark0'].shape
    #pos_shifts-=pos_shifts[0]


# In[4]:


theta = theta/180*np.pi


# In[5]:


ntheta,n = shape[:2]
ndark = shape_dark[0]
nref = shape_ref[0]

n//=2**bin


# In[6]:


print(ndist,ntheta,n)
print(nref,ndark)


# In[7]:


energy = 33.35  # [keV] xray energy
wavelength = 1.2398419840550367e-09/energy  # [m] wave length
z2 = focusToDetectorDistance-z1
magnifications = focusToDetectorDistance/z1
norm_magnifications = magnifications/magnifications[0]
distances = (z1*z2)/focusToDetectorDistance*norm_magnifications**2
voxelsize = detector_pixelsize/magnifications[0]*2048/n  # object voxel size
show = False


# In[8]:


pad = 0
npsi = int(np.ceil((2048+2*pad)/norm_magnifications[-1]/16))*16  # make multiple of 8
# npsi+=64
rotation_axis=npsi//2+145.147#(879-(1616-npsi//2)//2+2.5)*n/1024#n/2#(796.25+2)*n/1024#397.5*2#499.75*n//1024+npsi//2-n//2

print(rotation_axis)
npsi//=(2048//n)
rotation_axis/=(2048//n)

print(npsi)
# In[9]:


args = SimpleNamespace()
args.ngpus = ngpus

args.n = n
args.ndist = ndist
args.ntheta = ntheta
args.pad = pad
args.npsi = npsi
args.nq = n + 2 * pad
args.nchunk = 64
args.lam = 0

args.voxelsize = voxelsize
args.wavelength = wavelength
args.distance = distances
args.eps = 1e-12
args.rho = [1, 2, 1]
args.path_out = f"{path_out}/s1"
args.show = False

args.niter=10000
args.vis_step=1
args.err_step=1
args.rotation_axis =rotation_axis
print(args.rotation_axis)
args.theta = theta
args.norm_magnifications = norm_magnifications
print(norm_magnifications)
# create class
cl_rec = Rec(args)

# sss
# In[11]:


data = np.zeros([ntheta,ndist,n,n],dtype='float32')
with h5py.File(f'{path_out}/{pfile}_corr.h5') as fid:
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
    r = fid[f'/exchange/cshifts_final'][::step,:ndist]*n/2048#/norm_magnifications[:,np.newaxis]# in init coordinates! not scaled


# In[12]:


rdata = data/ref
srdata = np.zeros([ntheta,ndist,args.npsi,args.npsi],dtype='float32')
distances_pag = (distances/norm_magnifications**2)
npad=n//16
for j in np.arange(ndist)[::-1]:
    print(j)
    tmp = cl_rec.STa(r[:,j]*norm_magnifications[j],rdata[:,j].astype('complex64'),
                     'edge')    
    # tmp=cp.array(tmp)
    tmp = (cl_rec.MT(tmp,j)/norm_magnifications[j]**2).real    
    print(norm_magnifications)
    

    st = np.where(tmp[0]>1e-1)[0][0]+8
    
    if j==ndist-1:
         tmp = np.pad(tmp[:,st:-st,st:-st],((0,0),(st,st),(st,st)),'symmetric')
    if j<ndist-1:
        w = np.ones([args.npsi],dtype='float32')  
        v = np.linspace(0, 1, npad, endpoint=False)
        v = v**5*(126-420*v+540*v**2-315*v**3+70*v**4)             
        w[:st]=0
        w[st:st+npad] = v
        w[-st-npad:-st] = 1-v
        w[-st:]=0
        w=np.outer(w,w)
        #mshow(w,True)
        tmp=tmp*(w)+srdata[:,j+1]*(1-w)       
    srdata[:,j]=tmp
    mshow(srdata[0,j],args.show)
    


# In[13]:


mshow(srdata[0,0],args.show)
mshow(srdata[0,ndist-1],args.show)
mshow(srdata[0,0]-srdata[0,2],args.show,vmax=0.2,vmin=-0.2)


# In[14]:


def multiPaganin(data, distances, wavelength, voxelsize, delta_beta,  alpha):    
    
    fx = cp.fft.fftfreq(data.shape[-1], d=voxelsize).astype('float32')
    [fx, fy] = cp.meshgrid(fx, fx)
    numerator = 0
    denominator = 0
    for j in range(data.shape[0]):        
        rad_freq = cp.fft.fft2(data[j])
        taylorExp = 1 + wavelength * distances[j] * cp.pi * (delta_beta) * (fx**2+fy**2)
        numerator = numerator + taylorExp * (rad_freq)
        denominator = denominator + taylorExp**2

    numerator = numerator / len(distances)
    denominator = (denominator / len(distances)) + alpha

    phase = cp.log(cp.real(cp.fft.ifft2(numerator / denominator)))
    phase = (delta_beta) * 0.5 * phase

    return phase

def CTFPurePhase(data, distances, wavelength, voxelsize, alpha):   

    fx = cp.fft.fftfreq(data.shape[-1], d=voxelsize).astype('float32')
    [fx, fy] = cp.meshgrid(fx, fx)
    numerator = 0
    denominator = 0
    for j in range(data.shape[0]):
        rad_freq = cp.fft.fft2(data[j])
        taylorExp = cp.sin(cp.pi*wavelength*distances[j]*(fx**2+fy**2))
        numerator = numerator + taylorExp * (rad_freq)
        denominator = denominator + 2*taylorExp**2
    numerator = numerator / len(distances)
    denominator = (denominator / len(distances)) + alpha
    phase = cp.real(cp.fft.ifft2(numerator / denominator))
    phase = 0.5 * phase
    return phase

def rec_init(rdata):
    recMultiPaganin = np.zeros([args.ntheta,args.npsi, args.npsi], dtype="float32")
    for j in range(0, args.ntheta):
        r = cp.array(rdata[j])
        distances_pag = (distances/norm_magnifications**2)
        # print(distances_pag,wavelength,voxelsize)
        r = multiPaganin(r, distances_pag,wavelength, voxelsize,paganin, 1e-3)            
        # r = CTFPurePhase(r, distances_pag,wavelength, voxelsize, 1e-3)             
        # r[r>0]=0
        recMultiPaganin[j] = r.get()           
        
    recMultiPaganin-=np.mean(recMultiPaganin[:,:,:8])
    recMultiPaganin[recMultiPaganin>0]=0
    recMultiPaganin = np.exp(1j * recMultiPaganin)
    return recMultiPaganin

for pp in [110]:
    print(f"{rotation_axis=},{npsi}",flush=True)
    paganin = pp
    psi_init = rec_init(srdata)
    #mshow_complex(psi_init[0],True)
    psi_data = np.log(psi_init)/1j


    cl_rec = Rec(args)
    cl_rec.theta = np.ascontiguousarray(theta)
    psi_data = np.ascontiguousarray(psi_data)
    u_init = cl_rec.rec_tomo(psi_data,64)


    with h5py.File(f'{path_out}/{pfile}_corr.h5','a') as fid:
        try:
            del fid[f'/exchange/u_init_re{paganin}']
            del fid[f'/exchange/u_init_imag{paganin}']        
        except:
            pass
        fid.create_dataset(f'/exchange/u_init_re{paganin}',data = u_init.real)
        fid.create_dataset(f'/exchange/u_init_imag{paganin}',data = u_init.imag)    


