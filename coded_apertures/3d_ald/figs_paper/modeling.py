#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

from holotomocupy.tomo import R
from holotomocupy.holo import G
from holotomocupy.magnification import M
from holotomocupy.shift import S
from holotomocupy.utils import *
import sys
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(10)
get_ipython().system('jupyter nbconvert --to script modeling.ipynb')


# 

# # Init data sizes and parametes of the PXM of ID16A

# In[2]:


n = 256  # object size in each dimension

ntheta = 180  # number of angles (rotations)
noise = 0
npos = 3

# npos = int(sys.argv[1])  # number of angles (rotations)
# noise = int(sys.argv[2])#sys.argv[2]=='True'
# z1p = float(sys.argv[3])  # positions of the code and the probe for reconstruction


center = n/2 # rotation axis
theta = cp.linspace(0, np.pi, ntheta,endpoint=False).astype('float32')  # projection angles

detector_pixelsize = 3e-6/2
energy = 33.35  # [keV] xray energy
wavelength = 1.2398419840550367e-09/energy  # [m] wave length

focusToDetectorDistance = 1.28  # [m]
sx0 = 3.7e-4
z1 = np.array([4.584e-3, 4.765e-3, 5.488e-3, 6.9895e-3])[:npos]-sx0
z2 = focusToDetectorDistance-z1
distances = (z1*z2)/focusToDetectorDistance
magnifications = focusToDetectorDistance/z1
voxelsize = detector_pixelsize/magnifications[0]*2048/n  # object voxel size
# scaled propagation distances due to magnified probes
# magnification when propagating from the probe plane to the detector
magnifications2 = z1/z1[0]
distances = distances/magnifications2**2
# propagation distances after switching from the point source wave to plane wave,
distances2 = (z1-z1[0])/magnifications2

# allow padding if there are shifts of the probe
pad = n//8
# sample size after demagnification
ne = n+2*pad
show = False
flg = f'{n}_{ntheta}_{npos}_{z1[0]:.2e}_{noise}_conv'
# allow padding if there are shifts of the probe
# sample size after demagnification
ne = int(np.ceil((n+2*pad)*magnifications2[-1]/8))*8  # make multiple of 8
print(distances+distances2)
print(distances2)


# ## Read real and imaginary parts of the refractive index u = delta+i beta

# In[3]:


from scipy import ndimage

# cube_all = np.zeros([n, n, n], dtype='float32')
# rr = (np.ones(8)*n*0.25).astype(np.int32)
# amps = [3, -3, 1, 3, -4, 1, 4]  # , -2, -4, 5 ]
# dil = np.array([33, 28, 25, 21, 16, 10, 3])/256*n  # , 6, 3,1]
# for kk in range(len(amps)):
#     cube = np.zeros([n, n, n], dtype='bool')
#     r = rr[kk]
#     p1 = n//2-r//2
#     p2 = n//2+r//2
#     for k in range(3):
#         cube = cube.swapaxes(0, k)
#         cube[p1:p2, p1, p1] = True
#         cube[p1:p2, p1, p2] = True
#         cube[p1:p2, p2, p1] = True
#         cube[p1:p2, p2, p2] = True
#         # cube[p1:p2,p2,p2] = True

#     [x, y, z] = np.meshgrid(np.arange(-n//2, n//2),
#                             np.arange(-n//2, n//2), np.arange(-n//2, n//2))
#     circ = (x**2+y**2+z**2) < dil[kk]**2
#     # circ = (x**2<dil[kk]**2)*(y**2<dil[kk]**2)*(z**2<dil[kk]**2)

#     fcirc = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(circ)))
#     fcube = np.fft.fftshift(np.fft.fftn(
#         np.fft.fftshift(cube.astype('float32'))))
#     cube = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(fcube*fcirc))).real
#     cube = cube > 1
#     cube_all += amps[kk]*cube

# # cube_all = ndimage.rotate(cube_all,52,axes=(1,2),reshape=False,order=1)
# cube_all = ndimage.rotate(cube_all, 28, axes=(0, 1), reshape=False, order=3)
# cube_all = ndimage.rotate(cube_all, 45, axes=(0, 2), reshape=False, order=3)
# cube_all[cube_all < 0] = 0


# u0 = cube_all  # (-1*cube_all*1e-6+1j*cube_all*1e-8)/3

# u0 = np.roll(u0, -15*n//256, axis=2)
# u0 = np.roll(u0, -10*n//256, axis=1)
# v = np.arange(-n//2, n//2)/n
# [vx, vy, vz] = np.meshgrid(v, v, v)
# v = np.exp(-10*(vx**2+vy**2+vz**2))
# fu = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(u0)))
# u0 = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(fu*v))).real
# u0[u0 < 0] = 0
# u0 = u0*(-1*1e-6+1j*1e-8)/2
# u0 = u0.astype('complex64')  
# !mkdir -p data
# np.save('data/u', u0)


u = np.load('data/uc.npy').astype('complex64')
u = np.pad(u,((ne//2-n//2,ne//2-n//2),(ne//2-n//2,ne//2-n//2),(ne//2-n//2,ne//2-n//2)))

mshow_complex(u[:, ne//2],show)
mshow_complex(u[ne//2],show)


# ## Compute tomographic projection data via the Fourier based method, $\mathcal{R}u$:

# In[4]:


print(center,u.shape)
Ru = R(u, theta, center*ne/n)
Ru = Ru.swapaxes(0, 1)
mshow_complex(Ru[0])


# ## Convert it to the transmittance function $e^{\frac{2\pi j}{\lambda} \mathcal{R} u }$

# In[5]:


psi = np.exp(2*np.pi*1j/wavelength*voxelsize*Ru)

mshow_polar(psi[0])


# ## Read a reference image previously recovered by the NFP (Near-field ptychogarphy) method at ID16A. 

# In[6]:


# !wget -nc https://g-110014.fd635.8443.data.globus.org/holotomocupy/examples_synthetic/data/prb_id16a/prb_abs_2048.tiff -P ../data/prb_id16a
# !wget -nc https://g-110014.fd635.8443.data.globus.org/holotomocupy/examples_synthetic/data/prb_id16a/prb_phase_2048.tiff -P ../data/prb_id16a

prb_abs = read_tiff(f'../data/prb_id16a/prb_abs_2048.tiff')[0:1]
prb_phase = read_tiff(f'../data/prb_id16a/prb_phase_2048.tiff')[0:1]
prb = prb_abs*np.exp(1j*prb_phase).astype('complex64')


for k in range(2):
    prb = prb[:, ::2]+prb[:, 1::2]
    prb = prb[:, :, ::2]+prb[:, :, 1::2]/4

prb = prb[:, 128-pad:-128+pad, 128-pad:-128+pad]
prb /= np.mean(np.abs(prb))
# prb[:]=1

mshow_polar(prb[0])


# # Smooth the probe, the loaded one is too noisy

# In[7]:


v = np.arange(-(n+2*pad)//2,(n+2*pad)//2)/(n+2*pad)
[vx,vy] = np.meshgrid(v,v)
v=np.exp(-20*(vx**2+vy**2))
prb = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(prb)))
prb = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(prb*v)))
prb = prb.astype('complex64')
# prb[:]=1
mshow_polar(prb[0])


# # Shifts/drifts

# ### Compute holographic projections for all angles and all distances
# #### $d=\left|\mathcal{G}_{z_j}((\mathcal{G}_{z'_j}S_{s'_{kj}}q)(M_j S_{s_{kj}}\psi_k))\right|_2^2$, and reference data $d^r=\left|\mathcal{G}_{z'_j}S_{s^r_{j}}q\right|$

# In[8]:


from holotomocupy.chunking import gpu_batch

@gpu_batch
def fwd_holo(psi, prb):
    # print(prb.shape)
    prb = cp.array(prb)
    
    data = cp.zeros([psi.shape[0],npos,n,n],dtype='complex64')
    for i in range(npos):        
        # ill shift for each acquisition
        prbr = cp.tile(prb,[psi.shape[0],1,1])
        
        # propagate illumination
        prbr = G(prbr, wavelength, voxelsize, distances2[i])
        # object shift for each acquisition
        psir = psi.copy()
        
        # scale object        
        if ne != n+2*pad:
            psir = M(psir, 1/magnifications2[i]*ne/(n+2*pad), n+2*pad)                
        
        # multiply the ill and object
        psir *= prbr

        # propagate both
        psir = G(psir, wavelength, voxelsize, distances[i])   
        data[:,i] = psir[:,pad:n+pad,pad:n+pad]
    return data

@gpu_batch
def _fwd_holo0(prb):    
    data = cp.zeros([1,npos, n, n], dtype='complex64')
    for j in range(npos):
        # propagate illumination 
        data[:,j] = G(prb, wavelength, voxelsize, distances[0])[:,pad:n+pad,pad:n+pad]        
    return data

def fwd_holo0(prb): 
    return _fwd_holo0(prb)

fpsi = fwd_holo(psi,prb)
fref = fwd_holo0(prb)


# ### Take squared absolute value to simulate data on the detector and a reference image

# In[9]:


data = np.abs(fpsi)**2
ref = np.abs(fref)**2


# In[10]:


mshow(data[0,0],show)

if noise>0:
    data_noise = np.random.poisson(data*noise).astype('float32')/noise
    ref_noise = np.random.poisson(ref*noise).astype('float32')/noise
    mshow(data_noise[0,0],show)
    mshow(data_noise[0,0]-data[0,0],show)
    data=data_noise
    ref=ref_noise
# mshow(ref[0,0])


# ### Visualize data

# In[11]:


for k in range(npos):
    mshow(data[0,k],show)


# ### Visualize reference images

# In[12]:


for k in range(npos):
    mshow(ref[0,k],show,vmax=3)


# ### Save data, reference images

# In[13]:


print(f'/data2/vnikitin/coded_apertures_new3/data/data_{k}_{flg}')
for k in range(len(distances)):
    write_tiff(data[:,k],f'/data2/vnikitin/coded_apertures_new3/data/data_{k}_{flg}')
for k in range(len(distances)):
    write_tiff(ref[:,k],f'/data2/vnikitin/coded_apertures_new3/data/ref_{k}_{flg}')
np.save(f'/data2/vnikitin/coded_apertures_new3/data/prb_{flg}', prb)


# In[14]:


# from matplotlib_scalebar.scalebar import ScaleBar
# for k in range(npos):
#     fig, ax = plt.subplots(figsize=(3,3))
#     ax.imshow(data[ntheta//2,k],cmap='gray',vmin=0.2,vmax=2.8)
#     scalebar = ScaleBar(voxelsize*magnifications[0], "m", length_fraction=0.25, font_properties={
#             "family": "serif",
#             "size": "large",
#         },  # For more information, see the cell below
#         location="lower right")
#     ax.add_artist(scalebar)
#     ax.tick_params(axis='both', which='major', labelsize=11)
#     # plt.show()
#     plt.savefig(f'fig/nocoded90deg{k}dist_prb.png',bbox_inches='tight',dpi=300)

#     from matplotlib_scalebar.scalebar import ScaleBar
#     fig, ax = plt.subplots(figsize=(3,3))
#     im = ax.imshow(data[0,k],cmap='gray',vmin=0.1,vmax=3)
#     scalebar = ScaleBar(voxelsize*magnifications[0], "m", length_fraction=0.25, font_properties={
#             "family": "serif",
#             "size": "large",
#         },  # For more information, see the cell below
#         location="lower right")
#     ax.add_artist(scalebar)
#     # ax.xticks(fontsize=14)
#     # ax.yticks(fontsize=14)
#     ax.tick_params(axis='both', which='major', labelsize=11)
#     # fig.colorbar(im, ax=ax, orientation='vertical')

#     # plt.show()
#     plt.savefig(f'fig/nocoded0deg{k}dist_prb.png',bbox_inches='tight',dpi=300)

# from matplotlib_scalebar.scalebar import ScaleBar
# fig, ax = plt.subplots(figsize=(3,3))
# im = ax.imshow(ref[0,k],cmap='gray',vmin=0.2,vmax=2.8)
# # fig.colorbar(im, ax=ax)
# scalebar = ScaleBar(voxelsize*magnifications[0], "m", length_fraction=0.25, font_properties={
#         "family": "serif",
#         "size": "large",
#     },  # For more information, see the cell below
#     location="lower right")
# ax.add_artist(scalebar)
# # ax.xticks(fontsize=14)
# # ax.yticks(fontsize=14)
# ax.tick_params(axis='both', which='major', labelsize=11)
# # fig.colorbar(im, ax=ax, orientation='vertical')

# # plt.show()
# plt.savefig(f'fig/nocoded_prb.png',bbox_inches='tight',dpi=300)






# In[15]:


# from matplotlib_scalebar.scalebar import ScaleBar
# fig, ax = plt.subplots(figsize=(3,3))
# im = ax.imshow(np.abs(prb[0]),cmap='gray',vmin=0.2,vmax=2.8)
# # fig.colorbar(im, ax=ax)
# scalebar = ScaleBar(voxelsize, "m", length_fraction=0.25, font_properties={
#         "family": "serif",
#         "size": "large",
#     },  # For more information, see the cell below
#     location="lower right")
# ax.add_artist(scalebar)
# # ax.xticks(fontsize=14)
# # ax.yticks(fontsize=14)
# ax.tick_params(axis='both', which='major', labelsize=11)
# cbar = fig.colorbar(im, ax=ax, orientation='vertical',fraction=0.046, pad=0.04)
# cbar.ax.tick_params(labelsize=12) 
# # plt.show()
# plt.savefig(f'fig/prb_abs.png',bbox_inches='tight',dpi=300)


# from matplotlib_scalebar.scalebar import ScaleBar
# fig, ax = plt.subplots(figsize=(3,3))
# im = ax.imshow(np.angle(prb[0]),cmap='gray',vmin=-1.6,vmax=1.6)
# # fig.colorbar(im, ax=ax)
# scalebar = ScaleBar(voxelsize, "m", length_fraction=0.25, font_properties={
#         "family": "serif",
#         "size": "large",
#     },  # For more information, see the cell below
#     location="lower right")
# ax.add_artist(scalebar)
# # ax.xticks(fontsize=14)
# # ax.yticks(fontsize=14)
# ax.tick_params(axis='both', which='major', labelsize=11)
# cbar = fig.colorbar(im, ax=ax, orientation='vertical',fraction=0.046, pad=0.04)
# cbar.ax.tick_params(labelsize=12) 

# # plt.show()
# plt.savefig(f'fig/prb_angle.png',bbox_inches='tight',dpi=300)


# In[ ]:





# In[16]:


# from matplotlib_scalebar.scalebar import ScaleBar
# for k in range(npos):
#     fig, ax = plt.subplots(figsize=(3,3))
#     ax.imshow(data[ntheta//2,k],cmap='gray',vmax=2.8,vmin=0.2)
#     scalebar = ScaleBar(voxelsize, "m", length_fraction=0.25, font_properties={
#             "family": "serif",
#             "size": "large",
#         },  # For more information, see the cell below
#         location="lower right")
#     ax.add_artist(scalebar)
#     ax.tick_params(axis='both', which='major', labelsize=11)
#     # plt.show()
#     plt.savefig(f'fig/nocoded90deg{k}dist.png',bbox_inches='tight',dpi=300)

#     from matplotlib_scalebar.scalebar import ScaleBar
#     fig, ax = plt.subplots(figsize=(3,3))
#     im = ax.imshow(data[0,k],cmap='gray',vmax=2.8,vmin=0.2)
#     scalebar = ScaleBar(voxelsize, "m", length_fraction=0.25, font_properties={
#             "family": "serif",
#             "size": "large",
#         },  # For more information, see the cell below
#         location="lower right")
#     ax.add_artist(scalebar)
#     # ax.xticks(fontsize=14)
#     # ax.yticks(fontsize=14)
#     ax.tick_params(axis='both', which='major', labelsize=11)
#     # fig.colorbar(im, ax=ax, orientation='vertical')

#     # plt.show()
#     plt.savefig(f'fig/nocoded0deg{k}dist.png',bbox_inches='tight',dpi=300)


