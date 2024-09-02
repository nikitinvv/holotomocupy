#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cupy as cp

from holotomocupy.utils import *
from holotomocupy.holo import G
from holotomocupy.shift import S
from holotomocupy.tomo import R
from holotomocupy.chunking import gpu_batch
import sys
# get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(1) # fix randomness


# In[2]:


# !jupyter nbconvert --to script modeling_codes.ipynb


# # Init data sizes and parametes of the PXM of ID16A

# In[3]:


n = 256  # object size in each dimension

ntheta = 180  # number of angles (rotations)
noise = 1999
z1c = -12e-3
# thickness of the coded aperture
code_thickness = 1.5e-6 #in m
# feature size
ill_feature_size = 1e-6 #in m

ntheta = int(sys.argv[1])  # number of angles (rotations)
noise = int(sys.argv[2])#sys.argv[2]=='True'
# z1c = float(sys.argv[3])  # positions of the code and the probe for reconstruction

center = n/2 # rotation axis
theta = cp.linspace(0, np.pi, ntheta,endpoint=False).astype('float32')  # projection angles
npos = 1  # number of code positions
detector_pixelsize = 3e-6/2
energy = 33.35  # [keV] xray energy
wavelength = 1.2398419840550367e-09/energy  # [m] wave length
focusToDetectorDistance = 1.28  # [m]
sx0 = 3.7e-4
z1 = 4.584e-3-sx0# np.array([4.584e-3, 4.765e-3, 5.488e-3, 6.9895e-3])[:npos]-sx0
z1 = np.tile(z1, [npos])
z2 = focusToDetectorDistance-z1
distances = (z1*z2)/focusToDetectorDistance
magnifications = focusToDetectorDistance/z1
voxelsize = detector_pixelsize/magnifications[0]*2048/n  # object voxel size
# magnification when propagating from the probe plane to the detector
magnifications2 = z1/z1c
distances2 = (z1-z1c)/(z1c/z1)#magnifications2
# allow padding if there are shifts of the probe
pad = n//8
# sample size after demagnification
ne = n+2*pad

show = False

flg = f'{n}_{ntheta}_{npos}_{z1c}_{noise}_code'
# print(magnifications2,norm_magnifications)
# print(distances2,distances22)


# In[4]:


print(magnifications2)
# print(distances22)


# ### Form the refractive index u = delta+i beta for a sample

# In[5]:


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
# u = u0.astype('complex64')  

# !mkdir -p data
# np.save('data/u', u0)

# if exist then load and comment the above
u = np.load('data/u.npy').astype('complex64')
dxchange.write_tiff(-u.real,'/data/tmp/cube',overwrite=True)
print(u.shape)
u = np.pad(u,((pad,pad),(pad,pad),(pad,pad)))
# u[:]=0
mshow_complex(u[:, n//2],show)
mshow_complex(u[n//2],show)


# ### Compute tomographic projection data $\mathcal{R}u$:

# In[6]:


center_pad = u.shape[-1]//2
Ru = R(u, theta, center_pad)
Ru = Ru.swapaxes(0, 1)
mshow_complex(Ru[0],show)


# ### Convert it to the transmittance function $e^{\frac{2\pi j}{\lambda} \mathcal{R} u }$

# In[7]:


def exptomo(psi):
    """Exp representation of projections"""
    return np.exp(1j*psi * voxelsize * 2*cp.pi / wavelength)

psi = exptomo(Ru)
mshow_polar(psi[ntheta//4*3],show)


# ## Use prb==1 for this test

# In[8]:


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


# mshow_polar(prb[0])

v = np.arange(-(n+2*pad)//2,(n+2*pad)//2)/(n+2*pad)
[vx,vy] = np.meshgrid(v,v)
v=np.exp(-20*(vx**2+vy**2))
prb = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(prb)))
prb = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(prb*v)))
prb = prb.astype('complex64')

# prb[:]=1

mshow_polar(prb[0],show)
print(distances2)
prb = G(cp.array(prb),wavelength,voxelsize,-distances2[0])
mshow_polar(prb[0],show)


# ### Generate a coded aperture, make it twice bigger than the sample to allow motion

# In[9]:


import random
import xraylib
import scipy.ndimage as ndimage



random.seed(10)
nill = 2*ne
ill_global = np.zeros([1,nill,nill],dtype='bool')
for k in  range(ill_global.shape[0]):
    ill0 = np.zeros([nill*nill],dtype='bool')
    ill_ids = random.sample(range(0, nill*nill), int(nill*nill*0.55))
    ill0[ill_ids] = 1
    ill_global[k] = ill0.reshape(nill,nill)

# form codes for simulations
nill = int(ne*voxelsize/np.abs(magnifications2[0])//(ill_feature_size*2))*2
ill = np.zeros([1,nill,nill],dtype='bool')
for k in  range(ill.shape[0]):
    ill0 = ill_global[k]
    ill[k] = ill0[ill0.shape[0]//2-nill//2:ill0.shape[0]//2+(nill)//2,ill0.shape[1]//2-nill//2:ill0.shape[1]//2+(nill)//2]#.reshape(nill,nill)
    
ill = ndimage.zoom(ill,[1,2*ne/nill,2*ne/nill],order=0,grid_mode=True,mode='grid-wrap')

# print(np.sum(ill),2*ne*2*ne//2)

# fsize = int(2*ne//nill)+1
# x = np.linspace(-1,1,fsize)
# [x,y] = np.meshgrid(x,x)
# circ = x**2+y**2<0.9
# ill = ill*0
# while cp.sum(ill)<2*ne*2*ne//2:
#     stx = int(cp.random.randint(ill.shape[1]-fsize))
#     sty = int(cp.random.randint(ill.shape[1]-fsize))
#     # print(stx,sty,ill[:,sty:sty+fsize,stx:stx+fsize].shape)
#     ill[:,sty:sty+fsize,stx:stx+fsize]+=circ
#     if cp.amax(ill[:,sty:sty+fsize,stx:stx+fsize]>1):
#         ill[:,sty:sty+fsize,stx:stx+fsize]-=circ

#     stx = int(cp.random.randint(ill.shape[1]-fsize))
#     sty = int(cp.random.randint(ill.shape[1]-fsize))

# ill = ill*0 + np.random.random(ill.shape)*2

# ill_ids = np.array(ill_ids)
# idsy = ill_ids//(nill)
# idsx = ill_ids%(nill)
# ill[:]=0

# ill_ids = random.sample(range(0, 2*ne*2*ne), int(2*ne*2*ne/2))
# ill0 = np.zeros([2*ne*2*ne],dtype='bool')
# ill0[ill_ids] = 1
# ill[k] = ill0.reshape(2*ne,2*ne)

# for k in range(len(ill_ids)):
#     if idsy[k]*fsize+fsize<=ill.shape[1] and idsx[k]*fsize+fsize<=ill.shape[1]:
#         ill[:,idsy[k]*fsize:idsy[k]*fsize+fsize,idsx[k]*fsize:idsx[k]*fsize+fsize]=circ
# print(np.sum(ill),2*ne*2*ne//2)


# print(ill.shape)
# for k in range(ill.shape[1]//fsize):
#     for j in range(ill.shape[2]//fsize):
#         ill[0,k*fsize:(k+1)*fsize,j*fsize:(j+1)*fsize]*=circ

# print(nill)



delta = 1-xraylib.Refractive_Index_Re('Au',energy,19.3)
beta = xraylib.Refractive_Index_Im('Au',energy,19.3)
print(beta,delta,delta/beta)

# delta = 1-xraylib.Refractive_Index_Re('C21H25ClO',energy,1.19)
# beta = xraylib.Refractive_Index_Im('C21H25ClO',energy,1.19)

print(beta,delta,delta/beta)
thickness = code_thickness/voxelsize # thickness in pixels





# form Transmittance function

Rill = ill*(-delta+1j*beta)*thickness 
# Rill=ndimage.rotate(Rill, 45, axes=(2, 1), reshape=False, order=3, mode='reflect',
#                    prefilter=True)

v = np.arange(-2*ne//2,2*ne//2)/2/ne
[vx,vy] = np.meshgrid(v,v)
v = np.exp(-10*(vx**2+vy**2))
fill = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(Rill)))
Rill = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(fill*v)))
Rill = Rill.astype('complex64')

code = np.exp(1j * Rill * voxelsize * 2 * np.pi / wavelength).astype('complex64')

# code=code*0+1
mshow_polar(code[0],show)

mshow_polar(code[0,:n//4,:n//4],show)


# ### Shifts of the code

# In[10]:


# shifts of codes
shifts_code = np.round((np.random.random([ntheta, npos, 2]).astype('float32')-0.5)*ne/2)


# #### Forward holo: $d_k=\mathcal{P}_{z}\left(q\psi_k(\mathcal{P}_{z'}\mathcal{S}_{s_{k}}c)\right)$,

# In[11]:


@gpu_batch
def _fwd_holo(psi, shifts_code, code, prb):
    prb = cp.array(prb)
    code = cp.array(code)

    data = cp.zeros([psi.shape[0], npos, n, n], dtype='complex64')
    for i in range(npos):
        psir = psi.copy()
        prbr = cp.tile(prb, [psi.shape[0], 1, 1])
        coder = cp.tile(code, [psi.shape[0], 1, 1])
        
        # shift and crop thecode 
        coder = S(coder, shifts_code[:, i])
        coder = coder[:, ne-n//2-pad:ne+n//2+pad, ne-n//2-pad:ne+n//2+pad]
        coder *= prbr
        
        # propagate the code to the probe plane
        coder = G(coder, wavelength, voxelsize, distances2[i])
        
        # multiply the ill code and object
        psir *= (coder)                
        # propagate all to the detector
        psir = G(psir, wavelength, voxelsize, distances[i])
        # unpad
        data[:, i] = psir[:, pad:n+pad, pad:n+pad]
    return data

def fwd_holo(psi, prb):
    return _fwd_holo(psi, shifts_code, code, prb)

@gpu_batch
def _fwd_holo0(prb):    
    data = cp.zeros([1,npos, n, n], dtype='complex64')
    for j in range(npos):
        # propagate illumination 
        data[:,j] = G(prb, wavelength, voxelsize, distances[0]+distances2[0])[:,pad:n+pad,pad:n+pad]        
    return data

def fwd_holo0(prb): 
    return _fwd_holo0(prb)
# Apply the forward transform
psi = cp.array(psi)
prb = cp.array(prb)
code = cp.array(code)
shifts_code = cp.array(shifts_code)
fpsi = fwd_holo(psi, prb)
fref = fwd_holo0(prb)


# ### Take squared absolute value to simulate data on the detector and a reference image

# In[12]:


data = np.abs(fpsi)**2
ref = np.abs(fref)**2
# show a data example
# mshow(data[0,0],show,vmax=1.8,vmin=0.5)
# mshow(ref[0,0],show)
mshow(np.abs(fpsi[0,0]),show)
mshow(np.angle(fpsi[0,0]),show)
mshow(np.abs(ref[0,0]),show)
mshow(np.angle(ref[0,0]),show)
if noise>0:
    data_noise = cp.random.poisson(data*noise).astype('float32')/noise
    ref_noise = cp.random.poisson(ref*noise).astype('float32')/noise
    mshow(data[0,0]-data_noise[0,0])
    data=data_noise
    ref=ref_noise

mshow(data[0,0]/ref[0,0],show)


# ### Save data and the code

# In[13]:


print(f'/data2/vnikitin/coded_apertures_new3/data/data_{k}_{flg}.tiff')
for k in range(npos):
    write_tiff(data[:,k],f'/data2/vnikitin/coded_apertures_new3/data/data_{k}_{flg}')
for k in range(npos):
    write_tiff(ref[:,k],f'/data2/vnikitin/coded_apertures_new3/data/ref_{k}_{flg}')
np.save(f'/data2/vnikitin/coded_apertures_new3/data/code_{flg}',code)
np.save(f'/data2/vnikitin/coded_apertures_new3/data/shifts_code_{flg}', shifts_code)
np.save(f'/data2/vnikitin/coded_apertures_new3/data/prb_{flg}', prb)


# In[14]:


# from matplotlib_scalebar.scalebar import ScaleBar
# fig, ax = plt.subplots(figsize=(3,3))
# ax.imshow(data[ntheta//2,0],cmap='gray',vmin=0.2,vmax=2.8)
# scalebar = ScaleBar(voxelsize*magnifications[0], "m", length_fraction=0.25, font_properties={
#         "family": "serif",
#         "size": "large",
#     },  # For more information, see the cell below
#     location="lower right")
# ax.add_artist(scalebar)
# ax.tick_params(axis='both', which='major', labelsize=11)
# # plt.show()
# plt.savefig('fig/coded90deg_prb.png',bbox_inches='tight',dpi=300)

# from matplotlib_scalebar.scalebar import ScaleBar
# fig, ax = plt.subplots(figsize=(3,3))
# im = ax.imshow(data[0,0],cmap='gray',vmin=0.2,vmax=2.8)
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
# plt.savefig('fig/coded0deg_prb.png',bbox_inches='tight',dpi=300)


# In[15]:


# #for figure 1
# from matplotlib_scalebar.scalebar import ScaleBar
# fig, ax = plt.subplots(figsize=(3,3))
# ax.imshow(data[ntheta//2,0],cmap='gray',vmax=2.8,vmin=0.2)
# scalebar = ScaleBar(voxelsize, "m", length_fraction=0.25, font_properties={
#         "family": "serif",
#         "size": "large",
#     },  # For more information, see the cell below
#     location="lower right")
# ax.add_artist(scalebar)
# ax.tick_params(axis='both', which='major', labelsize=11)
# # plt.show()
# plt.savefig('fig/coded90deg.png',bbox_inches='tight',dpi=300)

# from matplotlib_scalebar.scalebar import ScaleBar
# fig, ax = plt.subplots(figsize=(3,3))
# im = ax.imshow(data[0,0],cmap='gray',vmax=2.8,vmin=0.2)
# scalebar = ScaleBar(voxelsize, "m", length_fraction=0.25, font_properties={
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
# plt.savefig('fig/coded0deg.png',bbox_inches='tight',dpi=300)

