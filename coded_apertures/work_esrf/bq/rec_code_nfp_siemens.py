#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cupy as cp
import h5py
from holotomocupy.holo import G, GT
from holotomocupy.shift import S, ST
from holotomocupy.recon_methods import multiPaganin
from holotomocupy.utils import *
from holotomocupy.proc import remove_outliers
import sys
# Use managed memory
# cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)


# # Init data sizes and parametes of the PXM of ID16A

# In[2]:


n = 2048  # object size in each dimension
pad = n//8
npos= int(sys.argv[1])
z1c = -17.75e-3

detector_pixelsize = 3.03751e-6
energy = 33.35  # [keV] xray energy
wavelength = 1.2398419840550367e-09/energy  # [m] wave length
focusToDetectorDistance = 1.28  # [m]
sx0 = 1.286e-3
z1 = np.tile(5.5e-3-sx0, [npos])
z2 = focusToDetectorDistance-z1
distances = (z1*z2)/focusToDetectorDistance
magnifications = focusToDetectorDistance/z1
voxelsize = np.abs(detector_pixelsize/magnifications[0]*2048/n)  # object voxel size

# magnification when propagating from the probe plane to the detector
magnifications2 = z1/z1c
distances2 = (z1-z1c)/(z1c/z1)#magnifications2

# sample size after demagnification
ne = 2048+2*pad
show = False

rho = 0.1
flg = f'{n}_{z1c}_{rho}_{npos}'
path = f'/data2/vnikitin/nfp_codes_siemens'
print(f'{voxelsize=}')


# ## Read data

# In[3]:


idsx = np.arange(4-np.ceil(np.sqrt(npos)/2),4+np.int32(np.sqrt(npos)/2))
idsy = np.arange(4-np.ceil(np.sqrt(npos)/2),4+np.int32(np.sqrt(npos)/2))
[idsx,idsy] = np.meshgrid(idsx,idsy)
ids = (idsy*9+idsx).flatten().astype('int32')
print(ids)
with h5py.File('/data/vnikitin/ESRF/ID16A/20240924/SiemensLH/SiemensLH_010nm_code2um_nfp9x9_01/SiemensLH_010nm_code2um_nfp9x9_010000.h5') as fid:
    data0 = fid['/entry_0000/measurement/data'][ids].astype('float32')
    
with h5py.File('/data/vnikitin/ESRF/ID16A/20240924/SiemensLH/SiemensLH_010nm_nfp_02/ref_0000.h5') as fid:
    ref0 = fid['/entry_0000/measurement/data'][:].astype('float32')
with h5py.File('/data/vnikitin/ESRF/ID16A/20240924/SiemensLH/SiemensLH_010nm_code2um_nfp9x9_01/dark_0000.h5') as fid:
    dark0 = fid['/entry_0000/measurement/data'][:].astype('float32')

data0 = data0[np.newaxis]


# In[4]:


data = data0.copy()
ref = ref0.copy()
dark = dark0.copy()
for k in range(npos):
    radius = 7
    threshold = 20000
    data[:,k] = remove_outliers(data[:,k], radius, threshold)
ref[:] = remove_outliers(ref[:], radius, threshold)     
dark[:] = remove_outliers(dark[:], radius, threshold)     

data/=np.mean(ref)
dark/=np.mean(ref)
ref/=np.mean(ref)

rdata = (data-np.mean(dark,axis=0))/(np.mean(ref,axis=0)-np.mean(dark,axis=0))

mshow_complex(data[0,0]+1j*rdata[0,0],show,vmax=2)
mshow_complex(ref[0]+1j*dark[0],show)


# ### load reconstructed code

# In[5]:


iter = 72
code_angle = dxchange.read_tiff(f'/data2/vnikitin/nfp_codes/crec_code_angle2048_-0.01775full0.25/{iter:03}.tiff')
code_abs = dxchange.read_tiff(f'/data2/vnikitin/nfp_codes/crec_code_abs2048_-0.01775full0.25/{iter:03}.tiff')
code = code_abs*np.exp(1j*code_angle)[np.newaxis]
mshow_polar(code[0],show)
prb_angle = dxchange.read_tiff(f'/data2/vnikitin/nfp_codes/crec_prb_angle2048_-0.01775full0.25/{iter:03}.tiff')
prb_abs = dxchange.read_tiff(f'/data2/vnikitin/nfp_codes/crec_prb_abs2048_-0.01775full0.25/{iter:03}.tiff')
prb = prb_abs*np.exp(1j*prb_angle)[np.newaxis]
mshow_polar(prb[0],show)

z1_code = cp.array([z1c])
z2_code = focusToDetectorDistance-z1_code
distances_code = (z1_code*z2_code)/focusToDetectorDistance
magnifications_code = focusToDetectorDistance/z1_code
voxelsize_code = np.abs(detector_pixelsize/magnifications_code[0]*2048/n)  # object voxel size
code_data = (np.abs(G(cp.array(code),wavelength,voxelsize_code,distances_code))**2).get()
code_data0 = code_data[0,code_data.shape[1]//2-n//2:code_data.shape[1]//2+n//2,code_data.shape[1]//2-n//2:code_data.shape[1]//2+n//2]

prb_data = (np.abs(G(cp.array(prb),wavelength,voxelsize_code,distances_code))**2).get()
prb_data0 = prb_data[0,prb_data.shape[1]//2-n//2:prb_data.shape[1]//2+n//2,prb_data.shape[1]//2-n//2:prb_data.shape[1]//2+n//2]
mshow_complex(code_data0+1j*rdata[0,0],show,vmax=1.8,vmin=0.6)
mshow_complex(code_data0[n//2-256:n//2+256,n//2-256:n//2+256]+1j*rdata[0,0,n//2-256:n//2+256,n//2-256:n//2+256],show,vmax=1.8,vmin=0.6)
mshow_complex(prb_data0+1j*ref[0],show)
mshow(prb_data0-ref[0],show)
mshow(prb_data0-ref[0],show,vmin=-0.05,vmax=0.05)



# ### find position of an image in another image

# In[6]:


def my_phase_corr(d1, d2):
    image_product = np.fft.fft2(d1) * np.fft.fft2(d2).conj()
    cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
    ind = np.unravel_index(np.argmax(cc_image.real, axis=None), cc_image.real.shape)
    shifts = cp.zeros(2,'float32')
    shifts[0] = ind[0]
    shifts[1] = ind[1]
    shifts -= d1.shape[-1]//2
    return shifts.get()

shifts_code = np.zeros([1,npos,2],dtype='float32')
a = cp.array(code_data[0])
nn = code_data.shape[-1]
rrdata=rdata.copy()
for k in range(rdata.shape[1]):        
    b = cp.pad(cp.array(rdata[0,k]),((nn//2-n//2,nn//2-n//2),(nn//2-n//2,nn//2-n//2)))
    shift = -my_phase_corr(a,b)
    shifts_code[0,k] = shift
    aa = a[nn//2-shift[0]-n//2:nn//2-shift[0]+n//2,nn//2-shift[1]-n//2:nn//2-shift[1]+n//2]
    bb = cp.array(rdata[0,k])
    rrdata[0,k] = (bb/aa).get()
mshow(rrdata[0,0],show,vmin=0.5,vmax=1.5)
print(shifts_code)


# In[7]:


print(voxelsize,distances)
def rec_init(rdata):
    recMultiPaganin = cp.zeros([1,npos,ne,ne],dtype='float32')
    recMultiPaganin = multiPaganin(cp.array(rdata),distances, wavelength, voxelsize,  24.05, 1e-3)
    recMultiPaganin = cp.pad(recMultiPaganin,((0,0), (ne//2-n//2,ne//2-n//2), (ne//2-n//2,ne//2-n//2)),'constant')   
    recMultiPaganin = cp.exp(1j*recMultiPaganin)
    return recMultiPaganin.get()

# a = np.sum(rrdata[0],axis=1)
# mshow(np.sum(rrdata,axis=1)[0],show,vmax=2,vmin=-1)
rec_paganin = rec_init(rrdata)
mshow_polar(rec_paganin[0],show)


# ### $$I({{x}},\Psi)=\left\||L_2\Big(L_1\big(J(q)\cdot S_{{x}}(c)\big)\cdot \Psi\Big)|-d\right\|^2,$$ where $\Psi$ is a datacube representing a collection of objects $(\psi_1,\ldots,\psi_K)$. Typically, one has $\Psi=J(\psi)$ so that the object is the same for every shot

# # Construct operators
# 

# In[8]:


def L2op(psi):
    data = np.zeros([psi.shape[0], npos, n, n], dtype='complex64')
    for i in range(npos):
        psir = cp.array(psi[:,i])       
        psir = G(psir, wavelength, voxelsize, distances[i],'symmetric')
        data[:, i] = psir[:, pad:n+pad, pad:n+pad].get()
    return data

def LT2op(data):
    psi = np.zeros([data.shape[0],npos, n+2*pad, n+2*pad], dtype='complex64')
    for j in range(npos):
        datar = cp.array(data[:, j])
        psir = cp.pad(datar, ((0, 0), (pad, pad), (pad, pad))).astype('complex64')
        psir = GT(psir, wavelength, voxelsize, distances[j],'symmetric')        
        psi[:,j] = psir.get()
    return psi

def L1op(psi):
    data = np.zeros([psi.shape[0], npos, n+2*pad, n+2*pad], dtype='complex64')
    for i in range(npos):
        psir = cp.array(psi[:,i])       
        psir = G(psir, wavelength, voxelsize, distances2[i],'symmetric')
        data[:, i] = psir.get()
    return data

def LT1op(data):
    psi = np.zeros([data.shape[0],npos, n+2*pad, n+2*pad], dtype='complex64')
    for j in range(npos):
        datar = cp.array(data[:, j])
        psir = datar
        psir = GT(psir, wavelength, voxelsize, distances2[j],'symmetric')        
        psi[:,j] = psir.get()
    return psi


def Sop(psi,shifts):
    data = np.zeros([psi.shape[0], npos, n+2*pad, n+2*pad], dtype='complex64')
    psi = cp.array(psi)
    for i in range(npos):
        psir = psi.copy()
        shiftsr = cp.array(shifts[:, i])
        psir = S(psir, shiftsr)
        nee = psir.shape[1]        
        data[:,i] = psir[:, nee//2-n//2-pad:nee//2+n//2+pad, nee//2-n//2-pad:nee//2+n//2+pad].get()
    return data

def STop(data,shifts):
    psi = cp.zeros([data.shape[0], ne, ne], dtype='complex64')

    for j in range(npos):
        datar = cp.array(data[:,j])
        shiftsr = cp.array(shifts[:, j])        
        psir = cp.pad(datar,((0,0),(ne//2-n//2-pad,ne//2-n//2-pad),(ne//2-n//2-pad,ne//2-n//2-pad)))        
        psi += ST(psir,shiftsr)
    return psi.get()

# adjoint tests
tmp = data.copy()
arr1 = np.pad(tmp[:,0],((0,0),(ne//2-n//2,ne//2-n//2),(ne//2-n//2,ne//2-n//2)),'symmetric')     
prb1 = np.ones([1,n+2*pad,n+2*pad],dtype='complex64')
shifts = np.array(shifts_code)
arr2 = Sop(arr1,shifts*rho)
arr3 = STop(arr2,shifts*rho)

arr4 = L1op(arr2)
arr5 = LT1op(arr4)

print(f'{np.sum(arr1*np.conj(arr3))}==\n{np.sum(arr2*np.conj(arr2))}')
print(f'{np.sum(arr2*np.conj(arr5))}==\n{np.sum(arr4*np.conj(arr4))}')

arr4 = L2op(arr2)
arr5 = LT2op(arr4)

print(f'{np.sum(arr1*np.conj(arr3))}==\n{np.sum(arr2*np.conj(arr2))}')
print(f'{np.sum(arr2*np.conj(arr5))}==\n{np.sum(arr4*np.conj(arr4))}')


# ## Reconstruction with the CG (Carlsson) with Hessians

# ### Define real-valued summation and dot product

# In[9]:


def reprod(a,b):
    return a.real*b.real+a.imag*b.imag

def redot(a,b,axis=None):    
    res = np.sum(reprod(a,b),axis=axis)        
    return res


# ### Reusing the functional $F$ from previously (with $L:=L_2$), and $M$ from previously, we can write 
# ### $$I({{x}},\Psi)=F(M(x)\cdot \Psi)$$ where we omit the variables $q$ and $c$ from the arguments of $M$, since we consider these constant. 
# ### Set $$N(x,\Psi)=L_1\big(M(x)\big)\cdot \Psi.$$
# 

# 

# ## Gradients

# #### $$\nabla F=2 \left(L^*_2\left( (L_2(N))-\tilde D\right)\right).$$
# #### where $$\tilde D = D \frac{(L_2(N))}{|L_2(N)|}$$
# 
# 

# In[10]:


def gradientF(vars,d):
    (psi,code,q,x) = (vars['psi'], vars['code'], vars['prb'], vars['shift'])
    nxpsi = psi*L1op(q*Sop(code,x))
    Lpsi = L2op(nxpsi)
    td = d*(Lpsi/np.abs(Lpsi))
    res = 2*LT2op(Lpsi - td)
    return res


# #### $\nabla_{{\Psi}} I|_{({x}_0,\Psi_0)}=\overline{L_1(J(q)\cdot S_{x_0}(c))}\cdot \nabla F|_{N({x}_0,\Psi_0)} $
# ##### $\nabla_{{x}} I|_{({x}_0,\Psi_0)}=\mathsf{Re} \Big(\big( \Big\langle (\nabla F|_{N({x}_0,\Psi_0)})_k, L_1\Big(q\cdot  C(\mathcal{F}^{-1}(-2\pi i\xi_1 e^{ -2\pi i{x}_{0,k}\cdot {\xi}}\hat{c}))\Big)\cdot \Psi_{0,k}\Big\rangle,\Big\langle (\nabla F|_{N({x}_0,\Psi_0)})_k,L_1\Big(q\cdot C(\mathcal{F}^{-1}(-2\pi i\xi_2 e^{ -2\pi i{x}_{0,k}\cdot {\xi}}\hat{c}))\Big) \Psi_{0,k}\Big\rangle\big)\Big)_{k=1}^K$
# 
# #### new operator $$T_{c,w}(x) = C(\mathcal{F}^{-1}(w e^{-2\pi i \boldsymbol{x}_{0}\cdot \boldsymbol{\xi}}\hat{c_0}))$$

# In[11]:


def gradientpsi(code,q,x,gradF):
    return np.sum(np.conj(L1op(q*Sop(code,x)))*gradF,axis=1)

def Twop_(code,x,w):
    data = np.zeros([code.shape[0], npos, n+2*pad, n+2*pad], dtype='complex64')
    code = cp.array(code)
    nn = code.shape[-1]
    xi1 = cp.fft.fftfreq(2*nn).astype('float32')
    [xi2, xi1] = cp.meshgrid(xi1, xi1)
    for i in range(npos):
        coder = code.copy()            
        p = cp.array(x[:,i])
        coder = cp.pad(coder, ((0, 0), (nn//2, nn//2), (nn//2, nn//2)), 'constant')
        pp = w*cp.exp(-2*cp.pi*1j*(xi1*p[:, 0, None, None]+xi2*p[:, 1, None, None]))    
        coder = cp.fft.ifft2(pp*cp.fft.fft2(coder))   
        data[:,i] = coder[:, nn-n//2-pad:nn+n//2+pad, nn-n//2-pad:nn+n//2+pad].get()        
    return data

# def Twop_(code,x,w):
#     data = np.zeros([code.shape[0], npos, n+2*pad, n+2*pad], dtype='complex64')
#     code = cp.array(code)
#     nn = code.shape[-1]
#     xi1 = cp.fft.fftfreq(2*code.shape[-1]).astype('float32')
#     [xi2, xi1] = cp.meshgrid(xi1, xi1)
#     for i in range(npos):
#         coder = code.copy()            
#         p = cp.array(x[:,i])
#         pint = p.astype('int32')
#         pfloat = p-pint

#         st = nn//2-pint[0,0]-ne//2-pad
#         end = st+ne+2*pad 
#         coder = coder[:,st:end,st:end]
#         pp = w*cp.exp(-2*cp.pi*1j*(xi1*pfloat[:, 0, None, None]+xi2*pfloat[:, 1, None, None]))    
        
#         coder = cp.fft.ifft2(pp*cp.fft.fft2(coder))           

#         st = ne//2+pad-n//2-pad
#         end = st+n+2*pad
#         data[:,i] = coder[:, st:end, st:end].get()        
#     return data

def gradientx(psi,code,q,x,gradF):
    xi1 = cp.fft.fftfreq(2*code.shape[-1]).astype('float32')    
    [xi2, xi1] = cp.meshgrid(xi1, xi1)  
    tksi1 = Twop_(code,x,-2*cp.pi*1j*xi1)
    tksi2 = Twop_(code,x,-2*cp.pi*1j*xi2)    

    tksi1 = psi*L1op(q*tksi1)
    tksi2 = psi*L1op(q*tksi2)

    gradx = np.zeros([1,npos,2],dtype='float32')    
    gradx[:,:,0] = redot(gradF,tksi1,axis=(2,3))
    gradx[:,:,1] = redot(gradF,tksi2,axis=(2,3))
    return gradx

def gradients(vars,d,gradF):
    (psi,code,q,x) = (vars['psi'], vars['code'], vars['prb'], vars['shift'])
    grads = {}
    grads['psi'] = gradientpsi(code, q,x,gradF)
    grads['shift'] = rho*gradientx(psi,code,q,x,gradF)
    return grads



# ##### $$\frac{1}{2}\mathcal{H}^F|_{x_0}(y,z)= \left\langle \mathbf{1}-d_{0}, \mathsf{Re}({L_2(y)}\overline{L(z)})\right\rangle+\left\langle d_{0},(\mathsf{Re} (\overline{l_0}\cdot L_2(y)))\cdot (\mathsf{Re} (\overline{l_0}\cdot L_2(z)))\right\rangle.$$
# ##### $$l_0=L_2(x_0)/|L_2(x_0)|$$
# ##### $$d_0=d/|L_2(x_0)|$$
# 

# In[12]:


def hessianF(hpsi,hpsi1,hpsi2,data):
    Lpsi = L2op(hpsi)        
    Lpsi1 = L2op(hpsi1)
    Lpsi2 = L2op(hpsi2)    
    l0 = Lpsi/np.abs(Lpsi)
    d0 = data/np.abs(Lpsi)
    v1 = np.sum((1-d0)*reprod(Lpsi1,Lpsi2))
    v2 = np.sum(d0*reprod(l0,Lpsi1)*reprod(l0,Lpsi2))    
    return 2*(v1+v2)


# #### $$ D T_c|_{{\boldsymbol{z}_0}}(\Delta \boldsymbol{z})=C(\mathcal{F}^{-1}(-2\pi i\xi_1 e^{ -2\pi i\boldsymbol{z}_0\cdot \boldsymbol{\xi}}\hat{c}))\Delta {z}_{1}+C(\mathcal{F}^{-1}(-2\pi i\xi_2 e^{-2\pi i \boldsymbol{z}_0\cdot \boldsymbol{\xi}}\hat{c}))\Delta {z}_2$$

# In[13]:


def DT(code,x,dx):
    xi1 = cp.fft.fftfreq(2*code.shape[-1]).astype('float32')
    [xi2, xi1] = cp.meshgrid(xi1, xi1)
    tksi1 = Twop_(code,x,-2*cp.pi*1j*xi1)
    tksi2 = Twop_(code,x,-2*cp.pi*1j*xi2)
    res = tksi1*dx[:,:,0,None,None]+tksi2*dx[:,:,1,None,None]
    return res


# #### $$D^2{T_c}|_{{\boldsymbol{z}_0}}(\Delta\boldsymbol{z},\Delta\boldsymbol{w})=$$
# #### $$\Delta {z}_{1}\Delta {w}_{1} C(\mathcal{F}^{-1}(-4\pi^2 \xi_1^2 e^{ -2\pi i\boldsymbol{z}_0\cdot \boldsymbol{\xi}}\hat{c})) +(\Delta {z}_{1}\Delta {w}_{2} +\Delta {w}_{1}\Delta {z}_{2})C(\mathcal{F}^{-1}(-4\pi^2 \xi_1\xi_2 e^{ -2\pi i\boldsymbol{z}_0\cdot \boldsymbol{\xi}}\hat{c}))+\Delta {z}_{2}\Delta {w}_{2} C(\mathcal{F}^{-1}(-4\pi^2\xi_2^2 e^{ -2\pi i\boldsymbol{z}_0\cdot \boldsymbol{\xi}}\hat{c}))$$

# In[14]:


def D2T(code,x,dx1,dx2):
    xi1 = cp.fft.fftfreq(2*code.shape[-1]).astype('float32')
    [xi2,xi1] = cp.meshgrid(xi1, xi1)
    dx11 = dx1[:,:,:,None,None] 
    dx22 = dx2[:,:,:,None,None] 
    res = dx11[:,:,0]*dx22[:,:,0]*Twop_(code,x,-4*cp.pi**2*xi1**2)+ \
         (dx11[:,:,0]*dx22[:,:,1]+dx11[:,:,1]*dx22[:,:,0])*Twop_(code,x,-4*cp.pi**2*xi1*xi2)+ \
          dx11[:,:,1]*dx22[:,:,1]*Twop_(code,x,-4*cp.pi**2*xi2**2)
    return res


# #### $$DM|_{{x}}(\Delta{x})=  \Big(q\cdot DT_{c}|_{{{x}_{0,k}}}( \Delta {x}_k) \Big)_{k=1}^K$$
#                                                                                          
# 

# In[15]:


def DM(code,q,x,dx):
    res = rho*q*DT(code,x,dx)   
    return res


# #### $$D^2M|_{{x}}\big(\Delta{x}^{(1)},\Delta{x}^{(2)}\big)= q\cdot D^2{T_c}|_{{{x}_0}}(\Delta{x}_k^{(1)},\Delta{x}_k^{(2)})    $$

# In[16]:


def D2M(code,q,x,dx1,dx2):    
    res =  rho**2*q*D2T(code,x,dx1,dx2)      
    return res


# ### $$DN|_{(x_0,\Psi_0)}(\Delta x,\Delta \Psi)=L_1\Big(DM|_{x_0}(\Delta x)\Big)\cdot \Psi_0+N(x_0,\Delta \Psi)$$

# In[17]:


def DN(psi,code,q,x,dpsi,dx):
    nxpsi = dpsi*L1op(q*Sop(code,x))
    res = psi*L1op(DM(code,q,x,dx))+nxpsi
    return res


# 
# ### $$D^2N|_{(x_0,\Psi_0)}\left((\Delta x^{(1)},\Delta \Psi^{(1)}),(\Delta x^{(2)},\Delta \Psi^{(2)})\right)= $$
# ### $$L_1\Big(D^2M|_{x_0}(\Delta x^{(1)},\Delta x^{(2)})\Big)\cdot\Psi_0+L_1\Big(DM|_{x_0}(\Delta x^{(1)})\Big)\cdot\Delta \Psi^{(2)}+L_1\Big(DM|_{x_0}(\Delta x^{(2)})\Big)\cdot\Delta \Psi^{(1)} $$
# 

# In[18]:


def D2N(psi,code,q,x,dpsi1,dpsi2,dx1,dx2):
    res = psi*L1op(D2M(code,q,x,dx1,dx2)) + dpsi2*L1op(DM(code,q,x,dx1)) + dpsi1*L1op(DM(code,q,x,dx2))
    return res


# ### $$H^I|_{x_0,\Psi_0}((\Delta x^{(1)},\Delta \Psi^{(1)}),(\Delta x^{(2)},\Delta \Psi^{(2)}))=$$
# ### $$\Big\langle \nabla F|_{N(x_0,\Psi_0)}, D^2N|_{(x_0,\Psi_0)}((\Delta x^{(1)},\Delta \Psi^{(1)}),(\Delta x^{(2)},\Delta \Psi^{(2)}))\Big\rangle$$
# ### $$H^F|_{N(x_0,\Psi_0)}\Big(DN|_{(x_0,\Psi_0)}(\Delta x^{(1)},\Delta \Psi^{(1)}),DN|_{(x_0,\Psi_0)}(\Delta x^{(1)},\Delta \Psi^{(1)})\Big)$$

# In[19]:


def hessian2(psi,q,x,dpsi1,dx1,dpsi2,dx2,d,gradF):
    d2n = D2N(psi,code,q,x,dpsi1,dpsi2,dx1,dx2)
    dn1 = DN(psi,code,q,x,dpsi1,dx1)
    dn2 = DN(psi,code,q,x,dpsi2,dx2)
    nxpsi = psi*L1op(q*Sop(code,x))  
    return redot(gradF,d2n)+hessianF(nxpsi,dn1,dn2,d) 


# In[20]:


def calc_beta(vars,grads,etas,d,gradF):
    (psi,q,x) = (vars['psi'], vars['prb'], vars['shift'])
    (dpsi1,dx1) = (grads['psi'], grads['shift'])
    (dpsi2,dx2) = (etas['psi'], etas['shift'])
    
    d2n = D2N(psi,code,q,x,dpsi1,dpsi2,dx1,dx2)
    dn1 = DN(psi,code,q,x,dpsi1,dx1)
    dn2 = DN(psi,code,q,x,dpsi2,dx2)
    nxpsi = psi*L1op(q*Sop(code,x))  
    top = redot(gradF,d2n)+hessianF(nxpsi,dn1,dn2,d) 

    bottom = redot(gradF,d2n)+hessianF(nxpsi,dn2,dn2,d) 
    return top/bottom

def calc_alpha(vars,grads,etas,d,gradF):    
    (psi,q,x) = (vars['psi'], vars['prb'], vars['shift'])
    (dpsi1,dx1) = (grads['psi'], grads['shift'])
    (dpsi2,dx2) = (etas['psi'], etas['shift'])
    
    d2n = D2N(psi,code,q,x,dpsi1,dpsi2,dx1,dx2)
    dn2 = DN(psi,code,q,x,dpsi2,dx2)
    nxpsi = psi*L1op(q*Sop(code,x))  
    
    top = -redot(dpsi1,dpsi2)-redot(dx1,dx2)
    bottom = redot(gradF,d2n)+hessianF(nxpsi,dn2,dn2,d)     
    return top/bottom, top, bottom


# ## debug functions

# In[21]:


def minf(fpsi,data):
    f = np.linalg.norm(np.abs(fpsi)-data)**2
    return f

def plot_debug2(vars,etas,top,bottom,alpha,data):
    (psi,code,q,x) = (vars['psi'],vars['code'], vars['prb'], vars['shift'])
    (dpsi2,dx2) = (etas['psi'], etas['shift'])
    npp = 17
    errt = np.zeros(npp*2)
    errt2 = np.zeros(npp*2)
    for k in range(0,npp*2):
        psit = psi+(alpha*k/(npp-1))*dpsi2
        xt = x+(alpha*k/(npp-1))*dx2*rho
        fpsit = L2op(psit*L1op(q*Sop(code,xt)))
        errt[k] = minf(fpsit,data)    

    t = alpha*(cp.arange(2*npp))/(npp-1)
    errt2 = minf(L2op(psi*L1op(q*Sop(code,x))),data)-top*t+0.5*bottom*t**2
    
    plt.plot(alpha*np.arange(2*npp)/(npp-1),errt,'.')
    plt.plot(alpha*np.arange(2*npp)/(npp-1),errt2.get(),'.')
    plt.show()

def plot_debug3(shifts,shifts_init):
    plt.plot(shifts_init[0,:,0]-(shifts[0,:,0]),'r.')
    plt.plot(shifts_init[0,:,1]-(shifts[0,:,1]),'b.')
    plt.show()

def vis_debug(vars,i):
    mshow_polar(vars['psi'][0],show)
    mshow_polar(vars['psi'][0,ne//2-128:ne//2+128,ne//2-128:ne//2+128],show)
    # mshow_polar(vars['prb'][0],show)
    dxchange.write_tiff(np.angle(vars['psi'][0]),f'{path}/crec_code_angle{flg}/{i:03}',overwrite=True)
    dxchange.write_tiff(np.angle(vars['prb'][0]),f'{path}/crec_prb_angle{flg}/{i:03}',overwrite=True)
    dxchange.write_tiff(np.abs(vars['psi'][0]),f'{path}/crec_code_abs{flg}/{i:03}',overwrite=True)
    dxchange.write_tiff(np.abs(vars['prb'][0]),f'{path}/crec_prb_abs{flg}/{i:03}',overwrite=True)
    
def err_debug(vars, grads, data):    
    err = minf(L2op(vars['psi']*L1op(vars['prb']*Sop(vars['code'],vars['shift']))),data)    
    
    print(f'gradient norms (psi, shift): {np.linalg.norm(grads['psi']):.2f}, {np.linalg.norm(grads['shift']):.2f}')                        
    return err


# # Main CG loop (fifth rule)

# In[22]:


def cg_holo(data, vars, pars):

    data = np.sqrt(data)    
    shifts_init = vars['shift'].copy()
    erra = np.zeros(pars['niter'])
    alphaa = np.zeros(pars['niter'])    
        
    for i in range(pars['niter']):         
        
        gradF = gradientF(vars,data)
        grads = gradients(vars,data,gradF)
        #grads['psi'][:] = 0
        # etas = {}
        # etas['psi'] = -grads['psi']
        # etas['shift'] = -grads['shift']
        if i==0:
            etas = {}
            etas['psi'] = -grads['psi']
            etas['shift'] = -grads['shift']
        else:      
            beta = calc_beta(vars, grads, etas, data, gradF)
            etas['psi'] = -grads['psi'] + beta*etas['psi']
            etas['shift'] = -grads['shift'] + beta*etas['shift']

        alpha,top,bottom = calc_alpha(vars, grads, etas, data, gradF)              
        # if i % pars['vis_step'] == 0:
        #     plot_debug2(vars,etas,top,bottom,alpha,data)
        
        vars['psi'] += alpha*etas['psi']
        vars['shift'] += alpha*rho*etas['shift']
        
        if i % pars['err_step'] == 0:
            err = err_debug(vars, grads, data)    
            print(f'{i}) {alpha=:.5f}, {err=:1.5e}',flush=True)
            erra[i] = err
            alphaa[i] = alpha

        if i % pars['vis_step'] == 0:
            vis_debug(vars, i)
            # plot_debug3(vars['shift'],shifts_init)     
            d = np.abs(L2op(vars['psi']*L1op(vars['prb']*Sop(vars['code'],vars['shift']))))
            mshow(d[0,0]-data[0,0],show)        
    
    return vars,erra,alphaa

vars = {}
vars['psi'] = rec_paganin.copy()*0+1
vars['prb'] = prb.copy()
vars['shift'] = shifts.copy()
vars['code'] = code.copy()

pars = {'niter': 513, 'err_step': 1, 'vis_step': 16}
vars,erra,alphaa = cg_holo(data, vars, pars)   


# In[ ]:




