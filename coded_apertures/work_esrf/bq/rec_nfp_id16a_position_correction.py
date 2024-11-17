#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cupy as cp
import h5py
from holotomocupy.holo import G, GT
from holotomocupy.shift import S, ST
from holotomocupy.recon_methods import multiPaganin
from holotomocupy.utils import *
from holotomocupy.proc import remove_outliers
# Use managed memory
cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)


# # Init data sizes and parametes of the PXM of ID16A

# In[ ]:


n = 2048  # object size in each dimension
pad = n//8
npos= 18*18
pos_step = 1
z1c = -17.75e-3
# thickness of the coded aperture
code_thickness = 1.8e-6 #in m
# feature size
ill_feature_size = 2e-6 #in m

detector_pixelsize = 3.03751e-6
energy = 33.35  # [keV] xray energy
wavelength = 1.2398419840550367e-09/energy  # [m] wave length
focusToDetectorDistance = 1.28  # [m]
sx0 = 3.7e-4
z1 = z1c
z1 = np.tile(z1, [npos])
z2 = focusToDetectorDistance-z1
distances = (z1*z2)/focusToDetectorDistance
magnifications = focusToDetectorDistance/z1
voxelsize = np.abs(detector_pixelsize/magnifications[0]*2048/n)  # object voxel size

# sample size after demagnification
ne = 4096+2048+1024+128+2*pad
# ne = 2048+2*pad
# ne = 3096+512+2*pad
show = True

flg = f'{n}_{z1c}full'
path = f'/data2/vnikitin/nfp_codes'
rho = 0.13


# ## Read data

# In[ ]:


with h5py.File('/data/vnikitin/ESRF/ID16A/20240924/SiemensLH/code2um_nfp18x18_01/code2um_nfp18x18_010000.h5') as fid:
    data0 = fid['/entry_0000/measurement/data'][:npos].astype('float32')
    
with h5py.File('/data/vnikitin/ESRF/ID16A/20240924/SiemensLH/code2um_nfp18x18_01/ref_0000.h5') as fid:
    ref0 = fid['/entry_0000/measurement/data'][:].astype('float32')
with h5py.File('/data/vnikitin/ESRF/ID16A/20240924/SiemensLH/code2um_nfp18x18_01/dark_0000.h5') as fid:
    dark0 = fid['/entry_0000/measurement/data'][:].astype('float32')

data0 = data0[np.newaxis]

shifts_code0 = np.loadtxt(f'/data/vnikitin/ESRF/ID16A/20240924/positions/shifts_code_nfp18x18ordered.txt')[np.newaxis,:,::-1]
shifts_code0 = shifts_code0/voxelsize*1e-6
print(shifts_code0[0,:4])

shifts_code0 = np.load(f'shifts_code_new.npy')
print(shifts_code0[0,:4])
shifts_code0[:,:,1]*=-1

#centering
shifts_code0[:,:,1]-=(np.amax(shifts_code0[:,:,1])+np.amin(shifts_code0[:,:,1]))/2
shifts_code0[:,:,0]-=(np.amax(shifts_code0[:,:,0])+np.amin(shifts_code0[:,:,0]))/2

shifts_code0 = shifts_code0.reshape(1,int(np.sqrt(npos)),int(np.sqrt(npos)),2)
shifts_code0 = shifts_code0[:,::pos_step,::pos_step,:].reshape(1,npos//pos_step**2,2)
data0 = data0.reshape(1,int(np.sqrt(npos)),int(np.sqrt(npos)),n,n)
data0 = data0[:,::pos_step,::pos_step,:].reshape(1,npos//pos_step**2,n,n)

ids = np.where((np.abs(shifts_code0[0,:,0])<ne//2-n//2-pad)*(np.abs(shifts_code0[0,:,1])<ne//2-n//2-pad))[0]#[0:2]
data0 = data0[:,ids]
shifts_code0 = shifts_code0[:,ids]

print(shifts_code0.shape)
# plt.plot(shifts_code0[0,:,0],shifts_code0[0,:,1],'.')
# plt.axis('square')
# plt.show()

mshow_complex(data0[0,0]/ref0[0]+1j*data0[0,1]/ref0[0],show)
npos = len(ids)
print(f'{npos=}')
print(np.amin(shifts_code0[0,:,0]),np.amax(shifts_code0[0,:,0]))
print(np.amin(shifts_code0[0,:,1]),np.amax(shifts_code0[0,:,1]))


# In[ ]:


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

mshow_complex(data[0,0]+1j*rdata[0,0],show)
mshow_complex(ref[0]+1j*dark[0],show)


# # Construct operators
# 

# In[ ]:


def Lop(psi):
    psi = cp.array(psi)
    data = cp.zeros([psi.shape[0], npos, n, n], dtype='complex64')
    for i in range(npos):
        psir = psi[:,i].copy()       
        psir = G(psir, wavelength, voxelsize, distances[i])
        data[:, i] = psir[:, pad:n+pad, pad:n+pad]
    return data

def LTop(data):
    psi = cp.zeros([data.shape[0],npos, n+2*pad, n+2*pad], dtype='complex64')
    for j in range(npos):
        psir = cp.pad(data[:, j], ((0, 0), (pad, pad), (pad, pad))).astype('complex64')
        psir = GT(psir, wavelength, voxelsize, distances[j])        
        psi[:,j] = psir
    return psi

def Sop(psi,shifts):
    psi = cp.array(psi)
    data = cp.zeros([psi.shape[0], npos, n+2*pad, n+2*pad], dtype='complex64')
    for i in range(npos):
        psir = psi.copy()    
        psir = S(psir, shifts[:, i])
        nee = psir.shape[1]        
        data[:,i] = psir[:, nee//2-n//2-pad:nee//2+n//2+pad, nee//2-n//2-pad:nee//2+n//2+pad]
    return data

def STop(data,shifts):
    psi = cp.zeros([data.shape[0], ne, ne], dtype='complex64')

    for j in range(npos):
        psir = cp.pad(data[:,j],((0,0),(ne//2-n//2-pad,ne//2-n//2-pad),(ne//2-n//2-pad,ne//2-n//2-pad)))        
        psi += ST(psir,shifts[:,j])
    return psi

# adjoint tests
tmp = cp.array(data)
arr1 = cp.pad(tmp[:,0],((0,0),(ne//2-n//2,ne//2-n//2),(ne//2-n//2,ne//2-n//2)),'symmetric')     
prb1 = cp.ones([1,n+2*pad,n+2*pad],dtype='complex64')
shifts = cp.array(shifts_code0)
arr2 = Sop(arr1,shifts*rho)
arr3 = STop(arr2,shifts*rho)

arr4 = Lop(arr2)
arr5 = LTop(arr4)

print(f'{cp.sum(arr1*cp.conj(arr3))}==\n{cp.sum(arr2*cp.conj(arr2))}')
print(f'{cp.sum(arr2*cp.conj(arr5))}==\n{cp.sum(arr4*cp.conj(arr4))}')


# ## Reconstruction with the CG (Carlsson) with Hessians

# ### Define real-valued summation and dot product

# In[ ]:


def reprod(a,b):
    return a.real*b.real+a.imag*b.imag

def redot(a,b,axis=None):
    return cp.sum(reprod(a,b),axis=axis)


# ## Gradients

# #### $$\nabla F=2 \left(L^*\left( L(M(q_0,\psi_0,\boldsymbol{x}_0))-\tilde D\right)\right).$$
# #### where $$\tilde D = D \frac{L(M(q_0,\psi_0,\boldsymbol{x}_0))}{|L(M(q_0,\psi_0,\boldsymbol{x}_0))|}$$
# 
# 

# In[ ]:


def gradientF(psi,q,x,d):
    Lpsi = Lop(Sop(psi,x)*q)
    td = d*(Lpsi/cp.abs(Lpsi))
    res = 2*LTop(Lpsi - td)
    return res


# ##### $$\nabla_{\psi} G|_{(q_0,\psi_0,\boldsymbol{x}_0)}=S_{\boldsymbol{x}_{0}}^*\left(\overline{J(q_0)}\cdot \nabla F\right).$$
# ##### $$\nabla_{q} G|_{(q_0,\psi_0,\boldsymbol{x}_0)}=J^*\left( \overline{S_{\boldsymbol{x}_{0}}(\psi_0)}\cdot \nabla F\right).$$
# ##### $$\nabla_{\boldsymbol{x}_0} G|_{(q_0,\psi_0,\boldsymbol{x}_0)}=\textsf{Re}\Big(\big( \Big\langle \overline{q_0}\cdot \nabla F,   C(\mathcal{F}^{-1}(-2\pi i \xi_1 e^{ -2\pi i \boldsymbol{x}_{0,k}\cdot \boldsymbol{\xi}}\hat{\psi_0}))\Big\rangle,\Big\langle \overline{q_0}\cdot \nabla F,C(\mathcal{F}^{-1}(-2\pi i \xi_2 e^{ -2\pi i \boldsymbol{x}_{0,k}\cdot \boldsymbol{\xi}}\hat{\psi_0})) \Big\rangle\big)\Big)_{k=1}^K. $$
# 
# #### new operator $$T_{\psi,w}(x) = C(\mathcal{F}^{-1}(w e^{-2\pi i \boldsymbol{x}_{0}\cdot \boldsymbol{\xi}}\hat{\psi_0}))$$
# 
# 
# 

# In[ ]:


def gradientpsi(q,x,gradF):
    return STop(cp.conj(q)*gradF,x)

def gradientq(psi,x,gradF):
    return cp.sum(cp.conj(Sop(psi,x))*gradF,axis=1)

def Twop_(psi,x,w):
    data = cp.zeros([psi.shape[0], npos, n+2*pad, n+2*pad], dtype='complex64')
    xi1 = cp.fft.fftfreq(2*ne).astype('float32')
    [xi2, xi1] = cp.meshgrid(xi1, xi1)
    for i in range(npos):
        psir = psi.copy()            
        p = x[:,i]
        psir = cp.pad(psir, ((0, 0), (ne//2, ne//2), (ne//2, ne//2)), 'constant')
        pp = w*cp.exp(-2*cp.pi*1j*(xi1*p[:, 0, None, None]+xi2*p[:, 1, None, None]))    
        psir = cp.fft.ifft2(pp*cp.fft.fft2(psir))   
        data[:,i] = psir[:, ne-n//2-pad:ne+n//2+pad, ne-n//2-pad:ne+n//2+pad]        
    return data

def gradientx(psi,q,x,gradF):
    xi1 = cp.fft.fftfreq(2*psi.shape[-1]).astype('float32')    
    [xi2, xi1] = cp.meshgrid(xi1, xi1)  
    tksi1 = Twop_(psi,x,-2*cp.pi*1j*xi1)
    tksi2 = Twop_(psi,x,-2*cp.pi*1j*xi2)    
    gradx = cp.zeros([1,npos,2],dtype='float32')
    tmp = cp.conj(q)*gradF
    gradx[:,:,0] = redot(tmp,tksi1,axis=(2,3))
    gradx[:,:,1] = redot(tmp,tksi2,axis=(2,3))
    return gradx

def gradients(psi,q,x,d):
    gradF = gradientF(psi,q,x,d)
    gradpsi = gradientpsi(q,x,gradF)
    gradq = gradientq(psi,x,gradF)
    gradx = rho*gradientx(psi,q,x,gradF)
    return gradpsi,gradq,gradx


# ##### $$\frac{1}{2}\mathcal{H}|_{x_0}(y,z)= \left\langle \mathbf{1}-d_{0}, \mathsf{Re}({L(y)}\overline{L(z)})\right\rangle+\left\langle d_{0},(\mathsf{Re} (\overline{l_0}\cdot L(y)))\cdot (\mathsf{Re} (\overline{l_0}\cdot L(z)))\right\rangle.$$
# ##### $$l_0=L(x_0)/|L(x_0)|$$
# ##### $$d_0=d/|L(x_0)|$$
# 

# In[ ]:


def hessianF(hpsi,hpsi1,hpsi2,data):
    Lpsi = Lop(hpsi)        
    Lpsi1 = Lop(hpsi1)
    Lpsi2 = Lop(hpsi2)    
    l0 = Lpsi/cp.abs(Lpsi)
    d0 = data/cp.abs(Lpsi)
    v1 = cp.sum((1-d0)*reprod(Lpsi1,Lpsi2))
    v2 = cp.sum(d0*reprod(l0,Lpsi1)*reprod(l0,Lpsi2))    
    return 2*(v1+v2)


# #### $$ D T_\psi|_{{\boldsymbol{z}_0}}(\Delta \boldsymbol{z})=C(\mathcal{F}^{-1}(-2\pi i\xi_1 e^{ -2\pi i\boldsymbol{z}_0\cdot \boldsymbol{\xi}}\hat{\psi}))\Delta {z}_{1}+C(\mathcal{F}^{-1}(-2\pi i\xi_2 e^{-2\pi i \boldsymbol{z}_0\cdot \boldsymbol{\xi}}\hat{\psi}))\Delta {z}_2$$

# In[ ]:


def DT(psi,x,dx):
    xi1 = cp.fft.fftfreq(2*psi.shape[-1]).astype('float32')
    [xi2, xi1] = cp.meshgrid(xi1, xi1)
    tksi1 = Twop_(psi,x,-2*cp.pi*1j*xi1)
    tksi2 = Twop_(psi,x,-2*cp.pi*1j*xi2)
    res = tksi1*dx[:,:,0,None,None]+tksi2*dx[:,:,1,None,None]
    return res


# #### $$D^2{T_\psi}|_{{\boldsymbol{z}_0}}(\Delta\boldsymbol{z},\Delta\boldsymbol{w})=$$
# #### $$\Delta {z}_{1}\Delta {w}_{1} C(\mathcal{F}^{-1}(-4\pi^2 \xi_1^2 e^{ -2\pi i\boldsymbol{z}_0\cdot \boldsymbol{\xi}}\hat{\psi})) +(\Delta {z}_{1}\Delta {w}_{2} +$$
# #### $$ \Delta {w}_{1}\Delta {z}_{2})C(\mathcal{F}^{-1}(-4\pi^2 \xi_1\xi_2 e^{ -2\pi i\boldsymbol{z}_0\cdot \boldsymbol{\xi}}\hat{\psi}))+\Delta {z}_{2}\Delta {w}_{2} C(\mathcal{F}^{-1}(-4\pi^2\xi_2^2 e^{ -2\pi i\boldsymbol{z}_0\cdot \boldsymbol{\xi}}\hat{\psi}))$$

# In[ ]:


def D2T(psi,x,dx1,dx2):
    xi1 = cp.fft.fftfreq(2*psi.shape[-1]).astype('float32')
    [xi2,xi1] = cp.meshgrid(xi1, xi1)
    dx11 = dx1[:,:,:,None,None] 
    dx22 = dx2[:,:,:,None,None] 
    res = dx11[:,:,0]*dx22[:,:,0]*Twop_(psi,x,-4*cp.pi**2*xi1**2)+ \
         (dx11[:,:,0]*dx22[:,:,1]+dx11[:,:,1]*dx22[:,:,0])*Twop_(psi,x,-4*cp.pi**2*xi1*xi2)+ \
          dx11[:,:,1]*dx22[:,:,1]*Twop_(psi,x,-4*cp.pi**2*xi2**2)
    return res


# #### $$ DM|_{(q_0,\psi_0,\boldsymbol{x})}(\Delta q, \Delta \psi,\Delta\boldsymbol{x})=$$
# #### $$ \Big(\Delta q\cdot T_{\psi_0}({\boldsymbol{x}_{0,k}})+ q_0\cdot \big(T_{\Delta \psi}({\boldsymbol{x}_{0,k}})+  DT_{\psi_0}|_{{\boldsymbol{x}_{0,k}}}( \Delta \boldsymbol{x}_k)\big) \Big)_{k=1}^K=$$
# #### $$ J(\Delta q)\cdot S_{\boldsymbol{x}_{0,k}}(\psi_0)+ J(q_0)\cdot S_{\boldsymbol{x}_{0}}{(\Delta \psi)}+  \Big(q_0\cdot DT_{\psi_0}|_{{\boldsymbol{x}_{0,k}}}( \Delta \boldsymbol{x}_k) \Big)_{k=1}^K$$
# 

# In[ ]:


def DM(psi,q,x,dpsi,dq,dx):
    res = dq*Sop(psi,x)+q*(Sop(dpsi,x)+rho*DT(psi,x,dx))   
    return res


# ##### $$ D^2M|_{(q_0,\psi_0,\boldsymbol{x})}\big((\Delta q^{(1)}, \Delta \psi^{(1)},\Delta\boldsymbol{x}^{(1)}),(\Delta q^{(2)}, \Delta \psi^{(2)},\Delta\boldsymbol{x}^{(2)})\big)= $$
# ##### $$\Big( q_0\cdot DT_{\Delta\psi^{(1)}}|_{{\boldsymbol{x}_{0,k}}}( \Delta \boldsymbol{x}^{(2)})+q_0\cdot DT_{\Delta\psi^{(2)}}|_{{\boldsymbol{x}_{0,k}}}( \Delta \boldsymbol{x}^{(1)})+ q_0\cdot D^2{T_\psi}|_{{\boldsymbol{x}_0}}(\Delta\boldsymbol{x}^{(1)},\Delta\boldsymbol{x}^{(2)})+$$
# ##### $$\Delta q^{(1)}\cdot T_{\Delta \psi^{(2)}}({\boldsymbol{x}_{0,k}})+\Delta q^{(2)}\cdot T_{\Delta \psi^{(1)}}({\boldsymbol{x}_{0,k}})+ $$
# ##### $$\Delta q^{(1)}\cdot DT_{\psi_0}|_{{\boldsymbol{x}_{0,k}}}( \Delta \boldsymbol{x}^{(2)})+\Delta q^{(2)}\cdot DT_{\psi_0}|_{{\boldsymbol{x}_{0,k}}}( \Delta \boldsymbol{x}^{(1)})\Big)_{k=1}^K.$$
# 

# In[ ]:


def D2M(psi,q,x,dpsi1,dq1,dx1,dpsi2,dq2,dx2):    
    res =  q*rho*DT(dpsi1,x,dx2) + q*rho*DT(dpsi2,x,dx1) + q*rho**2*D2T(psi,x,dx1,dx2)  
    res += dq1*Sop(dpsi2,x) + dq2*Sop(dpsi1,x) 
    res += dq1*rho*DT(psi,x,dx2) + dq2*rho*DT(psi,x,dx1)
    return res


# ##### $$\mathcal{H}^G|_{ (q_0,\psi_0,\boldsymbol{x}_0)}\Big((\Delta q^{(1)},\Delta \psi^{(1)},\Delta \boldsymbol{x}^{(1)}),(\Delta q^{(2)},\Delta \psi^{(2)},\Delta \boldsymbol{x}^{(2)})\Big)=$$
# ##### $$\Big\langle \nabla F|_{M(q_0,\psi_0,\boldsymbol{x}_0)}, D^2M|_{(q_0,\psi_0,\boldsymbol{x}_0)}\Big((\Delta q^{(1)},\Delta \psi^{(1)},\Delta \boldsymbol{x}^{(1)}),(\Delta q^{(2)},\Delta \psi^{(2)},\Delta \boldsymbol{x}^{(2)})\Big)\Big\rangle +$$
# ##### $$\mathcal{H}^F|_{M(q_0,\psi_0,\boldsymbol{x}_0)}\Big(DM|_{(q_0,\psi_0,\boldsymbol{x}_0)}(\Delta q^{(1)},\Delta \psi^{(1)},\Delta \boldsymbol{x}^{(1)}),DM|_{(q_0,\psi_0,\boldsymbol{x}_0)}(\Delta q^{(2)},\Delta \psi^{(2)},\Delta \boldsymbol{x}^{(2)})\Big).$$

# ##### $$\mathcal{H}^G|_{ (q_0,\psi_0,\boldsymbol{x}_0)}\Big((\Delta q^{(1)},\Delta \psi^{(1)},\Delta \boldsymbol{x}^{(1)}),(\Delta q^{(2)},\Delta \psi^{(2)},\Delta \boldsymbol{x}^{(2)})\Big)=$$
# ##### $$\Big\langle \nabla F|_{M(q_0,\psi_0,\boldsymbol{x}_0)}, D^2M|_{(q_0,\psi_0,\boldsymbol{x}_0)}\Big((\Delta q^{(1)},\Delta \psi^{(1)},\Delta \boldsymbol{x}^{(1)}),(\Delta q^{(2)},\Delta \psi^{(2)},\Delta \boldsymbol{x}^{(2)})\Big)\Big\rangle +$$
# ##### $$\mathcal{H}^F|_{M(q_0,\psi_0,\boldsymbol{x}_0)}\Big(DM|_{(q_0,\psi_0,\boldsymbol{x}_0)}(\Delta q^{(1)},\Delta \psi^{(1)},\Delta \boldsymbol{x}^{(1)}),DM|_{(q_0,\psi_0,\boldsymbol{x}_0)}(\Delta q^{(2)},\Delta \psi^{(2)},\Delta \boldsymbol{x}^{(2)})\Big).$$

# In[ ]:


def hessian2(psi,q,x,dpsi1,dq1,dx1,dpsi2,dq2,dx2,data):
    gradF = gradientF(psi,q,x,data)
    res = redot(gradF,D2M(psi,q,x,dpsi1,dq1,dx1,dpsi2,dq2,dx2))
    res += hessianF(Sop(psi,x)*q, DM(psi,q,x,dpsi1,dq1,dx1), DM(psi,q,x,dpsi2,dq2,dx2),data)         
    return res


# ### Initial guess for reconstruction (Paganin)

# In[ ]:


def rec_init(rdata,shifts):
    rdata = cp.array(rdata)
    recMultiPaganin = cp.zeros([1,npos,ne,ne],dtype='float32')
    recMultiPaganinr = cp.zeros([1,npos,ne,ne],dtype='float32')# to compensate for overlap
    for j in range(0,npos):
        r = multiPaganin(rdata[:,j:j+1],
                            distances[j:j+1], wavelength, voxelsize,  24.05, 5e-2)    
        rr = r*0+1 # to compensate for overlap
        r = cp.pad(r,((0,0), (ne//2-n//2,ne//2-n//2), (ne//2-n//2,ne//2-n//2)),'constant')   
        rr = cp.pad(rr,((0,0), (ne//2-n//2,ne//2-n//2), (ne//2-n//2,ne//2-n//2)),'constant')   
        recMultiPaganin[:,j] = ST(r,shifts[:,j]).real
        recMultiPaganinr[:,j] = ST(rr,shifts[:,j]).real
        
    recMultiPaganin = cp.sum(recMultiPaganin,axis=1)
    recMultiPaganinr = cp.sum(recMultiPaganinr,axis=1)

    # avoid division by 0
    recMultiPaganinr[np.abs(recMultiPaganinr)<5e-2] = 1

    # compensate for overlap
    recMultiPaganin /= recMultiPaganinr
    recMultiPaganin = cp.exp(1j*recMultiPaganin)

    return recMultiPaganin

rec_paganin = rec_init(rdata,shifts)
mshow_polar(rec_paganin[0],show)
mshow_polar(rec_paganin[0,ne//2-128:ne//2+128,ne//2-128:ne//2+128],show)


# In[ ]:


def minf(fpsi,data):
    f = np.linalg.norm(np.abs(fpsi)-data)**2
    return f

def plot_debug2(psi,prb,shifts,etapsi,etaprb,etashift,top,bottom,gammah,data):
    npp = 17
    errt = np.zeros(npp*2)
    errt2 = np.zeros(npp*2)
    for k in range(0,npp*2):
        psit = psi+(gammah*k/(npp-1))*etapsi
        prbt = prb+(gammah*k/(npp-1))*etaprb
        shiftst = shifts+(gammah*k/(npp-1))*etashift
        fpsit = Lop(Sop(psit,shiftst*rho)*prbt)
        errt[k] = minf(fpsit,data)    

    t = gammah*(cp.arange(2*npp))/(npp-1)
    errt2 = minf(Lop(Sop(psi,shifts*rho)*prb),data)-top*t+0.5*bottom*t**2
    
    plt.plot(gammah.get()*np.arange(2*npp)/(npp-1),errt,'.')
    plt.plot(gammah.get()*np.arange(2*npp)/(npp-1),errt2.get(),'.')
    plt.show()

def plot_debug3(shifts,shifts_init):
    plt.plot(shifts_init[0,:,0].get()-(shifts[0,:,0]).get(),'r.')
    plt.plot(shifts_init[0,:,1].get()-(shifts[0,:,1]).get(),'b.')
    plt.show()


# In[ ]:


def cg_holo(data, psi, prb, shifts, pars):

    data = np.sqrt(data)    
    shifts /= rho
    shifts_init = shifts.copy()
    conv = np.zeros(pars['niter'])
    alphaa = np.zeros(pars['niter'])    
    
    for i in range(pars['niter']):              
        gradpsi, gradprb, gradshift = gradients(psi,prb,shifts*rho,data)
        
        if i==0:
            etapsi = -gradpsi
            etaprb = -gradprb
            etashift = -gradshift
        else:      
            beta = hessian2(psi,prb,shifts*rho,gradpsi,gradprb,gradshift,etapsi,etaprb,etashift,data)/\
                   hessian2(psi,prb,shifts*rho, etapsi, etaprb, etashift,etapsi,etaprb,etashift,data)                        
            etapsi = -gradpsi + beta*etapsi
            etaprb = -gradprb + beta*etaprb
            etashift = -gradshift + beta*etashift            
        
        top = -(redot(gradpsi,etapsi)+
                redot(gradprb,etaprb)+
                redot(gradshift,etashift))       
        bottom = hessian2(psi,prb,shifts*rho,etapsi,etaprb,etashift,
                                         etapsi,etaprb,etashift,data)
        alpha = top/bottom        
        
        psi += alpha*etapsi
        prb += alpha*etaprb
        shifts += alpha*etashift
        
        if i % pars['vis_step'] == 0:
            mshow_polar(psi[0],show)
            mshow_polar(psi[0,ne//2-128:ne//2+128,ne//2-128:ne//2+128],show)
            mshow_polar(prb[0],show)
            dxchange.write_tiff(cp.angle(psi[0]).get(),f'{path}/crec_code_angle{flg}/{i:03}',overwrite=True)
            dxchange.write_tiff(cp.angle(prb[0]).get(),f'{path}/crec_prb_angle{flg}/{i:03}',overwrite=True)
            dxchange.write_tiff(cp.abs(psi[0]).get(),f'{path}/crec_code_abs{flg}/{i:03}',overwrite=True)
            dxchange.write_tiff(cp.abs(prb[0]).get(),f'{path}/crec_prb_abs{flg}/{i:03}',overwrite=True)
            # plot_debug3(shifts*rho,shifts_init*rho)
            

        if i % pars['err_step'] == 0:
            fpsi = Lop(Sop(psi,shifts*rho)*prb)
            err = minf(fpsi,data)
            conv[i] = err
            alphaa[i] = alpha
            print(f'{i}) {alpha=:.5f}, {err=:1.5e}')
            print(f'gradient norms (psi, prb, shift): {cp.linalg.norm(gradpsi):.2f}, {cp.linalg.norm(gradprb):.2f}, {cp.linalg.norm(gradshift):.2f}')                        
        
    
    shifts *= rho
    return psi,prb,shifts,conv,alphaa

data = cp.array(data)
rec_shifts = cp.array(shifts)
rec_prb = cp.ones([1,n+2*pad,n+2*pad],dtype='complex64')
rec_psi = rec_paganin

pars = {'niter': 2048, 'err_step': 4, 'vis_step': 8}
psi,prb,shifts,conv,alphaa = cg_holo(data,rec_psi,rec_prb,rec_shifts, pars)   

