#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cupy as cp
from holotomocupy.holo import G, GT
from holotomocupy.magnification import M, MT
from holotomocupy.shift import S, ST
from holotomocupy.recon_methods import multiPaganin
from holotomocupy.utils import *
from holotomocupy.proc import remove_outliers
import holotomocupy.chunking as chunking
import sys

chunk = 5
chunking.global_chunk = chunk


# # Init data sizes and parametes of the PXM of ID16A

# In[ ]:


n = 2048  # object size in each dimension
pad = 0
ndist = 4
lam = 0.1
show = False
ntheta = int(sys.argv[1])
st = int(sys.argv[2])
gpu = int(sys.argv[3])
# cp.cuda.Device(0).use()
# ntheta=30
# st = 600

cp.cuda.Device(gpu).use()
flg = f'{n}_{ntheta}_{pad}_{lam}_{st}'

detector_pixelsize = 3.03751e-6
energy = 33.35  # [keV] xray energy
wavelength = 1.2398419840550367e-09/energy  # [m] wave length
focusToDetectorDistance = 1.28  # [m]
sx0 = 1.286e-3
z1 = np.array([4.236e-3,4.3625e-3,4.86850e-3,5.91950e-3])[:ndist]-sx0
z2 = focusToDetectorDistance-z1
distances = (z1*z2)/focusToDetectorDistance
magnifications = focusToDetectorDistance/z1
voxelsize = detector_pixelsize/magnifications[0]*2048/n  # object voxel size

norm_magnifications = magnifications/magnifications[0]
# scaled propagation distances due to magnified probes
distances = distances*norm_magnifications**2

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

# sample size after demagnification
ne = int(np.ceil((n+2*pad)/norm_magnifications[-1]/32))*32  # make multiple of 32


path = f'/data/vnikitin/ESRF/ID16A/20240924/AtomiumS2/'
pfile = f'AtomiumS2_HT_007nm'
path_out = f'/data/vnikitin/ESRF/ID16A/20240924_rec/AtomiumS2/{pfile}/{flg}'
print(f'{voxelsize=}')
n0=n
ne0=ne
pad0=pad
voxelsize0=voxelsize
print(norm_magnifications)


# ## Read data

# In[ ]:


data00 = np.zeros([ntheta,ndist,2048,2048],dtype='float32')
ref00 = np.zeros([ndist,2048,2048],dtype='float32')
ref01 = np.zeros([ndist,2048,2048],dtype='float32')
dark00 = np.zeros([ndist,2048,2048],dtype='float32')
    
mmeans = np.zeros(8)

for k in range(ndist):
    for j in range(0,ntheta):
        jtheta=st+j
        fname = f'{path}{pfile}_{k+1}_/{pfile}_{k+1}_{jtheta:04}.edf'
        # print(fname)
        tmp = dxchange.read_edf(fname)[0]
        data00[j,k] = tmp
    
    tmp = np.zeros([2048,2048],dtype='float32')
    for l in range(20):
        fname=f'{path}{pfile}_{k+1}_/ref{l:04}_0000.edf'
        #print(fname)
        tmp += dxchange.read_edf(fname)[0]
    tmp/=20
    ref00[k] = tmp

    tmp = np.zeros([2048,2048],dtype='float32')
    for l in range(20):
        fname = f'{path}{pfile}_{k+1}_/ref{l:04}_1800.edf'
        #print(fname)
        tmp += dxchange.read_edf(fname)[0]
    tmp/=20
    ref01[k] = tmp

    tmp = np.zeros([2048,2048],dtype='float32')
    for l in range(20):
        fname = f'{path}{pfile}_{k+1}_/darkend{l:04}.edf'
        #print(fname)
        tmp += dxchange.read_edf(fname)[0]
    tmp/=20

    dark00[k] = tmp


# # Pre-processing

# In[ ]:


# Creating in advance
amps = np.load('amps.npy')[st:st+ntheta,:ndist]
amps_ref0 = np.load('amps_ref0.npy')[:ndist]
amps_ref1 = np.load('amps_ref1.npy')[:ndist]
shifts =np.load('shifts.npy')[st:st+ntheta,:ndist]*n/2048
shifts_ref =np.load('shifts_ref.npy')[:ndist]*n/2048


# In[ ]:


data = data00.copy()
ref0 = ref00.copy()
ref1 = ref01.copy()
dark = dark00.copy()

data-=dark
ref0-=dark
ref1-=dark

data[data<0]=0
ref0[ref0<0]=0
ref1[ref1<0]=0

# broken region
data[:,:,1320//3:1320//3+25//3,890//3:890//3+25//3] = data[:,:,1280//3:1280//3+25//3,890//3:890//3+25//3]
ref0[:,1320//3:1320//3+25//3,890//3:890//3+25//3] = ref0[:,1280//3:1280//3+25//3,890//3:890//3+25//3]
ref1[:,1320//3:1320//3+25//3,890//3:890//3+25//3] = ref1[:,1280//3:1280//3+25//3,890//3:890//3+25//3]

mshow(data[0,0],show)
for k in range(ndist):
    radius = 3
    threshold = 1.2
    for j in range(0,ntheta//30):
        st=j*30
        end=st+30
        data[st:end,k] = remove_outliers(data[st:end,k], radius, threshold)

ref0[:] = remove_outliers(ref0[:], radius, threshold)     
ref1[:] = remove_outliers(ref1[:], radius, threshold)     


data/=(amps[:,:,np.newaxis,np.newaxis])
ref0/=(amps_ref0[:,np.newaxis,np.newaxis])
ref1/=(amps_ref1[:,np.newaxis,np.newaxis])

print(np.mean(data[:,:,50:50+256,-50-256:-50],axis=(2,3)))
print(np.mean(ref0[:,50:50+256,-50-256:-50],axis=(1,2)))
print(np.mean(ref1[:,50:50+256,-50-256:-50],axis=(1,2)))

dark/=0.5*(amps_ref0[:,np.newaxis,np.newaxis]+amps_ref1[:,np.newaxis,np.newaxis])
v = np.linspace(0,1,1800)[:,np.newaxis,np.newaxis,np.newaxis].astype('float32')[st:st+ntheta]

ref = (1-v)*ref0+v*ref1
data/=np.mean(ref)
ref/=np.mean(ref)
for k in range(int(np.log2(2048//n))):
    data = (data[:,:,::2]+data[:,:,1::2])*0.5
    data = (data[:,:,:,::2]+data[:,:,:,1::2])*0.5
    ref = (ref[:,:,::2]+ref[:,:,1::2])*0.5
    ref = (ref[:,:,:,::2]+ref[:,:,:,1::2])*0.5    
    # dark = (dark[:,::2]+dark[:,1::2])*0.5
    # dark = (dark[:,:,::2]+dark[:,:,1::2])*0.5    


# In[ ]:


rdata = data/(ref+1e-6)

mshow_complex(data[0,0]+1j*rdata[0,0],show,vmin=0.05,vmax=2)

for k in range(ndist):
    r = M(rdata[:,k].astype('complex64'),1/norm_magnifications[k])
    rdata[:,k] = ST(r,shifts[:,k],mode='constant',constant_values=1).real

for k in range(ndist):
    mshow(rdata[0,0]-rdata[0,k],show)


# rdata = data/(ref+1e-6)

# mshow_complex(data[0,0]+1j*rdata[0,0],show,vmin=0.05,vmax=2)

# shifts_new = np.zeros([1800,4,2])
# iter = 0
# for st in range(0,1800,150):
#     flg = f'{n}_{150}_{pad}_{lam}_{st}'
#     shifts_new[st:st+150]=np.load(f'/data/vnikitin/ESRF/ID16A/20240924_rec/AtomiumS2/{pfile}/{flg}/crec_shift{flg}_{iter:03}.npy')


# shifts_new = shifts_new[:ntheta]
# print(shifts_new[0]-shifts)
# # print(shifts)
# rdata = np.pad(rdata,((0,0),(0,0),(pad,pad),(pad,pad)),'constant',constant_values=1)

# rdata_new = np.zeros([ntheta,ndist,ne,ne],dtype='float32')
# rdata_new1 = np.zeros([ntheta,ndist,ne,ne],dtype='float32')

# for k in range(ndist):
#     r = MTop(rdata[:,k].astype('complex64'),k)/norm_magnifications[k]**2
#     rdata_new[:,k] =ST2op(r,shifts[:,k]).real
#     rdata_new1[:,k] =ST2op(r,shifts_new[:,k]).real


# print(rdata_new.shape)
# rdata_new=rdata_new[:,:,ne//2-n//2:ne//2+n//2,ne//2-n//2:ne//2+n//2]
# rdata_new1=rdata_new1[:,:,ne//2-n//2:ne//2+n//2,ne//2-n//2:ne//2+n//2]

# for k in range(ndist):
#     diff = rdata_new[0,0]-rdata_new[0,k]
#     diff1 = rdata_new1[0,0]-rdata_new1[0,k]    
#     print(np.linalg.norm(diff),(np.linalg.norm(diff1)))
#     mshow_complex(diff+1j*diff1,show,vmax=0.5,vmin=-0.5)


# # Paganin reconstruction

# In[ ]:


rdatap = rdata.copy()
# distances should not be normalized
distances_pag = (distances/norm_magnifications**2)[:ndist]
rdatap = np.pad(rdatap,((0,0),(0,0),(ne//2-n//2,ne//2-n//2),(ne//2-n//2,ne//2-n//2)),'constant',constant_values=1)
rec_paganin = multiPaganin(rdatap, distances_pag, wavelength, voxelsize,100, 1e-12)
print(np.mean(rec_paganin[0,ne//2-n//2+n//16:ne//2-n//2+n//8,ne//2-n//2+n//16:ne//2-n//2+n//8]))
print(np.mean(rec_paganin[0,:n//8,:n//8]))
rec_paganin = np.exp(1j*rec_paganin)
mshow_polar(rec_paganin[-1],show)


# # Construct operators
# 

# In[ ]:


def L2op(psi,j):
    return G(psi, wavelength, voxelsize, distances[j],'symmetric')[:, pad:n+pad, pad:n+pad]

def LT2op(data,j):
    psir = np.pad(data, ((0, 0), (pad, pad), (pad, pad))).astype('complex64')
    psir = GT(psir, wavelength, voxelsize, distances[j],'symmetric')        
    return psir

def S2op(psi,shift):
    return S(psi, shift,mode='symmetric') 

def ST2op(data,shift):
    return ST(data,shift,mode='symmetric')
    
def Mop(psi, j):
    return M(psi, norm_magnifications[j]*ne/(n+2*pad), n+2*pad)                        

def MTop(psi,j):    
    return MT(psi, norm_magnifications[j]*ne/(n+2*pad), ne)        

def Cop(psi,crop):
    res = psi.copy()
    res[...,crop:res.shape[-1]-crop,crop:res.shape[-1]-crop]=0
    return res

def CTop(psi,crop):
    res = psi.copy()
    res[...,crop:res.shape[-1]-crop,crop:res.shape[-1]-crop]=0
    return res

def Cfop(psi,fcrop):
    return psi[:,fcrop:psi.shape[1]-fcrop,fcrop:psi.shape[2]-fcrop]

def CfTop(psi,fcrop):
    return np.pad(psi,((0,0),(fcrop,fcrop),(fcrop,fcrop)))

def Gop(psi):
    res = cp.zeros([2, *psi.shape], dtype='complex64')
    res[0, :, :, :-1] = psi[:, :, 1:]-psi[:, :, :-1]
    res[1, :, :-1, :] = psi[:, 1:, :]-psi[:, :-1, :]
    return res

def GTop( gr):
    res = cp.zeros(gr.shape[1:], dtype='complex64')
    res[:, :, 1:] = gr[0, :, :, 1:]-gr[0, :, :, :-1]
    res[:, :, 0] = gr[0, :, :, 0]
    res[:, 1:, :] += gr[1, :, 1:, :]-gr[1, :, :-1, :]
    res[:, 0, :] += gr[1, :, 0, :]
    return -res

arr1 = cp.random.random([chunk,ne,ne]).astype('complex64')
shifts_test = cp.random.random([chunk,2]).astype('float32')
arr2 = S2op(arr1,shifts_test)
arr3 = ST2op(arr2,shifts_test)
print(f'{np.sum(arr1*np.conj(arr3))}==\n{np.sum(arr2*np.conj(arr2))}')

arr1 = cp.random.random([chunk,ne,ne]).astype('complex64')
# arr1 = rec_paganin[:chunk].copy()#np.pad(np.exp(1j*recMultiPaganin[:chunk]),((0,0),(ne//2-n//2,ne//2-n//2),(ne//2-n//2,ne//2-n//2)),'edge')
arr2 = Mop(arr1,0)
arr3 = MTop(arr2,0)
print(f'{np.sum(arr1*np.conj(arr3))}==\n{np.sum(arr2*np.conj(arr2))}')

arr1 = cp.random.random([chunk,n+2*pad,n+2*pad]).astype('complex64')
arr2 = L2op(arr1,0)
arr3 = LT2op(arr2,0)
print(f'{np.sum(arr1*np.conj(arr3))}==\n{np.sum(arr2*np.conj(arr2))}')

arr1=arr2=arr3=[]


# # The fifth rule (Carlsson et al, 2024)

# ## Gradients

# \begin{align} &\nabla F=2 \left(L^*\left( L(M(q_0,\psi_0,\boldsymbol{x}_0))-\tilde D\right)\right)\\
# &\text{where } \tilde D = D \frac{L(M(q_0,\psi_0,\boldsymbol{x}_0))}{|L(M(q_0,\psi_0,\boldsymbol{x}_0))|}\end{align}
# 
# 

# In[ ]:


def gradientF(vars,d):
    (q,psi,x,crop,psifr) = (vars['prb'], vars['psi'], vars['shift'],vars['crop'],vars['psifr'])
    res = np.zeros([ntheta,ndist,n+2*pad,n+2*pad],dtype='complex64')
    for ichunk in range(0,int(np.ceil(ntheta/chunk))):
        st = ichunk*chunk
        end = min((ichunk+1)*chunk,ntheta)    
        psi_gpu = cp.asarray(psi[st:end])
        psi_gpu = CfTop(psi_gpu,crop[1])+psifr
        for j in range(ndist):
            x_gpu = cp.asarray(x[st:end,j])
            d_gpu = cp.asarray(d[st:end,j])            
            q_gpu = cp.asarray(q[j])            
            L2psi = L2op(q_gpu*Mop(S2op(psi_gpu,x_gpu),j),j)
            td = d_gpu*(L2psi/np.abs(L2psi))
            res[st:end,j] = cp.asnumpy(2*LT2op(L2psi - td,j))
    return res


# \begin{align*}\nabla_{\psi} G|_{(q_0,\psi_0,\boldsymbol{x}_0)}=S_{\boldsymbol{x}_{0}}^*\left(\overline{J(q_0)}\cdot \nabla F\right)
# \end{align*}
# 
# \begin{align*}
# \nabla_{q} G|_{(q_0,\psi_0,\boldsymbol{x}_0)}=J^*\left( \overline{S_{\boldsymbol{x}_{0}}(\psi_0)}\cdot \nabla F\right)
# \end{align*}
# 
# \begin{align*}
# \nabla_{\boldsymbol{x}_0} G|_{(q_0,\psi_0,\boldsymbol{x}_0)}=\textsf{Re}\Big(\big( \Big\langle \overline{q_0}\cdot \nabla F,   C(\mathcal{F}^{-1}(-2\pi i \xi_1 e^{ -2\pi i \boldsymbol{x}_{0,k}\cdot \boldsymbol{\xi}}\hat{\psi_0}))\Big\rangle,\Big\langle \overline{q_0}\cdot \nabla F,C(\mathcal{F}^{-1}(-2\pi i \xi_2 e^{ -2\pi i \boldsymbol{x}_{0,k}\cdot \boldsymbol{\xi}}\hat{\psi_0})) \Big\rangle\big)\Big)_{k=1}^K
# \end{align*}
# 

# In[ ]:


def gradientpsi(q,x,gradF,crop,j):
    # part 1 of gradpsi 
    t1 = Cfop(ST2op(MTop(np.conj(q)*gradF,j),x),crop[1])   
    return t1

def gradientq(x,gradF,psi,j):
    t1 = np.conj(Mop(S2op(psi,x),j))*gradF
    t1 = np.sum(t1,axis=0)
    return t1

def gradientx(q,x,gradF,psi,j):
    
    gradx = cp.zeros([x.shape[0],2],dtype='float32')    
    xi1 = cp.fft.fftfreq(2*ne).astype('float32')
    [xi2, xi1] = cp.meshgrid(xi1, xi1)
    psir = cp.pad(psi, ((0, 0), (ne//2, ne//2), (ne//2, ne//2)),'constant')
    
    xj = cp.asarray(x[:,:,np.newaxis,np.newaxis])
    pp = cp.exp(-2*cp.pi*1j*(xi1*xj[:, 0]+xi2*xj[:, 1]))                    
    t = cp.fft.ifft2(pp*xi1*cp.fft.fft2(psir))[:, ne//2:-ne//2, ne//2:-ne//2]      
    t = Mop(t, j)
    gradx[:,0] = -2*np.pi*imdot(gradF,q*t,axis=(1,2))    

    t = cp.fft.ifft2(pp*xi2*cp.fft.fft2(psir))[:, ne//2:-ne//2, ne//2:-ne//2]              
    t = Mop(t, j)
    gradx[:,1] = -2*np.pi*imdot(gradF,q*t,axis=(1,2))    

    return gradx

def gradients(vars,gradF):    
    (psi,q,x,rho,crop,psifr) = (vars['psi'], vars['prb'], vars['shift'], vars['rho'],vars['crop'],vars['psifr'])

    grads = {}
    grads['prb'] = np.zeros([ndist,n+2*pad,n+2*pad],dtype='complex64')
    grads['psi'] = np.zeros([ntheta,ne-2*crop[1],ne-2*crop[1]],dtype='complex64')
    grads['shift'] = np.zeros([ntheta,ndist,2],dtype='float32')
    
    for ichunk in range(0,int(np.ceil(ntheta/chunk))):
        st = ichunk*chunk
        end = min((ichunk+1)*chunk,ntheta)
        psi_gpu = cp.asarray(psi[st:end])    

        # part 2 of gradpsi
        grads['psi'][st:end] += cp.asnumpy(2*lam*GTop(CTop(Cop(Gop(psi_gpu),crop[0]),crop[0]))) 
        psi_gpu = CfTop(psi_gpu,crop[1])+psifr
        
        for j in range(ndist):
            x_gpu = cp.asarray(x[st:end,j])
            gradF_gpu = cp.asarray(gradF[st:end,j])
            q_gpu = cp.asarray(q[j])
            
            grads['prb'][j] += cp.asnumpy(rho[0]*gradientq(x_gpu,gradF_gpu,psi_gpu,j))
            grads['psi'][st:end] += cp.asnumpy(gradientpsi(q_gpu,x_gpu,gradF_gpu,crop,j))
            grads['shift'][st:end,j] = cp.asnumpy(rho[1]*gradientx(q_gpu,x_gpu,gradF_gpu,psi_gpu,j))
    
    grads['shift'][:,0] = 0

    return grads


# ## Hessians

# ##### $$\frac{1}{2}\mathcal{H}|_{x_0}(y,z)= \left\langle \mathbf{1}-d_{0}, \mathsf{Re}({L(y)}\overline{L(z)})\right\rangle+\left\langle d_{0},(\mathsf{Re} (\overline{l_0}\cdot L(y)))\cdot (\mathsf{Re} (\overline{l_0}\cdot L(z)))\right\rangle.$$
# ##### $$l_0=L(x_0)/|L(x_0)|$$
# ##### $$d_0=d/|L(x_0)|$$
# 

# In[ ]:


def hessianF(hpsi,hpsi1,hpsi2,data,j):
    Lpsi = L2op(hpsi,j)        
    Lpsi1 = L2op(hpsi1,j)
    Lpsi2 = L2op(hpsi2,j)    
    l0 = Lpsi/np.abs(Lpsi)
    d0 = data/np.abs(Lpsi)
    v1 = np.sum((1-d0)*reprod(Lpsi1,Lpsi2))
    v2 = np.sum(d0*reprod(l0,Lpsi1)*reprod(l0,Lpsi2))    
    return 2*(v1+v2)


# ### Functions for the shift operator:
# 
# \begin{align*}
# D T_c|_{{{z}_0}}(\Delta {z})=-2\pi iC\Big(\mathcal{F}^{-1}\big({\Delta z \cdot \xi}) e^{-2\pi i  {z}_0\cdot {\xi}}\hat{c}({\xi})\big)\Big)=-2\pi i C\Big(\mathcal{F}^{-1}\big((\Delta z_1 {\xi_1}+\Delta z_2 {\xi_2}) e^{-2\pi i  {z}_0\cdot {\xi}}\hat{c}({\xi})\big)\Big)
# \end{align*}
# 
# \begin{align*} D^2{T_c}|_{{{z}_0}}(\Delta{z},\Delta{w})
# =-4\pi^2C(\mathcal{F}^{-1}((\Delta{z_1}\Delta{w_1}\xi_1^2 + (\Delta{z_1}\Delta{w_2}+\Delta{z_2}\Delta{w_1})\xi_1\xi_2+\Delta{z_2}\Delta{w_2}\xi_2^2)\hat{c}))\end{align*}

# In[ ]:


def DT(psi,x,dx):
    psir = psi.copy()
    xi1 = cp.fft.fftfreq(2*ne).astype('float32')
    [xi2, xi1] = cp.meshgrid(xi1, xi1)
    
    xj = x[:,:,np.newaxis,np.newaxis]
    dxj = dx[:,:,np.newaxis,np.newaxis]

    psir = cp.pad(psir,((0,0),(ne//2,ne//2),(ne//2,ne//2)),'symmetric')
    pp = cp.exp(-2*cp.pi*1j*(xi1*xj[:, 0]+xi2*xj[:, 1]))    
    xiall = xi1*dxj[:,0]+xi2*dxj[:,1]

    psir = cp.fft.ifft2(pp*xiall*cp.fft.fft2(psir))   
    psir = psir[:,ne//2:-ne//2,ne//2:-ne//2]
    psir = -2*np.pi*1j*psir
    return psir

def D2T(psi,x,dx1,dx2):
    psir = psi.copy()
    
    xi1 = cp.fft.fftfreq(2*ne).astype('float32')
    [xi2, xi1] = cp.meshgrid(xi1, xi1)
    xj = x[:,:,np.newaxis,np.newaxis]
    dx1j = dx1[:,:,np.newaxis,np.newaxis]
    dx2j = dx2[:,:,np.newaxis,np.newaxis]

    pp = cp.exp(-2*cp.pi*1j*(xi1*xj[:, 0]+xi2*xj[:, 1]))    
    xiall = xi1**2*dx1j[:,0]*dx2j[:,0]+ \
            xi1*xi2*(dx1j[:,0]*dx2j[:,1]+dx1j[:,1]*dx2j[:,0])+ \
            xi2**2*dx1j[:,1]*dx2j[:,1]
    
    psir = cp.pad(psir,((0,0),(ne//2,ne//2),(ne//2,ne//2)),'symmetric')
    psir = cp.fft.ifft2(pp*xiall*cp.fft.fft2(psir))   
    psir = psir[:,ne//2:-ne//2,ne//2:-ne//2]
    psir = -4*np.pi**2*psir
    return psir


#  \begin{align*}
#  & DM|_{(q_0,\psi_0,{x}_0)}(\Delta q, \Delta \psi,\Delta{x})=L_1(q_0)\cdot M_j(T_{\Delta c}(z_0)+ DT_{c_0}|_{{{z}_0}}( \Delta {z}))+L_1(\Delta q)\cdot M_j(T_{c_0}({{z}_0}))
# \end{align*}
# 
# \begin{align*}
#  & D^2M|_{(q_0,\psi_0,{x}_0)}\big((\Delta q^{(1)},\Delta\psi^{(1)},\Delta{x}^{(1)}),(\Delta q^{(2)},\Delta\psi^{(2)},\Delta{x}^{(2)})\big)=\\&L_1(\Delta q^{(1)})\cdot M_j(T_{\Delta c^{(2)}}({{z}_0})
#  +DT_{c_0}|_{{{z}_0}}( \Delta {z}^{(2)})))+\\&
#  L_1(\Delta q^{(2)})\cdot M_j(T_{\Delta c^{(1)}}({{z}_0})
#  +DT_{c_0}|_{{{z}_0}}( \Delta {z}^{(1)})))\\&
#  +L_1(q_0)\cdot M_j(DT_{\Delta c^{(1)}}|_{{{z}_0}}( \Delta {z}^{(2)})
#  +DT_{\Delta c^{(2)}}|_{{{z}_0}}( \Delta {z}^{(1)})\\&
#  +\left(D^2{T_c}|_{{{z}_0}}(\Delta{z}^{(1)},\Delta{z}^{(2)})\right))
# \end{align*}

# In[ ]:


def DM(q,psi,x,dq,dpsi,dx,j):
    t1 = S2op(dpsi,x)+DT(psi,x,dx)    
    t2 = S2op(psi,x)
    return q*Mop(t1,j)+dq*Mop(t2,j)

def D2M(q,psi,x,dq1,dpsi1,dx1,dq2,dpsi2,dx2,j):
    t1 = DT(dpsi1,x,dx2)+DT(dpsi2,x,dx1)+D2T(psi,x,dx1,dx2)
    t2 = S2op(dpsi2,x)+DT(psi,x,dx2)
    t3 = S2op(dpsi1,x)+DT(psi,x,dx1)
    t1 = q*Mop(t1,j)
    t2 = dq1*Mop(t2,j)
    t3 = dq2*Mop(t3,j)
    return t1+t2+t3


# ## Form iterative scheme with alpha and beta

# $$x_{j+1}=x_j+\alpha_j s_j$$
# 
# where 
# $$
# s_{j+1}=-\nabla F|_{x_j}+\beta_j s_j, \quad s_0 = -\nabla F|_{x_j}
# $$
# 
# \begin{align*}
#               \alpha_j=\frac{\mathsf{Re}\langle \nabla F|_{x_j},s_j\rangle}{H|_{x_j}( {s_j},s_j)}.
#              \end{align*}
#              
# \begin{align*}
#     \beta_j=\frac{H(\nabla F|_{x_j},s_j)}{H|_{x_j}( {s_j},s_j)}.
# \end{align*}
# 

# In[ ]:


def calc_beta(vars,grads,etas,data,gradF):
    (q,psi,x,rho,crop,psifr) = (vars['prb'], vars['psi'], vars['shift'], vars['rho'], vars['crop'],vars['psifr'])
    (dq1,dpsi1,dx1) = (grads['prb']*rho[0], grads['psi'], grads['shift']*rho[1])
    (dq2,dpsi2,dx2) = (etas['prb']*rho[0], etas['psi'], etas['shift']*rho[1])
    
    top = 0
    bottom = 0
    for ichunk in range(0,int(np.ceil(ntheta/chunk))):
        st = ichunk*chunk
        end = min((ichunk+1)*chunk,ntheta)
        psi_gpu = cp.asarray(psi[st:end])        
        dpsi1_gpu = cp.asarray(dpsi1[st:end])
        dpsi2_gpu = cp.asarray(dpsi2[st:end])

        gpsi1 = Cop(Gop(dpsi1_gpu),crop[0])
        gpsi2 = Cop(Gop(dpsi2_gpu),crop[0])
        top += 2*lam*redot(gpsi1,gpsi2)
        bottom += 2*lam*redot(gpsi2,gpsi2)
        
        psi_gpu = CfTop(psi_gpu,crop[1])+psifr
        dpsi1_gpu = CfTop(dpsi1_gpu,crop[1])
        dpsi2_gpu = CfTop(dpsi2_gpu,crop[1])
        
        for j in range(ndist):                                
            data_gpu = cp.asarray(data[st:end,j])
            x_gpu = cp.asarray(x[st:end,j])
            dx1_gpu = cp.asarray(dx1[st:end,j])
            dx2_gpu = cp.asarray(dx2[st:end,j])
            gradF_gpu = cp.asarray(gradF[st:end,j])
            q_gpu = cp.asarray(q[j])
            dq1_gpu = cp.asarray(dq1[j])
            dq2_gpu = cp.asarray(dq2[j])

            L1psi = q_gpu*Mop(S2op(psi_gpu,x_gpu),j)        
            
            dv1 = DM(q_gpu,psi_gpu,x_gpu,dq1_gpu,dpsi1_gpu,dx1_gpu,j)    
            d2v1 = D2M(q_gpu,psi_gpu,x_gpu,dq1_gpu,dpsi1_gpu,dx1_gpu,dq2_gpu,dpsi2_gpu,dx2_gpu,j) 
            dv2 = DM(q_gpu,psi_gpu,x_gpu,dq2_gpu,dpsi2_gpu,dx2_gpu,j)    
            d2v2 = D2M(q_gpu,psi_gpu,x_gpu,dq2_gpu,dpsi2_gpu,dx2_gpu,dq2_gpu,dpsi2_gpu,dx2_gpu,j) 

            top += redot(gradF_gpu,d2v1)+hessianF(L1psi,dv1,dv2,data_gpu,j)                 
            bottom += redot(gradF_gpu,d2v2)+hessianF(L1psi,dv2,dv2,data_gpu,j)                     
    return float(top/bottom)

def _redot(a,b):
    res = 0    
    for ichunk in range(0,int(np.ceil(a.shape[0]/chunk))):
        st = ichunk*chunk
        end = min((ichunk+1)*chunk,a.shape[0])
        a_gpu = cp.asarray(a[st:end])
        b_gpu = cp.asarray(b[st:end])
        res+=redot(a_gpu,b_gpu)
    return res

def calc_alpha(vars,grads,etas,data,gradF):    
    (q,psi,x,rho,crop,psifr) = (vars['prb'], vars['psi'], vars['shift'], vars['rho'], vars['crop'],vars['psifr'])
    (dq1,dpsi1,dx1) = (grads['prb'], grads['psi'], grads['shift'])
    (dq2,dpsi2,dx2) = (etas['prb'], etas['psi'], etas['shift'])
    
    top = -_redot(dq1,dq2)-_redot(dpsi1,dpsi2)-_redot(dx1,dx2)        
    bottom = 0
    for ichunk in range(0,int(np.ceil(ntheta/chunk))):
        q = cp.asarray(q)
        dq2 = cp.asarray(dq2)
        st = ichunk*chunk
        end = min((ichunk+1)*chunk,ntheta)
        psi_gpu = cp.asarray(psi[st:end])
        dpsi2_gpu = cp.asarray(dpsi2[st:end])
        gpsi2 = Cop(Gop(dpsi2_gpu),crop[0])
        bottom += 2*lam*redot(gpsi2,gpsi2)
        
        psi_gpu = CfTop(psi_gpu,crop[1])+psifr
        dpsi2_gpu = CfTop(dpsi2_gpu,crop[1])
        
        for j in range(ndist):                    
            data_gpu = cp.asarray(data[st:end,j])
            x_gpu = cp.asarray(x[st:end,j])
            dx2_gpu = cp.asarray(dx2[st:end,j])            
            gradF_gpu = cp.asarray(gradF[st:end,j])
            q_gpu = cp.asarray(q[j])
            #dq1_gpu = cp.asarray(dq1[j])
            dq2_gpu = cp.asarray(dq2[j])
            
            L1psi = q_gpu*Mop(S2op(psi_gpu,x_gpu),j)                
            rdq2_gpu = dq2_gpu*rho[0]
            rdx2_gpu = dx2_gpu*rho[1]
            
            d2v2 = D2M(q_gpu,psi_gpu,x_gpu,rdq2_gpu,dpsi2_gpu,rdx2_gpu,rdq2_gpu,dpsi2_gpu,rdx2_gpu,j) 
            dv2 = DM(q_gpu,psi_gpu,x_gpu,rdq2_gpu,dpsi2_gpu,rdx2_gpu,j)
        
            bottom += redot(gradF_gpu,d2v2)+hessianF(L1psi,dv2,dv2,data_gpu,j)                     
        
    return float(top/bottom), float(top), float(bottom)


# ## Minimization functional
# 
# \begin{align*}
# \||L(q\cdot M(S_x(\psi,x)))|-d\|_2^2+\lambda\|C(\nabla \psi)\|_2^2
# \end{align*}

# In[ ]:


def minf(q,psi,x,crop,psifr,data):
    res = 0
    for ichunk in range(0,int(np.ceil(ntheta/chunk))):
        st = ichunk*chunk
        end = min((ichunk+1)*chunk,ntheta)
        psi_gpu = cp.asarray(psi[st:end])
        gpsi = Cop(Gop(psi_gpu),crop[0])
        res +=lam*np.linalg.norm(gpsi)**2
        
        psi_gpu = CfTop(psi_gpu,crop[1])+psifr                
        for j in range(ndist):               
            data_gpu = cp.asarray(data[st:end,j])
            x_gpu = cp.asarray(x[st:end,j])        
            q_gpu = cp.asarray(q[j])        
            L2psi = L2op(q_gpu*Mop(S2op(psi_gpu,x_gpu),j),j)            
            res += np.linalg.norm(np.abs(L2psi)-data_gpu)**2
    return float(res)


# ## debug functions

# In[ ]:


def plot_debug1(vars,etas,top,bottom,alpha,data):
    if show==False:
        return
    
    # check approximation with the gradient and hessian
    (q,psi,x,rho,crop,psifr) = (vars['prb'], vars['psi'], vars['shift'],vars['rho'],vars['crop'],vars['psifr'])
    (dq2,dpsi2,dx2) = (etas['prb'], etas['psi'], etas['shift'])
    npp = 3
    errt = np.zeros(npp*2)
    errt2 = np.zeros(npp*2)
    for k in range(0,npp*2):
        psit = psi+(alpha*k/(npp-1))*dpsi2        
        qt = q+(alpha*k/(npp-1))*rho[0]*dq2
        xt = x+(alpha*k/(npp-1))*rho[1]*dx2
        errt[k] = minf(qt,psit,xt,crop,psifr,data)    
        
    t = alpha*(np.arange(2*npp))/(npp-1)
    errt2 = minf(q,psi,x,crop,psifr,data) -top*t+0.5*bottom*t**2
    
    plt.plot((alpha*np.arange(2*npp)/(npp-1)),errt,'.',label='real')
    plt.plot((alpha*np.arange(2*npp)/(npp-1)),errt2,'.',label='approx')
    plt.legend()
    plt.show()

def plot_debug2(shifts,shifts_init):
    if show==False:
        return
    # show shift errors
    fig, axs = plt.subplots(1, 2, figsize=(9, 3))
    fig.suptitle('Error in shifts (for current level)')
    for k in range(ndist):        
        axs[0].plot(shifts_init[:,k,0]-shifts[:,k,0],'.')
        axs[1].plot(shifts_init[:,k,1]-shifts[:,k,1],'.')
    plt.show()

def vis_debug(vars,i):
    # visualization and write
    psie = CfTop(vars['psi'],vars['crop'][1])+vars['psifr'].get()
    mshow_polar(psie[0],show)
    mshow_polar(psie[0,ne//2-n//8:ne//2+n//8,ne//2+n//4:ne//2+n//2],show)
    mshow_polar(vars['prb'][0],show)
    dxchange.write_tiff(np.angle(psie),f'{path_out}/crec_psi_angle{flg}/{i:03}',overwrite=True)
    dxchange.write_tiff(np.angle(vars['prb']),f'{path_out}/crec_prb_angle{flg}/{i:03}',overwrite=True)
    dxchange.write_tiff(np.angle(psie[0]),f'{path_out}/crec_psio_angle{flg}/{i:03}',overwrite=True)
    dxchange.write_tiff(np.angle(vars['prb'][0]),f'{path_out}/crec_prbo_angle{flg}/{i:03}',overwrite=True)    
    dxchange.write_tiff(np.abs(psie),f'{path_out}/crec_code_abs{flg}/{i:03}',overwrite=True)
    dxchange.write_tiff(np.abs(vars['prb']),f'{path_out}/crec_prb_abs{flg}/{i:03}',overwrite=True)
    dxchange.write_tiff(np.abs(psie[0]),f'{path_out}/crec_psio_abs{flg}/{i:03}',overwrite=True)
    dxchange.write_tiff(np.abs(vars['prb'])[0],f'{path_out}/crec_prbo_abs{flg}/{i:03}',overwrite=True)
    np.save(f'{path_out}/crec_shift{flg}_{i:03}',vars['shift'])
    print(f'saved to {path_out}/crec_psi_angle{flg}/{i:03}',flush=True)
    
def err_debug(vars, grads, data):  
    # calculate error
    (q,psi,x,crop,psifr) = (vars['prb'], vars['psi'], vars['shift'],vars['crop'],vars['psifr'])  
    #(dq,dpsi,dx) = (grads['prb'], grads['psi'], grads['shift'])  
    err = minf(q,psi,x,crop,psifr,data)
    #print(f'gradient norms (prb, u, shift): {np.linalg.norm(dq):.2f}, {np.linalg.norm(du):.2f}, {np.linalg.norm(dx):.2f}',flush=True)                        
    return err


# # Main CG loop (fifth rule)

# In[ ]:


import time
def cg_holo(data, vars, pars):
    data = np.sqrt(data)    
    erra = np.zeros(pars['niter'])
    shifts_init = vars['shift'].copy()
    
    for i in range(pars['niter']):       

        if i % pars['vis_step'] == 0 and pars['vis_step'] != -1:
            vis_debug(vars, i)
            plot_debug2(vars['shift'],shifts_init)     

        tt = time.time()
        gradF = gradientF(vars,data)
        grads = gradients(vars,gradF)
        if i<16 and n==256:
            grads['shift'][:]=0
            
        if i==0:
            etas = {}
            etas['psi'] = -grads['psi']
            etas['prb'] = -grads['prb']
            etas['shift'] = -grads['shift']
        else:      
            beta = calc_beta(vars, grads, etas, data, gradF)
            etas['psi'] = -grads['psi'] + beta*etas['psi']
            etas['prb'] = -grads['prb'] + beta*etas['prb']
            etas['shift'] = -grads['shift'] + beta*etas['shift']    
            
        alpha,top,bottom = calc_alpha(vars, grads, etas, data, gradF) 
        if i % pars['vis_step'] == 0 and pars['vis_step'] != -1:
            plot_debug1(vars,etas,top,bottom,alpha,data)

        vars['psi'] += alpha*etas['psi']
        vars['prb'] += vars['rho'][0]*alpha*etas['prb']
        vars['shift'] += vars['rho'][1]*alpha*etas['shift']
        
        if i % pars['err_step'] == 0 and pars['err_step'] != -1:
            err = err_debug(vars, grads, data)    
            tt=time.time()-tt
            print(f'{i}) {vars['rho']} {alpha=:.5f}, {tt:.1f}s, {err=:1.5e}',flush=True)
            erra[i] = err
        

        # t={}        
        # # t[0]=np.linalg.norm(grads['prb'])
        # t[1]=np.linalg.norm(grads['shift'])
        # t[2]=np.linalg.norm(grads['psi'])
        
        # for k in range(1,2):
        #     if t[k]>2*t[2]:
        #         vars['rho'][k]/=2
        #     elif t[k]<t[2]/2:
        #         vars['rho'][k]*=2     

    return vars,erra

# vars = {}
# vars['psi'] = rec_paganin.copy()
# if one_probe:
#     vars['prb'] = cp.ones([1,n+2*pad,n+2*pad],dtype='complex64')
# else:
#     vars['prb'] = cp.ones([ndist,n+2*pad,n+2*pad],dtype='complex64')
# vars['shift'] = shifts.copy()
# vars['rho'] = [1,1]
# data_rec = data.copy()#cp.pad(cp.array(data),((0,0),(0,0),(pad,pad),(pad,pad)))
# pars = {'niter': 2049, 'err_step': 64, 'vis_step': 128}
# vars,erra = cg_holo(data_rec, vars, pars)    


# ## Hierarchical reconstruction

# In[ ]:


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

nlevels = 4
iters = np.array([2049,769,257,2570])
vis_steps = [64,64,32,16]
chunks = [150,60,15,5]


# init with most binned
n = n0//2**(nlevels-1)
pad =pad0//2**(nlevels-1)
ne = ne0//2**(nlevels-1)
voxelsize = voxelsize0*2**(nlevels-1)

vars = {}
vars['crop'] = np.array([ne//2-n//2,0])
rec = _downsample(rec_paganin,nlevels-1).astype('complex64')
vars['psi'] = rec[:,vars['crop'][1]:ne-vars['crop'][1],vars['crop'][1]:ne-vars['crop'][1]]
vars['psifr'] = cp.ones([ne,ne],dtype='complex64')
vars['psifr'][vars['crop'][1]:ne-vars['crop'][1],vars['crop'][1]:ne-vars['crop'][1]]=0


vars['shift'] = shifts/2**(nlevels-1)

vars['prb'] = np.ones([ndist,n+2*pad,n+2*pad],dtype='complex64')
vars['rho'] = [0.25,0.25]

for level in range(nlevels):
    print(f'{level=}')
    data_bin = _downsample(data,nlevels-level-1)    
    pars = {'niter': iters[level], 'err_step': vis_steps[level], 'vis_step': vis_steps[level]}

    chunk = chunks[level]
    chunking.global_chunk = chunk    
    vars,erra = cg_holo(data_bin, vars, pars)    
    if level==nlevels-1:
        break
    
    vars['psi'] = _fftupsample(vars['psi'],[1])
    vars['psi'] = _fftupsample(vars['psi'],[2])
    vars['prb'] = _fftupsample(vars['prb'],[1])
    vars['prb'] = _fftupsample(vars['prb'],[2])
    vars['shift']*=2
    vars['crop']*=2
    n*=2
    ne*=2
    pad*=2
    voxelsize/=2
    vars['psifr'] = cp.ones([ne,ne],dtype='complex64')
    vars['psifr'][vars['crop'][1]:ne-vars['crop'][1],vars['crop'][1]:ne-vars['crop'][1]]=0
    

