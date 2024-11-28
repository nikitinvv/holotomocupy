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
##!jupyter nbconvert --to script config_template.ipynb


# # Init data sizes and parametes of the PXM of ID16A

# In[ ]:


n = 2048  # object size in each dimension
pad = 768
npos= 16

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

# sample size after demagnification
ne = 2048//(2048//n)+2*pad#1024//(2048//n)#2*pad
show = False

flg = f'{n}'
path = f'/data/vnikitin/ESRF/ID16A/20240924/SiemensLH/SiemensLH_010nm_nfp_01/'
path_ref = f'/data/vnikitin/ESRF/ID16A/20240924/SiemensLH/SiemensLH_010nm_nfp_02/'
path_out = f'/data/vnikitin/ESRF/ID16A/20240924_rec/SiemensLH/SiemensLH_010nm_nfp_01_{n}_{pad}'


print(f'{voxelsize=}')


# ## Read data

# In[3]:


with h5py.File(f'{path}SiemensLH_010nm_nfp_010000.h5') as fid:
    data0 = fid['/entry_0000/measurement/data'][:npos].astype('float32')
with h5py.File(f'{path_ref}ref_0000.h5') as fid:
    ref0 = fid['/entry_0000/measurement/data'][:].astype('float32')
with h5py.File(f'{path}/dark_0000.h5') as fid:
    dark0 = fid['/entry_0000/measurement/data'][:].astype('float32')
with h5py.File(f'{path}SiemensLH_010nm_nfp_010000.h5','r') as fid:
    spz = np.array(str(np.array(str(np.array(fid['/entry_0000/instrument/PCIe/header/spz']))[1:]))[1:-1].split(' '),dtype='float32')*1e-6/voxelsize
    spy = np.array(str(np.array(str(np.array(fid['/entry_0000/instrument/PCIe/header/spy']))[1:]))[1:-1].split(' '),dtype='float32')*1e-6/voxelsize
data0 = data0[np.newaxis]

if show==True:
    plt.plot(spy,spz,'.')
    plt.axis('square')
    plt.show()
shifts_code0 = np.zeros([1,npos,2],dtype='float32')
shifts_code0[:,:,1] = spy[:npos]
shifts_code0[:,:,0] = -spz[:npos]


# In[4]:


import scipy.ndimage as ndimage
def remove_outliers(data, dezinger, dezinger_threshold):    
    res = data.copy()
    if (int(dezinger) > 0):
        w = int(dezinger)
        # print(data.shape)
        fdata = ndimage.median_filter(data, [1,w, w])
        print(np.sum(np.abs(data-fdata)>fdata*dezinger_threshold))
        res[:] = np.where(np.abs(data-fdata)>fdata*dezinger_threshold, fdata, data)
    return res


# In[5]:


data = data0.copy()
ref = ref0.copy()
dark = dark0.copy()
dark = np.mean(dark,axis=0)[np.newaxis]
ref = np.mean(ref,axis=0)[np.newaxis]
data-=dark
ref-=dark

data[data<0]=0
ref[ref<0]=0
# for k in range(data.shape[1]):
#     data[0,k,data[0,k]>ref[0]] = ref[0,data[0,k]>ref[0]]
data[:,:,1320//3:1320//3+25//3,890//3:890//3+25//3] = data[:,:,1280//3:1280//3+25//3,890//3:890//3+25//3]
ref[:,1320//3:1320//3+25//3,890//3:890//3+25//3] = ref[:,1280//3:1280//3+25//3,890//3:890//3+25//3]
for k in range(npos):
    radius = 3
    threshold = 0.8
    data[:,k] = remove_outliers(data[:,k], radius, threshold)
    
ref[:] = remove_outliers(ref[:], radius, threshold)     
data/=np.mean(ref)
dark/=np.mean(ref)
ref/=np.mean(ref)

data[np.isnan(data)] = 1
ref[np.isnan(ref)] = 1

for k in range(int(np.log2(2048//n))):
    data = (data[:,:,::2]+data[:,:,1::2])*0.5
    data = (data[:,:,:,::2]+data[:,:,:,1::2])*0.5
    ref = (ref[:,::2]+ref[:,1::2])*0.5
    ref = (ref[:,:,::2]+ref[:,:,1::2])*0.5    
    dark = (dark[:,::2]+dark[:,1::2])*0.5
    dark = (dark[:,:,::2]+dark[:,:,1::2])*0.5  

rdata = data/(ref+1e-11)

mshow_complex(data[0,0]+1j*rdata[0,0],show)
mshow_complex(ref[0]+1j*dark[0],show)


# # Construct operators
# 

# In[6]:


def Lop(psi):
    data = cp.zeros([*psi.shape[:2], ne, ne], dtype='complex64')
    for i in range(psi.shape[1]):
        psir = cp.array(psi[:,i])               
        psir = G(psir, wavelength, voxelsize, distances[i],'symmetric')
        data[:, i] = psir
    return data

def LTop(data):
    psi = cp.zeros([*data.shape[:2], ne, ne], dtype='complex64')
    for j in range(data.shape[1]):
        datar = cp.array(data[:, j]).astype('complex64')        
        datar = GT(datar, wavelength, voxelsize, distances[j],'symmetric')        
        psi[:,j] = datar
    return psi

def Sop(psi,shifts):
    data = cp.zeros([1, npos, ne, ne], dtype='complex64')
    psi = cp.array(psi)
    for j in range(npos):
        psir = psi.copy()
        shiftsr = cp.array(shifts[:, j])
        psir = S(psir, shiftsr)
        data[:,j] = psir
    return data

def STop(data,shifts):
    psi = cp.zeros([1, ne, ne], dtype='complex64')

    for j in range(npos):
        datar = cp.array(data[:,j])
        shiftsr = cp.array(shifts[:, j])
        psi += ST(datar,shiftsr)
    return psi

def Kop(e):
    res = e.copy()
    res[:,:,pad:ne-pad,pad:ne-pad] = 0
    return res

def KTop(e):
    res = e.copy()
    res[:,:,pad:ne-pad,pad:ne-pad] = 0
    return res

# adjoint tests
tmp = cp.array(data).copy()
arr1 = cp.pad(tmp[:,0],((0,0),(ne//2-n//2,ne//2-n//2),(ne//2-n//2,ne//2-n//2)),'symmetric')     
prb1 = cp.ones([1,n+2*pad,n+2*pad],dtype='complex64')
shifts = cp.array(shifts_code0)
arr2 = Sop(arr1,shifts)
arr3 = STop(arr2,shifts)

arr4 = Lop(arr2)
arr5 = LTop(arr4)

e = cp.random.random([1,npos,ne,ne]).astype('float32')
arr6 = Kop(e)
arr7 = KTop(arr6)


print(f'{np.sum(arr1*np.conj(arr3))}==\n{np.sum(arr2*np.conj(arr2))}')
print(f'{np.sum(arr2*np.conj(arr5))}==\n{np.sum(arr4*np.conj(arr4))}')
print(f'{np.sum(e*np.conj(arr7))}==\n{np.sum(arr6*np.conj(arr6))}')


# ## Reconstruction with the CG (Carlsson) with Hessians

# ## Gradients

# #### $$\nabla F=2 \left(L^*\left( L(M(q_0,\psi_0,\boldsymbol{x}_0))-\tilde D\right)\right).$$
# #### where $$\tilde D = (D+K(e)) \frac{L(M(q_0,\psi_0,\boldsymbol{x}_0))}{|L(M(q_0,\psi_0,\boldsymbol{x}_0))|}$$
# #### $$\nabla_e F=2 \left(K^*\left( |L(M(q_0,\psi_0,\boldsymbol{x}_0))|-(D+K(e_0))\right)\right).$$
# 
# 

# In[7]:


def gradientF(vars,d):
    (psi,q,x,e) = (vars['psi'], vars['prb'], vars['shift'],vars['e'])
    Lpsi = Lop(Sop(psi,x)*q)
    td = (d+Kop(e))*(Lpsi/np.abs(Lpsi))
    res = 2*LTop(Lpsi - td)
    return res



# ##### $$\nabla_{\psi} G|_{(q_0,\psi_0,\boldsymbol{x}_0)}=S_{\boldsymbol{x}_{0}}^*\left(\overline{J(q_0)}\cdot \nabla F\right).$$
# ##### $$\nabla_{q} G|_{(q_0,\psi_0,\boldsymbol{x}_0)}=J^*\left( \overline{S_{\boldsymbol{x}_{0}}(\psi_0)}\cdot \nabla F\right).$$
# ##### $$\nabla_{\boldsymbol{x}_0} G|_{(q_0,\psi_0,\boldsymbol{x}_0)}=\textsf{Re}\Big(\big( \Big\langle \overline{q_0}\cdot \nabla F,   C(\mathcal{F}^{-1}(-2\pi i \xi_1 e^{ -2\pi i \boldsymbol{x}_{0,k}\cdot \boldsymbol{\xi}}\hat{\psi_0}))\Big\rangle,\Big\langle \overline{q_0}\cdot \nabla F,C(\mathcal{F}^{-1}(-2\pi i \xi_2 e^{ -2\pi i \boldsymbol{x}_{0,k}\cdot \boldsymbol{\xi}}\hat{\psi_0})) \Big\rangle\big)\Big)_{k=1}^K. $$
# 
# 
# 

# In[8]:


def gradientpsi(q,x,gradF):
    return STop(np.conj(q)*gradF,x)

def gradientq(psi,x,gradF):
    return np.sum(np.conj(Sop(psi,x))*gradF,axis=1)

# def gradientx(psi,q,x,gradF):
#     xi1 = cp.fft.fftfreq(2*psi.shape[-1]).astype('float32')    
#     [xi2, xi1] = cp.meshgrid(xi1, xi1)  
#     tksi1 = Twop_(psi,x,-2*cp.pi*1j*xi1)
#     tksi2 = Twop_(psi,x,-2*cp.pi*1j*xi2)    
#     gradx = cp.zeros([1,npos,2],dtype='float32')
#     tmp = np.conj(q)*gradF
#     gradx[:,:,0] = redot(tmp,tksi1,axis=(2,3))
#     gradx[:,:,1] = redot(tmp,tksi2,axis=(2,3))
#     return gradx

def gradientx(psi,q,x,gradF):
    
    gradx = cp.zeros([1,npos,2],dtype='float32')    
    xi1 = cp.fft.fftfreq(ne).astype('float32')
    [xi2, xi1] = cp.meshgrid(xi1, xi1)
    psir = psi.copy()#cp.pad(eRu, ((0, 0), (ne//2, ne//2), (ne//2, ne//2)))
    for j in range(npos):        
        xj = cp.array(x[:,j,:,np.newaxis,np.newaxis])
        pp = cp.exp(-2*cp.pi*1j*(xi1*xj[:, 0]+xi2*xj[:, 1]))      
        #print(pp.shape,xi1.shape,psir.shape)              
        t = cp.fft.ifft2(pp*xi1*cp.fft.fft2(psir))
        gradx[:,j,0] = -2*np.pi*imdot(gradF[:,j],q*t,axis=(1,2))    
    
    for j in range(npos):        
        xj = cp.array(x[:,j,:,np.newaxis,np.newaxis])
        pp = cp.exp(-2*cp.pi*1j*(xi1*xj[:, 0]+xi2*xj[:, 1]))                    
        
        t = cp.fft.ifft2(pp*xi2*cp.fft.fft2(psir))
        gradx[:,j,1] = -2*np.pi*imdot(gradF[:,j],q*t,axis=(1,2))    

    return gradx

def gradiente(psi,q,x,e,d):
    Lpsi = Lop(Sop(psi,x)*q)
    res = -2*KTop(np.abs(Lpsi)-(d+Kop(e)))
    return res
    
def gradients(vars,d,gradF):
    (psi,q,x,e,rho) = (vars['psi'], vars['prb'], vars['shift'],vars['e'],vars['rho'])
    grads = {}
    grads['psi'] = rho[0]*gradientpsi(q,x,gradF)
    grads['prb'] = rho[1]*gradientq(psi,x,gradF)
    grads['shift'] = rho[2]*gradientx(psi,q,x,gradF)
    grads['e'] = rho[3]*gradiente(psi,q,x,e,d)
    return grads



# ##### $$\frac{1}{2}\mathcal{H}|_{x_0}(y,z)= \left\langle \mathbf{1}-d_{0}, \mathsf{Re}({L(y)}\overline{L(z)})\right\rangle+\left\langle d_{0},(\mathsf{Re} (\overline{l_0}\cdot L(y)))\cdot (\mathsf{Re} (\overline{l_0}\cdot L(z))) \right\rangle$$
# ##### $$+\mathsf{Re}\langle K(\Delta_{e_1}),K(\Delta_{e_2})\rangle-\langle K(\Delta_{e_1}),\mathsf{Re}(\overline{l_0}\cdot L(z))\rangle-\langle K(\Delta_{e_2}),\mathsf{Re}(\overline{l_0}\cdot L(y))\rangle$$
# ##### $$l_0=L(x_0)/|L(x_0)|$$
# ##### $$d_0=(d+K(e))/|L(x_0)|$$
# 

# In[9]:


def hessianF(hpsi,hpsi1,hpsi2,e,de1,de2,data):
    Lpsi = Lop(hpsi)        
    Lpsi1 = Lop(hpsi1)
    Lpsi2 = Lop(hpsi2)    
    l0 = Lpsi/np.abs(Lpsi)
    d0 = (data+Kop(e))/np.abs(Lpsi)
    v1 = np.sum((1-d0)*reprod(Lpsi1,Lpsi2))
    v2 = np.sum(d0*reprod(l0,Lpsi1)*reprod(l0,Lpsi2))        
    v3 = np.sum(Kop(de1)*Kop(de2))-np.sum(Kop(de1)*reprod(l0,Lpsi2))-np.sum(Kop(de2)*reprod(l0,Lpsi1))
    
    return 2*(v1+v2+v3)


# ##### $D T_c|_{{{z}_0}}(\Delta {z})=-2\pi iC\Big(\mathcal{F}^{-1}\big({\Delta z \cdot \xi}) e^{-2\pi i  {z}_0\cdot {\xi}}\hat{c}({\xi})\big)\Big)=-2\pi i C\Big(\mathcal{F}^{-1}\big((\Delta z_1 {\xi_1}+\Delta z_2 {\xi_2}) e^{-2\pi i  {z}_0\cdot {\xi}}\hat{c}({\xi})\big)\Big)$
# ##### $ D^2{T_c}|_{{{z}_0}}(\Delta{z},\Delta{w})=-4\pi^2C(\mathcal{F}^{-1}((\Delta{z}\cdot\xi)(\Delta{w}\cdot\xi)e^{-2\pi i  {z}_0\cdot {\xi}}\hat{c}))$
# ##### $=-4\pi^2C(\mathcal{F}^{-1}((\Delta{z_1}\Delta{w_1}\xi_1^2 + (\Delta{z_1}\Delta{w_2}+\Delta{z_2}\Delta{w_1})\xi_1\xi_2+\Delta{z_2}\Delta{w_2}\xi_2^2)\hat{c}))$

# In[10]:


def DT(psi,x,dx):
    res = cp.zeros([1,npos,ne,ne],dtype='complex64')
    xi1 = cp.fft.fftfreq(ne).astype('float32')
    [xi2, xi1] = cp.meshgrid(xi1, xi1)
    for j in range(npos):
        psir = psi.copy()#cp.pad(psi, ((0, 0), (ne//2, ne//2), (ne//2, ne//2)))
        xj = cp.array(x[:,j,:,np.newaxis,np.newaxis])
        dxj = cp.array(dx[:,j,:,np.newaxis,np.newaxis])

        pp = cp.exp(-2*cp.pi*1j*(xi1*xj[:, 0]+xi2*xj[:, 1]))    
        xiall = xi1*dxj[:,0]+xi2*dxj[:,1]

        psir = cp.fft.ifft2(pp*xiall*cp.fft.fft2(psir))   

        res[:,j] = -2*np.pi*1j*psir#[:, ne//2:-ne//2, ne//2:-ne//2]       
    return res


def D2T(psi,x,dx1,dx2):
    res = cp.zeros([1,npos,ne,ne],dtype='complex64')
    xi1 = cp.fft.fftfreq(ne).astype('float32')
    [xi2, xi1] = cp.meshgrid(xi1, xi1)
    for j in range(npos):
        psir = psi.copy()#cp.pad(psi, ((0, 0), (ne//2, ne//2), (ne//2, ne//2)))
        xj = cp.array(x[:,j,:,np.newaxis,np.newaxis])
        dx1j = cp.array(dx1[:,j,:,np.newaxis,np.newaxis])
        dx2j = cp.array(dx2[:,j,:,np.newaxis,np.newaxis])

        pp = cp.exp(-2*cp.pi*1j*(xi1*xj[:, 0]+xi2*xj[:, 1]))    
        xiall = xi1**2*dx1j[:,0]*dx2j[:,0]+ \
                xi1*xi2*(dx1j[:,0]*dx2j[:,1]+dx1j[:,1]*dx2j[:,0])+ \
                xi2**2*dx1j[:,1]*dx2j[:,1]

        psir = cp.fft.ifft2(pp*xiall*cp.fft.fft2(psir))   

        res[:,j] = -4*np.pi**2*psir#[:,ne//2:-ne//2, ne//2:-ne//2]       
    return res


# #### $$ DM|_{(q_0,\psi_0,\boldsymbol{x})}(\Delta q, \Delta \psi,\Delta\boldsymbol{x})=$$
# #### $$ \Big(\Delta q\cdot T_{\psi_0}({\boldsymbol{x}_{0,k}})+ q_0\cdot \big(T_{\Delta \psi}({\boldsymbol{x}_{0,k}})+  DT_{\psi_0}|_{{\boldsymbol{x}_{0,k}}}( \Delta \boldsymbol{x}_k)\big) \Big)_{k=1}^K=$$
# #### $$ J(\Delta q)\cdot S_{\boldsymbol{x}_{0,k}}(\psi_0)+ J(q_0)\cdot S_{\boldsymbol{x}_{0}}{(\Delta \psi)}+  \Big(q_0\cdot DT_{\psi_0}|_{{\boldsymbol{x}_{0,k}}}( \Delta \boldsymbol{x}_k) \Big)_{k=1}^K$$
# 

# In[11]:


def DM(psi,q,x,dpsi,dq,dx):
    res = dq*Sop(psi,x)+q*(Sop(dpsi,x)+DT(psi,x,dx))   
    return res


# ##### $$ D^2M|_{(q_0,\psi_0,\boldsymbol{x})}\big((\Delta q^{(1)}, \Delta \psi^{(1)},\Delta\boldsymbol{x}^{(1)}),(\Delta q^{(2)}, \Delta \psi^{(2)},\Delta\boldsymbol{x}^{(2)})\big)= $$
# ##### $$\Big( q_0\cdot DT_{\Delta\psi^{(1)}}|_{{\boldsymbol{x}_{0,k}}}( \Delta \boldsymbol{x}^{(2)})+q_0\cdot DT_{\Delta\psi^{(2)}}|_{{\boldsymbol{x}_{0,k}}}( \Delta \boldsymbol{x}^{(1)})+ q_0\cdot D^2{T_\psi}|_{{\boldsymbol{x}_0}}(\Delta\boldsymbol{x}^{(1)},\Delta\boldsymbol{x}^{(2)})+$$
# ##### $$\Delta q^{(1)}\cdot T_{\Delta \psi^{(2)}}({\boldsymbol{x}_{0,k}})+\Delta q^{(2)}\cdot T_{\Delta \psi^{(1)}}({\boldsymbol{x}_{0,k}})+ $$
# ##### $$\Delta q^{(1)}\cdot DT_{\psi_0}|_{{\boldsymbol{x}_{0,k}}}( \Delta \boldsymbol{x}^{(2)})+\Delta q^{(2)}\cdot DT_{\psi_0}|_{{\boldsymbol{x}_{0,k}}}( \Delta \boldsymbol{x}^{(1)})\Big)_{k=1}^K.$$
# 

# In[12]:


def D2M(psi,q,x,dpsi1,dq1,dx1,dpsi2,dq2,dx2):    
    res =  q*DT(dpsi1,x,dx2) + q*DT(dpsi2,x,dx1) + q*D2T(psi,x,dx1,dx2)  
    res += dq1*Sop(dpsi2,x) + dq2*Sop(dpsi1,x) 
    res += dq1*DT(psi,x,dx2) + dq2*DT(psi,x,dx1)
    return res


# ##### $$\mathcal{H}^G|_{ (q_0,\psi_0,\boldsymbol{x}_0)}\Big((\Delta q^{(1)},\Delta \psi^{(1)},\Delta \boldsymbol{x}^{(1)}),(\Delta q^{(2)},\Delta \psi^{(2)},\Delta \boldsymbol{x}^{(2)})\Big)=$$
# ##### $$\Big\langle \nabla F|_{M(q_0,\psi_0,\boldsymbol{x}_0)}, D^2M|_{(q_0,\psi_0,\boldsymbol{x}_0)}\Big((\Delta q^{(1)},\Delta \psi^{(1)},\Delta \boldsymbol{x}^{(1)}),(\Delta q^{(2)},\Delta \psi^{(2)},\Delta \boldsymbol{x}^{(2)})\Big)\Big\rangle +$$
# ##### $$\mathcal{H}^F|_{M(q_0,\psi_0,\boldsymbol{x}_0)}\Big(DM|_{(q_0,\psi_0,\boldsymbol{x}_0)}(\Delta q^{(1)},\Delta \psi^{(1)},\Delta \boldsymbol{x}^{(1)}),DM|_{(q_0,\psi_0,\boldsymbol{x}_0)}(\Delta q^{(2)},\Delta \psi^{(2)},\Delta \boldsymbol{x}^{(2)})\Big).$$

# In[13]:


def hessian2(psi,q,x,e,dpsi1,dq1,dx1,dpsi2,dq2,dx2,de1,de2,d,gradF):
    d2m = D2M(psi,q,x,dpsi1,dq1,dx1,dpsi2,dq2,dx2)
    dm1 = DM(psi,q,x,dpsi1,dq1,dx1)
    dm2 = DM(psi,q,x,dpsi2,dq2,dx2)
    sq = Sop(psi,x)*q    
        
    return redot(gradF,d2m)+hessianF(sq, dm1,dm2,e,de1,de2,d)


# In[14]:


def calc_beta(vars,grads,etas,d,gradF):
    (psi,q,x,e,rho) = (vars['psi'], vars['prb'], vars['shift'],vars['e'],vars['rho'])
    (dpsi1,dq1,dx1,de1) = (grads['psi']*rho[0], grads['prb']*rho[1], grads['shift']*rho[2], grads['e']*rho[3])
    (dpsi2,dq2,dx2,de2) = (etas['psi']*rho[0], etas['prb']*rho[1], etas['shift']*rho[2],etas['e']*rho[3])
    
    dm1 = DM(psi,q,x,dpsi1,dq1,dx1)
    dm2 = DM(psi,q,x,dpsi2,dq2,dx2)
    d2m1 = D2M(psi,q,x,dpsi1,dq1,dx1,dpsi2,dq2,dx2)
    d2m2 = D2M(psi,q,x,dpsi2,dq2,dx2,dpsi2,dq2,dx2)
    sq = Sop(psi,x)*q

    top = redot(gradF,d2m1)        
    top += hessianF(sq, dm1, dm2,e,de1,de2,d)    
    bottom = redot(gradF,d2m2)    
    bottom += hessianF(sq, dm2, dm2,e,de2,de2,d)
    return top/bottom

def calc_alpha(vars,grads,etas,d,gradF):    
    (psi,q,x,e,rho) = (vars['psi'], vars['prb'], vars['shift'],vars['e'],vars['rho'])
    (dpsi1,dq1,dx1,de1) = (grads['psi'], grads['prb'], grads['shift'], grads['e'])
    (dpsi2,dq2,dx2,de2) = (etas['psi'], etas['prb'], etas['shift'],etas['e'])        
    top = -redot(dpsi1,dpsi2)-redot(dq1,dq2)-redot(dx1,dx2)-redot(de1,de2)
    
    (dpsi2,dq2,dx2,de2) = (etas['psi']*rho[0], etas['prb']*rho[1], etas['shift']*rho[2],etas['e']*rho[3])
    dm2 = DM(psi,q,x,dpsi2,dq2,dx2)
    d2m2 = D2M(psi,q,x,dpsi2,dq2,dx2,dpsi2,dq2,dx2)
    sq = Sop(psi,x)*q
    bottom = redot(gradF,d2m2)+hessianF(sq, dm2, dm2,e,de2,de2,d)
    return top/bottom, top, bottom


# ### Initial guess for reconstruction (Paganin)

# In[15]:


def rec_init(rdata,shifts):
    recMultiPaganin = cp.zeros([1,npos,ne,ne],dtype='float32')
    recMultiPaganinr = cp.zeros([1,npos,ne,ne],dtype='float32')# to compensate for overlap
    for j in range(0,npos):
        rdatar = cp.array(rdata[:,j:j+1])
        r = multiPaganin(rdatar,
                            distances[j:j+1], wavelength, voxelsize,  24.05, 1.2e-2)    
        rr = r*0+1 # to compensate for overlap
        r = cp.pad(r,((0,0), (ne//2-n//2,ne//2-n//2), (ne//2-n//2,ne//2-n//2)))   
        rr = cp.pad(rr,((0,0), (ne//2-n//2,ne//2-n//2), (ne//2-n//2,ne//2-n//2)))   
        shiftsr = cp.array(shifts[:,j])
        recMultiPaganin[:,j] = ST(r,shiftsr).real
        recMultiPaganinr[:,j] = ST(rr,shiftsr).real
        
    recMultiPaganin = np.sum(recMultiPaganin,axis=1)
    recMultiPaganinr = np.sum(recMultiPaganinr,axis=1)

    # avoid division by 0
    recMultiPaganinr[np.abs(recMultiPaganinr)<5e-2] = 1

    # compensate for overlap
    recMultiPaganin /= recMultiPaganinr
    v = cp.ones(ne,dtype='float32')
    v[:pad] = np.sin(cp.linspace(0,1,pad)*np.pi/2)
    v[ne-pad:] = np.cos(cp.linspace(0,1,pad)*np.pi/2)
    v = np.outer(v,v)
    recMultiPaganin*=v
    recMultiPaganin = np.exp(1j*recMultiPaganin)

    return recMultiPaganin

rec_paganin = rec_init(rdata,shifts)
mshow_polar(rec_paganin[0],show)
mshow_polar(rec_paganin[0,ne//2-128:ne//2+128,ne//2-128:ne//2+128],show)


# ## debug functions

# In[16]:


def plot_debug2(vars,etas,top,bottom,alpha,data):
    if show==False:
        return
    (psi,q,x,e,rho) = (vars['psi'], vars['prb'], vars['shift'], vars['e'],vars['rho'])
    (dpsi2,dq2,dx2,de2) = (etas['psi'],etas['prb'],etas['shift'],etas['e'])
    npp = 17
    errt = cp.zeros(npp*2)
    errt2 = cp.zeros(npp*2)
    for k in range(0,npp*2):
        psit = psi+(alpha*k/(npp-1))*rho[0]*dpsi2
        qt = q+(alpha*k/(npp-1))*rho[1]*dq2
        xt = x+(alpha*k/(npp-1))*rho[2]*dx2
        et = e+(alpha*k/(npp-1))*rho[3]*de2
        fpsit = np.abs(Lop(Sop(psit,xt)*qt))-(data+Kop(et))
        errt[k] = np.linalg.norm(fpsit)**2
        
    t = alpha*(cp.arange(2*npp))/(npp-1)
    tmp = np.abs(Lop(Sop(psi,x)*q))-(data+Kop(e))
    errt2 = np.linalg.norm(tmp)**2-top*t+0.5*bottom*t**2
    
    plt.plot(alpha.get()*cp.arange(2*npp).get()/(npp-1),errt.get(),'.')
    plt.plot(alpha.get()*cp.arange(2*npp).get()/(npp-1),errt2.get(),'.')
    plt.show()

def plot_debug3(shifts,shifts_init):
    if show==False:
        return
    plt.plot(shifts_init[0,:,0].get()-(shifts[0,:,0].get()),'r.')
    plt.plot(shifts_init[0,:,1].get()-(shifts[0,:,1].get()),'b.')
    plt.show()

def vis_debug(vars,data,i):
    mshow_polar(vars['psi'][0],show)
    mshow_polar(vars['psi'][0,ne//2-n//8:ne//2+n//8,ne//2+n//4:ne//2+n//2],show)
    mshow_polar(vars['prb'][0],show)
    tmp=vars['e'][0,0].copy()+data[0,0]
    mshow(tmp,show)
    dxchange.write_tiff(np.angle(vars['psi'][0]).get(),f'{path_out}/crec_code_angle{flg}/{i:03}',overwrite=True)
    dxchange.write_tiff(np.angle(vars['prb'][0]).get(),f'{path_out}/crec_prb_angle{flg}/{i:03}',overwrite=True)
    dxchange.write_tiff(np.abs(vars['psi'][0]).get(),f'{path_out}/crec_code_abs{flg}/{i:03}',overwrite=True)
    dxchange.write_tiff(np.abs(vars['prb'][0]).get(),f'{path_out}/crec_prb_abs{flg}/{i:03}',overwrite=True)
    dxchange.write_tiff(vars['e'][0].get(),f'{path_out}/crec_e{flg}/{i:03}',overwrite=True)
    np.save(f'{path_out}/crec_shift{flg}_{i:03}',vars['shift'])

    
def err_debug(vars, grads, data):    
    (psi,q,x,e) = (vars['psi'], vars['prb'], vars['shift'], vars['e'])
    tmp = np.abs(Lop(Sop(psi,x)*q))-(data+Kop(e))
    err = np.linalg.norm(tmp)**2
    print(f'gradient norms (psi, prb, shift,e): {np.linalg.norm(grads['psi']):.2f}, {np.linalg.norm(grads['prb']):.2f}, {np.linalg.norm(grads['shift']):.2f}, {np.linalg.norm(grads['e']):.2f}')                        
    return err


# # Main CG loop (fifth rule)

# In[17]:


iter = 288
prb_abs = dxchange.read_tiff(f'/data2/vnikitin/nfp_codes_correct/crec_prb_abs2048_-0.01775full0.3/{iter}.tiff')
prb_angle = dxchange.read_tiff(f'/data2/vnikitin/nfp_codes_correct/crec_prb_angle2048_-0.01775full0.3/{iter}.tiff')
prb = prb_abs*np.exp(1j*prb_angle)[np.newaxis]
prb = prb[:,prb.shape[1]//2-n//2-pad:prb.shape[1]//2+n//2+pad,prb.shape[2]//2-n//2-pad:prb.shape[2]//2+n//2+pad]
# prb = np.pad(prb,((0,0),(n//2+pad-prb.shape[1]//2,(n//2+pad-prb.shape[1]//2)),(n//2+pad-prb.shape[1]//2,(n//2+pad-prb.shape[1]//2))),'edge')

mshow_polar(prb[0],show)
z1c = -17.75e-3
distances2 = (z1-z1c)/(z1c/z1)#magnifications2
prb = G(prb,wavelength,voxelsize,distances2[0],'symmetric')
mshow_polar(prb[0],show)



# In[ ]:


def cg_holo(data, vars, pars):

    data = np.sqrt(data)    
    vis_debug(vars, data, 0)
    erra = cp.zeros(pars['niter'])
    alphaa = cp.zeros(pars['niter'])    
    shifts_init = vars['shift'].copy()
    for i in range(pars['niter']):           
        gradF = gradientF(vars,data)        
        grads = gradients(vars,data,gradF)
        if i==0:
            etas = {}
            etas['psi'] = -grads['psi']
            etas['prb'] = -grads['prb']
            etas['shift'] = -grads['shift']
            etas['e'] = -grads['e']
        else:      
            beta = calc_beta(vars, grads, etas, data, gradF)
            etas['psi'] = -grads['psi'] + beta*etas['psi']
            etas['prb'] = -grads['prb'] + beta*etas['prb']
            etas['shift'] = -grads['shift'] + beta*etas['shift']
            etas['e'] = -grads['e'] + beta*etas['e']

        alpha,top,bottom = calc_alpha(vars, grads, etas, data, gradF) 
        if i % pars['vis_step'] == 0:
            plot_debug2(vars,etas,top,bottom,alpha,data)

        vars['psi'] += vars['rho'][0]*alpha*etas['psi']
        vars['prb'] += vars['rho'][1]*alpha*etas['prb']
        vars['shift'] += vars['rho'][2]*alpha*etas['shift']
        vars['e'] += vars['rho'][3]*alpha*etas['e']
        
        if i % pars['err_step'] == 0:
            err = err_debug(vars, grads, data)    
            print(f'{i}) {alpha=:.5f}, {err=:1.5e}',flush=True)
            erra[i] = err
            alphaa[i] = alpha

        if i % pars['vis_step'] == 0 and pars['vis_step'] != -1:
            vis_debug(vars, data, i)
            if vars['rho'][3]>0:
                plot_debug3(vars['shift'],shifts_init)            
            
    return vars,erra,alphaa

vars = {}
vars['psi'] = cp.array(rec_paganin).copy()
vars['prb'] = cp.ones([1,n+2*pad,n+2*pad],dtype='complex64')
vars['shift'] = cp.array(shifts)
vars['e'] = 1+0*cp.pad(cp.sqrt(cp.array(data)),((0,0),(0,0),(pad,pad),(pad,pad)),'symmetric')
vars['e'][:,:,pad:ne-pad,pad:ne-pad]=0
vars['rho'] = [1,1,0,4]
data_rec = cp.pad(cp.array(data),((0,0),(0,0),(pad,pad),(pad,pad)))

pars = {'niter': 2000, 'err_step': 32, 'vis_step': 128}
vars,erra,alphaa = cg_holo(data_rec, vars, pars)   


# In[ ]:




