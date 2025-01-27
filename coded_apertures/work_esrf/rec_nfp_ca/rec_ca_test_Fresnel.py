#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cupy as cp
import sys
import pandas as pd
import time

from holotomocupy.utils import *


# In[2]:


show = False
domain = sys.argv[1]
size_v = int(sys.argv[2])#48
size_1 = int(sys.argv[3])#164# 144
pad = int(sys.argv[4])
niter = int(sys.argv[5])

gpu = int(sys.argv[6])
cp.cuda.Device(gpu).use()

n = 2048  # object size in each dimension
# pad = n//8 # pad for the reconstructed probe
npos = 16 # total number of positions
z1 = -17.75e-3 # [m] position of the CA
detector_pixelsize = 3.03751e-6
energy = 33.35  # [keV] xray energy
wavelength = 1.24e-09/energy  # [m] wave length
focusToDetectorDistance = 1.28  # [m]
z2 = focusToDetectorDistance-z1
distance = (z1*z2)/focusToDetectorDistance
magnification = focusToDetectorDistance/z1
voxelsize = np.abs(detector_pixelsize/magnification)  # object voxel size

extra = 8
nobj = 3072+512+2*pad
nprb = n+2*pad
npatch = nprb+2*extra

flg = f'{domain}_{size_v}_{size_1}_{pad}_sym'

path = f'/data/vnikitin/ESRF/ID16A/20240924/SiemensLH/code2um_nfp18x18_01'
path_out = f'/data/vnikitin/ESRF/ID16A/20240924_rec2/SiemensLH/code2um_nfp18x18_01'


# In[3]:


from holotomocupy.holo import G,GT
# def Lop(psi):
    
#     ff = G(psi,wavelength, voxelsize, distance, mpad)
#     ff = ff[:,pad:nprb-pad,pad:nprb-pad]
#     return ff

# def LTop(psi):
#     ff = cp.pad(psi,((0,0),(pad,pad),(pad,pad)))        
#     ff = GT(ff,wavelength, voxelsize, distance, mpad)
#     return ff

def Lop(psi):
    data = cp.zeros([psi.shape[0], n, n], dtype='complex64')
    for k in range(psi.shape[0]):
        ff = G(psi[k:k+1],wavelength, voxelsize, distance, 'symmetric')
        data[k:k+1] = ff[:,pad:nprb-pad,pad:nprb-pad]
    return data

def LTop(psi):
    data = cp.zeros([psi.shape[0], n+2*pad, n+2*pad], dtype='complex64')
    for k in range(psi.shape[0]):
        ff = cp.pad(psi[k:k+1],((0,0),(pad,pad),(pad,pad)))        
        data[k:k+1] = GT(ff,wavelength, voxelsize, distance, 'symmetric')
    return data

def L2op(psi):
    data = cp.zeros([npos, n, n], dtype='complex64')
    for i in range(npos):
        psir = psi[i].copy()
        
        x = (cp.arange(-n-2*pad,n+2*pad))*voxelsize
        [y,x] = cp.meshgrid(x,x)
        P=1j*1/(wavelength*np.abs(distance))*np.exp(1j*np.pi*(x**2+y**2)/(wavelength*distance))/(2*n)**4/2
        v = cp.ones(2*n+4*pad)        
        vv=cp.linspace(0,1,size_v)
        vv = vv**5*(126-420*vv+540*vv**2-315*vv**3+70*vv**4)
        v[:n+2*pad-size_v-size_1] = 0
        v[n+2*pad-size_v-size_1:n+2*pad-size_1] = vv#cp.sin(cp.linspace(0,1,size_v)*cp.pi/2)        
        v[-(n+2*pad):] = v[:n+2*pad][::-1]
        v = cp.outer(v,v)
        
        P *= v
        fP = cp.fft.fft2(cp.fft.fftshift(P)                )
        fP = fP.astype('complex64')

        psir = G(psir[np.newaxis],wavelength,voxelsize,distance,'symmetric',fP)[0]
        data[i] = psir[pad:-pad,pad:-pad]
        # psir = cp.pad(psir,((n//2+pad,n//2+pad),(n//2+pad,n//2+pad)))        
        # psir = cp.fft.ifft2(cp.fft.fft2(psir)*fP)        
        # data[i] = psir[2*pad+n-n//2:2*pad+n+n//2,2*pad+n-n//2:2*pad+n+n//2]
    return data

def L2Top(data):
    psi = cp.zeros([npos, n+2*pad, n+2*pad], dtype='complex64')
    for j in range(npos):
        
        x = (cp.arange(-n-2*pad,n+2*pad))*voxelsize
        [y,x] = cp.meshgrid(x,x)
        P=-1j*1/(wavelength*np.abs(distance))*np.exp(1j*np.pi*(x**2+y**2)/(-wavelength*distance))/(2*n)**4/2
        v = cp.ones(2*n+4*pad)        
        vv=cp.linspace(0,1,size_v)
        vv = vv**5*(126-420*vv+540*vv**2-315*vv**3+70*vv**4)
        v[:n+2*pad-size_v-size_1] = 0
        v[n+2*pad-size_v-size_1:n+2*pad-size_1] = vv#cp.sin(cp.linspace(0,1,size_v)*cp.pi/2)        
        v[-(n+2*pad):] = v[:n+2*pad][::-1]
        v = cp.outer(v,v)
        P *= v        
        fP = cp.fft.fft2(cp.fft.fftshift(P)                )
        fP =fP.astype('complex64')
        psir = cp.array(cp.pad(data[j],((pad,pad),(pad,pad)))).astype('complex64')        

        psi[j] += GT(psir[np.newaxis],wavelength,voxelsize,distance,'symmetric',fP)[0]
        # psir = cp.pad(psir,((n//2+pad,n//2+pad),(n//2+pad,n//2+pad)))        
        # psir = cp.fft.ifft2(cp.fft.fft2(psir)*fP)        
        # psir = psir[2*pad+n-n//2-pad:2*pad+n+n//2+pad,2*pad+n-n//2-pad:2*pad+n+n//2+pad]
        
        # psi[j] += psir
    return psi

def Ex(psi,ix):
    res = cp.empty([ix.shape[0],npatch,npatch],dtype='complex64')
    stx = nobj//2-ix[:,1]-npatch//2
    endx = stx+npatch
    sty = nobj//2-ix[:,0]-npatch//2
    endy = sty+npatch
    for k in range(len(stx)):
        res[k] = psi[sty[k]:endy[k],stx[k]:endx[k]]     
    return res

def ExT(psi,psir,ix):
    stx = nobj//2-ix[:,1]-npatch//2
    endx = stx+npatch
    sty = nobj//2-ix[:,0]-npatch//2
    endy = sty+npatch
    for k in range(len(stx)):
        psi[sty[k]:endy[k],stx[k]:endx[k]] += psir[k]
    return psi

def S(psi,p):
    x = cp.fft.fftfreq(npatch).astype('float32')
    [y, x] = cp.meshgrid(x, x)
    pp = cp.exp(-2*cp.pi*1j * (y*p[:, 1, None, None]+x*p[:, 0, None, None])).astype('complex64')
    res = cp.fft.ifft2(pp*cp.fft.fft2(psi))
    return res

def Sop(psi,ix,x,ex):
    data = cp.zeros([x.shape[1], nprb, nprb], dtype='complex64')
    psir = Ex(psi,ix)     
    psir = S(psir,x)
    data = psir[:, ex:npatch-ex, ex:npatch-ex]
    return data

def STop(d,ix,x,ex):
    psi = cp.zeros([nobj, nobj], dtype='complex64')
    dr = cp.pad(d, ((0, 0), (ex, ex), (ex, ex)))
    dr = S(dr,-x)        
    ExT(psi,dr,ix)
    return psi

# adjoint tests
shifts_test = 30*(cp.random.random([npos,2])-0.5).astype('float32')
ishifts = shifts_test.astype('int32')
fshifts = shifts_test-ishifts

arr1 = (cp.random.random([nobj,nobj])+1j*cp.random.random([nobj,nobj])).astype('complex64')
arr2 = Ex(arr1,ishifts)
arr3 = arr1*0
ExT(arr3,arr2,ishifts)
print(f'{cp.sum(arr1*cp.conj(arr3))}==\n{cp.sum(arr2*cp.conj(arr2))}')

arr1 = (cp.random.random([nobj,nobj])+1j*cp.random.random([nobj,nobj])).astype('complex64')
arr2 = Sop(arr1,ishifts,fshifts,extra)
arr3 = STop(arr2,ishifts,fshifts,extra)
print(f'{cp.sum(arr1*cp.conj(arr3))}==\n{cp.sum(arr2*cp.conj(arr2))}')

arr1 = (cp.random.random([npos,nprb,nprb])+1j*cp.random.random([npos,nprb,nprb])).astype('complex64')
arr2 = Lop(arr1)
arr3 = LTop(arr2)
print(f'{cp.sum(arr1*cp.conj(arr3))}==\n{cp.sum(arr2*cp.conj(arr2))}')
arr1=arr2=arr3=[]



if domain=='space':    
    delta = cp.zeros([npos,n+2*pad,n+2*pad],dtype='complex64')
    delta[:,n//2+pad,n//2+pad] = 1
    Ldelta2 = Lop(delta)
    Ldelta = L2op(delta)
    Lop=L2op
    LTop=L2Top


    mshow_complex(np.abs(Ldelta[0])+1j*np.abs(Ldelta2[0]),False)
    # plt.plot(np.real(Ldelta[0,0])[n//2+pad].get())
    mshow_complex(np.real(Ldelta[0,n//2-128:n//2+128,n//2-128:n//2+128])+1j*np.real(Ldelta2[0,n//2-128:n//2+128,n//2-128:n//2+128]),False)

    fig, axs = plt.subplots(2, 1, figsize=(15, 5))
    plt.suptitle(flg)
    axs[0].set_title('abs')
    axs[0].plot(np.abs(Ldelta2[0])[n//2][n//2-256:n//2+256].get(),label='space')
    axs[0].plot(np.abs(Ldelta[0])[n//2][n//2-256:n//2+256].get(),label='freq')
    axs[0].set_title('real')
    axs[1].plot(np.real(Ldelta2[0])[n//2][n//2-256:n//2+256].get(),label='space')
    axs[1].plot(np.real(Ldelta[0])[n//2][n//2-256:n//2+256].get(),label='freq')
    plt.savefig(f'figs/{flg}.png',bbox_inches='tight',dpi=300)
    plt.show()
    delta=Ldelta=Ldetla2=[]


# # read data

# In[4]:


import h5py
npos = 18*18
pos_step = 1 # steps in positions
with h5py.File(f'{path}/code2um_nfp18x18_010000.h5') as fid:
    data0 = fid['/entry_0000/measurement/data'][:npos].astype('float32')
    
with h5py.File(f'{path}/ref_0000.h5') as fid:
    ref0 = fid['/entry_0000/measurement/data'][:].astype('float32')
with h5py.File(f'{path}/dark_0000.h5') as fid:
    dark0 = fid['/entry_0000/measurement/data'][:].astype('float32')

shifts = np.loadtxt(f'/data/vnikitin/ESRF/ID16A/20240924/positions/shifts_code_nfp18x18ordered.txt')[:,::-1]
shifts = shifts/voxelsize*(2048//n)*1e-6
shifts[:,1]*=-1

# print(shifts[-10:])
shifts = np.load(f'shifts_code_new.npy')[0]
# print(shifts[-10:])
#centering
shifts[:,1]-=(np.amax(shifts[:,1])+np.amin(shifts[:,1]))/2
shifts[:,0]-=(np.amax(shifts[:,0])+np.amin(shifts[:,0]))/2
shifts = shifts.reshape(int(np.sqrt(npos)),int(np.sqrt(npos)),2)
shifts = shifts[::pos_step,::pos_step,:].reshape(npos//pos_step**2,2)
data0 = data0.reshape(int(np.sqrt(npos)),int(np.sqrt(npos)),n,n)
data0 = data0[::pos_step,::pos_step,:].reshape(npos//pos_step**2,n,n)

ids = np.where((np.abs(shifts[:,0])<nobj//2-n//2-pad-extra)*(np.abs(shifts[:,1])<nobj//2-n//2-pad-extra))[0]#[0:2]
data0 = data0[ids]
shifts = shifts[ids]

# plt.plot(shifts[:,0],shifts[:,1],'.')
# plt.axis('square')
# plt.show()

npos = len(ids)
print(f'{npos=}')


# In[5]:


from holotomocupy.proc import remove_outliers
data = data0.copy()
ref = ref0.copy()
dark = dark0.copy()
dark = np.mean(dark,axis=0)
ref = np.mean(ref,axis=0)
data-=dark
ref-=dark

data[data<0]=0
ref[ref<0]=0
data[:,1320//3:1320//3+25//3,890//3:890//3+25//3] = data[:,1280//3:1280//3+25//3,890//3:890//3+25//3]
ref[1320//3:1320//3+25//3,890//3:890//3+25//3] = ref[1280//3:1280//3+25//3,890//3:890//3+25//3]
radius = 3
threshold = 0.8

data = remove_outliers(data, radius, threshold)
ref = remove_outliers(ref[np.newaxis], radius, threshold)[0]     

data[np.isnan(data)] = 1
ref[np.isnan(ref)] = 1

rdata = data/(ref+1e-11)

mshow_complex(data[0]+1j*rdata[0],show)
mshow_complex(ref+1j*dark,show)
data0=ref0=dark0=[]


# # Paganin reconstruction

# In[6]:


def Paganin(data, wavelength, voxelsize, delta_beta,  alpha):
    fx = cp.fft.fftfreq(data.shape[-1], d=voxelsize).astype('float32')
    [fx, fy] = cp.meshgrid(fx, fx)
    rad_freq = cp.fft.fft2(data)
    taylorExp = 1 + wavelength * distance * cp.pi * (delta_beta) * (fx**2+fy**2)
    numerator = taylorExp * (rad_freq)
    denominator = taylorExp**2 + alpha
    phase = cp.log(cp.real(cp.fft.ifft2(numerator / denominator)))
    phase = delta_beta * 0.5 * phase
    return phase

def rec_init(rdata,ishifts):
    recMultiPaganin = cp.zeros([nobj,nobj],dtype='float32')
    recMultiPaganinr = cp.zeros([nobj,nobj],dtype='float32')# to compensate for overlap
    for j in range(0,npos):
        r = rdata[j]        
        r = Paganin(r, wavelength, voxelsize,  24.05, 1e-1)
        rr = r*0+1 # to compensate for overlap                        
        rpsi = cp.zeros([nobj,nobj],dtype='float32')
        rrpsi = cp.zeros([nobj,nobj],dtype='float32')
        stx = nobj//2-ishifts[j,1]-n//2
        endx = stx+n
        sty = nobj//2-ishifts[j,0]-n//2
        endy = sty+n
        rpsi[sty:endy,stx:endx] = r
        rrpsi[sty:endy,stx:endx] = rr
        
        recMultiPaganin += rpsi
        recMultiPaganinr += rrpsi
        
    recMultiPaganinr[np.abs(recMultiPaganinr)<5e-2] = 1    
    recMultiPaganin /= recMultiPaganinr    
    recMultiPaganin = np.exp(1j*recMultiPaganin)
    return recMultiPaganin

ishifts = cp.round(cp.array(shifts)).astype('int32')
rdata = cp.array(data/(ref+1e-5))
mshow(rdata[0],show)
rec_paganin = rec_init(rdata,ishifts)
mshow_polar(rec_paganin,show)
mshow_polar(rec_paganin[:1000,:1000],show)

# smooth borders
v = cp.arange(-nobj//2, nobj//2)/nobj
[vx, vy] = cp.meshgrid(v, v)
v = cp.exp(-1000*(vx**2+vy**2)).astype('float32')

rec_paganin = cp.fft.fftshift(cp.fft.fftn(cp.fft.fftshift(rec_paganin)))
rec_paganin = cp.fft.fftshift(cp.fft.ifftn(cp.fft.fftshift(rec_paganin*v))).astype('complex64')
mshow_polar(rec_paganin,show)
mshow_polar(rec_paganin[:1000,:1000],show)

rdata=v=[]


# ##### $$\nabla F=2 \left(L^*\left( L\psi-\tilde d\right)\right).$$
# ##### where $$\tilde d = d \frac{L(\psi)}{|L(\psi)|}$$
# 
# 
# 

# In[7]:


def gradientF(vars, pars, reused, d):
    Lpsi =  reused['Lpsi']    
    if pars['model']=='Gaussian':
        td = d*(Lpsi/(np.abs(Lpsi)+pars['eps']))                
        res = 2*LTop(Lpsi - td)        
    elif pars['model']=='Poisson':
        dd = d*Lpsi/(cp.abs(Lpsi)**2+pars['eps']**2) 
        res = 2*LTop(Lpsi-dd)               
    return res


# ##### $$\nabla_{\psi} G|_{(q_0,\psi_0,\boldsymbol{x}_0)}= S_{\boldsymbol{x}_{0}}^*\left(\overline{J(q_0)}\cdot \nabla F\right)$$
# 
# ##### $$\nabla_{q} G|_{(q_0,\psi_0,\boldsymbol{x}_0)}=J^*\left( \overline{S_{\boldsymbol{x}_{0}}(C_f^*(\psi_0)+\psi_{fr})}\cdot \nabla F\right).$$
# ##### $$\nabla_{\boldsymbol{x}_0} G|_{(q_0,\psi_0,\boldsymbol{x}_0)}=\textsf{Re}\Big(\big( \Big\langle \overline{q_0}\cdot \nabla F,   C(\mathcal{F}^{-1}(-2\pi i \xi_1 e^{ -2\pi i \boldsymbol{x}_{0,k}\cdot \boldsymbol{\xi}}\hat{\psi_0}))\Big\rangle,\Big\langle \overline{q_0}\cdot \nabla F,C(\mathcal{F}^{-1}(-2\pi i \xi_2 e^{ -2\pi i \boldsymbol{x}_{0,k}\cdot \boldsymbol{\xi}}\hat{\psi_0})) \Big\rangle\big)\Big)_{k=1}^K. $$
# 
# 
# 

# In[8]:


def gradient_psi(q,ix,x,ex,gradF):
    res =  STop(np.conj(q)*gradF,ix,x,ex)
    return res

def gradient_prb(psi,ix,x,ex,gradF):
    return np.sum(np.conj(Sop(psi,ix,x,ex))*gradF,axis=0)

def gradient_shift(psi, q, ix, x, ex, gradF):    
    # frequencies
    xi1 = cp.fft.fftfreq(npatch).astype('float32')
    xi2, xi1 = cp.meshgrid(xi1, xi1)

    # multipliers in frequencies
    w = cp.exp(-2 * cp.pi * 1j * (xi2 * x[:, 1, None, None] + xi1 * x[:, 0, None, None]))
    w1 = xi1
    w2 = xi2
    
    # Gradient parts
    tmp = Ex(psi, ix)
    tmp = cp.fft.fft2(tmp) 
    dt1 = cp.fft.ifft2(w*w1*tmp)
    dt2 = cp.fft.ifft2(w*w2*tmp)
    dt1 = -2 * cp.pi * dt1[:,ex:nprb+ex,ex:nprb+ex]
    dt2 = -2 * cp.pi * dt2[:,ex:nprb+ex,ex:nprb+ex]
    
    # inner product with gradF
    gradx = cp.zeros([npos, 2], dtype='float32')
    gradx[:, 0] = imdot(gradF, q * dt1, axis=(1, 2))
    gradx[:, 1] = imdot(gradF, q * dt2, axis=(1, 2))

    return gradx

def gradients(vars,pars,reused):    
    (q,psi,x) = (vars['prb'], vars['psi'], vars['fshift'])
    (ix,ex,rho) = (pars['ishift'],pars['extra'],pars['rho'])
    gradF = reused['gradF']
    dpsi = gradient_psi(q,ix,x,ex,gradF)
    dprb = gradient_prb(psi,ix,x,ex,gradF)
    dx = gradient_shift(psi,q,ix,x,ex,gradF)
    grads={'psi': rho[0]*dpsi,'prb': rho[1]*dprb, 'fshift': rho[2]*dx}
    return grads


# ##### $$\frac{1}{2}\mathcal{H}|_{x_0}(y,z)= \left\langle \mathbf{1}-d_{0}, \mathsf{Re}({L(y)}\overline{L(z)})\right\rangle+\left\langle d_{0},(\mathsf{Re} (\overline{l_0}\cdot L(y)))\cdot (\mathsf{Re} (\overline{l_0}\cdot L(z))) \right\rangle$$
# ##### $$l_0=L(x_0)/|L(x_0)|$$
# ##### $$d_0=d/|L(x_0)|$$
# 

# In[9]:


def hessianF(Lm,Ldm1,Ldm2,data,pars):
    if pars['model']=='Gaussian':
        psi0p = Lm/(np.abs(Lm)+pars['eps'])
        d0 = data/(np.abs(Lm)+pars['eps'])
        v1 = np.sum((1-d0)*reprod(Ldm1,Ldm2))
        v2 = np.sum(d0*reprod(psi0p,Ldm1)*reprod(psi0p,Ldm2))        
    else:        
        psi0p = Lm/(cp.abs(Lm)+pars['eps'])            
        v1 = cp.sum((1-data/(cp.abs(Lm)**2+pars['eps']**2))*reprod(Ldm1,Ldm2))
        v2 = 2*cp.sum(data*reprod(psi0p,Ldm1)*reprod(psi0p,Ldm2)/(cp.abs(Lm)**2+pars['eps']**2))
    return 2*(v1+v2)


# ##### $D T_\psi|_{{{z}_0}}(\Delta {z})=-2\pi iC\Big(\mathcal{F}^{-1}\big({\Delta z \cdot \xi}) e^{-2\pi i  {z}_0\cdot {\xi}}\hat{\psi}({\xi})\big)\Big)=-2\pi i C\Big(\mathcal{F}^{-1}\big((\Delta z_1 {\xi_1}+\Delta z_2 {\xi_2}) e^{-2\pi i  {z}_0\cdot {\xi}}\hat{\psi}({\xi})\big)\Big)$
# ##### $ D^2{T_\psi}|_{{{z}_0}}(\Delta{z},\Delta{w})=-4\pi^2C(\mathcal{F}^{-1}((\Delta{z}\cdot\xi)(\Delta{w}\cdot\xi)e^{-2\pi i  {z}_0\cdot {\xi}}\hat{\psi}))$
# ##### $=-4\pi^2C(\mathcal{F}^{-1}((\Delta{z_1}\Delta{w_1}\xi_1^2 + (\Delta{z_1}\Delta{w_2}+\Delta{z_2}\Delta{w_1})\xi_1\xi_2+\Delta{z_2}\Delta{w_2}\xi_2^2)\hat{\psi}))$

# #### $$ DM|_{(q_0,\psi_0,\boldsymbol{x})}(\Delta q, \Delta \psi,\Delta\boldsymbol{x})=$$
# #### $$ \Big(\Delta q\cdot T_{\psi_0}({\boldsymbol{x}_{0,k}})+ q_0\cdot \big(T_{\Delta \psi}({\boldsymbol{x}_{0,k}})+  DT_{\psi_0}|_{{\boldsymbol{x}_{0,k}}}( \Delta \boldsymbol{x}_k)\big) \Big)_{k=1}^K=$$
# #### $$ J(\Delta q)\cdot S_{\boldsymbol{x}_{0,k}}(\psi_0)+ J(q_0)\cdot S_{\boldsymbol{x}_{0}}{(\Delta \psi)}+  \Big(q_0\cdot DT_{\psi_0}|_{{\boldsymbol{x}_{0,k}}}( \Delta \boldsymbol{x}_k) \Big)_{k=1}^K$$
# 

# ##### $$ D^2M|_{(q_0,\psi_0,\boldsymbol{x})}\big((\Delta q^{(1)}, \Delta \psi^{(1)},\Delta\boldsymbol{x}^{(1)}),(\Delta q^{(2)}, \Delta \psi^{(2)},\Delta\boldsymbol{x}^{(2)})\big)= $$
# ##### $$\Big( q_0\cdot DT_{\Delta\psi^{(1)}}|_{{\boldsymbol{x}_{0,k}}}( \Delta \boldsymbol{x}^{(2)})+q_0\cdot DT_{\Delta\psi^{(2)}}|_{{\boldsymbol{x}_{0,k}}}( \Delta \boldsymbol{x}^{(1)})+ q_0\cdot D^2{T_\psi}|_{{\boldsymbol{x}_0}}(\Delta\boldsymbol{x}^{(1)},\Delta\boldsymbol{x}^{(2)})+$$
# ##### $$\Delta q^{(1)}\cdot T_{\Delta \psi^{(2)}}({\boldsymbol{x}_{0,k}})+\Delta q^{(2)}\cdot T_{\Delta \psi^{(1)}}({\boldsymbol{x}_{0,k}})+ $$
# ##### $$\Delta q^{(1)}\cdot DT_{\psi_0}|_{{\boldsymbol{x}_{0,k}}}( \Delta \boldsymbol{x}^{(2)})+\Delta q^{(2)}\cdot DT_{\psi_0}|_{{\boldsymbol{x}_{0,k}}}( \Delta \boldsymbol{x}^{(1)})\Big)_{k=1}^K.$$
# 

# ##### $$\mathcal{H}^G|_{ (q_0,\psi_0,\boldsymbol{x}_0)}\Big((\Delta q^{(1)},\Delta \psi^{(1)},\Delta \boldsymbol{x}^{(1)}),(\Delta q^{(2)},\Delta \psi^{(2)},\Delta \boldsymbol{x}^{(2)})\Big)=$$
# ##### $$\Big\langle \nabla F|_{M(q_0,\psi_0,\boldsymbol{x}_0)}, D^2M|_{(q_0,\psi_0,\boldsymbol{x}_0)}\Big((\Delta q^{(1)},\Delta \psi^{(1)},\Delta \boldsymbol{x}^{(1)}),(\Delta q^{(2)},\Delta \psi^{(2)},\Delta \boldsymbol{x}^{(2)})\Big)\Big\rangle +$$
# ##### $$\mathcal{H}^F|_{M(q_0,\psi_0,\boldsymbol{x}_0)}\Big(DM|_{(q_0,\psi_0,\boldsymbol{x}_0)}(\Delta q^{(1)},\Delta \psi^{(1)},\Delta \boldsymbol{x}^{(1)}),DM|_{(q_0,\psi_0,\boldsymbol{x}_0)}(\Delta q^{(2)},\Delta \psi^{(2)},\Delta \boldsymbol{x}^{(2)})\Big)$$
# 

# ### Updates:
# 
# \begin{equation}
#                \alpha_j=\frac{\mathsf{Re}\langle \nabla F|_{x_j},s_j\rangle}{H|_{x_j}( {s_j},s_j)}
#              \end{equation}
# 
# \begin{equation}
#                 \beta_j=\frac{H(\nabla F|_{x_j},s_j)}{H|_{x_j}( {s_j},s_j)}.
# \end{equation}
# 
# ### Scaling variables:

# \begin{equation}
# \begin{aligned}
# \tilde{\beta}_j=\frac{H^{\tilde{F}}|_{\tilde{x}_j} (\nabla \tilde{F}|_{\tilde{x}_j},\tilde{\eta}_j)}{H^{\tilde{F}}|_{\tilde{x}_j} (\tilde{\eta}_j,\tilde{\eta}_j)}=\frac{H^{F}|_{x_j} (\rho\nabla \tilde{F}|_{\tilde{x}_j},\rho\tilde{\eta}_j)}{H^{F}|_{x_j} (\rho\tilde{\eta}_j,\rho\tilde{\eta}_j)}=\frac{H^{F}|_{x_j} (\rho^2\nabla F|_{x_j},\rho\tilde{\eta}_j)}{H^{F}|_{x_j} (\rho\tilde{\eta}_j,\rho\tilde{\eta}_j)}
# \end{aligned}
# \end{equation}
# 
# \begin{equation}
# \begin{aligned}
# \tilde{\alpha}_j=\frac{\langle\nabla \tilde{F}|_{\tilde{x}_j},\tilde{\eta}_j\rangle}{H^{\tilde{F}}|_{\tilde{x}_j} (\tilde{\eta}_j,\tilde{\eta}_j)}=\frac{\langle \rho\nabla F|_{x_j},\tilde{\eta}_j\rangle}{H^{F}|_{x_j} (\rho\tilde{\eta}_j,\rho\tilde{\eta}_j)}
# \end{aligned}
# \end{equation}
# 
# \begin{equation}
#     \begin{aligned}
#         \tilde{\eta}_{j+1} = -\nabla \tilde{F}|_{\tilde{x}_j}+\tilde{\beta}_j\tilde{\eta}_j=-\rho\nabla F|_{x_j}+\tilde{\beta}_j\tilde{\eta}_j,\quad \text{with } \tilde{\eta}_0=-\rho\nabla F|_{x_0}
#     \end{aligned}
# \end{equation}
# 
# \begin{equation}
#     \begin{aligned}
#         \tilde{x}_{j+1} = \tilde{x}_{j}+\tilde{\alpha}_j\tilde{\eta}_{j+1}
#     \end{aligned}
# \end{equation}
# 
# Multiplying both sides by $\rho$,
# 
# \begin{equation}
#     \begin{aligned}
#         x_{j+1} = x_j+\rho\tilde{\alpha}_j\tilde{\eta}_{j+1}
#     \end{aligned}
# \end{equation}

# # Optimized version, without extra functions

# In[10]:


def calc_beta(vars,grads,etas,pars,reused,d):
    (q,psi,x) = (vars['prb'], vars['psi'], vars['fshift'])    
    (ix,ex,rho) = (pars['ishift'],pars['extra'],pars['rho'])
    (Lpsi,gradF) = (reused['Lpsi'], reused['gradF'])
    
    # note scaling with rho
    (dpsi1,dq1,dx1) = (grads['psi']*rho[0], grads['prb']*rho[1], grads['fshift']*rho[2])
    (dpsi2,dq2,dx2) = (etas['psi']*rho[0],etas['prb']*rho[1], etas['fshift']*rho[2])
        
    # frequencies
    xi1 = cp.fft.fftfreq(npatch).astype('float32')
    [xi2, xi1] = cp.meshgrid(xi1, xi1)    

    # multipliers in frequencies
    dx1 = dx1[:,:,np.newaxis,np.newaxis]
    dx2 = dx2[:,:,np.newaxis,np.newaxis]
    w = cp.exp(-2*cp.pi*1j * (xi2*x[:, 1, None, None]+xi1*x[:, 0, None, None]))
    w1 = xi1*dx1[:,0]+xi2*dx1[:,1]
    w2 = xi1*dx2[:,0]+xi2*dx2[:,1]
    w12 = xi1**2*dx1[:,0]*dx2[:,0]+ \
                xi1*xi2*(dx1[:,0]*dx2[:,1]+dx1[:,1]*dx2[:,0])+ \
                xi2**2*dx1[:,1]*dx2[:,1]
    w22 = xi1**2*dx2[:,0]**2+ 2*xi1*xi2*(dx2[:,0]*dx2[:,1]) + xi2**2*dx2[:,1]**2
    
    # DT, D2T terms
    tmp1 = Ex(dpsi1,ix)     
    # tmp1 = dpsip1#Ex(dpsi1,ix)     
    tmp1 = cp.fft.fft2(tmp1)
    sdpsi1 = cp.fft.ifft2(w*tmp1)[:,ex:nprb+ex,ex:nprb+ex]
    dt12 = -2*np.pi*1j*cp.fft.ifft2(w*w2*tmp1)[:,ex:nprb+ex,ex:nprb+ex]
    
    tmp2 =Ex(dpsi2,ix)     
    # tmp2 = dpsip2#Ex(dpsi2,ix)     
    tmp2 = cp.fft.fft2(tmp2)
    sdpsi2 = cp.fft.ifft2(w*tmp2)[:,ex:nprb+ex,ex:nprb+ex]
    dt21 = -2*np.pi*1j*cp.fft.ifft2(w*w1*tmp2)[:,ex:nprb+ex,ex:nprb+ex]
    dt22 = -2*np.pi*1j*cp.fft.ifft2(w*w2*tmp2)[:,ex:nprb+ex,ex:nprb+ex]
    
    tmp = Ex(psi,ix)     
    spsi = S(tmp,x)[:, ex:npatch-ex, ex:npatch-ex]    
    
    # tmp = psip#Ex(psi,ix)     
    tmp = cp.fft.fft2(tmp)        
    dt1 = -2*np.pi*1j*cp.fft.ifft2(w*w1*tmp)[:,ex:nprb+ex,ex:nprb+ex]
    dt2 = -2*np.pi*1j*cp.fft.ifft2(w*w2*tmp)[:,ex:nprb+ex,ex:nprb+ex]
    d2t1 = -4*np.pi**2*cp.fft.ifft2(w*w12*tmp)[:,ex:nprb+ex,ex:nprb+ex]
    d2t2 = -4*np.pi**2*cp.fft.ifft2(w*w22*tmp)[:,ex:nprb+ex,ex:nprb+ex]
    
    # DM,D2M terms
    d2m1 =  q*dt12 + q*dt21 + q*d2t1
    d2m1 += dq1*sdpsi2 + dq2*sdpsi1
    d2m1 += dq1*dt2 + dq2*dt1
    top = redot(gradF,d2m1)
    d2m1=[]

    d2m2 =  q*dt22 + q*dt22 + q*d2t2
    d2m2 += dq2*sdpsi2 + dq2*sdpsi2
    d2m2 += dq2*dt2 + dq2*dt2
    bottom = redot(gradF,d2m2)
    d2m2=[]
    
    
    Ldm1 = Lop(dq1*spsi+q*(sdpsi1+dt1) )
    Ldm2 = Lop(dq2*spsi+q*(sdpsi2+dt2)  )
    spsi=sdpsi1=sdpsi2=dt=dt1=dt2=[]
    # top and bottom parts
    top+=hessianF(Lpsi, Ldm1, Ldm2, d, pars)            
    bottom+=hessianF(Lpsi, Ldm2, Ldm2,d, pars)
    
    return top/bottom, top,bottom

def calc_alpha(vars,grads,etas,pars,reused,d):    
    (q,psi,x) = (vars['prb'], vars['psi'], vars['fshift'])    
    (ix,ex,rho) = (pars['ishift'],pars['extra'],pars['rho'])
    (dpsi1,dq1,dx1) = (grads['psi'], grads['prb'], grads['fshift'])
    (dpsi2,dq2,dx2) = (etas['psi'], etas['prb'], etas['fshift'])    
    (Lpsi,gradF) = (reused['Lpsi'], reused['gradF'])

    # top part
    top = -redot(dpsi1,dpsi2)-redot(dq1,dq2)-redot(dx1,dx2)
    
    # scale variable for the hessian
    (dpsi,dq,dx) = (etas['psi']*rho[0],etas['prb']*rho[1], etas['fshift']*rho[2])

    # frequencies        
    xi1 = cp.fft.fftfreq(npatch).astype('float32')    
    [xi2, xi1] = cp.meshgrid(xi1, xi1)

    # multipliers in frequencies
    dx = dx[:,:,np.newaxis,np.newaxis]
    w = cp.exp(-2*cp.pi*1j * (xi2*x[:, 1, None, None]+xi1*x[:, 0, None, None]))
    w1 = xi1*dx[:,0]+xi2*dx[:,1]
    w2 = xi1**2*dx[:,0]**2+ 2*xi1*xi2*(dx[:,0]*dx[:,1]) + xi2**2*dx[:,1]**2
    
    # DT,D2T terms, and Spsi
    tmp = Ex(dpsi,ix)     
    # tmp = dpsip#Ex(dpsi,ix)     
    tmp = cp.fft.fft2(tmp)    
    sdpsi = cp.fft.ifft2(w*tmp)[:,ex:nprb+ex,ex:nprb+ex]
    dt2 = -2*np.pi*1j*cp.fft.ifft2(w*w1*tmp)[:,ex:nprb+ex,ex:nprb+ex]
    
    tmp = Ex(psi,ix)     
    spsi = S(tmp,x)[:, ex:npatch-ex, ex:npatch-ex]    
    # tmp = psip#Ex(psi,ix)     
    tmp = cp.fft.fft2(tmp)
    dt = -2*np.pi*1j*cp.fft.ifft2(w*w1*tmp)[:,ex:nprb+ex,ex:nprb+ex]
    d2t = -4*np.pi**2*cp.fft.ifft2(w*w2*tmp)[:,ex:nprb+ex,ex:nprb+ex]
    
    # DM and D2M terms
    d2m2 = q*(2*dt2 + d2t)+2*dq*sdpsi+2*dq*dt
    bottom = redot(gradF,d2m2)
    d2m2=[]

    Ldm = Lop(dq*spsi+q*(sdpsi+dt))   
    spsi=sdpsi=dt=[]
            
    # bottom part
    bottom += hessianF(Lpsi, Ldm, Ldm,d,pars)
    
    return top/bottom, top, bottom


# ## minimization functional and calculation of reused arrays

# In[11]:


def minf(Lpsi,d,pars):
    if pars['model']=='Gaussian':
        f = cp.linalg.norm(cp.abs(Lpsi)-d)**2/(n*n*npos)    
    else:        
        f = cp.sum(cp.abs(Lpsi)**2-2*d*cp.log(cp.abs(Lpsi)+pars['eps']))/(n*n*npos)          
        # loss = torch.nn.PoissonNLLLoss(log_input=False, full=True, size_average=None, eps=pars['eps'], reduce=None, reduction='sum')
        # f = loss(torch.as_tensor(cp.abs(Lpsi)**2,device='cuda'),torch.as_tensor(d,device='cuda'))    
    return f

def calc_reused(vars, pars):
    
    (q,psi,x) = (vars['prb'], vars['psi'],vars['fshift'])    
    (ix,ex,rho) = (pars['ishift'],pars['extra'],pars['rho'])
    
    reused = {}
    spsi = S(Ex(psi,ix),x)[:, ex:npatch-ex, ex:npatch-ex]    
    reused['Lpsi'] = Lop(spsi*q)     
    return reused


# ## debug functions

# In[12]:


def plot_debug(vars,etas,pars,top,bottom,alpha,data,i):
    '''Check the minimization functional behaviour'''
    if i % pars['vis_step'] == 0 and pars['vis_step'] != -1 and show:
        (q,psi,x) = (vars['prb'], vars['psi'], vars['fshift'])    
        (ix,ex,rho) = (pars['ishift'],pars['extra'],pars['rho'])
        (dpsi2,dq2,dx2) = (etas['psi'], etas['prb'], etas['fshift'])    

        npp = 3
        errt = cp.zeros(npp*2)
        errt2 = cp.zeros(npp*2)
        for k in range(0,npp*2):
            # psipt = psip+(alpha*k/(npp-1))*rho[0]*dpsip2
            psit = psi+(alpha*k/(npp-1))*rho[0]*dpsi2
            qt = q+(alpha*k/(npp-1))*rho[1]*dq2
            xt = x+(alpha*k/(npp-1))*rho[2]*dx2
            errt[k] = minf(Lop(S(Ex(psit,ix),xt)[:, ex:npatch-ex, ex:npatch-ex]*qt),data,pars)
                    
        t = alpha*(cp.arange(2*npp))/(npp-1)    
        errt2 = minf(Lop(S(Ex(psi,ix),x)[:, ex:npatch-ex, ex:npatch-ex]*q),data,pars)
        errt2 = errt2 -top*t/(n*n*npos)+0.5*bottom*t**2/(n*n*npos)    
        plt.plot(alpha.get()*cp.arange(2*npp).get()/(npp-1),errt.get(),'.')
        plt.plot(alpha.get()*cp.arange(2*npp).get()/(npp-1),errt2.get(),'.')
        plt.show()

def vis_debug(vars,pars,i):
    '''Visualization and data saving'''
    if i % pars['vis_step'] == 0 and pars['vis_step'] != -1:
        (q,psi,x) = (vars['prb'], vars['psi'], vars['fshift'])        
        mshow_polar(psi,show)
        mshow_polar(q,show)
        if show:
            plt.plot(vars['fshift'].get(),'.')
            plt.show()
        write_tiff(np.angle(psi),f'{path_out}_{flg}/crec_psi_angle/{i:03}')
        write_tiff(np.abs(psi),f'{path_out}_{flg}/crec_psi_abs/{i:03}')
        write_tiff(np.angle(q),f'{path_out}_{flg}/crec_prb_angle/{i:03}')
        write_tiff(np.abs(q),f'{path_out}_{flg}/crec_prb_abs/{i:03}')
        np.save(f'{path_out}_{flg}/crec_shift_{i:03}',x)

def error_debug(vars, pars, reused, data, i):
    '''Visualization and data saving'''
    if i % pars['err_step'] == 0 and pars['err_step'] != -1:
        err = minf(reused['Lpsi'],data,pars)
        print(f'{i}) {err=:1.5e}',flush=True)                        
        vars['table'].loc[len(vars['table'])] = [i, err.get(), time.time()]
        vars['table'].to_csv(f'{flg}', index=False)            

def grad_debug(alpha, grads, pars, i):
    if i % pars['grad_step'] == 0 and pars['grad_step'] != -1:
        print(f'(alpha,psi,prb,shift): {alpha:.1e} {np.linalg.norm(grads['psi']):.1e},{np.linalg.norm(grads['prb']):.1e},{np.linalg.norm(grads['fshift']):.1e}')


# # Bilinear Hessian method

# In[13]:


def BH(data, vars, pars):
   
    if pars['model']=='Gaussian':
        # work with sqrt
        data = cp.sqrt(data)
    alpha=1    
    for i in range(pars['niter']):                             
        reused = calc_reused(vars, pars)
        error_debug(vars, pars, reused, data, i)
        vis_debug(vars, pars, i)            
      
        reused['gradF'] = gradientF(vars,pars,reused,data) 
        grads = gradients(vars,pars,reused)
        
        if i==0 or pars['method']=='BH-GD':
            etas = {}
            etas['psi'] = -grads['psi']
            etas['prb'] = -grads['prb']
            etas['fshift'] = -grads['fshift']
        else:      
            beta,_,_ = calc_beta(vars, grads, etas, pars, reused, data)
                
            etas['psi'] = -grads['psi'] + beta*etas['psi']
            etas['prb'] = -grads['prb'] + beta*etas['prb']
            etas['fshift'] = -grads['fshift'] + beta*etas['fshift']

        
        alpha,top,bottom = calc_alpha(vars, grads, etas, pars, reused, data)         
        plot_debug(vars,etas,pars,top,bottom,alpha,data,i)
        grad_debug(alpha,grads,pars,i)
        
        vars['psi'] += pars['rho'][0]*alpha*etas['psi']
        
        vars['prb'] += pars['rho'][1]*alpha*etas['prb']
        vars['fshift'] += pars['rho'][2]*alpha*etas['fshift']
        
    return vars


# reconstruct probe
rho = [0,1,0]
probe = cp.ones([nprb,nprb],dtype='complex64')
positions_px = cp.array(shifts)

# fixed variables
pars = {'niter': 4, 'err_step': 1, 'vis_step': -1, 'grad_step': -1}
pars['rho'] = rho
pars['ishift'] = cp.round(positions_px).astype('int32')
pars['extra'] = extra
pars['eps'] = 1e-8
pars['model'] = 'Gaussian'


pars['method'] = 'BH-CG'
vars = {}
vars['psi'] = rec_paganin.copy()*0+1
vars['prb'] = cp.array(probe)
vars['fshift'] = cp.array(positions_px-cp.round(positions_px).astype('int32')).astype('float32')
vars['table'] = pd.DataFrame(columns=["iter", "err", "time"])

data_rec = cp.array(ref).copy()
vars = BH(data_rec, vars, pars)      
mshow_polar(vars['prb'],show)


# # Smooth borders for the probe

# In[14]:


probe = cp.pad(vars['prb'][3*pad//2:-3*pad//2,3*pad//2:-3*pad//2],((3*pad//2,3*pad//2),(3*pad//2,3*pad//2)),'symmetric')
# v = cp.ones(n+2*pad)
# vv = cp.sin(cp.linspace(0,np.pi/2,pad))
# v[:pad]=vv
# v[-pad:]=vv[::-1]
# v=cp.outer(v,v)
# probe*=v

mshow_polar(probe,show)


# # full reconstruction

# In[15]:


rho = [1,1.5*np.mean(np.abs(probe)),0.1]
positions_px = cp.array(shifts)

# fixed variables
pars = {'niter': niter, 'err_step': 32, 'vis_step': 32, 'grad_step': -1}
pars['rho'] = rho
pars['ishift'] = cp.round(positions_px).astype('int32')
pars['extra'] = extra
pars['eps'] = 1e-8
pars['model'] = 'Gaussian'


pars['method'] = 'BH-CG'
vars = {}
vars['psi'] = rec_paganin.copy()*0+1
vars['prb'] = cp.array(probe)
vars['fshift'] = cp.array(positions_px-cp.round(positions_px).astype('int32')).astype('float32')
vars['table'] = pd.DataFrame(columns=["iter", "err", "time"])

data_rec = cp.array(data).copy()
vars = BH(data_rec, vars, pars)      

