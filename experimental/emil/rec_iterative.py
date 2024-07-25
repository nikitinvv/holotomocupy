#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import h5py
from holotomocupy.holo import G, GT
from holotomocupy.magnification import M, MT
from holotomocupy.shift import S, ST, registration_shift
from holotomocupy.recon_methods import CTFPurePhase, multiPaganin
from holotomocupy.proc import linear, dai_yuan
from holotomocupy.chunking import gpu_batch
import holotomocupy.chunking as chunking
from holotomocupy.utils import *
from holotomocupy.proc import remove_outliers
import sys


chunking.global_chunk = 10


# # Init data sizes and parametes of the PXM of ID16A

# In[2]:


st = int(sys.argv[1])
n = 2048
ntheta = 150

# ID16B setup
ndist = 4
energy = 29.63  # [keV] xray energy
wavelength = 1.2398419840550367e-09/energy  # [m] wave length
detector_pixelsize= 0.65e-6
focusToDetectorDistance = 0.704433  # [m]
sx0h = 0.8525605999567023e-3; #1.077165773192669 for 75nm.
sx0v = 0.80170811624758109e-3; #1.110243284221266 for 75nm.
sx0 = 0.5*(sx0h+sx0v)
z1 = np.array([54.9598e-3, 55.96e-3, 59.1701e-3, 69.17e-3])[:ndist]-sx0
z2 = focusToDetectorDistance-z1
distances = (z1*z2)/focusToDetectorDistance
magnifications = focusToDetectorDistance/z1
voxelsize = detector_pixelsize/magnifications[0]*2048/n  # object voxel size
print(f"{voxelsize=}")
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

# allow padding if there are shifts of the probe
pad = n//32
# sample size after demagnification
ne = int(np.ceil((n+2*pad)/norm_magnifications[-1]/32))*32  # make multiple of 32


# In[3]:


data0 = np.zeros([ntheta,ndist,n,n],dtype='float32')
ref0 = np.zeros([1,ndist,n,n],dtype='float32')
dark0 = np.zeros([1,n,n],dtype='float32')
cropx_left=cropx_right=cropy_up=cropy_down=0
binning = 0
# theta = np.loadtxt(f'/data/viktor/id16a/3d_ald4/3d_ald4_ht_10nm_/angles_file.txt').astype('float32')[:][:900:900//ntheta]
# print(f'{theta=}')
for k in range(ndist):
    for j in range(0,ntheta):
        # print(j)
        jtheta=st+j*900//ntheta
        fname = f'/data/vnikitin/ESRF/ID16B/009/033_009_50nm_{k+1}_/033_009_50nm_{k+1}_{jtheta:04}.edf'
        tmp = dxchange.read_edf(fname)[0,cropy_up:2048-cropy_down,cropx_left:2048-cropx_right]
        for kb in range(binning):
            tmp = (tmp[::2]+tmp[1::2])/2
            tmp = (tmp[:,::2]+tmp[:,1::2])/2
        data0[j,k] = tmp

    tmp = dxchange.read_edf(f'/data/vnikitin/ESRF/ID16B/009/033_009_50nm_{k+1}_/refHST0000.edf')[0,cropy_up:2048-cropy_down,cropx_left:2048-cropx_right]
    for kb in range(binning):
        tmp = (tmp[::2]+tmp[1::2])/2
        tmp = (tmp[:,::2]+tmp[:,1::2])/2
    ref0[:,k] = tmp

tmp = dxchange.read_edf(f'/data/vnikitin/ESRF/ID16B/009/033_009_50nm_1_/dark.edf')[0,cropy_up:2048-cropy_down,cropx_left:2048-cropx_right]
for kb in range(binning):
    tmp = (tmp[::2]+tmp[1::2])/2
    tmp = (tmp[:,::2]+tmp[:,1::2])/2
dark0[:] = tmp

# data0-=dark0
# ref0 -= dark0
# mean_all = np.mean(data0[:,:,0:n//16,0:n//16])

# mean_each = np.mean(data0[:,:,0:n//16,0:n//16],axis=(2,3))
# data0 = data0*mean_all/mean_each[:,:,np.newaxis,np.newaxis]

# ref0_mean = np.mean(ref0[:,0:n//16,0:n//16],axis=(1,2))
# ref0 = ref0*mean_all/ref0_mean[:,np.newaxis,np.newaxis]


# In[4]:


print(data0.shape,ref0.shape)
# remove outliers
for k in range(ndist):
    radius = 7
    threshold = 20000
    data0[:,k] = remove_outliers(data0[:,k], radius, threshold)
    ref0[:,k] = remove_outliers(ref0[:,k], radius, threshold)        


# ## Take mean for flat and dark

# In[5]:


ref0 = np.mean(ref0,axis=0)[np.newaxis]
dark0 = np.mean(dark0,axis=0)[np.newaxis]


# ### Normalize everything wrt to the mean of the reference image

# In[6]:


mean_value = np.mean(ref0)#*0+1
dark00 = dark0.astype('float32')/mean_value
ref00 = ref0.astype('float32')/mean_value
data00 = data0.astype('float32')/mean_value


data00 = data00-np.mean(dark00,axis=0)
ref00 = ref00-np.mean(dark00,axis=0)

data00[data00<0] = 0
ref00[ref00<0] = 0


# # Find shifts of reference images

# In[7]:


shifts_ref0 = np.zeros([1, ndist, 2], dtype='float32')
for k in range(ndist):
    shifts_ref0[:, k] = registration_shift(ref00[:, k], ref00[:, 0], upsample_factor=1000)
    

shifts_ref = np.zeros([ntheta, ndist, 2], dtype='float32')
for k in range(ndist):
    im = np.tile(ref00[0, 0], [ntheta, 1, 1])
    shifts_ref[:, k] = registration_shift(data00[:, k], im, upsample_factor=1000)
    

# ### Assuming the shifts are calculated, shifts refs back

# In[8]:


data0 = data00.copy()
ref0 = ref00.copy()
# shifted refs for correction
for k in range(ndist):
    # shift refs back
    ref0[:, k] = ST(ref0[:, k].astype('complex64'), shifts_ref0[:, k]).real

ref0c = np.tile(np.array(ref0), (ntheta, 1, 1, 1))
for k in range(ndist):
    # shift refs the position where they were when collecting data
    ref0c[:, k] = S(ref0c[:, k].astype('complex64'), shifts_ref[:, k]).real


# ### divide data by the reference image

# In[9]:


rdata = data0/(ref0+1e-9)


# In[10]:




# ### Scale images

# In[11]:


rdata_scaled = rdata.copy()

for k in range(ndist):
    rdata_scaled[:, k] = M(rdata_scaled[:, k], 1/norm_magnifications[k], n).real




# ### Align images between different planes
# 

# #### Approach 1. Align data

# In[12]:


# shifts_drift = np.zeros([ntheta,ndist,2],dtype='float32')

# for k in range(1,ndist):
#     shifts_drift[:,k] = registration_shift(rdata_scaled[:,k],rdata_scaled[:,0],upsample_factor=1000)

# # note shifts_drift should be after magnification.
# shifts_drift*=norm_magnifications[np.newaxis,:,np.newaxis]

# shifts_drift_median = shifts_drift.copy()
# shifts_drift_median[:] = np.median(shifts_drift,axis=0)

# print(shifts_drift_median[0],shifts_drift_init[0])
# for k in range(ndist):
#     fig, axs = plt.subplots(1, 2, figsize=(10, 3))
#     im=axs[0].plot(shifts_drift[:,k,0],'.')
#     im=axs[0].plot(shifts_drift_median[:,k,0],'.')
#     im=axs[0].plot(shifts_drift_init[:,k,0],'r.')
#     axs[0].set_title(f'distance {k}, shifts y')
#     im=axs[1].plot(shifts_drift[:,k,1],'.')
#     im=axs[1].plot(shifts_drift_median[:,k,1],'.')
#     im=axs[1].plot(shifts_drift_init[:,k,1],'r.')
#     axs[1].set_title(f'distance {k}, shifts x')
#     # plt.show()


# #### Approach 2. Align CTF reconstructions from 1 distance

# In[13]:


recCTF_1dist = np.zeros([ntheta, ndist, n, n], dtype='float32')
distances_ctf = (distances/norm_magnifications**2)[:ndist]

for k in range(ndist):
    recCTF_1dist[:, k] = CTFPurePhase(
        rdata_scaled[:, k:k+1], distances_ctf[k:k+1],wavelength, voxelsize,  1e-1)



shifts_drift = np.zeros([ntheta, ndist, 2], dtype='float32')

for k in range(1, ndist):
    shifts_drift[:, k] = registration_shift(
        recCTF_1dist[:, k], recCTF_1dist[:, 0], upsample_factor=1000)

# note shifts_drift should be after magnification.
shifts_drift *= norm_magnifications[np.newaxis, :, np.newaxis]

shifts_drift_median = shifts_drift.copy()
shifts_drift_median[:] = np.median(shifts_drift, axis=0)

# print(shifts_drift_median[0], shifts_drift_init[0])



# set mediam
shifts_drift = shifts_drift_median


# In[14]:


rdata_scaled_aligned = rdata_scaled.copy()
for k in range(ndist):
    rdata_scaled_aligned[:, k] = ST(rdata_scaled[:, k], shifts_drift[:, k]/norm_magnifications[k]).real


# #### Reconstruction by the MultiPaganin method

# In[15]:


# distances should not be normalized
distances_pag = (distances/norm_magnifications**2)[:ndist]
recMultiPaganin = multiPaganin(rdata_scaled_aligned, distances_pag, wavelength, voxelsize, 30, 1e-12)
#mshow(recMultiPaganin[0])


# #### Reconstruction by the CTF pure phase method

# In[16]:


distances_ctf = (distances/norm_magnifications**2)[:ndist]

recCTF = CTFPurePhase(rdata_scaled_aligned, distances_ctf, wavelength, voxelsize, 1e-2)
#mshow(recCTF[0])
#mshow(recCTF[0,750:750+500,500:1000])


# In[17]:


shifts = shifts_drift


# # Construct operators

# #### Forward holo: $d=\mathcal{G}_{z_j}\left((\mathcal{G}_{z_j'}\mathcal{S}_{s'_{j}}q)\mathcal{M}_j\mathcal{S}_{s_{j}}\psi\right)$,
# #### Adjoint holo: $\psi=\sum_j\mathcal{S}^H_{s_j}\mathcal{M}_j^H\left((\mathcal{G}_{z_j'}\mathcal{S}_{s'_{j}}q)^*\mathcal{G}^H_{z_j}d\right)$, 
# #### Adjoint holo wrt probe: $q=\sum_{j,k}\mathcal{S}_{s_{k,j}'}\mathcal{G}_{z_j'}^H\left((\mathcal{M}_j\mathcal{S}_{s_{k,j}}\psi_k)^*\mathcal{G}^H_{z_j}d\right)$
# 
# 

# In[18]:


from holotomocupy.chunking import gpu_batch


@gpu_batch
def _fwd_holo(psi, shifts_ref, shifts, prb):    
    prb = cp.array(prb)
    shifts_ref = cp.array(shifts_ref)
    shifts = cp.array(shifts)

    data = cp.zeros([psi.shape[0], ndist, n, n], dtype='complex64')
    for i in range(ndist):
        # ill shift for each acquisition
        prbr = cp.tile(prb, [psi.shape[0], 1, 1])

        prbr = S(prbr, shifts_ref[:, i])
        # propagate illumination
        prbr = G(prbr, wavelength, voxelsize, distances2[i])
        # object shift for each acquisition
        psir = S(psi, shifts[:, i]/norm_magnifications[i])

        # scale object
        if ne != n:
            psir = M(psir, norm_magnifications[i]*ne/(n+2*pad), n+2*pad)

        # multiply the ill and object
        psir *= prbr

        # propagate both
        psir = G(psir, wavelength, voxelsize, distances[i])
        data[:, i] = psir[:, pad:n+pad, pad:n+pad]
    return data


def fwd_holo(psi, prb):
    return _fwd_holo(psi, shifts_ref, shifts, prb)


@gpu_batch
def _adj_holo(data, shifts_ref, shifts, prb):
    prb = cp.array(prb)
    shifts_ref = cp.array(shifts_ref)
    shifts = cp.array(shifts)
    psi = cp.zeros([data.shape[0], ne, ne], dtype='complex64')
    for j in range(ndist):
        psir = cp.pad(data[:, j], ((0, 0), (pad, pad), (pad, pad)))

        # propagate data back
        psir = GT(psir, wavelength, voxelsize, distances[j])

        # ill shift for each acquisition
        prbr = cp.tile(prb, [data.shape[0], 1, 1])
        prbr = S(prbr, shifts_ref[:, j])

        # propagate illumination
        prbr = G(prbr, wavelength, voxelsize, distances2[j])

        # multiply the conj ill and object
        psir *= cp.conj(prbr)

        # scale object
        if ne != n:
            psir = MT(psir, norm_magnifications[j]*ne/(n+2*pad), ne)
        # object shift for each acquisition
        psi += ST(psir, shifts[:, j]/norm_magnifications[j])
    return psi


def adj_holo(data, prb):
    return _adj_holo(data, shifts_ref, shifts, prb)


@gpu_batch
def _adj_holo_prb(data, shifts_ref, shifts, psi):
    shifts_ref = cp.array(shifts_ref)
    shifts = cp.array(shifts)
    
    prb = cp.zeros([data.shape[0], n+2*pad, n+2*pad], dtype='complex64')
    for j in range(ndist):
        prbr = np.pad(data[:, j], ((0, 0), (pad, pad), (pad, pad)))
        psir = psi.copy()

        # propagate data back
        prbr = GT(prbr, wavelength, voxelsize, distances[j])

        # object shift for each acquisition
        psir = S(psir, shifts[:, j]/norm_magnifications[j])

        # scale object
        psir = M(psir, norm_magnifications[j]*ne/(n+2*pad), n+2*pad)

        # multiply the conj object and ill
        prbr *= cp.conj(psir)

        # propagate illumination
        prbr = GT(prbr, wavelength, voxelsize, distances2[j])

        # ill shift for each acquisition
        prbr = ST(prbr, shifts_ref[:, j])
        prb += prbr
    return prb


def adj_holo_prb(data, psi):
    ''' Adjoint Holography operator '''
    return np.sum(_adj_holo_prb(data, shifts_ref, shifts, psi), axis=0)[np.newaxis]


# adjoint test
data = data0.copy()
ref = ref0.copy()
arr1 = np.pad(np.array(data[:, 0]+1j*data[:, 0]).astype('complex64'),
              ((0, 0), (ne//2-n//2, ne//2-n//2), (ne//2-n//2, ne//2-n//2)), 'symmetric')
prb1 = np.array(ref[0, :1]+1j*ref[0, :1]).astype('complex64')
prb1 = np.pad(prb1, ((0, 0), (pad, pad), (pad, pad)))

arr2 = fwd_holo(arr1, prb1)
arr3 = adj_holo(arr2, prb1)
arr4 = adj_holo_prb(arr2, arr1)

print(f'{np.sum(arr1*np.conj(arr3))}==\n{np.sum(arr2*np.conj(arr2))}')
print(f'{np.sum(prb1*np.conj(arr4))}==\n{np.sum(arr2*np.conj(arr2))}')


# #### Forward holo without sample: $d=\mathcal{G}_{z_j}\mathcal{S}_{s'_{j}}q$, 
# #### Adjoint holo without sample: $q=\sum_j\mathcal{S}^H_{s'_{j}}\mathcal{G}^H_{z_j}d$

# In[19]:


@gpu_batch
def _fwd_holo0(prb, shifts_ref0):
    shifts_ref0 = cp.array(shifts_ref0)
    data = cp.zeros([1, ndist, n, n], dtype='complex64')
    for j in range(ndist):
        # ill shift for each acquisition
        prbr = S(prb, shifts_ref0[:, j])
        # propagate illumination
        data[:, j] = G(prbr, wavelength, voxelsize, distances[0])[:, pad:n+pad, pad:n+pad]
    return data


def fwd_holo0(prb):
    return _fwd_holo0(prb, shifts_ref0)


@gpu_batch
def _adj_holo0(data, shifts_ref0):
    shifts_ref0 = cp.array(shifts_ref0)
    prb = cp.zeros([1, n+2*pad, n+2*pad], dtype='complex64')
    for j in range(ndist):
        # ill shift for each acquisition
        prbr = cp.pad(data[:, j], ((0, 0), (pad, pad), (pad, pad)))
        # propagate illumination
        prbr = GT(prbr, wavelength, voxelsize, distances[0])
        # ill shift for each acquisition
        prb += ST(prbr, shifts_ref0[:, j])
    return prb


def adj_holo0(data):
    return _adj_holo0(data, shifts_ref0)


# adjoint test
data = data0[0, :].copy()
ref = ref0.copy()
prb1 = np.array(ref[0, :1]+1j*ref[0, :1]).astype('complex64')
prb1 = np.pad(prb1, ((0, 0), (pad, pad), (pad, pad)))
arr2 = fwd_holo0(prb1)
arr3 = adj_holo0(arr2)

print(f'{np.sum(prb1*np.conj(arr3))}==\n{np.sum(arr2*np.conj(arr2))}')


# #### Approximate the probe by solving the L2-norm minimization problem for reference images

# In[20]:


def line_search(minf, gamma, fu, fd):
    """ Line search for the step sizes gamma"""
    while (minf(fu)-minf(fu+gamma*fd) < 0 and gamma > 1e-12):
        gamma *= 0.5
    if (gamma <= 1e-12):  # direction not found
        # print('no direction')
        gamma = 0
    return gamma


def cg_holo(ref, init_prb,  pars):
    """Conjugate gradients method for holography"""
    # minimization functional
    def minf(fprb):
        f = np.linalg.norm(np.abs(fprb)-ref)**2
        return f

    ref = np.sqrt(ref)
    prb = init_prb.copy()

    for i in range(pars['niter']):
        fprb0 = fwd_holo0(prb)
        gradprb = adj_holo0(fprb0-ref*np.exp(1j*np.angle(fprb0)))

        if i == 0:
            dprb = -gradprb
        else:
            dprb = dai_yuan(dprb,gradprb,gradprb0)
        gradprb0 = gradprb

        # line search
        fdprb0 = fwd_holo0(dprb)
        gammaprb = line_search(minf, pars['gammaprb'], fprb0, fdprb0)
        prb = prb + gammaprb*dprb

        if i % pars['err_step'] == 0:
            fprb0 = fwd_holo0(prb)
            err = minf(fprb0)
            print(f'{i}) {gammaprb=}, {err=:1.5e}')

        # if i % pars['vis_step'] == 0:
        #     mshow_polar(prb[0])

    return prb


rec_prb0 = np.ones([1, n+2*pad, n+2*pad], dtype='complex64')
ref = ref00.copy()
pars = {'niter': 17, 'err_step': 1, 'vis_step': 16, 'gammaprb': 0.5}
rec_prb0 = cg_holo(ref, rec_prb0, pars)


# In[21]:


def line_search(minf, gamma, fu, fu0, fd, fd0):
    """ Line search for the step sizes gamma"""
    while (minf(fu, fu0)-minf(fu+gamma*fd, fu0+gamma*fd0) < 0 and gamma >= 1/64):
        gamma *= 0.5
    if (gamma < 1/64):  # direction not found
        # print('no direction')
        gamma = 0
    return gamma


@gpu_batch
def _gradient(psi, data, shifts_ref, shifts, prb):
    prb = cp.array(prb)
    shifts_ref = cp.array(shifts_ref)
    shifts = cp.array(shifts)
    res = cp.zeros([psi.shape[0], ne, ne], dtype='complex64')
    fpsires = cp.zeros([psi.shape[0], ndist, n, n], dtype='complex64')
    for j in range(ndist):
        # ill shift for each acquisition
        prbr = cp.tile(prb, [psi.shape[0], 1, 1])
        prbr = S(prbr, shifts_ref[:, j])
        # propagate illumination
        prbr = G(prbr, wavelength, voxelsize, distances2[j])
        # object shift for each acquisition
        psir = S(psi, shifts[:, j]/norm_magnifications[j])

        # scale object
        if ne != n:
            psir = M(psir, norm_magnifications[j]*ne/(n+2*pad), n+2*pad)

        # multiply the ill and object
        psir *= prbr

        # propagate both
        psir = G(psir, wavelength, voxelsize, distances[j])
        fpsi = psir[:, pad:n+pad, pad:n+pad]
        fpsires[:, j] = fpsi

        ###########################
        psir = fpsi-data[:, j]*np.exp(1j*(np.angle(fpsi)))

        psir = cp.pad(psir, ((0, 0), (pad, pad), (pad, pad)))

        # propagate data back
        psir = GT(psir, wavelength, voxelsize, distances[j])

        # ill shift for each acquisition
        prbr = cp.tile(prb, [psi.shape[0], 1, 1])
        prbr = S(prbr, shifts_ref[:, j])

        # propagate illumination
        prbr = G(prbr, wavelength, voxelsize, distances2[j])

        # multiply the conj ill and object
        psir *= cp.conj(prbr)

        # scale object
        if ne != n:
            psir = MT(psir, norm_magnifications[j]*ne/(n+2*pad), ne)
        # object shift for each acquisition
        res += ST(psir, shifts[:, j]/norm_magnifications[j])

    # probe normalization
    res /= cp.amax(cp.abs(prb))**2
    return [res, fpsires]


def gradient(psi, data, prb):
    ''' Gradient wrt psi'''
    return _gradient(psi, data, shifts_ref, shifts, prb)


@gpu_batch
def _gradientprb(psi, data, shifts_ref, shifts, prb):
    prb = cp.array(prb)
    shifts_ref = cp.array(shifts_ref)
    shifts = cp.array(shifts)
    res = cp.zeros([psi.shape[0], n+2*pad, n+2*pad], dtype='complex64')
    fpsires = cp.zeros([psi.shape[0], ndist, n, n], dtype='complex64')
    
    for j in range(ndist):
        # ill shift for each acquisition
        prbr = cp.tile(prb, [psi.shape[0], 1, 1])
        prbr = S(prbr, shifts_ref[:, j])
        # propagate illumination
        prbr = G(prbr, wavelength, voxelsize, distances2[j])
        # object shift for each acquisition
        psir = S(psi, shifts[:, j]/norm_magnifications[j])

        # scale object
        if ne != n:
            psir = M(psir, norm_magnifications[j]*ne/(n+2*pad), n+2*pad)

        # multiply the ill and object
        psir *= prbr

        # propagate both
        psir = G(psir, wavelength, voxelsize, distances[j])
        fpsi = psir[:, pad:n+pad, pad:n+pad]
        fpsires[:, j] = fpsi

    ###########################
        fpsi = fpsi-data[:, j]*np.exp(1j*(np.angle(fpsi)))

        prbr = np.pad(fpsi, ((0, 0), (pad, pad), (pad, pad)))
        psir = psi.copy()

        # propagate data back
        prbr = GT(prbr, wavelength, voxelsize, distances[j])

        # object shift for each acquisition
        psir = S(psir, shifts[:, j]/norm_magnifications[j])

        # scale object
        psir = M(psir, norm_magnifications[j]*ne/(n+2*pad), n+2*pad)

        # multiply the conj object and ill
        prbr *= cp.conj(psir)

        # propagate illumination
        prbr = GT(prbr, wavelength, voxelsize, distances2[j])

        # ill shift for each acquisition
        prbr = ST(prbr, shifts_ref[:, j])
        res += prbr

    return [res, fpsires]


def gradientprb(psi, data, prb):
    ''' Gradient wrt prb'''
    [gradprb, fprb] = _gradientprb(psi, data, shifts_ref, shifts, prb)
    gradprb = np.sum(gradprb, axis=0)[np.newaxis]
    return [gradprb, fprb]


def cg_holo(data, ref, init, init_prb,  pars):
    """Conjugate gradients method for holography"""
    # minimization functional
    @gpu_batch
    def _minf(fpsi,data):
        res = cp.empty(data.shape[0],dtype='float32')
        for k in range(data.shape[0]):
            res[k] = np.linalg.norm(cp.abs(fpsi[k])-data[k])**2
        return res
    
    def minf(fpsi,fprb):
        res = np.sum(_minf(fpsi,data))
        if isinstance(fprb, np.ndarray) or isinstance(fprb, cp.ndarray):
            res += np.linalg.norm(np.abs(fprb)-ref)**2
        return res

    #work with sqrt data
    data = np.sqrt(data)
    ref = np.sqrt(ref)

    psi = init.copy()
    prb = init_prb.copy()
    conv = np.zeros(1+pars['niter']//pars['err_step'])

    for i in range(pars['niter']):
        if pars['upd_psi']:
            [grad, fpsi] = gradient(psi, data, prb)
            # Dai-Yuan direction
            if i == 0:
                d = -grad
            else:
                d = dai_yuan(d,grad,grad0)

            grad0 = grad
            fd = fwd_holo(d, prb)
            gammapsi = line_search(minf, pars['gammapsi'], fpsi, 0, fd, 0)
            psi = linear(psi,d,1,gammapsi)

        if pars['upd_prb']:
            [gradprb, fprb] = gradientprb(psi, data, prb)
            fprb0 = fwd_holo0(prb)
            gradprb += adj_holo0(fprb0-ref*np.exp(1j*np.angle(fprb0)))
            gradprb *= 1/(ntheta+1)

            # Dai-Yuan direction
            if i == 0:
                dprb = -gradprb
            else:
                dprb = dai_yuan(dprb,gradprb,gradprb0)
            gradprb0 = gradprb

            # line search
            fdprb = fwd_holo(psi, dprb)
            fdprb0 = fwd_holo0(dprb)
            gammaprb = line_search(minf, pars['gammaprb'], fprb, fprb0, fdprb, fdprb0)
            prb = linear(prb,dprb,1,gammaprb)

        if i % pars['err_step'] == 0:
            fprb = fwd_holo(psi, prb)
            fprb0 = fwd_holo0(prb)
            err = minf(fprb, fprb0)
            conv[i//pars['err_step']] = err
            print(f'{i}) {gammapsi=} {gammaprb=}, {err=:1.5e}')

        if i % pars['vis_step'] == 0:
            # mshow_polar(psi[0])
            # mshow_polar(prb[0])
            dxchange.write_tiff(np.angle(psi[0]),f'/data/vnikitin/ESRF/ID16B/009/033_009_50nm_rec/o{st}/{i:03}.tiff', overwrite=True)
            dxchange.write_tiff(np.angle(psi),f'/data/vnikitin/ESRF/ID16B/009/033_009_50nm_rec/r{st}/{i:03}.tiff', overwrite=True)            
            print(f'/data/vnikitin/ESRF/ID16B/009/033_009_50nm_rec/r{st}/{i:03}')
            
    return psi, prb, conv


# if by chunk on gpu
rec = np.pad(np.exp(1j*recMultiPaganin), ((0, 0), (ne//2-n//2, ne//2-n//2),
             (ne//2-n//2, ne//2-n//2)), 'edge')
rec_prb = rec_prb0.copy()
ref = ref00.copy()
data = data00.copy()


# #if fully on gpu:
# rec = cp.pad(cp.array(np.exp(1j*recMultiPaganin)), ((0, 0), (ne//2-n//2, ne//2-n//2),
#              (ne//2-n//2, ne//2-n//2)), 'edge')
# rec_prb = cp.array(rec_prb0)#*0+1
# ref = cp.array(ref00)#*0+1
# data = cp.array(data00)#*0+cp.array(rdata)

# pars = {'niter': 1024, 'upd_psi': True, 'upd_prb': True,
#         'err_step': 16, 'vis_step': 32, 'gammapsi': 0.5, 'gammaprb': 0.5}
# rec, rec_prb, conv = cg_holo(data, ref, rec, rec_prb, pars)


# ### Hierarchical approach

# In[22]:


# if by chunk on gpu
reci = np.pad(np.exp(1j*recMultiPaganin), ((0, 0), (ne//2-n//2, ne//2-n//2),
             (ne//2-n//2, ne//2-n//2)), 'symmetric')
refi = ref00.copy()
datai = data00.copy()
prbi = rec_prb0.copy()
# prb0[:] = np.abs(rec_prb0)
ni = n
nei = ne
padi = pad
voxelsizei = voxelsize
shiftsi = shifts.copy()
shifts_refi = shifts_ref.copy()


# In[23]:


import scipy as sp
def downsample(data, binning):
    res = data.copy()
    for k in range(binning):
        res = 0.5*(res[..., ::2,:]+res[..., 1::2,:])
        res = 0.5*(res[..., ::2]+res[..., 1::2])        
    return res

lbinninga = [3,2,1,0]
iters = [512,256,128,65]

# lbinninga = [2,1,0]
# iters = [2,2,2,2]

# lbinninga = [0]
# iters = [1]
rec = downsample(reci,lbinninga[0])
prb = downsample(prbi,lbinninga[0])       
for k,lbinning in enumerate(lbinninga):
    n = ni//2**lbinning
    ne = nei//2**lbinning
    pad = padi//2**lbinning
    voxelsize = voxelsizei*2**lbinning
    shifts = shiftsi/2**lbinning
    shifts_ref = shifts_refi/2**lbinning
    print(f'{n=},{ne=},{voxelsize=}')   
    
    data = downsample(datai,lbinning)
    ref = downsample(refi,lbinning)

    pars = {'niter':iters[k], 'upd_psi': True, 'upd_prb': True,
        'err_step': 16, 'vis_step': 32, 'gammapsi': 0.5, 'gammaprb': 0.5}
    print(rec.shape,data.shape,prb.shape,ref.shape)
    rec, prb, conv = cg_holo(data, ref, rec, prb, pars)
    
    rec = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(rec)))
    rec = np.pad(rec,((0,0),(rec.shape[1]//2,rec.shape[1]//2),(rec.shape[1]//2,rec.shape[1]//2)))
    rec = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(rec)))*4
    rec = rec.astype('complex64')

    
    prb = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(prb)))
    prb = np.pad(prb,((0,0),(prb.shape[1]//2,prb.shape[1]//2),(prb.shape[1]//2,prb.shape[1]//2)))
    prb = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(prb)))*4
    prb = prb.astype('complex64')
    
    

