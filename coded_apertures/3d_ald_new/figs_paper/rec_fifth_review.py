#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import cupy as cp
from holotomocupy.holo import G, GT
from holotomocupy.shift import S
from holotomocupy.tomo import R, RT
from holotomocupy.chunking import gpu_batch
from holotomocupy.recon_methods import multiPaganin
from holotomocupy.utils import *
from holotomocupy.proc import linear, dai_yuan
import sys
# cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)
# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().system('jupyter nbconvert --to script rec_fifth_review.ipynb')


# # Init data sizes and parametes of the PXM of ID16A

# In[12]:


n = 256  # object size in each dimension

ntheta = int(sys.argv[4])  # number of angles (rotations)
noise = 0
z1c = -12e-3
# thickness of the coded aperture
code_thickness = 1.5e-6 #in m
# feature size
ill_feature_size = 1e-6 #in m

# ntheta = int(sys.argv[1])  # number of angles (rotations)
# noise = int(sys.argv[2])#sys.argv[2]=='True'
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
flg1 = f'{n}_{ntheta}_{npos}_{z1c}_{noise}_code'
# print(magnifications2,norm_magnifications)
# print(distances2,distances22)
ampshifterr = float(sys.argv[1])
ampcodeerr = float(sys.argv[2])


# ## Read data

# In[13]:


data00 = np.zeros([ntheta, npos, n, n], dtype='float32')
ref0 = np.zeros([1, npos, n, n], dtype='float32')
print(data00.shape)
print(f'/data2/vnikitin/coded_apertures_new3/data/data_{0}_{flg1}.tiff')
for k in range(npos):
    data00[:, k] = read_tiff(f'/data2/vnikitin/coded_apertures_new3/data/data_{k}_{flg1}.tiff')[:ntheta]
for k in range(npos):
    ref0[:, k] = read_tiff(f'/data2/vnikitin/coded_apertures_new3/data/ref_{k}_{flg1}.tiff')[:]
code = np.load(f'/data2/vnikitin/coded_apertures_new3/data/code_{flg1}.npy')
shifts_code = np.load(f'/data2/vnikitin/coded_apertures_new3/data/shifts_code_{flg1}.npy')[:, :npos]

shifts_code += (np.random.random(shifts_code.shape)-0.5)*2*ampshifterr
err_abs = (np.random.random(code.shape)-0.5)*np.abs(code)*2*ampcodeerr
err_angle = (np.random.random(code.shape)-0.5)*np.angle(code)*2*ampcodeerr
code = (np.abs(code)+err_abs)*np.exp(1j*(np.angle(code)+err_angle))
code = code.astype('complex64')
# # code = np.pad(code,((0,0),(ne//2,ne//2),(ne//2,ne//2)),'edge')
# print(code.shape)
# fig, ax = plt.subplots()

# # Display the image
# ax.imshow(data00[0,0],'gray')
# import matplotlib.patches as patches

# rect = patches.Rectangle((128, 128), 256, 256, linewidth=4, edgecolor='r', facecolor='none')
# ax.add_patch(rect)
# data00 = data00[:,:,256-128:256+128,256-128:256+128]
# # data00 = np.pad(data00,((0,0),(0,0),(n//4,n//4),(n//4,n//4)),'symmetric')
# ref0 = ref0[:,:,256-128:256+128,256-128:256+128]
# # ref0 = np.pad(ref0,((0,0),(0,0),(n//4,n//4),(n//4,n//4)),'symmetric')
# #ref0 = ref0[:,:,n//2:-n//2,n//2:-n//2]
# code = code[:,320:-320,320:-320]


# # Construct operators
# 

# In[14]:


@gpu_batch
def _fwd_holo(psi, shifts_code, code, prb):
    #print(psi.shape)
    prb = cp.array(prb)
    code = cp.array(code)

    data = cp.zeros([psi.shape[0], npos, n, n], dtype='complex64')
    for i in range(npos):
        psir = psi.copy()
        prbr = cp.tile(prb, [psi.shape[0], 1, 1])
        coder = cp.tile(code, [psi.shape[0], 1, 1])
        
        # shift and crop the code 
        coder = S(coder, shifts_code[:, i])
        coder = coder[:, coder.shape[1]//2-n//2-pad:coder.shape[1]//2+n//2+pad, coder.shape[1]//2-n//2-pad:coder.shape[1]//2+n//2+pad]
        # multiply by the probe
        coder *= prbr

        # propagate both to the sample plane
        coder = G(coder, wavelength, voxelsize, distances2[i],'symmetric')
        
        # multiply by the sample
        psir *= coder           

        # propagate all to the detector
        psir = G(psir, wavelength, voxelsize, distances[i],'symmetric')
        # mshow_polar(coder[0],show)
        # unpad
        data[:, i] = psir[:, psir.shape[1]//2-n//2:psir.shape[1]//2+n//2, psir.shape[1]//2-n//2:psir.shape[1]//2+n//2]
        
    return data

@gpu_batch
def _adj_holo(data, shifts_code, prb, code):
    prb = cp.array(prb)
    code = cp.array(code)
    shifts_code = cp.array(shifts_code)
    psi = cp.zeros([data.shape[0], ne, ne], dtype='complex64')
    for j in range(npos):
        prbr = cp.tile(prb,[psi.shape[0],1,1])        
        coder = cp.tile(code,[psi.shape[0],1,1])
        psir = cp.pad(data[:, j], ((0, 0), (pad, pad), (pad, pad)))
        psir = GT(psir, wavelength, voxelsize, distances[j],'symmetric')
        coder = S(coder, shifts_code[:,j])            
        coder = coder[:,ne-n//2-pad:ne+n//2+pad,ne-n//2-pad:ne+n//2+pad]        
        coder *= prbr
        coder = G(coder, wavelength, voxelsize, distances2[j],'symmetric')
        psir *= cp.conj(coder)
        psi += psir
    return psi

@gpu_batch
def _adj_holo_prb(data, shifts_code, psi, code):
    psi = cp.array(psi)
    code = cp.array(code)
    shifts_code = cp.array(shifts_code)       
    prb = cp.zeros([data.shape[0], n+2*pad, n+2*pad], dtype='complex64')
    for j in range(npos):
        prbr = np.pad(data[:, j], ((0, 0), (pad, pad), (pad, pad)))
        coder = cp.tile(code,[psi.shape[0],1,1])
        psir = psi.copy()
        prbr = GT(prbr, wavelength, voxelsize, distances[j],'symmetric')
        prbr*=cp.conj(psir)
        prbr = GT(prbr, wavelength, voxelsize, distances2[j],'symmetric')
        coder = S(coder, shifts_code[:,j])            
        coder = coder[:, coder.shape[1]//2-n//2-pad:coder.shape[1]//2+n//2+pad, coder.shape[1]//2-n//2-pad:coder.shape[1]//2+n//2+pad]
        prbr *= cp.conj(coder)
        prb += prbr
    return prb

def fwd_holo(psi, prb):
    return _fwd_holo(psi, shifts_code, code, prb)
def adj_holo(data, prb):
    return _adj_holo(data, shifts_code, prb, code)
def adj_holo_prb(data, psi):
    ''' Adjoint Holography operator '''
    return np.sum(_adj_holo_prb(data, shifts_code, psi, code), axis=0)[np.newaxis]

# adjoint tests
data = data00.copy()
arr1 = cp.pad(cp.array(data[:, 0]+1j*data[:, 0]).astype('complex64'),
              ((0, 0), (ne//2-n//2, ne//2-n//2), (ne//2-n//2, ne//2-n//2)), 'symmetric')

prb1 = cp.ones([1,n+2*pad,n+2*pad],dtype='complex64')
code = cp.array(code)
shifts_code = cp.array(shifts_code)
arr2 = fwd_holo(arr1, prb1)
arr3 = adj_holo(arr2, prb1)
arr4 = adj_holo_prb(arr2, arr1)

print(f'{cp.sum(arr1*cp.conj(arr3))}==\n{cp.sum(arr2*cp.conj(arr2))}')
print(f'{cp.sum(prb1*cp.conj(arr4))}==\n{cp.sum(arr2*cp.conj(arr2))}')

arr1 = arr1.swapaxes(0,1)
a = RT(arr1,theta,ne//2)
b = R(a,theta,ne//2)
c = RT(b,theta,ne//2)
print(f'{cp.sum(arr1*cp.conj(b))}==\n{cp.sum(a*cp.conj(a))}')
print(f'{cp.sum(a*cp.conj(a))}==\n{cp.sum(a*cp.conj(c))}')


# ### Propagate the code to the detector and divide all data by it

# In[15]:


psi = cp.ones([ntheta,ne,ne],dtype='complex64')
prb = cp.ones([1,n+2*pad,n+2*pad],dtype='complex64')
ref0 = cp.array(ref0)
data00 = cp.array(data00)
d = cp.abs(fwd_holo(psi,prb))**2
mshow_polar(code[0],show)
rdata = data00/d/ref0
mshow((rdata)[0,0],show)


# In[16]:


# distances should not be normalized
distances_pag = (distances)[:npos]
recMultiPaganin = np.exp(1j*multiPaganin(rdata,
                         distances_pag, wavelength, voxelsize,  100, 1e-12))
mshow(np.angle(recMultiPaganin[0]),show)


# #### Exponential and logarithm functions for the Transmittance function

# In[17]:


def exptomo(psi):
    """Exp representation of projections"""
    return np.exp(1j*psi * voxelsize * 2*cp.pi / wavelength*np.sqrt(ne*ntheta))
    
def logtomo(psi):
    """Log representation of projections, -i/\nu log(psi)"""
    res = psi.copy()
    res[np.abs(psi) < 1e-32] = 1e-32
    res = np.log(res)
    res = -1j * wavelength / (2*cp.pi) * res / voxelsize/np.sqrt(ne*ntheta)
    return res


# # Operators for the flat field

# In[18]:


@gpu_batch
def _fwd_holo0(prb):
    data = cp.zeros([1, npos, n, n], dtype='complex64')
    for j in range(npos):
        # propagate illumination
        data[:, j] = G(prb, wavelength, voxelsize, distances[0])[:, pad:n+pad, pad:n+pad]
    return data

def fwd_holo0(prb):
    return _fwd_holo0(prb)

@gpu_batch
def _adj_holo0(data):
    prb = cp.zeros([1, n+2*pad, n+2*pad], dtype='complex64')
    for j in range(npos):
        # ill shift for each acquisition
        prbr = cp.pad(data[:, j], ((0, 0), (pad, pad), (pad, pad)))
        # propagate illumination
        prbr = GT(prbr, wavelength, voxelsize, distances[0])
        # ill shift for each acquisition
        prb += prbr
    return prb

def adj_holo0(data):
    return _adj_holo0(data)


# adjoint test
data = data[0, :].copy()
ref = ref0.copy()
prb1 = cp.array(ref[0, :1]+1j*ref[0, :1]).astype('complex64')
prb1 = cp.pad(prb1, ((0, 0), (pad, pad), (pad, pad)))
arr2 = fwd_holo0(prb1)
arr3 = adj_holo0(arr2)


print(f'{np.sum(prb1*np.conj(arr3))}==\n{np.sum(arr2*np.conj(arr2))}')


# In[19]:


def line_search(minf, gamma, fu, fd):
    """ Line search for the step sizes gamma"""
    while(minf(fu)-minf(fu+gamma*fd) < 0 and gamma > 1e-3):
        gamma *= 0.5
    if(gamma <= 1e-3):  # direction not found
        #print(f'{fu.shape} r no direction')
        gamma = 0
    return gamma

def cg_tomo(data, init, pars):
    """Conjugate gradients method for tomogarphy"""
    # minimization functional    
    @gpu_batch
    def _minf(Ru,data):
        res = cp.empty(data.shape[0],dtype='float32')
        for k in range(data.shape[0]):
            res[k] = np.linalg.norm(Ru[k]-data[k])**2
        return res
    
    def minf(Ru):
        res = np.sum(_minf(Ru,data))
        return res
    
    u = init.copy()
    center_pad = u.shape[-1]//2
    for i in range(pars['titer']):
        fu = R(u,theta,center_pad)
        grad = RT(fu-data,theta,center_pad)#/np.float32(np.prod(data.shape[1:]))
        # Dai-Yuan direction
        if i == 0:
            d = -grad
        else:
            d = dai_yuan(d,grad,grad0)

        grad0 = grad
        fd = R(d, theta, center_pad)
        gamma = line_search(minf, pars['gammau'], fu, fd)
        u = linear(u,d,1,gamma)   
        # print('t',i,minf(R(u,theta,center_pad)))
    return u


# In[20]:


def hessian2(psi,ksi,eta,prb,data):
    Lksi = fwd_holo(ksi,prb)
    Leta = fwd_holo(eta,prb)
    Lpsi = fwd_holo(psi,prb)        

    v1 = cp.abs(Lpsi)-data
    v2 = cp.real(cp.conj(Lksi)*Leta)/cp.abs(Lpsi)            
    v3 = cp.real(cp.conj(Lpsi)*Lksi) * cp.real(cp.conj(Lpsi)*Leta) / cp.abs(Lpsi)**3    
    v4 = cp.real(cp.conj(Lpsi)*Lksi)/cp.abs(Lpsi)
    v5 = cp.real(cp.conj(Lpsi)*Leta)/cp.abs(Lpsi)
    return 2*(cp.sum(v1 * cp.conj(v2-v3)) + cp.sum(v4*cp.conj(v5)))


def cg(data, init_u, prb, pars):
    def minf(fpsi):
        f = np.linalg.norm(np.abs(fpsi)-data)**2
        return f

    data = np.sqrt(data)
    u = init_u.copy()    
    conv = np.zeros(pars['niter'])
    step = np.zeros(pars['niter'])    
    center_pad = u.shape[-1]//2
    for i in range(pars['niter']):
        
        # \nabla(F)_X
        eR = cp.exp(1j*R(u,theta,center_pad).swapaxes(0,1))
        Lpsi = fwd_holo(eR,prb)        
        gradx = 2*adj_holo(Lpsi-data*np.exp(1j*np.angle(Lpsi)),prb)        
               
        # \nabla(G)_U0
        grad = cp.conj(eR)*gradx
        grad = -1j*RT(grad.swapaxes(0,1),theta,center_pad)                                
        
        Rgrad = R(grad,theta,center_pad).swapaxes(0,1)
        
        # eta
        if i == 0:
            eta = -grad            
            Reta = -Rgrad
        else:                     
        
            h2u = cp.sum(cp.real(gradx*cp.conj(eR*(1j*Rgrad)*(1j*Reta))))         
            h2u += hessian2(eR,eR*(1j*Rgrad),eR*(1j*Reta),prb,data)

            h2b = cp.sum(cp.real(gradx*cp.conj(eR*(1j*Reta)*(1j*Reta))))         
            h2b += hessian2(eR,eR*(1j*Reta),eR*(1j*Reta),prb,data)

            beta = h2u/h2b
            
            eta = -grad + beta*eta
            Reta = -Rgrad + beta*Reta  

        # hessian
        
        h2 = cp.sum(cp.real(gradx*cp.conj(eR*(1j*Reta)**2)))         
        h2 += hessian2(eR,eR*(1j*Reta),eR*(1j*Reta),prb,data)
                
        gammah = -cp.sum(cp.real(grad*cp.conj(eta)))/h2
        u += gammah*eta
        
        if i % pars['err_step'] == 0:
            eR = cp.exp(1j*R(u,theta,u.shape[-1]//2).swapaxes(0,1))
            Lpsi = fwd_holo(eR,prb)
            err = minf(Lpsi)
            conv[i] = err
            step[i] = gammah
            print(f'{i}), {float(gammah)=} {err=:1.5e}')

        if i % pars['vis_step'] == 0:
            mshow_complex(u[:,ne//2+3,:],show)            
            mshow_complex(u[90,:,:],show)            
            
    return u,conv,step
psirec = cp.pad(cp.array(recMultiPaganin),((0,0),(ne//2-n//2,ne//2-n//2),(ne//2-n//2,ne//2-n//2)),'edge')

#tomo initial guess
xi = 1/1j*cp.log(psirec).swapaxes(0,1)


urec = cp.zeros([ne,ne,ne],dtype='complex64')
pars = {'titer':8, 'gammau':0.5}
urec = cg_tomo(xi,urec,pars)

data = cp.array(data00)
rec_prb = np.load(f'/data2/vnikitin/coded_apertures_new3/data/prb_{flg1}.npy')#[:, :npos]
# rec_prb = rec_prb[:,rec_prb.shape[1]//4:-rec_prb.shape[1]//4,rec_prb.shape[2]//4:-rec_prb.shape[2]//4]
# admm
mshow_polar(rec_prb[0],show)
mshow_polar(code[0],show)
shifts_code = cp.array(shifts_code)
pars = {'niter': int(sys.argv[3]), 'err_step': 1, 'vis_step': 4}

urec,conv,step = cg(data, urec,rec_prb, pars)



# In[ ]:


dxchange.write_tiff(urec.real.get(),
           f'/data2/vnikitin/coded_apertures_new3/rec/rec_{ntheta}_{ampshifterr}_{ampcodeerr}_{pars['niter']}',
           overwrite=True)


# In[ ]:





# 
