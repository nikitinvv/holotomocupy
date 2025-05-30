#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cupy as cp
import matplotlib.pyplot as plt
import pandas as pd
import sys
from types import SimpleNamespace
import warnings
warnings.filterwarnings("ignore", message=f".*peer.*")


sys.path.insert(0, '..')
from utils import *
from rec import Rec


# In[ ]:





# ## Sizes and propagation settings

# In[2]:


energy = 33.5
wavelength = 1.24e-09 / energy
z1 = 3.393*1e-3-1.286*1e-3  # [m] position of the sample
detector_pixelsize = 3.03751e-6
focusToDetectorDistance = 1.28  # [m]
# adjustments for the cone beam
z2 = focusToDetectorDistance - z1
distance = (z1 * z2) / focusToDetectorDistance
magnification = focusToDetectorDistance / z1
voxelsize = float(cp.abs(detector_pixelsize / magnification))
path = f"/data/vnikitin/ESRF/ID16A/20240924/Chip/Chip_005nm_155deg_binning1_nfpPSEUDO_RANDOM/"
path2 = f"/data/vnikitin/ESRF/ID16A/20240924/Chip/Chip_005nm_155deg_binning1_nfpPSEUDO_RANDOM_repeat/"
voxelsize


# In[3]:


args = SimpleNamespace()

args.ngpus = 2#int(sys.args[1])
args.lam = 0.0#float(sys.args[2])

args.n = 6144//3
args.pad = 0
args.npsi = args.n+2*args.pad + args.n // 8
args.nq = args.n + 2 * args.pad
args.ex = 16
args.npatch = args.nq + 2 * args.ex
args.npos = 16
args.nchunk = 1

args.voxelsize = voxelsize
args.wavelength = wavelength
args.distance = distance
args.eps = 1e-12
args.rho = [1, 2, 0]
args.crop = 2 * args.pad
args.path_out = f"/data/vnikitin/ESRF/ID16A/20240924_rec0224/Chip_nfp/s22bin/"

args.niter = 10000
args.err_step = 64
args.vis_step = 64
args.method = "BH-CG"
args.show = False

# create class
cl_rec = Rec(args)


# In[ ]:





# ## read data

# In[4]:


import h5py

with h5py.File(f"{path}/Chip_005nm_155deg_binning1_nfpPSEUDO_RANDOM0000.h5") as fid:
    data = fid["/entry_0000/measurement/data"][: args.npos].astype("float32")

with h5py.File(f"{path}/Chip_005nm_155deg_binning1_nfpPSEUDO_RANDOM0000.h5") as fid:
    ref = fid["/entry_0000/measurement/data"][:].astype("float32")
with h5py.File(f"{path}/dark_0000.h5") as fid:
    dark = fid["/entry_0000/measurement/data"][:].astype("float32")

with h5py.File(f'{path}Chip_005nm_155deg_binning1_nfpPSEUDO_RANDOM0000.h5','r') as fid:
    spz = np.array(str(np.array(str(np.array(fid['/entry_0000/instrument/PCIe/header/spz']))[1:]))[1:-1].split(' '),dtype='float32')*1e-6/voxelsize
    spy = np.array(str(np.array(str(np.array(fid['/entry_0000/instrument/PCIe/header/spy']))[1:]))[1:-1].split(' '),dtype='float32')*1e-6/voxelsize
                    

pos_init = np.zeros([args.npos,2],dtype='float32')
pos_init[:,1] = spy
pos_init[:,0] = -spz


# In[5]:


print(pos_init.shape)
# plt.plot(pos_init[:,1],pos_init[:,0],'.')


# # remove outliers from data

# In[6]:


import cupyx.scipy.ndimage as ndimage


def remove_outliers(data, dezinger, dezinger_threshold):
    res = data.copy()
    w = [dezinger, dezinger]
    for k in range(data.shape[0]):
        data0 = cp.array(data[k])
        fdata = ndimage.median_filter(data0, w)
        print(np.sum(np.abs(data0 - fdata) > fdata * dezinger_threshold))
        res[k] = np.where(
            np.abs(data0 - fdata) > fdata * dezinger_threshold, fdata, data0
        ).get()
    return res


dark = np.mean(dark, axis=0)
ref = np.mean(ref, axis=0)
data -= dark
ref -= dark

data[data < 0] = 0
ref[ref < 0] = 0
data[:, 1320 : 1320 + 25 , 890 : 890 + 25 ] = data[
    :, 1280 : 1280 + 25 , 890 : 890 + 25 
]
ref[1320 : 1320 + 25 , 890 : 890 + 25 ] = ref[
    1280 : 1280 + 25 , 890 : 890 + 25 
]

data = remove_outliers(data, 5, 0.995)
ref = remove_outliers(ref[None], 5, 0.995)[0]

data /= np.mean(ref)
ref /= np.mean(ref)

data[np.isnan(data)] = 1
ref[np.isnan(ref)] = 1

mshow(data[0],args.show)
mshow(ref,args.show)

data=(data[:,::3]+data[:,1::3]+data[:,2::3])/3
data=(data[:,:,::3]+data[:,:,1::3]+data[:,:,2::3])/3
dark=(dark[:,::3]+dark[:,1::3]+dark[:,2::3])/3
ref=(ref[:,::3]+ref[:,1::3]+ref[:,2::3])/3
dark=(dark[::3]+dark[1::3]+dark[2::3])/3
ref=(ref[::3]+ref[1::3]+ref[2::3])/3

# # initial guess for the object

# In[7]:


def Paganin(data, wavelength, voxelsize, delta_beta, alpha):
    fx = cp.fft.fftfreq(data.shape[-1], d=voxelsize).astype("float32")
    [fx, fy] = cp.meshgrid(fx, fx)
    rad_freq = cp.fft.fft2(data)
    taylorExp = 1 + wavelength * distance * cp.pi * (delta_beta) * (fx**2 + fy**2)
    numerator = taylorExp * (rad_freq)
    denominator = taylorExp**2 + alpha
    phase = cp.log(cp.real(cp.fft.ifft2(numerator / denominator)))
    phase = delta_beta * 0.5 * phase
    return phase


def rec_init(rdata, ipos_init):
    recMultiPaganin = cp.zeros([args.npsi, args.npsi], dtype="float32")
    recMultiPaganinr = cp.zeros(
        [args.npsi, args.npsi], dtype="float32"
    )  # to compensate for overlap
    for j in range(0, args.npos):
        r = cp.array(rdata[j])
        r = Paganin(r, wavelength, voxelsize, 24.05, 1e-2)
        rr = r * 0 + 1  # to compensate for overlap
        rpsi = cp.zeros([args.npsi, args.npsi], dtype="float32")
        rrpsi = cp.zeros([args.npsi, args.npsi], dtype="float32")
        stx = args.npsi // 2 - ipos_init[j, 1] - args.n // 2
        endx = stx + args.n
        sty = args.npsi // 2 - ipos_init[j, 0] - args.n // 2
        endy = sty + args.n
        rpsi[sty:endy, stx:endx] = r
        rrpsi[sty:endy, stx:endx] = rr

        recMultiPaganin += rpsi
        recMultiPaganinr += rrpsi

    recMultiPaganinr[np.abs(recMultiPaganinr) < 5e-2] = 1
    recMultiPaganin /= recMultiPaganinr
    recMultiPaganin = np.exp(1j * recMultiPaganin)
    return recMultiPaganin


ipos_init = np.round(np.array(pos_init)).astype("int32")
rdata = np.array(data / (ref + 1e-5))
psi_init = rec_init(rdata, ipos_init)
mshow_polar(psi_init,args.show)
mshow_polar(psi_init[:1000, :1000],args.show)

# smooth borders
v = cp.arange(-args.npsi // 2,args.npsi // 2) / args.npsi
[vx, vy] = cp.meshgrid(v, v)
v = cp.exp(-1000 * (vx**2 + vy**2)).astype("float32")

psi_init = cp.fft.fftshift(cp.fft.fftn(cp.fft.fftshift(psi_init)))
psi_init = cp.fft.fftshift(cp.fft.ifftn(cp.fft.fftshift(psi_init * v))).astype(
    "complex64"
)
v=[]
mshow_polar(psi_init,args.show)
mshow_polar(psi_init[:1000, :1000],args.show)

rdata = v = []


# #### Initial guess for the probe calculated by backpropagating the square root of the reference image
# #### Smooth the probe borders for stability

# In[8]:


b = cl_rec.S(pos_init.astype('int32')*0,0*pos_init.astype('float32'),cp.array(psi_init))
a = cl_rec.D(b)
c = cl_rec.DT(a)
print(np.sum(b*c.conj()))
print(np.sum(a*a.conj()))


# In[9]:


q_init = cp.array(cl_rec.DT(np.sqrt(ref[np.newaxis]))[0])
mshow_polar(q_init,args.show)


# In[10]:


# variables

vars = {}
vars["psi"] = cp.array(psi_init)
vars["q"] = cp.array(q_init)
vars["ri"] = np.round(pos_init).astype("int32")
vars["r"] = np.array(pos_init - vars["ri"]).astype("float32")

a = np.load(f'/home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/march2025/Chip_nfp/r.npy')[:16]/3
# plt.plot(a[:,1],a[:,0],'.')
# plt.grid()
vars["r"][:]=a

vars["r_init"] = np.array(pos_init - vars["ri"]).astype("float32")
vars["table"] = pd.DataFrame(columns=["iter", "err", "time"])

# reconstruction
vars = cl_rec.BH(data, vars)

