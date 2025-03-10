#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cupy as cp
import matplotlib.pyplot as plt
import pandas as pd
from utils import *
import sys
from rec import Rec
from types import SimpleNamespace

import warnings

warnings.filterwarnings("ignore", message=f".*peer.*")


# ## Sizes and propagation settings

# In[2]:


energy = 33.5
wavelength = 1.24e-09 / energy
z1 = -17.75e-3  # [m] position of the sample
detector_pixelsize = 3.03751e-6
focusToDetectorDistance = 1.28  # [m]
# adjustments for the cone beam
z2 = focusToDetectorDistance - z1
distance = (z1 * z2) / focusToDetectorDistance
magnification = focusToDetectorDistance / z1
voxelsize = float(cp.abs(detector_pixelsize / magnification))
path = f"/data/vnikitin/ESRF/ID16A/20240924/SiemensLH/code2um_nfp18x18_01"


# In[3]:


args = SimpleNamespace()

args.ngpus = int(sys.argv[1])
args.lam = float(sys.argv[2])

args.n = 2048
args.npsi = 8192+512
args.pad = int(sys.argv[3])
args.nq = args.n + 2 * args.pad
args.ex = 16
args.npatch = args.nq + 2 * args.ex
args.npos = 18 * 18
args.nchunk = 4

args.voxelsize = voxelsize
args.wavelength = wavelength
args.distance = distance
args.eps = 1e-12
args.rho = [1, 2, 0.1]
args.crop = 2 * args.pad
args.path_out = f"/data/vnikitin/ESRF/ID16A/20240924_rec0224/SiemensLH/code2um_nfp18x18_01/bets_final_{args.pad}_{args.lam}_{args.rho[1]}"


args.niter = 4097
args.err_step = 1
args.vis_step = 128
args.method = "BH-CG"
args.show = False

# create class
cl_rec = Rec(args)


# ## read data

# In[4]:


import h5py

npos = args.npos
pos_step = 1  # steps in positions
with h5py.File(f"{path}/code2um_nfp18x18_010000.h5") as fid:
    data = fid["/entry_0000/measurement/data"][: args.npos].astype("float32")

with h5py.File(f"{path}/ref_0000.h5") as fid:
    ref = fid["/entry_0000/measurement/data"][:].astype("float32")
with h5py.File(f"{path}/dark_0000.h5") as fid:
    dark = fid["/entry_0000/measurement/data"][:].astype("float32")

pos_init = np.loadtxt(
    f"/data/vnikitin/ESRF/ID16A/20240924/positions/shifts_code_nfp18x18ordered.txt"
)[:, ::-1]
pos_init = pos_init / voxelsize * (2048 // args.n) * 1e-6
pos_init[:, 1] *= -1

print(pos_init[-4:])
pos_init = np.load(f"shifts_new.npy")
print(pos_init[-4:])
# centering
pos_init[:, 1] -= (np.amax(pos_init[:, 1]) + np.amin(pos_init[:, 1])) / 2
pos_init[:, 0] -= (np.amax(pos_init[:, 0]) + np.amin(pos_init[:, 0])) / 2
pos_init = pos_init.reshape(int(np.sqrt(args.npos)), int(np.sqrt(args.npos)), 2)
pos_init = pos_init[::pos_step, ::pos_step, :].reshape(args.npos // pos_step**2, 2)
data = data.reshape(int(np.sqrt(args.npos)), int(np.sqrt(args.npos)), args.n, args.n)
data = data[::pos_step, ::pos_step, :].reshape(npos // pos_step**2, args.n, args.n)

ids = np.where(
    (np.abs(pos_init[:, 0]) < args.npsi // 2 - args.n // 2 - args.pad - args.ex)
    * (np.abs(pos_init[:, 1]) < args.npsi // 2 - args.n // 2 - args.pad - args.ex)
)[0]  
data = data[ids]
pos_init = pos_init[ids]

mplot_positions(pos_init,args.show)

npos = len(ids)
args.npos = npos
print(f"{npos=}")


# # remove outliers from data

# In[ ]:


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
data[:, 1320 // 3 : 1320 // 3 + 25 // 3, 890 // 3 : 890 // 3 + 25 // 3] = data[
    :, 1280 // 3 : 1280 // 3 + 25 // 3, 890 // 3 : 890 // 3 + 25 // 3
]
ref[1320 // 3 : 1320 // 3 + 25 // 3, 890 // 3 : 890 // 3 + 25 // 3] = ref[
    1280 // 3 : 1280 // 3 + 25 // 3, 890 // 3 : 890 // 3 + 25 // 3
]

data = remove_outliers(data, 3, 0.8)
ref = remove_outliers(ref[None], 3, 0.8)[0]

data /= np.mean(ref)
ref /= np.mean(ref)

data[np.isnan(data)] = 1
ref[np.isnan(ref)] = 1

mshow(data[0],args.show)
mshow(ref,args.show)


# # initial guess for the object

# In[6]:


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
    for j in range(0, npos):
        r = cp.array(rdata[j])
        r = Paganin(r, wavelength, voxelsize, 24.05, 1e-1)
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
mshow_polar(psi_init,args.show)
mshow_polar(psi_init[:1000, :1000],args.show)

rdata = v = []


# #### Initial guess for the probe calculated by backpropagating the square root of the reference image
# #### Smooth the probe borders for stability

# In[7]:


q_init = cp.array(cl_rec.DT(np.sqrt(ref[np.newaxis]))[0])

ppad = 3 * args.pad // 2
q_init = np.pad(
    q_init[ppad : args.nq - ppad, ppad : args.nq - ppad],
    ((ppad, ppad), (ppad, ppad)),
    "symmetric",
)
v = cp.ones(args.nq, dtype="float32")
vv = cp.sin(cp.linspace(0, cp.pi / 2, ppad))
v[:ppad] = vv
v[args.nq - ppad :] = vv[::-1]
v = cp.outer(v, v)
q_init = cp.abs(q_init * v) * cp.exp(1j * cp.angle(q_init) * v)

mshow_polar(q_init,args.show)


# In[ ]:


# variables
vars = {}
vars["psi"] = cp.array(psi_init)
vars["q"] = cp.array(q_init)
vars["ri"] = np.floor(pos_init).astype("int32")
vars["r"] = np.array(pos_init - vars["ri"]).astype("float32")
vars["table"] = pd.DataFrame(columns=["iter", "err", "time"])

# reconstruction
vars = cl_rec.BH(data, ref, vars)


# In[ ]:


# # results
# erra = vars["table"]["err"].values
# plt.plot(erra)
# plt.yscale("log")
# plt.grid()
# mshow_polar(vars["psi"],args.show)
# mshow_polar(vars["q"],args.show)
# pos_rec = vars["ri"] + vars["r"]
# if args.show:
#     plt.plot((pos_init[:, 1] - pos_rec[:, 1]), ".", label="x difference")
#     plt.plot((pos_init[:, 0] - pos_rec[:, 0]), ".", label="y difference")
#     plt.legend()
#     plt.grid()
#     plt.plot()

