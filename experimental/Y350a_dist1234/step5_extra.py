#!/usr/bin/env python
"""
Step 5 Extra — Rotation-centre scan on the middle slice.

Stitches the distances, applies multi-distance Paganin on the middle slice only,
then reconstructs that slice with a sweep of rotation-axis offsets.

Output datasets written to the same .h5 file:
  /exchange/rec_centers_{paganin}_{bin}         [n_centers, nobj, nobj]  float32
  /exchange/rec_centers_values_{paganin}_{bin}  [n_centers]              float32
"""

import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as cndimage
import h5py
from holotomocupy.utils import *
from holotomocupy.tomo import Tomo
from holotomocupy.shift import Shift

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
ntheta = 900
st     = 0
bin    = 2
show   = True
rotation_center_shift = -8.780113000000028

paganin     = 20
filter_name = 'ramp'

# Offsets (pixels) relative to rotation_center_shift to scan
center_offsets = np.arange(-5, 6, 1.0)

ids = np.arange(st, 4500, 4500 / ntheta).astype('int')

path_out = '/data3/vnikitin/brain/20251115/Y350a_new_rec'
pfile    = 'Y350a_HT_20nm_8dist'
fpath    = f'{path_out}/{pfile}.h5'

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
with h5py.File(fpath, 'r') as fid:
    detector_pixelsize      = fid['/exchange/detector_pixelsize'][0]
    focusToDetectorDistance = fid['/exchange/focusdetectordistance'][0]
    z1                      = fid['/exchange/z1'][:]
    energy                  = fid['/exchange/energy'][0]
    theta                   = fid['/exchange/theta'][ids, 0]
    theta                   = (-theta / 180 * np.pi).astype('float32')
    shape                   = np.array(fid['/exchange/data0'].shape)

ndist = len(z1)

wavelength          = 1.24e-09 / energy
z2                  = focusToDetectorDistance - z1
magnifications      = focusToDetectorDistance / z1
norm_magnifications = magnifications / magnifications[0]
distances           = (z1 * z2) / focusToDetectorDistance * norm_magnifications**2
voxelsize           = detector_pixelsize / magnifications[0]

n    = shape[1] // (2**bin)
nobj = int(np.ceil(shape[1] / norm_magnifications[-1] / 64)) * 64 // (2**bin)
voxelsize *= 2**bin

print(f'{energy=}')
print(f'{z1=}')
print(f'{focusToDetectorDistance=}')
print(f'{detector_pixelsize=}')
print(f'{magnifications=}')
print(f'{voxelsize=}')
print(f'{distances=}')
print(f'{nobj=}  {n=}')

# ---------------------------------------------------------------------------
# Shifts
# ---------------------------------------------------------------------------
with h5py.File(fpath, 'r') as fid:
    ref = fid[f'/exchange/pref_{bin}'][:ndist].astype('float32')
    r   = (fid['/exchange/cshifts_final'][:] / 2**bin).astype('float32')

s = rotation_center_shift
for _ in range(bin):
    s = (s - 0.5) / 2
r[..., 1] += s

# ---------------------------------------------------------------------------
# Helper: multi-distance Paganin
# ---------------------------------------------------------------------------
def multiPaganin(data, distances, wavelength, voxelsize, delta_beta, alpha):
    fx = cp.fft.fftfreq(data.shape[-1], d=voxelsize).astype('float32')
    fx, fy = cp.meshgrid(fx, fx)
    numerator   = 0
    denominator = 0
    for j in range(data.shape[0]):
        rad_freq   = cp.fft.fft2(data[j].astype('complex64'))
        taylorExp  = 1 + wavelength * distances[j] * cp.pi * delta_beta * (fx**2 + fy**2)
        numerator  += taylorExp * rad_freq
        denominator += taylorExp**2
    numerator   /= len(distances)
    denominator  = denominator / len(distances) + alpha
    phase = cp.log(cp.real(cp.fft.ifft2(numerator / denominator)))
    phase *= delta_beta * 0.5
    return phase

# ---------------------------------------------------------------------------
# Stitch distances + Paganin — middle slice only
# ---------------------------------------------------------------------------
mid  = nobj // 2
pad8 = nobj // 8

cl_shift = Shift(n, nobj, n, nobj, 1 / norm_magnifications, 'complex64')
npad  = n // 16
cref  = cp.array(ref)
r_gpu = cp.array(r)

v = cp.linspace(0, 1, npad, endpoint=False)
v = v**5 * (126 - 420*v + 540*v**2 - 315*v**3 + 70*v**4)

# sinogram of the middle slice after Paganin: [ntheta, nobj]
sino = np.empty([ntheta, nobj], dtype='float32')

srdata    = cp.zeros([ndist, nobj, nobj], dtype='float32')
mm_fixed  = None
global_bg = None

with h5py.File(fpath, 'r') as fid:
    for i, j in enumerate(ids):
        data_j = cp.empty([ndist, n, n], dtype='float32')
        for k in range(ndist):
            data_j[k] = cp.array(fid[f'/exchange/pdata{k}_{bin}'][j].astype('float32'))

        rdata = data_j / (cref + 1e-5)

        for k in range(ndist - 1, -1, -1):
            tmp = rdata[k].astype('complex64')
            tmp = cl_shift.curlySback(cp.log(tmp[None]).astype('complex64'), r_gpu[j:j+1, k], k)[0].real
            tmp /= norm_magnifications[k]**2
            tmp = cp.exp(tmp)

            padx0 = int((nobj - n / norm_magnifications[k]) / 2) - int(r[j, k, 1])
            pady0 = int((nobj - n / norm_magnifications[k]) / 2) - int(r[j, k, 0])
            padx1 = int((nobj - n / norm_magnifications[k]) / 2) + int(r[j, k, 1])
            pady1 = int((nobj - n / norm_magnifications[k]) / 2) + int(r[j, k, 0])
            padx0 = min(nobj, max(0, padx0)) + 5
            pady0 = min(nobj, max(0, pady0)) + 5
            padx1 = min(nobj, max(0, padx1)) + 5
            pady1 = min(nobj, max(0, pady1)) + 5

            tmp = cp.pad(tmp[pady0:-pady1], ((pady0, pady1), (0, 0)), 'edge')
            tmp = cp.pad(tmp[:, padx0:-padx1], ((0, 0), (padx0, padx1)),
                         'linear_ramp', end_values=((1, 1), (1, 1)))

            if k < ndist - 1:
                denom = tmp[pady0:-pady1, padx0:-padx1].mean() + 1e-10
                mmm   = float(srdata[k+1][pady0:-pady1, padx0:-padx1].mean() / denom)
                tmp  *= mmm

                wx = cp.ones(nobj, dtype='float32')
                wy = cp.ones(nobj, dtype='float32')
                wx[:padx0] = 0;  wx[padx0:padx0+npad] = v;  wx[-padx1-npad:-padx1] = 1-v;  wx[-padx1:] = 0
                wy[:pady0] = 0;  wy[pady0:pady0+npad] = v;  wy[-pady1-npad:-pady1] = 1-v;  wy[-pady1:] = 0

                w   = cp.outer(wy, wx)
                tmp = tmp * w + srdata[k+1] * (1 - w)
            srdata[k] = tmp

        # Paganin on the middle row only (pad to a thin 2D patch)
        pj_mid = srdata[:, mid:mid+1, :]   # [ndist, 1, nobj]
        if i == 0:
            mm_fixed  = float(srdata[:, :32 * n // 512, :32 * n // 512].mean())
            pj_pad    = cp.pad(pj_mid, ((0, 0), (pad8, pad8), (pad8, pad8)),
                               'constant', constant_values=mm_fixed)
            ph0       = multiPaganin(pj_pad, distances / norm_magnifications**2,
                                     wavelength, voxelsize, paganin, 1e-5)
            global_bg = float(cp.median(ph0[pad8, pad8:pad8 + nobj]))

        pj_pad = cp.pad(pj_mid, ((0, 0), (pad8, pad8), (pad8, pad8)),
                        'constant', constant_values=mm_fixed)
        phase  = multiPaganin(pj_pad, distances / norm_magnifications**2,
                              wavelength, voxelsize, paganin, 1e-5)
        sino[i] = phase[pad8, pad8:pad8 + nobj].get()

        if i % 300 == 0:
            print(f'proj {i}/{ntheta}')
            mshow_complex(srdata[0] + 1j * srdata[ndist - 1], show)

sino -= global_bg

sino_complex = (sino + 1j * sino / paganin).astype('complex64')  # [ntheta, nobj]
mshow_complex(cp.array(sino_complex).T + 0j, show)

# ---------------------------------------------------------------------------
# FBP sweep over rotation centres
# ---------------------------------------------------------------------------
nchunk  = 1
cl_tomo = Tomo(nobj, nchunk, theta, mask_r=0.9)

sino_re_gpu = cp.array(sino_complex.real)  # [ntheta, nobj]
sino_im_gpu = cp.array(sino_complex.imag)

recs = np.zeros([len(center_offsets), nobj, nobj], dtype='float32')

for i, dc in enumerate(center_offsets):
    re_shifted = cndimage.shift(sino_re_gpu, [0, dc])
    im_shifted = cndimage.shift(sino_im_gpu, [0, dc])
    sino_shifted = (re_shifted + 1j * im_shifted).astype('complex64')  # [ntheta, nobj]

    rec_slice = cl_tomo.fbp(sino_shifted[:, None, :], filter_name)  # [1, nobj, nobj]
    recs[i] = rec_slice[0].real.get()

    center_val = rotation_center_shift + dc
    print(f'dc={dc:+.1f}  center={center_val:.3f}')
    mshow_complex(cp.array(recs[i]) + 0j, show)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
center_values = (rotation_center_shift + center_offsets).astype('float32')

with h5py.File(fpath, 'a') as fid:
    for key, data in [
        (f'/exchange/rec_centers_{paganin}_{bin}',        recs),
        (f'/exchange/rec_centers_values_{paganin}_{bin}', center_values),
    ]:
        if key in fid:
            del fid[key]
        fid.create_dataset(key, data=data)

print(f'Saved {len(center_offsets)} reconstructions to {fpath}')
print(f'Rotation centres: {center_values}')
