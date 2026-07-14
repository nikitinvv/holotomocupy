"""Generate a 512^3 phantom of 3 concentric hollow vertical cylinders
(equal height), lightly smoothed; compute parallel-beam tomography
projections for 360 angles spanning [0, 2π) via holotomocupy.tomo.Tomo,
and plot slices and sample projections + sinogram."""
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import h5py

from holotomocupy.tomo import Tomo

N = 512

# All concentric on (cy, cx), span the full volume height.
# Each: (r_outer, r_inner, value)
cy, cx = N // 2, N // 2
z0, z1 = 0, N                             # full height
cylinders = [
    (160, 150, 1.0),                      # outer
    (110, 100, 1.0),                      # middle
    ( 60,  50, 1.0),                      # inner
]

yy, xx = np.mgrid[0:N, 0:N]
r2 = (yy - cy) ** 2 + (xx - cx) ** 2
vol = np.zeros((N, N, N), dtype=np.float32)
for r_out, r_in, val in cylinders:
    ring2d = (r2 >= r_in ** 2) & (r2 <= r_out ** 2)
    vol[z0:z1, ring2d] = val

# Light smoothing — softens edges (avoids high-freq artifacts in projections)
vol = gaussian_filter(vol, sigma=0.8).astype(np.float32)

# ------------------------------------------------------------
# Phantom slices
# ------------------------------------------------------------
mid = N // 2
slice_h  = vol[mid, :, :]      # axial / horizontal (xy)
slice_v1 = vol[:, mid, :]      # vertical (xz)
slice_v2 = vol[:, :, mid]      # vertical (yz)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for ax, s, title, xlabel, ylabel in [
    (axs[0], slice_h,  f'horizontal slice  z={mid}  (xy plane)', 'x', 'y'),
    (axs[1], slice_v1, f'vertical slice    y={mid}  (xz plane)', 'x', 'z'),
    (axs[2], slice_v2, f'vertical slice    x={mid}  (yz plane)', 'y', 'z'),
]:
    im = ax.imshow(s, cmap='gray', origin='lower')
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
out_png = '/home/beams2/VNIKITIN/holotomocupy_mpi/tests/sample_3cyl.png'
plt.savefig(out_png, dpi=120); print(f'saved {out_png}')

out_npy = '/home/beams2/VNIKITIN/holotomocupy_mpi/tests/sample_3cyl.npy'
np.save(out_npy, vol)
print(f'saved {out_npy}  shape={vol.shape}  max={vol.max():.3f}')

# ------------------------------------------------------------
# Tomography: parallel-beam Radon at 360 angles in [0, 2π)
# ------------------------------------------------------------
ntheta = 720
theta  = np.linspace(0, 2 * np.pi, ntheta, endpoint=False).astype('float32')

tomo = Tomo(n=N, nz=N, theta=theta, mask_r=0)
Rf = tomo.R(cp.asarray(vol))                   # (ntheta, nz, n) — line integrals
data = cp.exp(-Rf)                             # Beer-Lambert intensity: exp(-Rf)
data_np = cp.asnumpy(data).astype('float32')

# Crop projections to a half-width window with the rotation axis at column
# `axis_pos` in the cropped frame. The Tomo class rotates around column N/2
# of the full projection, so the crop starts at  x0 = N/2 - axis_pos.
axis_pos = 205                                  # desired axis column in the cropped frame
x0       = N // 2 - axis_pos                    # crop start (= 51 for axis_pos=205)
data_np  = np.ascontiguousarray(data_np[:, :N // 2, x0:x0 + N // 2])
Nz_det, Nx_det = data_np.shape[1], data_np.shape[2]
print(f'cropped projections to {data_np.shape}  (rotation axis at column {axis_pos})')

out_proj = '/home/beams2/VNIKITIN/holotomocupy_mpi/tests/sample_3cyl_proj.npy'
np.save(out_proj, data_np)
print(f'saved {out_proj}  shape={data_np.shape}  range=[{data_np.min():.3f}, {data_np.max():.3f}]')

# ------------------------------------------------------------
# Write a DXchange-style HDF5 file (the layout tomocupy expects):
#   /exchange/data        — projections     (ntheta, nz, n) float32
#   /exchange/data_white  — flat fields     (nflat,  nz, n) float32
#   /exchange/data_dark   — dark fields     (ndark,  nz, n) float32
#   /exchange/theta       — rotation angles (ntheta,) float32, **degrees**
#
# Our data is already exp(-Rf) (= normalized intensity, no noise, no offset),
# so we write flats=1 and darks=0; tomocupy's normalization step then yields
#   (data - dark) / (flat - dark) = data = exp(-Rf),
# and the -log brings it back to Rf for reconstruction.
# ------------------------------------------------------------
nflat, ndark = 10, 10
flat = np.ones((nflat, Nz_det, Nx_det), dtype='float32')
dark = np.zeros((ndark, Nz_det, Nx_det), dtype='float32')
theta_deg = np.degrees(theta).astype('float32')

out_h5 = '/home/beams2/VNIKITIN/holotomocupy_mpi/tests/sample_3cyl.h5'
with h5py.File(out_h5, 'w') as f:
    ex = f.create_group('exchange')
    ex.create_dataset('data',       data=data_np,   compression='gzip', compression_opts=4)
    ex.create_dataset('data_white', data=flat,      compression='gzip', compression_opts=4)
    ex.create_dataset('data_dark',  data=dark,      compression='gzip', compression_opts=4)
    ex.create_dataset('theta',      data=theta_deg)
    ex.create_dataset('init',       data=vol,       compression='gzip', compression_opts=4)
print(f'saved {out_h5}  '
      f'(data {data_np.shape} float32, white {flat.shape}, dark {dark.shape}, '
      f'theta {theta_deg.shape} deg, init {vol.shape})')

# ------------------------------------------------------------
# Plot: sample projections (4 angles spanning [0, 2π)) + sinogram of mid-z
# ------------------------------------------------------------
angle_picks = [0, ntheta // 4, ntheta // 2, 3 * ntheta // 4]
sino_z = Nz_det // 2                            # middle of cropped z range
sino_mid = data_np[:, sino_z, :]                # (ntheta, Nx_det)

fig, axs = plt.subplots(1, 5, figsize=(22, 4))
for ax, ai in zip(axs[:4], angle_picks):
    im = ax.imshow(data_np[ai], cmap='gray', origin='lower', aspect='auto',
                   vmin=0, vmax=1)
    deg = np.degrees(theta[ai])
    ax.set_title(f'intensity exp(-Rf)  θ={deg:.1f}°')
    ax.set_xlabel('detector x'); ax.set_ylabel('z')
    plt.colorbar(im, ax=ax, fraction=0.046)

im = axs[4].imshow(sino_mid, cmap='gray', origin='lower', aspect='auto',
                   extent=(0, Nx_det, 0, 360), vmin=0, vmax=1)
axs[4].set_title(f'sinogram exp(-Rf)  z={sino_z}')
axs[4].set_xlabel('detector x'); axs[4].set_ylabel('θ [deg]')
plt.colorbar(im, ax=axs[4], fraction=0.046)

plt.tight_layout()
out_png2 = '/home/beams2/VNIKITIN/holotomocupy_mpi/tests/sample_3cyl_proj.png'
plt.savefig(out_png2, dpi=120); print(f'saved {out_png2}')
