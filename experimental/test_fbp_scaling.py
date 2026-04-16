"""
Test FBP vs rec_tomo scaling.

Usage:
    python test_fbp_scaling.py

Applies R to a phantom, then applies FBP and rec_tomo to the sinogram,
and plots f, fbp(Rf), and rec_tomo(Rf) side by side.
"""

import numpy as np
import cupy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from holotomocupy.tomo import Tomo

# --- Parameters ---
n      = 256          # object / detector size
nz     = 1            # single slice
ntheta = 360          # number of angles
theta  = np.linspace(0, np.pi, ntheta, endpoint=False).astype('float32')

# --- Build a simple phantom (filled disk + inner ring) ---
t  = np.linspace(-1, 1, n, endpoint=False)
x, y = np.meshgrid(t, t)
r  = np.sqrt(x**2 + y**2)
f_np = (r < 0.8).astype('float32') * 1.0
f_np[r < 0.4] += 0.5          # brighter inner disk
f_np = f_np[np.newaxis]        # shape [nz=1, n, n]

f_gpu = cp.array(f_np)

# --- Build Tomo ---
cl = Tomo(n, nz, theta, mask_r=1.0)

# --- Forward: d = R(f) ---
d = cl.R(f_gpu)                # [ntheta, nz, n]
print(f"f    : min={float(f_gpu.min()):.4f}  max={float(f_gpu.max()):.4f}  "
      f"norm={float(cp.linalg.norm(f_gpu)):.4f}")
print(f"R(f) : min={float(d.real.min()):.4f}  max={float(d.real.max()):.4f}  "
      f"norm={float(cp.linalg.norm(d)):.4f}")

# --- FBP with different filters ---
fbp_ramp   = cl.fbp(d, filter_name='ramp')
fbp_shepp  = cl.fbp(d, filter_name='shepp')

# --- Plain backprojection RT(d) ---
rt = cl.RT(d)

# --- rec_tomo (CG, 1 iter) ---
rec1 = cl.rec_tomo(d, niter=1)

# --- rec_tomo (CG, 16 iter) ---
rec16 = cl.rec_tomo(d, niter=64)

def _sl(arr):
    """Extract 2D middle slice from [nz,n,n] or [ntheta,nz,n] cupy array."""
    a = arr.real.get()
    if a.ndim == 3:
        return a[a.shape[0]//2]
    return a

# --- Plot ---
imgs = [
    (f_np[0],              'phantom f'),
    (_sl(rt),              'RT(Rf)\n(plain backproj)'),
    (_sl(fbp_ramp),        'FBP ramp\n= RT(|ω|·Rf)'),
    (_sl(fbp_shepp),       'FBP shepp'),
    (_sl(rec1),            'rec_tomo 1 iter'),
    (_sl(rec16),           'rec_tomo 16 iter'),
]

fig, axes = plt.subplots(1, len(imgs), figsize=(4*len(imgs), 4))
for ax, (img, title) in zip(axes, imgs):
    vmax = max(abs(img.max()), abs(img.min()))
    im = ax.imshow(img, cmap='gray', vmin=-vmax, vmax=vmax, origin='lower')
    ax.set_title(title, fontsize=9)
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle(f'n={n}  ntheta={ntheta}  — compare FBP / RT / rec_tomo scaling', fontsize=10)
plt.tight_layout()
out = 'test_fbp_scaling.png'
plt.savefig(out, dpi=120, bbox_inches='tight')
print(f"\nSaved: {out}")

# --- Print amplitude ratios ---
norm_f     = float(cp.linalg.norm(f_gpu))
norm_rt    = float(cp.linalg.norm(rt))
norm_ramp  = float(cp.linalg.norm(fbp_ramp))
norm_shepp = float(cp.linalg.norm(fbp_shepp))
norm_rec1  = float(cp.linalg.norm(rec1))
norm_rec16 = float(cp.linalg.norm(rec16))

print(f"\nNorm ratios  (ratio / norm_f):")
print(f"  RT(Rf)         / f  = {norm_rt    / norm_f:.4f}")
print(f"  FBP ramp       / f  = {norm_ramp  / norm_f:.4f}")
print(f"  FBP shepp      / f  = {norm_shepp / norm_f:.4f}")
print(f"  rec_tomo 1it   / f  = {norm_rec1  / norm_f:.4f}")
print(f"  rec_tomo 16it  / f  = {norm_rec16 / norm_f:.4f}")

print(f"\nNorm ratios  (ratio / norm_rt):")
print(f"  FBP ramp       / RT = {norm_ramp  / norm_rt:.4f}")
print(f"  FBP shepp      / RT = {norm_shepp / norm_rt:.4f}")
print(f"  rec_tomo 1it   / RT = {norm_rec1  / norm_rt:.4f}")
print(f"  rec_tomo 16it  / RT = {norm_rec16 / norm_rt:.4f}")
