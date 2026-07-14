#!/usr/bin/env python
"""
FFP step0 — synthetic 2-D far-field ptychography reconstruction.

Structured like experimental/*/step0.py but with inline config (no .conf
file) and synthetic data (there is no real-scan reader for FFP yet).
Reconstructs a complex projected refractive-index object obj = δ + i·β
against a far-field diffraction dataset generated on the fly.

Launch with:
    mpirun -n <N> ./set_affinity_gpu.sh python step0.py
"""

import os
import numpy as np
import cupy as cp
import h5py
from types import SimpleNamespace
from scipy.spatial import cKDTree
from mpi4py import MPI

from holotomocupy.rec_ffp_mpi import RecFFP
from holotomocupy.utils       import read_tiff, logger

import logging
logger.setLevel(logging.INFO)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


# ---------------------------------------------------------------------------
# Config parameters
# ---------------------------------------------------------------------------
# --- Acquisition ---
n          = 256          # detector / probe size (pixels)
nobj       = 2048         # object grid size (pixels)
ntheta     = 300          # number of scan positions
overlap    = 0.70         # target linear probe overlap between neighbours

# --- Reconstruction ---
niter           = 257
nchunk          = 32
rho             = [1.0, 2.0, 0.1]    # [obj, prb, pos] gradient step-size scales
shift_type      = 'cubic'               # 'fft' or 'cubic'
checkpoint_step = 32                  # save tiffs every N iters (-1 = off)
error_step      = 32                  # log error every N iters (-1 = off)

# --- I/O ---
path_out = '/data2/vnikitin/tmp/ffp_step0'
h5_out   = os.path.join(path_out, 'result.h5')


# ---------------------------------------------------------------------------
# Synthetic object / probe / positions
# ---------------------------------------------------------------------------

def siemens_star(nobj, step_deg=15):
    """(nobj, nobj) float32 Siemens-star mask."""
    def rotate_pts(pts, ang, center):
        c, s = np.cos(ang), np.sin(ang)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        return (pts - center) @ R.T + center

    def tri_mask(X, Y, tri):
        (x1, y1), (x2, y2), (x3, y3) = tri
        d = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if d == 0:
            return np.zeros_like(X, dtype=bool)
        a = ((y2 - y3) * (X - x3) + (x3 - x2) * (Y - y3)) / d
        b = ((y3 - y1) * (X - x3) + (x1 - x3) * (Y - y3)) / d
        return (a >= 0) & (b >= 0) & (1 - a - b >= 0)

    tri0 = np.array([
        (1.5 * nobj // 16, nobj // 2 - nobj // 32),
        (1.5 * nobj // 16, nobj // 2 + nobj // 32),
        (nobj // 2 - nobj // 128, nobj // 2),
    ], dtype=np.float32)
    yy, xx = np.mgrid[0:nobj, 0:nobj]
    center = np.array([nobj / 2, nobj / 2], dtype=np.float32)
    star   = np.zeros((nobj, nobj), dtype=np.float32)
    for deg in range(0, 360, step_deg):
        star += tri_mask(xx, yy, rotate_pts(tri0, np.deg2rad(deg), center)).astype(np.float32)
    return star / (star.max() or 1.0)


def gen_obj(nobj):
    """obj = δ + i·β, δ = −phase_amp·star_smoothed, β = δ / delta_beta."""
    star = cp.array(siemens_star(nobj))
    v  = cp.arange(-nobj // 2, nobj // 2, dtype='float32') / nobj
    vx, vy = cp.meshgrid(v, v)
    g  = cp.fft.fftshift(cp.exp(-8 * (vx ** 2 + vy ** 2)))
    star = cp.fft.ifft2(cp.fft.fft2(star) * g).real
    star = star / (star.max() or 1.0) * np.pi / 8
    return (-star + 1j * star / 29).astype('complex64')


def load_probe(n):
    """Ground-truth probe from the 256x256 tiffs next to this file (centred crop for n<256)."""
    _here   = os.path.dirname(os.path.abspath(__file__))
    p_amp   = read_tiff(os.path.join(_here, 'prb_amp_256.tiff')).astype('float32')
    p_phase = read_tiff(os.path.join(_here, 'prb_phase_256.tiff')).astype('float32')
    if p_amp.shape[0] < n or p_amp.shape[1] < n:
        raise ValueError(f'probe tiff ({p_amp.shape}) smaller than n={n}')
    cy, cx = p_amp.shape[0] // 2, p_amp.shape[1] // 2
    p_amp   = p_amp  [cy - n // 2 : cy + n // 2, cx - n // 2 : cx + n // 2]
    p_phase = p_phase[cy - n // 2 : cy + n // 2, cx - n // 2 : cx + n // 2]
    prb = (p_amp * np.exp(1j * p_phase)).astype('complex64')
    prb /= np.mean(np.abs(prb))
    return prb


def blurred_prb(prb_cpu, sigma):
    """Fourier-Gaussian blur — initial guess for the probe."""
    v  = np.arange(-n // 2, n // 2, dtype='float32') / n
    vy, vx = np.meshgrid(v, v, indexing='ij')
    gker = np.fft.fftshift(np.exp(-2 * (np.pi * sigma) ** 2 *
                                  (vy ** 2 + vx ** 2)).astype('float32'))
    return (np.fft.ifft2(np.fft.fft2(prb_cpu) * gker)).astype('complex64')


def gen_positions(ntheta, n, overlap):
    """Fermat spiral calibrated so mean NN distance ≈ (1 − overlap)·n pixels."""
    golden = np.pi * (3 - np.sqrt(5))
    step   = (1.0 - overlap) * n
    idx    = np.arange(ntheta)
    p0     = np.stack([np.sqrt(idx + 0.5) * np.sin(idx * golden),
                       np.sqrt(idx + 0.5) * np.cos(idx * golden)], axis=-1)
    c      = float(step / cKDTree(p0).query(p0, k=2)[0][:, 1].mean())
    return np.stack([c * np.sqrt(idx + 0.5) * np.sin(idx * golden),
                     c * np.sqrt(idx + 0.5) * np.cos(idx * golden)],
                    axis=-1).astype('float32')


# ---------------------------------------------------------------------------
# Build ground truth (same on every rank — small)
# ---------------------------------------------------------------------------
prb_gt   = load_probe(n)
prb_init = blurred_prb(prb_gt, sigma=5.0)
pos_gt   = gen_positions(ntheta, n, overlap)

max_r = (nobj - n) // 2
r_max = float(np.hypot(pos_gt[:, 0], pos_gt[:, 1]).max())
if r_max > max_r:
    raise ValueError(f'spiral radius {r_max:.1f} pix > max scan radius {max_r} pix')

rng     = np.random.default_rng(10)
pos_err = (8 * (2 * rng.random((ntheta, 2)) - 1)).astype('float32')

if rank == 0:
    _nn = cKDTree(pos_gt).query(pos_gt, k=2)[0][:, 1]
    logger.info(f'FFP step0 config:')
    logger.info(f'  n={n}  nobj={nobj}  ntheta={ntheta}  nchunk={nchunk}')
    logger.info(f'  shift_type={shift_type}  overlap≈{(1 - _nn.mean()/n)*100:.1f}%  '
                f'r_max={r_max:.1f} pix (limit {max_r})')
    logger.info(f'  niter={niter}  rho={rho}')
    logger.info(f'  path_out = {path_out}')


# ---------------------------------------------------------------------------
# Init RecFFP
# ---------------------------------------------------------------------------
rec_args = SimpleNamespace(
    energy          = 8.0,          # keV — reporting only (FFP does not use it)
    ntheta          = ntheta,
    nz              = n,
    n               = n,
    nzobj           = nobj,
    nobj            = nobj,
    obj_dtype       = 'complex64',
    rho             = rho,
    niter           = niter,
    nchunk          = nchunk,
    checkpoint_step = checkpoint_step,
    error_step      = error_step,
    start_iter      = 0,
    shift_type      = shift_type,
    path_out        = path_out,
    comm            = comm,
)

cl = RecFFP(rec_args)


# ---------------------------------------------------------------------------
# Generate synthetic sqrt-intensity data
# ---------------------------------------------------------------------------
obj_gt = gen_obj(nobj)

cl.vars['obj'][:] = obj_gt
cl.vars['prb'][:] = cp.array(prb_gt)
cl.vars['pos'][:] = cp.array(pos_gt[cl.st_theta:cl.end_theta])
cl.gen_sqrt_data(cl.vars, cl.data)


# ---------------------------------------------------------------------------
# Reset to initial guess and reconstruct
# ---------------------------------------------------------------------------
cl.vars['obj'][:] = 0                                       # flat δ = β = 0
cl.vars['prb'][:] = cp.array(prb_init)                      # blurred-GT probe
cl.vars['pos'][:] = cp.array((pos_gt + pos_err)[cl.st_theta:cl.end_theta])

cl.BH()


# ---------------------------------------------------------------------------
# Collect position errors (across ranks) and write final HDF5
# ---------------------------------------------------------------------------
pos_final_local = cl.vars['pos'].get()
pos_init_local  = cl.pos_init.get()
pos_drift_local = pos_final_local - pos_init_local
all_pos_final   = comm.gather(pos_final_local, root=0)
all_pos_drift   = comm.gather(pos_drift_local, root=0)

if rank == 0:
    pos_final     = np.concatenate(all_pos_final, axis=0)
    pos_drift     = np.concatenate(all_pos_drift, axis=0)
    pos_recov_err = pos_final - pos_gt

    def _stats(label, e):
        logger.info(f'{label} y (pix): max={np.abs(e[:,0]).max():.4f}  '
                    f'mean={np.abs(e[:,0]).mean():.4f}  std={e[:,0].std():.4f}')
        logger.info(f'{label} x (pix): max={np.abs(e[:,1]).max():.4f}  '
                    f'mean={np.abs(e[:,1]).mean():.4f}  std={e[:,1].std():.4f}')

    _stats('init  err (pos_gt+pos_err − pos_gt)', pos_err)
    _stats('recov err (pos_final     − pos_gt)', pos_recov_err)
    _stats('drift     (pos_final − pos_init)',   pos_drift)

    prb_np = cl.vars['prb'].get()
    obj_np = cl.vars['obj'].get()

    os.makedirs(os.path.dirname(h5_out) or '.', exist_ok=True)
    with h5py.File(h5_out, 'w') as f:
        f.create_dataset('prb_amp',       data=np.abs(prb_np)[None])
        f.create_dataset('prb_phase',     data=np.angle(prb_np)[None])
        f.create_dataset('obj_delta',     data=obj_np.real[None])
        f.create_dataset('obj_beta',      data=obj_np.imag[None])
        f.create_dataset('pos_final',     data=pos_final)
        f.create_dataset('pos_gt',        data=pos_gt)
        f.create_dataset('pos_recov_err', data=pos_recov_err)
        f.create_dataset('pos_drift',     data=pos_drift)
        # ground-truth fields for offline comparison
        f.create_dataset('prb_amp_gt',    data=np.abs(prb_gt)[None])
        f.create_dataset('prb_phase_gt',  data=np.angle(prb_gt)[None])
        f.create_dataset('obj_delta_gt',  data=cp.asnumpy(obj_gt.real)[None])
        f.create_dataset('obj_beta_gt',   data=cp.asnumpy(obj_gt.imag)[None])
    logger.info(f'Saved to {h5_out}')
