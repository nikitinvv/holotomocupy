#!/usr/bin/env python
"""
NFPDelta — Synthetic Self-Test (step0-style: path_out + periodic checkpoints + tiffs).

Same data as test.py, but reconstruction uses RecNFPDelta where
   u = delta * (1 + i * bd)
with delta REAL and bd a scalar (δ/β = 1/bd).

Launch with:
    mpirun -n <N> python test_delta.py
"""

import os
import sys
import subprocess
import numpy as np
import cupy as cp
import h5py
import scipy.ndimage as ndimage
from scipy.fft import fft2, ifft2, fftshift
from types import SimpleNamespace
from mpi4py import MPI

from holotomocupy.rec_nfp_mpi_delta import RecNFPDelta
from holotomocupy.utils import read_tiff, logger

import logging
logger.setLevel(logging.INFO)


# ── Acquisition parameters ───────────────────────────────────────────────────
n      = 1024
ntheta = 16

energy                  = 17.1
detector_pixelsize      = 1.4760147601476e-6 * 4
focustodetectordistance = 1.217
z1                      = 5.110e-3

# ── Run config ──────────────────────────────────────────────────────────────
path_out        = '/data2/vnikitin/tmp/test_nfp_delta_results'
niter           = 1025
nchunk          = 8
checkpoint_step = 32
error_step      = 32
rho             = [1, 2, 0.1, 0.001]   # [proj, prb, pos, bd]
photons         = None                  # mean photons per pixel for Poisson noise; None to disable

# ground-truth δ/β
delta_beta_gt = 29.0
bd_gt         = 1.0 / delta_beta_gt
# initial guess for bd (intentionally off so we can see it converge)
bd_init       = 1.0 / 50.0

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


# ── Phantom object — Siemens star (real delta) ───────────────────────────────
def siemens_star(nobj, step_deg=15):
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
    star /= star.max() or 1.0
    return star


def gen_proj_real(nobj):
    """Return real-valued delta (negative phase, scaled to [-π/4, 0])."""
    star = cp.array(siemens_star(nobj))
    v  = cp.arange(-nobj // 2, nobj // 2, dtype='float32') / nobj
    vx, vy = cp.meshgrid(v, v)
    g  = cp.fft.fftshift(cp.exp(-8 * (vx ** 2 + vy ** 2)))
    star = cp.fft.ifft2(cp.fft.fft2(star) * g).real
    star = star / star.max() * (np.pi / 4)/2
    return (-star).astype('float32')      # delta only — bd handles imag part


# ── Probe — ID16A tiff files ─────────────────────────────────────────────────
_prb_dir = 'data/prb_id16a'
_urls = [
    'https://g-110014.fd635.8443.data.globus.org/holotomocupy/examples_synthetic/data/prb_id16a/prb_abs_2048.tiff',
    'https://g-110014.fd635.8443.data.globus.org/holotomocupy/examples_synthetic/data/prb_id16a/prb_phase_2048.tiff',
]
if rank == 0:
    os.makedirs(_prb_dir, exist_ok=True)
    for url in _urls:
        dest = os.path.join(_prb_dir, os.path.basename(url))
        if not os.path.exists(dest):
            subprocess.run(['wget', '-q', '-O', dest, url], check=True)
comm.Barrier()

prb_abs   = read_tiff(os.path.join(_prb_dir, 'prb_abs_2048.tiff'))[:1]
prb_phase = read_tiff(os.path.join(_prb_dir, 'prb_phase_2048.tiff'))[:1]
prb = (prb_abs * np.exp(1j * prb_phase)).astype('complex64')
prb = prb[:, prb.shape[1] // 2 - n // 2 : prb.shape[1] // 2 + n // 2,
             prb.shape[2] // 2 - n // 2 : prb.shape[2] // 2 + n // 2]

v = np.arange(-n // 2, n // 2, dtype='float32') / n
vx, vy = np.meshgrid(v, v, indexing='ij')
filt = fftshift(np.exp(-4.0 * (vx ** 2 + vy ** 2)).astype('float32'))
prb  = ifft2(fft2(prb) * filt).astype('complex64')
prb = prb[0]
prb /= np.mean(np.abs(prb))
prb_gt = cp.array(prb)


# ── Positions ────────────────────────────────────────────────────────────────
rng     = np.random.default_rng(10)
pos_gt  = (30 / 512 * n * (rng.random((ntheta, 2)) - 0.5)).astype('float32')
pos_err = 4 * (rng.random((ntheta, 2)) - 0.5).astype('float32')

pos_range = int(np.ceil(np.abs(pos_gt + pos_err).max())) + 8
nobj      = int(np.ceil((n + 2 * pos_range) / 32)) * 32
if rank == 0:
    logger.info(f'pos_range = ±{pos_range} pix → nobj = {nobj} (n = {n})')

proj_gt_real = gen_proj_real(nobj)


# ── Initialise RecNFPDelta ───────────────────────────────────────────────────
rec_args = SimpleNamespace(
    energy                  = energy,
    detector_pixelsize      = detector_pixelsize,
    focustodetectordistance = focustodetectordistance,
    z1                      = z1,
    ntheta                  = ntheta,
    nz                      = n,
    n                       = n,
    nzobj                   = nobj,
    nobj                    = nobj,
    obj_dtype               = 'complex64',
    rho                     = rho,
    niter                   = niter,
    nchunk                  = nchunk,
    checkpoint_step         = checkpoint_step,
    error_step              = error_step,
    start_iter              = 0,
    path_out                = path_out,
    comm                    = comm,
)

cl = RecNFPDelta(rec_args)

if rank == 0:
    logger.info(f'nobj={nobj}, n={n}, ntheta={ntheta}, niter={niter}')
    logger.info(f'GT  delta/beta = {delta_beta_gt}  (bd_gt = {bd_gt:.6e})')
    logger.info(f'init delta/beta = {1.0/bd_init:.2f}  (bd_init = {bd_init:.6e})')


# ── Generate synthetic data with GT (proj_real, bd_gt) ───────────────────────
cl.vars['proj'][:] = proj_gt_real
cl.vars['prb'][:]  = prb_gt
cl.vars['pos'][:]  = cp.array(pos_gt[cl.st_theta:cl.end_theta])
cl.vars['bd'][:]   = bd_gt

cl.gen_sqrt_data(cl.vars, cl.data)

# ── Add Poisson noise on the intensity ──────────────────────────────────────
# Per-theta RNG keyed by GLOBAL theta index so each theta's noise realisation
# is identical regardless of how many MPI ranks split the job (reproducibility).
# cl.data is sqrt(intensity): square → Poisson(I·photons)/photons → sqrt.
if photons is not None:
    seeds = np.random.SeedSequence(20251119).spawn(ntheta)
    for j_local in range(cl.end_theta - cl.st_theta):
        rng = np.random.default_rng(seeds[cl.st_theta + j_local])
        I   = cl.data[j_local].astype('float32') ** 2
        I   = rng.poisson(I * photons).astype('float32') / photons
        cl.data[j_local] = np.sqrt(I)
    if rank == 0:
        logger.info(f'Poisson noise: {photons} photons/pixel  '
                    f'(sqrt-data std ≈ 1/(2·sqrt(photons)) = {0.5 / np.sqrt(photons):.4f})')


# ── Reconstruction: reset to initial guess, then BH ──────────────────────────
cl.vars['proj'][:] = 0
cl.vars['prb'][:]  = 1
cl.vars['pos'][:]  = cp.array((pos_gt + pos_err)[cl.st_theta:cl.end_theta])
cl.vars['bd'][:]   = bd_init

cl.BH()


# ── Collect & write final HDF5 ───────────────────────────────────────────────
pos_final_local = cl.vars['pos'].get()
pos_init_local  = cl.pos_init.get()
pos_drift_local = pos_final_local - pos_init_local

all_pos_final = comm.gather(pos_final_local, root=0)
all_pos_drift = comm.gather(pos_drift_local, root=0)
if rank == 0:
    pos_final     = np.concatenate(all_pos_final, axis=0)
    pos_drift     = np.concatenate(all_pos_drift, axis=0)
    pos_recov_err = pos_final - pos_gt

    def _stats(label, e):
        logger.info(f'{label} y (pix): max={np.abs(e[:,0]).max():.4f}  '
                    f'mean={np.abs(e[:,0]).mean():.4f}  std={e[:,0].std():.4f}')
        logger.info(f'{label} x (pix): max={np.abs(e[:,1]).max():.4f}  '
                    f'mean={np.abs(e[:,1]).mean():.4f}  std={e[:,1].std():.4f}')

    _stats('init guess error  (pos_init - pos_gt)',     pos_err)
    _stats('recovered  vs GT  (pos_final - pos_gt)',    pos_recov_err)
    _stats('drift from init   (pos_final - pos_init)',  pos_drift)

    bd_final = float(cl.vars['bd'][0])
    delta_beta_final = 1.0 / bd_final if bd_final != 0 else float('inf')
    logger.info(f'bd final     = {bd_final:.6e}   GT = {bd_gt:.6e}   rel.err = {abs(bd_final-bd_gt)/bd_gt:.3e}')
    logger.info(f'delta/beta   = {delta_beta_final:.3f}  GT = {delta_beta_gt:.3f}')

    prb_np  = cl.vars['prb'].get()
    proj_np = cl.vars['proj'].get()      # REAL delta

    h5_out = os.path.join(path_out, 'result.h5')
    os.makedirs(path_out, exist_ok=True)
    with h5py.File(h5_out, 'w') as f:
        f.create_dataset('prb_amp',       data=np.abs(prb_np)[None])
        f.create_dataset('prb_phase',     data=np.angle(prb_np)[None])
        f.create_dataset('delta',         data=proj_np[None])
        f.create_dataset('delta_gt',      data=cp.asnumpy(proj_gt_real)[None])
        f.create_dataset('bd_final',      data=np.array([bd_final], dtype='float32'))
        f.create_dataset('bd_gt',         data=np.array([bd_gt],    dtype='float32'))
        f.create_dataset('pos_final',     data=pos_final)
        f.create_dataset('pos_gt',        data=pos_gt)
        f.create_dataset('pos_recov_err', data=pos_recov_err)
        f.create_dataset('pos_drift',     data=pos_drift)
    logger.info(f'Saved final result to {h5_out}')
    logger.info(f'Periodic tiffs in       {path_out}/checkpoints_tiff/')
