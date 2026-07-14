#!/usr/bin/env python
"""
FFP — rho hyperparameter search (two phases).

Same data sizes as tests/ffp/test.py. Each trial runs a short reconstruction
(niter_trial=8 BH iterations) from an identical initial guess and reports
the final data-fit error via cl.min(). A directional line search picks the
next rho:

    start at rho = 1.0
    evaluate at rho and 2·rho
    if err(2·rho) < err(rho): follow the improving direction (keep doubling)
    else:                     switch to halving (keep dividing by 2)
    stop when the error stops decreasing, or after ~10 evaluations.

Phase 1 — Probe rho (rho[1]):
    * positions FROZEN (init_err = 0; rho[pos] set to a tiny value so BH
      barely moves pos even if it tries).
    * search rho[1] with rho[0] = 1 fixed.

Phase 2 — Position rho (rho[2]):
    * introduce a random per-position error (init_pos_err pix, uniform).
    * use best rho[1] found in Phase 1, rho[0] = 1.
    * search rho[2].

Launch:
    mpirun -n <N> python rho_search.py
"""

import os
import time
import numpy as np
import cupy as cp
import pandas as pd
from scipy.spatial import cKDTree
from types import SimpleNamespace
from mpi4py import MPI

from holotomocupy.rec_ffp_mpi import RecFFP
from holotomocupy.utils       import read_tiff, logger

import logging
logger.setLevel(logging.WARNING)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


# ---------------------------------------------------------------------------
# Config parameters (mirror tests/ffp/test.py)
# ---------------------------------------------------------------------------
n            = 256
nobj         = 2048
ntheta       = 300
overlap      = 0.70
nchunk       = 32
shift_type   = 'fft'

delta_beta   = 29.0
phase_amp    = np.pi / 8
blur_sigma   = 5.0

# --- Search config ---
niter_trial     = 8               # BH iterations per trial
max_evals       = 10              # max rho evaluations per phase
rho_obj         = 1.0             # rho[0], kept fixed throughout
rho_pos_frozen  = 1e-7            # tiny rho[pos] used in Phase 1 (can't be 0)
init_pos_err    = 4.0             # ± pix uniform for Phase 2 init pos error


# ---------------------------------------------------------------------------
# Synthetic object / probe / positions (same generators as test.py)
# ---------------------------------------------------------------------------
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
    return star / (star.max() or 1.0)

def gen_obj(nobj):
    star = cp.array(siemens_star(nobj))
    v  = cp.arange(-nobj // 2, nobj // 2, dtype='float32') / nobj
    vx, vy = cp.meshgrid(v, v)
    g  = cp.fft.fftshift(cp.exp(-8 * (vx ** 2 + vy ** 2)))
    star = cp.fft.ifft2(cp.fft.fft2(star) * g).real
    star = star / (star.max() or 1.0) * phase_amp
    return (-star + 1j * star / delta_beta).astype('complex64')

def load_probe(n):
    _here   = os.path.dirname(os.path.abspath(__file__))
    p_amp   = read_tiff(os.path.join(_here, 'prb_amp_256.tiff')).astype('float32')
    p_phase = read_tiff(os.path.join(_here, 'prb_phase_256.tiff')).astype('float32')
    cy, cx = p_amp.shape[0] // 2, p_amp.shape[1] // 2
    p_amp   = p_amp  [cy - n // 2 : cy + n // 2, cx - n // 2 : cx + n // 2]
    p_phase = p_phase[cy - n // 2 : cy + n // 2, cx - n // 2 : cx + n // 2]
    prb = (p_amp * np.exp(1j * p_phase)).astype('complex64')
    prb /= np.mean(np.abs(prb))
    return prb

def blurred_prb(prb_cpu, sigma):
    v  = np.arange(-n // 2, n // 2, dtype='float32') / n
    vy, vx = np.meshgrid(v, v, indexing='ij')
    gker = np.fft.fftshift(np.exp(-2 * (np.pi * sigma) ** 2 *
                                  (vy ** 2 + vx ** 2)).astype('float32'))
    return (np.fft.ifft2(np.fft.fft2(prb_cpu) * gker)).astype('complex64')

def gen_positions(ntheta, n, overlap):
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
# Build shared ground truth
# ---------------------------------------------------------------------------
prb_gt     = load_probe(n)
prb_init   = cp.array(blurred_prb(prb_gt, sigma=blur_sigma))
prb_gt_cp  = cp.array(prb_gt)
pos_gt     = gen_positions(ntheta, n, overlap)

max_r = (nobj - n) // 2
r_max = float(np.hypot(pos_gt[:, 0], pos_gt[:, 1]).max())
if r_max > max_r:
    raise ValueError(f'spiral radius {r_max:.1f} pix > max {max_r} pix')

# Fixed random pos-error direction (reproducible, scaled by init_pos_err in Phase 2).
_rng     = np.random.default_rng(42)
pos_err_direction = (2 * _rng.random((ntheta, 2)) - 1).astype('float32')

obj_gt = gen_obj(nobj)


# ---------------------------------------------------------------------------
# Instantiate RecFFP once; reuse across all trials
# ---------------------------------------------------------------------------
rec_args = SimpleNamespace(
    energy          = 8.0,
    ntheta          = ntheta,
    nz              = n,
    n               = n,
    nzobj           = nobj,
    nobj            = nobj,
    obj_dtype       = 'complex64',
    rho             = [rho_obj, 1.0, 1.0],       # overwritten per trial
    niter           = niter_trial,
    nchunk          = nchunk,
    checkpoint_step = -1,
    error_step      = -1,                         # no in-loop error prints
    start_iter      = 0,
    shift_type      = shift_type,
    path_out        = None,
    comm            = comm,
)
cl = RecFFP(rec_args)

# Fill vars with GT and generate sqrt-intensity data ONCE (reused every trial).
cl.vars['obj'][:] = obj_gt
cl.vars['prb'][:] = prb_gt_cp
cl.vars['pos'][:] = cp.array(pos_gt[cl.st_theta:cl.end_theta])
cl.gen_sqrt_data(cl.vars, cl.data)


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------
def run_trial(rho, init_err):
    """Reset to initial guess (blurred-GT probe, obj=0, pos=pos_gt+init_err·dir),
    run niter_trial BH iterations at the given rho, return final data-fit error."""
    # Update rho-scaling on the existing instance
    cl.rho_sq = {
        'obj': rho[0] ** 2,
        'prb': rho[1] ** 2,
        'pos': rho[2] ** 2,
    }
    # Reset to initial guess
    cl.vars['obj'][:] = 0
    cl.vars['prb'][:] = prb_init
    pos_local = (pos_gt + init_err * pos_err_direction)[cl.st_theta:cl.end_theta]
    cl.vars['pos'][:] = cp.array(pos_local)
    # Fresh convergence table so BH doesn't append rows across trials
    cl.table = pd.DataFrame(columns=['iter', 'err', 'time'])
    cl.niter      = niter_trial
    cl.start_iter = 0

    cl.BH()

    return cl.min(cl.vars['prb'], cl.vars['obj'], cl.vars['pos'])


# ---------------------------------------------------------------------------
# Directional line search
# ---------------------------------------------------------------------------
def search(evaluate, rho_init=1.0, max_evals=10):
    """
    Bidirectional line search. Evaluate at rho and 2·rho; then explore the
    direction that looked more promising first (doubling if err(2·rho) < err(rho),
    else halving), keep going in that direction until the error stops
    decreasing; then also explore the OTHER direction from the initial
    (rho, 2·rho) pair until it stops decreasing or budget runs out.
    Returns (best_rho, best_err, log[list of (rho, err)]).
    """
    log = []
    def _eval(r):
        e = evaluate(r)
        log.append((float(r), float(e)))
        if rank == 0:
            print(f'    rho = {r:9.4g}   err = {e:.6e}', flush=True)
        return e

    def _explore(start_rho, start_err, factor):
        """From (start_rho, start_err), multiply rho by `factor` while error
        keeps decreasing (and evaluation budget lasts). Returns the best
        (rho, err) found on this branch (== start if no step improved)."""
        cur_rho, cur_err = start_rho, start_err
        while len(log) < max_evals:
            r = cur_rho * factor
            e = _eval(r)
            if e < cur_err:
                cur_rho, cur_err = r, e
            else:
                break
        return cur_rho, cur_err

    rho = float(rho_init)
    err = _eval(rho)
    if len(log) >= max_evals:
        return rho, err, log

    rho_up = rho * 2
    err_up = _eval(rho_up)
    if len(log) >= max_evals:
        best = (rho, err) if err <= err_up else (rho_up, err_up)
        return best[0], best[1], log

    # Explore the promising direction first, then the other one from the
    # initial (rho, rho_up) pair. Continue each branch until error stops
    # improving; both branches share the remaining evaluation budget.
    if err_up < err:
        up_rho, up_err = _explore(rho_up, err_up, 2.0)   # continue doubling
        dn_rho, dn_err = _explore(rho,    err,    0.5)   # then halve from rho
    else:
        dn_rho, dn_err = _explore(rho,    err,    0.5)   # halve from rho
        up_rho, up_err = _explore(rho_up, err_up, 2.0)   # then continue doubling

    candidates = [(rho, err), (rho_up, err_up), (up_rho, up_err), (dn_rho, dn_err)]
    best_rho, best_err = min(candidates, key=lambda p: p[1])
    return best_rho, best_err, log


# ---------------------------------------------------------------------------
# Phase 1 — probe rho (fixed positions)
# ---------------------------------------------------------------------------
if rank == 0:
    print(f'\n[FFP rho search]  n={n} nobj={nobj} ntheta={ntheta} '
          f'nchunk={nchunk} niter/trial={niter_trial}\n')
    print('Phase 1: probe rho (positions frozen: init_err=0, '
          f'rho[pos]={rho_pos_frozen:g})')
t0 = time.time()
best_rho_prb, best_err_prb, log_prb = search(
    lambda r: run_trial([rho_obj, r, rho_pos_frozen], init_err=0.0),
    rho_init=1.0, max_evals=max_evals,
)
if rank == 0:
    print(f'  → best rho[prb] = {best_rho_prb:g}   '
          f'err = {best_err_prb:.6e}   ({time.time()-t0:.1f} s)\n')


# ---------------------------------------------------------------------------
# Phase 2 — position rho (fixed probe rho, position error injected)
# ---------------------------------------------------------------------------
if rank == 0:
    print(f'Phase 2: position rho (init_pos_err=±{init_pos_err} pix, '
          f'rho[prb]={best_rho_prb:g})')
t0 = time.time()
best_rho_pos, best_err_pos, log_pos = search(
    lambda r: run_trial([rho_obj, best_rho_prb, r], init_err=init_pos_err),
    rho_init=1.0, max_evals=max_evals,
)
if rank == 0:
    print(f'  → best rho[pos] = {best_rho_pos:g}   '
          f'err = {best_err_pos:.6e}   ({time.time()-t0:.1f} s)\n')

    print('Summary:')
    print(f'  Phase 1 (probe): rho[prb] = {best_rho_prb:g}   err = {best_err_prb:.6e}')
    print(f'  Phase 2 (pos):   rho[pos] = {best_rho_pos:g}   err = {best_err_pos:.6e}')
    print(f'  → suggested rho = [{rho_obj}, {best_rho_prb:g}, {best_rho_pos:g}]')
