#!/usr/bin/env python
"""
2D near-field ptychography reconstruction for MAX IV / DanMAX scan 096.

Inputs:
  scan-0096.h5 / scan-0096_orca.h5  - positions + data
  scan-0097_orca.h5                 - flat (timescan, no sample)
  scan-0115_orca.h5                 - dark (timescan, no beam)   [see note below]

Note on the dark: user asked for scan-0114 as dark, but 114's mean counts (~7200)
are consistent with a FLAT, not a dark. Only scan-0115 (mean ~200 cts) looks like
a beam-off dark, so this script uses 115. Flip DARK_SCAN if you want 114 instead.

Run:
  python step0.py                      # single-GPU
  mpirun -n <N> python step0.py        # multi-GPU (positions split across ranks)
"""

import os
import numpy as np
import cupy as cp
import h5py
from types import SimpleNamespace
from mpi4py import MPI
from holotomocupy.rec_nfp_mpi import RecNFP
from holotomocupy.utils import logger

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR   = '/data2/maxiv'
DATA_SCAN  = 96
DARK_SCAN  = 115   # ~200 cts   (no beam)     -- see docstring
FLAT_SCAN  = 97    # ~6800 cts  (no sample)

# Beamline geometry (user-provided; same as scan-113)
ENERGY                     = 19.55        # keV
DETECTOR_PIXELSIZE         = 550e-9       # m
Z1                         = 131e-3       # focus-to-sample distance [m]
Z2                         = 1.430        # sample-to-detector distance [m]
FOCUS_TO_DETECTOR_DISTANCE = Z1 + Z2      # m

# Reconstruction
N               = 2048                    # detector crop (n x n, must be /32)
N_POSITIONS     = 32                      # None = all frames; int = N closest to center
NITER           = 193
NCHUNK          = 4
RHO             = [1.0, 2.0, 0.1]         # object, probe, positions
CHECKPOINT_STEP = 4
ERROR_STEP      = 4
SHIFT_TYPE      = 'fft'
PC_SIGMA        = 0.0                     # horizontal partial-coherence Gaussian σ [pixels]; 0 disables
REG_LAMBDA      = 2e-7                    # Tikhonov ||∇proj||² weight; 0 disables. Try 1e-4..1e-2.
LOG_LEVEL       = 'INFO'

OUT_DIR = f'/data2/vnikitin/maxiv/rec_scan-{DATA_SCAN:04d}'
H5_OUT  = os.path.join(OUT_DIR, 'nfp_results.h5')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def orca_path(n): return os.path.join(DATA_DIR, f'scan-{n:04d}_orca.h5')
def meta_path(n): return os.path.join(DATA_DIR, f'scan-{n:04d}.h5')


def read_positions(scan):
    """tom_sam_x, tom_y in mm."""
    with h5py.File(meta_path(scan), 'r') as f:
        x = f['entry/instrument/tom_sam_x/value'][:]
        y = -f['entry/instrument/tom_y/value'][:]
    return x.astype('float32'), y.astype('float32')


def read_mean_frame(scan, sty, stx, n):
    """Mean of a timescan cropped to [sty:sty+n, stx:stx+n]."""
    with h5py.File(orca_path(scan), 'r') as f:
        d = f['entry/instrument/orca/data'][:, sty:sty+n, stx:stx+n]
    return d.mean(axis=0).astype('float32')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

logger.setLevel(LOG_LEVEL)

# --- Geometry ---
wavelength    = 1.24e-9 / ENERGY
magnification = FOCUS_TO_DETECTOR_DISTANCE / Z1
voxelsize     = DETECTOR_PIXELSIZE / magnification
fresnel_z     = Z1 * Z2 / FOCUS_TO_DETECTOR_DISTANCE

# --- Detector geometry ---
with h5py.File(orca_path(DATA_SCAN), 'r') as f:
    ntheta_all, ny, nx = f['entry/instrument/orca/data'].shape

n   = N
sty = (ny - n) // 2
stx = (nx - n) // 2

# --- Positions: read raw, then center (scan-096 has ~0.56 mm x-offset, ~55 mm y-offset) ---
tom_x_mm_all, tom_y_mm_all = read_positions(DATA_SCAN)
x_mean = tom_x_mm_all.mean()
y_mean = tom_y_mm_all.mean()
tom_x_mm_all = tom_x_mm_all - x_mean
tom_y_mm_all = tom_y_mm_all - y_mean

# --- Select subset of frames: N closest to (0,0) after centering ---
if N_POSITIONS is None or N_POSITIONS >= ntheta_all:
    sel = np.arange(ntheta_all, dtype=np.int64)
else:
    r2 = tom_x_mm_all**2 + tom_y_mm_all**2
    sel = np.sort(np.argsort(r2)[:N_POSITIONS]).astype(np.int64)

tom_x_mm = tom_x_mm_all[sel]
tom_y_mm = tom_y_mm_all[sel]
ntheta   = len(sel)

spx = tom_x_mm * 1e-3
spy = tom_y_mm * 1e-3
pos = np.stack([-spy / voxelsize, spx / voxelsize], axis=-1).astype('float32')

pos_range = int(np.ceil(np.abs(pos).max())) + 8
nobj      = int(np.ceil((n + 2 * pos_range) / 32)) * 32

if rank == 0:
    logger.info('=' * 64)
    logger.info(f'MPI ranks               = {comm.Get_size()}')
    logger.info('--- files ---')
    logger.info(f'data                    = {orca_path(DATA_SCAN)}')
    logger.info(f'dark                    = {orca_path(DARK_SCAN)}')
    logger.info(f'flat                    = {orca_path(FLAT_SCAN)}')
    logger.info(f'positions               = {meta_path(DATA_SCAN)}')
    logger.info(f'output dir              = {OUT_DIR}')
    logger.info(f'output h5               = {H5_OUT}')
    logger.info('--- beam / geometry ---')
    logger.info(f'energy                  = {ENERGY} keV')
    logger.info(f'wavelength              = {wavelength*1e12:.4f} pm')
    logger.info(f'z1 (focus-sample)       = {Z1*1e3:.3f} mm')
    logger.info(f'z2 (sample-detector)    = {Z2*1e3:.3f} mm')
    logger.info(f'focus-detector          = {FOCUS_TO_DETECTOR_DISTANCE*1e3:.3f} mm')
    logger.info(f'fresnel_z = z1*z2/zTot  = {fresnel_z*1e3:.3f} mm')
    logger.info(f'detector pixelsize      = {DETECTOR_PIXELSIZE*1e9:.3f} nm')
    logger.info(f'magnification           = {magnification:.4f}')
    logger.info(f'voxelsize               = {voxelsize*1e9:.4f} nm')
    logger.info(f'fov (n * voxelsize)     = {n*voxelsize*1e6:.3f} um')
    logger.info('--- detector / scan ---')
    logger.info(f'raw frame shape         = {ntheta_all} x {ny} x {nx}')
    logger.info(f'crop n                  = {n} (sty={sty}, stx={stx})')
    logger.info(f'ntheta_all              = {ntheta_all}')
    logger.info(f'ntheta used             = {ntheta}  (N_POSITIONS={N_POSITIONS}, closest to centered origin)')
    if ntheta != ntheta_all:
        logger.info(f'selected indices        = {sel.tolist()}')
    logger.info(f'position centering (mm) : x_mean={x_mean:.4f}, y_mean={y_mean:.4f}  (subtracted)')
    logger.info(f'centered positions (mm) : x in [{tom_x_mm.min():.4f}, {tom_x_mm.max():.4f}], '
                f'y in [{tom_y_mm.min():.4f}, {tom_y_mm.max():.4f}]')
    logger.info(f'positions (pix)         : y in [{pos[:,0].min():.1f}, {pos[:,0].max():.1f}], '
                f'x in [{pos[:,1].min():.1f}, {pos[:,1].max():.1f}]')
    logger.info(f'pos_range               = ±{pos_range} pix')
    logger.info(f'nobj                    = {nobj}')
    logger.info('--- reconstruction ---')
    logger.info(f'niter                   = {NITER}')
    logger.info(f'nchunk                  = {NCHUNK}')
    logger.info(f'rho (proj, prb, pos)    = {RHO}')
    logger.info(f'shift_type              = {SHIFT_TYPE}')
    logger.info(f'data_model              = intensity  (pc_sigma={PC_SIGMA} pix, horizontal)')
    logger.info(f'reg_lambda (||∇proj||²) = {REG_LAMBDA}')
    logger.info(f'checkpoint_step         = {CHECKPOINT_STEP}')
    logger.info(f'error_step              = {ERROR_STEP}')
    logger.info(f'obj_dtype               = complex64')
    logger.info('=' * 64)

# --- Init RecNFP ---
if rank == 0:
    os.makedirs(OUT_DIR, exist_ok=True)
comm.Barrier()

rec_args = SimpleNamespace(
    energy                  = ENERGY,
    detector_pixelsize      = DETECTOR_PIXELSIZE,
    focustodetectordistance = FOCUS_TO_DETECTOR_DISTANCE,
    z1                      = Z1,
    ntheta                  = ntheta,
    nz                      = n,
    n                       = n,
    nzobj                   = nobj,
    nobj                    = nobj,
    obj_dtype               = 'complex64',
    rho                     = RHO,
    niter                   = NITER,
    nchunk                  = NCHUNK,
    checkpoint_step         = CHECKPOINT_STEP,
    error_step              = ERROR_STEP,
    shift_type              = SHIFT_TYPE,
    data_model              = 'intensity',
    pc_sigma                = PC_SIGMA,
    reg_lambda              = REG_LAMBDA,
    start_iter              = 0,
    path_out                = OUT_DIR,
    comm                    = comm,
)
cl = RecNFP(rec_args)

# --- Load and flat/dark correct data (per-rank slice of positions) ---
dark_mean = read_mean_frame(DARK_SCAN, sty, stx, n)
flat_mean = read_mean_frame(FLAT_SCAN, sty, stx, n)
denom     = np.clip(flat_mean - dark_mean, 1.0, None)

local_sel = sel[cl.st_theta:cl.end_theta].tolist()
with h5py.File(orca_path(DATA_SCAN), 'r') as f:
    raw = f['entry/instrument/orca/data'][local_sel,
                                          sty:sty+n, stx:stx+n].astype('float32')

corr = (raw - dark_mean) / denom
cl.data[:] = np.maximum(corr, 0)          # intensity (data_model='intensity')

cl.vars['proj'][:] = 0
cl.vars['prb'][:]  = 1
cl.vars['pos'][:]  = cp.array(pos[cl.st_theta:cl.end_theta])

# --- Reconstruct ---
cl.BH()

# --- Gather results on rank 0 ---
pos_err_local = (cl.vars['pos'] - cl.pos_init).get()
all_pos_err   = comm.gather(pos_err_local, root=0)

if rank == 0:
    pos_err = np.concatenate(all_pos_err, axis=0)
    prb_np  = cl.vars['prb'].get()
    proj_np = cl.vars['proj'].get()

    logger.info(f'position errors y (pix): max={np.abs(pos_err[:,0]).max():.4f}, '
                f'mean={np.abs(pos_err[:,0]).mean():.4f}')
    logger.info(f'position errors x (pix): max={np.abs(pos_err[:,1]).max():.4f}, '
                f'mean={np.abs(pos_err[:,1]).mean():.4f}')

    with h5py.File(H5_OUT, 'w') as f:
        f.create_dataset('prb_amp',    data=np.abs(prb_np))
        f.create_dataset('prb_phase',  data=np.angle(prb_np))
        f.create_dataset('proj_delta', data=proj_np.real)
        f.create_dataset('proj_beta',  data=proj_np.imag)
        f.create_dataset('pos_init',   data=cl.pos_init.get())
        f.create_dataset('pos_final',  data=cl.pos_init.get() + pos_err)
        f.create_dataset('pos_err',    data=pos_err)
        f.attrs['energy_keV']         = ENERGY
        f.attrs['voxelsize_m']        = voxelsize
        f.attrs['magnification']      = magnification
        f.attrs['n']                  = n
        f.attrs['nobj']               = nobj
        f.attrs['data_scan']          = DATA_SCAN
        f.attrs['dark_scan']          = DARK_SCAN
        f.attrs['flat_scan']          = FLAT_SCAN
        f.attrs['ntheta']             = ntheta
        f.attrs['ntheta_all']         = ntheta_all
        f.attrs['pos_center_x_mm']    = float(x_mean)
        f.attrs['pos_center_y_mm']    = float(y_mean)
        f.create_dataset('sel_idx', data=sel)
    logger.info(f'saved {H5_OUT}')
