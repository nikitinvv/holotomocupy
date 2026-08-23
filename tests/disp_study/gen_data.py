#!/usr/bin/env python
"""Step 1 of the displacement study — generate synthetic holotomography data
with random sample displacements.

The phantom, the probe and the acquisition geometry are the ones from
../holotomo3d/test.py.  Three study parameters:

* `--amp` - the half-width (in detector pixels) of the uniform random
  displacement applied to the sample at every angle and every distance;
* `--prb-smooth` - the standard deviation (detector px) of the Gaussian blur
  applied to the ID16A probe, i.e. how much structure the illumination still
  has.  0 = the measured probe, the default 0.2251 px = the mild filter
  ../holotomo3d/test.py uses, several px = an almost flat beam.
* `--obj-smooth` - the same, in object voxels, for the phantom itself: how
  sharp the edges are that the reconstruction has to recover.  0 = the raw
  hard-edged shells, the default 0.2251 voxel = the same mild filter.  Has no
  effect with --obj-vol, which uses the volume as it is.

`--ndist 1` (the default) gives the single-distance case the study is about.

Run (one rank per GPU):

    mpirun -np 4 ./set_affinity_gpu.sh python gen_data.py --amp 16 --prb-smooth 2

Output: {out}/amp{amp}_ndist{ndist}[_prbs{s}][_objs{s}]/data.h5  (layout in common.py).
"""

import argparse
import os
import sys
import time

import numpy as np
import cupy as cp
import h5py
from mpi4py import MPI
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'src'))
sys.path.insert(0, _HERE)

# Pinned buffers here are big and long-lived; the cupy pinned pool only adds
# fragmentation on top of that (same reasoning as tests/mosaic_brain/gen_data.py).
cp.cuda.set_pinned_memory_allocator(None)

from holotomocupy.rec_mpi import Rec                              # noqa: E402
from holotomocupy.logger_config import logger, set_log_level      # noqa: E402

import common as C                                                # noqa: E402


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--amp',    type=float, default=12.0,
                   help='half-width of the uniform random displacement [detector px]')
    p.add_argument('--prb-smooth', type=float, default=C.PRB_SMOOTH,
                   help='sigma [detector px] of the Gaussian blur applied to the probe; '
                        '0 = the measured probe, the default reproduces holotomo3d/test.py')
    p.add_argument('--obj-smooth', type=float, default=C.OBJ_SMOOTH,
                   help='sigma [object voxels] of the Gaussian blur applied to the phantom; '
                        '0 = hard edges, the default reproduces holotomo3d/test.py.  '
                        'Ignored with --obj-vol, which takes the volume as it is.')
    p.add_argument('--ndist',  type=int,   default=1,   help='number of propagation distances')
    p.add_argument('--n',      type=int,   default=512, help='detector size [px]')
    p.add_argument('--ntheta', type=int,   default=900, help='number of projection angles')
    p.add_argument('--nobj', type=int, default=0,
                   help='object grid in px, set directly; 0 = derive it from --nobj-factor')
    p.add_argument('--nobj-factor', type=float, default=0,
                   help='object grid as a multiple of n (nobj = round(factor*n/2)*2), used '
                        'only when --nobj is 0; 0 = the tight value n + 2*amp, the smallest '
                        'grid the sliding crop can read without touching mirrored edge data')
    p.add_argument('--delta',  type=float, default=1.0,  help='phantom delta scale')
    p.add_argument('--beta',   type=float, default=1e-2, help='phantom beta scale')
    p.add_argument('--obj-vol', default=None,
                   help='use a real sample volume instead of the synthetic phantom: '
                        'a path, or "path::dataset" for HDF5 (see common.open_volume). '
                        'The file holds the real part of the object directly; the '
                        'imaginary part follows from delta/beta.')
    p.add_argument('--obj-scale', type=float, default=1.0,
                   help='multiplies --obj-vol; the file carries arbitrary grey levels, '
                        'and this is what sets the projected phase excursion (the '
                        '"projected phase" line printed below is the thing to tune it on)')
    p.add_argument('--photons', type=float, default=0.0,
                   help='mean photons/pixel for Poisson noise (0 = noiseless)')
    p.add_argument('--seed',   type=int,   default=10,  help='RNG seed for the displacements')
    p.add_argument('--nchunk', type=int,   default=4,  help='angles per GPU pass')
    p.add_argument('--prb-dir', default=C.PRB_DIR, help='directory with the ID16A probe TIFFs')
    p.add_argument('--phantom-cache', default=None,
                   help='HDF5 file the ground-truth object (phantom or rescaled --obj-vol) '
                        'is cached in, so a sweep builds it only once (default: beside the '
                        "datasets, named for nobj and a hash of everything it depends on; "
                        "'none' rebuilds it every time)")
    p.add_argument('--out',    default=None,
                   help=f'output directory (default {C.OUT_ROOT}/amp<amp>_ndist<ndist>)')
    p.add_argument('--log-level', default='INFO')
    return p.parse_args()


a = parse()
set_log_level(a.log_level)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

n     = a.n
if a.nobj:
    nobj = int(a.nobj) // 2 * 2                  # the grid, in px, as asked for
else:
    nobj = int(round((a.nobj_factor or (n + 2.0 * a.amp) / n) * n / 2)) * 2
nobj_factor = nobj / float(n)
nzobj = nobj
amp_max = (nobj - n) / 2
detector_pixelsize = C.detector_pixelsize(n)
# --obj-vol only: the whole source array is rescaled to the detector width, so
# the sample is whatever fraction of that array is not air; nobj adds the blank
# margin around it.  Plus the delta/beta ratio that turns the volume's
# (already -delta-like) values into obj_im.
span       = n
delta_beta = a.delta / a.beta
out = a.out or os.path.join(C.OUT_ROOT,
                            C.case_name(a.amp, a.ndist, a.prb_smooth, a.obj_smooth))
path = os.path.join(out, 'data.h5')

# The ground-truth object depends only on (nobj, delta, beta) for the phantom,
# and on (source file, nobj, span, scale, delta/beta) for a real volume -- in
# both cases on nothing that varies across a sweep, so the whole sweep can share
# one copy: build it on the first run, read it back on every later one.  Worth
# it either way, but most of all for --obj-vol: rescaling the 3072^3 source
# reads the entire file (~150 s, almost all of it disk) and the dose comparison
# alone generates twice.
# obj_id hashes everything the result depends on -- the phantom's layer list,
# or the source's size/mtime plus the rescale parameters -- so an edited
# phantom or a replaced source file can never make an old cache silently
# reappear.
# next to the datasets (the parent of --out), not in C.OUT_ROOT: run_study.sh
# points --out at its own root, and the file is a full nobj^3 volume
obj_id = (C.volume_id(a.obj_vol, span, a.obj_scale, delta_beta, nobj, nzobj)
          if a.obj_vol else C.phantom_id(smooth=a.obj_smooth))
default_cache = os.path.join(
    os.path.dirname(os.path.normpath(out)),
    (f'objvol_nobj{nobj}_scale{a.obj_scale:g}_{obj_id}.h5' if a.obj_vol else
     f'phantom_nobj{nobj}_delta{a.delta:g}_beta{a.beta:g}_{obj_id}.h5'))
cache = None if a.phantom_cache == 'none' else (a.phantom_cache or default_cache)

if rank == 0:
    os.makedirs(out, exist_ok=True)
    logger.info('=' * 62)
    logger.info(f'  displacement amplitude : +-{a.amp:g} px  (edge-extension limit {amp_max:g} px)')
    logger.info(f'  probe smoothing        : sigma={a.prb_smooth:g} px'
                f'{" (as measured)" if a.prb_smooth <= 0 else ""}')
    if a.obj_vol:
        if abs(a.obj_smooth - C.OBJ_SMOOTH) > 1e-6:
            logger.warning(f'  --obj-smooth {a.obj_smooth:g} is ignored with --obj-vol: the '
                           f'volume is used as it is, only the phantom is filtered')
    else:
        logger.info(f'  object smoothing       : sigma={a.obj_smooth:g} voxel'
                    f'{" (hard edges)" if a.obj_smooth <= 0 else ""}')
    logger.info(f'  geometry               : {C.GEOMETRY}  '
                f'({C.ENERGY:g} keV, focus-to-detector {C.FOCUSTODETECTORDISTANCE:g} m)')
    logger.info(f'  distances              : {a.ndist}  z1={C.Z1_ALL[:a.ndist]*1e3} mm')
    logger.info(f'  detector / object grid : {n} x {n}   /   {nzobj} x {nobj} x {nobj}'
                f'  (nobj = {nobj_factor:.4g} x n)')
    logger.info(f'  detector pixel         : {detector_pixelsize*1e9:.2f} nm '
                f'({C.DETECTOR_NDET // n}x binned)')
    m0  = C.FOCUSTODETECTORDISTANCE / C.Z1_ALL[0]
    vox = detector_pixelsize / m0
    logger.info(f'  object voxel           : {vox*1e9:.2f} nm  '
                f'(x{m0:.1f} magnification, field of view {n*vox*1e6:.1f} um)')
    logger.info(f'  angles                 : {a.ntheta}')
    if a.obj_vol:
        logger.info(f'  sample volume          : {a.obj_vol}')
        logger.info(f'  ... span / scale       : {span} px = {span*vox*1e6:.1f} um  '
                    f'(the source array, rescaled to the detector width)  '
                    f'x {a.obj_scale:g}, delta/beta={delta_beta:g}')
    logger.info(f'  photons per pixel      : {a.photons:g} (0 = noiseless)')
    logger.info(f'  MPI ranks              : {comm.Get_size()}')
    logger.info(f'  output                 : {path}')
    logger.info(f'  object cache           : {cache or "disabled"}'
                f'{"" if not cache or os.path.isfile(cache) else "  (building it)"}')
    logger.info('=' * 62)
    if a.obj_vol and span > nobj:
        logger.warning(f'the sample is {span} px wide but the object grid is only {nobj} px: '
                       f'it is cropped. Raise --nobj to at least {span}.')
    if a.amp > amp_max:
        logger.warning(f'amp={a.amp:g} exceeds (nobj-n)/2={amp_max:g}: the crop will sample '
                       f'mirrored edge data. Increase --nobj-factor.')
comm.Barrier()

# --- Rec instance (used only as the forward operator here) ------------------
args = SimpleNamespace()
args.energy                  = C.ENERGY
args.detector_pixelsize      = C.detector_pixelsize(n)
args.focustodetectordistance = C.FOCUSTODETECTORDISTANCE
args.z1                      = C.Z1_ALL[:a.ndist]
args.theta                   = np.linspace(0, np.pi, a.ntheta, dtype='float32')
args.ndist   = a.ndist
args.ntheta  = a.ntheta
args.nz      = n
args.n       = n
args.nzobj   = nzobj
args.nobj    = nobj
args.mask            = 0.9
args.lam_prbfit      = 0.0
args.lam_laplacian   = 0.0
args.rho             = [1, 0.05, 0.02]
args.niter           = 0
args.nchunk          = a.nchunk
args.checkpoint_step = -1
args.error_step      = -1
args.start_iter      = 0
args.comm            = comm
# generation only touches vars / data / proj_tmp — skip the gradient buffers
args.alloc_mode      = 'gen'

cl = Rec(args)
local_nzobj  = cl.end_obj - cl.st_obj
local_ntheta = cl.end_theta - cl.st_theta

# --- probe and true positions (identical on every rank) ---------------------
prb = C.load_probe(n, a.ndist, a.prb_dir, smooth=a.prb_smooth)
prb_contrast = C.probe_contrast(prb)
if rank == 0:
    logger.info(f'  probe contrast std|prb|/mean|prb| : '
                + ', '.join(f'{c:.4f}' for c in prb_contrast))
pos = C.gen_positions(a.ntheta, a.ndist, a.amp, a.seed)

# --- create the file and write the ground-truth object (rank 0) -------------
# The phantom needs several nobj^3 work arrays, so only rank 0 builds it; the
# other ranks pick up their z-slice from the file.
def fill_object(ds_re, ds_im):
    """Fill /obj_re,/obj_im of the new data file with the ground-truth object.

    With --obj-vol that is a rescaled slice-by-slice copy of a real volume,
    otherwise the synthetic phantom.  Either way it is a deterministic function
    of arguments that do not change across a sweep, so it is built on the first
    call and cached, and later calls copy it back slab by slab (which also keeps
    a cache hit down to one z-batch of memory instead of a whole nobj^3 volume).
    """
    if cache and os.path.isfile(cache):
        with h5py.File(cache, 'r') as g:
            if (int(g.attrs.get('nobj', -1)) == nobj
                    and float(g.attrs.get('delta', np.nan)) == a.delta
                    and float(g.attrs.get('beta',  np.nan)) == a.beta
                    and str(g.attrs.get('phantom_id', '')) == obj_id):
                logger.info(f'reading the object from {cache}')
                batch = C.h5_batch(nobj * nobj * 4)
                for i0 in range(0, nzobj, batch):
                    i1 = min(i0 + batch, nzobj)
                    ds_re[i0:i1] = g['obj_re'][i0:i1]
                    ds_im[i0:i1] = g['obj_im'][i0:i1]
                return
        logger.warning(f'{cache} holds an object for a different (nobj, delta, beta, source) '
                       f'- rebuilding and overwriting it')

    if a.obj_vol:
        # streamed slice by slice: an nobj^3 volume does not fit in memory at n=2048
        logger.info(f'reading the sample volume from {a.obj_vol}')
        C.fill_volume(a.obj_vol, ds_re, ds_im, nzobj, nobj, span,
                      scale=a.obj_scale, delta_beta=delta_beta, log=logger.info)
    else:
        logger.info('generating the phantom')
        obj = C.gen_object(nobj, a.delta, a.beta, smooth=a.obj_smooth)
        ds_re[:] = obj.real
        ds_im[:] = obj.imag
        del obj

    if cache:
        # copied back out of the data file rather than held in memory: the
        # --obj-vol path never has the whole volume at once, and at n=2048 a
        # cache write that did would need 67 GB.
        # via a temporary file: an interrupted run must not leave a half-written
        # cache behind for the next run to read back as if it were complete
        os.makedirs(os.path.dirname(cache) or '.', exist_ok=True)
        tmp = f'{cache}.tmp{os.getpid()}'
        with h5py.File(tmp, 'w') as g:
            g.attrs['nobj']  = nobj
            g.attrs['delta'] = a.delta
            g.attrs['beta']  = a.beta
            g.attrs['phantom_id'] = obj_id
            gre = g.create_dataset('obj_re', (nzobj, nobj, nobj), dtype='float32')
            gim = g.create_dataset('obj_im', (nzobj, nobj, nobj), dtype='float32')
            batch = C.h5_batch(nobj * nobj * 4)
            for i0 in range(0, nzobj, batch):
                i1 = min(i0 + batch, nzobj)
                gre[i0:i1] = ds_re[i0:i1]
                gim[i0:i1] = ds_im[i0:i1]
        os.replace(tmp, cache)
        logger.info(f'object cached -> {cache}  '
                    f'({2 * nzobj * nobj * nobj * 4 / 2**30:.1f} GiB)')


t0 = time.time()
if rank == 0:
    with h5py.File(path, 'w') as f:
        f.attrs['n'] = n; f.attrs['nz'] = n
        f.attrs['nobj'] = nobj; f.attrs['nzobj'] = nzobj
        f.attrs['ntheta'] = a.ntheta; f.attrs['ndist'] = a.ndist
        f.attrs['energy'] = C.ENERGY
        f.attrs['detector_pixelsize'] = detector_pixelsize
        f.attrs['focustodetectordistance'] = C.FOCUSTODETECTORDISTANCE
        f.attrs['amp'] = a.amp; f.attrs['seed'] = a.seed
        f.attrs['photons'] = a.photons
        f.attrs['delta'] = a.delta; f.attrs['beta'] = a.beta
        f.attrs['nobj_factor'] = nobj_factor
        f.attrs['prb_smooth'] = a.prb_smooth
        f.attrs['obj_smooth'] = a.obj_smooth
        if a.obj_vol:
            f.attrs['obj_vol']   = a.obj_vol
            f.attrs['obj_scale'] = a.obj_scale
            f.attrs['obj_span']  = span
        f.attrs['prb_contrast'] = prb_contrast.astype('float32')

        f.create_dataset('theta', data=args.theta)
        f.create_dataset('z1',    data=np.asarray(args.z1))
        f.create_dataset('pos',   data=pos)
        f.create_dataset('prb_abs',   data=np.abs(prb).astype('float32'))
        f.create_dataset('prb_phase', data=np.angle(prb).astype('float32'))
        # datasets filled later / collectively
        f.create_dataset('data', shape=(a.ndist, a.ntheta, n, n), dtype='float32')
        f.create_dataset('ref',  shape=(a.ndist, n, n),           dtype='float32')
        ds_re = f.create_dataset('obj_re', shape=(nzobj, nobj, nobj), dtype='float32')
        ds_im = f.create_dataset('obj_im', shape=(nzobj, nobj, nobj), dtype='float32')
        fill_object(ds_re, ds_im)
    logger.info(f'ground-truth object ready in {time.time()-t0:.1f} s')
comm.Barrier()

# --- every rank loads its own slice of the object ---------------------------
with h5py.File(path, 'r', driver='mpio', comm=comm) as f:
    batch = C.h5_batch(nobj * nobj * 4)
    for i0 in range(0, local_nzobj, batch):
        i1 = min(i0 + batch, local_nzobj)
        sl  = slice(cl.st_obj + i0, cl.st_obj + i1)
        dst = cl.vars['obj'][i0:i1]
        dst.real[:] = f['obj_re'][sl]
        dst.imag[:] = f['obj_im'][sl]

cl.vars['prb'][:] = prb          # prb is numpy; vars['prb'] is pinned numpy
C.set_pos(cl, pos)

# --- forward model ----------------------------------------------------------
logger.info('generating data')
t0 = time.time()
cl.gen_sqrt_data(cl.vars, cl.data)

# The projected phase actually seen by exp(1j*proj), measured rather than
# derived: the Radon normalisation folds n, ntheta and norm_const together, so
# this is the only reliable way to check the magnitude.  A few rad to a few tens
# of rad is a well-conditioned test; rescale --obj-scale if it is off.
pr, pi = cl.vars['proj'].real, cl.vars['proj'].imag
_lo  = comm.allreduce(float(pr.min()), op=MPI.MIN)
_hi  = comm.allreduce(float(pr.max()), op=MPI.MAX)
_alo = comm.allreduce(float(pi.min()), op=MPI.MIN)
_ahi = comm.allreduce(float(pi.max()), op=MPI.MAX)
if rank == 0:
    logger.info(f'projected phase  [{_lo:+.4g}, {_hi:+.4g}] rad')
    logger.info(f'projected absorp [{_alo:+.4g}, {_ahi:+.4g}]   transmission '
                f'exp(-imag) in [{np.exp(-_ahi):.4g}, {np.exp(-_alo):.4g}]')

ref = C.gen_ref(cl, prb)
comm.Barrier()
if rank == 0:
    logger.info(f'forward model done in {time.time()-t0:.1f} s')

# --- noise ------------------------------------------------------------------
if a.photons > 0:
    C.add_poisson_noise(cl.data, a.photons, a.seed + 1000 + rank)
    C.add_poisson_noise(ref,     a.photons, a.seed + 500)

# --- write the measurements -------------------------------------------------
# cl.data is [ndist, local_ntheta, nz, n]; /data has the same distance-major order.
with h5py.File(path, 'a', driver='mpio', comm=comm) as f:
    ds = f['data']
    batch = C.h5_batch(n * n * 4)
    for k in range(a.ndist):
        for i0 in range(0, local_ntheta, batch):
            i1 = min(i0 + batch, local_ntheta)
            ds[k, cl.st_theta + i0:cl.st_theta + i1] = cl.data[k, i0:i1]
comm.Barrier()
if rank == 0:
    with h5py.File(path, 'a') as f:
        f['ref'][:] = ref
comm.Barrier()

if rank == 0:
    d = cl.data
    logger.info(f'rank-0 data range [{float(d.min()):.4f}, {float(d.max()):.4f}] '
                f'(sqrt intensity), mean {float(d.mean()):.4f}')
    logger.info(f'true displacements: max |y| = {np.abs(pos[...,0]).max():.3f} px, '
                f'max |x| = {np.abs(pos[...,1]).max():.3f} px')
    logger.info(f'saved -> {path}  ({os.path.getsize(path)/1024**3:.2f} GB)')
