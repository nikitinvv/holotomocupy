#!/usr/bin/env python
"""
YY037A 5-tile mosaic pipeline — Steps 1–5 (MPI + GPU parallel).

Structure follows experimental/Y350c_new/steps15.py. Steps 1–4 run per tile
inside a loop over ``tiles`` (e.g. ``farright, right, center, left, farleft``),
each writing its own ``.h5`` file:

    1  convert EDF → HDF5
    2  outlier removal + intensity normalisation
    3  combine encoder / RHAPP / motion / 3-D shifts → cshifts_final
    4  binned data: multi-distance alignment + amplitude correction
       → pdata{k}_{bin}, pref_{bin}   (tile-local object grid, nobj_tile)

Step 5 then runs ONCE over all tiles: each tile is stitched across distances
on its own object grid, the results are composited onto one wide mosaic grid
(nzobj x nobj), Paganin phase retrieval runs on the wide grid, and FBP
follows for the bin levels that fit in memory.

Deltas versus Y350c_new/steps15.py:
  * Per-tile loop wrapping Steps 1–4.
  * Geometry is read from the ID16A ``.info`` file + ``angles_file.txt``
    rather than a per-distance HDF5 scan file; ``refHST`` is used as the
    pre-averaged flat and ``dark*.edf`` as darks (matches YY037A layout).
  * Encoder shifts in Step 3 come from
    ``{path}/{pfile_tile}/projections/{pfile_tile}_{k:04d}.txt``
    (guarded — falls back to zeros with a warning).
  * Step 5 is a mosaic: it composites the tiles and matches their grey levels
    across the seams. Mind the volume — it is nobj**2 * nzobj voxels, i.e.
    3.4 TB at bin 0 and 52 GB at bin 2, so use ``start_level_rec`` to pick
    which bins are actually reconstructed.
  * Where each tile sits on the mosaic is measured inside Step 5, not supplied.
    It starts from the nominal ``tile_step`` and, with ``estimate_overlap``,
    correlates each pair of neighbours once their distances have been assembled
    into one projection — the point at which magnification and per-projection
    motion are already out of the picture and placement is all that is left.
    Done per bin, coarsest first, each bin refining the one before it.
  * The old adjacent-tile phase-correlation diagnostic on raw frames is still
    there but off by default (``overlap_check=true``).

Launch with:
    mpirun -n <N> python steps15.py config_steps15.conf
"""

import sys
import logging
import h5py
import fabio
logging.getLogger('fabio').setLevel(logging.ERROR)
import glob
import os
import time
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as ndimage
from mpi4py import MPI
from holotomocupy.shift import Shift
from holotomocupy.tomo import Tomo
from holotomocupy.chunking import Chunking
from holotomocupy.mpi_functions import MPIClass, get_local_chunk
from holotomocupy.logger_config import logger, set_log_level
from holotomocupy.config import parse_args_steps15
from holotomocupy.reader import (load_octave_text_mat, load_shrink_from_mats,
                                 load_shrink_profile, load_scan_infos)
from holotomocupy.utils import *

args = parse_args_steps15(sys.argv[1])
start_step            = args.start_step
start_level_rec       = args.start_level_rec
rotation_center_shift = args.rotation_center_shift
nlevels               = args.nlevels
paganin               = args.paganin
nchunk                = args.nchunk
ref_dist              = args.ref_dist
set_log_level(args.log_level)

path        = args.path + '/'
pfile_base  = args.pfile
scan_suffix = args.scan_suffix
tiles       = args.tiles if args.tiles else [""]
tile_order  = args.tile_order
overlap_check   = args.overlap_check
overlap_width   = args.overlap_width
overlap_nangles = args.overlap_nangles
estimate_overlap = args.estimate_overlap
tile_step        = args.tile_step
ntheta_rec       = args.ntheta_rec

# --- TEMPORARY: placement-error sweep ---------------------------------------
# HOLO_TILE_ERR=<px> nudges one tile horizontally by that many object px on the
# finest grid, on top of whatever step 5 measured, and makes step 5 drop the
# middle FBP slice as a tiff. HOLO_TILE_PROC picks which tile — a name from
# `tiles=` in the config, or an index; unset means the middle one. One image
# per (tile, error) pair; see run.sh for the full sweep.
#
#   for e in $(seq -10 1 10); do
#     HOLO_TILE_ERR=$e HOLO_TILE_PROC=center \
#       mpirun -n 8 python steps15.py config_steps15.conf
#   done
#
# The files are named fbp_{tile}_{error+100}.tiff, so they group by tile and
# sort in error order with no minus sign: -10 -> 90, 0 -> 100, +10 -> 110.
#
# HOLO_TILE_ERR unset = normal run, nothing extra written. Delete this block
# when the sweep has served its purpose.
_tile_err  = os.environ.get('HOLO_TILE_ERR')
tile_err   = float(_tile_err) if _tile_err is not None else None
tile_err_dir = os.environ.get('HOLO_TILE_ERR_DIR', '/data2/vnikitin/tmp')

# Which tile carries the error. Resolved here, against the same `tiles` list the
# rest of the script uses, so a typo fails immediately instead of silently
# sweeping the wrong tile.
_tile_proc = os.environ.get('HOLO_TILE_PROC')
if _tile_proc is None:
    proc_tile = len(tiles) // 2
elif _tile_proc.lstrip('+-').isdigit():
    proc_tile = int(_tile_proc) % len(tiles)
elif _tile_proc in tiles:
    proc_tile = tiles.index(_tile_proc)
else:
    raise SystemExit(f'HOLO_TILE_PROC={_tile_proc!r} is neither an index nor '
                     f'one of the configured tiles {tiles}')

_script_dir = os.path.dirname(os.path.abspath(__file__))

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Assign one GPU per rank (round-robin if fewer GPUs than ranks)
ngpus = cp.cuda.runtime.getDeviceCount()
cp.cuda.Device(rank % ngpus).use()


# ---------------------------------------------------------------------------
# Helpers — ID16A .info + angles_file.txt
# ---------------------------------------------------------------------------

def _pfile_tile(tile):
    """Compose the per-tile prefix used to build directory / file names."""
    if not tile:
        return pfile_base
    if scan_suffix:
        return f'{pfile_base}_{tile}_{scan_suffix}'
    return f'{pfile_base}_{tile}'


def nominal_tile_offsets(ntiles, step):
    """Nominal tile placement: a uniform row, ``step`` object px apart.

    ``tiles`` is listed left to right on the mosaic, and the offsets are
    *minus* the frame position (see ``s_kernel``), so they run downwards with
    the tile index and are centred on zero. Vertical is nominally zero — the
    stage only moves horizontally between tiles.

    Only a starting guess; ``estimate_overlap`` measures the rest from the data.
    """
    idx = np.arange(ntiles) - (ntiles - 1) / 2
    out = np.zeros([ntiles, 2], dtype='float32')
    out[:, 1] = -idx * step
    return out


def phase_corr_2d(A, B, max_shift=0):
    """Subpixel phase correlation of two equal-sized cupy images.

    Returns ``((dv, dh), peak)`` where ``B`` looks like ``A`` displaced by
    ``+d``: a feature at index ``i`` of ``A`` sits at ``i + d`` in ``B``. The
    peak height (0..1) says how much of the correlation energy landed in that
    one pixel, so it doubles as a confidence measure.

    ``max_shift`` restricts the search to that many px around zero, which keeps
    a periodic sample from locking onto the wrong repeat.
    """
    ny, nx = A.shape
    wy = cp.hanning(ny).astype('float32')[:, None]
    wx = cp.hanning(nx).astype('float32')[None, :]
    FA = cp.fft.rfft2((A - A.mean()) * wy * wx)
    FB = cp.fft.rfft2((B - B.mean()) * wy * wx)
    R  = FA.conj() * FB
    R /= cp.abs(R) + 1e-9
    c  = cp.fft.irfft2(R, s=(ny, nx))

    if max_shift:
        m = int(min(max_shift, min(ny, nx) // 2 - 1))
        keep = cp.zeros((ny, nx), dtype=bool)
        for ys in (slice(0, m + 1), slice(ny - m, ny)):
            for xs in (slice(0, m + 1), slice(nx - m, nx)):
                keep[ys, xs] = True
        c = cp.where(keep, c, float(c.min()) - 1.0)

    iy, ix = (int(v) for v in cp.unravel_index(cp.argmax(c), c.shape))
    c0 = float(c[iy, ix])

    def _sub(cm, cc, cpp):
        den = cm - 2 * cc + cpp
        return 0.0 if abs(den) < 1e-12 else float(np.clip(0.5 * (cm - cpp) / den,
                                                          -0.5, 0.5))

    dy = _sub(float(c[(iy - 1) % ny, ix]), c0, float(c[(iy + 1) % ny, ix]))
    dx = _sub(float(c[iy, (ix - 1) % nx]), c0, float(c[iy, (ix + 1) % nx]))
    dv = (iy + dy + ny / 2) % ny - ny / 2
    dh = (ix + dx + nx / 2) % nx - nx / 2
    return (float(dv), float(dh)), c0


def combine_seam_shifts(meas, ntiles, ref=None, min_peak=0.02, log=None):
    """Turn per-angle seam measurements into a per-tile correction.

    ``meas[s]`` is the list of ``(dv, dh, peak)`` measured on the seam between
    tiles ``s`` and ``s+1``, one entry per angle. Each is a *residual*: how far
    tile ``s+1`` still sits from where the current offsets put it.

    The angles are combined with a median (one bad projection should not move
    a seam) and the residuals then accumulate along the chain — a seam that is
    off displaces every tile downstream of it. A chain only fixes the tiles
    relative to each other, so the result is anchored on tile ``ref``
    (the middle one by default): that tile does not move, and the rotation axis
    stays where the reference tile's geometry puts it.

    Returns ``(corr, per_seam)``: an ``(ntiles, 2)`` correction in whatever px
    units the measurements were made in, and a
    ``(dv, dh, sem_v, sem_h, nused, peak)`` tuple per seam for logging.
    """
    if ref is None:
        ref = ntiles // 2
    per_seam, step = [], np.zeros([ntiles - 1, 2])
    for s in range(ntiles - 1):
        v = np.array([m for m in meas[s] if m[2] >= min_peak],
                     dtype='float64').reshape(-1, 3)
        if len(v) == 0:
            per_seam.append((0.0, 0.0, np.nan, np.nan, 0, 0.0))
            if log:
                log(f'  seam {s}->{s+1}: no usable angle '
                    f'(peak < {min_peak}), residual left at zero')
            continue
        step[s] = np.median(v[:, :2], axis=0)
        # MAD -> sem, so a couple of outliers do not inflate it
        mad = np.median(np.abs(v[:, :2] - step[s]), axis=0) * 1.4826
        sem = mad / np.sqrt(len(v))
        per_seam.append((step[s, 0], step[s, 1], sem[0], sem[1],
                         len(v), float(np.median(v[:, 2]))))

    corr = np.zeros([ntiles, 2])
    for s in range(ntiles - 1):
        corr[s + 1] = corr[s] + step[s]
    corr -= corr[ref]                        # anchor: the reference tile stays put
    return corr, per_seam


def paste_geometry(off_bin, nobj_tile_bin, nzobj_bin, nobj_bin):
    """Split binned tile offsets into a paste origin and a sub-px remainder.

    ``r`` is minus the position of the frame, so moving ``off`` out of ``r``
    moves the paste origin by ``-off``. Only the integer part can be a paste;
    the fraction has to go back into ``r``.
    """
    ioff     = np.round(off_bin).astype(int)
    frac     = off_bin - ioff
    origin_y = (nzobj_bin - nobj_tile_bin) // 2 - ioff[:, 0]
    origin_x = (nobj_bin - nobj_tile_bin) // 2 - ioff[:, 1]
    return frac, origin_y, origin_x


path_out = args.path_out if args.path_out else path.rstrip('/') + '_rec'
if rank == 0:
    os.makedirs(path_out, exist_ok=True)
comm.Barrier()


# ===========================================================================
# Per-tile loop: Steps 1 → 2 → 3
# ===========================================================================

# Cached across the tile loop for Steps 4–5 (identical for every tile: the
# tiles are the same optics at a different stage position, and Step 1 checks
# the .info geometry of each one).
ntheta = ndist = n = nobj_tile = None
norm_magnifications = distances = None
voxelsize = wavelength = None
local_ids = None

# Shrinkage is NOT identical for every tile — each tile is its own scan with
# its own shapp.mat — so it is kept per tile and indexed by tile in Step 5,
# which runs after the loop. `shrink_nd` is the current tile's, for Step 4.
shrink_nd  = None
shrink_all = [None] * len(tiles)


# ---------------------------------------------------------------------------
# Shrinkage baseline across tiles
#
# The sample shrinks for as long as the session lasts, not just for as long as
# one tile takes: by the time the last tile is acquired the sample is already
# smaller than it was for the first. Each tile's shapp.mat measures only its own
# scan and starts from zero, so on its own it describes a sample that resets
# between tiles. Every tile therefore gets an offset — what the sample had
# already shrunk before it — which lifts its curve without touching its shape,
# and the shape stays the direct measurement.
#
# The tiles were acquired one at a time — every distance of a tile before the
# next tile started — so the offset is one constant per tile: the running sum of
# the tiles' own totals, taken in ACQUISITION order. That order is not the
# mosaic order and is not reliably recoverable from the files, so it is a plain
# config parameter:
#
#     tile_order=center,left,right,farright,farleft        (YY037A)
#
# Applied to `shrink_nd` inside the loop, i.e. before Step 4 resamples with it
# and before Step 3 writes /exchange/shrink — so step 6 reads the accumulated
# values too, and its `tp` fit starts from the right offset.
#
# The time between tiles counts for nothing, by assumption: the sample only
# deforms while it is being scanned, so the (long) idle gaps add nothing.
# ---------------------------------------------------------------------------

shrink_base = [np.zeros(2, dtype='float32') for _ in tiles]
if len(tiles) > 1:
    if not tile_order:
        raise SystemExit(
            'tile_order= is empty. With more than one tile every tile\'s shrink '
            'has to be offset by what the sample had already shrunk before it, '
            'which needs the order the tiles were ACQUIRED in — not the mosaic '
            f'order tiles={tiles}. For YY037A: '
            'tile_order=center,left,right,farright,farleft')
    if sorted(tile_order) != sorted(tiles):
        raise SystemExit(
            f'tile_order={tile_order} must be a permutation of tiles={tiles} — '
            f'it is the same tiles in acquisition order, so every tile has to '
            f'appear exactly once.')

    _running = np.zeros(2, dtype='float32')
    _report  = []
    for _tl in tile_order:
        _t     = tiles.index(_tl)
        _pf    = _pfile_tile(_tl)
        _infos = load_scan_infos(path, _pf)
        _, _total = load_shrink_profile(path, _pf, len(_infos),
                                        int(_infos[0]['TOMO_N']))
        shrink_base[_t] = _running.copy()
        _running        = _running + _total
        _report.append((_tl, shrink_base[_t], _total))
    if rank == 0:
        logger.info('=' * 78)
        logger.info(f'Shrink chained over the tiles in acquisition order '
                    f'{",".join(tile_order)}:')
        for _tl, _base, _total in _report:
            logger.info(f'  {_tl:<10s} baseline v={_base[0]:+.6f} '
                        f'h={_base[1]:+.6f}   own total '
                        f'v={_total[0]:+.6f} h={_total[1]:+.6f}')
        logger.info(f'  end of session       v={_running[0]:+.6f} '
                    f'h={_running[1]:+.6f}')
        logger.info('=' * 78)

for tile_idx, tile in enumerate(tiles):
    pfile   = _pfile_tile(tile)
    dname0  = f'{path}/{pfile}_1_'
    fpath   = f'{path_out}/{pfile}.h5'

    if rank == 0:
        logger.info('=' * 74)
        logger.info(f'TILE  {tile or "(single)"}   pfile={pfile}')
        logger.info('=' * 74)

    # ---- Geometry from .info files (one per distance dir) -----------------
    infos = load_scan_infos(path, pfile)
    ndist = len(infos)

    ntheta = int(infos[0]['TOMO_N'])
    energy = float(infos[0]['Energy'])                              # keV

    # ``Optic_used`` is the DETECTOR pixel — the optic's pixel size at the
    # detector, the same for every distance (the ESRF octave script carries it
    # as pixelsize_detector = 2.95203e-06). ``PixelSize`` in the same .info is
    # already the *object* pixel: it varies per distance exactly as
    # Optic_used / magnification, so using it here divides by the magnification
    # a second time and shrinks the voxel ~29x.
    detector_pixelsize = float(infos[0]['Optic_used']) * 1e-6       # µm → m

    # The .info geometry, as ht_*.m reads it: z1h = -SourceDistance/1000 and
    # z2h = Distance/1000. So ``Distance`` is the sample-to-detector distance,
    # NOT the focus-to-detector one — the latter is z1 + z2, which comes out
    # constant (1282.00 mm here) as it must, while ``Distance`` alone varies
    # with the sample position. ``SourceDistance`` is already measured from the
    # focus, so there is no separate sx0 to subtract (contrast the NX pipelines,
    # where z1 = sx - sx0).
    sx0 = None
    z1  = np.array([-float(i['SourceDistance']) * 1e-3 for i in infos], dtype='float64')
    z2  = np.array([ float(i['Distance']) * 1e-3       for i in infos], dtype='float64')
    focustodetectordistance = float((z1 + z2).mean())

    wavelength          = 1.24e-09 / energy
    z2                  = focustodetectordistance - z1
    magnifications      = focustodetectordistance / z1
    norm_magnifications = magnifications / magnifications[0]
    distances           = (z1 * z2) / focustodetectordistance * norm_magnifications**2
    voxelsizes          = np.abs(detector_pixelsize / magnifications)
    voxelsize           = voxelsizes[0]

    # Cross-check: with both of the above right, Optic_used / magnification must
    # reproduce the .info PixelSize of every distance. It does, to ~3 ppm.
    _esrf = np.array([float(i['PixelSize']) * 1e-6 for i in infos], dtype='float64')
    if rank == 0 and np.max(np.abs(voxelsizes / _esrf - 1)) > 0.01:
        logger.warning(f'voxelsize disagrees with the .info PixelSize by '
                       f'{np.max(np.abs(voxelsizes / _esrf - 1)) * 100:.2f}% — check the geometry: '
                       f'ours {np.array2string(voxelsizes * 1e9, precision=3)} nm vs '
                       f'{np.array2string(_esrf * 1e9, precision=3)} nm')

    # Offset by everything the sample had already shrunk before this tile was
    # acquired (see the block above). Broadcasts over angles and distances:
    # (ntheta, ndist, 2) + (2,). Zero for the tile acquired first.
    shrink_nd = load_shrink_from_mats(path, pfile, ndist, ntheta,
                                      angle_ramp=args.shrink_angle_ramp)
    shrink_nd = shrink_nd + shrink_base[tile_idx]
    shrink_all[tile_idx] = shrink_nd
    shrink    = shrink_nd[0]

    # n from actual EDF size, overrideable via config n=
    n0, n1 = fabio.open(f'{dname0}/refHST0000.edf').data.shape
    n = args.n if args.n is not None else n0
    sty, endy = n0 // 2 - n // 2, n0 // 2 + n // 2
    stx, endx = n1 // 2 - n // 2, n1 // 2 + n // 2

    # YY037A uses refHST (pre-averaged flat, one per angle position) →
    # nref = 1 in the y350c convention. Darks are dark*.edf.
    nref  = 1
    ndark = len(glob.glob(f'{dname0}/dark[0-9]*.edf'))

    nobj_tile = args.nobj_tile if args.nobj_tile is not None else int(np.ceil(n / norm_magnifications[-1] / 64)) * 64

    if rank == 0:
        logger.info(f'path                    = {path}')
        logger.info(f'pfile                   = {pfile}')
        logger.info(f'ntheta                  = {ntheta}')
        logger.info(f'energy                  = {energy} keV')
        logger.info(f'detector_pixelsize      = {detector_pixelsize*1e6:.5f} um (Optic_used)')
        logger.info(f'voxelsize               = {voxelsize*1e9:.3f} nm '
                    f'(.info PixelSize says {float(infos[0]["PixelSize"])*1e3:.3f} nm)')
        logger.info(f'voxelsizes              = {np.array2string(voxelsizes*1e9, precision=3)} nm')
        logger.info(f'focustodetectordistance = {focustodetectordistance*1e3:.3f} mm (z1 + Distance)')
        logger.info(f'z1                      = {np.array2string(z1*1e3, precision=4)} mm')
        logger.info(f'z2                      = {np.array2string(z2*1e3, precision=4)} mm')
        logger.info(f'magnifications          = {np.array2string(magnifications, precision=4)}')
        logger.info(f'norm_magnifications     = {np.array2string(norm_magnifications, precision=6)}')
        logger.info(f'distances (propagation) = {np.array2string(distances*1e3, precision=4)} mm')
        logger.info(f'ndist={ndist}  n={n}  nobj_tile={nobj_tile}  nref={nref}  ndark={ndark}')
        logger.info(f'shrink_v cumulative     = {[round(float(s), 6) for s in shrink[:, 0]]}')
        logger.info(f'shrink_h cumulative     = {[round(float(s), 6) for s in shrink[:, 1]]}')
    comm.Barrier()

    # Distribute ntheta projections across ranks
    ids_per_rank = np.array_split(np.arange(ntheta), size)
    local_ids    = ids_per_rank[rank]
    local_start  = int(local_ids[0])
    local_end    = int(local_ids[-1]) + 1
    logger.info(f'[{tile}] theta-range [{local_start}:{local_end}), '
                f'local_ntheta={local_end - local_start}')

    # -----------------------------------------------------------------------
    # STEP 1: Convert EDF → HDF5
    # -----------------------------------------------------------------------
    if start_step > 1:
        if rank == 0:
            logger.info(f'[{tile}] Step 1: skipped.')
        comm.Barrier()
    else:
        if rank == 0:
            logger.info(f'[{tile}] Step 1: converting EDF files to HDF5...')

        # Angles from angles_file.txt (first ntheta rows; extras at end are
        # return-to-start frames we discard). Sign convention matches y350c:
        # we store the raw values and negate at read time (see reader.py).
        ang_path = f'{dname0}/angles_file.txt'
        if rank == 0:
            theta_vals = np.loadtxt(ang_path, dtype='float32')[:ntheta]
            theta_vals = -theta_vals   # match y350c_new/steps15.py:230

        with h5py.File(fpath, 'w', driver='mpio', comm=comm) as fid:
            data_ds   = [fid.create_dataset(f'/exchange/data{k}',             shape=(ntheta, n, n), dtype='uint16') for k in range(ndist)]
            white0_ds = [fid.create_dataset(f'/exchange/data_white_start{k}', shape=(nref,  n, n),  dtype='uint16') for k in range(ndist)]
            white1_ds = [fid.create_dataset(f'/exchange/data_white_end{k}',   shape=(nref,  n, n),  dtype='uint16') for k in range(ndist)]
            dark_ds   = [fid.create_dataset(f'/exchange/data_dark{k}',        shape=(ndark, n, n),  dtype='uint16') for k in range(ndist)]
            theta_ds  = fid.create_dataset('/exchange/theta',  shape=(ntheta, ndist), dtype='float32')
            vs_ds     = fid.create_dataset('/exchange/voxelsize',             shape=voxelsizes.shape, dtype='float32')
            z1_ds     = fid.create_dataset('/exchange/z1',                    shape=z1.shape,         dtype='float32')
            dpx_ds    = fid.create_dataset('/exchange/detector_pixelsize',    shape=(1,),             dtype='float32')
            en_ds     = fid.create_dataset('/exchange/energy',                shape=(1,),             dtype='float32')
            fdd_ds    = fid.create_dataset('/exchange/focusdetectordistance', shape=(1,),             dtype='float32')

            if rank == 0:
                vs_ds[:]    = voxelsizes
                z1_ds[:]    = z1
                dpx_ds[:]   = [detector_pixelsize]
                en_ds[:]    = [energy]
                fdd_ds[:]   = [focustodetectordistance]
                theta_ds[:] = theta_vals[:, None]

            for k in range(ndist):
                dname = f'{path}/{pfile}_{k + 1}_'

                if rank == 0:
                    # refHST is the averaged flat at each angle position
                    # (0 and ntheta) — one file per position.
                    white0_ds[k][0] = fabio.open(f'{dname}/refHST0000.edf').data[sty:endy, stx:endx]
                    white1_ds[k][0] = fabio.open(f'{dname}/refHST{ntheta:04d}.edf').data[sty:endy, stx:endx]
                    for id in range(ndark):
                        dark_ds[k][id] = fabio.open(f'{dname}/dark{id:04d}.edf').data[sty:endy, stx:endx]

                norms = np.empty(len(local_ids), dtype='float64')
                for ii, id in enumerate(local_ids):
                    fname = f'{dname}/{pfile}_{k + 1}_{id:04}.edf'
                    frame = fabio.open(fname).data[sty:endy, stx:endx]
                    data_ds[k][id] = frame
                    norms[ii] = np.linalg.norm(frame)
                    if ii % 100 == 0:
                        logger.info(f'[{tile}] step1: proj {int(id):4d}/{ntheta}, dist {k+1}/{ndist}, norm={norms[ii]:.3e}')

                ref_norm = np.median(norms)
                for ii, id in enumerate(local_ids):
                    if norms[ii] < ref_norm / 10:
                        logger.warning(f'[{tile}] step1: broken frame proj={int(id)} dist={k+1} '
                                       f'norm={norms[ii]:.3e}  median={ref_norm:.3e}')
                        prev_id = local_ids[ii - 1] if ii > 0 else None
                        next_id = local_ids[ii + 1] if ii < len(local_ids) - 1 else None
                        if prev_id is not None and next_id is not None:
                            rep = 0.5 * (data_ds[k][prev_id].astype('float32') +
                                         data_ds[k][next_id].astype('float32'))
                        elif prev_id is not None:
                            rep = data_ds[k][prev_id].astype('float32')
                        else:
                            rep = data_ds[k][next_id].astype('float32')
                        data_ds[k][id] = np.round(rep).astype(data_ds[k].dtype)

        comm.Barrier()
        if rank == 0:
            logger.info(f'[{tile}] Step 1: done.')

    # -----------------------------------------------------------------------
    # STEP 2: Preprocessing (outlier removal + intensity normalisation)
    # -----------------------------------------------------------------------
    if start_step > 2:
        if rank == 0:
            logger.info(f'[{tile}] Step 2: skipped.')
        comm.Barrier()
    else:
        if rank == 0:
            logger.info(f'[{tile}] Step 2: preprocessing...')

        radius     = 9
        threshold  = 0.9
        chunk_size = 16

        def remove_outliers(data, radius, threshold):
            fdata = ndimage.median_filter(data, size=(1, radius, radius))
            mask  = cp.abs(data - fdata) > fdata * threshold
            return cp.where(mask, fdata, data)

        if rank == 0:
            ref0_arr  = np.empty([nref,  ndist, n, n], dtype='float32')
            dark_arr  = np.empty([ndark, ndist, n, n], dtype='float32')
            with h5py.File(fpath) as fid:
                for k in range(ndist):
                    ref0_arr[:, k]  = fid[f'/exchange/data_white_start{k}'][:, :n, :n]
                    dark_arr[:, k]  = fid[f'/exchange/data_dark{k}'][:, :n, :n]

            dark = np.mean(dark_arr, axis=0).astype('float32')

            ref  = np.mean(ref0_arr, axis=0).astype('float32')
            ref_gpu  = cp.array(ref) - cp.array(dark)
            ref_gpu[ref_gpu < 0] = 1e-3
            ref_gpu[:] = remove_outliers(ref_gpu, radius, threshold)
            ref = ref_gpu.get()
        else:
            ref  = np.empty([ndist, n, n], dtype='float32')
            dark = np.empty([ndist, n, n], dtype='float32')

        comm.Bcast(ref,  root=0)
        comm.Bcast(dark, root=0)

        dark_gpu = cp.array(dark)

        if rank == 0:
            mean_data_ref = np.zeros(ndist, dtype='float32')
            with h5py.File(fpath) as fid:
                for k in range(ndist):
                    data = cp.array(fid[f'/exchange/data{k}'][0, :n, :n].astype('float32'))
                    data -= dark_gpu[k]
                    data[data < 0] = 0
                    data = remove_outliers(data[None], radius, threshold)[0]
                    mean_data_ref[k] = float(data.mean())

            mmr = np.mean(ref, axis=(1, 2))
            mean_data_ref *= mmr[0] / mmr[:]
            ref           *= mmr[0] / mmr[:, None, None]
            mean_data_ref /= mmr[0]
            ref           /= mmr[0]
        else:
            mean_data_ref = np.zeros(ndist, dtype='float32')

        comm.Bcast(mean_data_ref, root=0)
        comm.Bcast(ref,           root=0)

        if rank == 0:
            with h5py.File(fpath, 'a') as fid:
                if '/exchange/pref' in fid:
                    del fid['/exchange/pref']
                fid.create_dataset('/exchange/pref', data=ref)
                if '/exchange/pref_end' in fid:
                    del fid['/exchange/pref_end']
                for k in range(ndist):
                    if f'/exchange/pdata{k}' in fid:
                        del fid[f'/exchange/pdata{k}']
        comm.Barrier()

        with h5py.File(fpath, 'a', driver='mpio', comm=comm) as fid:
            for k in range(ndist):
                fid.create_dataset(f'/exchange/pdata{k}', shape=(ntheta, n, n), dtype='float32')
        comm.Barrier()

        with h5py.File(fpath, 'a', driver='mpio', comm=comm) as fid:
            pdata_ds = [fid[f'/exchange/pdata{k}'] for k in range(ndist)]

            for k in range(ndist):
                for j in range(local_start, local_end, chunk_size):
                    end = min(j + chunk_size, local_end)

                    data = cp.array(fid[f'/exchange/data{k}'][j:end, :n, :n].astype('float32'))
                    data -= dark_gpu[k]
                    data[data < 0] = 0
                    data[:] = remove_outliers(data, radius, threshold)

                    _mean = data.mean(axis=(1, 2), keepdims=True)
                    _mean[_mean == 0] = 1
                    data *= float(mean_data_ref[k]) / _mean
                    data[~cp.isfinite(data)] = 1

                    pdata_ds[k][j:end] = data.get()

                    if j % 100 == 0:
                        logger.info(f'[{tile}] step2: proj {j:4d}/{ntheta}, dist {k+1}/{ndist}, mean={float(data[0].mean()):.4f}')

        with h5py.File(fpath, 'r', driver='mpio', comm=comm) as fid:
            _norm_sq = 0.0
            _rbatch = max(1, (1 << 28) // (n * n))
            for k in range(ndist):
                ds = fid[f'/exchange/pdata{k}']
                for _i0 in range(local_start, local_end, _rbatch):
                    _i1 = min(_i0 + _rbatch, local_end)
                    _chunk = cp.array(ds[_i0:_i1])
                    _norm_sq += float(cp.linalg.norm(_chunk)**2)
        logger.info(f'[{tile}] step2: rank {rank:4d}  pdata norm = {_norm_sq**0.5:.6e}')

        if rank == 0:
            logger.info(f'[{tile}] Step 2: done.')

    # -----------------------------------------------------------------------
    # STEP 3: Combine shifts
    # -----------------------------------------------------------------------
    if rank == 0:
        if start_step > 3:
            logger.info(f'[{tile}] Step 3: skipped.')
        else:
            logger.info(f'[{tile}] Step 3: combining shifts...')

            # Encoder (random) shifts — YY037A stores one text file per
            # distance under {path}/{pfile}/projections/{pfile}_{k:04d}.txt
            # (shape [ntheta, 2], columns = h, v). Fall back to zeros with
            # a warning if missing (same pattern as rhapp/motion below).
            _enc_dir = f'{path}/{pfile}/projections'
            shifts   = np.empty([ntheta, ndist, 2], dtype='float32')
            enc_ok   = True
            for k in range(ndist):
                enc_path = f'{_enc_dir}/{pfile}_{k + 1:04d}.txt'
                if not os.path.exists(enc_path):
                    logger.warning(f'[{tile}] Step 3: encoder file not found, '
                                   f'using zeros: {enc_path}')
                    shifts[:, k] = 0.0
                    enc_ok = False
                else:
                    logger.info(f'[{tile}] Step 3: reading encoder    from {enc_path}')
                    shifts[:, k] = np.loadtxt(enc_path, dtype='float32')[:ntheta]
            if not enc_ok:
                logger.warning(f'[{tile}] Step 3: one or more encoder files missing; '
                               'random_shifts partially zeroed.')

            random_shifts = np.empty([ntheta, ndist, 2], dtype='float32')
            random_shifts[..., 0] = shifts[..., 1] / norm_magnifications  # (y, row)
            random_shifts[..., 1] = shifts[..., 0] / norm_magnifications  # (x, col)

            # RHAPP inter-plane shifts (from Peter's MATLAB pipeline)
            _rhapp_path = f'{path}/{pfile}_/rhapp.mat'
            if not os.path.exists(_rhapp_path):
                logger.warning(f'[{tile}] Step 3: rhapp.mat not found, using zeros: {_rhapp_path}')
                rhapp_shifts = np.zeros([ntheta, ndist, 2], dtype='float32')
            else:
                logger.info(f'[{tile}] Step 3: reading rhapp       from {_rhapp_path}')
                rhapp_raw = load_octave_text_mat(_rhapp_path, 'rhapp')
                rhapp_reordered = rhapp_raw.swapaxes(0, 2)[:ntheta]
                rhapp_reordered -= rhapp_reordered[:, ref_dist:ref_dist + 1]
                avg_plane_zero = rhapp_reordered[:, 0].mean(axis=0)   # [2]
                rhapp_reordered -= avg_plane_zero[np.newaxis, np.newaxis, :]
                logger.info(f'[{tile}] Step 3: avg_plane_zero  y={avg_plane_zero[0]:.4f} px   '
                            f'x={avg_plane_zero[1]:.4f} px')
                rhapp_shifts = (-rhapp_reordered).astype('float32')

            # Motion shifts (slow drift of reference plane)
            _motion_dname = f'{path}/{pfile}_{ref_dist + 1}_'
            _motion_path  = f'{_motion_dname}/correct_motion.txt'
            if not os.path.exists(_motion_path):
                logger.warning(f'[{tile}] Step 3: correct_motion.txt not found, '
                               f'using zeros: {_motion_path}')
                motion_shifts = np.zeros([ntheta, ndist, 2], dtype='float32')
            else:
                logger.info(f'[{tile}] Step 3: reading motion      from {_motion_path}')
                raw_motion = np.loadtxt(_motion_path)[:ntheta, ::-1].astype('float32')
                motion_base   = raw_motion / norm_magnifications[ref_dist] - random_shifts[:, ref_dist]
                motion_shifts = np.tile(motion_base[:, np.newaxis], (1, ndist, 1))

            # 3-D tomographic correction shifts (usually absent on the first pass)
            _c3d_path = f'{path}/{pfile}_/correct_correct3D.txt'
            if os.path.exists(_c3d_path):
                logger.info(f'[{tile}] Step 3: reading correct3D   from {_c3d_path}')
                raw_3d = np.loadtxt(_c3d_path)[:ntheta, ::-1].astype('float32')
                correct3d_shifts = np.tile(raw_3d[:, np.newaxis], (1, ndist, 1))
            else:
                logger.info(f'[{tile}] Step 3: correct3D file not found, using zeros: {_c3d_path}')
                correct3d_shifts = np.zeros([ntheta, ndist, 2], dtype='float32')

            shifts_final = random_shifts + rhapp_shifts + motion_shifts + correct3d_shifts

            # cshifts_final stays tile-local, so Step 4 and any single-tile
            # reconstruction work unchanged. Where the tile sits on the mosaic
            # is not decided here at all — Step 5 starts from the nominal
            # tile_step and measures the rest off the assembled projections.
            with h5py.File(fpath, 'a') as fid:
                if '/exchange/cshifts_final' in fid:
                    del fid['/exchange/cshifts_final']
                fid.create_dataset('/exchange/cshifts_final', data=shifts_final)
                if '/exchange/shrink' in fid:
                    del fid['/exchange/shrink']
                fid.create_dataset('/exchange/shrink', data=shrink_nd)

            logger.info(f'[{tile}] Step 3: done.')

    comm.Barrier()

    # -----------------------------------------------------------------------
    # STEP 4: Binned data (multi-distance alignment + amplitude correction)
    #
    # Per tile, on the tile-local object grid (nobj_tile), using the tile-local
    # cshifts_final. The stitch across distances is only a means to an end here:
    # what gets written out is pdata{k}_{bin} with the inter-distance amplitude
    # corrections applied, plus the binned refs. Ported from
    # Y350c_new/steps15.py Step 4.
    # -----------------------------------------------------------------------
    if start_step > 4:
        if rank == 0:
            logger.info(f'[{tile}] Step 4: skipped.')
        comm.Barrier()
    else:
        if rank == 0:
            logger.info(f'[{tile}] Step 4: making binned data...')

        npad = n // 16

        if rank == 0:
            with h5py.File(fpath) as fid:
                ref = fid['/exchange/pref'][:, :n, :n].astype('float32')   # [ndist, n, n]
                r   = fid['/exchange/cshifts_final'][:].astype('float32')
            r[..., 1] += rotation_center_shift
        else:
            ref = np.empty([ndist, n, n],      dtype='float32')
            r   = np.empty([ntheta, ndist, 2], dtype='float32')

        comm.Bcast(ref, root=0)
        comm.Bcast(r,   root=0)

        if rank == 0:
            ref0 = ref.copy()
            with h5py.File(fpath, 'a') as fid:
                for bin in range(nlevels):
                    if f'/exchange/pref_{bin}' in fid:
                        del fid[f'/exchange/pref_{bin}']
                    fid.create_dataset(f'/exchange/pref_{bin}', data=ref0)
                    ref0 = 0.5 * (ref0[..., ::2]    + ref0[..., 1::2])
                    ref0 = 0.5 * (ref0[..., ::2, :] + ref0[..., 1::2, :])
        comm.Barrier()

        cl_shift = Shift(n, nobj_tile, n, nobj_tile, 'complex64')
        cref     = cp.array(ref)

        fwhm_ref    = 17.0 * (n / 2048)
        sigma_ref   = fwhm_ref / (2 * np.sqrt(2 * np.log(2)))
        cref_smooth = cp.stack([ndimage.gaussian_filter(cref[k], sigma_ref)
                                for k in range(ndist)])

        with h5py.File(fpath, 'a', driver='mpio', comm=comm) as fid:
            data_out = [[fid.require_dataset(f'/exchange/pdata{k}_{bin}',
                                             shape=(ntheta, n // 2**bin, n // 2**bin),
                                             dtype='float32', exact=True)
                         for k in range(ndist)]
                        for bin in range(nlevels)]

            srdata = cp.zeros([ndist, nobj_tile, nobj_tile], dtype='float32')

            v = cp.linspace(0, 1, npad, endpoint=False)
            v = v**5 * (126 - 420*v + 540*v**2 - 315*v**3 + 70*v**4)

            for j in local_ids:
                data = cp.empty([ndist, n, n], dtype='float32')
                for k in range(ndist):
                    data[k] = cp.array(fid[f'/exchange/pdata{k}'][j, :n, :n].astype('float32'))

                data_smooth = cp.stack([ndimage.gaussian_filter(data[k], sigma_ref)
                                        for k in range(ndist)])
                rdata = data_smooth / (cref_smooth + 1e-5)

                for k in range(ndist - 1, -1, -1):
                    shrink_jk  = shrink_nd[j, k]                          # (2,) (y, x)
                    eff_mag_jk = norm_magnifications[k] / (1 + shrink_jk)  # (2,) (y, x)
                    mag = cp.array(1.0 / eff_mag_jk, dtype='float32')[None]
                    tmp = rdata[k].astype('complex64')
                    tmp = cl_shift.curlySback(
                        cp.log(tmp[None]).astype('complex64'),
                        cp.array(r[j:j+1, k]), mag
                    )[0].real
                    tmp = cp.exp(tmp)

                    padx0 = int((nobj_tile - n / eff_mag_jk[1]) / 2) - int(r[j, k, 1])
                    pady0 = int((nobj_tile - n / eff_mag_jk[0]) / 2) - int(r[j, k, 0])
                    padx1 = int((nobj_tile - n / eff_mag_jk[1]) / 2) + int(r[j, k, 1])
                    pady1 = int((nobj_tile - n / eff_mag_jk[0]) / 2) + int(r[j, k, 0])
                    padx0 = min(nobj_tile, max(0, padx0)) + 5
                    pady0 = min(nobj_tile, max(0, pady0)) + 5
                    padx1 = min(nobj_tile, max(0, padx1)) + 5
                    pady1 = min(nobj_tile, max(0, pady1)) + 5

                    tmp = cp.pad(tmp[pady0:-pady1], ((pady0, pady1), (0, 0)), 'edge')
                    tmp = cp.pad(tmp[:, padx0:-padx1], ((0, 0), (padx0, padx1)),
                                 'linear_ramp', end_values=((1, 1), (1, 1)))

                    if k < ndist - 1:
                        mmm = float(srdata[k + 1][pady0:-pady1, padx0:-padx1].mean() /
                                    (tmp[pady0:-pady1, padx0:-padx1].mean() + 1e-10))
                        tmp     *= mmm
                        data[k] *= mmm
                        if k == 0:
                            cs   = min(nobj_tile // 16, (nobj_tile - pady0 - pady1) // 2,
                                       (nobj_tile - padx0 - padx1) // 2)
                            ch   = cs // 2
                            midy = nobj_tile // 2
                            midx = nobj_tile // 2
                            ys   = [pady0, midy - ch, nobj_tile - pady1 - cs]
                            xs   = [padx0, midx - ch, nobj_tile - padx1 - cs]
                            _rf  = srdata[k + 1]
                            R = cp.array([[float(_rf[y:y+cs, x:x+cs].mean() /
                                                 (tmp[y:y+cs, x:x+cs].mean() + 1e-10))
                                           for x in xs] for y in ys], dtype='float32')
                            ratio_map = ndimage.zoom(R, nobj_tile / 3, order=1)
                            tmp *= ratio_map[:nobj_tile, :nobj_tile]
                            ratio_crop = ratio_map[pady0:nobj_tile-pady1, padx0:nobj_tile-padx1]
                            data[k] *= ndimage.zoom(
                                ratio_crop,
                                (n / ratio_crop.shape[0], n / ratio_crop.shape[1]),
                                order=1)[:n, :n]
                        wx = cp.ones(nobj_tile, dtype='float32')
                        wy = cp.ones(nobj_tile, dtype='float32')
                        wx[:padx0]               = 0
                        wx[padx0:padx0 + npad]   = v
                        wx[-padx1 - npad:-padx1] = 1 - v
                        wx[-padx1:]              = 0
                        wy[:pady0]               = 0
                        wy[pady0:pady0 + npad]   = v
                        wy[-pady1 - npad:-pady1] = 1 - v
                        wy[-pady1:]              = 0
                        w   = cp.outer(wy, wx)
                        tmp = tmp * w + srdata[k + 1] * (1 - w)
                    srdata[k] = tmp

                if j % 100 == 0:
                    logger.info(f'[{tile}] step4: proj {int(j):4d}/{ntheta}')

                for k in range(ndist):
                    datak = data[k]
                    for bin in range(nlevels):
                        data_out[bin][k][j] = datak.get()
                        datak = 0.5 * (datak[::2, :] + datak[1::2, :])
                        datak = 0.5 * (datak[:, ::2] + datak[:, 1::2])

        del cl_shift, cref, cref_smooth, srdata
        cp.get_default_memory_pool().free_all_blocks()
        comm.Barrier()
        if rank == 0:
            logger.info(f'[{tile}] Step 4: done.')


# ===========================================================================
# Optional diagnostic: adjacent-tile overlap phase correlation
#
# Superseded by estimate_overlap in Step 5, which registers the seams on the
# assembled projections instead of raw frames. Kept as a quick sanity check on
# the data as it comes off Step 3; enable with overlap_check=true.
# ===========================================================================

def phase_correlation(a, b):
    """Sub-pixel shift (dy, dx) of b relative to a via FFT phase correlation.

    Positive dy/dx means b is shifted DOWN / RIGHT relative to a. A Hann window
    is applied to both inputs to suppress edge ringing on narrow overlap strips.
    Sub-pixel refinement is a 3-point parabolic fit around the correlation peak.
    """
    a = np.asarray(a, dtype='float32'); b = np.asarray(b, dtype='float32')
    ny, nx = a.shape
    wy = np.hanning(ny)[:, None]; wx = np.hanning(nx)[None, :]
    win = wy * wx
    a = (a - a.mean()) * win
    b = (b - b.mean()) * win
    fa = np.fft.fft2(a); fb = np.fft.fft2(b)
    r  = fa * np.conj(fb)
    r  /= np.abs(r) + 1e-12
    c  = np.fft.ifft2(r).real
    py, px = np.unravel_index(np.argmax(c), c.shape)
    dy0 = py - ny if py > ny // 2 else py
    dx0 = px - nx if px > nx // 2 else px
    def _parab(v):
        return 0.5 * (v[0] - v[2]) / (v[0] - 2 * v[1] + v[2] + 1e-12)
    ry = [c[(py + t) % ny, px] for t in (-1, 0, 1)]
    rx = [c[py, (px + t) % nx] for t in (-1, 0, 1)]
    return float(dy0 + _parab(ry)), float(dx0 + _parab(rx))


if not overlap_check or len(tiles) < 2:
    pass
elif rank == 0:
    logger.info('=' * 74)
    logger.info(f'DIAGNOSTIC: adjacent-tile overlap correlation '
                f'(width={overlap_width}px, {overlap_nangles} angles)')
    logger.info('=' * 74)

    angle_idx = np.linspace(0, ntheta - 1, overlap_nangles, dtype=int)

    logger.info(f'{"pair":<22s} {"dist":>4s} {"angle":>6s}   {"dy":>8s} {"dx":>8s}')
    for i in range(len(tiles) - 1):
        left_tile, right_tile = tiles[i], tiles[i + 1]
        fA_path = f'{path_out}/{_pfile_tile(left_tile )}.h5'
        fB_path = f'{path_out}/{_pfile_tile(right_tile)}.h5'
        pair_lbl = f'{left_tile}<->{right_tile}'
        with h5py.File(fA_path, 'r') as fA, h5py.File(fB_path, 'r') as fB:
            for k in range(ndist):
                dsA = fA[f'/exchange/pdata{k}']
                dsB = fB[f'/exchange/pdata{k}']
                for j in angle_idx:
                    A = dsA[int(j), :, -overlap_width:].astype('float32')  # right edge of left tile
                    B = dsB[int(j), :,  :overlap_width ].astype('float32')  # left  edge of right tile
                    dy, dx = phase_correlation(A, B)
                    logger.info(f'{pair_lbl:<22s} {k:>4d} {int(j):>6d}   '
                                f'{dy:+8.2f} {dx:+8.2f}')

comm.Barrier()


# ===========================================================================
# STEP 5: Mosaic stitching + Paganin phase retrieval + FBP
#
# Runs ONCE, over all tiles at once. Each tile is stitched across distances on
# its own tile-local object grid (exactly as in Y350c_new/steps15.py — doing it
# on the full mosaic grid would be ~len(tiles)x the FFT work for no gain), then
# the five tile-local results are composited onto one wide object grid at the
# offsets measured by estimate_overlap. Paganin then runs on the wide grid.
#
# FBP on the mosaic costs nobj**2 * nzobj voxels (3.4 TB at bin 0, 52 GB at
# bin 2) — pick the bins with start_level_rec.
# ===========================================================================

if start_step > 5:
    if rank == 0:
        logger.info('Step 5: skipped.')
    comm.Barrier()
else:
    if rank == 0:
        logger.info('=' * 74)
        logger.info('STEP 5: mosaic stitching + Paganin (+ FBP)')
        logger.info('=' * 74)

    tile_paths = [f'{path_out}/{_pfile_tile(t)}.h5' for t in tiles]
    ref_tile   = len(tiles) // 2                     # geometry source (== "center")

    # --- Tile offsets: nominal to start with, measured below -----------------
    # (v, h) in object px on the finest grid. Only the nominal step goes in
    # here; estimate_overlap corrects it from the assembled projections, once
    # per bin, coarsest first.
    if len(tiles) > 1 and tile_step <= 0:
        raise SystemExit('tile_step must be set (object px on the finest grid, '
                         'the nominal spacing between adjacent tiles)')
    offsets = nominal_tile_offsets(len(tiles), tile_step)
    if rank == 0:
        for t, tl in enumerate(tiles):
            logger.info(f'  {tl or "(single)":<10s} tile_offset  v={offsets[t,0]:+9.4f}  '
                        f'h={offsets[t,1]:+11.4f}   [nominal]')

    # --- Mosaic grid size ----------------------------------------------------
    # Set in the config, never derived. Deriving it from the nominal offsets
    # made the grid move whenever tile_step was touched, which silently changes
    # the voxel-to-index mapping of everything already reconstructed.
    if args.nzobj is None or args.nobj is None:
        raise SystemExit('nzobj and nobj must both be set in the config '
                         '(mosaic grid, in object px on the finest grid)')
    nzobj, nobj = args.nzobj, args.nobj

    # How far a tile is allowed to move away from the nominal, in finest-grid
    # object px. Doubles as the seam search radius, so a bad correlation lock
    # can never put a tile somewhere the grid does not reach.
    estimate_margin = 64.0

    if rank == 0:
        logger.info(f'  tile grid nobj_tile={nobj_tile}   mosaic grid '
                    f'nzobj={nzobj}  nobj={nobj}   (tile_step={tile_step:g})')
        # Both are divided by 2**bin at every level, and paste_geometry splits
        # (nobj_bin - nobj_tile_bin) / (nzobj_bin - nobj_tile_bin) evenly, so an
        # odd result would drift the tile placement by half a pixel between levels.
        for _nm, _v in (('nzobj', nzobj), ('nobj', nobj)):
            if _v % 2**(nlevels - 1) != 0:
                logger.warning(f'  {_nm}={_v} is not a multiple of '
                               f'2**(nlevels-1)={2**(nlevels - 1)}; the coarse '
                               f'levels will not line up exactly with bin 0')
        # The grid has to hold a tile centred at +-max|offset|, plus room for
        # the measured correction applied after the grid is fixed.
        _margin = estimate_margin if (estimate_overlap and len(tiles) > 1) else 0.0
        for _nm, _v, _ax in (('nzobj', nzobj, 0), ('nobj', nobj, 1)):
            _need = 2 * (float(np.abs(offsets[:, _ax]).max()) + _margin) + nobj_tile
            if _v < _need:
                logger.warning(f'  {_nm}={_v} is smaller than the '
                               f'{_need:.0f} px the tiles need at their nominal '
                               f'positions — the outer tiles will be cropped')

    def multiPaganin(data, distances, wavelength, voxelsize, delta_beta, alpha):

        """Multi-distance Paganin phase retrieval on GPU. data: [ndist, ny, nx]."""
        fx = cp.fft.fftfreq(data.shape[-1], d=voxelsize).astype('float32')
        fy = cp.fft.fftfreq(data.shape[-2], d=voxelsize).astype('float32')
        fx, fy = cp.meshgrid(fx, fy)
        numerator   = 0
        denominator = 0
        for j in range(data.shape[0]):
            rad_freq    = cp.fft.fft2(data[j].astype('complex64'))
            taylorExp   = 1 + wavelength * distances[j] * cp.pi * delta_beta * (fx**2 + fy**2)
            numerator   += taylorExp * rad_freq
            denominator += taylorExp**2
        numerator   /= len(distances)
        denominator  = denominator / len(distances) + alpha
        phase = cp.log(cp.real(cp.fft.ifft2(numerator / denominator)))
        phase *= delta_beta * 0.5
        return phase

    def _overlap_slices(o, w_src, w_dst):
        """Slices for pasting a length-w_src source at offset o into a length-w_dst
        destination, clipped at both ends."""
        s0, s1 = max(0, -o), min(w_src, w_dst - o)
        return slice(s0, s1), slice(o + s0, o + s1)

    # Common geometry (identical for every tile — taken from the reference tile)
    fpath_mosaic = f'{path_out}/{pfile_base}_mosaic'
    if rank == 0:
        with h5py.File(tile_paths[ref_tile]) as fid:
            theta_raw = fid['/exchange/theta'][:, 0].astype('float32')
    else:
        theta_raw = np.empty(ntheta, dtype='float32')
    comm.Bcast(theta_raw, root=0)
    theta = (-theta_raw / 180 * np.pi).astype('float32')

    # --- optional angle subset, for fast tests ------------------------------
    # Steps 1-4 always process every angle; only the reconstruction is thinned.
    # The subset is spread evenly over the whole rotation, so the FBP still
    # spans the full angular range and only gets sparser (more streaks). Angle
    # indices stay absolute — they index the tile h5 and shrink_nd directly —
    # while positions within the subset index the output datasets.
    if 0 < ntheta_rec < ntheta:
        theta_ids = np.unique(np.linspace(0, ntheta, ntheta_rec,
                                          endpoint=False).astype('int64'))
    else:
        theta_ids = np.arange(ntheta, dtype='int64')
    ntheta5 = len(theta_ids)
    theta5  = np.ascontiguousarray(theta[theta_ids])

    # Same partition MPIClass will use for the theta slab below, taken from the
    # same helper so the two cannot drift apart.
    start5, _end5 = get_local_chunk(ntheta5, rank, size)
    ids5 = theta_ids[start5:_end5]
    # With every angle kept, keep the historic every-10th sampling of the
    # written projections; on a subset the point is to look at them, so keep all.
    proj_step = 10 if ntheta5 == ntheta else 1

    if rank == 0 and ntheta5 != ntheta:
        logger.warning(f'Step 5: reconstructing {ntheta5} of {ntheta} angles '
                       f'(ntheta_rec={ntheta_rec}), every '
                       f'{theta_ids[1] - theta_ids[0]}th, {theta_ids[0]}..{theta_ids[-1]} '
                       f'— TEST MODE, not a full run')
    logger.debug(f'step5: rank {rank} angles [{start5}:{_end5}) of {ntheta5}'
                 + (f', absolute {ids5[0]}..{ids5[-1]}' if len(ids5) else ' (empty)'))

    for _f in ('_srdata.h5', '_proj.h5', '_obj.h5'):
        if rank == 0 and os.path.exists(fpath_mosaic + _f):
            os.remove(fpath_mosaic + _f)
    comm.Barrier()
    if rank == 0:
        for _f in ('_srdata.h5', '_proj.h5', '_obj.h5'):
            with h5py.File(fpath_mosaic + _f, 'w'):
                pass
    comm.Barrier()

    # --- Shrinkage across the tiles ------------------------------------------
    # The stitch is per tile, so it uses that tile's own shrink_all[t]. Paganin
    # is not: it runs on the composited mosaic, where every tile is present in
    # one image and there is a single distance per distance index. The tile
    # mean is what that image corresponds to, so that is what it gets — with
    # the spread across tiles logged, since a large one would mean the tiles
    # deform differently enough that one Paganin filter no longer fits them all.
    shrink_tiles = np.stack(shrink_all)            # [ntiles, ntheta, ndist, 2]
    shrink_mean  = shrink_tiles.mean(axis=0)       # [ntheta, ndist, 2]
    if rank == 0:
        spread = np.abs(shrink_tiles - shrink_mean[None]).max()
        logger.info(f'Step 5: shrink per tile, |tile - mean| max {spread:.3e} '
                    f'(mean over tiles used for Paganin, each tile\'s own for '
                    f'the stitch)')
        for _t, _tl in enumerate(tiles):
            _s = shrink_tiles[_t]
            logger.info(f'  {_tl or "(single)":<10s} base '
                        f'v {shrink_base[_t][0]:+.3e} '
                        f'h {shrink_base[_t][1]:+.3e}'
                        f'   shrink v '
                        f'{_s[..., 0].min():+.3e}..{_s[..., 0].max():+.3e}  h '
                        f'{_s[..., 1].min():+.3e}..{_s[..., 1].max():+.3e}')
        if spread > 1e-3:
            _why = ('mostly the acquisition baseline, which is expected'
                    if np.abs(np.stack(shrink_base)).max() > 0 else
                    'and the acquisition baseline is zero everywhere, so none '
                    'of it is that')
            logger.warning(f'Step 5: the tiles differ in shrink by up to '
                           f'{spread:.3e}, i.e. {spread*nobj_tile:.1f} px over '
                           f'a tile width ({_why}) — the mosaic Paganin uses '
                           f'their mean')

    # range(start_level_rec, nlevels) is empty if the two are set the same way
    # round, which would make step 5 a silent no-op. The coarsest bin is
    # nlevels-1, so anything at or above nlevels reconstructs nothing.
    if start_level_rec >= nlevels and rank == 0:
        logger.warning(f'Step 5: start_level_rec={start_level_rec} >= '
                       f'nlevels={nlevels}, so no bin level is reconstructed. '
                       f'The coarsest available level is {nlevels - 1}.')

    for bin in range(start_level_rec, nlevels):
        n_bin         = n         // 2**bin
        nobj_tile_bin = nobj_tile // 2**bin      # tile-local grid
        nzobj_bin     = nzobj     // 2**bin      # mosaic grid
        nobj_bin      = nobj      // 2**bin
        voxelsize_bin = voxelsize * 2**bin
        scale         = 1.0 / 2**bin
        vol_gb        = nobj_bin**2 * nzobj_bin * 8 / 2**30

        if rank == 0:
            logger.info(f'Step 5: bin={bin}  n_bin={n_bin}  nobj_tile_bin={nobj_tile_bin}  '
                        f'mosaic {nzobj_bin}x{nobj_bin}  '
                        f'voxelsize={voxelsize_bin*1e9:.3f} nm  '
                        f'FBP volume {vol_gb:.1f} GB')

        # --- Per-tile shifts and paste offsets -------------------------------
        off_bin = offsets * scale                                   # (ntiles, 2)
        frac, origin_y, origin_x = paste_geometry(off_bin, nobj_tile_bin,
                                                  nzobj_bin, nobj_bin)

        def _feather():
            """Seam feather: a good fraction of the tile-tile overlap, on the
            sides that actually face a neighbour."""
            if len(tiles) > 1:
                gap = int(np.abs(np.diff(origin_x)).max())
                ov  = max(0, nobj_tile_bin - gap)
            else:
                ov = 0
            return ov, (max(8, min(ov // 2, nobj_tile_bin // 8)) if ov > 16 else 0)

        def _apply_offsets():
            """Recompute everything that hangs off ``offsets``.

            Changing ``offsets`` (or ``off_bin``) on its own does nothing: the
            paste origins, the fractional part folded into the shift kernels and
            the seam feather are all derived here, once, and it is those that
            the stitch actually reads. ``rs``/``r_gpu`` already carry the old
            fraction, so only the difference goes in.
            """
            global off_bin, frac, origin_y, origin_x, ov_bin, tfeather
            off_bin  = offsets * scale
            frac_old = frac
            frac, origin_y, origin_x = paste_geometry(off_bin, nobj_tile_bin,
                                                      nzobj_bin, nobj_bin)
            for t in range(len(tiles)):
                d = (frac[t] - frac_old[t]).astype('float32')
                rs[t][..., 0] += d[0]
                rs[t][..., 1] += d[1]
                r_gpu[t][..., 0] += float(d[0])
                r_gpu[t][..., 1] += float(d[1])
            ov_bin, tfeather = _feather()

        ov_bin, tfeather = _feather()
        if rank == 0:
            logger.info(f'Step 5: bin={bin}  tile overlap {ov_bin} px, '
                        f'seam feather {tfeather} px')
            logger.info(f'Step 5: bin={bin}  paste origins x={list(origin_x)}  '
                        f'y={list(origin_y)}')

        rs, crefs = [], []
        for t, tp in enumerate(tile_paths):
            if rank == 0:
                with h5py.File(tp) as fid:
                    _cs  = fid['/exchange/cshifts_final'][:].astype('float32')
                    _ref = fid[f'/exchange/pref_{bin}'][:ndist].astype('float32')
            else:
                _cs  = np.empty([ntheta, ndist, 2],   dtype='float32')
                _ref = np.empty([ndist, n_bin, n_bin], dtype='float32')
            comm.Bcast(_cs,  root=0)
            comm.Bcast(_ref, root=0)
            _r = (_cs * scale).astype('float32')
            _r[..., 1] += rotation_center_shift * scale + 0.5 * (scale - 1)
            _r[..., 0] += frac[t, 0]
            _r[..., 1] += frac[t, 1]
            rs.append(_r)
            crefs.append(_ref)


        
        fwhm_ref    = 17.0 * (n_bin / 2048)
        sigma_ref   = fwhm_ref / (2 * np.sqrt(2 * np.log(2)))
        cref_smooth = [cp.stack([ndimage.gaussian_filter(cp.array(cr[k]), sigma_ref)
                                 for k in range(ndist)]) for cr in crefs]
        r_gpu = [cp.array(_r) for _r in rs]

        cl_shift = Shift(n_bin, nobj_tile_bin, n_bin, nobj_tile_bin, 'complex64')
        npad_bin = n_bin // 16
        v_bin    = cp.linspace(0, 1, npad_bin, endpoint=False)
        v_bin    = v_bin**5 * (126 - 420*v_bin + 540*v_bin**2 - 315*v_bin**3 + 70*v_bin**4)

        def _stitch(fid, srdata, j, t):
            """Multi-distance stitch of tile ``t`` on its own object grid.

            Identical to Y350c_new/steps15.py; returns the footprint of the
            least-magnified distance (the widest of the ndist), which is where
            this tile actually holds data.
            """
            data_j = cp.empty([ndist, n_bin, n_bin], dtype='float32')
            for k in range(ndist):
                data_j[k] = cp.array(fid[f'/exchange/pdata{k}_{bin}'][j].astype('float32'))
            data_j_smooth = cp.stack([ndimage.gaussian_filter(data_j[k], sigma_ref)
                                      for k in range(ndist)])
            rdata = data_j_smooth / (cref_smooth[t] + 1e-5)
            srdata.fill(0)
            _r = rs[t]
            box = None
            for k in range(ndist - 1, -1, -1):
                shrink_jk  = shrink_all[t][j, k]                      # (2,) (y, x)
                eff_mag_jk = norm_magnifications[k] / (1 + shrink_jk)  # (2,) (y, x)
                mag = cp.array(1.0 / eff_mag_jk, dtype='float32')[None]
                tmp = rdata[k].astype('complex64')
                tmp = cl_shift.curlySback(
                    cp.log(tmp[None]).astype('complex64'), r_gpu[t][j:j+1, k], mag
                )[0].real
                tmp = cp.exp(tmp)
                padx0 = int((nobj_tile_bin - n_bin / eff_mag_jk[1]) / 2) - int(_r[j, k, 1])
                pady0 = int((nobj_tile_bin - n_bin / eff_mag_jk[0]) / 2) - int(_r[j, k, 0])
                padx1 = int((nobj_tile_bin - n_bin / eff_mag_jk[1]) / 2) + int(_r[j, k, 1])
                pady1 = int((nobj_tile_bin - n_bin / eff_mag_jk[0]) / 2) + int(_r[j, k, 0])
                padx0 = min(nobj_tile_bin, max(0, padx0)) + 5
                pady0 = min(nobj_tile_bin, max(0, pady0)) + 5
                padx1 = min(nobj_tile_bin, max(0, padx1)) + 5
                pady1 = min(nobj_tile_bin, max(0, pady1)) + 5
                if box is None:
                    box = (pady0, pady1, padx0, padx1)
                tmp = cp.pad(tmp[pady0:-pady1], ((pady0, pady1), (0, 0)), 'edge')
                tmp = cp.pad(tmp[:, padx0:-padx1], ((0, 0), (padx0, padx1)),
                             'linear_ramp', end_values=((1, 1), (1, 1)))
                if k < ndist - 1:
                    denom = tmp[pady0:-pady1, padx0:-padx1].mean() + 1e-10
                    mmm   = float(srdata[k+1][pady0:-pady1, padx0:-padx1].mean() / denom)
                    tmp  *= mmm
                    if k == 0:
                        cs   = min(nobj_tile_bin // 16, (nobj_tile_bin - pady0 - pady1) // 2,
                                   (nobj_tile_bin - padx0 - padx1) // 2)
                        ch   = cs // 2
                        ys   = [pady0, nobj_tile_bin // 2 - ch, nobj_tile_bin - pady1 - cs]
                        xs   = [padx0, nobj_tile_bin // 2 - ch, nobj_tile_bin - padx1 - cs]
                        _rf  = srdata[k + 1]
                        R = cp.array([[float(_rf[y:y+cs, x:x+cs].mean() /
                                             (tmp[y:y+cs, x:x+cs].mean() + 1e-10))
                                       for x in xs] for y in ys], dtype='float32')
                        ratio_map = ndimage.zoom(R, nobj_tile_bin / 3, order=1)
                        tmp *= ratio_map[:nobj_tile_bin, :nobj_tile_bin]
                    wx = cp.ones(nobj_tile_bin, dtype='float32')
                    wy = cp.ones(nobj_tile_bin, dtype='float32')
                    wx[:padx0]                 = 0
                    wx[padx0:padx0+npad_bin]   = v_bin
                    wx[-padx1-npad_bin:-padx1] = 1 - v_bin
                    wx[-padx1:]                = 0
                    wy[:pady0]                 = 0
                    wy[pady0:pady0+npad_bin]   = v_bin
                    wy[-pady1-npad_bin:-pady1] = 1 - v_bin
                    wy[-pady1:]                = 0
                    w   = cp.outer(wy, wx)
                    tmp = tmp * w + srdata[k+1] * (1 - w)
                srdata[k] = tmp
            return box

        def _tile_window(box, t):
            """Weight of tile ``t`` over its own grid: 1 inside the footprint,
            quintic ramp over ``tfeather`` on the sides facing a neighbour,
            hard cut elsewhere (the wsum normalisation takes care of those)."""
            pady0, pady1, padx0, padx1 = box
            wy = cp.zeros(nobj_tile_bin, dtype='float32')
            wx = cp.zeros(nobj_tile_bin, dtype='float32')
            wy[pady0:nobj_tile_bin - pady1] = 1
            wx[padx0:nobj_tile_bin - padx1] = 1
            if tfeather:
                ramp = cp.linspace(0, 1, tfeather, endpoint=False, dtype='float32')
                ramp = ramp**5 * (126 - 420*ramp + 540*ramp**2 - 315*ramp**3 + 70*ramp**4)
                if t > 0:                                # feather towards the left neighbour
                    wx[padx0:padx0 + tfeather] = ramp
                if t < len(tiles) - 1:                   # ... and the right one
                    wx[nobj_tile_bin - padx1 - tfeather:nobj_tile_bin - padx1] = 1 - ramp
            return cp.outer(wy, wx)

        def _mosaic(fids, srdata, mosaic, wsum, j):
            """Composite every tile of projection ``j`` onto the wide grid."""
            mosaic.fill(0)
            wsum.fill(0)
            for t in range(len(tiles)):
                box = _stitch(fids[t], srdata, j, t)
                
                wt  = _tile_window(box, t)
                sy, dy_ = _overlap_slices(int(origin_y[t]), nobj_tile_bin, nzobj_bin)
                sx, dx_ = _overlap_slices(int(origin_x[t]), nobj_tile_bin, nobj_bin)
                wts = wt[sy, sx]
                if t:
                    # Cumulative amplitude match: put this tile on the grey scale
                    # of what is already down, using the shared overlap only.
                    ov = (wsum[dy_, dx_] > 0.5) & (wts > 0.5)
                    npx = int(ov.sum())
                    if npx > 256:
                        acc = mosaic[0][dy_, dx_][ov] / wsum[dy_, dx_][ov]
                        mmm = float(acc.mean() / (srdata[0][sy, sx][ov].mean() + 1e-10))
                        srdata *= mmm
                    else:
                        mmm = 1.0
                    if j == 0:
                        logger.debug(f'step5 bin={bin}: tile {tiles[t]!r} seam scale '
                                     f'{mmm:.5f} over {npx} px')
                mosaic[:, dy_, dx_] += srdata[:, sy, sx] * wts
                wsum[dy_, dx_]      += wts
            mosaic /= cp.maximum(wsum, 1e-2)
            mosaic[:, wsum < 1e-2] = 1.0          # uncovered = no absorption

        def _estimate_overlap(fids, srdata, angles):
            """Re-measure the tile placement from the assembled projections.

            Once ``_stitch`` has run, a tile is a single multi-distance image on
            the object grid, with its magnification and its per-projection
            motion already taken out — the only thing left between two
            neighbours is where they sit relative to each other, which is what
            the correlation sees. Doing it here rather than on raw frames also
            means one measurement per seam instead of one per distance.

            Each seam is correlated over the strip the two neighbours share
            under the *current* origins, restricted to where both actually hold
            data (the ``_stitch`` footprint), so what comes back is a residual.

            Returns the correction in finest-grid object px, ``(ntiles, 2)``.
            """
            ntiles = len(tiles)
            if ntiles < 2:
                return np.zeros([ntiles, 2])

            max_shift = max(2, min(int(np.ceil(estimate_margin * scale)),
                                   ov_bin // 4))
            min_ov    = 32
            meas      = [[] for _ in range(ntiles - 1)]
            for j in angles:
                prev = None
                for t in range(ntiles):
                    box = _stitch(fids[t], srdata, j, t)
                    cur = (srdata[0].copy(), box)
                    if prev is not None:
                        s = t - 1
                        (pi, pb), (ci, cb) = prev, cur
                        # Where each neighbour holds data, in mosaic coords.
                        ax0 = origin_x[s] + pb[2]; ax1 = origin_x[s] + nobj_tile_bin - pb[3]
                        bx0 = origin_x[t] + cb[2]; bx1 = origin_x[t] + nobj_tile_bin - cb[3]
                        ay0 = origin_y[s] + pb[0]; ay1 = origin_y[s] + nobj_tile_bin - pb[1]
                        by0 = origin_y[t] + cb[0]; by1 = origin_y[t] + nobj_tile_bin - cb[1]
                        x0, x1 = max(ax0, bx0), min(ax1, bx1)
                        y0, y1 = max(ay0, by0), min(ay1, by1)
                        if x1 - x0 >= min_ov and y1 - y0 >= min_ov:
                            A = pi[y0 - origin_y[s]:y1 - origin_y[s],
                                   x0 - origin_x[s]:x1 - origin_x[s]]
                            B = ci[y0 - origin_y[t]:y1 - origin_y[t],
                                   x0 - origin_x[t]:x1 - origin_x[t]]
                            (dv, dh), pk = phase_corr_2d(A, B, max_shift)
                            meas[s].append((dv, dh, pk))
                    prev = cur

            corr, per_seam = combine_seam_shifts(meas, ntiles, ref=ref_tile,
                                                 log=logger.warning)
            logger.info(f'step5 bin={bin}: overlap estimate over '
                        f'{len(angles)} angles, search +-{max_shift} px, '
                        f'anchored on {tiles[ref_tile]!r}')
            for s, (dv, dh, sv, sh, nu, pk) in enumerate(per_seam):
                logger.debug(f'  seam {tiles[s]}|{tiles[s+1]}: '
                             f'dv={dv:+7.2f}+-{sv:5.2f}  dh={dh:+7.2f}+-{sh:5.2f} '
                             f'px (bin {bin}), {nu}/{len(angles)} angles, '
                             f'peak {pk:.3f}')
            return corr / scale                   # -> finest-grid object px

        srdata = cp.zeros([ndist, nobj_tile_bin,   nobj_tile_bin],   dtype='float32')
        mosaic = cp.zeros([ndist, nzobj_bin, nobj_bin], dtype='float32')
        wsum   = cp.zeros([nzobj_bin, nobj_bin],        dtype='float32')
        pad8   = min(nzobj_bin, nobj_bin) // 8

        pag_alpha = 0.001

        def _pag_distances(j):
            """Effective propagation distance per distance index, for angle j.

            The object-plane distance is the detector one divided by the
            magnification squared; shrinkage rescales the magnification, so it
            enters as (1 + shrink)**2 and makes this angle-dependent.

            One mosaic, one filter, so this is the mean over the tiles (see the
            spread logged above); the per-tile values only differ in shrink.
            """
            return (distances * (1 + shrink_mean[j, :].mean(axis=-1))**2
                    / norm_magnifications**2)

        def _paganin(j):
            pj = cp.pad(mosaic, ((0, 0), (pad8, pad8), (pad8, pad8)), 'reflect')
            ph = multiPaganin(pj, _pag_distances(j), wavelength, voxelsize_bin,
                              paganin, pag_alpha)
            return ph[pad8:pad8 + nzobj_bin, pad8:pad8 + nobj_bin]

        if rank == 0:
            eff = (distances[None] * (1 + shrink_mean.mean(axis=-1))**2
                   / norm_magnifications[None]**2)          # [ntheta, ndist]
            # multiPaganin's filter is 1 + pi*lambda*d*(delta/beta)*f**2, so it
            # is flat below f = 1/sqrt(pi*lambda*d*delta_beta) and rolls off as
            # f**-2 above. 1/f is the length it smooths over.
            cutoff = np.sqrt(np.pi * wavelength * eff[0] * paganin)
            logger.info(f'step5 bin={bin}: Paganin  delta/beta={paganin:g}  '
                        f'alpha={pag_alpha:g}  '
                        f'energy={energy:g} keV  lambda={wavelength*1e12:.4f} pm  '
                        f'voxel={voxelsize_bin*1e9:.3f} nm  pad={pad8} px')
            logger.info(f'step5 bin={bin}: Paganin  detector distances (m) '
                        f'{np.array2string(distances, precision=6)}')
            logger.info(f'step5 bin={bin}: Paganin  norm_magnifications '
                        f'{np.array2string(norm_magnifications, precision=6)}')
            for k in range(ndist):
                logger.info(f'  k={k}: object distance {eff[0, k]*1e3:9.4f} mm '
                            f'(over angles {eff[:, k].min()*1e3:9.4f} .. '
                            f'{eff[:, k].max()*1e3:9.4f})  smoothing length '
                            f'{cutoff[k]*1e9:8.1f} nm = '
                            f'{cutoff[k]/voxelsize_bin:6.2f} px')

        # --- Re-measure the tile placement (rank 0), broadcast ---------------
        # Cheap enough to redo per bin, and each bin is a finer measurement of
        # the same thing; the result is carried in `offsets` (finest-grid px),
        # so a coarse bin also hands its answer to the next one down.
        if estimate_overlap and len(tiles) > 1:
            dcorr = np.zeros([len(tiles), 2], dtype='float32')
            if rank == 0:
                nang  = max(2, min(overlap_nangles, ntheta))
                angs  = [int(j) for j in
                         np.linspace(0, ntheta, nang, endpoint=False)]
                fids0 = [h5py.File(tp, 'r') for tp in tile_paths]
                try:
                    dcorr[:] = _estimate_overlap(fids0, srdata, angs)
                finally:
                    for f in fids0:
                        f.close()
            comm.Bcast(dcorr, root=0)
            offsets = (offsets + dcorr).astype('float32')
            _apply_offsets()
            if rank == 0:
                for t, tl in enumerate(tiles):
                    logger.info(f'step5 bin={bin}: {tl:<9s} shift  '
                                f'v={offsets[t, 0]:+9.2f}  h={offsets[t, 1]:+10.2f}   '
                                f'(moved {dcorr[t, 0]:+7.2f} {dcorr[t, 1]:+7.2f})  '
                                f'finest-grid object px')
                logger.debug(f'step5 bin={bin}: overlap now {ov_bin} px, '
                             f'feather {tfeather} px, paste origins '
                             f'x={list(origin_x)} y={list(origin_y)}')

        # --- Record the measured placement next to cshifts_final -------------
        # cshifts_final is tile-local and says nothing about where the tile sits
        # on the mosaic; that is what this adds. The whole [ntiles, 2] table
        # goes into every tile file, together with the tile order and this
        # tile's row, so a single file is enough to rebuild the layout.
        # Rewritten once per bin, so the finest measurement is what survives,
        # and written *before* the pin and any HOLO_TILE_ERR perturbation —
        # what lands on disk is always the real measurement.
        if rank == 0:
            for _t, _tp in enumerate(tile_paths):
                with h5py.File(_tp, 'a') as _fid:
                    if '/exchange/tile_offsets' in _fid:
                        del _fid['/exchange/tile_offsets']
                    _ds = _fid.create_dataset('/exchange/tile_offsets',
                                              data=offsets.astype('float32'))
                    _ds.attrs['tiles']   = [str(x) for x in tiles]
                    _ds.attrs['index']   = _t
                    _ds.attrs['columns'] = ['v', 'h']
                    _ds.attrs['units']   = 'object px on the finest grid'
                    _ds.attrs['bin']     = bin
                    _ds.attrs['estimated'] = bool(estimate_overlap and len(tiles) > 1)
            logger.info(f'step5 bin={bin}: wrote /exchange/tile_offsets '
                        f'({len(tiles)}x2, finest-grid px) to the '
                        f'{len(tile_paths)} tile files')

        # TEMPORARY: pin the placement to a measured run, so the sweep varies
        # only HOLO_TILE_ERR and not estimate_overlap's re-measurement. Applied
        # after tile_offsets is written, so what lands on disk stays the real
        # measurement. (v, h) in object px on the finest grid.
        offsets = np.array([[1.2769033e+01,  4.0267793e+03],
                            [1.0637471e+01,  2.0140543e+03],
                            [0.0000000e+00,  1.6000000e+01],
                            [1.3013465e+00, -1.9891675e+03],
                            [1.1986929e+01, -3.9903440e+03]], dtype='float32')
        _apply_offsets()
        if rank == 0:
            logger.warning(f'step5 bin={bin}: placement PINNED to the hardcoded '
                           f'offsets h={list(offsets[:, 1])}, paste origins '
                           f'x={list(origin_x)}')
        comm.Barrier()

        # --- TEMPORARY: deliberate placement error on one tile ---------------
        # Applied last, so it survives estimate_overlap. Meant for a run over a
        # single bin (start_level_rec == nlevels - 1): `offsets` carries across
        # bins, so on a multi-bin run the error would both accumulate and be
        # measured away again by the next bin's estimate_overlap.

        # offsets[2, 1] += 16
        # offsets[1, 1] += 4
        # offsets[3, 1] += 26
        # offsets[4, 1] += 26
        # offsets[0, 1] += 1
        # print(offsets)
        if tile_err is not None:
            offsets = offsets.copy()
            offsets[proc_tile, 1] += tile_err
            _apply_offsets()
            if rank == 0:
                logger.warning(f'step5 bin={bin}: TEST SWEEP, HOLO_TILE_ERR='
                               f'{tile_err:g} finest-grid px on '
                               f'{tiles[proc_tile]!r} -> h={offsets[proc_tile, 1]:+.2f}, '
                               f'paste origins x={list(origin_x)}')

        # --- Background level from projection 0 (rank 0), broadcast ----------
        # Top rows of the covered area, which are air on a tomographic scan.
        calib = np.zeros(1, dtype='float32')
        if rank == 0:
            fids0 = [h5py.File(tp, 'r') for tp in tile_paths]
            try:
                _mosaic(fids0, srdata, mosaic, wsum, 0)
                ph0  = _paganin(0)
                cov  = wsum > 0.5
                rows = max(8, 16 * n_bin // 512)
                ys   = int(cp.argmax(cov.any(axis=1).astype('int8')))
                strip = ph0[ys:ys + rows][cov[ys:ys + rows]]
                calib[0] = float(cp.median(strip)) if strip.size else 0.0
            finally:
                for f in fids0:
                    f.close()
        comm.Bcast(calib, root=0)
        global_bg = float(calib[0])
        if rank == 0:
            logger.info(f'step5 bin={bin}: global_bg = {global_bg:.6f}')

        # --- Stitch + Paganin for this rank's projections --------------------
        # positions within theta_ids, not absolute angle indices
        proj_pos     = [start5 + i for i in range(len(ids5))
                        if (start5 + i) % proj_step == 0]
        proj_buf     = np.empty([len(proj_pos), nzobj_bin, nobj_bin], dtype='float32')
        local_recPag = np.empty([len(ids5), nzobj_bin, nobj_bin], dtype='float32')

        n_srdata_save = min(8, ntheta5)
        with h5py.File(fpath_mosaic + '_srdata.h5', 'a', driver='mpio', comm=comm) as fsr:
            srdata_ds = fsr.create_dataset(f'/exchange/srdata_bin{bin}',
                                           shape=(n_srdata_save, nzobj_bin, nobj_bin),
                                           dtype='float32')
            fids = [h5py.File(tp, 'r') for tp in tile_paths]
            try:
                ip = 0
                for i, j in enumerate(ids5):
                    j, g = int(j), start5 + i      # absolute angle, subset position
                    _mosaic(fids, srdata, mosaic, wsum, j)
                    if g < n_srdata_save:
                        srdata_ds[g] = mosaic[0].get()
                    phase = _paganin(j) - global_bg
                    local_recPag[i] = phase.get()
                    if g % proj_step == 0:
                        proj_buf[ip] = local_recPag[i]
                        ip += 1
                    if i % 50 == 0:
                        logger.info(f'step5 bin={bin}: proj {j:4d}/{ntheta} '
                                    f'({g + 1}/{ntheta5})')
            finally:
                for f in fids:
                    f.close()

        logger.info(f'step5 bin={bin}: rank {rank:4d}  paganin norm = '
                    f'{np.linalg.norm(proj_buf):.6e}')

        # --- Stitched Paganin projections (every proj_step-th) ---------------
        n_proj_10 = len(range(0, ntheta5, proj_step))
        with h5py.File(fpath_mosaic + '_proj.h5', 'a', driver='mpio', comm=comm) as fid:
            proj_ds = fid.create_dataset(f'/exchange/proj_bin{bin}',
                                         shape=(n_proj_10, nzobj_bin, nobj_bin),
                                         dtype='float32')
            for i, g in enumerate(proj_pos):
                proj_ds[g // proj_step] = proj_buf[i]
        del proj_buf
        if rank == 0:
            logger.info(f'step5 bin={bin}: wrote /exchange/proj_bin{bin} '
                        f'({n_proj_10} frames, every {proj_step}th of the '
                        f'{ntheta5} reconstructed angles) to {fpath_mosaic}_proj.h5')

        del cl_shift, srdata, mosaic, wsum, cref_smooth, r_gpu
        cp.get_default_memory_pool().free_all_blocks()

        # --- Redistribute theta-slab -> z-slab, then FBP ---------------------
        cl_mpi5  = MPIClass(comm, nzobj_bin, ntheta5, nobj_bin, 'float32')
        local_nz = cl_mpi5.local_nzobj
        z_start  = cl_mpi5.st_obj
        logger.debug(f'step5 bin={bin}: z-range [{z_start}:{cl_mpi5.end_obj}), '
                     f'local_nz={local_nz}')

        psi_z = np.empty((ntheta5, local_nz, nobj_bin), dtype='float32')
        cl_mpi5.redist(local_recPag, psi_z, direction='backward')
        del local_recPag

        psi_z_c = np.empty((ntheta5, local_nz, nobj_bin), dtype='complex64')
        psi_z_c.real[:] = psi_z
        psi_z_c.imag[:] = psi_z / paganin
        del psi_z

        # --- TEMPORARY: in sweep mode reconstruct the middle slice only ------
        # It is the only slice the sweep looks at, and dropping the rest makes
        # the FBP ~nzobj_bin times cheaper. The rank owning that z keeps one
        # slice, the others keep none; the FBP is rank-local and gpu_batch
        # returns immediately on an empty output, so idle ranks cost nothing.
        # z_start/local_nz are narrowed with it, which is why the object volume
        # is not written below — rec_loc no longer holds one.
        _zmid = nzobj_bin // 2
        if tile_err is not None:
            if z_start <= _zmid < z_start + local_nz:
                _k = _zmid - z_start
                psi_z_c = psi_z_c[:, _k:_k + 1].copy()
                z_start, local_nz = _zmid, 1
            else:
                psi_z_c = psi_z_c[:, :0].copy()
                local_nz = 0
            logger.info(f'step5 bin={bin}: TEST SWEEP, rank {rank:4d} FBP of '
                        f'{local_nz} slice(s) (middle slice is {_zmid})')

        rec_loc = np.zeros((local_nz, nobj_bin, nobj_bin), dtype='complex64')
        cl_tomo = Tomo(nobj_bin, nchunk, theta5, mask_r=0.9)
        nbytes  = 2 * (ntheta5 * nchunk * nobj_bin + nchunk * nobj_bin**2) \
            * np.dtype('complex64').itemsize
        cl      = Chunking(nbytes, nchunk)

        @cl.gpu_batch(axis_out=0, axis_inp=1, nout=1)
        def _fbp(_, rec_loc, psi_z_c):
            rec_loc[:] = cl_tomo.fbp(psi_z_c, 'ramp')

        logger.info(f'step5 bin={bin}: FBP start, local_nz={local_nz}, '
                    f'nobj_bin={nobj_bin}')
        _fbp(cl, rec_loc, psi_z_c)
        logger.info(f'step5 bin={bin}: rank {rank:4d}  fbp norm = '
                    f'{np.linalg.norm(rec_loc):.6e}')
        del psi_z_c

        # --- TEMPORARY: drop the middle slice of the sweep as a tiff ---------
        if tile_err is not None and z_start <= _zmid < z_start + local_nz:
            import tifffile
            os.makedirs(tile_err_dir, exist_ok=True)
            _sl = rec_loc[_zmid - z_start].real.astype('float32')
            # Named by the tile being nudged, and by the error offset by +100 so
            # the names sort in error order and carry no minus sign:
            # -10 -> fbp_center_90, 0 -> fbp_center_100, +10 -> fbp_center_110.
            _tag = f'{tiles[proc_tile] or proc_tile}_{tile_err + 100:g}'
            _fn = f'{tile_err_dir}/fbp_{_tag}.tiff'
            tifffile.imwrite(_fn, _sl)
            logger.warning(f'step5 bin={bin}: TEST SWEEP, wrote slice {_zmid} '
                           f'to {_fn}')

        paganin_tag = int(paganin) if paganin == int(paganin) else paganin
        _wbatch = max(1, (1 << 28) // (nobj_bin * nobj_bin * 4))
        if tile_err is not None:
            # TEMPORARY: only the middle slice was reconstructed, so there is no
            # volume to write. _obj.h5 is left empty on purpose — step 6 cannot
            # use the output of a sweep run.
            if rank == 0:
                logger.warning(f'step5 bin={bin}: TEST SWEEP, skipping '
                               f'{fpath_mosaic}_obj.h5 (middle slice only)')
        else:
            with h5py.File(fpath_mosaic + '_obj.h5', 'a', driver='mpio', comm=comm) as fid:
                re_ds = fid.create_dataset(f'/exchange/obj_init_re{paganin_tag}_{bin}',
                                           shape=(nzobj_bin, nobj_bin, nobj_bin),
                                           dtype='float32')
                im_ds = fid.create_dataset(f'/exchange/obj_init_imag{paganin_tag}_{bin}',
                                           shape=(nzobj_bin, nobj_bin, nobj_bin),
                                           dtype='float32')
                for _i0 in range(0, local_nz, _wbatch):
                    _i1 = min(_i0 + _wbatch, local_nz)
                    re_ds[z_start + _i0: z_start + _i1] = rec_loc[_i0:_i1].real
                    im_ds[z_start + _i0: z_start + _i1] = rec_loc[_i0:_i1].imag
        del rec_loc
        cp.get_default_memory_pool().free_all_blocks()

        if rank == 0:
            logger.info(f'Step 5: bin={bin} done.')
        comm.Barrier()

    if rank == 0:
        logger.info('Step 5: done.')

comm.Barrier()
