"""
Step 6 (mosaic) — full iterative reconstruction of a multi-tile scan (MPI).

Launch with mpirun / mpiexec, one rank per GPU:

    mpirun -np <ngpus> python step6.py config_step6_bin3.conf

Steps 1-5 of ``steps15.py`` produced, per tile, the preprocessed data
(``pdata{k}_{bin}``), the flat fields (``pref_{bin}``), the per-angle shifts
(``cshifts_final``) and the measured tile placement (``tile_offsets``); and, for
the mosaic as a whole, the Paganin/FBP starting volume in
``{pfile}_mosaic_obj.h5``. This driver hands all of that to the Bilinear-Hessian
solver and refines object, probes and positions together.

The solver never learns that the scan was tiled. ``MosaicReader`` presents the
N tiles as one scan with ``ndist = ntiles * ndist_tile`` entries in tile-major
order, the tile's place on the mosaic folded into the position
(``cshifts_final + tile_offsets``). ``rec_mpi.Rec`` derives the propagator,
magnifications and voxel size from ``z1`` alone, which is simply tiled to match
— so no solver change is involved.

Shrinkage is on, matching what ``steps15.py`` applies in Steps 4-5: this is
``rec_mpi_shrink.Rec``, whose ``tp`` variable holds the two linear coefficients
(A, B) of ``shrink(theta) = A*t + B`` per (tile x distance, axis), giving
``demag = (1 + shrink) / norm_magnifications``. ``init_tp_from_shrink`` seeds it
with a least-squares fit to each tile's stored ``/exchange/shrink``, and the
solver refines it. ``rho`` therefore needs a 4th entry (the tp step scale).

Set ``pos_per_dist = True`` in the config to swap the position model for
``rec_mpi_shrink_posdist.Rec``: the measured per-angle shifts from
``cshifts_final`` are then held FIXED, and the solver refines a single (y, x)
shift per (tile x distance) — ``ndist*2`` unknowns instead of
``ntheta*ndist*2``. Checkpoints keep the same ``(ntheta, ndist, 2)`` ``/pos``
dataset (base + refined offset), so they remain interchangeable with a
per-projection run.

Checkpoints are written periodically to ``path_out``; if one exists the run
resumes from the latest saved iteration.
"""

import sys
from mpi4py import MPI
from holotomocupy.rec_mpi_shrink import Rec as RecPosTheta
from holotomocupy.rec_mpi_shrink_posdist import Rec as RecPosDist
from holotomocupy.config import parse_args
from holotomocupy.mpi_functions import MPIClass
from holotomocupy.reader import MosaicReader, find_latest_checkpoint
from holotomocupy.writer import Writer
from holotomocupy.logger_config import logger, set_log_level

import numpy as np
import cupy as cp
cp.cuda.set_pinned_memory_allocator(None)

# --- Parse configuration file -------------------------------------------
args = parse_args(sys.argv[1])
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
args.comm = comm
set_log_level(args.log_level)

if not args.tiles:
    raise SystemExit(f'{sys.argv[1]}: no tiles= listed. This driver is for the '
                     f'mosaic; use the single-tile step6.py otherwise.')
if args.init_vol and rank == 0:
    logger.warning(f'init_vol={args.init_vol} is ignored by the mosaic driver; '
                   f'the starting object always comes from the step-5 mosaic '
                   f'volume (or a checkpoint).')

# --- Object grid ---------------------------------------------------------
# Set in the config, never derived: nzobj/nobj here are step 5's divided by
# 2**bin, and both are set by hand there too. Checked against the volume that
# is actually going to be loaded, since a mismatch is a config error, not
# something to paper over by resizing the grid.
if args.nobj <= 0 or args.nzobj <= 0:
    raise SystemExit('nzobj and nobj must both be set in the config '
                     '(step 5 grid / 2**bin)')
# Warn only: a run resuming from a checkpoint never touches the step-5 volume,
# so a missing or unreadable one is not by itself an error here.
shape = np.zeros(2, dtype='int64')
if rank == 0:
    try:
        shape[:] = MosaicReader.mosaic_obj_shape(args.mosaic_file,
                                                 args.paganin, args.bin)
    except Exception as _e:
        logger.warning(f'could not read the step-5 mosaic shape, so nzobj/nobj '
                       f'go unchecked: {_e}')
    if shape.any() and (args.nzobj, args.nobj) != (int(shape[0]), int(shape[1])):
        logger.warning(f'config nzobj={args.nzobj} nobj={args.nobj} does not '
                       f'match {args.mosaic_file.replace(".h5", "_obj.h5")}, '
                       f'which is {shape[0]}x{shape[1]}')

# --- Flatten tile x distance into the solver's distance axis -------------
# `ndist` in the config is per tile; entry t*ndist_tile + k is tile t at
# distance k.
args.ndist_tile = args.ndist
args.ndist      = args.ndist_tile * len(args.tiles)

# --- Distribute object and projection slices across MPI ranks -----------
cl_mpi = MPIClass(comm, args.nzobj, args.ntheta, args.nobj, args.obj_dtype)

# --- Build I/O helpers --------------------------------------------------
reader = MosaicReader(
    args.tile_files, args.mosaic_file, comm,
    cl_mpi.st_obj, cl_mpi.end_obj, args.nzobj, args.nobj,
    cl_mpi.st_theta, cl_mpi.end_theta, args.ntheta,
    args.ndist_tile, args.nz, args.n, args.obj_dtype,
    args.paganin, args.rotation_center_shift, args.start_theta, args.bin,
    tiles=args.tiles,
)
writer = Writer(
    args.path_out, comm,
    cl_mpi.st_obj, cl_mpi.end_obj, args.nzobj, args.nobj,
    cl_mpi.st_theta, cl_mpi.end_theta, args.ntheta,
    args.ndist, args.nz, args.n, args.obj_dtype,
    ndist_tile=args.ndist_tile,   # pos-error plot: one tile per row
)

# Physics parameters come from the converted /exchange/* HDF5 archive (via
# MosaicReader, which tiles z1 across the tiles). steps15.py is the single
# source of truth — it bakes binning and preprocessing into what it writes.
args.energy                  = args.energy if args.energy is not None else reader.energy
args.focustodetectordistance = reader.focustodetectordistance
args.z1                      = reader.z1
args.detector_pixelsize      = reader.detector_pixelsize
args.theta                   = reader.theta

# --- Print run summary (rank 0 only) ------------------------------------
if rank == 0:
    mag      = args.focustodetectordistance / args.z1[:args.ndist_tile]
    voxel_nm = args.detector_pixelsize / mag[0] * 1e9
    logger.info("=" * 60)
    logger.info(f"  energy               : {args.energy:.4f} keV")
    logger.info(f"  detector pixel size  : {args.detector_pixelsize*1e9:.3f} nm  (bin={args.bin})")
    logger.info(f"  voxel size           : {voxel_nm:.3f} nm")
    logger.info(f"  focus-det distance   : {args.focustodetectordistance*100:.3f} cm")
    logger.info(f"  z1 per tile          : {[f'{v*100:.3f} cm' for v in args.z1[:args.ndist_tile]]}")
    logger.info(f"  magnifications       : {np.array2string(mag, precision=4)}")
    logger.info(f"  detector size        : {args.nz} x {args.n}")
    logger.info(f"  object size          : {args.nzobj} x {args.nobj} x {args.nobj}")
    logger.info(f"  n angles             : {args.ntheta}  (start={args.start_theta})")
    logger.info(f"  tiles                : {args.tiles}")
    logger.info(f"  n distances          : {args.ndist}  = {len(args.tiles)} tiles x {args.ndist_tile}")
    logger.info(f"  rotation center shift: {args.rotation_center_shift:.4f} px")
    logger.info(f"  paganin              : {args.paganin}")
    logger.info(f"  mask                 : {args.mask}")
    logger.info(f"  n MPI ranks          : {comm.Get_size()}")
    logger.info(f"  position unknowns    : "
                + (f"{args.ndist} x 2  (one shift per tile x distance, "
                   f"per-angle shifts held fixed)" if args.pos_per_dist else
                   f"{args.ntheta} x {args.ndist} x 2  (one shift per projection)"))
    logger.info(f"  pfile                : {args.pfile}")
    logger.info(f"  path_out             : {args.path_out}")
    logger.info("=" * 60)

# --- Initialise the reconstruction class --------------------------------
# pos_per_dist swaps the position model: instead of one refined shift per
# (projection, distance), the measured per-angle shifts stay fixed in
# `cl.pos_base` and a single (y, x) shift per (tile, distance) is refined.
# Every read below therefore targets `pos_target`, which is `cl.pos_base` in
# that mode and `cl.vars['pos']` otherwise — both are (local_ntheta, ndist, 2).
Rec = RecPosDist if args.pos_per_dist else RecPosTheta
logger.info("Create class")
cl = Rec(args)
pos_target = cl.pos_base if args.pos_per_dist else cl.vars['pos']
logger.info(f"obj-range [{cl.st_obj}:{cl.end_obj}), local size: {cl.end_obj-cl.st_obj} x {cl.nobj} x {cl.nobj}")
logger.info(f"projt-range [{cl.st_theta}:{cl.end_theta}), local size: {cl.end_theta-cl.st_theta} x {cl.nzobj} x {cl.nobj}")

# --- Load measurements and reference (flat-field) data -----------------
logger.info("Read data")
reader.read_data(out=cl.data)
reader.read_ref(out=cl.ref)
# Initial guess for the linear-parameterized shrink variable vars['tp']: a
# closed-form 2-parameter linear LS fit to each tile's stored /exchange/shrink,
# per (tile x distance, axis). The solver refines it from there. No persistent
# shrink buffer is kept — MosaicReader.read_shrink is called inside
# init_tp_from_shrink and freed after the fit.
cl.init_tp_from_shrink(reader)

# --- Load initial variables (object, probe, positions) ------------------
# Resume from the latest checkpoint if one exists; otherwise start from the
# step-5 mosaic Paganin/FBP volume.
logger.info("Read initial variables")
ckpt = find_latest_checkpoint(args.path_out, args.start_iter)
if ckpt:
    logger.info(f"Resuming from checkpoint: {ckpt}")
    reader.read_checkpoint(ckpt, out_obj=cl.vars['obj'], out_pos=pos_target,
                           out_prb=cl.vars['prb'], out_tp=cl.vars['tp'])
else:
    reader.read_obj(out=cl.vars['obj'])
    reader.read_pos(out=pos_target)
    if args.prb_file:
        logger.info(f"Loading probes from: {args.prb_file}")
    reader.read_prb(prb_file=args.prb_file, out=cl.vars['prb'])
if args.pos_checkpoint:
    logger.info(f"Overriding positions from: {args.pos_checkpoint}")
    reader.read_pos_checkpoint(args.pos_checkpoint, out=pos_target)

# A wrong tile placement is the one failure mode that produces a plausible but
# meaningless result, so print, before the first iteration: the tile_offsets as
# read from /exchange/tile_offsets, and the window centre they put each tile at.
# These must match what step 5 stitched the initial object with — steps15.py
# logs its own "placement PINNED / tile_offsets" lines for comparison.
if rank == 0 and reader.tile_offsets is not None:
    scale = 1.0 / 2**args.bin
    logger.warning(f"  /exchange/tile_offsets read from the {len(args.tiles)} "
                   f"tile files (object px on the finest grid):")
    for t, tl in enumerate(args.tiles):
        o = reader.tile_offsets[t]
        logger.warning(f"    {tl:<10s} v={o[0]:+9.4f} h={o[1]:+11.4f}   "
                       f"-> bin {args.bin}: v={o[0]*scale:+8.3f} "
                       f"h={o[1]*scale:+10.3f}")
    pos0 = cp.asnumpy(pos_target[0])
    logger.warning(f"  tile window centres at angle {args.start_theta} "
                   f"(x = (nobj-1)/2 - pos[...,1], nobj={args.nobj}):")
    for t, tl in enumerate(args.tiles):
        p = pos0[t * args.ndist_tile]
        logger.warning(f"    {tl:<10s} pos=({p[0]:+9.3f},{p[1]:+10.3f})  "
                       f"x={(args.nobj - 1) / 2 - p[1]:9.3f}")
elif rank == 0:
    logger.warning("  tile_offsets not read this run (resumed from a "
                   "checkpoint, so positions came from it, not from "
                   "/exchange/tile_offsets)")

# --- Run iterative reconstruction ---------------------------------------
logger.info("Run reconstruction")
vars = cl.BH(writer)
