"""
Step 6E -- energy sweep on top of a FINISHED reconstruction (MPI).

    mpiexec -n <ngpus> ./bind.sh python step6_energy.py config_energy_bin1_e17p000.conf

Same solver as step6.py, three differences, all of them because this is a
*probe of one parameter* and not a reconstruction:

  1. `energy=` in the config overrides /exchange/energy from the data file.
     Energy enters the solver in exactly one place -- rec_mpi.py builds
     `wavelength = 1.24e-9 / energy` and hands it to Propagation -- so the
     only thing that changes between the runs of a sweep is the Fresnel
     kernel.  Object, probe, positions, shrinkage, mask, rho and the data
     itself are bit-identical.

  2. `init_checkpoint=` names the checkpoint to start from EXPLICITLY, instead
     of step6.py's find_latest_checkpoint(path_out, start_iter).  A sweep run
     must write somewhere other than the reconstruction it seeds from, and
     find_latest_checkpoint only ever looks inside path_out.

  3. NOTHING IS CHECKPOINTED.  The only output is one TIFF per vertical slice
     per saved iteration (plus conv.csv).  A bin-1 checkpoint of this scan is
     68 GB and a bin-0 one is 550 GB; at 17 energies that is 1.2 TB / 9 TB
     of volumes nobody is going to open.  See SliceWriter below -- it is a
     drop-in for Writer as far as Rec.vis_debug is concerned, but it writes
     slices, not volumes.

WHAT TO LOOK AT WHEN IT FINISHES

  conv.csv in each path_out.  The energies share the data, the seed and the
  iteration count, so `err` is directly comparable between them and the
  minimum over the sweep is the answer.  Two rows matter:
    iter=-1    the residual of the SEED under this energy's propagator, before
               a single iteration -- the propagator error on its own, with
               nothing yet refitted to hide it;
    iter=<last> the residual after the run has had 128 iterations to absorb it.
  A sweep where -1 separates cleanly but the last row does not means the
  refinement is absorbing the mismatch (probe and positions are free here);
  rerun with rho[prb]=rho[pos]=0 to see the propagator alone.

  slices/vert_E*_y*_it*.tiff.  Same y, same iteration, every energy -- open
  them as a stack.

FINAL ITERATION.  The loop runs i = start_iter .. niter-1, and Rec logs the
error and fires the writer only on multiples of error_step / checkpoint_step.
With start_iter=1280 and 128 iterations the last index is 1407, which is odd
and so lands on no step at all -- the run would end without ever recording the
state it ended in.  Rather than round the iteration count up to make 1408
divisible, this script runs precalc / _iterate / postcalc itself and forces one
final readout between the loop and postcalc, which is where it has to happen:
postcalc multiplies obj back by norm_const, and self.min() expects the
normalised obj the loop worked with.
"""

import os
import sys
import configparser

import h5py
import numpy as np
import tifffile
from mpi4py import MPI

import cupy as cp
cp.cuda.set_pinned_memory_allocator(None)

from holotomocupy.rec_mpi import Rec
from holotomocupy.config import parse_args
from holotomocupy.mpi_functions import MPIClass
from holotomocupy.reader import Reader
from holotomocupy.logger_config import logger, set_log_level


def parse_extra(config_file):
    """The two keys this script adds on top of holotomocupy.config.parse_args.

    parse_args builds a fixed namespace and silently ignores anything it does
    not know, so the extras are re-read here rather than added to the shared
    config.py -- no other step has any use for them.

      init_checkpoint  path to the checkpoint_*.h5 to start from.  Absolute, or
                       relative to the directory holding the config file.
      vert_slices      comma-separated y indices (object columns) to save, in
                       THIS level's grid.  Empty = the middle one, nobj//2,
                       which is the cut through the rotation axis.
    """
    p = configparser.ConfigParser(inline_comment_prefixes=("#",))
    with open(config_file, "r", encoding="utf-8") as f:
        p.read_string("[DEFAULT]\n" + f.read())
    c    = p["DEFAULT"]
    here = os.path.dirname(os.path.abspath(config_file))

    ckpt = c.get("init_checkpoint", fallback="")
    ckpt = ckpt.strip()
    if not ckpt:
        raise ValueError(f"Missing required field in {config_file}: init_checkpoint")
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(here, ckpt)

    ys = [int(x.strip()) for x in c.get("vert_slices", fallback="").split(",") if x.strip()]
    return ckpt, ys


class SliceWriter:
    """Vertical-slice writer, drop-in for Writer in Rec.vis_debug.

    Rec touches a writer through exactly two names -- `path_out` (only
    check_approximation, which is off here) and `write_checkpoint(...)` -- so
    this does not subclass Writer: inheriting would create the `checkpoints/`
    and `checkpoints_tiff/` directories this script exists to avoid, and every
    line of Writer.write_checkpoint that matters is the collective HDF5 write
    that is being dropped.

    The gather is the same trick Writer uses for its own vertical preview: a
    vertical cut is nzobj scattered reads out of a file, but every rank already
    holds its own contiguous z-slab of it, so it is one comm.gather of
    (local_nzobj, len(ys), nobj) floats -- 2 MB per rank at bin 1 -- and no
    file I/O off rank 0.
    """

    def __init__(self, path_out, comm, nzobj, nobj, energy, ys=None):
        self.path_out = path_out
        self.comm     = comm
        self.rank     = comm.Get_rank()
        self.nzobj    = nzobj
        self.nobj     = nobj
        self.energy   = energy
        self.ys       = list(ys) if ys else [nobj // 2]
        for y in self.ys:
            if not 0 <= y < nobj:
                raise ValueError(f"vert_slices: y={y} outside [0, {nobj})")
        # 17.1 -> "17p100": a tag that survives being copied into a directory
        # next to the other energies of the sweep.
        self.tag  = f"E{energy:.3f}".replace(".", "p")
        self.sdir = os.path.join(path_out, "slices")
        if self.rank == 0:
            os.makedirs(self.sdir, exist_ok=True)
        comm.Barrier()

    @staticmethod
    def _cpu(x):
        return x.get() if isinstance(x, cp.ndarray) else np.asarray(x)

    def write_checkpoint(self, vars, i, norm_const, pos_init=None,
                         shrink=None, shrink_init=None, shrink_gt=None):
        """Save one TIFF per requested vertical slice.  Signature matches
        Writer.write_checkpoint because Rec.vis_debug calls it by keyword; the
        shrink arguments are ignored (rho[tp]=0 here, shrinkage never moves)
        and pos_init is used for the drift log line only, not for the PNG."""
        # (local_nzobj, len(ys), nobj) -- one strided read, no full-volume copy.
        v_local = self._cpu(vars['obj'][:, self.ys, :].real).astype('float32')
        v_local = v_local * np.float32(norm_const)
        v_parts = self.comm.gather(v_local, root=0)

        if self.rank == 0:
            vol = np.concatenate(v_parts, axis=0)          # (nzobj, len(ys), nobj)
            for j, y in enumerate(self.ys):
                path = os.path.join(
                    self.sdir, f"vert_{self.tag}_y{y:04d}_it{i:05d}.tiff")
                tifffile.imwrite(path, np.ascontiguousarray(vol[:, j, :]))
            logger.info(f"SliceWriter: iter={i}  {len(self.ys)} vertical slice(s) "
                        f"-> {self.sdir}")

        if pos_init is not None:
            self._log_pos_drift(vars['pos'] - pos_init, i)

    def _log_pos_drift(self, delta_local, i):
        """How far positions have moved from the seed, per distance and axis.

        Worth one line per saved iteration in a sweep: a wrong propagator and a
        position error look alike to the solver, so if one energy's positions
        run away while another's sit still, the residuals are not measuring the
        same thing and the comparison needs rho[pos]=0 to be clean."""
        parts = self.comm.gather(self._cpu(delta_local), root=0)
        if self.rank != 0:
            return
        d = np.abs(np.concatenate(parts, axis=1))          # [ndist, ntheta, 2]
        msg = "  ".join(
            f"d{k}: y={d[k,:,0].mean():.4f}+-{d[k,:,0].std():.4f} "
            f"(max {d[k,:,0].max():.4f})"
            f"  x={d[k,:,1].mean():.4f}+-{d[k,:,1].std():.4f} "
            f"(max {d[k,:,1].max():.4f})"
            for k in range(d.shape[0]))
        logger.warning(f"iter={i}: pos drift from seed [px]  {msg}")


# --- Parse configuration file -------------------------------------------
args = parse_args(sys.argv[1])
init_ckpt, vert_ys = parse_extra(sys.argv[1])
comm = MPI.COMM_WORLD
args.comm = comm
set_log_level(args.log_level)

if args.energy is None:
    raise ValueError(f"{sys.argv[1]}: energy= is required -- this script exists "
                     f"to override it, and leaving it out would silently run "
                     f"the nominal energy from the data file")
if not os.path.exists(init_ckpt):
    raise FileNotFoundError(f"init_checkpoint does not exist: {init_ckpt}")

# --- Distribute object and projection slices across MPI ranks -----------
cl_mpi = MPIClass(comm, args.nzobj, args.ntheta, args.nobj, 'complex64')

# --- Build I/O helpers --------------------------------------------------
reader = Reader(
    args.in_file, comm,
    cl_mpi.st_obj, cl_mpi.end_obj, args.nzobj, args.nobj,
    cl_mpi.st_theta, cl_mpi.end_theta, args.ntheta,
    args.ndist, args.nz, args.n,
    args.paganin, args.rotation_center_shift, args.start_theta, args.bin,
)
writer = SliceWriter(args.path_out, comm, args.nzobj, args.nobj,
                     args.energy, vert_ys)

# Geometry comes from the data file as usual; ENERGY DOES NOT.
energy_nominal               = reader.energy
args.focustodetectordistance = reader.focustodetectordistance
args.z1                      = reader.z1
args.detector_pixelsize      = reader.detector_pixelsize
args.theta                   = reader.theta

# Reader.read_checkpoint infers its upsampling factor as self.n // ckpt_n, so a
# checkpoint FINER than this level silently gives scale=0.  Catch it here, with
# the two paths in the message, rather than in an integer divide 20 minutes in.
# Broadcast rather than raise on rank 0: a lone rank raising leaves the others
# blocked in the next collective, and the job then burns its walltime on a
# deadlock instead of printing the reason it stopped.
_ckpt_n = np.zeros(1, dtype='int64')
if comm.Get_rank() == 0:
    with h5py.File(init_ckpt, 'r') as f:
        _ckpt_n[0] = f['prb_abs'].shape[-1]
comm.Bcast(_ckpt_n, root=0)
_ckpt_n = int(_ckpt_n[0])
if _ckpt_n > args.n or args.n % _ckpt_n:
    raise ValueError(
        f"init_checkpoint is n={_ckpt_n} but this level is n={args.n}. "
        f"read_checkpoint upsamples only (scale = n // ckpt_n), so seed a "
        f"level from a checkpoint at its own resolution or coarser.\n"
        f"  checkpoint: {init_ckpt}\n  config    : {sys.argv[1]}")

# --- Print run summary (rank 0 only) ------------------------------------
if comm.Get_rank() == 0:
    mag  = args.focustodetectordistance / args.z1[0]
    voxel_nm = args.detector_pixelsize / mag * 1e9
    niter_run = args.niter - args.start_iter
    logger.info("=" * 60)
    logger.info(f"  ENERGY SWEEP RUN -- no checkpoints, vertical slices only")
    logger.info(f"  energy               : {args.energy:.4f} keV   "
                f"(data file says {energy_nominal:.4f}, "
                f"delta {args.energy - energy_nominal:+.4f})")
    logger.info(f"  seed checkpoint      : {init_ckpt}")
    logger.info(f"  iterations           : {args.start_iter} .. {args.niter - 1}"
                f"  ({niter_run} iters)")
    logger.info(f"  vertical slices at y : {writer.ys}")
    logger.info(f"  detector pixel size  : {args.detector_pixelsize*1e9:.3f} nm  (bin={args.bin})")
    logger.info(f"  voxel size           : {voxel_nm:.3f} nm")
    logger.info(f"  focus-det distance   : {args.focustodetectordistance*100:.3f} cm")
    logger.info(f"  z1 distances         : {[f'{v*100:.3f} cm' for v in args.z1]}")
    logger.info(f"  detector size        : {args.nz} x {args.n}")
    logger.info(f"  object size          : {args.nzobj} x {args.nobj} x {args.nobj}")
    logger.info(f"  n angles             : {args.ntheta}  (start={args.start_theta})")
    logger.info(f"  n distances          : {args.ndist}")
    logger.info(f"  rotation center shift: {args.rotation_center_shift:.4f} px")
    logger.info(f"  rho                  : {args.rho}")
    logger.info(f"  n MPI ranks          : {comm.Get_size()}")
    logger.info(f"  path_out             : {args.path_out}")
    logger.info("=" * 60)

# --- Initialise the reconstruction class --------------------------------
logger.info("Create class")
cl = Rec(args)

# --- Load measurements and reference (flat-field) data -----------------
logger.info("Read data")
reader.read_data(out=cl.data)
reader.read_ref(out=cl.ref)
reader.read_shrink(out=cl.shrink_nd)
cl.init_tp_from_shrink()

# --- Seed every variable from the finished reconstruction ---------------
# obj, prb, pos and tp all come from the checkpoint; nothing is read from the
# Paganin init.  This is the whole point -- every energy of a sweep starts
# from bit-identical state and differs only in the propagator.
logger.info(f"Seeding from checkpoint: {init_ckpt}")
reader.read_checkpoint(init_ckpt,
                       out_obj=cl.vars['obj'], out_pos=cl.vars['pos'],
                       out_prb=cl.vars['prb'], out_bd=cl.vars.get('bd'),
                       out_tp=cl.vars['tp'])

# --- Run, with a forced final readout -----------------------------------
# BH() open-coded so the last iteration is recorded even when it does not land
# on error_step / checkpoint_step -- see the module docstring.  estimate_rho is
# not replicated: a sweep must not retune rho per energy, or the energies stop
# being comparable, so the configs set estimate_rho=False and this asserts it.
if cl.estimate_rho:
    raise ValueError("estimate_rho must be False in a sweep: retuning rho per "
                     "energy would make the residuals incomparable")

logger.info("Run reconstruction")
vars = cl.vars
cl.precalc(vars)
cl.error_debug(vars, -1)                       # residual of the seed, this energy
cl._iterate(vars, cl.grads, cl.etas, writer)

last = cl.niter - 1
if last % cl.error_step or cl.error_step == -1:
    cl.error_step = 1                          # force; the loop is over
    cl.error_debug(vars, last)
if last % cl.checkpoint_step or cl.checkpoint_step == -1:
    writer.write_checkpoint(vars, last, cl.norm_const, pos_init=cl.pos_init)

cl.postcalc(vars)
logger.info("Done")
