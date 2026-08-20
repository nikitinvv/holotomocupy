#!/usr/bin/env python
"""
Multi-distance holotomography performance benchmark on fully-synthetic data.

Generates everything in-process (no Reader / h5 I/O): probe per distance,
3-D tomographic object slab, rotation angles, positions. Forward-models
sqrt(intensity) + sqrt(reference) via Rec.gen_sqrt_data / Rec.cl_prb_term.gen_sqrt_ref,
then runs BH for `--niter` iterations and prints a timing summary.

Launch with:
    mpirun -n <N> python rec_iterative_mpi_syn.py --n 1024 --ntheta 64 --nchunk 4
"""

import argparse
import logging
import time
import numpy as np
import cupy as cp
from types import SimpleNamespace
from mpi4py import MPI

from holotomocupy.rec_mpi       import Rec
from holotomocupy.logger_config import logger, set_log_level, add_file_handler

cp.cuda.set_pinned_memory_allocator(None)


# ── CLI ─────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument('--n',      type=int, required=True, help='detector size (square: nz=n=n)')
ap.add_argument('--ntheta', type=int, required=True, help='number of projection angles')
ap.add_argument('--nchunk', type=int, default=4,     help='theta chunk size for batched ops (default 4)')
ap.add_argument('--ndistchunk', type=int, default=0,
                help='distances sharing one upload of a theta chunk of proj '
                     '(0 = all ndist, the default; 1 = the old outer-distance loop)')
ap.add_argument('--log',    type=str, default='perf.log',
                help='log file path (rank 0; pass an empty string to disable file logging)')
args_cli = ap.parse_args()

n       = args_cli.n
ntheta  = args_cli.ntheta
nchunk  = args_cli.nchunk
ndistchunk = args_cli.ndistchunk
log_path = args_cli.log


# ── Fixed perf-test config (mirrors config1.conf style) ─────────────────────
nz              = n
nobj            = 3264*n//2048                       # small padding around detector for tomo
nzobj           = nobj                          # cubic object volume
ndist           = 4
# 0 on the command line means "all of them"; resolve it here so the plan, the
# log header and Rec all quote the same number.
ndistchunk = min(ndistchunk, ndist) if ndistchunk > 0 else ndist
niter           = 1                             # short — measures steady-state per-iter cost
obj_dtype       = 'complex64'
rho             = [1, 0.05, 0.02]
mask            = 1.1
lam_prbfit      = 1e-2                           # disable probe-fit regularization
lam_laplacian   = 0
start_iter      = 0
checkpoint_step = -1                            # no disk I/O
error_step      = 1                            # no cost computation in hot loop
log_level       = 'DEBUG'
shift_type ='cubic'

# Physics (brain-Y350 style; values are illustrative — perf timing doesn't depend on them)
energy                  = 17.1
detector_pixelsize      = 1.4760147601476e-6 * n / 4096
focustodetectordistance = 1.217
z1                      = np.linspace(5.10e-3, np.pi/2*5.10e-3, ndist)


# ── MPI ─────────────────────────────────────────────────────────────────────
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
set_log_level(log_level)

# Tee logger output to a file on rank 0 so we can parse timings/memory later.
if rank == 0 and log_path:
    add_file_handler(log_path)
    logger.warning(f"writing log to {log_path}")


# ── Machine info (rank 0 only — written to the perf log for reproducibility) ─


def _log_machine_info():
    """Log CPU model, total RAM, GPU model + memory, and GPU count."""
    # CPU model
    cpu_model = 'unknown'
    try:
        with open('/proc/cpuinfo') as fh:
            for line in fh:
                if line.startswith('model name'):
                    cpu_model = line.split(':', 1)[1].strip()
                    break
    except Exception:
        pass

    # CPU count + total RAM (psutil — already a runtime dep of holotomocupy.utils)
    cpu_count = ram_gb = None
    try:
        import psutil
        cpu_count = psutil.cpu_count(logical=True)
        ram_gb    = psutil.virtual_memory().total / 1024**3
    except Exception:
        pass

    # GPU model, memory, and how many devices CUDA sees
    gpu_name = gpu_mem_gb = gpu_count = dev_id = None
    try:
        gpu_count = cp.cuda.runtime.getDeviceCount()
        dev_id    = cp.cuda.Device().id
        props     = cp.cuda.runtime.getDeviceProperties(dev_id)
        raw_name  = props.get('name', b'')
        gpu_name  = raw_name.decode() if isinstance(raw_name, (bytes, bytearray)) else str(raw_name)
        gpu_mem_gb = props.get('totalGlobalMem', 0) / 1024**3
    except Exception:
        pass

    logger.warning(f"machine: CPU={cpu_model} ({cpu_count} logical cores)")
    if ram_gb is not None:
        logger.warning(f"machine: RAM total={ram_gb:.1f} GB")
    if gpu_name is not None:
        logger.warning(f"machine: GPU={gpu_name}  memory={gpu_mem_gb:.1f} GB  "
                       f"count_visible={gpu_count}  (this rank uses dev {dev_id})")

# Gather (hostname, physical_device_id) from every rank so rank 0 can count
# actual GPUs in use. Note: when CUDA_VISIBLE_DEVICES pins one GPU per rank
# (the usual bind.sh layout) cp.cuda.Device().id always returns 0 because the
# pinned GPU is the only visible one — so we derive the *physical* device id
# from CUDA_VISIBLE_DEVICES instead.
import os as _os
def _physical_dev_id():
    cvd = _os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    local_id = cp.cuda.Device().id if cp.cuda.is_available() else -1
    if not cvd:
        return local_id
    parts = [p.strip() for p in cvd.split(',') if p.strip()]
    if 0 <= local_id < len(parts):
        try:    return int(parts[local_id])
        except ValueError: return parts[local_id]
    return local_id

rank_devices = comm.gather((MPI.Get_processor_name(), _physical_dev_id()), root=0)

if rank == 0:
    _log_machine_info()
    # Distinct (host, device) pairs == distinct physical GPUs across the job.
    unique_gpus = sorted(set(rank_devices))
    from collections import defaultdict
    per_host_ranks = defaultdict(int)
    per_host_gpus  = defaultdict(set)
    for h, d in rank_devices:
        per_host_ranks[h] += 1
        per_host_gpus[h].add(d)
    logger.warning(f"job: ranks={size} hosts={len(per_host_ranks)} gpus_used={len(unique_gpus)}")
    for h in sorted(per_host_ranks):
        devs = sorted(per_host_gpus[h])
        logger.warning(f"job: host={h}  ranks={per_host_ranks[h]}  "
                       f"gpus_used={len(devs)}  devs={devs}")
    for i, (h, d) in enumerate(rank_devices):
        logger.warning(f"job: rank={i}  host={h}  dev={d}")


# ── Assemble Rec args ───────────────────────────────────────────────────────
args = SimpleNamespace(
    # sizes
    nz=nz, n=n, nzobj=nzobj, nobj=nobj,
    ntheta=ntheta, ndist=ndist,
    nchunk=nchunk, ndistchunk=ndistchunk, niter=niter, start_iter=start_iter,
    # dtypes / regs
    obj_dtype=obj_dtype, rho=rho,
    lam_prbfit=lam_prbfit, lam_laplacian=lam_laplacian,
    # logging / I/O
    checkpoint_step=checkpoint_step, error_step=error_step,
    # physics (Rec also expects theta, mask, z1, ...)
    energy=energy,
    focustodetectordistance=focustodetectordistance,
    z1=z1,
    detector_pixelsize=detector_pixelsize,
    theta=np.linspace(0, np.pi, ntheta, endpoint=False).astype('float32'),
    mask=mask,
    shift_type=shift_type,
    # MPI
    comm=comm,
)


# ── Build the reconstruction class ──────────────────────────────────────────
if rank == 0:
    logger.warning(f"perf-test: n={n} nz={nz} nobj={nobj} nzobj={nzobj} "
                   f"ntheta={ntheta} ndist={ndist} nchunk={nchunk} "
                   f"ndistchunk={ndistchunk} niter={niter} "
                   f"nranks={size} obj_dtype={obj_dtype}")
cl = Rec(args)


# ── Synthetic inputs ────────────────────────────────────────────────────────
# Reproducibility model:
#   * single MASTER_SEED — same on every rank, every machine
#   * SeedSequence.spawn() produces three top-level streams (prb / obj / pos)
#   * each is then spawned into one independent sub-seed per GLOBAL index
#     (distance / z-slice / theta) so the synthetic data depends only on the
#     global index, NOT on how MPI happens to split the work
#   * random numbers are drawn on the CPU via numpy.default_rng — that's
#     bit-stable across NumPy versions and machines (cp.random would not be)
#   * no FFTs are used in the synthesis — everything is direct random fill
MASTER_SEED = 20260525
_ss_root                  = np.random.SeedSequence(MASTER_SEED)
_ss_prb, _ss_obj, _ss_pos = _ss_root.spawn(3)


# Probe per distance: Gaussian-envelope amplitude × small random phase.
# No FFTs — direct random fill, scaled to keep |prb| ~ 1.
def synth_prb(nz, n, ndist, ss):
    v  = np.linspace(-0.5, 0.5, n,  endpoint=False, dtype='float32')
    u  = np.linspace(-0.5, 0.5, nz, endpoint=False, dtype='float32')
    vy, vx = np.meshgrid(u, v, indexing='ij')
    amp = np.exp(-2 * (vx**2 + vy**2)).astype('float32')
    out = np.empty((ndist, nz, n), dtype='complex64')
    seeds_per_dist = ss.spawn(ndist)
    for j, sseed in enumerate(seeds_per_dist):
        ph = 0.1 * np.random.default_rng(sseed).standard_normal((nz, n), dtype='float32')
        p  = (amp * np.exp(1j * ph)).astype('complex64')
        p /= np.mean(np.abs(p))                              # |prb| ~ 1
        out[j] = p
    return out


# Object slab [st:end) of the GLOBAL nzobj × nobj × nobj volume.
# Plain random fill (no FFT smoothing) — fast, and bit-reproducible because
# each global slice is drawn from its own seed spawned off MASTER_SEED.
def synth_obj(st, end, nzobj_global, nobj, ss, dtype):
    out_dtype = 'complex64' if dtype == 'complex64' else 'float32'
    out = np.empty((end - st, nobj, nobj), dtype=out_dtype)
    if end - st == 0:
        return out

    slice_seeds = ss.spawn(nzobj_global)[st:end]
    is_cplx     = (dtype == 'complex64')
    scale       = np.float32(1e-3)

    for k, sseed in enumerate(slice_seeds):
        rng_k = np.random.default_rng(sseed)
        re = rng_k.standard_normal((nobj, nobj), dtype='float32') * scale
        if is_cplx:
            im = rng_k.standard_normal((nobj, nobj), dtype='float32') * (scale / np.float32(30))
            out[k] = (-re + 1j * im).astype('complex64')
        else:
            out[k] = -re
    return out


# Positions for global theta range [st:end). Dist-major layout to match Rec.
def synth_pos(st, end, ntheta_global, ndist, ss):
    if end - st == 0:
        return np.empty((ndist, 0, 2), dtype='float32')
    theta_seeds = ss.spawn(ntheta_global)[st:end]
    out = np.empty((ndist, end - st, 2), dtype='float32')
    for j, sseed in enumerate(theta_seeds):
        out[:, j] = (10.0 * (np.random.default_rng(sseed).random((ndist, 2), dtype='float32') - 0.5))
    return out


cl.vars['prb'][:] = synth_prb(nz, n, ndist, _ss_prb)   # vars['prb'] is pinned numpy
cl.vars['obj'][:] = synth_obj(cl.st_obj,   cl.end_obj,   nzobj,  nobj, _ss_obj, obj_dtype)
cl.vars['pos'][:] = synth_pos(cl.st_theta, cl.end_theta, ntheta, ndist, _ss_pos)


# ── Forward-model synthetic data + reference ────────────────────────────────
comm.Barrier()
cl.gen_sqrt_data(cl.vars, cl.data)
cl.cl_prb_term.gen_sqrt_ref(cl.vars['prb'], cl.ref)


# ── Initial guess: zero obj, unit prb, slightly-perturbed pos ──────────────
cl.vars['obj'][:] = 0
cl.vars['prb'][:] = 1
# Keep cl.vars['pos'] as the synthetic positions (matches the data) — Rec recovers from them as init


# ── Time BH ────────────────────────────────────────────────────────────────
comm.Barrier()
cp.cuda.Device().synchronize()
t0 = time.time()

cl.BH()

cp.cuda.Device().synchronize()
comm.Barrier()
elapsed = time.time() - t0


# ── Timing summary ──────────────────────────────────────────────────────────
if rank == 0:
    per_iter = elapsed / max(niter, 1)
    logger.warning(f"BH done: {niter} iters in {elapsed:.3f} s  "
                   f"[nranks={size}, n={n}, nobj={nobj}, ntheta={ntheta}, ndist={ndist}, nchunk={nchunk}]")


# ── Append the parsed summary to the log ───────────────────────────────────
# The log is now complete, so parse it and write the breakdown at its end --
# one file carries both the raw @timer lines and the summary of the iteration
# that matters (the last one, i.e. after the JIT warmup).
if rank == 0 and log_path:
    for _h in logger.handlers:
        _h.flush()
    try:
        import sys as _sys
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from parse_perf_log import report
        with open(log_path, 'a') as _fh:
            _fh.write('\n')
            report(log_path, max(niter - 1, 0), out=_fh,
                   show_info=False, show_header=True)
    except Exception as e:                      # never let this kill a finished run
        logger.warning(f"could not append the parsed summary: {e}")
