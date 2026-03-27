"""
test_redist.py — timing test for MPIClass.redist

Run with:
    mpirun -np <nranks> python test_redist.py
"""

import time
import numpy as np
from mpi4py import MPI


def get_local_chunk(total_size, rank, size):
    chunk_size = total_size // size
    remainder = total_size % size
    if rank < remainder:
        start_idx = rank * (chunk_size + 1)
        end_idx = start_idx + chunk_size + 1
    else:
        start_idx = remainder * (chunk_size + 1) + (rank - remainder) * chunk_size
        end_idx = start_idx + chunk_size
    return start_idx, end_idx


class MPIClass:
    def __init__(self, comm, n_src, n_dst, nz, dtype):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.n_src = int(n_src)
        self.n_dst = int(n_dst)
        self.nz = int(nz)
        self.dtype = np.dtype(dtype)

        self.st_src, self.end_src = get_local_chunk(self.n_src, self.rank, self.size)
        self.st_dst, self.end_dst = get_local_chunk(self.n_dst, self.rank, self.size)
        self.local_n_src = self.end_src - self.st_src
        self.local_n_dst = self.end_dst - self.st_dst

        self._chunks_dst = [get_local_chunk(self.n_dst, p, self.size) for p in range(self.size)]
        self._chunks_src = [get_local_chunk(self.n_src, p, self.size) for p in range(self.size)]

        base = MPI._typedict[self.dtype.char]
        self._src_slab_types = [
            base.Create_subarray(
                [self.n_dst, self.local_n_src, self.nz],
                [de - ds, self.local_n_src, self.nz],
                [ds, 0, 0],
            ).Commit()
            for ds, de in self._chunks_dst
        ]
        self._dst_slab_types = [
            base.Create_subarray(
                [self.local_n_dst, self.n_src, self.nz],
                [self.local_n_dst, se - ss, self.nz],
                [0, ss, 0],
            ).Commit()
            for ss, se in self._chunks_src
        ]
        self._ones  = [1] * self.size
        self._zeros = [0] * self.size

    def redist(self, src, dst, direction="forward"):
        if direction == "forward":
            self.comm.Alltoallw(
                (src, self._ones, self._zeros, self._src_slab_types),
                (dst, self._ones, self._zeros, self._dst_slab_types),
            )
        elif direction == "backward":
            self.comm.Alltoallw(
                (src, self._ones, self._zeros, self._dst_slab_types),
                (dst, self._ones, self._zeros, self._src_slab_types),
            )
        else:
            raise ValueError("direction must be 'forward' or 'backward'")

    def close(self):
        for t in self._src_slab_types + self._dst_slab_types:
            t.Free()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ── Parameters ───────────────────────────────────────────────────────────────
n_src  = 1024
n_dst  = 1500
nz     = 1024
dtype  = np.float32
NREP   = 3
WARM   = 1

# ── Setup ────────────────────────────────────────────────────────────────────
cl = MPIClass(comm, n_src, n_dst, nz, dtype)

st_src, end_src = get_local_chunk(n_src, rank, size)
st_dst, end_dst = get_local_chunk(n_dst, rank, size)
local_n_src = end_src - st_src
local_n_dst = end_dst - st_dst

src_fwd = np.zeros((n_dst, local_n_src, nz), dtype=dtype)
dst_fwd = np.zeros((local_n_dst, n_src, nz), dtype=dtype)
src_bwd = np.zeros((local_n_dst, n_src, nz), dtype=dtype)
dst_bwd = np.zeros((n_dst, local_n_src, nz), dtype=dtype)

def time_redist(src, dst, direction, nrep=NREP, warm=WARM):
    """Return (max_ms, bw_GBps) across ranks."""
    for _ in range(warm):
        cl.redist(src, dst, direction=direction)
    comm.Barrier()
    t0 = time.perf_counter()
    for _ in range(nrep):
        cl.redist(src, dst, direction=direction)
    comm.Barrier()
    elapsed_ms = (time.perf_counter() - t0) / nrep * 1e3
    max_ms = float(comm.allreduce(elapsed_ms, op=MPI.MAX))
    total_bytes = float(comm.allreduce(src.nbytes + dst.nbytes, op=MPI.SUM))
    bw_GBps = total_bytes / (max_ms * 1e-3) / 1e9
    return max_ms, bw_GBps

# ── Timing ───────────────────────────────────────────────────────────────────
t_fwd, bw_fwd = time_redist(src_fwd, dst_fwd, direction="forward")
t_bwd, bw_bwd = time_redist(src_bwd, dst_bwd, direction="backward")

if rank == 0:
    print(f"nranks={size}, n_src={n_src}, n_dst={n_dst}, nz={nz}", flush=True)
    print(f"  forward  {t_fwd:.3f} ms/call  {bw_fwd:.2f} GB/s  ({NREP} reps, {WARM} warm-up)", flush=True)
    print(f"  backward {t_bwd:.3f} ms/call  {bw_bwd:.2f} GB/s  ({NREP} reps, {WARM} warm-up)", flush=True)
    
cl.close()
