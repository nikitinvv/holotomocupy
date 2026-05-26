"""Regularization terms used by Rec: 3-D biharmonic (Laplacian) and probe-fit."""

import numpy as np
import cupy as cp
from mpi4py import MPI

from .utils import lap, redot, reprod, timer, make_pinned


class LaplacianTerm:
    """3-D biharmonic regularization: (lam / obj_size) * ||∇²u||² where u = vars['obj'].

    Owns the padded scratch buffers u_pad / e_pad of shape [local_nzobj+4, nobj, nobj]
    whose middle [2:-2] slices (exposed as `obj_view` / `etas_view`) alias vars['obj']
    and etas['obj']. The 2 ghost rows on each z-side let a chunk compute (∇²)² in a
    single padded pass. When `lam == 0` the term is inactive and no padding is
    allocated; obj_view / etas_view return None so the caller can allocate plain
    obj-shape buffers instead.
    """

    def __init__(self, lam, obj_size, local_nzobj, nobj, obj_dtype, cl_mpi, gpu_batch):
        self.lam       = lam
        self.obj_size  = obj_size
        self.cl_mpi    = cl_mpi
        self.gpu_batch = gpu_batch
        if lam != 0:
            self.u_pad = make_pinned([local_nzobj + 4, nobj, nobj], dtype=obj_dtype)
            self.e_pad = make_pinned([local_nzobj + 4, nobj, nobj], dtype=obj_dtype)
            self.u_pad[:] = 0
            self.e_pad[:] = 0
        else:
            self.u_pad = None
            self.e_pad = None

    @property
    def obj_view(self):
        """Storage for vars['obj'] (view into u_pad); None when this term is inactive."""
        return None if self.u_pad is None else self.u_pad[2:-2]

    @property
    def etas_view(self):
        """Storage for etas['obj'] (view into e_pad); None when this term is inactive."""
        return None if self.e_pad is None else self.e_pad[2:-2]

    def exchange_ghosts(self, pad):
        """Fill pad[0:2] / pad[-2:] from neighbouring ranks; zero at the global boundary."""
        rank, size = self.cl_mpi.rank, self.cl_mpi.size
        left  = rank - 1 if rank > 0        else MPI.PROC_NULL
        right = rank + 1 if rank < size - 1 else MPI.PROC_NULL
        self.cl_mpi.comm.Sendrecv(
            sendbuf=np.ascontiguousarray(pad[-4:-2]), dest=right,
            recvbuf=pad[0:2], source=left)
        self.cl_mpi.comm.Sendrecv(
            sendbuf=np.ascontiguousarray(pad[2:4]), dest=left,
            recvbuf=pad[-2:], source=right)
        if left  == MPI.PROC_NULL: pad[0:2] = 0
        if right == MPI.PROC_NULL: pad[-2:] = 0

    @timer
    def gradient(self, grad_obj):
        """Add 2*lam/obj_size * (∇²)²u to grad_obj in-place."""
        if self.lam == 0:
            return
        scale = np.float32(2.0 * self.lam / self.obj_size)
        self.exchange_ghosts(self.u_pad)

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1, inp_pad=4)
        def _biharm_grad(self, g, u_pad_chunk, g_in):
            lap_zm1 = lap(u_pad_chunk[:-4], u_pad_chunk[1:-3], u_pad_chunk[2:-2])
            lap_z   = lap(u_pad_chunk[1:-3], u_pad_chunk[2:-2], u_pad_chunk[3:-1])
            lap_zp1 = lap(u_pad_chunk[2:-2], u_pad_chunk[3:-1], u_pad_chunk[4:])
            g[:] = g_in + scale * lap(lap_zm1, lap_z, lap_zp1)

        _biharm_grad(self, grad_obj, self.u_pad, grad_obj)

    @timer
    def hessian(self, dobj1):
        """2*lam/obj_size * Re<dobj1, (∇²)²e>, allreduced over ranks."""
        if self.lam == 0:
            return 0
        scale = np.float32(2.0 * self.lam / self.obj_size)
        self.exchange_ghosts(self.e_pad)
        acc = cp.zeros(1, dtype='float32')

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1, inp_pad=4)
        def _biharm_dot(self, acc, e_pad_chunk, d1):
            lap_zm1 = lap(e_pad_chunk[:-4], e_pad_chunk[1:-3], e_pad_chunk[2:-2])
            lap_z   = lap(e_pad_chunk[1:-3], e_pad_chunk[2:-2], e_pad_chunk[3:-1])
            lap_zp1 = lap(e_pad_chunk[2:-2], e_pad_chunk[3:-1], e_pad_chunk[4:])
            acc[:] += redot(d1, lap(lap_zm1, lap_z, lap_zp1))

        _biharm_dot(self, acc, self.e_pad, dobj1)
        return float(self.cl_mpi.allreduce(np.array(scale * float(acc[0]), dtype='float32')))

    def energy_local(self):
        """Local biharmonic energy (lam/obj_size) * ||∇²u||². No allreduce."""
        if self.lam == 0:
            return np.float32(0)
        scale = np.float32(self.lam / self.obj_size)
        self.exchange_ghosts(self.u_pad)
        acc = cp.zeros(1, dtype='float32')

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1, inp_pad=4)
        def _biharm_e(self, acc, u_pad_chunk):
            l = lap(u_pad_chunk[1:-3], u_pad_chunk[2:-2], u_pad_chunk[3:-1])
            acc[:] += redot(l, l)

        _biharm_e(self, acc, self.u_pad)
        return scale * float(acc[0])


class PrbfitTerm:
    """Probe-fit regularization: (lam / prb_size) * ||(|D·prb| - ref)||².

    Owns `ref` (shape [ndist, nz, n] float32) — the per-distance reference probe magnitude
    that the regularizer fits against. Allocated unconditionally so external code can
    seed it (via `gen_sqrt_ref` or a reader) regardless of whether the term is active."""

    def __init__(self, lam, prb_size, ndist, nz, n, cl_prop):
        self.lam      = lam
        self.prb_size = prb_size
        self.ndist    = ndist
        self.cl_prop  = cl_prop
        self.ref      = cp.empty([ndist, nz, n], dtype='float32')

    @timer
    def gradient(self, grad_prb, prb, rho_sq_prb):
        """Add (lam / prb_size) * 2 * D^T(|D·prb| - ref) * rho_sq_prb to grad_prb in-place."""
        if self.lam == 0:
            return
        for j in range(self.ndist):
            tmp = self.cl_prop.D(prb[j:j+1], j)
            td  = self.ref[j:j+1] * (tmp / cp.abs(tmp))
            td  = self.lam / self.prb_size * self.cl_prop.DT(2 * (tmp - td), j)
            grad_prb[j:j+1] += td * rho_sq_prb

    @timer
    def hessian(self, prb, dprb1, dprb2):
        """Probe-fit hessian: 2*lam/prb_size * Σ_j [(1-d0)Re<Ddprb1,Ddprb2> + d0 Re<l0,Ddprb1> Re<l0,Ddprb2>]."""
        if self.lam == 0:
            return 0
        out = 0
        for j in range(self.ndist):
            Dprb   = self.cl_prop.D(prb[j:j+1], j)
            Ddprb1 = self.cl_prop.D(dprb1[j:j+1], j)
            Ddprb2 = self.cl_prop.D(dprb2[j:j+1], j)
            l0 = Dprb / cp.abs(Dprb)
            d0 = self.ref[j:j+1] / cp.abs(Dprb)
            v1 = cp.sum((1 - d0) * reprod(Ddprb1, Ddprb2))
            v2 = cp.sum(d0 * reprod(l0, Ddprb1) * reprod(l0, Ddprb2))
            out += 2 * (v1 + v2)
        out = self.lam * out / self.prb_size
        return out.get()

    def energy_local(self, prb):
        """Local probe-fit energy (lam / prb_size) * Σ_j ||(|D·prb_j| - ref_j)||²."""
        if self.lam == 0:
            return 0
        out = 0
        for j in range(self.ndist):
            Dprb = self.cl_prop.D(prb[j:j+1], j)[0]
            out += self.lam / self.prb_size * cp.linalg.norm(cp.abs(Dprb) - self.ref[j]) ** 2
        return out

    def gen_sqrt_ref(self, prb, out):
        """Populate `out` with the synthetic reference: out[j] = |D·prb_j| for each distance.
        Used by tests/perf scripts to seed self.ref so the regularizer has something to fit."""
        for j in range(self.ndist):
            out[j] = cp.abs(self.cl_prop.D(prb[j:j+1], j)[0])
