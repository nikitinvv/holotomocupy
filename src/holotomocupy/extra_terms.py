"""Regularization terms used by Rec: 3-D biharmonic (Laplacian) and probe-fit."""

import numpy as np
import cupy as cp
from mpi4py import MPI

from .utils import lap, redot, reprod, timer, make_pinned


def biharm(pad_chunk):
    """(∇²)² over the owned slices of a chunk carrying 2 ghost rows per z-side.

    Returns an array shaped like pad_chunk[2:-2]: the inner Laplacians consume
    one ghost row, the outer one the second.
    """
    lap_zm1 = lap(pad_chunk[:-4], pad_chunk[1:-3], pad_chunk[2:-2])
    lap_z   = lap(pad_chunk[1:-3], pad_chunk[2:-2], pad_chunk[3:-1])
    lap_zp1 = lap(pad_chunk[2:-2], pad_chunk[3:-1], pad_chunk[4:])
    return lap(lap_zm1, lap_z, lap_zp1)


class LaplacianTerm:
    """3-D biharmonic regularization: (lam / obj_size) * ||∇²u||² where u = vars['obj'].

    Owns the padded scratch buffers u_pad / e_pad / g_pad of shape
    [local_nzobj+4, nobj, nobj] whose middle [2:-2] slices (exposed as `obj_view` /
    `etas_view` / `grads_view`) alias vars['obj'], etas['obj'] and grads['obj'].
    The 2 ghost rows on each z-side let a chunk compute (∇²)² in a single padded
    pass; g_pad exists so the *gradient* direction can be differentiated too,
    which is what lets hessian3 return all three bilinear forms at once. When
    `lam == 0` the term is inactive and no padding is allocated; the views
    return None so the caller can allocate plain obj-shape buffers instead.

    Halo cost over plain obj-shape buffers: 4 z-slabs per padded array.
    """

    def __init__(self, lam, obj_size, local_nzobj, nobj, obj_dtype, cl_mpi, gpu_batch,
                 grad_pad=True):
        self.lam       = lam
        self.obj_size  = obj_size
        self.cl_mpi    = cl_mpi
        self.gpu_batch = gpu_batch
        self.u_pad = self.e_pad = self.g_pad = None
        if lam != 0:
            shape = [local_nzobj + 4, nobj, nobj]
            self.u_pad = make_pinned(shape, dtype=obj_dtype); self.u_pad[:] = 0
            self.e_pad = make_pinned(shape, dtype=obj_dtype); self.e_pad[:] = 0
            # grads/etas are reconstruction-only; generation (alloc_mode='gen')
            # never touches them, so skip the buffer there.
            if grad_pad:
                self.g_pad = make_pinned(shape, dtype=obj_dtype); self.g_pad[:] = 0

    @property
    def obj_view(self):
        """Storage for vars['obj'] (view into u_pad); None when this term is inactive."""
        return None if self.u_pad is None else self.u_pad[2:-2]

    @property
    def etas_view(self):
        """Storage for etas['obj'] (view into e_pad); None when this term is inactive."""
        return None if self.e_pad is None else self.e_pad[2:-2]

    @property
    def grads_view(self):
        """Storage for grads['obj'] (view into g_pad); None when inactive."""
        return None if self.g_pad is None else self.g_pad[2:-2]

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
            g[:] = g_in + scale * biharm(u_pad_chunk)

        _biharm_grad(self, grad_obj, self.u_pad, grad_obj)

    @timer
    def hessian(self, dobj1):
        """2*lam/obj_size * Re<dobj1, (∇²)²e> over the LOCAL obj slab.

        Local, like energy_local and PrbfitTerm.hessian: Rec.hessian sums the
        three terms and its caller allreduces once. (This used to allreduce
        internally, which made the regularization term count comm.size times
        in the total — latent, since lam_laplacian is 0 in every config.)"""
        if self.lam == 0:
            return 0
        scale = np.float32(2.0 * self.lam / self.obj_size)
        self.exchange_ghosts(self.e_pad)
        acc = cp.zeros(1, dtype='float32')

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1, inp_pad=4)
        def _biharm_dot(self, acc, e_pad_chunk, d1):
            acc[:] += redot(d1, biharm(e_pad_chunk))

        _biharm_dot(self, acc, self.e_pad, dobj1)
        return float(scale * float(acc[0]))

    @timer
    def hessian3(self, *_):
        """The three regularization bilinear forms {B(g,g), B(g,e), B(e,e)} from
        ONE pass — the Laplacian counterpart of Rec.hessian3.

        Takes no directions: g and e are g_pad[2:-2] and e_pad[2:-2], so the pass
        streams the two padded slabs and reads both the biharmonics and the
        contraction vectors out of them. (Any positional arguments are ignored,
        which keeps the call site uniform with the other hessian3's.)

        Both fields carry ghost rows, so both biharmonics are available in the
        same chunk and the three redots cost only arithmetic on top. Local, like
        hessian(); the caller allreduces.

        Half the traffic of the two-pass route it replaces, and it lets the
        caller derive `bottom` arithmetically instead of re-measuring it after
        the etas update."""
        if self.lam == 0:
            return 0.0, 0.0, 0.0
        scale = np.float32(2.0 * self.lam / self.obj_size)
        # The 3-element accumulator is a non-proper output only because 3 is not
        # the chunked axis length; a degenerate slab would silently make it one.
        assert self.g_pad.shape[0] - 4 != 3, "local_nzobj==3 aliases the accumulator shape"
        self.exchange_ghosts(self.g_pad)
        self.exchange_ghosts(self.e_pad)
        acc = cp.zeros(3, dtype='float32')

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1, inp_pad=4)
        def _biharm_dot3(self, acc, g_pad_chunk, e_pad_chunk):
            bg = biharm(g_pad_chunk)
            be = biharm(e_pad_chunk)
            g  = g_pad_chunk[2:-2]
            acc[0:1] += redot(g, bg)
            acc[1:2] += redot(g, be)
            acc[2:3] += redot(e_pad_chunk[2:-2], be)

        _biharm_dot3(self, acc, self.g_pad, self.e_pad)
        a = acc.get()
        return float(scale * a[0]), float(scale * a[1]), float(scale * a[2])

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
        """Add (lam / prb_size) * 2 * D^T(|D·prb| - ref) * rho_sq_prb to grad_prb in-place.
        grad_prb may be pinned numpy (vars/grads/etas['prb'] are all pinned); the per-j
        contribution is .get()'d to host before accumulating."""
        if self.lam == 0:
            return
        for j in range(self.ndist):
            tmp = self.cl_prop.D(prb[j:j+1], j)
            td  = self.ref[j:j+1] * (tmp / cp.abs(tmp))
            td  = self.lam / self.prb_size * self.cl_prop.DT(2 * (tmp - td), j)
            contrib = (td * rho_sq_prb)
            if isinstance(grad_prb, cp.ndarray):
                grad_prb[j:j+1] += contrib
            else:
                grad_prb[j:j+1] += cp.asnumpy(contrib)

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

    @timer
    def hessian3(self, prb, dg, de):
        """The three probe-fit bilinear forms {B(g,g), B(g,e), B(e,e)}.

        Same contraction as hessian(), but the direction-independent Dprb/l0/d0
        and the two direction propagations are each computed once per distance
        instead of once per pair — 3 D() calls per j instead of 9."""
        if self.lam == 0:
            return 0, 0, 0
        ogg = oge = oee = 0
        for j in range(self.ndist):
            Dprb = self.cl_prop.D(prb[j:j+1], j)
            Dg   = self.cl_prop.D(dg[j:j+1], j)
            De   = self.cl_prop.D(de[j:j+1], j)
            aDprb = cp.abs(Dprb)
            l0 = Dprb / aDprb
            d0 = self.ref[j:j+1] / aDprb
            pg = reprod(l0, Dg)
            pe = reprod(l0, De)
            ogg += 2 * (cp.sum((1 - d0) * reprod(Dg, Dg)) + cp.sum(d0 * pg * pg))
            oge += 2 * (cp.sum((1 - d0) * reprod(Dg, De)) + cp.sum(d0 * pg * pe))
            oee += 2 * (cp.sum((1 - d0) * reprod(De, De)) + cp.sum(d0 * pe * pe))
        s = self.lam / self.prb_size
        return (s * ogg).get(), (s * oge).get(), (s * oee).get()

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
