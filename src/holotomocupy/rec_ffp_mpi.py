import numpy as np
import cupy as cp
import os
import tifffile
import warnings
import pandas as pd
import nvtx
import cupy.fft

from .propagation_far import PropagationFar
from .shift import Shift
from .shift_fft import ShiftFFT
from .chunking import Chunking
from .cuda_kernels import (
    patch_extract_c64_kernel,     patch_extract_f32_kernel,
    patch_scatter_add_c64_kernel, patch_scatter_add_f32_kernel,
)
from .utils import *
from .mpi_functions import *
from .logger_config import logger

np.set_printoptions(legacy="1.25")
warnings.filterwarnings("ignore", message=f".*peer.*")
cupy.fft.config.get_plan_cache().set_size(0)  # dont waste GPU memory


class RecFFP:
    """2-D far-field ptychography reconstruction (MPI-parallel over theta).

    Forward model (per scan position pos_j):
        x0_j = F1( F2( F3(prb, obj, pos_j) ) )
             = D( prb * exp(1j · S_pos_j(obj)) )
    where obj = δ + i·β is the complex projected refractive-index decrement
    (δ = phase, β = absorption) and D is a centered orthogonal 2-D FFT
    (Fraunhofer / far-field).

    Variables: prb (nz×n, complex64), obj (nzobj×nobj, complex64 or float32),
    pos (ntheta×2, float32).
    Parallelisation: theta is distributed across MPI ranks; prb/obj are
    replicated.
    """

    # Guard against |x|→0 in F0-related divisions. Probes with dim outer
    # speckle produce D(prb·exp(iobj)) values close to zero on the detector,
    # which would otherwise give NaNs in dF0 / d2F_dF0 / gF0.
    _ABS_EPS = np.float32(1e-4)

    # F3 patch margin (each side). F3 extracts a
    # (nz + 2·MARGIN) × (n + 2·MARGIN) window around each round(pos), then
    # applies the subpixel shift on that small grid. Bump if aliasing shows
    # up with high-frequency object content.
    MARGIN = 4

    _var_names = ("prb", "obj", "pos")

    def __init__(self, args):

        for key, value in vars(args).items():
            setattr(self, key, value)

        # cascade: F0 ◦ F1 ◦ F2 ◦ F3
        self.F      = [self.F0,      self.F1,      self.F2,      self.F3]
        self.gF     = [self.gF0,     self.gF1,     self.gF2,     self.gF3]
        self.dF     = [self.dF0,     self.dF1,     self.dF2,     self.dF3]
        self.d2F_dF = [self.d2F_dF0, self.d2F_dF1, self.d2F_dF2, self.d2F_dF3]

        multiplier   = 4
        float_item   = np.dtype("float32").itemsize
        complex_item = np.dtype("complex64").itemsize
        # double-buffered data chunks (dominant) + overhead for other proper arrays
        nbytes = int(multiplier * self.nchunk * (self.nz * self.n * float_item + self.nobj * self.nobj * complex_item))

        # MPI: distribute theta; prb/obj replicated on all ranks
        self.cl_mpi       = MPIClass(args.comm, self.nzobj, self.ntheta, self.nobj, args.obj_dtype)
        self.local_ntheta = self.cl_mpi.local_ntheta
        self.rank         = self.cl_mpi.rank
        self.st_theta     = self.cl_mpi.st_theta
        self.end_theta    = self.cl_mpi.end_theta

        if self.rank == 0 and hasattr(self, 'path_out') and self.path_out:
            os.makedirs(os.path.join(self.path_out, 'checkpoints_tiff'), exist_ok=True)
        args.comm.Barrier()

        self.rho_sq = {
            'obj': args.rho[0]**2,
            'prb': args.rho[1]**2,
            'pos': args.rho[2]**2,
        }

        # Patch-grid parameters for the F3 extract-then-shift pipeline.
        self._npad_y = self.nz + 2 * self.MARGIN
        self._npad_x = self.n  + 2 * self.MARGIN
        if self._npad_y > self.nzobj or self._npad_x > self.nobj:
            raise ValueError(f'patch ({self._npad_y}, {self._npad_x}) larger than object '
                             f'({self.nzobj}, {self.nobj}); reduce MARGIN')
        # Base extract offsets: patch center coincides with obj center at ipos=0.
        self._cy = self.nzobj // 2 - self._npad_y // 2
        self._cx = self.nobj  // 2 - self._npad_x // 2

        self.cl_chunking = Chunking(nbytes, self.nchunk)
        self.cl_prop     = PropagationFar(self.n, self.nz, self.nchunk)

        # cl_shift now lives on the SMALL patch grid — F3 does the subpixel
        # shift on (npad_y, npad_x) after extracting the patch around each
        # round(pos). ShiftFFT / Shift share the same constructor signature.
        if self.shift_type == 'fft':
            InnerShift = ShiftFFT
        elif self.shift_type == 'cubic':
            InnerShift = Shift
        else:
            raise ValueError(f"shift_type must be 'cubic' or 'fft', got {self.shift_type!r}")
        self.cl_shift = InnerShift(n=self.n, npsi=self._npad_x,
                                   nz=self.nz, nzpsi=self._npad_y,
                                   obj_dtype=self.obj_dtype,
                                   nchunk=self.nchunk)

        # Full-grid Shift used ONLY for the B-spline coeff prefilter in the
        # cubic case (prefilter is non-local — must run on the whole object,
        # not on the patch). None for shift_type='fft' since coeff is identity.
        if self.shift_type == 'cubic':
            self._coeff_shift = Shift(n=self.nobj, npsi=self.nobj,
                                      nz=self.nzobj, nzpsi=self.nzobj,
                                      obj_dtype=self.obj_dtype)
        else:
            self._coeff_shift = None

        self.alloc_arrays()

        self.table = pd.DataFrame(columns=["iter", "err", "time"])

        self.data_size = self.ntheta * self.nz * self.n
        self.prb_size  = self.nz * self.n

        self.gpu_batch    = self.cl_chunking.gpu_batch
        self.redot_batch  = self.cl_chunking.redot_batch
        self.linear_batch = self.cl_chunking.linear_batch
        self.mulc_batch   = self.cl_chunking.mulc_batch
        self.allreduce    = self.cl_mpi.allreduce
        self.allreduce2   = self.cl_mpi.allreduce2

    def alloc_arrays(self):
        self.vars = {
            'prb': cp.empty([self.nz, self.n],         dtype='complex64'),
            'obj': cp.zeros([self.nzobj, self.nobj],   dtype=self.obj_dtype),
            'pos': cp.zeros([self.local_ntheta, 2],    dtype='float32'),
        }
        self.data = make_pinned([self.local_ntheta, self.nz, self.n], dtype='float32')
        self.grads, self.etas = {}, {}
        for ge in self.grads, self.etas:
            ge['prb'] = cp.zeros([self.nz, self.n],       dtype='complex64')
            ge['obj'] = cp.zeros([self.nzobj, self.nobj], dtype=self.obj_dtype)
            ge['pos'] = cp.zeros([self.local_ntheta, 2],  dtype='float32')

    def BH(self, writer=None):
        vars  = self.vars
        grads = self.grads
        etas  = self.etas

        self.precalc(vars)
        self.error_debug(vars, -1)

        self.time_start = time.time()
        for i in range(self.start_iter, self.niter):
            with nvtx.annotate(f"::BH:ffp:{i}"):
                self.compute_gradient(vars, grads)
                self.compute_beta(vars, grads, etas, i)
                alpha = self.compute_alpha(vars, grads, etas)
                self.apply_step(vars, etas, alpha)
                self.log_iter(vars, i, writer)

        return vars

    def precalc(self, vars):
        """One-time setup at the start of BH: snapshot initial positions."""
        self.pos_init = vars['pos'].copy()

    def compute_gradient(self, vars, grads):
        """Gradients + per-variable rho_sq scaling."""
        with nvtx.annotate("gradients"):
            self.gradients(vars, grads)
        for v in self._var_names:
            self.mulc_batch(grads[v], grads[v], self.rho_sq[v])

    def compute_beta(self, vars, grads, etas, i):
        """Update etas in place: first iter is pure steepest descent (etas = -grads);
        subsequent iters apply etas = beta*etas - grads with the CG coefficient."""
        if i == self.start_iter:
            for v in self._var_names:
                self.mulc_batch(etas[v], grads[v], -1)
            return
        with nvtx.annotate(":::BH:calc beta"):
            top, bottom = self.allreduce2(
                self.hessian(vars, grads, etas),
                self.hessian(vars, etas,  etas),
            )
            beta = top / bottom
            for v in self._var_names:
                self.linear_batch(etas[v], grads[v], beta, -1)

    def compute_alpha(self, vars, grads, etas):
        """Step size: alpha = top / bottom with top = -<grad, eta>/rho_sq (probe & obj
        contributions only on rank 0 since they're replicated), bottom = <eta, H·eta>."""
        with nvtx.annotate(":::BH:calc_alpha"):
            top = -self.redot_batch(grads['pos'], etas['pos']) / self.rho_sq['pos']
            if self.rank == 0:
                for v in self._var_names:
                    if v == 'pos':
                        continue
                    top -= self.redot_batch(grads[v], etas[v]) / self.rho_sq[v]
            bottom = self.hessian(vars, etas, etas)
            top, bottom = self.allreduce2(top, bottom)
            return top / bottom

    def apply_step(self, vars, etas, alpha):
        """var ← var + alpha·eta for every variable."""
        for v in self._var_names:
            self.linear_batch(vars[v], etas[v], 1, alpha)

    def log_iter(self, vars, i, writer):
        """Error + checkpoint hooks for this iter."""
        with nvtx.annotate(":::BH:calc error", color='gray'):
            self.error_debug(vars, i)
        with nvtx.annotate(":::BH:vis_debug", color='gray'):
            self.vis_debug(vars, i, writer)

    def hessian(self, vars, grads, etas):
        return self.hessian_cascade(vars, grads, etas)

    @timer
    def hessian_cascade(self, vars, grads, etas):
        out = cp.zeros(1, dtype="float32")

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _hessian_cascade(
            self, out, d,
            x2, y2, z2,   # pos  — proper (theta-distributed)
            x0, y0, z0,   # prb  — non-proper gpu
            x1, y1, z1,   # obj  — non-proper gpu
        ):
            self._coeff_cache_reset()
            x = [x0, x1, x2]
            y = [y0, y1, y2]
            z = [z0, z1, z2]
            w = [None, None, None]
            y_is_z = y[0] is z[0]

            for id in range(1, len(self.F))[::-1]:
                w = self.d2F_dF[id](x, y, z, w)
                fx, y = self.dF[id](x, y)
                if y_is_z:
                    z = y
                else:
                    z = self.dF[id](x, z, return_x=False)
                x = fx

            out[:] += self.d2F_dF[0](x, y, z, w, d)

        _hessian_cascade(
            self, out, self.data,
            vars["pos"], grads["pos"], etas["pos"],
            vars["prb"], grads["prb"], etas["prb"],
            vars["obj"], grads["obj"], etas["obj"],
        )
        return out[0].get()

    def gradients(self, vars, grads):
        self.gradients_cascade(vars, grads)
        grads['prb'][:] = cp.array(self.allreduce(grads['prb'].get()))
        grads['obj'][:] = cp.array(self.allreduce(grads['obj'].get()))

    @timer
    def gradients_cascade(self, vars, grads):
        grads['prb'][:] = 0
        grads['obj'][:] = 0

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=3)
        def _gradients_cascade(self, gradpos, gradprb, gradobj, d, pos, prb, obj):
            self._coeff_cache_reset()
            x = [prb, obj, pos]
            y = d
            for id in range(len(self.gF)):
                y = self.gF[id](x, y)
            gradprb[:] += y[0]
            gradobj[:] += y[1]
            gradpos[:]  = y[2]

        _gradients_cascade(
            self, grads['pos'], grads['prb'], grads['obj'],
            self.data, vars['pos'], vars['prb'], vars['obj'],
        )

    ####################### Cascade functions #######################
    # Variables: x = [prb, obj, pos]
    # F3: (prb, obj, pos) → (prb, S_pos(obj))
    # F2: (prb, shifted_obj) → (prb, exp(1j · shifted_obj))
    # F1: (prb, exp_obj) → D(prb · exp_obj)          [D = far-field FFT]
    # F0: ||·| - d||²
    #################################################################

    def apply_F_from(self, x, from_level):
        """Apply F[from_level], F[from_level+1], ..., F[len(F)-1] inside-out."""
        for k in range(from_level, len(self.F))[::-1]:
            x = self.F[k](x)
        return x

    ####### F0: ||x0| - d||² / data_size
    @staticmethod
    @cp.fuse()
    def _F0_fused(x, d):
        t = cp.abs(x) - d
        return t * t

    def F0(self, x, d):
        return 1 / self.data_size * cp.sum(self._F0_fused(x, d))

    @staticmethod
    @cp.fuse()
    def _dF0_fused(x, d, eps):
        return x - d * (x / (cp.abs(x) + eps))

    def dF0(self, x, y, d, return_x=False):
        return 2 / self.data_size * redot(self._dF0_fused(x, d, self._ABS_EPS), y)

    @staticmethod
    @cp.fuse()
    def _d2F_dF0_fused(x, y, z, w, d, eps):
        absval = cp.abs(x) + eps
        l0 = x / absval
        d0 = d / absval
        v = (1 - d0) * reprod(y, z) + d0 * reprod(l0, y) * reprod(l0, z)
        if w is not None:
            v += reprod(x - d * l0, w)
        return v

    def d2F_dF0(self, x, y, z, w, d):
        return 2 / self.data_size * cp.sum(self._d2F_dF0_fused(x, y, z, w, d, self._ABS_EPS))

    @staticmethod
    @cp.fuse()
    def _gF0_fused(x, y, scale, eps):
        td = y * (x / (cp.abs(x) + eps))
        return scale * (x - td)

    def gF0(self, x, y):
        x = self.apply_F_from(x, 1)
        return self._gF0_fused(x, y, np.float32(2 / self.data_size), self._ABS_EPS)

    ####### F1: (prb, exp_obj) → D(prb · exp_obj)
    def F1(self, x):
        x11, x12 = x
        return self.cl_prop.D(x11 * x12, 0)

    def dF1(self, x, y, return_x=True):
        x11, x12 = x
        y11, y12 = y
        y0 = self.cl_prop.D(y11 * x12 + x11 * y12, 0)
        if return_x:
            return self.cl_prop.D(x11 * x12, 0), y0
        return y0

    def d2F_dF1(self, x, y, z, w):
        x11, x12 = x
        y11, y12 = y
        z11, z12 = z
        w11, w12 = w
        if y12 is z12:
            y0 = 2 * y11 * y12
        else:
            y0 = y11 * z12 + z11 * y12
        if w11 is not None:
            y0 = y0 + w11 * x12
        if w12 is not None:
            y0 = y0 + x11 * w12
        return self.cl_prop.D(y0, 0)

    def gF1(self, x, y):
        y0 = y
        x = self.apply_F_from(x, 2)
        x11, x12 = x
        y12 = self.cl_prop.DT(y0, 0)
        y11 = cp.sum(y12 * cp.conj(x12), axis=0)  # sum over theta → (nz, n)
        y12 = y12 * cp.conj(x11)
        return y11, y12

    ####### F2: (prb, shifted_obj) → (prb, exp(1j · shifted_obj))
    @staticmethod
    @cp.fuse()
    def _F2_fused(x22):
        return cp.exp(1j * x22)

    def F2(self, x):
        x21, x22 = x
        return x21, self._F2_fused(x22)

    @staticmethod
    @cp.fuse()
    def _dF2_fused(x22, y22):
        e = cp.exp(1j * x22)
        return e, e * 1j * y22

    def dF2(self, x, y, return_x=True):
        x21, x22 = x
        y21, y22 = y
        x12, y12 = self._dF2_fused(x22, y22)
        return ([x21, x12], [y21, y12]) if return_x else [y21, y12]

    @staticmethod
    @cp.fuse()
    def _d2F_dF2_fused(x22, y22, z22, w22):
        e = cp.exp(1j * x22)
        r = e * (-y22 * z22)
        if w22 is not None:
            r = r + e * 1j * w22
        return r

    def d2F_dF2(self, x, y, z, w):
        x21, x22 = x
        y21, y22 = y
        z21, z22 = z
        w21, w22 = w
        return [w21, self._d2F_dF2_fused(x22, y22, z22, w22)]

    @staticmethod
    @cp.fuse()
    def _gF2_fused(x22, y12):
        return (-1j) * y12 * cp.conj(cp.exp(1j * x22))

    def gF2(self, x, y):
        y11, y12 = y
        x = self.apply_F_from(x, 3)
        x21, x22 = x
        y22 = self._gF2_fused(x22, y12)
        y22 = y22.real if self.obj_dtype == 'float32' else y22
        return [y11, y22]

    ####### F3: (prb, obj, pos) → (prb, S_pos(obj))
    #
    # Patch-based: split pos into ipos = round(pos) + fpos = pos − ipos, then
    #     patch_t = obj_coeff[cy − ipos_y[t] : cy − ipos_y[t] + npad_y,
    #                         cx − ipos_x[t] : cx − ipos_x[t] + npad_x]
    # and the subpixel shift by fpos runs on the (npad_y, npad_x) patch via
    # cl_shift.curlySc — much cheaper than shifting the full (nzobj, nobj)
    # grid when nobj ≫ n.
    #
    # Position derivatives flow through fpos only (d(round)/d(pos) = 0), so
    # dF3 / d2F_dF3 pass y33 / z33 straight to the small-grid derivatives.
    # gF3's dcurlySadjc returns a per-theta (npad_y, npad_x) patch adjoint
    # that we scatter-add back into a (nzobj, nobj) coeff-space gradient,
    # then apply the (self-adjoint) coeff prefilter once — identity for
    # shift_type='fft', B-spline prefilter for shift_type='cubic'.
    #
    # coeff(x32) is cached on id(x32); callers (gradients_cascade /
    # hessian_cascade closures) MUST invoke self._coeff_cache_reset() at
    # chunk boundaries since id() is recycled once earlier arrays are GC'd.

    def _coeff_cached(self, psi):
        """coeff(psi) via the FULL-grid Shift (identity for FFT)."""
        if self._coeff_shift is None:
            return psi
        return self._coeff_shift.coeff_cached(psi)

    def _coeff_cache_reset(self):
        if self._coeff_shift is not None:
            self._coeff_shift.coeff_cache_reset()

    def _coeff_apply(self, psi):
        """Explicit (non-cached) coeff apply — used in gF3 to apply the
        prefilter adjoint on the accumulated obj-space gradient."""
        if self._coeff_shift is None:
            return psi
        return self._coeff_shift.coeff(psi)

    @staticmethod
    def _split_pos(pos):
        ipos = cp.round(pos).astype(cp.int32)
        fpos = (pos - ipos.astype(pos.dtype)).astype('float32')
        return ipos, fpos

    def _extract(self, obj, ipos):
        ntheta = ipos.shape[0]
        obj    = cp.ascontiguousarray(obj)
        patches = cp.empty((ntheta, self._npad_y, self._npad_x), dtype=obj.dtype)
        ipy = cp.ascontiguousarray(ipos[:, 0].astype(cp.int32))
        ipx = cp.ascontiguousarray(ipos[:, 1].astype(cp.int32))
        block = (16, 16, 1)
        grid  = ((self._npad_x + 15) // 16, (self._npad_y + 15) // 16, ntheta)
        fn = patch_extract_c64_kernel if obj.dtype == cp.complex64 else patch_extract_f32_kernel
        fn(grid, block,
           (obj, ipy, ipx, patches,
            cp.int32(self.nzobj), cp.int32(self.nobj),
            cp.int32(self._npad_y), cp.int32(self._npad_x), cp.int32(ntheta),
            cp.int32(self._cy), cp.int32(self._cx)))
        return patches

    def _scatter_add(self, out_obj, patches, ipos):
        ntheta = ipos.shape[0]
        out_obj = cp.ascontiguousarray(out_obj)
        patches = cp.ascontiguousarray(patches)
        ipy = cp.ascontiguousarray(ipos[:, 0].astype(cp.int32))
        ipx = cp.ascontiguousarray(ipos[:, 1].astype(cp.int32))
        block = (16, 16, 1)
        grid  = ((self._npad_x + 15) // 16, (self._npad_y + 15) // 16, ntheta)
        fn = patch_scatter_add_c64_kernel if out_obj.dtype == cp.complex64 else patch_scatter_add_f32_kernel
        fn(grid, block,
           (out_obj, ipy, ipx, patches,
            cp.int32(self.nzobj), cp.int32(self.nobj),
            cp.int32(self._npad_y), cp.int32(self._npad_x), cp.int32(ntheta),
            cp.int32(self._cy), cp.int32(self._cx)))

    def F3(self, x):
        x31, x32, x33 = x
        n = len(x33)
        coeff       = self._coeff_cached(x32)
        ipos, fpos  = self._split_pos(x33)
        patches     = self._extract(coeff, ipos)
        m = cp.ones([n, 2], dtype='float32')
        return x31, self.cl_shift.curlySc(patches, fpos, m)

    def dF3(self, x, y, return_x=True):
        x31, x32, x33 = x
        y31, y32, y33 = y
        n = len(x33)
        coeff       = self._coeff_cached(x32)
        coeff1      = self._coeff_cached(y32)
        ipos, fpos  = self._split_pos(x33)
        patches     = self._extract(coeff,  ipos)
        patches1    = self._extract(coeff1, ipos)
        m = cp.ones([n, 2], dtype='float32')
        y22 = self.cl_shift.dcurlySc(patches, fpos, m, patches1, y33)
        if return_x:
            x22 = self.cl_shift.curlySc(patches, fpos, m)
            return [x31, x22], [y31, y22]
        return [y31, y22]

    def d2F_dF3(self, x, y, z, w):
        x31, x32, x33 = x
        y31, y32, y33 = y
        z31, z32, z33 = z
        w31, w32, w33 = w
        n = len(x33)
        coeff       = self._coeff_cached(x32)
        coeff_y     = self._coeff_cached(y32)
        coeff_z     = self._coeff_cached(z32)
        ipos, fpos  = self._split_pos(x33)
        patches     = self._extract(coeff,   ipos)
        patches_y   = self._extract(coeff_y, ipos)
        patches_z   = self._extract(coeff_z, ipos)
        m = cp.ones([n, 2], dtype='float32')
        y22 = self.cl_shift.d2curlySc(patches, fpos, m,
                                      patches_y, y33, patches_z, z33)
        if w32 is not None:
            coeff_w   = self._coeff_cached(w32)
            patches_w = self._extract(coeff_w, ipos)
            y22 = y22 + self.cl_shift.dcurlySc(patches, fpos, m, patches_w, w33)
        return [w31, y22]

    def gF3(self, x, y):
        y21, y22 = y
        x = self.apply_F_from(x, 4)     # no-op: len(self.F) == 4
        x31, x32, x33 = x
        n = len(x33)
        coeff       = self._coeff_cached(x32)
        ipos, fpos  = self._split_pos(x33)
        patches     = self._extract(coeff, ipos)
        m = cp.ones([n, 2], dtype='float32')
        # Small shift returns (delta_patches [ntheta, npad_y, npad_x], delta_pos).
        delta_patches, y33 = self.cl_shift.dcurlySadjc(patches, fpos, m, y22)
        # Scatter per-theta patch adjoints into full-grid coeff-space grad,
        # then apply the (self-adjoint) prefilter — identity for FFT.
        delta_coeff = cp.zeros((self.nzobj, self.nobj), dtype=coeff.dtype)
        self._scatter_add(delta_coeff, delta_patches, ipos)
        y32 = self._coeff_apply(delta_coeff)
        return [y21, y32, y33]

    @timer
    def min(self, prb, obj, pos):
        out = cp.zeros(1, dtype="float32")

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _min(self, out, pos, data, prb, obj):
            self._coeff_cache_reset()
            x = [prb, obj, pos]
            y = self.apply_F_from(x, 1)
            out[:] += self.F0(y, data)

        _min(self, out, pos, self.data, prb, obj)
        return float(self.allreduce(np.array([out[0].get()], dtype='float32'))[0])

    def vis_debug(self, vars, i, writer=None):
        if not (i % self.checkpoint_step == 0 and self.checkpoint_step != -1):
            return
        if writer is not None:
            if i > self.start_iter:
                writer.write_checkpoint(vars, i)
            if self.rank == 0 and hasattr(self, 'path_out') and self.path_out:
                tiff_dir  = os.path.join(self.path_out, 'checkpoints_tiff')
                obj_np = cp.asnumpy(vars['obj'])
                prb_np = cp.asnumpy(vars['prb'])
                tifffile.imwrite(os.path.join(tiff_dir, f'checkpoint_{i:04}_obj_delta.tiff'),
                                 obj_np.real if np.iscomplexobj(obj_np) else obj_np)
                if np.iscomplexobj(obj_np):
                    tifffile.imwrite(os.path.join(tiff_dir, f'checkpoint_{i:04}_obj_beta.tiff'),
                                     obj_np.imag)
                tifffile.imwrite(os.path.join(tiff_dir, f'checkpoint_{i:04}_prb_amp.tiff'),
                                 np.abs(prb_np))
                tifffile.imwrite(os.path.join(tiff_dir, f'checkpoint_{i:04}_prb_phase.tiff'),
                                 np.angle(prb_np))
                logger.info(f"FFP: obj + prb TIFFs saved → {tiff_dir}")
        elif self.rank == 0:
            if hasattr(self, 'path_out'):
                tiff_dir = os.path.join(self.path_out, 'checkpoints_tiff')
                logger.info(f"Saving iter {i}: obj, prb to {tiff_dir}")
                obj_np = cp.asnumpy(vars['obj'])
                prb_np = cp.asnumpy(vars['prb'])
                write_tiff(obj_np.real if np.iscomplexobj(obj_np) else obj_np,
                           f'{tiff_dir}/obj_delta{i:04}')
                if np.iscomplexobj(obj_np):
                    write_tiff(obj_np.imag, f'{tiff_dir}/obj_beta{i:04}')
                write_tiff(np.abs(prb_np),   f'{tiff_dir}/prb_amp{i:04}')
                write_tiff(np.angle(prb_np), f'{tiff_dir}/prb_phase{i:04}')
                np.save(f'{tiff_dir}/prb{i:04}.npy', prb_np)
            else:
                mshow_polar(vars['obj'], True) if self.obj_dtype == 'complex64' else mshow(vars['obj'].real, True)
                mshow_polar(vars['prb'], True)
                mshow_pos(vars['pos'] - self.pos_init, True)

    def error_debug(self, vars, i):
        """i=-1 is the initial-state call from BH (before the loop) and is always
        logged regardless of error_step."""
        if i != -1 and not (i % self.error_step == 0 and self.error_step != -1):
            return
        err = self.min(vars['prb'], vars['obj'], vars['pos'])

        # Gather position errors from all ranks to rank 0
        pos_err = (vars['pos'] - self.pos_init).get()   # [local_ntheta, 2]
        all_pos_err = self.cl_mpi.comm.gather(pos_err, root=0)

        if self.rank == 0:
            if i == -1:
                logger.warning(f"Initial {err=:1.5e}")
                self.table.loc[len(self.table)] = [i, err, 0]
            else:
                ittime = time.time() - self.time_start
                logger.warning(f"iter={i}: {ittime:.4f}sec {err=:1.5e}")
                self.table.loc[len(self.table)] = [i, err, ittime]
            pos_err_all = np.concatenate(all_pos_err, axis=0)
            _fmt_head = lambda a, k=8: (
                '[' + ', '.join(f'{v:.4f}' for v in a[:k].tolist())
                + (', ...' if len(a) > k else '') + ']'
            )
            logger.warning(f"  pos err y: {_fmt_head(pos_err_all[:, 0])}")
            logger.warning(f"  pos err x: {_fmt_head(pos_err_all[:, 1])}")
            self.time_start = time.time()
            if hasattr(self, 'path_out'):
                name = f"{self.path_out}/conv_ffp.csv"
                os.makedirs(os.path.dirname(name), exist_ok=True)
                self.table.to_csv(name, index=False)

    def gen_sqrt_data(self, vars, out):
        """Generate synthetic sqrt(intensity) data."""
        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _gen_data(self, out, pos, prb, obj):
            self._coeff_cache_reset()
            x = [prb, obj, pos]
            y = self.apply_F_from(x, 1)
            out[:] = cp.abs(y)
        _gen_data(self, out, vars['pos'], vars['prb'], vars['obj'])
