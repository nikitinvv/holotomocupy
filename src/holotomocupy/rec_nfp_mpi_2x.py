"""Near-field ptychography reconstruction at 2× super-resolution.

Sibling of :mod:`rec_nfp_mpi`: same MPI parallelisation, same BH machinery,
same cascade structure. The difference is that prb and proj live on a grid
twice as fine as the detector (and twice as fine as the user-facing object
grid). A real detector integrates *intensity* over the pixel area, so the
2×2 bin acts on |E|² and the loss is

    L = (1/N) Σ_θ ‖ sqrt( B(|E_θ|²) ) − d_θ ‖²,
    E_θ = D( P · exp(i · S_{2r_θ} ψ) )                       (dense field)
    B(I)[i, j] = (1/4) · Σ_{a,b ∈ {0,1}} I[2i+a, 2j+b]      (intensity bin)

i.e. *intensity-bin-then-sqrt*, not field-bin-then-magnitude. The bin
therefore lives inside F0; F1 / F2 / F3 stay in the dense complex domain
and the rest of the cascade is structurally identical to ``RecNFP``.

Sizes (with user-facing detector dims n, nz and object dims nobj, nzobj):
    prb      : (2·nz,    2·n)        — dense
    proj     : (2·nzobj, 2·nobj)     — dense
    pos      : (local_ntheta, 2)     — user-facing units (detector pixels);
                                       multiplied by 2 internally when handed
                                       to cl_shift (which works in dense px)
    data     : (local_ntheta, nz, n) — detector resolution, unchanged
    cl_prop  : built at (2·nz, 2·n) with voxelsize/2
    cl_shift : built at (2·n, 2·nobj, 2·nz, 2·nzobj)

Position scaling: pos values in ``vars['pos']`` are interpreted in detector-
pixel units (same as ``RecNFP``); the cascade multiplies them by 2 before
calling cl_shift and scales the returned position gradient by 2 (chain rule).
This keeps the user-facing meaning of ``rho_sq['pos']`` unchanged.
"""
import numpy as np
import cupy as cp
import os
import tifffile
import warnings
import pandas as pd
import nvtx
import cupy.fft

from .propagation import Propagation
from .shift import Shift
from .shift_fft import ShiftFFT
from .chunking import Chunking
from .utils import *
from .mpi_functions import *
from .logger_config import logger

np.set_printoptions(legacy="1.25")
warnings.filterwarnings("ignore", message=f".*peer.*")
cupy.fft.config.get_plan_cache().set_size(0)


class RecNFP2x:
    """Near-field ptychography at 2× super-resolution. See module docstring."""

    _var_names = ("prb", "proj", "pos")

    # Chain-rule factor for pos: cl_shift sees pos in dense-grid px (= 2·user px).
    POS_SCALE = np.float32(2.0)

    def __init__(self, args):

        for key, value in vars(args).items():
            setattr(self, key, value)

        self.F      = [self.F0,      self.F1,      self.F2,      self.F3]
        self.gF     = [self.gF0,     self.gF1,     self.gF2,     self.gF3]
        self.dF     = [self.dF0,     self.dF1,     self.dF2,     self.dF3]
        self.d2F_dF = [self.d2F_dF0, self.d2F_dF1, self.d2F_dF2, self.d2F_dF3]

        # Dense grid sizes used internally for prb / proj / propagation / shift.
        self.n2     = 2 * self.n
        self.nz2    = 2 * self.nz
        self.nobj2  = 2 * self.nobj
        self.nzobj2 = 2 * self.nzobj

        multiplier   = 4
        float_item   = np.dtype("float32").itemsize
        complex_item = np.dtype("complex64").itemsize
        # Dense per-theta scratch is 4× the native; data slab stays at (nz, n).
        nbytes = int(multiplier * self.nchunk * (
            self.nz * self.n * float_item
            + self.nz2 * self.n2 * complex_item))

        self.cl_mpi       = MPIClass(args.comm, self.nzobj2, self.ntheta, self.nobj2, args.obj_dtype)
        self.local_ntheta = self.cl_mpi.local_ntheta
        self.rank         = self.cl_mpi.rank
        self.st_theta     = self.cl_mpi.st_theta
        self.end_theta    = self.cl_mpi.end_theta

        if self.rank == 0 and hasattr(self, 'path_out') and self.path_out:
            os.makedirs(os.path.join(self.path_out, 'checkpoints_tiff'), exist_ok=True)
        args.comm.Barrier()

        wavelength    = 1.24e-09 / self.energy
        z1            = self.z1
        z2            = self.focustodetectordistance - z1
        magnification = self.focustodetectordistance / z1
        distance      = z1 * z2 / self.focustodetectordistance
        voxelsize     = self.detector_pixelsize / magnification
        # Dense object voxel is half the detector-projected pixel.
        voxelsize_dense = voxelsize / 2

        self.rho_sq = {
            'proj': args.rho[0]**2,
            'prb':  args.rho[1]**2,
            'pos':  args.rho[2]**2,
        }

        self.cl_chunking = Chunking(nbytes, self.nchunk)
        # Propagator on the dense grid with the dense voxel size.
        self.cl_prop     = Propagation(self.n2, self.nz2, self.nchunk, 1,
                                       wavelength, voxelsize_dense,
                                       np.array([distance]))
        # Shift on the dense grid; pos passed in dense-grid px (×POS_SCALE).
        if self.shift_type == 'fft':
            self.cl_shift = ShiftFFT(self.n2, self.nobj2, self.nz2, self.nzobj2,
                                     self.obj_dtype, symmetric=self.shift_symmetric)
        elif self.shift_type == 'cubic':
            self.cl_shift = Shift(self.n2, self.nobj2, self.nz2, self.nzobj2,
                                  self.obj_dtype, symmetric=self.shift_symmetric)
        else:
            raise ValueError(f"shift_type must be 'cubic' or 'fft', got {self.shift_type!r}")

        self.alloc_arrays()

        self.table = pd.DataFrame(columns=["iter", "err", "time"])

        # data_size: number of measured values (data is at detector resolution).
        self.data_size = self.ntheta * self.nz * self.n
        self.prb_size  = self.nz2 * self.n2

        self.gpu_batch    = self.cl_chunking.gpu_batch
        self.redot_batch  = self.cl_chunking.redot_batch
        self.linear_batch = self.cl_chunking.linear_batch
        self.mulc_batch   = self.cl_chunking.mulc_batch
        self.allreduce    = self.cl_mpi.allreduce
        self.allreduce2   = self.cl_mpi.allreduce2

    def alloc_arrays(self):
        self.vars = {
            'prb':  cp.empty([self.nz2, self.n2],         dtype='complex64'),
            'proj': cp.zeros([self.nzobj2, self.nobj2],   dtype=self.obj_dtype),
            'pos':  cp.zeros([self.local_ntheta, 2],      dtype='float32'),
        }
        self.data = make_pinned([self.local_ntheta, self.nz, self.n], dtype='float32')
        self.grads, self.etas = {}, {}
        for ge in self.grads, self.etas:
            ge['prb']  = cp.zeros([self.nz2, self.n2],       dtype='complex64')
            ge['proj'] = cp.zeros([self.nzobj2, self.nobj2], dtype=self.obj_dtype)
            ge['pos']  = cp.zeros([self.local_ntheta, 2],    dtype='float32')

    def BH(self, writer=None):
        vars  = self.vars
        grads = self.grads
        etas  = self.etas

        self.precalc(vars)
        self.error_debug(vars, -1)

        self.time_start = time.time()
        for i in range(self.start_iter, self.niter):
            with nvtx.annotate(f"::BH:nfp2x:{i}"):
                self.compute_gradient(vars, grads)
                self.compute_beta(vars, grads, etas, i)
                alpha = self.compute_alpha(vars, grads, etas)
                self.apply_step(vars, etas, alpha)
                self.log_iter(vars, i, writer)

        return vars

    def precalc(self, vars):
        self.pos_init = vars['pos'].copy()

    def compute_gradient(self, vars, grads):
        with nvtx.annotate("gradients"):
            self.gradients(vars, grads)
        for v in self._var_names:
            self.mulc_batch(grads[v], grads[v], self.rho_sq[v])

    def compute_beta(self, vars, grads, etas, i):
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
        for v in self._var_names:
            self.linear_batch(vars[v], etas[v], 1, alpha)

    def log_iter(self, vars, i, writer):
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
            x2, y2, z2,
            x0, y0, z0,
            x1, y1, z1,
        ):
            self.cl_shift.coeff_cache_reset()
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
            vars["pos"],  grads["pos"],  etas["pos"],
            vars["prb"],  grads["prb"],  etas["prb"],
            vars["proj"], grads["proj"], etas["proj"],
        )
        return out[0].get()

    def gradients(self, vars, grads):
        self.gradients_cascade(vars, grads)
        grads['prb'][:]  = cp.array(self.allreduce(grads['prb'].get()))
        grads['proj'][:] = cp.array(self.allreduce(grads['proj'].get()))

    @timer
    def gradients_cascade(self, vars, grads):
        grads['prb'][:]  = 0
        grads['proj'][:] = 0

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=3)
        def _gradients_cascade(self, gradpos, gradprb, gradproj, d, pos, prb, proj):
            self.cl_shift.coeff_cache_reset()
            x = [prb, proj, pos]
            y = d
            for id in range(len(self.gF)):
                y = self.gF[id](x, y)
            gradprb[:]  += y[0]
            gradproj[:] += y[1]
            gradpos[:]   = y[2]

        _gradients_cascade(
            self, grads['pos'], grads['prb'], grads['proj'],
            self.data, vars['pos'], vars['prb'], vars['proj'],
        )

    ####################### 2×2 average-bin operator ########################
    # bin2x2 acts on real-valued intensity (Σ/4 over 2×2 blocks).
    # upsample_2x replicates each detector-grid scalar across its 4 dense
    # children with no scaling — used to broadcast small-grid scalars (e.g.
    # 1 − d/A) into the dense field's pointwise multiplications.

    @staticmethod
    def bin2x2(y):
        """Average over 2×2 blocks of the last two axes. (..., 2H, 2W) → (..., H, W)."""
        s = y.shape
        return y.reshape(*s[:-2], s[-2] // 2, 2, s[-1] // 2, 2).mean(axis=(-3, -1))

    @staticmethod
    def upsample_2x(z):
        """Broadcast each pixel to its 2×2 dense block (no scaling)."""
        return cp.repeat(cp.repeat(z, 2, axis=-2), 2, axis=-1)

    ####################### Cascade functions #######################

    def apply_F_from(self, x, from_level):
        for k in range(from_level, len(self.F))[::-1]:
            x = self.F[k](x)
        return x

    ####### F0: ‖ sqrt(B(|E|²)) − d ‖² / data_size  (physical detector model)
    # E is the dense complex field (the F1 output, shape (·, 2nz, 2n));
    # d is the detector-resolution measured amplitude (shape (·, nz, n)).
    # Quantities used throughout (all live on the detector grid):
    #     I = B(|E|²),   A = sqrt(I + eps),   res = 1 − d/A,   d/A = d0
    # eps avoids 0/0 in the rare case the predicted intensity is exactly
    # zero; in practice A > 0 strictly because |prb|·|exp(iSψ)| > 0.

    _EPS = np.float32(1e-30)

    def F0(self, x, d):
        I = self.bin2x2(reprod(x, x))
        A = cp.sqrt(I + self._EPS)
        diff = A - d
        return np.float32(1 / self.data_size) * cp.sum(diff * diff)

    def dF0(self, x, y, d, return_x=False):
        # dL/dE · y  =  (2/N) Σ_{ij} (1 − d/A) · B(reprod(E, y))[i,j]
        I = self.bin2x2(reprod(x, x))
        A = cp.sqrt(I + self._EPS)
        BEV = self.bin2x2(reprod(x, y))
        return np.float32(2 / self.data_size) * cp.sum((1 - d / A) * BEV)

    def d2F_dF0(self, x, y, z, w, d):
        # d²L/dE² (y, z) + dL/dE · w, all reduced through B to the small grid.
        # Per detector pixel:
        #   v = (1 − d/A)·B(reprod(y,z))
        #     + (d/A)   · (B(reprod(E,y))/A) · (B(reprod(E,z))/A)
        #     + [w]      (1 − d/A) · B(reprod(E, w))
        I = self.bin2x2(reprod(x, x))
        A = cp.sqrt(I + self._EPS)
        BEV = self.bin2x2(reprod(x, y))
        BEW = self.bin2x2(reprod(x, z))
        BVW = self.bin2x2(reprod(y, z))
        d0  = d / A
        ll0 = 1 - d0
        v = ll0 * BVW + d0 * (BEV / A) * (BEW / A)
        if w is not None:
            BEw = self.bin2x2(reprod(x, w))
            v = v + ll0 * BEw
        return np.float32(2 / self.data_size) * cp.sum(v)

    def gF0(self, x, y):
        # Codebase convention: gradient = 2 ∂L/∂Ē.
        # gF0 = (1/(2N)) · E · upsample_2x(1 − d/A).
        E = self.apply_F_from(x, 1)
        I = self.bin2x2(reprod(E, E))
        A = cp.sqrt(I + self._EPS)
        res_dense = self.upsample_2x(1 - y / A)
        return np.float32(1 / (2 * self.data_size)) * E * res_dense

    ####### F1: (prb, exp_proj) → D(prb · exp_proj)   (dense complex wave)
    # The bin lives in F0; F1 / F2 / F3 are identical to RecNFP modulo the
    # fact that everything here is on the dense grid.

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
        # y arrives on the dense grid (gF0 returns dense); no B^T lifting.
        x = self.apply_F_from(x, 2)
        x11, x12 = x
        y12 = self.cl_prop.DT(y, 0)
        y11 = cp.sum(y12 * cp.conj(x12), axis=0)
        y12 = y12 * cp.conj(x11)
        return y11, y12

    ####### F2: exp(i·shifted_proj)  (unchanged shape-wise; runs on dense grid)
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

    ####### F3: (prb, proj, pos) → (prb, S_{POS_SCALE·pos}(proj))
    # Position scaling: cl_shift works in dense-grid px; user-facing pos is in
    # detector px. We multiply by POS_SCALE (=2) on input and pull the same
    # factor out via chain rule on the gradient.

    def _tiled_coeff(self, psi, n):
        return cp.tile(self.cl_shift.coeff_cached(psi)[None], [n, 1, 1])

    def F3(self, x):
        x31, x32, x33 = x
        n = len(x33)
        c = self._tiled_coeff(x32, n)
        m = cp.ones(n, dtype='float32')
        return x31, self.cl_shift.curlySc(c, self.POS_SCALE * x33, m)

    def dF3(self, x, y, return_x=True):
        x31, x32, x33 = x
        y31, y32, y33 = y
        n  = len(x33)
        c  = self._tiled_coeff(x32, n)
        c1 = self._tiled_coeff(y32, n)
        m  = cp.ones(n, dtype='float32')
        r  = self.POS_SCALE * x33
        dr = self.POS_SCALE * y33
        y22 = self.cl_shift.dcurlySc(c, r, m, c1, dr)
        if return_x:
            x22 = self.cl_shift.curlySc(c, r, m)
            return [x31, x22], [y31, y22]
        return [y31, y22]

    def d2F_dF3(self, x, y, z, w):
        x31, x32, x33 = x
        y31, y32, y33 = y
        z31, z32, z33 = z
        w31, w32, w33 = w
        n  = len(x33)
        c  = self._tiled_coeff(x32, n)
        cy = self._tiled_coeff(y32, n)
        cz = self._tiled_coeff(z32, n)
        m  = cp.ones(n, dtype='float32')
        r  = self.POS_SCALE * x33
        dy = self.POS_SCALE * y33
        dz = self.POS_SCALE * z33
        y22 = self.cl_shift.d2curlySc(c, r, m, cy, dy, cz, dz)
        if w32 is not None:
            cw = self._tiled_coeff(w32, n)
            dw = self.POS_SCALE * w33
            y22 = y22 + self.cl_shift.dcurlySc(c, r, m, cw, dw)
        return [w31, y22]

    def gF3(self, x, y):
        y21, y22 = y
        x = self.apply_F_from(x, 4)
        x31, x32, x33 = x
        n = len(x33)
        c = self._tiled_coeff(x32, n)
        m = cp.ones(n, dtype='float32')
        Deltapsi, y33 = self.cl_shift.dcurlySadjc(c, self.POS_SCALE * x33, m, y22)
        # Chain rule: f(p) = S(POS_SCALE·p)  ⇒  ∇_p f = POS_SCALE · ∇_q S.
        y33 = self.POS_SCALE * y33
        y32 = cp.sum(Deltapsi, axis=0)
        y32 = self.cl_shift.coeff(y32)
        return [y21, y32, y33]

    @timer
    def min(self, prb, proj, pos):
        out = cp.zeros(1, dtype="float32")

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _min(self, out, pos, data, prb, proj):
            self.cl_shift.coeff_cache_reset()
            x = [prb, proj, pos]
            y = self.apply_F_from(x, 1)
            out[:] += self.F0(y, data)

        _min(self, out, pos, self.data, prb, proj)
        return float(self.allreduce(np.array([out[0].get()], dtype='float32'))[0])

    def vis_debug(self, vars, i, writer=None):
        if not (i % self.checkpoint_step == 0 and self.checkpoint_step != -1):
            return
        if writer is not None:
            if i > self.start_iter:
                writer.write_checkpoint(vars, i)
            if self.rank == 0 and hasattr(self, 'path_out') and self.path_out:
                tiff_dir  = os.path.join(self.path_out, 'checkpoints_tiff')
                tiff_path = os.path.join(tiff_dir, f'checkpoint_{i:04}_proj_re.tiff')
                tifffile.imwrite(tiff_path, cp.asnumpy(vars['proj'].real))
                logger.info(f"NFP2x: proj_re TIFF saved → {tiff_path}")
        elif self.rank == 0:
            if hasattr(self, 'path_out'):
                tiff_dir = os.path.join(self.path_out, 'checkpoints_tiff')
                logger.info(f"Saving iter {i}: proj, prb to {tiff_dir}")
                write_tiff(vars['proj'].real,     f'{tiff_dir}/proj{i:04}')
                write_tiff(cp.angle(vars['prb']), f'{tiff_dir}/prb{i:04}')
                np.save(f'{tiff_dir}/prb{i:04}.npy', vars['prb'].get())
            else:
                mshow(vars['proj'].real, True)
                mshow_polar(vars['prb'], True)
                mshow_pos(vars['pos'] - self.pos_init, True)

    def error_debug(self, vars, i):
        if i != -1 and not (i % self.error_step == 0 and self.error_step != -1):
            return
        err = self.min(vars['prb'], vars['proj'], vars['pos'])

        pos_err = (vars['pos'] - self.pos_init).get()
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
            logger.warning(f"  pos err y: {np.array2string(pos_err_all[:, 0], precision=4, separator=', ')}")
            logger.warning(f"  pos err x: {np.array2string(pos_err_all[:, 1], precision=4, separator=', ')}")
            self.time_start = time.time()
            if hasattr(self, 'path_out'):
                name = f"{self.path_out}/conv_nfp.csv"
                os.makedirs(os.path.dirname(name), exist_ok=True)
                self.table.to_csv(name, index=False)

    def gen_sqrt_data(self, vars, out):
        # Synthetic detector amplitudes are sqrt of the binned dense intensity,
        # matching the F0 forward model.
        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _gen_data(self, out, pos, prb, proj):
            self.cl_shift.coeff_cache_reset()
            x = [prb, proj, pos]
            E = self.apply_F_from(x, 1)
            out[:] = cp.sqrt(self.bin2x2(reprod(E, E)) + self._EPS)
        _gen_data(self, out, vars['pos'], vars['prb'], vars['proj'])
