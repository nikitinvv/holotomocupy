import numpy as np
import cupy as cp
import os
import warnings
import pandas as pd
import nvtx

from .propagation import Propagation
from .shift import Shift
from .chunking import Chunking
from .utils import *
from .mpi_functions import *
from .logger_config import logger

np.set_printoptions(legacy="1.25")
warnings.filterwarnings("ignore", message=f".*peer.*")


class RecNFP:
    """Near-field ptychography reconstruction (MPI-parallel over theta).

    Forward model:
        x0 = F1(F2(F3(prb, proj, pos)))
           = D( prb * exp(i * S_pos(proj)) )

    Variables: prb (nz×n, complex), proj (nzobj×nobj, real), pos (ntheta×2)
    Parallelisation: theta distributed across MPI ranks; prb/proj replicated.
    """

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

        # MPI: distribute theta; prb/proj replicated on all ranks
        self.cl_mpi       = MPIClass(args.comm, self.nzobj, self.ntheta, self.nobj, args.obj_dtype)
        self.local_ntheta = self.cl_mpi.local_ntheta
        self.rank         = self.cl_mpi.rank
        self.st_theta     = self.cl_mpi.st_theta
        self.end_theta    = self.cl_mpi.end_theta

        wavelength    = 1.24e-09 / self.energy
        z1            = self.z1
        z2            = self.focustodetectordistance - z1
        magnification = self.focustodetectordistance / z1
        distance      = z1 * z2 / self.focustodetectordistance
        voxelsize     = self.detector_pixelsize / magnification

        self.rho_sq = {
            'proj': args.rho[0]**2,
            'prb':  args.rho[1]**2,
            'pos':  args.rho[2]**2,
        }

        self.cl_chunking = Chunking(nbytes, self.nchunk)
        self.cl_prop     = Propagation(self.n, self.nz, self.nchunk, 1, wavelength, voxelsize,
                                       np.array([distance]))
        self.cl_shift    = Shift(self.n, self.nobj, self.nz, self.nzobj,self.obj_dtype)

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
            'prb':  cp.empty([self.nz, self.n],       dtype='complex64'),
            'proj': cp.zeros([self.nzobj, self.nobj],  dtype=self.obj_dtype),
            'pos':  cp.zeros([self.local_ntheta, 2],   dtype='float32'),
        }
        self.data = make_pinned([self.local_ntheta, self.nz, self.n], dtype='float32')
        self.grads, self.etas = {}, {}
        for ge in self.grads, self.etas:
            ge['prb']  = cp.zeros([self.nz, self.n],      dtype='complex64')
            ge['proj'] = cp.zeros([self.nzobj, self.nobj], dtype=self.obj_dtype)
            ge['pos']  = cp.zeros([self.local_ntheta, 2],  dtype='float32')

    def BH(self, writer=None):
        vars  = self.vars
        grads = self.grads
        etas  = self.etas

        self.pos_init = vars['pos'].copy()
        self.error_debug(vars, -1)

        self.time_start = time.time()
        for i in range(self.start_iter, self.niter):
            nvtx.push_range("::BH:nfp:" + str(i))

            self.gradients(vars, grads)

            for v in ["proj", "prb", "pos"]:
                self.mulc_batch(grads[v], grads[v], self.rho_sq[v])

            if i == self.start_iter:
                for v in ["prb", "proj", "pos"]:
                    self.mulc_batch(etas[v], grads[v], -1)
            else:
                top, bottom = self.allreduce2(
                    self.hessian(vars, grads, etas),
                    self.hessian(vars, etas,  etas),
                )
                beta = top / bottom
                for v in ["prb", "proj", "pos"]:
                    self.linear_batch(etas[v], grads[v], beta, -1)

            top = -self.redot_batch(grads['pos'], etas['pos']) / self.rho_sq['pos']
            if self.rank == 0:
                top -= self.redot_batch(grads['prb'],  etas['prb'])  / self.rho_sq['prb']
                top -= self.redot_batch(grads['proj'], etas['proj']) / self.rho_sq['proj']

            bottom = self.hessian(vars, etas, etas)
            top, bottom = self.allreduce2(top, bottom)
            alpha = top / bottom

            for v in ["prb", "proj", "pos"]:
                self.linear_batch(vars[v], etas[v], 1, alpha)

            self.error_debug(vars, i)
            self.vis_debug(vars, i, writer)
            nvtx.pop_range()

        return vars

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
            x1, y1, z1,   # proj — non-proper gpu
        ):
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

    ####################### Cascade functions #######################
    # Variables: x = [prb, proj, pos]
    # F3: (prb, proj, pos) → (prb, S_pos(proj))
    # F2: (prb, shifted_proj) → (prb, exp(i·shifted_proj))
    # F1: (prb, exp_proj) → D(prb · exp_proj)
    # F0: ||·| - d||²
    #################################################################

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
    def _dF0_fused(x, d):
        return x - d * (x / cp.abs(x))

    def dF0(self, x, y, d, return_x=False):
        return 2 / self.data_size * redot(self._dF0_fused(x, d), y)

    @staticmethod
    @cp.fuse()
    def _d2F_dF0_fused(x, y, z, w, d):
        absval = cp.abs(x)
        l0 = x / absval
        d0 = d / absval
        v = (1 - d0) * reprod(y, z) + d0 * reprod(l0, y) * reprod(l0, z)
        if w is not None:
            v += reprod(x - d * l0, w)
        return v

    def d2F_dF0(self, x, y, z, w, d):
        return 2 / self.data_size * cp.sum(self._d2F_dF0_fused(x, y, z, w, d))

    @staticmethod
    @cp.fuse()
    def _gF0_fused(x, y, scale):
        td = y * (x / cp.abs(x))
        return scale * (x - td)

    def gF0(self, x, y):
        for id in range(1, 4)[::-1]:
            x = self.F[id](x)
        return self._gF0_fused(x, y, np.float32(2 / self.data_size))

    ####### F1: (prb, exp_proj) → D(prb · exp_proj)
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
        for id in range(2, 4)[::-1]:   # apply F3, F2 to reach F1's input space
            x = self.F[id](x)
        x11, x12 = x
        y12 = self.cl_prop.DT(y0, 0)
        y11 = cp.sum(y12 * cp.conj(x12), axis=0)  # sum over theta → (nz, n)
        y12 = y12 * cp.conj(x11)
        return y11, y12

    ####### F2: (prb, shifted_proj) → (prb, exp(i·shifted_proj))
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
        for id in range(3, 4)[::-1]:   # apply F3 to reach F2's input space
            x = self.F[id](x)
        x21, x22 = x
        y22 = self._gF2_fused(x22, y12)
        y22 = y22.real if self.obj_dtype == 'float32' else y22
        return [y11, y22]

    ####### F3: (prb, proj, pos) → (prb, S_pos(proj))
    def F3(self, x):
        x31, x32, x33 = x
        c = self.cl_shift.coeff(x32)
        c = cp.tile(c[None], [len(x33), 1, 1])
        m = cp.ones(len(x33), dtype='float32')
        return x31, self.cl_shift.curlySc(c, x33, m)

    def dF3(self, x, y, return_x=True):
        x31, x32, x33 = x
        y31, y32, y33 = y
        c  = self.cl_shift.coeff(x32)
        c  = cp.tile(c[None], [len(x33), 1, 1])
        c1 = self.cl_shift.coeff(y32)
        c1 = cp.tile(c1[None], [len(x33), 1, 1])
        m = cp.ones(len(x33), dtype='float32')
        y22 = self.cl_shift.dcurlySc(c, x33, m, c1, y33)
        if return_x:
            x22 = self.cl_shift.curlySc(c, x33, m)
            return [x31, x22], [y31, y22]
        return [y31, y22]

    def d2F_dF3(self, x, y, z, w):
        x31, x32, x33 = x
        y31, y32, y33 = y
        z31, z32, z33 = z
        w31, w32, w33 = w
        c  = self.cl_shift.coeff(x32)
        cy = self.cl_shift.coeff(y32)
        cz = self.cl_shift.coeff(z32)
        n  = len(x33)
        c  = cp.tile(c[None],  [n, 1, 1])
        cy = cp.tile(cy[None], [n, 1, 1])
        cz = cp.tile(cz[None], [n, 1, 1])
        m = cp.ones(n, dtype='float32')
        y22 = self.cl_shift.d2curlySc(c, x33, m, cy, y33, cz, z33)
        if w32 is not None:
            cw = cp.tile(self.cl_shift.coeff(w32)[None], [n, 1, 1])
            y22 = y22 + self.cl_shift.dcurlySc(c, x33, m, cw, w33)
        return [w31, y22]

    def gF3(self, x, y):
        y21, y22 = y
        # x is already at the (prb, proj, pos) level — no forward apply needed
        x31, x32, x33 = x
        c = self.cl_shift.coeff(x32)
        c = cp.tile(c[None], [len(x33), 1, 1])
        m = cp.ones(len(x33), dtype='float32')
        Deltapsi, y33 = self.cl_shift.dcurlySadjc(c, x33, m, y22)
        y32 = cp.zeros([self.nzobj, self.nobj], dtype=self.obj_dtype)
        y32[:] = cp.sum(Deltapsi, axis=0)
        y32[:] = self.cl_shift.coeff(y32)
        return [y21, y32, y33]

    @timer
    def min(self, prb, proj, pos):
        out = cp.zeros(1, dtype="float32")

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _min(self, out, pos, data, prb, proj):
            x = [prb, proj, pos]
            y = x
            for id in range(1, len(self.F))[::-1]:
                y = self.F[id](y)
            out[:] += self.F0(y, data)

        _min(self, out, pos, self.data, prb, proj)
        return float(self.allreduce(np.array([out[0].get()], dtype='float32'))[0])

    def vis_debug(self, vars, i, writer=None):
        if not (i % self.vis_step == 0 and self.vis_step != -1):
            return
        if writer is not None:
            if i > self.start_iter:
                writer.write_checkpoint(vars, i)
        elif self.rank == 0:
            if hasattr(self, 'path_out'):
                logger.info(f"Saving iter {i}: proj, prb to {self.path_out}")
                write_tiff(vars['proj'].real,     f'{self.path_out}/proj{i:04}')
                write_tiff(cp.angle(vars['prb']), f'{self.path_out}/prb{i:04}')
                np.save(f'{self.path_out}/prb{i:04}.npy', vars['prb'].get())
            else:
                mshow(vars['proj'].real, True)
                mshow_polar(vars['prb'], True)
                mshow_pos(vars['pos'] - self.pos_init, True)

    def error_debug(self, vars, i):
        if not (i % self.err_step == 0 and self.err_step != -1):
            return
        err = self.min(vars['prb'], vars['proj'], vars['pos'])
        if self.rank == 0:
            if i == -1:
                logger.warning(f"Initial {err=:1.5e}")
                self.table.loc[len(self.table)] = [i, err, 0]
            else:
                ittime = time.time() - self.time_start
                logger.warning(f"iter={i}: {ittime:.4f}sec {err=:1.5e}")
                self.table.loc[len(self.table)] = [i, err, ittime]
            self.time_start = time.time()
            if hasattr(self, 'path_out'):
                name = f"{self.path_out}/conv_nfp.csv"
                os.makedirs(os.path.dirname(name), exist_ok=True)
                self.table.to_csv(name, index=False)

    def gen_sqrt_data(self, vars, out):
        """Generate synthetic sqrt(intensity) data."""
        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _gen_data(self, out, pos, prb, proj):
            x = [prb, proj, pos]
            y = x
            for id in range(1, len(self.F))[::-1]:
                y = self.F[id](y)
            out[:] = cp.abs(y)
        _gen_data(self, out, vars['pos'], vars['prb'], vars['proj'])
