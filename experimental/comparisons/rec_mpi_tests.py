import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import os
import warnings
import pandas as pd

from holotomocupy.tomo import Tomo
from holotomocupy.propagation import Propagation
from holotomocupy.shift import Shift
from holotomocupy.chunking import Chunking
from holotomocupy.utils import *
from holotomocupy.mpi_functions import *
from holotomocupy.logger_config import logger
from holotomocupy.conv2d_cufftdx import precompile as cufftdx_precompile

np.set_printoptions(legacy="1.25")
warnings.filterwarnings("ignore", message=f".*peer.*")


# --- method groupings ---
_LBFGS_METHODS      = frozenset({3, 6, 8, 9, 11, 12})
_BH_BETA_METHODS    = frozenset({0, 4, 5, 10})
_BH_ALPHA_METHODS   = frozenset({0, 1, 4, 6, 12})   # always use quadratic BH step
_SCAN_METHODS       = frozenset({4})                  # scan LS on top of BH step
_ARMIJO_METHODS     = frozenset({7, 9, 10, 11})
_LBFGS_LS_INIT      = frozenset({3, 8, 9, 11})       # set alpha=1/16 at i==1
_LBFGS_M10_METHODS  = frozenset({8, 9, 11, 12})


class Rec:
    def __init__(self, args):

        # copy args to elements of the class
        for key, value in vars(args).items():
            setattr(self, key, value)

        # list of functionals, gradients, differentials, and second-order differentials
        self.F = [self.F0, self.F1, self.F2, self.F3]
        self.gF = [self.gF0, self.gF1, self.gF2, self.gF3]
        self.dF = [self.dF0, self.dF1, self.dF2, self.dF3]
        self.d2F_dF = [self.d2F_dF0, self.d2F_dF1, self.d2F_dF2,self.d2F_dF3]

        # estimate memory footprint for pinned + device buffer per GPU (complex64)
        multiplier = 8  # related to the number of arrays, experimentally chosen. the scheme will diverge if too low
        complex_item = np.dtype("complex64").itemsize
        max_dim = max(self.nzobj, self.ntheta)
        nbytes = int(multiplier * self.nchunk * self.nobj * max_dim * complex_item)

        ### multinode processing
        self.cl_mpi = MPIClass(args.comm, self.nzobj, self.ntheta, self.nobj, args.obj_dtype)
        self.local_nzobj = self.cl_mpi.local_n_src
        self.local_ntheta = self.cl_mpi.local_n_dst
        self.rank      = self.cl_mpi.rank
        self.st_obj    = self.cl_mpi.st_src
        self.end_obj   = self.cl_mpi.end_src
        self.st_theta  = self.cl_mpi.st_dst
        self.end_theta = self.cl_mpi.end_dst

        # X-ray propagation and magnification parameters for classes
        wavelength = 1.24e-09 / self.energy
        z2 = self.focustodetectordistance - self.z1
        magnifications = self.focustodetectordistance / self.z1
        norm_magnifications = magnifications / magnifications[0]
        distance = (self.z1 * z2) / self.focustodetectordistance * norm_magnifications**2
        voxelsize = self.detector_pixelsize / magnifications[0]

        # scaling variables
        self.rho_sq = {'obj': args.rho[0]**2, 'prb': args.rho[1]**2, 'pos': args.rho[2]**2}

        # cuFFTDx JIT compile: rank 0 builds the .so, then all ranks proceed
        if self.rank == 0:
            cufftdx_precompile(2 * self.nz, 2 * self.n)
        self.cl_mpi.comm.Barrier()

        # create classes (one GPU per MPI rank via CUDA_VISIBLE_DEVICES)
        self.cl_chunking = Chunking(nbytes, self.nchunk)
        self.cl_tomo  = Tomo(self.nobj, self.nchunk, self.theta, self.mask)
        self.cl_prop  = Propagation(self.n, self.nz, self.nchunk, self.ndist, wavelength, voxelsize, distance)
        self.cl_shift = Shift(self.n, self.nobj, self.nz, self.nzobj, 1.0 / norm_magnifications, self.obj_dtype)

        self.alloc_arrays()

        # save convergence results
        self.table = pd.DataFrame(columns=["iter", "alpha", "beta", "err", "time"])

        # normalization constant to address work with normal operators
        self.norm_const = np.float32(np.sqrt(self.nobj / self.ntheta))
        # sizes for normalization
        self.data_size = self.ntheta * self.ndist * self.nz * self.n
        self.prb_size = self.ndist * self.nz * self.n
        self.obj_size = self.nzobj * self.nobj**2

        # fast refs
        self.gpu_batch = self.cl_chunking.gpu_batch
        self.redot_batch = self.cl_chunking.redot_batch
        self.linear_batch = self.cl_chunking.linear_batch
        self.mulc_batch = self.cl_chunking.mulc_batch
        self.redist = self.cl_mpi.redist
        self.allreduce  = self.cl_mpi.allreduce
        self.allreduce2 = self.cl_mpi.allreduce2

        # precomputed method flags (avoid per-iteration if-chains)
        self._use_lbfgs      = self.method in _LBFGS_METHODS
        self._use_bh_beta    = self.method in _BH_BETA_METHODS
        self._use_bh_alpha   = self.method in _BH_ALPHA_METHODS
        self._use_scan       = self.method in _SCAN_METHODS
        self._use_armijo     = self.method in _ARMIJO_METHODS
        self._lbfgs_ls_init  = self.method in _LBFGS_LS_INIT

    def alloc_arrays(self):
        """Allocate all pinned CPU and CuPy GPU buffers used during reconstruction."""
        prb_shape = [self.ndist, self.nz, self.n]
        # reconstruction variables
        self.vars = {
            'obj':  make_pinned([self.local_nzobj,  self.nobj, self.nobj], dtype=self.obj_dtype),
            'pos':  cp.zeros([self.local_ntheta, self.ndist, 2],           dtype='float32'),
            'prb':  cp.empty(prb_shape,                                    dtype='complex64'),
            'proj': make_pinned([self.local_ntheta, self.nzobj, self.nobj], dtype=self.obj_dtype),
        }
        # measurement data and reference
        self.data = make_pinned([self.local_ntheta, self.ndist, self.nz, self.n], dtype='float32')
        self.ref  = cp.empty(prb_shape,                                           dtype='float32')
        # gradient and conjugate-direction buffers
        self.grads, self.grads1, self.grads2, self.etas = {}, {}, {}, {}
        for ge in self.grads, self.grads1, self.grads2, self.etas:
            ge["obj"]  = make_pinned([self.local_nzobj,  self.nobj, self.nobj], dtype=self.obj_dtype)
            ge["pos"]  = cp.zeros([self.local_ntheta, self.ndist, 2],           dtype='float32')
            ge["proj"] = make_pinned([self.local_ntheta, self.nzobj, self.nobj], dtype=self.obj_dtype)
            ge["prb"]  = cp.empty(prb_shape, dtype='complex64')
        self.proj_tmp = make_pinned([self.ntheta, self.local_nzobj, self.nobj], dtype=self.obj_dtype)

        # L-BFGS history buffers (methods 3,6,8,9,11,12), keyed by obj/prb/pos (proj excluded)
        self.lbfgs_m = 10 if self.method in _LBFGS_M10_METHODS else 5
        lbfgs_m = self.lbfgs_m
        self.lbfgs_s, self.lbfgs_y = [], []
        for _ in range(lbfgs_m):
            self.lbfgs_s.append({
                'obj': make_pinned([self.local_nzobj, self.nobj, self.nobj], dtype=self.obj_dtype),
                'pos': cp.zeros([self.local_ntheta, self.ndist, 2],          dtype='float32'),
                'prb': cp.empty(prb_shape,                                   dtype='complex64'),
            })
            self.lbfgs_y.append({
                'obj': make_pinned([self.local_nzobj, self.nobj, self.nobj], dtype=self.obj_dtype),
                'pos': cp.zeros([self.local_ntheta, self.ndist, 2],          dtype='float32'),
                'prb': cp.empty(prb_shape,                                   dtype='complex64'),
            })
        self.lbfgs_rho = [0.0] * lbfgs_m
        self.lbfgs_k = 0   # number of valid pairs (capped at lbfgs_m)
        self.lbfgs_t = 0   # circular-buffer write pointer

    def BH(self, writer=None):
        vars = self.vars
        grads = self.grads
        grads1 = self.grads1
        etas = self.etas
        proj_tmp = self.proj_tmp

        vars["obj"] /= self.norm_const
        if self.start_iter == 0:
            vars["obj"] *= self.cl_tomo.mask

        self.pos_init = vars['pos'].copy()

        self.fwd_tomo(vars["obj"], out=proj_tmp)
        self.redist(proj_tmp, vars['proj'])

        self.error_debug(vars, 0, 0, -1)

        st = 2
        self.time_start = time.time()
        for i in range(self.start_iter, self.niter):

            # --- gradients ---
            for v in ["obj", "prb", "pos", 'proj']:
                grads1[v][:] = grads[v]
            self.gradients(vars, grads)

            for v in ["obj", "prb", "pos"]:
                self.mulc_batch(grads[v], grads[v], self.rho_sq[v])

            self.fwd_tomo(grads["obj"], out=proj_tmp)
            self.cl_mpi.redist(proj_tmp, grads['proj'])

            beta = 0.0

            # --- search direction ---
            if i < self.start_method or alpha==0:
                #reset history for lbfgs
                self.lbfgs_k = 0
                self.lbfgs_t = 0
                for v in ["obj", "prb", "pos", "proj"]:
                    self.mulc_batch(etas[v], grads[v], -1)

            elif self._use_lbfgs:
                if i > self.start_iter:
                    self._lbfgs_push(grads, grads1, alpha)
                self._lbfgs_direction(grads, etas)
                # check descent; reset and fall back to steepest descent if needed
                g0_test = 0.0
                for v in ["obj", "pos"]:
                    g0_test += self.redot_batch(grads[v], etas[v]) / self.rho_sq[v]
                if self.rank == 0:
                    g0_test += self.redot_batch(grads['prb'], etas['prb']) / self.rho_sq['prb']
                g0_test, _ = self.allreduce2(g0_test, 0.0)
                if g0_test >= 0:
                    if self.rank == 0:
                        print("L-BFGS: non-descent direction, resetting history")
                    self.lbfgs_k = 0
                    self.lbfgs_t = 0
                    for v in ["obj", "prb", "pos", "proj"]:
                        self.mulc_batch(etas[v], grads[v], -1)
                self.fwd_tomo(etas["obj"], out=proj_tmp)
                self.cl_mpi.redist(proj_tmp, etas['proj'])

            else:
                if self._use_bh_beta:
                    top = self.hessian(vars, grads, etas)
                    bottom = self.hessian(vars, etas, etas)
                    top, bottom = self.allreduce2(top, bottom)
                    beta = max(top / bottom, 0)
                else:  # PR+
                    beta = self.beta_pr_plus(grads1, grads)
                for v in ["obj", "prb", "pos", "proj"]:
                    self.linear_batch(etas[v], grads[v], beta, -1)

            # --- step length ---
            if i<self.start_method or self._use_bh_alpha:
                top = 0
                for v in ["obj", "pos"]:
                    top -= self.redot_batch(grads[v], etas[v]) / self.rho_sq[v]
                if self.cl_mpi.rank == 0:
                    top -= self.redot_batch(grads['prb'], etas['prb']) / self.rho_sq['prb']
                bottom = self.hessian(vars, etas, etas)
                top, bottom = self.allreduce2(top, bottom)
                alpha = top / bottom
            else:
                if i == self.start_method and self._lbfgs_ls_init:
                    alpha = 1 / 16
                g0_eta = 0.0
                for v in ["obj", "pos"]:
                    g0_eta += self.redot_batch(grads[v], etas[v]) / self.rho_sq[v]
                if self.rank == 0:
                    g0_eta += self.redot_batch(grads['prb'], etas['prb']) / self.rho_sq['prb']
                g0_eta, _ = self.allreduce2(g0_eta, 0.0)
                if self._use_armijo:
                    alpha = self.line_search_armijo(vars, etas, alpha * 16, g0_eta)
                else:
                    alpha = self.line_search(vars, etas, alpha * 16, g0_eta)

            if self._use_scan:
                alpha = self.line_search_scan(vars, etas, alpha)

            # --- update ---
            for v in ["obj", "prb", "pos", "proj"]:
                self.linear_batch(vars[v], etas[v], 1, alpha)

            self.error_debug(vars, alpha, beta, i)
            self.vis_debug(vars, i, writer)

        vars["obj"] *= self.norm_const
        return vars

    def hessian(self, vars, grads, etas):
        """Hessian for the full functional, is a sum of 3 terms:
        1. main data fit term calcuated with the cascade rule,
        2. probe fit term,
        3. regularization term"""


        w = self.hessian_cascade(vars, grads, etas)
        if self.rank==0:
            w += self.hessian_prbfit(vars["prb"], grads["prb"], etas["prb"])

        return w

    @timer
    def hessian_cascade(self, vars, grads, etas):
        """"Cascade computation of the hessian for the main term,
            following the composition rule (Carlsson, 2025):
            For f = F1 ◦ F2 the hessian is
                d2f = dF1 ◦ d2F2 + d2F1 ◦ dF2
                where dF are differentials,
                d2F are second order terms.
            The function implements it for f = F0 ◦ F1 ◦ F2 ◦ F3 ...
            parameters to functions are unified as (x,y,z,w)
        """

        out = cp.zeros(1, dtype="float32")

        @self.gpu_batch(axis_out=0, axis_inp=0,nout=1)
        def _hessian_cascade(
            self, out, d,
            x2, y2, z2,
            x1, y1, z1,
            x0, y0, z0,
        ):
            # reorganize inputs into ordered lists for cascade traversal
            x = [x0, x1, x2]
            y = [y0, y1, y2]
            z = [z0, z1, z2]
            w = [None,None,None]

            # check whether y and z share same object (avoid duplicate work)
            y_is_z = y[0] is z[0]

            for id in range(1,len(self.F))[::-1]:
                # compute d2F(dFy,dFz)+dF(d2F(y,z))
                w = self.d2F_dF[id](x, y, z, w)

                # propagate differentials to the next level: fx, dF(x)(y)
                fx, y = self.dF[id](x, y)  # returns (fx, dfx(y))
                if y_is_z:
                    z = y
                else:
                    z = self.dF[id](x, z, return_x=False)  # returns dfx(z)
                x = fx

            # outer functional
            out[:] += self.d2F_dF[0](x, y, z, w, d)


        _hessian_cascade(
            self, out, self.data,
            vars["pos"], grads["pos"], etas["pos"],
            vars["proj"], grads["proj"], etas["proj"],
            vars["prb"], grads["prb"], etas["prb"],### reordered to keep syntax for the gpu_batch (last 4 are on gpu)
        )

        return out[0].get()

    def gradients(self, vars, grads):
        """Full gradient, consists of 3 terms:
        1. main data fit term calcuated with the cascade rule,
        2. probe fit term,
        """

        self.gradients_cascade(vars,grads)

        self.cl_mpi.redist(grads['proj'], self.proj_tmp,direction='backward')

        # part2, parallelization over object slices, formally gF4
        self.gF4(grads['obj'], self.proj_tmp)

        if self.rank==0:
            self.gradient_prbfit(grads["prb"], vars["prb"])

        ## copying to cpu before reduce for now
        grads['prb'][:] = cp.array(self.allreduce(grads['prb'].get()))

    @timer
    def gradients_cascade(self, vars, grads):
        """Cascade gradient for the main term
            following the composition rule (Carlsson, 2025):
            For f = F1 ◦ F2 the gradient is
                gradf = dF_2^*(\nabla F_1)),
                where dF_2^* is the adjoint to the differential
            The function implements it for f = F0 ◦ F1 ◦ F2 ◦ F3 ...
            parameters to functions are unified as (x,y,z)
        """

        # part1, parallelization over angles
        grads['prb'][:] = 0
        @self.gpu_batch(axis_out=0, axis_inp=0, nout=3)
        def _gradients_cascade(self,gradproj,gradpos,gradprb,d,proj,pos,prb):

            x = [prb, proj, pos]
            y = d
            # compute gradient by applying operators in forward order
            for id in range(len(self.gF)):  #last one computed separately because of different chunking
                y = self.gF[id](x,y)
            gradprb[:] += y[0]
            gradproj[:] = y[1]
            gradpos[:] = y[2]

        _gradients_cascade(self,grads['proj'],grads['pos'],grads['prb'],self.data,vars["proj"],vars["pos"],vars["prb"])

    @timer
    def gF4(self, gradu, gradproj):
        @self.gpu_batch(axis_out=0, axis_inp=1,nout=1)
        def _gF4(self, gradu, gradproj):
            gradu[:] = self.cl_tomo.RT(gradproj)
        _gF4(self, gradu, gradproj)

    #### probe fit term
    @timer
    def gradient_prbfit(self, grad_prb, prb):
        """Gradient with respect to the term
        lam_prbfit|||Dprb|-ref||_2^2"""

        if self.lam_prbfit == 0:
            return
        for j in range(self.ndist):
            tmp = self.cl_prop.D(prb[j : j + 1], j)
            td = self.ref[j : j + 1] * (tmp / (cp.abs(tmp)))
            grad_prb[j : j + 1] += self.lam_prbfit / self.prb_size * self.cl_prop.DT(2 * (tmp - td), j)

    @timer
    def hessian_prbfit(self, prb, dprb1, dprb2):
        """Hessian with respect to the term
        lam_prbfit|||Dprb|-ref||_2^2"""

        if self.lam_prbfit == 0:
            return 0

        out = 0
        for j in range(self.ndist):
            Dprb   = self.cl_prop.D(prb[j : j + 1], j)
            Ddprb1 = self.cl_prop.D(dprb1[j : j + 1], j)
            Ddprb2 = self.cl_prop.D(dprb2[j : j + 1], j)
            l0 = Dprb / (cp.abs(Dprb))
            d0 = self.ref[j : j + 1] / (cp.abs(Dprb))
            v1 = cp.sum((1 - d0) * reprod(Ddprb1, Ddprb2))
            v2 = cp.sum(d0 * reprod(l0, Ddprb1) * reprod(l0, Ddprb2))
            out += 2 * (v1 + v2)
        out = self.lam_prbfit * out / self.prb_size
        return out.get()### copy to cpu

    @timer
    def fwd_tomo(self, obj, out):
        """Forward tomography operator"""

        @self.gpu_batch(axis_out=1, axis_inp=0,nout=1)
        def _fwd_tomo(self, out, obj):
            out[:] = self.cl_tomo.R(obj)

        _fwd_tomo(self, out, obj)
        return out

    ####################### Functions for the cascade (following math notes for variables)
    # F* - functional
    # dF* - differential
    # d2F* - second order term for hessian
    # gF* - gradient
    #######################################################################################


    ####### F0(x0) = 1/n\||x0|-d\|_2^2
    def F0(self, x, d):
        """In: (x0), Out: const"""

        @cp.fuse()
        def F0_fused(x, d):
            t = cp.abs(x) - d
            return t*t
        return 1/ self.data_size * cp.sum(F0_fused(x, d))

    def dF0(self, x, y, d, return_x=False):
        """In: (x0,y0), Out: const"""

        @cp.fuse()
        def dF0_fused(x, d):
            return (x - d * (x / cp.abs(x)))
        return 2 / self.data_size * redot(dF0_fused(x, d), y)

    def d2F_dF0(self, x, y, z, w, d):
        """In: (x0,y0,z0,w0), Out: const"""

        @cp.fuse()
        def d2F_dF0_fused(x, y, z, w, d):
            absval = cp.abs(x)
            l0 = x / absval
            d0 = d / absval
            v = (1 - d0) * reprod(y, z) + d0 * reprod(l0, y) * reprod(l0, z)
            if w is not None:
                v += reprod(x - d * l0, w)
            return v

        return 2 / self.data_size * cp.sum(d2F_dF0_fused(x, y, z, w, d))

    def gF0(self, x, y):
        """In: x, y = F0(F1(..(x)))), Out: y0"""

        # calc fwd starting from 1
        for id in range(1, 4)[::-1]:
            x = self.F[id](x)

        @cp.fuse()
        def gF0_fused(x,y):
            td = y * (x / (cp.abs(x)))
            y0 = (2 / self.data_size) * (x - td)
            return y0

        return gF0_fused(x,y)

    ####### x0 = F1(x11,x12) = D(x11\cdot x12)
    def F1(self, x):
        """In: (x11,x12), Out: x0"""

        x11, x12 = x

        x0 = cp.empty([x12.shape[0], self.ndist, self.nz, self.n], dtype="complex64")
        for j in range(self.ndist):
            x0[:, j] = self.cl_prop.D(x11[j] * x12[:, j], j)

        return x0

    def dF1(self, x, y, return_x=True):
        """In: (x11,x12),(y11,y12) Out: y0"""

        x11, x12 = x
        y11, y12 = y

        y0 = y11[None] * x12 + x11[None] * y12
        for j in range(self.ndist):
            y0[:, j] = self.cl_prop.D(y0[:, j], j)

        if return_x:
            x0 = x11[None] * x12
            for j in range(self.ndist):
                x0[:, j] = self.cl_prop.D(x0[:, j], j)

        return (x0, y0) if return_x else y0

    def d2F_dF1(self, x, y, z, w):
        """In: (x11,x12),(y11,y12),(z11,z12) Out: y0"""

        x11, x12 = x
        y11, y12 = y
        z11, z12 = z
        w11, w12 = w

        if y12 is z12:
            y0 = 2 * y11[None] * y12
        else:
            y0 = y11[None] * z12 + z11[None] * y12

        if w11 is not None:
            y0 += w11[None] * x12
        if w12 is not None:
            y0 += x11[None] * w12

        for j in range(self.ndist):
            y0[:, j] = self.cl_prop.D(y0[:, j], j)

        return y0

    def gF1(self, x, y):
        """In: x=(x01,x02,x03),(y0) Out: y11,y12"""

        y0 = y

        # calc fwd starting from 2
        for id in range(2, 4)[::-1]:
            x = self.F[id](x)

        x11, x12 = x
        y11 = cp.zeros([self.ndist, self.nz, self.n], dtype="complex64")
        y12 = cp.empty([y0.shape[0], self.ndist, self.nz, self.n], dtype="complex64")
        for j in range(self.ndist):
            y12[:,j] = self.cl_prop.DT(y0[:, j], j)

        y11 = cp.sum(y12 * np.conj(x12), axis=0)
        y12 *= np.conj(x11[None])
        return y11, y12

    ######## (x11,x12) = F2(x21,x22) = (x21,e^{1j x22})
    def F2(self, x):
        """In: (x21,x22) Out: (x11,x12)"""

        x21, x22 = x
        x11 = x21
        @cp.fuse()
        def F2_fused(x22):
            return cp.exp(1j*x22)
        x12 = F2_fused(x22)
        return x11, x12

    def dF2(self, x, y, return_x=True):
        """In: (x21,x22),(y21,y22) Out: (x11,x12),(y11,y12)"""

        x21, x22 = x
        y21, y22 = y

        @cp.fuse()
        def dF2_fused(x22,y22):
            x12 = cp.exp(1j*x22)
            y12 = x12 * 1j * y22
            return x12,y12
        x12, y12 = dF2_fused(x22,y22)
        x11 = x21
        y11 = y21

        return ([x11, x12], [y11, y12]) if return_x else [y11, y12]



    def d2F_dF2(self, x, y, z, w):
        """In: (x21,x22),(y21,y22),(z21,z22),(w21,w22) Out: (y11,y12)"""

        x21, x22 = x
        y21, y22 = y
        z21, z22 = z
        w21, w22 = w

        @cp.fuse()
        def d2F_dF2(x22,y22,z22,w22):
            if y22 is z22:
                y12 = x22 * (-y22**2)
            else:
                y12 = x22 * (-y22 * z22)

            if w22 is not None:
                y12 = y12 + cp.exp(1j*x22) * 1j * w22
            return y12

        y12 = d2F_dF2(x22,y22,z22,w22)
        y11 = w21

        return [y11, y12]

    def gF2(self,x,y):
        """In: x(x01, x02, x03) ,(y11,y12) Out: (y21,y22)"""

        y11, y12 = y

        # calc fwd starting from 3
        for id in range(3, 4)[::-1]:
            x = self.F[id](x)
        x21,x22 = x

        @cp.fuse()
        def gF2_fused(x22,y12):
            y22 = (-1j) * y12 * cp.conj(cp.exp(1j*x22))
            return y22

        y22 = gF2_fused(x22,y12)
        y22 = y22.real if self.obj_dtype == 'float32' else y22

        y21 = y11
        return [y21, y22]

    ####### (x21,x22) = F3(x31,x32,x33) = (x31,S_{x_33}(x32))
    def F3(self, x):
        """In: (x31, x32, x33)  Out: (x21,x22)"""

        x31, x32, x33 = x

        x22 = cp.empty([len(x33), self.ndist, self.nz, self.n], dtype=self.obj_dtype)
        c = self.cl_shift.coeff(x32)
        for k in range(self.ndist):
            x22[:, k] = self.cl_shift.curlySc(c, x33[:, k], k)

        x21 = x31
        return [x21, x22]

    def dF3(self, x, y, return_x=True):
        """In: (x31, x32, x33),(y31, y32, y33)  Out: (y31, y22)"""

        x31, x32, x33 = x
        y31, y32, y33 = y

        y22 = cp.zeros([len(y32), self.ndist, self.nz, self.n], dtype=self.obj_dtype)
        c  = self.cl_shift.coeff(x32)
        c1 = self.cl_shift.coeff(y32)
        if return_x:
            x22 = cp.zeros([len(x32), self.ndist, self.nz, self.n], dtype=self.obj_dtype)
            for k in range(self.ndist):
                x22[:, k] = self.cl_shift.curlySc(c, x33[:, k], k)

        for k in range(self.ndist):
            y22[:, k] = self.cl_shift.dcurlySc(c, x33[:, k], k, c1, y33[:, k])

        x21 = x31
        y21 = y31
        return ([x21, x22], [y21, y22]) if return_x else [y21, y22]



    def d2F_dF3(self, x, y, z, w):
        """In: (x31, x32, x33),(y31, y32, y33),(z31, z32, z33),(w31, w32, w33)  Out: (y21, y22)"""

        x31, x32, x33 = x
        y31, y32, y33 = y
        z31, z32, z33 = z
        w31, w32, w33 = w

        y22 = cp.zeros([len(y32), self.ndist, self.nz, self.n], dtype=self.obj_dtype)
        c  = self.cl_shift.coeff(x32)
        cy = self.cl_shift.coeff(y32)
        cz = self.cl_shift.coeff(z32)
        for k in range(self.ndist):
            y22[:, k] = self.cl_shift.d2curlySc(c, x33[:, k], k, cy, y33[:, k], cz, z33[:, k])

        if w32 is not None:
            cy = self.cl_shift.coeff(w32)
            for k in range(self.ndist):
                y22[:, k] += self.cl_shift.dcurlySc(c, x33[:, k], k, cy, w33[:, k])

        y21 = w31

        return [y21, y22]

    def gF3(self, x, y):
        """In: x(x01, x02, x03) ,(y21,y22) Out: (y31,y32)"""

        y21, y22 = y

        for id in range(4, 4)[::-1]:
            x = self.F[id](x)
        x31, x32, x33 = x

        y32 = cp.zeros([y22.shape[0], self.nzobj, self.nobj], dtype=self.obj_dtype)
        y33 = cp.empty([y22.shape[0], self.ndist, 2], dtype="float32")
        c = self.cl_shift.coeff(x32)
        for k in range(self.ndist):
            Deltapsi, Deltar = self.cl_shift.dcurlySadjc(c, x33[:, k], k, y22[:, k])
            y32[:] += Deltapsi
            y33[:, k] = Deltar

        y32[:] = self.cl_shift.coeff(y32)

        y31 = y21
        return [y31, y32, y33]

    @timer
    def min(self, prb, obj, pos, proj):
        out = cp.zeros(1, dtype="float32")

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _min(self, out, proj, pos, data, prb):
            x = [prb, proj, pos]
            y = x
            for id in range(1, len(self.F))[::-1]:
                y = self.F[id](y)
            out[:] += self.F0(y, data)

        _min(self, out, proj, pos, self.data, prb)

        out = out[0]

        if self.rank == 0:
            for j in range(self.ndist):
                Dprb = self.cl_prop.D(prb[j : j + 1], j)[0]
                out += self.lam_prbfit / self.prb_size * cp.linalg.norm(cp.abs(Dprb) - self.ref[j]) ** 2
        return self.allreduce(out.get())

    def vis_debug(self, vars, i,writer=None):
        """Save reconstruction checkpoint to HDF5."""
        if not (i % self.vis_step == 0 and self.vis_step != -1):
            return
        if writer is not None:
            if i > self.start_iter:
                writer.write_checkpoint(vars, i, self.norm_const)
        else:
            mshow_complex(vars['obj'][self.local_nzobj//2],True)
            mshow_polar(vars['prb'][0],True)
            mshow_pos(vars['pos']-self.pos_init,True)


    def error_debug(self, vars, alpha, beta, i):
        """Visualization and data saving"""
        if not (i % self.err_step == 0 and self.err_step != -1):
            return

        err = self.min(vars["prb"], vars["obj"], vars["pos"], vars["proj"])
        if self.rank==0:
            if i==-1:
                logger.warning(f"Initial {err=:1.5e} ")
                self.table.loc[len(self.table)] = [i, beta, alpha,err, 0]
            else:
                ittime = time.time()-self.time_start
                logger.warning(f"iter={i}: {ittime:.4f}sec {err=:1.5e} ")
                self.table.loc[len(self.table)] = [i, alpha, beta, err, ittime]
            self.time_start = time.time()
            if hasattr(self, 'path_out'):
                name = f"{self.path_out}/conv.csv"
                os.makedirs(os.path.dirname(name), exist_ok=True)
                self.table.to_csv(name, index=False)

    def gen_sqrt_data(self, vars, out):
        """Generate synthetic data"""

        vars["obj"] /= self.norm_const
        self.fwd_tomo(vars["obj"],out = self.proj_tmp)
        self.redist(self.proj_tmp, vars['proj'])
        @self.gpu_batch(axis_out=0, axis_inp=0,nout=1)
        def _gen_data(self, out, proj, pos, prb):
            x = [prb, proj, pos]
            y = x  # forming output
            # compute functional by applying operators in reverse order
            for id in range(1, len(self.F))[::-1]:
                y = self.F[id](y)
            out[:] = cp.abs(y)
        _gen_data(self, out, vars['proj'], vars['pos'], vars['prb'])
        vars["obj"] *= self.norm_const

    def gen_sqrt_ref(self, prb, out):
        """Generate synthetic reference"""
        for j in range(self.ndist):
            out[j] = cp.abs(self.cl_prop.D(prb[j : j + 1], j)[0])

    ############### polak ribiera
    def beta_pr(self, gk, gk1):
        """
        Polak–Ribière beta for real or complex gradients.
        Uses Hermitian inner products; returns a real scalar.
        """
        top = 0
        bottom = 0
        for v in ['pos', 'obj']:
            y = gk1[v] - gk[v]
            top    += self.redot_batch(gk1[v], y)      # Re<g_{k+1}, g_{k+1}-g_k>
            bottom += self.redot_batch(gk[v],  gk[v])  # Re<g_k, g_k>
        if self.rank == 0:
            y = gk1['prb'] - gk['prb']
            top    += self.redot_batch(gk1['prb'], y)
            bottom += self.redot_batch(gk['prb'],  gk['prb'])
        top, bottom = self.allreduce2(top, bottom)
        return top / bottom

    def beta_pr_plus(self,gk, gk1):
        """PR+ safeguard: max(beta_PR, 0)."""
        return max(0.0, self.beta_pr(gk, gk1))

    ############### L-BFGS (method 3)

    def _dot(self, a, b):
        """MPI-aware inner product: obj+pos on all ranks, prb on rank 0 only."""
        d = 0.0
        for v in ['obj', 'pos']:
            d += self.redot_batch(a[v], b[v])
        if self.rank == 0:
            d += self.redot_batch(a['prb'], b['prb'])
        d, _ = self.allreduce2(d, 0.0)
        return d

    def _lbfgs_push(self, grads, grads1, alpha, curvature_eps=1e-10):
        """Store s = alpha*etas, y = grads-grads1.

        Skips the pair if the curvature condition ys > eps*||y||*||s|| is not met
        (relative threshold from the reference implementation).
        """
        m = self.lbfgs_m
        t = self.lbfgs_t
        # write s and y to slot first, then check curvature
        for v in ['obj', 'prb', 'pos']:
            self.mulc_batch(self.lbfgs_s[t][v], self.etas[v], alpha)
            self.linear_batch(self.lbfgs_y[t][v], grads[v],  0,  1)
            self.linear_batch(self.lbfgs_y[t][v], grads1[v], 1, -1)
        ys = self._dot(self.lbfgs_s[t], self.lbfgs_y[t])
        yy = self._dot(self.lbfgs_y[t], self.lbfgs_y[t])
        ss = self._dot(self.lbfgs_s[t], self.lbfgs_s[t])
        thresh = curvature_eps * max(1.0, float(np.sqrt(max(yy, 0.0))) * float(np.sqrt(max(ss, 0.0))))
        if ys <= thresh:
            if self.rank == 0:
                print(f"L-BFGS: skip pair (ys={ys:.3e} thresh={thresh:.3e})")
            return
        self.lbfgs_rho[t] = 1.0 / ys
        self.lbfgs_t = (t + 1) % m
        self.lbfgs_k += 1

    def _lbfgs_direction(self, grads, etas):
        """Standard L-BFGS two-loop (N&W Alg 7.4) using grads directly as q."""
        m = self.lbfgs_m
        k = min(self.lbfgs_k, m)
        t = self.lbfgs_t

        # q = grads
        for v in ['obj', 'prb', 'pos']:
            self.linear_batch(etas[v], grads[v], 0, 1.0)

        # first loop (backward)
        alpha_l = np.zeros(k)
        for j in range(k - 1, -1, -1):
            idx = (t - k + j) % m
            a_j = self.lbfgs_rho[idx] * self._dot(self.lbfgs_s[idx], etas)
            alpha_l[j] = a_j
            for v in ['obj', 'prb', 'pos']:
                self.linear_batch(etas[v], self.lbfgs_y[idx][v], 1, -a_j)

        # H0 = gamma*I,  gamma = <s,y> / <y,y>  (with safety checks)
        if k > 0:
            idx_last = (t - 1) % m
            sy = 1.0 / self.lbfgs_rho[idx_last]
            yy = self._dot(self.lbfgs_y[idx_last], self.lbfgs_y[idx_last])
            gamma = sy / yy if yy > 1e-20 else 1.0
            if not np.isfinite(gamma) or gamma <= 0:
                if self.rank == 0:
                    print(f"L-BFGS: bad gamma={gamma:.3e}, resetting to 1.0")
                gamma = 1.0
            for v in ['obj', 'prb', 'pos']:
                self.mulc_batch(etas[v], etas[v], gamma)

        # second loop (forward)
        for j in range(k):
            idx = (t - k + j) % m
            b_j = self.lbfgs_rho[idx] * self._dot(self.lbfgs_y[idx], etas)
            for v in ['obj', 'prb', 'pos']:
                self.linear_batch(etas[v], self.lbfgs_s[idx][v], 1, alpha_l[j] - b_j)

        # descent direction = -H_k * g
        for v in ['obj', 'prb', 'pos']:
            self.mulc_batch(etas[v], etas[v], -1)

    def line_search_scan(self, vars, etas, alpha, n_scan=21, lo=0.5, hi=1.5):
        """Exact line search by scanning n_scan alpha values in [lo*alpha, hi*alpha].

        Returns the alpha with the lowest objective value.
        Uses self.grads1 as trial buffer (self.grads is preserved).
        """
        f_best = np.inf
        alpha_best = alpha
        for a in np.linspace(lo * alpha, hi * alpha, n_scan):
            for v in ["obj", "prb", "pos", "proj"]:
                self.grads1[v][:] = vars[v] + a * etas[v]
            f_a = self.min(self.grads1['prb'], self.grads1['obj'],
                           self.grads1['pos'], self.grads1['proj'])
            if self.rank == 0:
                print(f"ScanLS alpha={a:.3e}  f={f_a:.6e}")
            if f_a < f_best:
                f_best = f_a
                alpha_best = a
        return alpha_best

    def line_search_armijo(self, vars, etas, alpha, g0_eta, c1=1e-4, rho=0.5, max_iter=50):
        """Backtracking Armijo line search: sufficient decrease only, no curvature check."""
        f0 = self.min(vars['prb'], vars['obj'], vars['pos'], vars['proj'])
        for _ in range(max_iter):
            for v in ["obj", "prb", "pos", "proj"]:
                self.grads1[v][:] = vars[v] + alpha * etas[v]
            f_a = self.min(self.grads1['prb'], self.grads1['obj'],
                           self.grads1['pos'], self.grads1['proj'])
            if f_a <= f0 + c1 * alpha * g0_eta:
                return alpha
            alpha *= rho
        return alpha

    def line_search(self, vars, etas, alpha, g0_eta, c1=1e-4, c2=0.9, max_iter=30):
        """Strong Wolfe line search (N&W Algorithm 3.5/3.6).

        phi  → trial point in self.grads1.
        dphi → gradient in self.grads2 (self.grads = g_k preserved).
        zoom and helpers are inner closures following lbfgs_minimal.ipynb.
        """
        f0 = self.min(vars['prb'], vars['obj'], vars['pos'], vars['proj'])
        alpha_max = alpha * 10.0

        def phi(a):
            for v in ["obj", "prb", "pos", "proj"]:
                self.grads1[v][:] = vars[v] + a * etas[v]
            return self.min(self.grads1['prb'], self.grads1['obj'],
                            self.grads1['pos'], self.grads1['proj'])

        def dphi():
            # uses self.grads1 set by the preceding phi(a) call
            self.gradients(self.grads1, self.grads2)
            return self._dot(self.grads2, etas)

        def phi_dphi(a):
            f = phi(a)
            if not np.isfinite(f):
                return f, float('nan')
            g = dphi()
            return f, g

        if g0_eta >= 0:
            if self.rank == 0:
                print(f"WolfeLS: non-descent g0_eta={g0_eta:.3e}, skipping step")
            return 0.0

        def cubic_minimizer(a0, f0_, g0_, a1, f1, g1):
            if a0 == a1:
                return None
            d1 = g0_ + g1 - 3.0 * (f0_ - f1) / (a0 - a1)
            disc = d1 * d1 - g0_ * g1
            if disc < 0.0:
                return None
            d2 = float(np.sqrt(disc))
            den = g1 - g0_ + 2.0 * d2
            if den == 0.0:
                return None
            return a1 - (a1 - a0) * (g1 + d2 - d1) / den

        def quad_minimizer(a0, f0_, g0_, a1, f1):
            da = a1 - a0
            if da == 0.0:
                return None
            den = 2.0 * (f1 - f0_ - g0_ * da)
            if den == 0.0:
                return None
            return a0 - (g0_ * da * da) / den

        def choose_trial(a_lo, f_lo, g_lo, a_hi, f_hi, g_hi):
            lo, hi = (a_lo, a_hi) if a_lo < a_hi else (a_hi, a_lo)
            w = hi - lo
            if w <= 0.0:
                return lo
            lo_safe = lo + 0.1 * w
            hi_safe = hi - 0.1 * w
            a = cubic_minimizer(a_lo, f_lo, g_lo, a_hi, f_hi, g_hi)
            if a is None or not np.isfinite(a) or a <= lo_safe or a >= hi_safe:
                a = quad_minimizer(a_lo, f_lo, g_lo, a_hi, f_hi)
            if a is None or not np.isfinite(a) or a <= lo_safe or a >= hi_safe:
                a = 0.5 * (lo + hi)
            return float(np.clip(a, lo_safe, hi_safe))

        def zoom(a_lo, a_hi, phi_lo, dphi_lo, phi_hi, dphi_hi):
            if a_lo > a_hi:
                a_lo, a_hi = a_hi, a_lo
                phi_lo, phi_hi = phi_hi, phi_lo
                dphi_lo, dphi_hi = dphi_hi, dphi_lo
            for iss in range(max_iter):
                if self.rank == 0:
                    logger.info(f"{iss=}")
                a_j = choose_trial(a_lo, phi_lo, dphi_lo, a_hi, phi_hi, dphi_hi)
                phi_j, dphi_j = phi_dphi(a_j)
                if not np.isfinite(phi_j):
                    a_hi, phi_hi, dphi_hi = a_j, phi_j, dphi_j
                    continue
                if (phi_j > f0 + c1 * a_j * g0_eta) or (phi_j >= phi_lo):
                    a_hi, phi_hi, dphi_hi = a_j, phi_j, dphi_j
                else:
                    if np.isfinite(dphi_j) and abs(dphi_j) <= -c2 * g0_eta:
                        return a_j
                    if not np.isfinite(dphi_j) or dphi_j * (a_hi - a_lo) >= 0.0:
                        a_hi, phi_hi, dphi_hi = a_lo, phi_lo, dphi_lo
                    a_lo, phi_lo, dphi_lo = a_j, phi_j, dphi_j
                if abs(a_hi - a_lo) <= 1e-16 * max(1.0, abs(a_lo), abs(a_hi)):
                    return a_lo
            return (a_lo + a_hi) / 2.0

        a_prev = 0.0
        phi_prev = f0
        dphi_prev = g0_eta
        a = float(alpha)
        if not np.isfinite(a) or a <= 0.0:
            a = 1.0
        a = min(a, alpha_max)

        for i in range(1, max_iter + 1):
            phi_a, dphi_a = phi_dphi(a)
            if self.rank == 0:
                print(f"WolfeLS[{i}] alpha={a:.3e}  f={phi_a:.6e}")
            if (not np.isfinite(phi_a) or
                    phi_a > f0 + c1 * a * g0_eta or
                    (i > 1 and phi_a >= phi_prev)):
                return zoom(a_prev, a, phi_prev, dphi_prev, phi_a, dphi_a)
            if np.isfinite(dphi_a) and abs(dphi_a) <= -c2 * g0_eta:
                return a
            if not np.isfinite(dphi_a) or dphi_a >= 0.0:
                return zoom(a_prev, a, phi_prev, dphi_prev, phi_a, dphi_a)
            a_prev, phi_prev, dphi_prev = a, phi_a, dphi_a
            a = min(2.0 * a, alpha_max)

        return a if phi(a) < f0 else 0.0
