import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import os
import warnings
import pandas as pd
import cupyx.scipy.ndimage as ndimage
from datetime import datetime

from .tomo import Tomo
from .propagation import Propagation
from .shift import Shift
from .chunking import Chunking 
from .utils import *

np.set_printoptions(legacy="1.25")
warnings.filterwarnings("ignore", message=f".*peer.*")


class Rec:
    def __init__(self, args):
        # copy args to elements of the class
        for key, value in vars(args).items():
            setattr(self, key, value)        

        # list of functionals, gradients, differentials, and second-order differentials
        self.F = [self.F0, self.F1, self.F2, self.F3]
        self.gF = [self.gF0, self.gF1, self.gF2, self.gF31, self.gF32]
        self.dF = [self.dF0, self.dF1, self.dF2, self.dF3]
        self.d2F = [self.d2F0, self.d2F1, self.d2F2, self.d2F3]
        self.noper = len(self.F)

        # estimate memory footprint for pinned + device buffer per GPU (complex64)
        multiplier = 16  # related to the number of arrays, experimentally chosen. the scheme will diverge if too low
        complex_item = np.dtype("complex64").itemsize
        max_dim = max(self.nzobj, self.ntheta)
        nbytes = int(multiplier * self.nchunk * self.nobj * max_dim * complex_item)

        # X-ray propagation and magnification parameters for classes
        wavelength = 1.24e-09 / args.energy
        z2 = args.focustodetectordistance - args.z1
        magnifications = args.focustodetectordistance / args.z1
        norm_magnifications = magnifications / magnifications[0]
        distance = (args.z1 * z2) / args.focustodetectordistance * norm_magnifications**2
        voxelsize = args.detector_pixelsize / magnifications[0]
        self.rho_sq = {'obj':args.rho[0]**2,'prb':args.rho[1]**2,'pos':args.rho[2]**2}

        # create classes
        self.cl_chunking = Chunking(nbytes, self.nchunk, self.ngpus)
        self.cl_tomo = [None for _ in range(self.ngpus)]
        self.cl_prop = [None for _ in range(self.ngpus)]
        self.cl_shift = [None for _ in range(self.ngpus)]

        
        
        for igpu in range(self.ngpus):
            with cp.cuda.Device(igpu):
                # initialize processing classes per gpu
                self.cl_tomo[igpu] = Tomo(self.nobj, self.theta, self.mask)
                self.cl_prop[igpu] = Propagation(self.n, self.nz, self.ndist, wavelength, voxelsize, distance)
                self.cl_shift[igpu] = Shift(self.n, self.nobj, self.nz, self.nzobj, 1.0 / norm_magnifications)

        # preallocate memory for the gradient and conjugate direction
        self.grads, self.etas = {}, {}
        for ge in self.grads, self.etas:
            ge["obj"] = np.empty([self.nzobj, self.nobj, self.nobj], dtype=self.obj_dtype)
            ge["prb"] = cp.empty([self.ndist, self.nz, self.n], dtype="complex64")
            ge["pos"] = np.empty([self.ntheta, self.ndist, 2], dtype="float32")
            ge["proj"] = np.empty([self.ntheta, self.nzobj, self.nobj], dtype=self.obj_dtype)

        # normalization constant to address work with normal operators
        self.norm_const = np.float32(np.sqrt(self.nobj / self.ntheta))
        
        # save convergence results
        self.table=pd.DataFrame(columns=["iter", "err", "time"])    

        # sizes for normalization
        self.data_size = self.ntheta * self.ndist * self.nz * self.n
        self.prb_size = self.ndist * self.nz * self.n
        self.obj_size = self.nzobj * self.nobj**2

        # fast refs
        self.gpu_batch = self.cl_chunking.gpu_batch
        self.redot_batch = self.cl_chunking.redot_batch
        self.linear_batch = self.cl_chunking.linear_batch
        self.mulc_batch = self.cl_chunking.mulc_batch

    def BH(self, data, ref, vars):

        # keep data and initial shifts in class
        self.data = data
        self.ref = ref
        self.pos_init = vars["pos"].copy()

        # refs to preallocated memory for gradients
        grads = self.grads
        etas = self.etas

        # normalize to work with normal operators (do this once, restore in finally)
        vars["obj"] /= self.norm_const
        # precalculate ewave
        vars["ewave"] = self.fwd_tomo(vars["obj"], exp=True)
        
        for i in range(self.start_iter,self.niter):
            # error and visualization debug
            self.error_debug(vars, i)
            self.vis_debug(vars, i)

            # compute gradients
            self.gradients(vars, grads)
            
            # scale
            for v in ["obj", "prb", "pos"]:
                self.mulc_batch(grads[v], grads[v], self.rho_sq[v])
            # keep projections in memory
            
            self.fwd_tomo(grads["obj"], out=grads["proj"])
            
            if i == self.start_iter:
                # initial search direction (negative gradient)
                for v in ["obj", "prb", "pos"]:
                    self.mulc_batch(etas[v], grads[v], -1)
            else:
                # calc beta using Hessian-weighted inner products
                beta = self.hessian(vars, grads, etas) / self.hessian(vars, etas, etas)
                # update search direction: eta = beta * previous_eta - grad
                for v in ["obj", "prb", "pos"]:
                    self.linear_batch(etas[v], grads[v], beta, -1)
            
            # keep projections in memory            
            self.fwd_tomo(etas["obj"], out=etas["proj"])
            
            # calc alpha (step length)            
            top = 0
            for v in ["obj", "prb", "pos"]:
                top -= self.redot_batch(grads[v], etas[v]) / self.rho_sq[v]
            
            bottom = self.hessian(vars, etas, etas)            
            
            alpha = top / bottom
            
            # check approximation with the Hessian
            self.check_approximation(vars, etas, top, bottom, alpha, i)

            # update variables: var = var+alpha*eta
            for v in ["obj", "prb", "pos"]:
                self.linear_batch(vars[v], etas[v], 1, alpha)
            
            
            # update ewave for current u            
            self.fwd_tomo(vars["obj"], exp=True, out=vars["ewave"])
                        
        # normalize back
        vars["obj"] *= self.norm_const
        

        return vars

    def hessian(self, vars, grads, etas):
        """Hessian for the full functional, is a sum of 3 terms:
        1. main data fit term calcuated with the cascade rule,
        2. probe fit term,
        3. regularization term"""

        w = self.hessian_cascade(vars, grads, etas)
        w += self.hessian_prbfit(vars["prb"], grads["prb"], etas["prb"])
        w += self.hessian_reg(vars["obj"], grads["obj"], etas["obj"])
        return w

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

        # allocate per-GPU accumulators (float32 scalars)
        out = [None]*self.ngpus
        for igpu in range(self.ngpus):
            with cp.cuda.Device(igpu):
                out[igpu] = cp.zeros(1, dtype="float32")

        @self.gpu_batch(axis_out=0, axis_inp=0,nout=1)
        def _hessian_cascade(
            self, out, d,
            x1, y1, z1, 
            x2, y2, z2, 
            x0, y0, z0, 
        ):
            # reorganize inputs into ordered lists for cascade traversal
            x = [x0, x1, x2]
            y = [y0, y1, y2]
            z = [z0, z1, z2]
            w = [cp.zeros_like(x0),cp.zeros_like(x1),cp.zeros_like(x2)]
            
            for id in range(self.noper)[::-1]:
                # compute second derivative and first differentials for this level
                if id == 0:
                    d2f1 = self.d2F[id](x, y, z, d)
                    d2f2 = self.dF[id](x, w, d, return_x=False)
                else:
                    d2f1 = self.d2F[id](x, y, z)
                    d2f2 = self.dF[id](x, w, return_x=False)
                    
                # reuse d2f1 as working accumulator and add contribution from d2f2
                w = d2f1
                if isinstance(w, list):
                    for iv in range(len(w)):
                        w[iv] += d2f2[iv]
                else:
                    w += d2f2
                d2f2 = []  # clear memory
                # last iteration produces scalar contribution -> accumulate and exit
                if id == 0:
                    break
                    
                # check whether y and z share same object (avoid duplicate work)
                y_is_z = y[0] is z[0]

                # propagate differentials to the next level: fx, dF(x)(y)
                fx, y = self.dF[id](x, y)  # returns (fx, dfx(y))
                if y_is_z:
                    z = y
                else:
                    z = self.dF[id](x, z, return_x=False)  # returns dfx(z)
                x = fx

            # accumulate scalar result into per-GPU accumulator
            out[:] += w

        _hessian_cascade(
            self, out, self.data,
            vars["pos"], grads["pos"], etas["pos"],
            vars["ewave"], grads["proj"], etas["proj"],
            vars["prb"], grads["prb"], etas["prb"],### reordered to keep syntax for the gpu_batch (last 4 are on gpu)
        )
        # reduce per-GPU accumulators to a single scalar
        out = sum(out)[0]
        return out.get()### copy to cpu

    def gradients(self, vars, grads):
        """Full gradient, consists of 3 terms: 
        1. main data fit term calcuated with the cascade rule,
        2. probe fit term,
        3. regularization term"""

        grads["prb"], grads["obj"], grads["pos"] = self.gradients_cascade(vars)
        self.gradient_prbfit(grads["prb"], vars["prb"])
        self.gradient_reg(grads["obj"], vars["obj"])
        
    def gradients_cascade(self, vars):
        """Cascade gradient for the main term
            following the composition rule (Carlsson, 2025):
            For f = F1 ◦ F2 the gradient is 
                gradf = dF_2^*(\nabla F_1)),
                where dF_2^* is the adjoint to the differential
            The function implements it for f = F0 ◦ F1 ◦ F2 ◦ F3 ...
            parameters to functions are unified as (x,y,z)
        """

        x = [vars["prb"], vars["obj"], vars["pos"], vars["ewave"]]  # assume ewave is precalculated
        
        # part1, parallelization over angles
        gradprb = [None]*self.ngpus
        for igpu in range(self.ngpus):
            with cp.cuda.Device(igpu):
                gradprb[igpu] = cp.zeros([self.ndist, self.nz, self.n], dtype="complex64")
        
        gradpos = np.empty([self.ntheta,self.ndist, 2], dtype="float32")
        gradewave = np.empty([self.ntheta, self.nzobj, self.nobj], dtype=self.obj_dtype)
        
        @self.gpu_batch(axis_out=0, axis_inp=0, nout=3)
        def _gradients_cascade(self,gradewave,gradpos,gradprb,d,ewave,pos,prb):
            x = [prb, ewave, pos]
            y = x  # forming output
            # compute functional by applying operators in reverse order
            for id in range(1, len(self.F) - 1)[::-1]:  ####dont take last one, ignore F3
                y = self.F[id](y)
            # compute gradient by applying operators in forward order
            y = self.gF[0](x, y, d)
            for id in range(1, len(self.gF) - 1):  
                y = self.gF[id](x,y)            
            gradprb[:] += y[0]
            gradewave[:] = y[1]
            gradpos[:] = y[2]
        _gradients_cascade(self,gradewave,gradpos,gradprb,self.data,vars["ewave"],vars["pos"],vars["prb"])
        gradprb = sum(gradprb[1:], gradprb[0])
    
        # part2, parallelization over object slices
        y = [gradprb,gradewave,gradpos]                
        y = self.gF[-1](x, y)
        return y        
    
    #### probe fit and regularization terms
    def gradient_prbfit(self, grad_prb, prb):
        """Gradient with respect to the term 
        lam_prbfit|||Dprb|-ref||_2^2"""
        
        if self.lam_prbfit == 0:
            return
        grad_prbfit = cp.zeros_like(prb)
        igpu = cp.cuda.Device().id
        for j in range(self.ndist):
            tmp = self.cl_prop[igpu].D(prb[j : j + 1], j)
            td = self.ref[j : j + 1] * (tmp / (cp.abs(tmp)))
            grad_prbfit[j : j + 1] += self.cl_prop[igpu].DT(2 * (tmp - td), j)
        grad_prb[:] += self.lam_prbfit / self.prb_size * grad_prbfit

    def gradient_reg(self, gu, u):
        """Gradient with respect to the regularization term (laplace)
        lam_reg ||Delta u||_2^2"""
        
        if self.lam_reg == 0:
            return
        # process by chunk to save memory
        step = self.nzobj // 8
        for k in range(int(np.ceil(self.nzobj / step))):
            st = k * step
            end = min((k + 1) * step, self.nzobj)
            st1 = max(st - 6, 0)
            end1 = min(end + 6, self.nzobj)
            gg = self.laplacian_sq(u[st1:end1])[st - st1 : end1 - st1 - (end1 - end)]
            self.linear_batch(gu[st:end], gg, 1, 2 * self.lam_reg / self.obj_size)

    def hessian_prbfit(self, prb, dprb1, dprb2):
        """Hessian with respect to the term 
        lam_prbfit|||Dprb|-ref||_2^2"""

        if self.lam_prbfit == 0:
            return 0

        igpu = cp.cuda.Device().id
        out = 0
        for j in range(self.ndist):
            Dprb = self.cl_prop[igpu].D(prb[j : j + 1], j)
            Ddprb1 = self.cl_prop[igpu].D(dprb1[j : j + 1], j)
            Ddprb2 = self.cl_prop[igpu].D(dprb2[j : j + 1], j)
            l0 = Dprb / (cp.abs(Dprb))
            d0 = self.ref[j : j + 1] / (cp.abs(Dprb))
            v1 = cp.sum((1 - d0) * reprod(Ddprb1, Ddprb2))
            v2 = cp.sum(d0 * reprod(l0, Ddprb1) * reprod(l0, Ddprb2))
            out += 2 * (v1 + v2)
        out = self.lam_prbfit * out / self.prb_size
        return out.get()### copy to cpu

    def hessian_reg(self, obj, dobj1, dobj2):
        """Hessian with respect to the regularization term (laplace)
        lam_reg ||Delta obj||_2^2"""
        
        if self.lam_reg == 0:
            return 0

        out = 0
        step = self.nzobj // 8
        for k in range(int(np.ceil(self.nzobj / step))):
            st = k * step
            end = min((k + 1) * step, self.nzobj)
            st1 = max(st - 4, 0)
            end1 = min(end + 4, self.nzobj)
            Lobj1 = self.laplacian(dobj1[st1:end1])[st - st1 : end1 - st1 - (end1 - end)]
            Lobj2 = self.laplacian(dobj2[st1:end1])[st - st1 : end1 - st1 - (end1 - end)]
            out += self.redot_batch(Lobj1, Lobj2)
        return 2 * self.lam_reg * out / self.obj_size

    def fwd_ewave(self, ewave, pos, prb):
        """Forward operator with precalculated ewave=exp(1j R(u))"""

        out = np.empty([self.ntheta, self.ndist, self.nz, self.n], dtype="complex64")

        @self.gpu_batch(axis_out=0, axis_inp=0,nout=1)
        def _fwd_ewave(self, out, ewave, pos, prb):
            x = [prb, ewave, pos]
            y = x  # forming output
            # compute functional by applying operators in reverse order
            for id in range(1, len(self.F) - 1)[::-1]:  ####dont take last one, ignore F3
                y = self.F[id](y)
            out[:] = y

        _fwd_ewave(self, out, ewave, pos, prb)
        return out

    def fwd_tomo(self, obj, exp=False, out=None):
        """Forward tomography operator"""

        if out is None:
            if exp:
                out = np.empty([self.ntheta, obj.shape[0], self.nobj], dtype="complex64")
            else:
                out = np.empty([self.ntheta, obj.shape[0], self.nobj], dtype=obj.dtype)

        @self.gpu_batch(axis_out=1, axis_inp=0,nout=1)
        def _fwd_tomo(self, out, obj):
            igpu = cp.cuda.Device().id
            out[:] = self.cl_tomo[igpu].R(obj)
            if exp:
                out[:] = cp.exp(1j * out)

        _fwd_tomo(self, out, obj)
        return out    

    ####################### Functions for the cascade (following math notes for variables) 
    # F* - functional
    # dF* - differential
    # d2F* - second order term for hessian
    # gF* - gradient    
    #######################################################################################


    ####### F0(x0) = 1/n\||x|-d\|_2^2
    def F0(self, x, d):
        return cp.linalg.norm(cp.abs(x) - d) ** 2 / self.data_size

    def dF0(self, x, y, d, return_x=False):
        tmp = 2 * (x - d * (x / cp.abs(x)))
        return redot(tmp, y) / self.data_size

    def d2F0(self, x, y, z, d):
        l0 = x / (cp.abs(x))
        d0 = d / (cp.abs(x))
        v1 = cp.sum((1 - d0) * reprod(y, z))
        v2 = cp.sum(d0 * reprod(l0, y) * reprod(l0, z))
        return 2 * (v1 + v2)/ self.data_size
    
    def gF0(self, x, y, d):
        td = d * (y / (cp.abs(y)))
        y11 = (2 / self.data_size) * (y - td) 
        return y11
    
    ####### F1(x21,x22) = D(x21\cdot x22)
    def F1(self, x):
        x21, x22 = x
        igpu = cp.cuda.Device().id
        out = cp.empty([x22.shape[0], self.ndist, self.nz, self.n], dtype="complex64")
        for j in range(self.ndist):
            out[:, j] = self.cl_prop[igpu].D(x21[j] * x22[:, j], j)
        return out

    def dF1(self, x, y, return_x=True):
        x21, x22 = x
        y21, y22 = y

        y1 = cp.empty([x22.shape[0], self.ndist, self.nz, self.n], dtype="complex64")
        x1 = cp.empty_like(y1) if return_x else None

        igpu = cp.cuda.Device().id
        for j in range(self.ndist):
            y1[:, j] = self.cl_prop[igpu].D(y21[j] * x22[:, j] + x21[j] * y22[:, j], j)
            if return_x:
                x1[:, j] = self.cl_prop[igpu].D(x21[j] * x22[:, j], j)

        return (x1, y1) if return_x else y1
    
    def d2F1(self, x, y, z):
        x21, x22 = x
        y21, y22 = y
        z21, z22 = z
        y1 = cp.empty([len(y22), self.ndist, self.nz, self.n], dtype="complex64")
        igpu = cp.cuda.Device().id
        if y22 is z22:
            for j in range(self.ndist):
                y1[:, j] = 2 * self.cl_prop[igpu].D(y21[j] * y22[:, j], j)
        else:
            for j in range(self.ndist):
                y1[:, j] = self.cl_prop[igpu].D(y21[j] * z22[:, j], j) + self.cl_prop[igpu].D(z21[j] * y22[:, j], j)

        return y1
    
    def gF1(self, x, y):
        x11, x12, x13 = x
        y11 = y
                    
        igpu = cp.cuda.Device().id
        y21 = cp.zeros([self.ndist,self.nz,self.n], dtype="complex64")
        y22 = cp.empty([y11.shape[0], self.ndist, self.nz, self.n], dtype="complex64")
        for j in range(self.ndist):
            mspsi = self.cl_shift[igpu].curlyS(x12, x13[:, j], j)
            tmp = self.cl_prop[igpu].DT(y11[:, j], j)             
            y21[j] += cp.sum(tmp* np.conj(mspsi), axis=0)
            y22[:, j] = tmp * np.conj(x11[j])
        return [y21,y22]        
        
    ####### F2(x32,x33) = S_{x_33}(x32)
    def F2(self, x):
        x31, x32, x33 = x
        x22 = cp.empty([len(x33), self.ndist, self.nz, self.n], dtype="complex64")
        igpu = cp.cuda.Device().id
        for k in range(self.ndist):
            x22[:, k] = self.cl_shift[igpu].curlyS(x32, x33[:, k], k)

        return [x31, x22]

    def dF2(self, x, y, return_x=True):
        x31, x32, x33 = x
        y31, y32, y33 = y

        y22 = cp.zeros([len(y32), self.ndist, self.nz, self.n], dtype="complex64")
        igpu = cp.cuda.Device().id

        if return_x:
            x22 = cp.zeros([len(x32), self.ndist, self.nz, self.n], dtype="complex64")
            for k in range(self.ndist):
                r = x33[:, k]
                x22[:, k] = self.cl_shift[igpu].curlyS(x32, r, k)

        for k in range(self.ndist):
            r = x33[:, k]
            Deltar = y33[:, k]
            y22[:, k] = self.cl_shift[igpu].dcurlyS(x32, r, k, y32, Deltar)

        return ([x31, x22], [y31, y22]) if return_x else [y31, y22]

    def d2F2(self, x, y, z):
        x31, x32, x33 = x
        y31, y32, y33 = y
        z31, z32, z33 = z
        y22 = cp.zeros([len(y32), self.ndist, self.nz, self.n], dtype="complex64")
        igpu = cp.cuda.Device().id
        for k in range(self.ndist):
            r = x33[:, k]
            Deltar_y = y33[:, k]
            Deltar_z = z33[:, k]
            y22[:, k] = self.cl_shift[igpu].d2curlyS(x32, r, k, y32, Deltar_y, z32, Deltar_z)

        return [cp.zeros_like(y31), y22]
    
    def gF2(self, x, y):        
        x21,x22, x23 = x
        y21, y22 = y

        y33 = cp.empty([y22.shape[0], self.ndist, 2], dtype="float32")
        y32 = cp.zeros([y22.shape[0], self.nzobj, self.nobj], dtype="complex64")
        igpu = cp.cuda.Device().id
        for k in range(self.ndist):
            Deltapsi, Deltar = self.cl_shift[igpu].dcurlySadj(x22, x23[:, k], k, y22[:, k])
            y32[:] += Deltapsi
            y33[:, k] = Deltar
        
        return [y21, y32, y33]
    
    ######## F3(x41,x42,x43)=(x41,e^{1j R(x42)},x43)
    def F3(self, x):

        x41, x42, x43, x44 = x
        # out = self.expR(self.cl_tomo[cp.cuda.Device().id].R(x42))
        out = x44
        return x41, out, x43

    def dF3(self, x, y, return_x=True):

        x41, x43, x44 = x
        y41, y43, y44 = y

        if return_x:
            x32 = x44  # already computed
        y32 = cp.zeros([len(y44), self.nzobj, self.nobj], dtype="complex64")

        y32[:] = x44 * 1j * y44

        return ([x41, x32, x43], [y41, y32, y43]) if return_x else [y41, y32, y43]

    def d2F3(self, x, y, z):
        x41, x43, x44 = x
        y41, y43, y44 = y
        z41, z43, z44 = z
        y32 = cp.empty([len(y44), self.nzobj, self.nobj], dtype="complex64")

        if y44 is z44:
            y32[:] = x44 * (-(y44**2))
        else:
            y32[:] = x44 * (-y44 * z44)

        return [cp.zeros_like(y41), y32, cp.zeros_like(y43)]
    
    def gF31(self,x,y):
        x31,x32,x33 = x
        y31,y32,y33 = y

        y42 = (-1j)*y32 * cp.conj(x32)
        y42 = y42.real if self.obj_dtype == 'float32' else y42

        y41,y43 = y31,y33
        return [y41,y42,y43]
    
    def gF32(self, x, y):
        y31, y32, y33 = y
        @self.gpu_batch(axis_out=0, axis_inp=1,nout=1)
        def _gF32(self, y42, y32):
            igpu = cp.cuda.Device().id
            y42[:] = self.cl_tomo[igpu].RT(y32)
            
        y42 = np.empty([self.nzobj, self.nobj, self.nobj], dtype=self.obj_dtype)

        _gF32(self, y42, y32)
        y41,y43 = y31,y33
        return [y41, y42, y43]    

    ############################ Regularization #########################
    def laplacian(self, obj):
        """3D Laplacian"""
        
        out = np.empty_like(obj)

        @self.gpu_batch(axis_out=0, axis_inp=0,nout=1)
        def _laplacian0(self, out, obj):
            stencil = cp.array([1, -2, 1]).astype("float32")
            out[:] = ndimage.convolve1d(obj, stencil, axis=1)
            out[:] += ndimage.convolve1d(obj, stencil, axis=2)

        @self.gpu_batch(axis_out=1, axis_inp=1,nout=1)
        def _laplacian1(self, out, out0, obj):
            stencil = cp.array([1, -2, 1]).astype("float32")
            out[:] = out0 + ndimage.convolve1d(obj, stencil, axis=0)

        _laplacian0(self, out, obj)
        _laplacian1(self, out, out, obj)
        return out

    def laplacian_sq(self, obj, out=None):
        """Twice 3D Laplacian"""
        
        out = np.empty_like(obj)

        @self.gpu_batch(axis_out=0, axis_inp=0,nout=1)
        def _laplacian_sq0(self, out, obj):
            stencil = cp.array([1, -4, 6, -4, 1]).astype("float32")
            out[:] = ndimage.convolve1d(obj, stencil, axis=1)
            out[:] += ndimage.convolve1d(obj, stencil, axis=2)

        @self.gpu_batch(axis_out=1, axis_inp=1,nout=1)
        def _laplacian_sq1(self, out, out0, obj):
            stencil = cp.array([1, -4, 6, -4, 1]).astype("float32")
            out[:] = out0 + ndimage.convolve1d(obj, stencil, axis=0)

        _laplacian_sq0(self, out, obj)
        _laplacian_sq1(self, out, out, obj)
        return out

    def min(self, prb, obj, pos,ewave):
        """Minimization functional
        1/data.size F0(F1(F2(F3(prb,pbj,pos))),data) +
        lam_prbfit/prb.size ||Dprb|-ref|_2^2 +
        lam_reg/u.size |Delta obj|_2^2"""

        ## batched computation
        out = [None]*self.ngpus
        for igpu in range(self.ngpus):
            with cp.cuda.Device(igpu):
                out[igpu] = cp.zeros(1, dtype="float32")

        ### main term
        @self.gpu_batch(axis_out=0, axis_inp=0,nout=1)
        def _min(self, out, x, data):
            out[:] += self.F0(x, data)

        x = self.fwd_ewave(ewave, pos, prb)
        _min(self, out, x, self.data)

        ### regularization term
        if self.lam_reg>0:
        
            @self.gpu_batch(axis_out=0, axis_inp=0,nout=1)
            def _min(self, out, Lobj):
                out[:] += self.lam_reg / self.obj_size * cp.linalg.norm(Lobj) ** 2

            Lobj = self.laplacian(obj)
            _min(self, out, Lobj)

        # collect results
        out = sum(out)[0]

        ### probe fit term
        igpu = cp.cuda.Device().id
        for j in range(self.ndist):
            Dprb = self.cl_prop[igpu].D(prb[j : j + 1], j)[0]
            out += self.lam_prbfit / self.prb_size * cp.linalg.norm(cp.abs(Dprb) - self.ref[j]) ** 2

        return out.get()

    def check_approximation(self, vars, etas, top, bottom, alpha, i):
        """Check the minimization functional behaviour"""
        if not (i % self.vis_step == 0 and self.vis_step != -1 and self.show):
            return

        (prb, obj, pos) = (vars["prb"], vars["obj"], vars["pos"])
        (dprb, dobj, dpos) = (etas["prb"], etas["obj"], etas["pos"])

        npp = 5
        t = np.linspace(0, 2 * alpha, npp).astype("float32")
        err_real = np.zeros(npp)
        err_approx = np.zeros(npp)
        objt = np.empty_like(obj)
        prbt = cp.empty_like(prb)
        post = np.empty_like(pos)
        
        for k in range(0, npp):            
            self.linear_batch(obj,dobj,1,t[k],out=objt)
            self.linear_batch(prb,dprb,1,t[k],out=prbt)
            self.linear_batch(pos,dpos,1,t[k],out=post)            
            err_real[k] = self.min(prbt, objt, post)

        err_approx = self.min(prb, obj, pos) - top * t + 0.5 * bottom * t**2
        mshow_approx(t,err_real,err_approx,self.show)

    def vis_debug(self, vars, i):
        """Visualization and data saving"""
        if not (i % self.vis_step == 0 and self.vis_step != -1):
            return

        (prb, obj, pos) = (vars["prb"], vars["obj"], vars["pos"])
        # denormalize u for visualization
        objs = empty_like(obj)
        self.mulc_batch(objs, obj, self.norm_const)

        # visualization
        mshow(objs[obj.shape[0] // 2].real, self.show)
        mshow_polar(prb[0], self.show)
        mshow_pos(vars["pos"] - self.pos_init,self.show)
        plt.savefig(f"{self.path_out}/rerr{i:04}.png")
        plt.close()
        
        save_intermediate(objs,prb,pos,f'{self.path_out}',i)

    def error_debug(self, vars, i):
        """Visualization and data saving"""
        if not (i % self.err_step == 0 and self.err_step != -1):
            return
            
        err = self.min(vars["prb"], vars["obj"], vars["pos"], vars["ewave"])
        print(f"{datetime.now().strftime("%H:%M:%S")} ngpus={self.ngpus} n={self.n} ntheta={self.ntheta} iter={i} {err=:1.5e}", flush=True)
        self.table.loc[len(self.table)] = [i, err, time.time()]
        name = f"{self.path_out}/conv.csv"
        os.makedirs(os.path.dirname(name), exist_ok=True)
        self.table.to_csv(name, index=False)
