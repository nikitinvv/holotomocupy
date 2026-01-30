import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import os
import warnings
import pandas as pd
import cupyx.scipy.ndimage as ndimage
from datetime import datetime
import nvtx

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
        self.gF = [self.gF0, self.gF1, self.gF2, self.gF3, self.gF4]
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
                self.cl_shift[igpu] = Shift(self.n, self.nobj, self.nz, self.nzobj, 1.0 / norm_magnifications,self.obj_dtype)

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

    # def change_rho(self,vars,grads):
    #     niter = 2
    #     nzobj = self.nzobj
    #     nobj = self.nobj
    #     nz = self.nz
    #     n = self.n
    #     ntheta = self.ntheta
    #     ndist = self.ndist
    #     energy = np.zeros(3)#[0,0,0]
    #     for j in range(niter):

    #         gobj = grads['obj']/np.linalg.norm(grads['obj'])
    #         gprb = cp.zeros([ndist,nz,n],dtype='complex64')    
    #         gpos = np.zeros([ntheta,ndist,2],dtype='float32')
            
    #         ggrads = {'obj':gobj,'prb':gprb,'pos':gpos, 'proj':self.fwd_tomo(gobj)}    
    #         energy[0] += self.hessian(vars,ggrads,ggrads)
            
    #     for j in range(niter):
    #         gobj = np.zeros([nzobj,nobj,nobj],dtype='complex64')
            
    #         gprb = grads['prb']/np.linalg.norm(grads['prb'])#cp.random.random([ndist,nz,n]).astype('float32')+1j*cp.random.random([ndist,nz,n]).astype('float32')            
    #         gpos = np.zeros([ntheta,ndist,2],dtype='float32')            
    #         ggrads = {'obj':gobj,'prb':gprb,'pos':gpos, 'proj':self.fwd_tomo(gobj)}    
            
    #         energy[1] += self.hessian(vars,ggrads,ggrads)
            
    #     for j in range(niter):
    #         gobj = np.zeros([nzobj,nobj,nobj],dtype='complex64')            
    #         gprb = cp.zeros([ndist,nz,n],dtype='complex64')            
    #         gpos = grads['pos']/np.linalg.norm(grads['pos'])#np.random.random([ntheta,ndist,2]).astype('float32')
    #         # gpos /= np.linalg.norm(gpos)
            
    #         ggrads = {'obj':gobj,'prb':gprb,'pos':gpos, 'proj':self.fwd_tomo(gobj)}        

    #         energy[2] += self.hessian(vars,ggrads,ggrads)
    #     rho = np.sqrt(energy[0]/energy)
    #     return rho
        
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
        # precalculate proj
        vars["proj"] = self.fwd_tomo(vars["obj"])
        self.time_start = time.time()
        for i in range(self.start_iter,self.niter):
            nvtx.push_range("::BH:"+str(i))
            # error and visualization debug
            self.error_debug(vars, i)
            self.vis_debug(vars, i)

            # compute gradients
            self.gradients(vars, grads)     

            # if i==0:
            #     rho = self.change_rho(vars,grads)
            #     print(f'set rho to {rho}')
            #     self.rho_sq['obj'],self.rho_sq['prb'],self.rho_sq['pos'] = rho**2

            # scale
            for v in ["obj", "prb", "pos"]:                
                self.mulc_batch(grads[v], grads[v], self.rho_sq[v])
            # keep projections in memory
            
            self.fwd_tomo(grads["obj"], out=grads["proj"])
            
            
            if i == self.start_iter:
                nvtx.push_range(":::BH:mulc_batch")
                # initial search direction (negative gradient)
                for v in ["obj", "prb", "pos"]:
                    self.mulc_batch(etas[v], grads[v], -1)
                nvtx.pop_range()
            else:
                # calc beta using Hessian-weighted inner products
                nvtx.push_range(":::BH:hessian")
                beta = self.hessian(vars, grads, etas) / self.hessian(vars, etas, etas)
                # update search direction: eta = beta * previous_eta - grad
                for v in ["obj", "prb", "pos"]:
                    self.linear_batch(etas[v], grads[v], beta, -1)
                nvtx.pop_range()
            
            # keep projections in memory  
            nvtx.push_range(":::BH:fwd_tomo")          
            self.fwd_tomo(etas["obj"], out=etas["proj"])
            nvtx.pop_range()
            
            # calc alpha (step length)            
            nvtx.push_range(":::BH:redot_batch")
            top = 0
            for v in ["obj", "prb", "pos"]:                
                top -= self.redot_batch(grads[v], etas[v]) / self.rho_sq[v]

            nvtx.pop_range()
            nvtx.push_range(":::BH:hessian1")
            bottom = self.hessian(vars, etas, etas)            
            nvtx.pop_range()      

            nvtx.push_range(":::BH:rest")
            alpha = top / bottom     
            
            
            # check approximation with the Hessian
            self.check_approximation(vars, etas, top, bottom, alpha, i)

            # update variables: var = var+alpha*eta
            for v in ["obj", "prb", "pos"]:
                self.linear_batch(vars[v], etas[v], 1, alpha)
            nvtx.pop_range()
            
            
            # update proj for current u  
            nvtx.push_range(":::BH:fwd_tomo")          
            self.fwd_tomo(vars["obj"], out=vars["proj"])
            nvtx.pop_range()
            nvtx.pop_range()         
                        
        # normalize back
        vars["obj"] *= self.norm_const
        

        return vars

    def hessian(self, vars, grads, etas):
        """Hessian for the full functional, is a sum of 3 terms:
        1. main data fit term calcuated with the cascade rule,
        2. probe fit term,
        3. regularization term"""

        nvtx.push_range("hessian")
        w = self.hessian_cascade(vars, grads, etas)
        w += self.hessian_prbfit(vars["prb"], grads["prb"], etas["prb"])
        w += self.hessian_reg(vars["obj"], grads["obj"], etas["obj"])
        nvtx.pop_range()
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
            x2, y2, z2, 
            x1, y1, z1, 
            x0, y0, z0, 
        ):
            nvtx.push_range("hessian_cascade")
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
            nvtx.pop_range()
            out[:] += w

        _hessian_cascade(
            self, out, self.data,
            vars["pos"], grads["pos"], etas["pos"],
            vars["proj"], grads["proj"], etas["proj"],
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
        
        nvtx.push_range("gradients")
        grads["prb"], grads["obj"], grads["pos"] = self.gradients_cascade(vars)
        self.gradient_prbfit(grads["prb"], vars["prb"])
        self.gradient_reg(grads["obj"], vars["obj"])
        nvtx.pop_range()
        
    def gradients_cascade(self, vars):
        """Cascade gradient for the main term
            following the composition rule (Carlsson, 2025):
            For f = F1 ◦ F2 the gradient is 
                gradf = dF_2^*(\nabla F_1)),
                where dF_2^* is the adjoint to the differential
            The function implements it for f = F0 ◦ F1 ◦ F2 ◦ F3 ...
            parameters to functions are unified as (x,y,z)
        """

        x = [vars["prb"], vars["obj"], vars["pos"], vars["proj"]]  # assume proj is precalculated
        
        # part1, parallelization over angles
        gradprb = [None]*self.ngpus
        for igpu in range(self.ngpus):
            with cp.cuda.Device(igpu):
                gradprb[igpu] = cp.zeros([self.ndist, self.nz, self.n], dtype="complex64")
        
        gradpos = np.empty([self.ntheta,self.ndist, 2], dtype="float32")
        gradproj = np.empty([self.ntheta, self.nzobj, self.nobj], dtype=self.obj_dtype)
        
        @self.gpu_batch(axis_out=0, axis_inp=0, nout=3)
        def _gradients_cascade(self,gradproj,gradpos,gradprb,d,proj,pos,prb):
            nvtx.push_range("gradients_cascade")
            x = [prb, proj, pos]
            y = d 
            # compute gradient by applying operators in forward order
            for id in range(0, len(self.gF) - 1):  #last one computed separately because of different chunking
                y = self.gF[id](x,y)                           
            gradprb[:] += y[0]
            gradproj[:] = y[1]
            gradpos[:] = y[2]
            nvtx.pop_range()
        _gradients_cascade(self,gradproj,gradpos,gradprb,self.data,vars["proj"],vars["pos"],vars["prb"])
        gradprb = sum(gradprb[1:], gradprb[0])
    
        # part2, parallelization over object slices
        
        y = [gradprb,gradproj,gradpos]                
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

    def fwd_proj(self, proj, pos, prb):
        """Forward operator with precalculated proj=R(u)"""

        out = np.empty([self.ntheta, self.ndist, self.nz, self.n], dtype="complex64")

        @self.gpu_batch(axis_out=0, axis_inp=0,nout=1)
        def _fwd_proj(self, out, proj, pos, prb):
            nvtx.push_range("fwd_proj")
            x = [prb, proj, pos]
            y = x  # forming output
            # compute functional by applying operators in reverse order
            for id in range(1, len(self.F))[::-1]:  
                y = self.F[id](y)
                
            out[:] = y
            nvtx.pop_range()

        _fwd_proj(self, out, proj, pos, prb)
        return out

    def fwd_tomo(self, obj, out=None):
        """Forward tomography operator"""

        if out is None:
            out = np.empty([self.ntheta, obj.shape[0], self.nobj], dtype=obj.dtype)

        @self.gpu_batch(axis_out=1, axis_inp=0,nout=1)
        def _fwd_tomo(self, out, obj):
            nvtx.push_range("fwd_tomo")
            igpu = cp.cuda.Device().id
            out[:] = self.cl_tomo[igpu].R(obj)
            nvtx.push_range("fwd_tomo")
            
        _fwd_tomo(self, out, obj)
        return out    

    ####################### Functions for the cascade (following math notes for variables) 
    # F* - functional
    # dF* - differential
    # d2F* - second order term for hessian
    # gF* - gradient    
    #######################################################################################


    ####### F0(x0) = 1/n\||x0|-d\|_2^2
    @nvtx.annotate("F0", color="green")
    def F0(self, x, d):
        """In: (x0), Out: const"""
        x0 = x
        return cp.linalg.norm(cp.abs(x0) - d) ** 2 / self.data_size

    @nvtx.annotate("dF0", color="green")
    def dF0(self, x, y, d, return_x=False):
        """In: (x0,y0), Out: const"""
        
        x0 = x
        y0 = y

        tmp = 2 * (x - d * (x0 / cp.abs(x0)))
        return redot(tmp, y0) / self.data_size
    
    @nvtx.annotate("d2F0", color="green")
    def d2F0(self, x, y, z, d):
        """In: (x0,y0,z0), Out: const"""
        
        x0 = x
        y0 = y
        z0 = z

        l0 = x0 / (cp.abs(x0))
        d0 = d / (cp.abs(x0))
        v1 = cp.sum((1 - d0) * reprod(y0, z0))
        v2 = cp.sum(d0 * reprod(l0, y0) * reprod(l0, z0))
        return 2 * (v1 + v2)/ self.data_size
    
    @nvtx.annotate("gF0", color="green")
    def gF0(self, x, y):
        """In: x, y = F0(F1(..(x)))), Out: y0"""
        
        # calc fwd starting from 1
        for id in range(1, 4)[::-1]: 
            x = self.F[id](x)
               
        td = y * (x / (cp.abs(x)))
        y0 = (2 / self.data_size) * (x - td) 
        
        return y0
    
    ####### x0 = F1(x11,x12) = D(x11\cdot x12)
    @nvtx.annotate("F1", color="green")
    def F1(self, x):
        """In: (x11,x12), Out: x0"""

        x11, x12 = x

        igpu = cp.cuda.Device().id
        x0 = cp.empty([x12.shape[0], self.ndist, self.nz, self.n], dtype="complex64")
        for j in range(self.ndist):
            x0[:, j] = self.cl_prop[igpu].D(x11[j] * x12[:, j], j)
        
        return x0
    
    @nvtx.annotate("dF1", color="green")
    def dF1(self, x, y, return_x=True):
        """In: (x11,x12),(y11,y12) Out: y0"""

        x11, x12 = x
        y11, y12 = y

        y0 = cp.empty([x12.shape[0], self.ndist, self.nz, self.n], dtype="complex64")
        x0 = cp.empty_like(y0) if return_x else None

        igpu = cp.cuda.Device().id
        for j in range(self.ndist):
            y0[:, j] = self.cl_prop[igpu].D(y11[j] * x12[:, j] + x11[j] * y12[:, j], j)
            if return_x:
                x0[:, j] = self.cl_prop[igpu].D(x11[j] * x12[:, j], j)

        return (x0, y0) if return_x else y0
    
    @nvtx.annotate("d2F1", color="green")
    def d2F1(self, x, y, z):
        """In: (x11,x12),(y11,y12),(z11,z12) Out: y0"""

        x11, x12 = x
        y11, y12 = y
        z11, z12 = z
        y0 = cp.empty([len(y12), self.ndist, self.nz, self.n], dtype="complex64")
        igpu = cp.cuda.Device().id
        if y12 is z12:
            for j in range(self.ndist):
                y0[:, j] = 2 * self.cl_prop[igpu].D(y11[j] * y12[:, j], j)
        else:
            for j in range(self.ndist):
                y0[:, j] = self.cl_prop[igpu].D(y11[j] * z12[:, j], j) + self.cl_prop[igpu].D(z11[j] * y12[:, j], j)

        return y0
    
    @nvtx.annotate("gF1", color="green")
    def gF1(self, x, y):
        """In: x=(x01,x02,x03),(y0) Out: y11,y12"""

        y0 = y
        
        # calc fwd starting from 2        
        for id in range(2, 4)[::-1]: 
            x = self.F[id](x)

        x11, x12 = x
                    
        igpu = cp.cuda.Device().id
        y11 = cp.zeros([self.ndist,self.nz,self.n], dtype="complex64")
        y12 = cp.empty([y0.shape[0], self.ndist, self.nz, self.n], dtype="complex64")
        for j in range(self.ndist):
            tmp = self.cl_prop[igpu].DT(y0[:, j], j)             
            y11[j] += cp.sum(tmp* np.conj(x12[:,j]), axis=0)
            y12[:, j] = tmp * np.conj(x11[j])
        return y11, y12        

    ######## (x11,x12) = F2(x21,x22) = (x21,e^{1j x22})
    @nvtx.annotate("F2", color="green")
    def F2(self, x):
        """In: (x21,x22) Out: (x11,x12)"""

        x21, x22 = x
        x11 = x21
        x12 = cp.exp(1j*x22)
        return x11, x12

    @nvtx.annotate("dF2", color="green")
    def dF2(self, x, y, return_x=True):
        """In: (x21,x22),(y21,y22) Out: (x11,x12),(y11,y12)"""

        x21, x22 = x
        y21, y22 = y

        x12 = cp.exp(1j*x22)             
        y12 = x12 * 1j * y22

        x11 = x21
        y11 = y21        

        return ([x11, x12], [y11, y12]) if return_x else [y11, y12]

    @nvtx.annotate("d2F2", color="green")
    def d2F2(self, x, y, z):
        """In: (x21,x22),(y21,y22),(z21,z22) Out: (y11,y12)"""

        x21, x22 = x
        y21, y22 = y
        z21, z22 = z
        
        if y22 is z22:
            y12 = x22 * (-(y22**2))
        else:
            y12 = x22 * (-y22 * z22)

        y11 = cp.zeros_like(y21)
        
        return [y11, y12]
    
    @nvtx.annotate("gF2", color="green")
    def gF2(self,x,y):
        """In: x(x01, x02, x03) ,(y11,y12) Out: (y21,y22)"""

        y11, y12 = y

        # calc fwd starting from 3        
        for id in range(3, 4)[::-1]: 
            x = self.F[id](x)
        x21,x22 = x

        y22 = (-1j) * y12 * cp.conj(cp.exp(1j*x22))
        y22 = y22.real if self.obj_dtype == 'float32' else y22

        y21 = y11
        return [y21, y22]
    
    ####### (x21,x22) = F3(x31,x32,x33) = (x31,S_{x_33}(x32))
    @nvtx.annotate("F3", color="green")
    def F3(self, x):
        """In: (x31, x32, x33)  Out: (x21,x22)"""

        x31, x32, x33 = x

        x22 = cp.empty([len(x33), self.ndist, self.nz, self.n], dtype="complex64")
        igpu = cp.cuda.Device().id
        c = self.cl_shift[igpu].coeff(x32)
        for k in range(self.ndist):
            x22[:, k] = self.cl_shift[igpu].curlySc(c, x33[:, k], k)

        x21 = x31
        return [x21, x22]

    @nvtx.annotate("dF3", color="green")
    def dF3(self, x, y, return_x=True):
        """In: (x31, x32, x33),(y31, y32, y33)  Out: (y31, y22)"""
        
        x31, x32, x33 = x
        y31, y32, y33 = y

        y22 = cp.zeros([len(y32), self.ndist, self.nz, self.n], dtype="complex64")
        igpu = cp.cuda.Device().id
        c = self.cl_shift[igpu].coeff(x32)
        c1 = self.cl_shift[igpu].coeff(y32)
        if return_x:
            x22 = cp.zeros([len(x32), self.ndist, self.nz, self.n], dtype="complex64")
            for k in range(self.ndist):
                r = x33[:, k]
                x22[:, k] = self.cl_shift[igpu].curlySc(c, r, k)

        for k in range(self.ndist):
            r = x33[:, k]
            Deltar = y33[:, k]
            y22[:, k] = self.cl_shift[igpu].dcurlySc(c, r, k, c1, Deltar)

        x21 = x31
        y21 = y31
        return ([x21, x22], [y21, y22]) if return_x else [y21, y22]

    @nvtx.annotate("d2F3", color="green")
    def d2F3(self, x, y, z):
        """In: (x31, x32, x33),(y31, y32, y33),(z31, z32, z33)  Out: (y21, y22)"""
        
        x31, x32, x33 = x
        y31, y32, y33 = y
        z31, z32, z33 = z
        y22 = cp.zeros([len(y32), self.ndist, self.nz, self.n], dtype="complex64")
        igpu = cp.cuda.Device().id
        c = self.cl_shift[igpu].coeff(x32)
        c1 = self.cl_shift[igpu].coeff(y32)
        c2 = self.cl_shift[igpu].coeff(z32)
        for k in range(self.ndist):
            r = x33[:, k]
            Deltar_y = y33[:, k]
            Deltar_z = z33[:, k]
            y22[:, k] = self.cl_shift[igpu].d2curlySc(c, r, k, c1, Deltar_y, c2, Deltar_z)

        y21 = cp.zeros_like(y31)
        
        return [y21, y22]
    
    @nvtx.annotate("gF3", color="green")
    def gF3(self, x, y):     
        """In: x(x01, x02, x03) ,(y21,y22) Out: (y31,y32)"""   

        y21, y22 = y

        for id in range(4, 4)[::-1]: 
            x = self.F[id](x)
        x31,x32,x33 = x

        y32 = cp.zeros([y22.shape[0], self.nzobj, self.nobj], dtype='complex64')
        y33 = cp.empty([y22.shape[0], self.ndist, 2], dtype="float32")
        igpu = cp.cuda.Device().id
        c = self.cl_shift[igpu].coeff(x32)
        for k in range(self.ndist):
            Deltapsi, Deltar = self.cl_shift[igpu].dcurlySadjc(c, x33[:, k], k, y22[:, k])
            y32[:] += Deltapsi
            y33[:, k] = Deltar
        
        y32[:] = self.cl_shift[igpu].coeff(y32)
        if self.obj_dtype=='float32':
            y32=y32.real
        
        
        y31 = y21
        return [y31, y32, y33]     
    
    ######## (x31,x32,x33) = F4(x41,x42,x43) = (x41,R(x42),x43)
    # F4=R(x),dF4=R(y),d2F4=0: Rx,Ry are stored between iterations    
    @nvtx.annotate("gF4", color="green")
    def gF4(self, x, y):
        """In: x(x01, x02, x03) ,(y31,y32,y33) Out: (y41,y42,y43)"""   
        y31, y32, y33 = y
        @self.gpu_batch(axis_out=0, axis_inp=1,nout=1)
        def _gF4(self, y42, y32):
            igpu = cp.cuda.Device().id
            y42[:] = self.cl_tomo[igpu].RT(y32)
            
        y42 = np.empty([self.nzobj, self.nobj, self.nobj], dtype=self.obj_dtype)

        _gF4(self, y42, y32)
        y41 = y31
        y43 = y33
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

    def min(self, prb, obj, pos,proj):
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

        x = self.fwd_proj(proj, pos, prb)
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
            proj = self.fwd_tomo(objt)
            err_real[k] = self.min(prbt, objt, post,proj)
        
        proj = self.fwd_tomo(obj)
        err_approx = self.min(prb, obj, pos,proj) - top * t + 0.5 * bottom * t**2
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
        mshow_complex(objs[obj.shape[0] // 2].real+1j*objs[:,obj.shape[1] // 2].real, self.show)
        mshow_polar(prb[0], self.show)
        mshow_pos(vars["pos"] - self.pos_init,self.show)
        plt.savefig(f"{self.path_out}/rerr{i:04}.png")
        plt.close()
        
        save_intermediate(objs,prb,pos,f'{self.path_out}',i)

    def error_debug(self, vars, i):
        """Visualization and data saving"""
        if not (i % self.err_step == 0 and self.err_step != -1):
            return
            
        err = self.min(vars["prb"], vars["obj"], vars["pos"], vars["proj"])
        ittime = time.time()-self.time_start
        print(f"{datetime.now().strftime("%H:%M:%S")} ngpus={self.ngpus} n={self.n} ntheta={self.ntheta} iter={i} {ittime=:.0f}s {err=:1.5e} ", flush=True)                
        # print(f"{i} {err=:1.5e} ", flush=True)                
        
        self.table.loc[len(self.table)] = [i, err, ittime]
        self.time_start = time.time()
        name = f"{self.path_out}/conv.csv"
        os.makedirs(os.path.dirname(name), exist_ok=True)
        self.table.to_csv(name, index=False)
