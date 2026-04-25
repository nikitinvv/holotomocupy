import numpy as np
import cupy as cp
import os
import warnings
import pandas as pd
import nvtx

from .tomo import Tomo
from .propagation import Propagation
from .shift import Shift
from .chunking import Chunking
from .utils import *
from .mpi_functions import *
from .logger_config import logger
from .conv2d_cufftdx import precompile as cufftdx_precompile

np.set_printoptions(legacy="1.25")
warnings.filterwarnings("ignore", message=f".*peer.*")


class Rec:
    def __init__(self, args):

        # copy args to elements of the class
        for key, value in vars(args).items():
            setattr(self, key, value)

        # defaults for optional regularisation weights
        if not hasattr(self, 'lam_laplacian'):
            self.lam_laplacian = 0

        # list of functionals, gradients, differentials, and second-order differentials
        self.F = [self.F0, self.F1, self.F2, self.F3]
        self.gF = [self.gF0, self.gF1, self.gF2, self.gF3]
        self.dF = [self.dF0, self.dF1, self.dF2, self.dF3]
        self.d2F_dF = [self.d2F_dF0, self.d2F_dF1, self.d2F_dF2,self.d2F_dF3]

        # estimate memory footprint for pinned + device buffer per GPU (complex64)
        multiplier = 16  # related to the number of arrays, experimentally chosen. the scheme will diverge if too low
        complex_item = np.dtype("complex64").itemsize
        max_dim = max(self.nobj, self.nzobj, self.ntheta)
        nbytes = int(multiplier * self.nchunk * self.nobj * max_dim * complex_item)

        ### multinode processing
        self.cl_mpi = MPIClass(args.comm, self.nzobj, self.ntheta, self.nobj, args.obj_dtype)
        self.local_nzobj = self.cl_mpi.local_nzobj
        self.local_ntheta = self.cl_mpi.local_ntheta
        self.rank      = self.cl_mpi.rank
        self.st_obj    = self.cl_mpi.st_obj
        self.end_obj   = self.cl_mpi.end_obj
        self.st_theta  = self.cl_mpi.st_theta
        self.end_theta = self.cl_mpi.end_theta

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
        self.cl_shift = Shift(self.n, self.nobj, self.nz, self.nzobj, self.obj_dtype, self.nchunk)

        # All hot-path FFTs now use manually pre-built plans; the auto-cache
        # is only hit by one-time init calls (Paganin FBP, coeffback), so
        # caching provides no benefit and wastes GPU memory.
        import cupy.fft
        cupy.fft.config.get_plan_cache().set_size(0)

        self.alloc_arrays()

        # save convergence results
        self.table = pd.DataFrame(columns=["iter", "err", "time"])

        # normalization constant to address work with normal operators
        self.norm_const = np.float32(np.sqrt(self.nobj / self.ntheta))        
        self.norm_magnifications = norm_magnifications
        # sizes for normalization        
        self.data_size = self.ntheta * self.ndist * self.nz * self.n
        self.prb_size = self.ndist * self.nz * self.n
        self.obj_size = self.nzobj * self.nobj**2

        # fast refs
        self.gpu_batch = self.cl_chunking.gpu_batch
        self.redot_batch = self.cl_chunking.redot_batch
        self.linear_batch = self.cl_chunking.linear_batch
        self.linear_redot_batch = self.cl_chunking.linear_redot_batch
        self.mulc_batch = self.cl_chunking.mulc_batch
        self.redist = self.cl_mpi.redist
        self.allreduce  = self.cl_mpi.allreduce
        self.allreduce2 = self.cl_mpi.allreduce2

    def alloc_arrays(self):
        """Allocate all pinned CPU and CuPy GPU buffers used during reconstruction."""
        prb_shape = [self.ndist, self.nz, self.n]
        # pre-allocated padded buffers for 3-D biharmonic regularization.
        # u_pad[2:-2] IS vars['obj'];  e_pad[2:-2] IS etas['obj'].
        # 2 ghost rows on each side let us compute (∇²)² in a single 5-slice pass.
        self.u_pad = make_pinned([self.local_nzobj + 4, self.nobj, self.nobj], dtype=self.obj_dtype)
        self.e_pad = make_pinned([self.local_nzobj + 4, self.nobj, self.nobj], dtype=self.obj_dtype)
        self.u_pad[:] = 0
        self.e_pad[:] = 0
        # reconstruction variables
        self.vars = {
            'obj':  self.u_pad[2:-2],                                        # view — lives in u_pad
            'pos':  cp.zeros([self.local_ntheta, self.ndist, 2],           dtype='float32'),
            'prb':  cp.empty(prb_shape,                                    dtype='complex64'),
            'proj': make_pinned([self.local_ntheta, self.nzobj, self.nobj], dtype=self.obj_dtype),
        }
        # measurement data and reference
        self.data = make_pinned([self.local_ntheta, self.ndist, self.nz, self.n], dtype='float32')
        self.ref  = cp.empty(prb_shape,                                           dtype='float32')
        # gradient and conjugate-direction buffers
        self.grads, self.etas = {}, {}
        for ge in self.grads, self.etas:
            ge["obj"]  = make_pinned([self.local_nzobj,  self.nobj, self.nobj], dtype=self.obj_dtype)
            ge["pos"]  = cp.zeros([self.local_ntheta, self.ndist, 2],           dtype='float32')
            ge["proj"] = make_pinned([self.local_ntheta, self.nzobj, self.nobj], dtype=self.obj_dtype)
            ge["prb"]  = cp.empty(prb_shape, dtype='complex64')
        self.etas["obj"] = self.e_pad[2:-2]                                  # view — lives in e_pad
        self.proj_tmp  = make_pinned([self.ntheta, self.local_nzobj, self.nobj], dtype=self.obj_dtype)
        
        self.shrink_nd = cp.zeros((self.local_ntheta, self.ndist), dtype='float32')
        self.eff_demagnifications = self.shrink_nd #reuse memory

    def BH(self, writer=None):
        # refs to preallocated memory for gradients                
        vars = self.vars        
        grads = self.grads
        etas = self.etas
        proj_tmp = self.proj_tmp

        # shrinkage
        self.eff_demagnifications[:]  = (1 + self.shrink_nd) / cp.array(self.norm_magnifications[None, :])

        # normalize to work with normal operators (do this once, restore in finally)
        vars["obj"] /= self.norm_const
        if self.start_iter==0:
            vars["obj"]*=self.cl_tomo.mask
        
        self.pos_init = vars['pos'].copy()
        
        # precalculate proj             
        self.fwd_tomo(vars["obj"],out = proj_tmp)                    
        self.redist(proj_tmp, vars['proj'])
        
        # calc init error
        self.error_debug(vars, -1)  
        
        self.time_start = time.time()
        for i in range(self.start_iter,self.niter):
            
            nvtx.push_range("::BH:"+str(i))
                        
            # compute gradients
            nvtx.push_range("gradients")
            self.gradients(vars, grads)      
            nvtx.pop_range()       
                                        
            nvtx.push_range(":::BH:fwd_tomo")                                  
            self.fwd_tomo(grads["obj"], out=proj_tmp)
            nvtx.pop_range()

            nvtx.push_range(":::BH:redist",color='red')          
            self.cl_mpi.redist(proj_tmp, grads['proj'])
            nvtx.pop_range()
            
            if i == self.start_iter:
                # initial search direction (negative gradient)
                beta = 0

            else:
                # calc beta using Hessian-weighted inner products
                nvtx.push_range(":::BH:calc beta")

                top = self.hessian(vars, grads, etas)
                bottom = self.hessian(vars, etas, etas)
                top, bottom = self.allreduce2(top, bottom)
                beta = top / bottom

                nvtx.pop_range()

            # update search direction and accumulate top for alpha in one pass
            nvtx.push_range(":::BH:calc_alpha")

            top = 0
            for v in ["obj", "pos"]:
                top -= self.linear_redot_batch(etas[v], grads[v], beta, -1) / self.rho_sq[v]
            
            # probe is shared
            dot_prb = self.linear_redot_batch(etas['prb'], grads['prb'], beta, -1)
            if self.rank == 0:
                top -= dot_prb / self.rho_sq['prb']
            
            # also update proj
            self.linear_batch(etas['proj'], grads['proj'], beta, -1)

            bottom = self.hessian(vars, etas, etas)

            top, bottom = self.allreduce2(top, bottom)
            alpha = top / bottom
            nvtx.pop_range()      
            
            # update variables: var = var+alpha*eta
            for v in ["obj", "prb", "pos", "proj"]:
                self.linear_batch(vars[v], etas[v], 1, alpha)            
            
            # error and visualization debug
            nvtx.push_range(":::BH:calc error",color='gray')  
            self.error_debug(vars, i)
            nvtx.pop_range()  
            
            nvtx.push_range(":::BH:vis_debug", color='gray')
            self.vis_debug(vars, i, writer)
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
        if self.rank==0:
            w += self.hessian_prbfit(vars["prb"], grads["prb"], etas["prb"])
        w += self.hessian_laplacian(grads["obj"])

        nvtx.pop_range()
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
        """Full gradient, consists of 2 terms:
        1. main data fit term calcuated with the cascade rule,
        2. probe fit term,
        """
        
        self.gradients_cascade(vars,grads)

        nvtx.push_range(":::BH:redist back",color='red')             
        self.cl_mpi.redist(grads['proj'], self.proj_tmp,direction='backward')
        nvtx.pop_range() 
        
        # part2, parallelization over object slices, formally gF4
        self.gF4(grads['obj'], self.proj_tmp)
        self.gradient_laplacian(grads['obj'])

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
            # move variable scaling here to avoid additional data transfers
            gradprb[:] += y[0]*self.rho_sq['prb']
            gradproj[:] = y[1]*self.rho_sq['obj']
            gradpos[:] = y[2]*self.rho_sq['pos']

        _gradients_cascade(self,grads['proj'],grads['pos'],grads['prb'],self.data,vars["proj"],vars["pos"],vars["prb"])
        
    @timer
    def gF4(self, gradu, gradproj):
        @self.gpu_batch(axis_out=0, axis_inp=1,nout=1)
        def _gF4(self, gradu, gradproj):
            gradu[:] = self.cl_tomo.RT(gradproj)        
        nvtx.push_range("gF4",color='green')            
        _gF4(self, gradu, gradproj)      
        nvtx.pop_range()     

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
            td = self.lam_prbfit / self.prb_size * self.cl_prop.DT(2 * (tmp - td), j)
            # scaling moved here here
            grad_prb[j : j + 1] += td*self.rho_sq['prb']                
        
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

    #### 3-D Laplacian regularization

    def _exchange_ghosts(self, pad):
        """Fill pad[0:2] and pad[-2:] from neighbouring ranks (pad[2:-2] is the data).
        2 ghost rows on each side are needed for the single-pass biharmonic (∇²)².
        Zeroes ghost cells at the global domain boundary (no neighbour = zero padding).
        """
        left  = self.rank - 1 if self.rank > 0                      else MPI.PROC_NULL
        right = self.rank + 1 if self.rank < self.cl_mpi.size - 1   else MPI.PROC_NULL
        self.cl_mpi.comm.Sendrecv(
            sendbuf=np.ascontiguousarray(pad[-4:-2]), dest=right,
            recvbuf=pad[0:2], source=left)
        self.cl_mpi.comm.Sendrecv(
            sendbuf=np.ascontiguousarray(pad[2:4]),   dest=left,
            recvbuf=pad[-2:], source=right)
        if left  == MPI.PROC_NULL: pad[0:2]  = 0
        if right == MPI.PROC_NULL: pad[-2:]  = 0

    @timer
    def gradient_laplacian(self, grad_obj):
        """Add 2*lam/obj_size * (∇²)²u to grad_obj in-place (energy = lam/N * ||∇²u||²).
        u_pad (size local_nzobj+4) is transferred as a single padded proper input:
        each chunk gets chunk+4 rows so the kernel can compute (∇²)² without extra views.
        """
        if self.lam_laplacian == 0:
            return
        scale = np.float32(2.0 * self.lam_laplacian / self.obj_size)
        self._exchange_ghosts(self.u_pad)

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1, inp_pad=4)
        def _biharm_grad(self, g, u_pad_chunk, g_in):
            # u_pad_chunk: [n+4, nobj, nobj]; g, g_in: [n, nobj, nobj]
            def _lap(a, b, c):
                return (a + c
                        + cp.roll(b, -1, axis=1) + cp.roll(b,  1, axis=1)
                        + cp.roll(b, -1, axis=2) + cp.roll(b,  1, axis=2)
                        - 6 * b)
            lap_zm1 = _lap(u_pad_chunk[:-4], u_pad_chunk[1:-3], u_pad_chunk[2:-2])
            lap_z   = _lap(u_pad_chunk[1:-3], u_pad_chunk[2:-2], u_pad_chunk[3:-1])
            lap_zp1 = _lap(u_pad_chunk[2:-2], u_pad_chunk[3:-1], u_pad_chunk[4:])
            g[:] = g_in + scale * _lap(lap_zm1, lap_z, lap_zp1)

        _biharm_grad(self, grad_obj, self.u_pad, grad_obj)

    @timer
    def hessian_laplacian(self, dobj1):
        """2*lam/obj_size * Re<dobj1, (∇²)²e>, e = self.e_pad[2:-2] = etas['obj'].
        e_pad (size local_nzobj+4) transferred as single padded proper input (inp_pad=4).
        Allreduced over MPI ranks.
        """
        if self.lam_laplacian == 0:
            return 0
        scale = np.float32(2.0 * self.lam_laplacian / self.obj_size)
        self._exchange_ghosts(self.e_pad)
        acc = cp.zeros(1, dtype='float32')

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1, inp_pad=4)
        def _biharm_dot(self, acc, e_pad_chunk, d1):
            # e_pad_chunk: [n+4, nobj, nobj]; d1: [n, nobj, nobj]
            def _lap(a, b, c):
                return (a + c
                        + cp.roll(b, -1, axis=1) + cp.roll(b,  1, axis=1)
                        + cp.roll(b, -1, axis=2) + cp.roll(b,  1, axis=2)
                        - 6 * b)
            lap_zm1 = _lap(e_pad_chunk[:-4], e_pad_chunk[1:-3], e_pad_chunk[2:-2])
            lap_z   = _lap(e_pad_chunk[1:-3], e_pad_chunk[2:-2], e_pad_chunk[3:-1])
            lap_zp1 = _lap(e_pad_chunk[2:-2], e_pad_chunk[3:-1], e_pad_chunk[4:])
            acc[:] += redot(d1, _lap(lap_zm1, lap_z, lap_zp1))

        _biharm_dot(self, acc, self.e_pad, dobj1)
        return float(self.allreduce(np.array(scale * float(acc[0]), dtype='float32')))

    def _lap_energy_local(self):
        """Local biharmonic energy (lam/obj_size)*||∇²u||² = (lam/obj_size)*Re<∇²u,∇²u>.
        Only first Laplacian needed; 3 proper z-slice views of u_pad (each size local_nzobj).
        u = self.u_pad[2:-2] = vars['obj']. No allreduce.
        """
        if self.lam_laplacian == 0:
            return np.float32(0)
        scale = np.float32(self.lam_laplacian / self.obj_size)
        self._exchange_ghosts(self.u_pad)
        acc = cp.zeros(1, dtype='float32')

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1, inp_pad=4)
        def _biharm_e(self, acc, u_pad_chunk):
            # u_pad_chunk: [n+4, nobj, nobj]; only first Laplacian needed for energy
            u_zm1 = u_pad_chunk[1:-3]
            u_z   = u_pad_chunk[2:-2]
            u_zp1 = u_pad_chunk[3:-1]
            lap = (u_zm1 + u_zp1
                   + cp.roll(u_z, -1, axis=1) + cp.roll(u_z,  1, axis=1)
                   + cp.roll(u_z, -1, axis=2) + cp.roll(u_z,  1, axis=2)
                   - 6 * u_z)
            acc[:] += redot(lap, lap)

        _biharm_e(self, acc, self.u_pad)
        return scale * float(acc[0])

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
    @staticmethod
    @cp.fuse()
    def _F0_fused(x, d):
        t = cp.abs(x) - d
        return t * t

    @nvtx.annotate("F0", color="green")
    def F0(self, x, d):
        """In: (x0), Out: const"""
        return 1 / self.data_size * cp.sum(self._F0_fused(x, d))

    @staticmethod
    @cp.fuse()
    def _dF0_fused(x, d):
        return x - d * (x / cp.abs(x))

    @nvtx.annotate("dF0", color="green")
    def dF0(self, x, y, d, return_x=False):
        """In: (x0,y0), Out: const"""
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

    @nvtx.annotate("d2F0_dF0", color="purple")
    def d2F_dF0(self, x, y, z, w, d):
        """In: (x0,y0,z0,w0), Out: const"""
        return 2 / self.data_size * cp.sum(self._d2F_dF0_fused(x, y, z, w, d))
        
    @staticmethod
    @cp.fuse()
    def _gF0_fused(x, y, scale):
        td = y * (x / cp.abs(x))
        return scale * (x - td)

    @nvtx.annotate("gF0", color="green")
    def gF0(self, x, y):
        """In: x, y = F0(F1(..(x)))), Out: y0"""

        # calc fwd starting from 1
        for id in range(1, 4)[::-1]:
            x = self.F[id](x)

        return self._gF0_fused(x, y, np.float32(2 / self.data_size))
    
    ####### x0 = F1(x11,x12) = D(x11\cdot x12)
    @nvtx.annotate("F1", color="green")
    def F1(self, x):
        """In: (x11,x12), Out: x0"""

        x11, x12 = x

        x0 = cp.empty([x12.shape[0], self.ndist, self.nz, self.n], dtype="complex64")
        for j in range(self.ndist):
            x0[:, j] = self.cl_prop.D(x11[j] * x12[:, j], j)

        return x0

    @nvtx.annotate("dF1", color="green")
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
    
    @nvtx.annotate("d2F_dF1", color="purple")
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
   
    @nvtx.annotate("gF1", color="green")
    def gF1(self, x, y):
        """In: x=(x01,x02,x03),(y0) Out: y11,y12"""

        y0 = y

        # calc fwd starting from 2
        for id in range(2, 4)[::-1]:
            x = self.F[id](x)

        x11, x12 = x
        y12 = cp.empty([y0.shape[0], self.ndist, self.nz, self.n], dtype="complex64")
        for j in range(self.ndist):
            y12[:,j] = self.cl_prop.DT(y0[:, j], j)

        y11 = cp.sum(y12 * np.conj(x12), axis=0)
        y12 *= np.conj(x11[None])
        return y11, y12        

    ######## (x11,x12) = F2(x21,x22) = (x21,e^{1j x22})
    @staticmethod
    @cp.fuse()
    def _F2_fused(x22):
        return cp.exp(1j * x22)

    @nvtx.annotate("F2", color="green")
    def F2(self, x):
        """In: (x21,x22) Out: (x11,x12)"""

        x21, x22 = x
        x11 = x21
        x12 = self._F2_fused(x22)
        return x11, x12

    @staticmethod
    @cp.fuse()
    def _dF2_fused(x22, y22):
        x12 = cp.exp(1j * x22)
        y12 = x12 * 1j * y22
        return x12, y12

    @nvtx.annotate("dF2", color="green")
    def dF2(self, x, y, return_x=True):
        """In: (x21,x22),(y21,y22) Out: (x11,x12),(y11,y12)"""

        x21, x22 = x
        y21, y22 = y

        x12, y12 = self._dF2_fused(x22, y22)
        x11 = x21
        y11 = y21

        return ([x11, x12], [y11, y12]) if return_x else [y11, y12]

   
    
    @staticmethod
    @cp.fuse()
    def _d2F_dF2_fused(x22, y22, z22, w22):
        y12 = cp.exp(1j * x22) * (-y22 * z22)
        if w22 is not None:
            y12 = y12 + cp.exp(1j * x22) * 1j * w22
        return y12

    @nvtx.annotate("d2F_dF2", color="purple")
    def d2F_dF2(self, x, y, z, w):
        """In: (x21,x22),(y21,y22),(z21,z22),(w21,w22) Out: (y11,y12)"""

        x21, x22 = x
        y21, y22 = y
        z21, z22 = z
        w21, w22 = w

        y12 = self._d2F_dF2_fused(x22, y22, z22, w22)
        y11 = w21

        return [y11, y12]
    
    @staticmethod
    @cp.fuse()
    def _gF2_fused(x22, y12):
        return (-1j) * y12 * cp.conj(cp.exp(1j * x22))

    @nvtx.annotate("gF2", color="green")
    def gF2(self, x, y):
        """In: x(x01, x02, x03) ,(y11,y12) Out: (y21,y22)"""

        y11, y12 = y

        # calc fwd starting from 3
        for id in range(3, 4)[::-1]:
            x = self.F[id](x)
        x21, x22 = x

        y22 = self._gF2_fused(x22, y12)
        y22 = y22.real if self.obj_dtype == 'float32' else y22

        y21 = y11
        return [y21, y22]
    
    ####### (x21,x22) = F3(x31,x32,x33) = (x31,S_{x_33}(x32))
    @nvtx.annotate("F3", color="green")
    def F3(self, x):
        """In: (x31, x32, x33)  Out: (x21,x22)"""

        x31, x32, x33 = x

        x22 = cp.empty([len(x33), self.ndist, self.nz, self.n], dtype=self.obj_dtype)
        c = self.cl_shift.coeff(x32)
        for k in range(self.ndist):
            x22[:, k] = self.cl_shift.curlySc(c, x33[:, k], self.eff_demagnifications[:, k])

        x21 = x31
        return [x21, x22]

    @nvtx.annotate("dF3", color="green")
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
                x22[:, k] = self.cl_shift.curlySc(c, x33[:, k], self.eff_demagnifications[:, k])

        for k in range(self.ndist):
            y22[:, k] = self.cl_shift.dcurlySc(c, x33[:, k], self.eff_demagnifications[:, k], c1, y33[:, k])

        x21 = x31
        y21 = y31
        return ([x21, x22], [y21, y22]) if return_x else [y21, y22]

    

    @nvtx.annotate("d2F_dF3", color="purple")
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
            y22[:, k] = self.cl_shift.d2curlySc(c, x33[:, k], self.eff_demagnifications[:, k], cy, y33[:, k], cz, z33[:, k])

        if w32 is not None:
            cy = self.cl_shift.coeff(w32)
            for k in range(self.ndist):
                y22[:, k] += self.cl_shift.dcurlySc(c, x33[:, k], self.eff_demagnifications[:, k], cy, w33[:, k])

        y21 = w31

        return [y21, y22]

    @nvtx.annotate("gF3", color="green")
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
            Deltapsi, Deltar = self.cl_shift.dcurlySadjc(c, x33[:, k], self.eff_demagnifications[:, k], y22[:, k])
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
        return self.allreduce(np.array(out.get() + self._lap_energy_local(), dtype='float32'))

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

        if writer is not None and i > self.start_iter:
            delta = cp.asnumpy(vars['pos'] - self.pos_init)   # [local_ntheta, ndist, 2]
            abs_local = np.sum(np.abs(delta), axis=0)          # [ndist, 2]
            abs_global = self.comm.allreduce(abs_local)
            mean_err = abs_global / self.ntheta                # [ndist, 2]
            if self.rank == 0:
                parts = "  ".join(
                    f"d{j}:({mean_err[j,0]:.4f},{mean_err[j,1]:.4f})"
                    for j in range(self.ndist)
                )
                logger.warning(f"iter={i}: pos mean abs error [px]  {parts}")
            
    
    def error_debug(self, vars, i):
        """Error logging and CSV checkpoint export."""
        if not (i % self.err_step == 0 and self.err_step != -1):
            return
            
        err = self.min(vars["prb"], vars["obj"], vars["pos"], vars["proj"])        
        if self.rank==0:
            if i==-1:
                logger.warning(f"Initial {err=:1.5e} ")                        
                self.table.loc[len(self.table)] = [i, err, 0]
            else:                
                ittime = time.time()-self.time_start           
                logger.warning(f"iter={i}: {ittime:.4f}sec {err=:1.5e} ")                        
                self.table.loc[len(self.table)] = [i, err, ittime]
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