import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time
from concurrent.futures import ThreadPoolExecutor
from cupy.cuda import runtime
import cupyx.scipy.ndimage as ndimage
import os
import sys


sys.path.insert(0, '..')
from cuda_kernels import *
from utils import *
from chunking import gpu_batch
import time
from functools import wraps

class Rec:
    def __init__(self, args):
        ngpus = args.ngpus
        ntheta = args.ntheta
        npsi = args.npsi
        nq = args.nq
        npatch = args.npatch
        ncode = args.ncode
        n = args.n
        pad = args.pad
        nchunk = args.nchunk
        ex = args.ex
        voxelsize = args.voxelsize
        wavelength = args.wavelength
        distance = args.distance
        distancec = args.distancec
        eps = args.eps
        rho = args.rho
        lam = args.lam
        path_out = args.path_out
        show = args.show
        theta = args.theta
        rotation_axis = args.rotation_axis

        # allocate gpu and pinned memory buffers
        # calculate max buffer size
        nbytes = (
            16 * nchunk * npatch * npatch * np.dtype("complex64").itemsize            
        )

        # create CUDA streams and allocate pinned memory
        self.stream = [[[] for _ in range(3)] for _ in range(ngpus)]
        self.pinned_mem = [[] for _ in range(ngpus)]
        self.gpu_mem = [[] for _ in range(ngpus)]
        self.fker = [[] for _ in range(ngpus)]
        self.fkerc = [[] for _ in range(ngpus)]
        self.pool_inp = [[] for _ in range(ngpus)]
        self.pool_out = [[] for _ in range(ngpus)]
        self.pool = ThreadPoolExecutor(ngpus)    
        
        for igpu in range(ngpus):
            with cp.cuda.Device(igpu):
                self.pinned_mem[igpu] = cp.cuda.alloc_pinned_memory(nbytes)
                self.gpu_mem[igpu] = cp.cuda.alloc(nbytes)
                for k in range(3):
                    self.stream[igpu][k] = cp.cuda.Stream(non_blocking=False)
                fx = cp.fft.fftfreq(nq, d=voxelsize).astype("float32")
                [fx, fy] = cp.meshgrid(fx, fx)
                self.fker[igpu] = cp.exp(-1j * cp.pi * wavelength * distance * (fx**2 + fy**2)).astype('complex64')
                self.fkerc[igpu] = cp.exp(-1j * cp.pi * wavelength * distancec * (fx**2 + fy**2)).astype('complex64')
                self.pool_inp[igpu] = ThreadPoolExecutor(16 // ngpus)    
                self.pool_out[igpu] = ThreadPoolExecutor(16 // ngpus)
        
        self.pool_cpu = ThreadPoolExecutor(16)
        self.npsi = npsi
        self.ntheta = ntheta
        self.nq = nq
        self.npatch = npatch
        self.ncode = ncode
        self.n = n
        self.nchunk = nchunk
        self.ngpus = ngpus
        self.ex = ex
        self.pad = pad
        self.eps = eps
        self.rho = rho
        self.lam = lam
        self.path_out = path_out        
        self.niter = args.niter
        self.err_step = args.err_step
        self.vis_step = args.vis_step
        self.rho = args.rho
        self.show = show
        self.theta = theta
        self.voxelsize = voxelsize
        self.wavelength = wavelength
        self.rotation_axis = rotation_axis
    

    def redot_batch(self,x,y):
        flg = chunking_flg(locals().values())        
        if flg:
            res = []
            for igpu in range(self.ngpus):
                with cp.cuda.Device(igpu):
                    res.append(cp.zeros(1, dtype="float32"))
        else:
            res = cp.zeros(1, dtype="float32")
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _redot(self, res, x,y):
            res[:] += redot(x, y)
            return res
        _redot(self, res, x,y)
        if flg:
            for k in range(1,len(res)):
                res[0] += res[k]
            res = res[0]    
        res = res[0]    
        return res
    
    def BH(self, d, vars):
        d = np.sqrt(d)
        
        alpha = 1
        rho = self.rho
        reused = {}
        
        for i in range(self.niter):
            self.calc_reused(reused, vars)
            self.gradientF(reused, d)
            self.error_debug(vars, reused, d, i)
            self.vis_debug(vars, i)
            grads = self.gradients(vars, reused)
            if i == 0:
                etas = {}
                etas["q"] = mulc(grads['q'],-rho[1]**2,self.pool_cpu)
                etas["r"] = mulc(grads['r'],-rho[2]**2,self.pool_cpu)
            else:
                beta = self.calc_beta(vars, grads, etas, reused, d)
                print(f'{beta=}')
                etas["q"] = -grads["q"] * rho[1] ** 2 + beta * etas["q"]
                etas["r"] = -grads["r"] * rho[2] ** 2 + beta * etas["r"]
            
            alpha, top, bottom = self.calc_alpha(vars, grads, etas, reused, d)
            print(alpha,top,bottom)
            self.plot_debug(vars, etas, top, bottom, alpha, d, i)
            linear(vars["q"],vars["q"],etas['q'],1,alpha,self.pool_cpu)                                
            linear(vars["r"],vars["r"],etas['r'],1,alpha,self.pool_cpu)                                
            
        return vars
    
    def E(self, ri, code):
        """Extract patches"""

        flg = chunking_flg(locals().values())
        # memory for result
        if flg:
            patches = np.empty([len(ri), self.npatch, self.npatch], dtype="complex64")
        else:
            patches = cp.empty([len(ri), self.npatch, self.npatch], dtype="complex64")

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _E(self, patches, ri, code):
            stx = self.ncode // 2 - ri[:, 1] - self.npatch // 2
            sty = self.ncode // 2 - ri[:, 0] - self.npatch // 2
            Efast_kernel(
                (
                    int(cp.ceil(self.npatch / 16)),
                    int(cp.ceil(self.npatch / 16)),
                    int(cp.ceil(len(ri) / 4)),
                ),
                (16, 16, 4),
                (patches, code, stx, sty, len(ri), self.npatch, self.ncode),
            )

        _E(self, patches, ri, code)

        return patches

    def ET(self, patches, ri):
        """Place patches, note only on 1 gpu for now"""
        
        flg = chunking_flg(locals().values())        
        if flg:
            code = []
            for igpu in range(self.ngpus):
                with cp.cuda.Device(igpu):
                    code.append(cp.zeros([self.ncode, self.ncode], dtype="complex64"))
        else:
            code = cp.zeros([self.ncode, self.ncode], dtype="complex64")

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _ET(self, code, patches, ri):
            """Adjoint extract patches"""
            stx = self.ncode // 2 - ri[:, 1] - self.npatch // 2
            sty = self.ncode // 2 - ri[:, 0] - self.npatch // 2
            ETfast_kernel(
                (
                    int(cp.ceil(self.npatch / 16)),
                    int(cp.ceil(self.npatch / 16)),
                    int(cp.ceil(len(ri) / 4)),
                ),
                (16, 16, 4),
                (patches, code, stx, sty, len(ri), self.npatch, self.ncode),
            )

        _ET(self, code, patches, ri)
        if flg:
            for k in range(1,len(code)):
                code[0] += code[k]
            code = code[0]    
        return code

    def S(self, ri, r, code):
        # memory for result
        flg = chunking_flg(locals().values())        
        if flg:
            patches = np.empty([len(ri), self.nq, self.nq], dtype="complex64")
        else:
            patches = cp.empty([len(ri), self.nq, self.nq], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _S(self, patches, ri, r, code):
            """Extract patches with subpixel shift"""
            coder = self.E(ri, code)

            x = cp.fft.fftfreq(self.npatch).astype("float32")
            [y, x] = cp.meshgrid(x, x)
            tmp = cp.exp(
                -2 * cp.pi * 1j * (y * r[:, 1, None, None] + x * r[:, 0, None, None])
            ).astype("complex64")
            coder = cp.fft.ifft2(tmp * cp.fft.fft2(coder))
            patches[:] = coder[
                :, self.ex : self.npatch - self.ex, self.ex : self.npatch - self.ex
            ]

        _S(self, patches, ri, r, code)
        return patches

    def ST(self, patches, ri, r):
        """Place patches, note only on 1 gpu for now"""

        flg = chunking_flg(locals().values())        
        if flg:
            code = []
            for igpu in range(self.ngpus):
                with cp.cuda.Device(igpu):
                    code.append(cp.zeros([self.ncode, self.ncode], dtype="complex64"))
        else:
            code = cp.zeros([self.ncode, self.ncode], dtype="complex64")

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)                
        def _ST(self, code, patches , ri, r):
            """Adjont extract patches with subpixel shift"""
            coder = cp.pad(patches, ((0, 0), (self.ex, self.ex), (self.ex, self.ex)))

            x = cp.fft.fftfreq(self.npatch).astype("float32")
            [y, x] = cp.meshgrid(x, x)
            tmp = cp.exp(
                2 * cp.pi * 1j * (y * r[:, 1, None, None] + x * r[:, 0, None, None])
            ).astype("complex64")
            coder = cp.fft.ifft2(tmp * cp.fft.fft2(coder))

            code[:] += self.ET(coder, ri)

        _ST(self, code, patches, ri, r)
        if flg:
            for k in range(1,len(code)):
                code[0] += code[k]
            code = code[0]    
        return code

    def _fwd_pad(self,f):
        """Fwd data padding"""
        [ntheta, n] = f.shape[:2]
        fpad = cp.zeros([ntheta, 2*n, 2*n], dtype='complex64')
        pad_kernel((int(cp.ceil(2*n+2/32)), int(cp.ceil(2*n/32)), ntheta),
                (32, 32, 1), (fpad, f, n, ntheta, 0))
        return fpad/2
    
    def _adj_pad(self, fpad):
        """Adj data padding"""
        [ntheta, n] = fpad.shape[:2]
        n //= 2
        f = cp.zeros([ntheta, n, n], dtype='complex64')
        pad_kernel((int(cp.ceil(2*n/32)), int(cp.ceil(2*n/32)), ntheta),
                (32, 32, 1), (fpad, f, n, ntheta, 1))
        return f/2
    def D(self, psi):
        """Forward propagator"""
         # memory for result
        flg = chunking_flg(locals().values())        
        if flg:
            big_psi = np.empty([len(psi), self.n, self.n], dtype="complex64")
        else:
            big_psi = cp.empty([len(psi), self.n, self.n], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _D(self, big_psi, psi):
            ff = psi.copy()
            # ff = self._fwd_pad(ff)
            # v = cp.ones(2*self.nq,dtype='float32')
            # v[:self.nq//2] = cp.sin(cp.linspace(0,1,self.nq//2)*cp.pi/2)
            # v[-self.nq//2:] = cp.cos(cp.linspace(0,1,self.nq//2)*cp.pi/2)
            # v = cp.outer(v,v)
            # ff *= v*2        
            ff = cp.fft.ifft2(cp.fft.fft2(ff)*self.fker[cp.cuda.Device().id])
            # ff = ff[:,self.nq//2:-self.nq//2,self.nq//2:-self.nq//2]
            
            big_psi[:] = ff[:, self.pad : self.nq - self.pad, self.pad : self.nq - self.pad]
            
        
        _D(self, big_psi, psi)
        return big_psi


    def DT(self, big_psi):
        """Adjoint propagator"""
        # memory for result
        flg = chunking_flg(locals().values())        
        if flg:
            psi = np.empty([len(big_psi), self.nq, self.nq], dtype="complex64")
        else:
            psi = cp.empty([len(big_psi), self.nq, self.nq], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _DT(self, psi, big_psi):
            # pad to the probe size
            ff = cp.pad(big_psi, ((0, 0), (self.pad, self.pad), (self.pad, self.pad)))
            ff = cp.fft.ifft2(cp.fft.fft2(ff)/self.fker[cp.cuda.Device().id])
            psi[:] = ff
        _DT(self, psi, big_psi)
        return psi
    
    def Dc(self, psi):
        return psi
        """Forward propagator"""
         # memory for result
        flg = chunking_flg(locals().values())        
        if flg:
            big_psi = np.empty([len(psi), self.nq, self.nq], dtype="complex64")
        else:
            big_psi = cp.empty([len(psi), self.nq, self.nq], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _Dc(self, big_psi, psi):
    
            ff = psi.copy()            
            ff = cp.fft.ifft2(cp.fft.fft2(ff)*self.fkerc[cp.cuda.Device().id])
            big_psi[:] = ff

        _Dc(self, big_psi, psi)
        return big_psi

    def DcT(self, big_psi):
        """Adjoint propagator"""
        return big_psi
        # memory for result
        flg = chunking_flg(locals().values())        
        if flg:
            psi = np.empty([len(big_psi), self.nq, self.nq], dtype="complex64")
        else:
            psi = cp.empty([len(big_psi), self.nq, self.nq], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _DcT(self, psi, big_psi):            
            # pad to the probe size
            ff = big_psi.copy()
            psi[:] = cp.fft.ifft2(cp.fft.fft2(ff)/self.fkerc[cp.cuda.Device().id])
            
        _DcT(self, psi, big_psi)
        return psi      
    # def D(self, psi):
    #     """Forward propagator"""
    #      # memory for result
    #     flg = chunking_flg(locals().values())        
    #     if flg:
    #         big_psi = np.empty([len(psi), self.n, self.n], dtype="complex64")
    #     else:
    #         big_psi = cp.empty([len(psi), self.n, self.n], dtype="complex64")
        
    #     @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
    #     def _D(self, big_psi, psi):
    #         ff = psi.copy()
    #         ff = self._fwd_pad(ff)
    #         v = cp.ones(2*self.nq,dtype='float32')
    #         v[:self.nq//2] = cp.sin(cp.linspace(0,1,self.nq//2)*cp.pi/2)
    #         v[-self.nq//2:] = cp.cos(cp.linspace(0,1,self.nq//2)*cp.pi/2)
    #         v = cp.outer(v,v)
    #         ff *= v*2        
    #         ff = cp.fft.ifft2(cp.fft.fft2(ff)*self.fker[cp.cuda.Device().id])
    #         ff = ff[:,self.nq//2:-self.nq//2,self.nq//2:-self.nq//2]
            
    #         big_psi[:] = ff[:, self.pad : self.nq - self.pad, self.pad : self.nq - self.pad]
            
        
    #     _D(self, big_psi, psi)
    #     return big_psi


    # def DT(self, big_psi):
    #     """Adjoint propagator"""
    #     # memory for result
    #     flg = chunking_flg(locals().values())        
    #     if flg:
    #         psi = np.empty([len(big_psi), self.nq, self.nq], dtype="complex64")
    #     else:
    #         psi = cp.empty([len(big_psi), self.nq, self.nq], dtype="complex64")
        
    #     @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
    #     def _DT(self, psi, big_psi):
    #         # pad to the probe size
    #         ff = cp.pad(big_psi, ((0, 0), (self.pad, self.pad), (self.pad, self.pad)))
    #         ff = cp.pad(ff,((0,0),(self.nq//2,self.nq//2),(self.nq//2,self.nq//2)))
    #         ff = cp.fft.ifft2(cp.fft.fft2(ff)/self.fker[cp.cuda.Device().id])
    #         v = cp.ones(2*self.nq,dtype='float32')
    #         v[:self.nq//2] = cp.sin(cp.linspace(0,1,self.nq//2)*cp.pi/2)
    #         v[-self.nq//2:] = cp.cos(cp.linspace(0,1,self.nq//2)*cp.pi/2)
    #         v = cp.outer(v,v)
    #         ff *= v*2        
    #         psi[:] = self._adj_pad(ff)
    #     _DT(self, psi, big_psi)
    #     return psi
    
    # def Dc(self, psi):
    #     """Forward propagator"""
    #      # memory for result
    #     flg = chunking_flg(locals().values())        
    #     if flg:
    #         big_psi = np.empty([len(psi), self.nq, self.nq], dtype="complex64")
    #     else:
    #         big_psi = cp.empty([len(psi), self.nq, self.nq], dtype="complex64")
        
    #     @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
    #     def _Dc(self, big_psi, psi):
    
    #         ff = psi.copy()
    #         ff = self._fwd_pad(ff)
    #         v = cp.ones(2*self.nq,dtype='float32')
    #         v[:self.nq//2] = cp.sin(cp.linspace(0,1,self.nq//2)*cp.pi/2)
    #         v[-self.nq//2:] = cp.cos(cp.linspace(0,1,self.nq//2)*cp.pi/2)
    #         v = cp.outer(v,v)
    #         ff *= v*2        
    #         ff = cp.fft.ifft2(cp.fft.fft2(ff)*self.fkerc[cp.cuda.Device().id])
    #         ff = ff[:,self.nq//2:-self.nq//2,self.nq//2:-self.nq//2]
    #         big_psi[:] = ff

    #     _Dc(self, big_psi, psi)
    #     return big_psi

    # def DcT(self, big_psi):
    #     """Adjoint propagator"""
    #     # memory for result
    #     flg = chunking_flg(locals().values())        
    #     if flg:
    #         psi = np.empty([len(big_psi), self.nq, self.nq], dtype="complex64")
    #     else:
    #         psi = cp.empty([len(big_psi), self.nq, self.nq], dtype="complex64")
        
    #     @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
    #     def _DcT(self, psi, big_psi):            
    #         # pad to the probe size
    #         ff = big_psi.copy()
    #         ff = cp.pad(ff,((0,0),(self.nq//2,self.nq//2),(self.nq//2,self.nq//2)))
    #         ff = cp.fft.ifft2(cp.fft.fft2(ff)/self.fkerc[cp.cuda.Device().id])
    #         v = cp.ones(2*self.nq,dtype='float32')
    #         v[:self.nq//2] = cp.sin(cp.linspace(0,1,self.nq//2)*cp.pi/2)
    #         v[-self.nq//2:] = cp.cos(cp.linspace(0,1,self.nq//2)*cp.pi/2)
    #         v = cp.outer(v,v)
    #         ff *= v*2        
    #         psi[:] = self._adj_pad(ff)
            
    #     _DcT(self, psi, big_psi)
    #     return psi    

    def G(self,psi):
        if self.lam==0:
            return np.zeros_like(psi)
            
        flg = chunking_flg(locals().values())        
        if flg:
            res = np.empty([len(psi), self.nq, self.nq], dtype="complex64")
        else:
            res = cp.empty([len(psi), self.nq, self.nq], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _G0(self, res, psi):
            stencil = cp.array([1, -2, 1]).astype('float32')            
            res[:] = ndimage.convolve1d(psi, stencil, axis=1)
            res[:] += ndimage.convolve1d(psi, stencil, axis=2)
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=1, axis_inp=1)
        def _G1(self, res, res0, psi):
            stencil = cp.array([1, -2, 1]).astype('float32')            
            res[:] = res0 + ndimage.convolve1d(psi, stencil, axis=0)
        
        _G0(self,res,psi)
        _G1(self,res,res,psi)
        return res
            
    def gradientF(self, reused, d):
        big_psi = reused["big_psi"]
        flg = chunking_flg(locals().values())        
        if flg:
            gradF = np.empty([len(big_psi), self.nq, self.nq], dtype="complex64")
        else:
            gradF = cp.empty([len(big_psi), self.nq, self.nq], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _gradientF(self, gradF, big_psi, d):
            td = d * (big_psi / (cp.abs(big_psi) + self.eps))
            gradF[:] = self.DT(2 * (big_psi - td))
            
        _gradientF(self, gradF, big_psi, d)
        reused['gradF'] = gradF                
    
    
    def gradient_q(self, scode, gradF):
        flg = chunking_flg(locals().values())        
        if flg:
            gradq = []
            for igpu in range(self.ngpus):
                with cp.cuda.Device(igpu):
                    gradq.append(cp.zeros([self.nq, self.nq], dtype="complex64"))
        else:
            gradq = cp.zeros([self.nq, self.nq], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)                
        def _gradient_q(self, gradq, scode, gradF):
            gradq[:] += cp.sum(cp.conj(scode) * self.DcT(gradF),axis=0)        
        _gradient_q(self, gradq, scode, gradF)

        if flg:
            for k in range(1,len(gradq)):
                gradq[0] += gradq[k]
            gradq = gradq[0]    
            
        return gradq
   
    def gradient_r(self, ri, r, gradF, code, q):
        
        flg = chunking_flg(locals().values())        
        if flg:
            gradr = np.empty([len(ri), 2], dtype="float32")
        else:
            gradr = cp.empty([len(ri), 2], dtype="float32")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)                               
        def _gradient_r(self, gradr, ri, r, gradF, code, q):

            # frequencies
            xi1 = cp.fft.fftfreq(self.npatch).astype("float32")
            xi2, xi1 = cp.meshgrid(xi1, xi1)

            # multipliers in frequencies
            w = cp.exp(
                -2 * cp.pi * 1j * (xi2 * r[:, 1, None, None] + xi1 * r[:, 0, None, None])
            ).astype('complex64')

            # Gradient parts
            tmp = self.E(ri, code)
            
            tmp = cp.fft.fft2(tmp)
            
            dt1 = cp.fft.ifft2(w * xi1 * tmp)
            dt2 = cp.fft.ifft2(w * xi2 * tmp)
            dt1 = -2 * cp.pi * 1j * dt1[:, self.ex : self.nq + self.ex, self.ex : self.nq + self.ex]
            dt2 = -2 * cp.pi * 1j * dt2[:, self.ex : self.nq + self.ex, self.ex : self.nq + self.ex]

            # inner product with gradF
            dm1 = self.Dc(dt1*q)
            dm2 = self.Dc(dt2*q)
            gradr[:, 0] = redot(gradF, dm1, axis=(1, 2))
            gradr[:, 1] = redot(gradF, dm2, axis=(1, 2))
        
        _gradient_r(self, gradr, ri, r, gradF, code, q)
        return gradr
    
    def gradients(self, vars, reused):
        (q, code, ri, r) = (vars["q"], vars["code"], vars["ri"], vars["r"])
        (gradF, scode) = (reused["gradF"], reused["scode"])
        dq = self.gradient_q(scode, gradF)
        dr = self.gradient_r(ri, r, gradF, code, q)
        grads = {"q": dq, "r": dr}        
        return grads
    
    def minF(self, big_psi, d):
        flg = chunking_flg(locals().values())        
        if flg:
            res = []
            for igpu in range(self.ngpus):
                with cp.cuda.Device(igpu):
                    res.append(cp.zeros(1, dtype="float32"))
        else:
            res = cp.zeros(1, dtype="float32")

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)          
        def _minF(self, res, big_psi, d):            
            res[:]+=cp.linalg.norm(cp.abs(big_psi) - d) ** 2
        _minF(self, res, big_psi, d)
        if flg:
            for k in range(1,len(res)):
                res[0] += res[k]
            res = res[0]    
        res = res[0]    
        return res

    def calc_reused(self, reused, vars):
        """Calculate variables reused during calculations"""
        (q, code, ri, r) = (vars["q"], vars['code'], vars["ri"], vars["r"])
        flg = chunking_flg(vars.values())   
        if flg:
            scode = np.zeros([len(ri), self.nq, self.nq], dtype="complex64")
            big_psi = np.zeros([len(ri), self.n, self.n], dtype="complex64")
        else:
            scode = cp.zeros([len(ri), self.nq, self.nq], dtype="complex64")
            big_psi = cp.zeros([len(ri), self.n, self.n], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _calc_reused1(self, scode, ri, r, code):
            scode[:] = self.S(ri, r, code)
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _calc_reused2(self, big_psi, scode, q):
            big_psi[:] = self.D(self.Dc(scode*q))                    

        _calc_reused1(self, scode, ri, r, code)        
        _calc_reused2(self, big_psi, scode, q)            
        
        reused["scode"] = scode
        reused["big_psi"] = big_psi
    
    def hessian_F(self, big_psi, dbig_psi1, dbig_psi2, d):    
        flg = chunking_flg(locals().values())        
        if flg:
            res = []
            for igpu in range(self.ngpus):
                with cp.cuda.Device(igpu):
                    res.append(cp.zeros(1, dtype="float32"))
        else:
            res = cp.zeros(1, dtype="float32")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _hessian_F(self, res, big_psi, dbig_psi1, dbig_psi2, d):
            l0 = big_psi / (cp.abs(big_psi) + self.eps)
            d0 = d / (cp.abs(big_psi) + self.eps)
            v1 = cp.sum((1 - d0) * reprod(dbig_psi1, dbig_psi2))
            v2 = cp.sum(d0 * reprod(l0, dbig_psi1) * reprod(l0, dbig_psi2))
            res[:] = 2 * (v1 + v2)
            return res
        _hessian_F(self, res, big_psi, dbig_psi1, dbig_psi2, d)
        if flg:
            for k in range(1,len(res)):
                res[0] += res[k]
        res = res[0]    
        return res

    def redot_batch(self,x,y):
        flg = chunking_flg(locals().values())        
        if flg:
            res = []
            for igpu in range(self.ngpus):
                with cp.cuda.Device(igpu):
                    res.append(cp.zeros(1, dtype="float32"))
        else:
            res = cp.zeros(1, dtype="float32")
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _redot(self, res, x,y):
            res[:] += redot(x, y)
            return res
        _redot(self, res, x,y)
        if flg:
            for k in range(1,len(res)):
                res[0] += res[k]
            res = res[0]    
        res = res[0]    
        return res
        
    def hessian(self, scode, dq1, dq2, big_psi, d):    
        dm1 = dq1 * scode
        dm2 = dq2 * scode
        dn1 = self.Dc(dm1)
        dn2 = self.Dc(dm2)
        Ddn1 = self.D(dn1)
        Ddn2 = self.D(dn2)
        return self.hessian_F(big_psi, Ddn1, Ddn2, d)
            
    def calc_beta(self, vars, grads, etas, reused, d):
        (q, code, ri, r) = (vars["q"], vars["code"], vars["ri"], vars["r"])
        (scode, big_psi, gradF) = (reused["scode"], reused["big_psi"], reused["gradF"])
        (dq1, dr1) = (grads["q"], grads["r"])
        (dq2, dr2) = (etas["q"], etas["r"])        

        beta = (self.hessian(scode, dq1, dq2, big_psi, d).get()/
                self.hessian(scode, dq2, dq2, big_psi, d).get())        
        return beta
    

    def calc_alpha(self, vars, grads, etas, reused, d):
        (q, code, ri, r) = (vars["q"], vars["code"], vars["ri"], vars["r"])
        (dq1, dr1) = (grads["q"], grads["r"])
        (dq2, dr2) = (etas["q"], etas["r"])       
        (scode, big_psi, gradF) = (reused["scode"], reused["big_psi"], reused["gradF"])
        
        top = -redot(dq1, dq2).get()
        bottom = self.hessian(scode, dq2, dq2, big_psi, d).get()        
        return top/bottom,top,bottom
    
    def fwd(self,ri,r,code,q):
                
        # memory for result
        flg = chunking_flg(locals().values())        
        if flg:
            res = np.empty([len(ri), self.n, self.n], dtype="complex64")
        else:
            res = cp.empty([len(ri), self.n, self.n], dtype="complex64")
                
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _fwd(self,res,ri,r,code,q):            
            scode = self.S(ri,r,code)
            res[:] = self.D(self.Dc(scode * q))
        
        _fwd(self,res,ri,r,code,q)
        return res

    def plot_debug(self, vars, etas, top, bottom, alpha, d, i):
        """Check the minimization functional behaviour"""
        if i % self.vis_step == 0 and self.vis_step != -1 and self.show:
            (q, code, ri, r) = (vars["q"], vars["code"], vars["ri"], vars["r"])
            (dq2, dr2) = (etas["q"], etas["r"])
                        
            npp = 9
            errt = cp.zeros(npp * 2)
            errt2 = cp.zeros(npp * 2)
            for k in range(0, npp * 2):
                qt = q + (alpha * k / (npp - 1)) * dq2
                rt = r + (alpha * k / (npp - 1)) * dr2
                tmp=self.fwd(ri,rt,code,qt)                
                errt[k] = self.minF(tmp, d)
                
            t = alpha * (cp.arange(2 * npp)) / (npp - 1)            
            tmp = self.fwd(ri,r,code,q)
            errt2 = self.minF(tmp, d)
            errt2 = errt2 - top * t + 0.5 * bottom * t**2

            plt.plot(
                alpha * cp.arange(2 * npp).get() / (npp - 1),
                errt.get(),
                ".",
                label="real",
            )
            plt.plot(
                alpha * cp.arange(2 * npp).get() / (npp - 1),
                errt2.get(),
                ".",
                label="approximation",
            )
            plt.legend()
            plt.grid()
            plt.show()
        

    def vis_debug(self, vars, i):
        """Visualization and data saving"""
        if i % self.vis_step == 0 and self.vis_step != -1:
            pass
            # (q, u) = (vars["q"], vars["u"])
            
            # mshow_complex(u[u.shape[0]//2].real+1j*u[:,u.shape[1]//2].real,self.show)
            # mshow_polar(q,self.show)
            # write_tiff(u.real,f'{self.path_out}/rec_u_real/{i:04}')
            # # write_tiff(u.imag,f'{self.path_out}/rec_u_imag/{i:04}')
            # write_tiff(u[self.npsi//2].real,f'{self.path_out}/rec_uz/{i:04}')
            # write_tiff(u[:,self.npsi//2].real,f'{self.path_out}/rec_uy/{i:04}')            
            # write_tiff(cp.angle(q),f'{self.path_out}/rec_prb_angle/{i:04}')
            # write_tiff(cp.abs(q),f'{self.path_out}/rec_prb_abs/{i:04}')
            # if self.show:
            #     plt.plot(vars['r'][:,1].get()-vars['r_init'][:,1].get(),'.')
            #     plt.plot(vars['r'][:,0].get()-vars['r_init'][:,0].get(),'.')
            #     plt.show()

    def error_debug(self,vars, reused, d, i):
        """Visualization and data saving"""
        if i % self.err_step == 0 and self.err_step != -1:
            err = self.minF(reused["big_psi"], d)
            print(f"{i}) {err=:1.5e}", flush=True)
            vars["table"].loc[len(vars["table"])] = [i, err.get(), time.time()]
            name = f'{self.path_out}/conv.csv'
            os.makedirs(os.path.dirname(name), exist_ok=True)
            vars["table"].to_csv(name, index=False)
    