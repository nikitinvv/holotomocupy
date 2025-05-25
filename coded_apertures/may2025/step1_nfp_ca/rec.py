import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time
from concurrent.futures import ThreadPoolExecutor
import cupyx.scipy.ndimage as ndimage
import os
import sys


sys.path.insert(0, '..')
from cuda_kernels import *
from utils import *
from chunking import gpu_batch
import copy

cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)

class Rec:
    def __init__(self, args):

        for attr in args.__dict__:
            setattr(self, attr, copy.deepcopy(args.__dict__[attr]))
        
        # allocate gpu and pinned memory buffers
        # calculate max buffer size
        nbytes = 32 * self.nchunk * self.npatch**2 * np.dtype("complex64").itemsize            
        
        # create CUDA streams and allocate pinned memory
        self.stream = [[[] for _ in range(3)] for _ in range(self.ngpus)]
        self.pinned_mem = [[] for _ in range(self.ngpus)]
        self.gpu_mem = [[] for _ in range(self.ngpus)]
        self.fker = [[] for _ in range(self.ngpus)]
        self.pool_inp = [[] for _ in range(self.ngpus)]
        self.pool_out = [[] for _ in range(self.ngpus)]
        self.pool = ThreadPoolExecutor(16)

        for igpu in range(self.ngpus):
            with cp.cuda.Device(igpu):
                self.pinned_mem[igpu] = cp.cuda.alloc_pinned_memory(nbytes)
                self.gpu_mem[igpu] = cp.cuda.alloc(nbytes)
                for k in range(3):
                    self.stream[igpu][k] = cp.cuda.Stream(non_blocking=False)
                fx = cp.fft.fftfreq(self.nq * 2, d=self.voxelsize).astype("float32")
                [fx, fy] = cp.meshgrid(fx, fx)
                self.fker[igpu] = cp.exp(-1j * cp.pi * self.wavelength * self.distance * (fx**2 + fy**2))#*unimod
                self.pool_inp[igpu] = ThreadPoolExecutor(16 // self.ngpus)    
                self.pool_out[igpu] = ThreadPoolExecutor(16 // self.ngpus)

    def BH(self, d, dref, vars):
        d = np.sqrt(d)
        dref = np.sqrt(cp.asarray(dref))

        reused = {}
        for i in range(self.niter):
            self.calc_reused(reused, vars)
            self.gradientF(reused, d)

            self.error_debug(vars, reused, d, dref, i)            
            self.vis_debug(vars, i)
            
            grads = self.gradients(vars, reused, dref)
            
            if i == 0:
                etas = {}
                etas["psi"] = mulc(grads["psi"], -self.rho[0] ** 2, self.pool)
                etas["q"] = mulc(grads["q"], -self.rho[1] ** 2, self.pool)
                etas["r"] = mulc(grads["r"], -self.rho[2] ** 2, self.pool)
            else:
                beta = self.calc_beta(vars, grads, etas, reused, d, dref)
                linear(etas['psi'],grads['psi'],etas['psi'],-self.rho[0]**2,beta,self.pool)                                
                linear(etas['q'],grads['q'],etas['q'],-self.rho[1]**2,beta,self.pool)                                
                linear(etas['r'],grads['r'],etas['r'],-self.rho[2]**2,beta,self.pool)                                
                
            alpha, top, bottom = self.calc_alpha(vars, grads, etas, reused, d, dref)
            self.plot_debug(vars, etas, top, bottom, alpha, d, dref, i)
            linear(vars["psi"],vars["psi"],etas['psi'],1,alpha,self.pool)                                
            linear(vars["q"],vars["q"],etas['q'],1,alpha,self.pool)                                
            linear(vars["r"],vars["r"],etas['r'],1,alpha,self.pool)                                

        return vars
    
    
    def E(self, ri, psi):
        """Extract patches"""

        flg = chunking_flg(locals().values())
        # memory for result
        if flg:
            patches = np.empty([len(ri), self.npatch, self.npatch], dtype="complex64")
        else:
            patches = cp.empty([len(ri), self.npatch, self.npatch], dtype="complex64")

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _E(self, patches, ri, psi):
            stx = self.npsi // 2 - ri[:, 1] - self.npatch // 2
            sty = self.npsi // 2 - ri[:, 0] - self.npatch // 2
            Efast_kernel(
                (
                    int(cp.ceil(self.npatch / 16)),
                    int(cp.ceil(self.npatch / 16)),
                    int(cp.ceil(len(ri) / 4)),
                ),
                (16, 16, 4),
                (patches, psi, stx, sty, len(ri), self.npatch, self.npsi),
            )

        _E(self, patches, ri, psi)

        return patches

    def ET(self, ri, patches):
        """Place patches, note only on 1 gpu for now"""
        
        flg = chunking_flg(locals().values())        
        if flg:
            psi = []
            for igpu in range(self.ngpus):
                with cp.cuda.Device(igpu):
                    psi.append(cp.zeros([self.npsi, self.npsi], dtype="complex64"))
        else:
            psi = cp.zeros([self.npsi, self.npsi], dtype="complex64")

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _ET(self, psi, ri, patches):
            """Adjoint extract patches"""
            stx = self.npsi // 2 - ri[:, 1] - self.npatch // 2
            sty = self.npsi // 2 - ri[:, 0] - self.npatch // 2
            ETfast_kernel(
                (
                    int(cp.ceil(self.npatch / 16)),
                    int(cp.ceil(self.npatch / 16)),
                    int(cp.ceil(len(ri) / 4)),
                ),
                (16, 16, 4),
                (patches, psi, stx, sty, len(ri), self.npatch, self.npsi),
            )

        _ET(self, psi, ri, patches)
        if flg:
            for k in range(1,len(psi)):
                psi[0] += psi[k]
            psi = psi[0]    
        return psi

    def S(self, ri, r, psi):
        flg = chunking_flg(locals().values())        
        if flg:
            patches = np.empty([len(ri), self.nq, self.nq], dtype="complex64")
        else:
            patches = cp.empty([len(ri), self.nq, self.nq], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _S(self, patches, ri, r, psi):
            """Extract patches with subpixel shift"""
            psir = self.E(ri, psi)

            x = cp.fft.fftfreq(self.npatch).astype("float32")
            [y, x] = cp.meshgrid(x, x)
            tmp = cp.exp(
                -2 * cp.pi * 1j * (y * r[:, 1, None, None] + x * r[:, 0, None, None])
            ).astype("complex64")
            psir = cp.fft.ifft2(tmp * cp.fft.fft2(psir))
            patches[:] = psir[:, self.ex : self.nq + self.ex, self.ex : self.nq + self.ex]

        _S(self, patches, ri, r, psi)
        return patches

    def ST(self, ri, r, patches):
        flg = chunking_flg(locals().values())        
        if flg:
            psi = []
            for igpu in range(self.ngpus):
                with cp.cuda.Device(igpu):
                    psi.append(cp.zeros([self.npsi, self.npsi], dtype="complex64"))
        else:
            psi = cp.zeros([self.npsi, self.npsi], dtype="complex64")

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)                
        def _ST(self, psi, ri,r,patches):
            """Adjont extract patches with subpixel shift"""
            psir = cp.pad(patches, ((0, 0), (self.ex, self.ex), (self.ex, self.ex)))

            x = cp.fft.fftfreq(self.npatch).astype("float32")
            [y, x] = cp.meshgrid(x, x)
            tmp = cp.exp(
                2 * cp.pi * 1j * (y * r[:, 1, None, None] + x * r[:, 0, None, None])
            ).astype("complex64")
            psir = cp.fft.ifft2(tmp * cp.fft.fft2(psir))

            psi[:] += self.ET(ri,psir)

        _ST(self, psi, ri,r,patches)
        if flg:
            for k in range(1,len(psi)):
                psi[0] += psi[k]
            psi = psi[0]    
        return psi


    def _fwd_pad(self,f):
        """Fwd data padding"""
        [npos, n] = f.shape[:2]
        fpad = cp.zeros([npos, 2*n, 2*n], dtype='complex64')
        pad_kernel((int(cp.ceil(2*n/32)), int(cp.ceil(2*n/32)), npos),
                (32, 32, 1), (fpad, f, n, npos, 0))
        return fpad/2
    
    def _adj_pad(self, fpad):
        """Adj data padding"""
        [npos, n] = fpad.shape[:2]
        n //= 2
        f = cp.zeros([npos, n, n], dtype='complex64')
        pad_kernel((int(cp.ceil(2*n/32)), int(cp.ceil(2*n/32)), npos),
                (32, 32, 1), (fpad, f, n, npos, 1))
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
            ff = self._fwd_pad(psi)
            ff *= 2        
            ff = cp.fft.ifft2(cp.fft.fft2(ff)*self.fker[cp.cuda.Device().id])
            ff = ff[:,self.nq//2:-self.nq//2,self.nq//2:-self.nq//2]            
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
            # ff = big_psi.copy
            ff = cp.pad(ff,((0,0),(self.nq//2,self.nq//2),(self.nq//2,self.nq//2)))
            ff = cp.fft.ifft2(cp.fft.fft2(ff)/self.fker[cp.cuda.Device().id])
            ff *= 2        
            psi[:] = self._adj_pad(ff)
        _DT(self, psi, big_psi)
        return psi
    
    def G(self,psi):
        if self.lam==0:
            return np.zeros_like(psi)
            
        flg = chunking_flg(locals().values())        
        if flg:
            res = np.empty_like(psi)
        else:
            res = cp.empty_like(psi)
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _G(self, res, psi):
            stencil = cp.array([1, -2, 1]).astype('float32')            
            res[:] = ndimage.convolve1d(psi, stencil, axis=1)
            res[:] += ndimage.convolve1d(psi, stencil, axis=2)
                
        _G(self,res,psi)
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

    def gradientF_q(self, q, dref):
        tmp = self.D(q[cp.newaxis])        
        td = dref * (tmp / (cp.abs(tmp) + self.eps))
        return  self.DT(2 * (tmp - td))[0]
    
    def gradient_psi(self, ri, r, gradF, q):
        flg = chunking_flg(locals().values())        
        if flg:
            gradpsi = []
            for igpu in range(self.ngpus):
                with cp.cuda.Device(igpu):
                    gradpsi.append(cp.zeros([self.npsi, self.npsi], dtype="complex64"))
        else:
            gradpsi = cp.zeros([self.npsi, self.npsi], dtype="complex64")
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)                
        def _gradient_psi(self, gradpsi, ri, r, gradF, q):            
            gradpsi[:] += self.ST(ri, r,cp.conj(q) * gradF)
        _gradient_psi(self, gradpsi, ri, r, gradF, q)
        if flg:
            for k in range(1,len(gradpsi)):
                gradpsi[0] += gradpsi[k]
            gradpsi = gradpsi[0]    
        return gradpsi
    
    def gradient_q(self, spsi, gradF, q, dref):
        flg = chunking_flg(locals().values())        
        if flg:
            gradq = []
            for igpu in range(self.ngpus):
                with cp.cuda.Device(igpu):
                    gradq.append(cp.zeros([self.nq, self.nq], dtype="complex64"))
        else:
            gradq = cp.zeros([self.nq, self.nq], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)                
        def _gradient_q(self, gradq, spsi, gradF):
            gradq[:] += cp.sum(cp.conj(spsi) * gradF,axis=0)        
        _gradient_q(self, gradq, spsi, gradF)

        if flg:
            for k in range(1,len(gradq)):
                gradq[0] += gradq[k]
            gradq = gradq[0]    

        # regularization and probe fitting terms
        gradq += 2*self.lam*self.G(self.G(q))
        gradq += self.gradientF_q(q,dref)
            
        return gradq
   
    def gradient_r(self, ri, r, gradF, psi, q):
        
        flg = chunking_flg(locals().values())        
        if flg:
            gradr = np.empty([len(ri), 2], dtype="float32")
        else:
            gradr = cp.empty([len(ri), 2], dtype="float32")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)                               
        def _gradient_r(self, gradr, ri, r, gradF, psi, q):

            # frequencies
            xi1 = cp.fft.fftfreq(self.npatch).astype("float32")
            xi2, xi1 = cp.meshgrid(xi1, xi1)

            # multipliers in frequencies
            w = cp.exp(
                -2 * cp.pi * 1j * (xi2 * r[:, 1, None, None] + xi1 * r[:, 0, None, None])
            )

            # Gradient parts
            tmp = self.E(ri, psi)
            tmp = cp.fft.fft2(tmp)
            
            dt1 = cp.fft.ifft2(w * xi1 * tmp)
            dt2 = cp.fft.ifft2(w * xi2 * tmp)
            dt1 = -2 * cp.pi * 1j * dt1[:, self.ex : self.nq + self.ex, self.ex : self.nq + self.ex]
            dt2 = -2 * cp.pi * 1j * dt2[:, self.ex : self.nq + self.ex, self.ex : self.nq + self.ex]

            # inner product with gradF
            
            gradr[:, 0] = redot(gradF, q * dt1, axis=(1, 2))
            gradr[:, 1] = redot(gradF, q * dt2, axis=(1, 2))
        
        _gradient_r(self, gradr, ri, r, gradF, psi, q)
        return gradr
    
    def gradients(self, vars, reused, dref):
        (q, psi, ri, r) = (vars["q"], vars["psi"], vars["ri"], vars["r"])
        (gradF, spsi) = (reused["gradF"], reused["spsi"])

        dpsi = self.gradient_psi(ri, r, gradF, q)
        dprb = self.gradient_q(spsi, gradF, q, dref)
        dr = self.gradient_r(ri, r, gradF, psi, q)
        grads = {"psi": dpsi, "q": dprb, "r": dr}        
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
        (q, psi, ri, r) = (vars["q"], vars["psi"], vars["ri"], vars["r"])
        flg = chunking_flg(vars.values())   
        if flg:
            spsi = np.empty([len(ri), self.nq, self.nq], dtype="complex64")
            big_psi = np.empty([len(ri), self.n, self.n], dtype="complex64")
        else:
            spsi = cp.empty([len(ri), self.nq, self.nq], dtype="complex64")
            big_psi = cp.empty([len(ri), self.n, self.n], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _calc_reused1(self, spsi, ri, r, psi):
            spsi[:] = self.S(ri, r, psi)
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _calc_reused2(self, big_psi, spsi, q):
            big_psi[:] = self.D(spsi * q)

        _calc_reused1(self, spsi, ri, r, psi)
        _calc_reused2(self, big_psi, spsi, q)            
        
        reused["spsi"] = spsi
        reused["big_psi"] = big_psi
    
    def hessian_F(self, big_psi, dbig_psi1, dbig_psi2, d):    
        flg = chunking_flg(locals().values())        
        if flg:
            res = []
            for igpu in range(self.ngpus):
                with cp.cuda.Device(igpu):
                    res.append(cp.zeros(1, dtype="complex64"))
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

    def hessianFq(self, big_q,dbig_q1,dbig_q2,dref):        
        l0 = big_q/(np.abs(big_q) + self.eps)
        d0 = dref/(np.abs(big_q) + self.eps)
        v1 = np.sum((1-d0)*reprod(dbig_q1,dbig_q2))
        v2 = np.sum(d0*reprod(l0,dbig_q1)*reprod(l0,dbig_q2))        
        return 2*(v1+v2)
    
    def calc_beta(self, vars, grads, etas, reused, d, dref):
        (q, psi, ri, r) = (vars["q"], vars["psi"], vars["ri"], vars["r"])
        (spsi, big_psi, gradF) = (reused["spsi"], reused["big_psi"], reused["gradF"])
        (dpsi1, dq1, dr1) = (grads["psi"], grads["q"], grads["r"])
        (dpsi2, dq2, dr2) = (etas["psi"], etas["q"], etas["r"])        

        flg = (chunking_flg(vars.values()) 
               or chunking_flg(reused.values()) 
               or chunking_flg(grads.values()) 
               or chunking_flg(etas.values()))                 
        if flg:
            res = []
            for igpu in range(self.ngpus):
                with cp.cuda.Device(igpu):
                    res.append(cp.zeros([2], dtype="float32"))
        else:
            res = cp.zeros([2], dtype="float32")

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _calc_beta(self, res, ri, r, spsi, big_psi, gradF, dr1, dr2, d, q, dq1, dq2, psi,dpsi1, dpsi2):            
            # note scaling with rho
            dpsi1 = dpsi1 * self.rho[0] ** 2
            dq1 = dq1 * self.rho[1] ** 2
            dr1 = dr1 * self.rho[2] ** 2
            # frequencies
            xi1 = cp.fft.fftfreq(self.npatch).astype("float32")
            [xi2, xi1] = cp.meshgrid(xi1, xi1)

            # multipliers in frequencies            
            dr1 = dr1[:, :, cp.newaxis, cp.newaxis]
            dr2 = dr2[:, :, cp.newaxis, cp.newaxis]
            w = cp.exp(
                -2 * cp.pi * 1j * (xi2 * r[:, 1, None, None] + xi1 * r[:, 0, None, None])
            )
            w1 = xi1 * dr1[:, 0] + xi2 * dr1[:, 1]
            w2 = xi1 * dr2[:, 0] + xi2 * dr2[:, 1]
            w12 = (
                xi1**2 * dr1[:, 0] * dr2[:, 0]
                + xi1 * xi2 * (dr1[:, 0] * dr2[:, 1] + dr1[:, 1] * dr2[:, 0])
                + xi2**2 * dr1[:, 1] * dr2[:, 1]
            )
            w22 = (
                xi1**2 * dr2[:, 0] ** 2
                + 2 * xi1 * xi2 * (dr2[:, 0] * dr2[:, 1])
                + xi2**2 * dr2[:, 1] ** 2
            )

            # DT, D2T terms
            tmp = self.E(ri, dpsi1)
            tmp = cp.fft.fft2(tmp)
            sdpsi1 = cp.fft.ifft2(w * tmp)[:, self.ex : self.nq + self.ex, self.ex : self.nq + self.ex]
            dt12 = -2 * cp.pi * 1j * cp.fft.ifft2(w * w2 * tmp)[:, self.ex : self.nq + self.ex, self.ex : self.nq + self.ex]

            tmp = self.E(ri, dpsi2)
            tmp  = cp.fft.fft2(tmp)
            sdpsi2 = cp.fft.ifft2(w * tmp)[:, self.ex : self.nq + self.ex, self.ex : self.nq + self.ex]
            dt21 = -2 * cp.pi * 1j * cp.fft.ifft2(w * w1 * tmp)[:, self.ex : self.nq + self.ex, self.ex : self.nq + self.ex]
            dt22 = -2 * cp.pi * 1j * cp.fft.ifft2(w * w2 * tmp)[:, self.ex : self.nq + self.ex, self.ex : self.nq + self.ex]

            tmp = self.E(ri, psi)
            tmp = cp.fft.fft2(tmp)
            dt1 = -2 * cp.pi * 1j * cp.fft.ifft2(w * w1 * tmp)[:, self.ex : self.nq + self.ex, self.ex : self.nq + self.ex]
            dt2 = -2 * cp.pi * 1j * cp.fft.ifft2(w * w2 * tmp)[:, self.ex : self.nq + self.ex, self.ex : self.nq + self.ex]
            d2t1 = -4 * cp.pi**2 * cp.fft.ifft2(w * w12 * tmp)[:, self.ex : self.nq + self.ex, self.ex : self.nq + self.ex]
            d2t2 = -4 * cp.pi**2 * cp.fft.ifft2(w * w22 * tmp)[:, self.ex : self.nq + self.ex, self.ex : self.nq + self.ex]

            # DM,D2M terms
            d2m1 = q * dt12 + q * dt21 + q * d2t1
            d2m1 += dq1 * sdpsi2 + dq2 * sdpsi1
            d2m1 += dq1 * dt2 + dq2 * dt1

            d2m2 = q * dt22 + q * dt22 + q * d2t2
            d2m2 += dq2 * sdpsi2 + dq2 * sdpsi2
            d2m2 += dq2 * dt2 + dq2 * dt2

            dm1 = dq1 * spsi + q * (sdpsi1 + dt1)
            dm2 = dq2 * spsi + q * (sdpsi2 + dt2)

            # top and bottom parts
            Ddm1 = self.D(dm1)
            Ddm2 = self.D(dm2)

            top = redot(gradF, d2m1) + self.hessian_F(big_psi, Ddm1, Ddm2, d)
            bottom = redot(gradF, d2m2) + self.hessian_F(big_psi, Ddm2, Ddm2, d)
            res[0] += top
            res[1] += bottom
            
        _calc_beta(self, res, ri, r, spsi, big_psi, gradF, dr1, dr2, d, q, dq1, dq2, psi,dpsi1, dpsi2)           
        if flg:
            for k in range(1,len(res)):
                res[0][0] += res[k][0]
                res[0][1] += res[k][1]
            res = res[0]      

        # regularization and probe fitting terms,
        gq1 = self.G(dq1)
        gq2 = self.G(dq2)
        Dq = self.D(q[cp.newaxis])[0]
        Ddq1 = self.D(dq1[cp.newaxis])[0]
        Ddq2 = self.D(dq2[cp.newaxis])[0] 
        res[0] += 2*self.lam*redot(gq1,gq2) + self.hessianFq(Dq,Ddq1,Ddq2,dref)
        res[1] += 2*self.lam*redot(gq2,gq2) + self.hessianFq(Dq,Ddq2,Ddq2,dref)
        
        top = cp.float32((res[0]).get()) 
        bottom = cp.float32((res[1]).get()) 
        return top/bottom
    

    def calc_alpha(self, vars, grads, etas, reused, d, dref):
        (q, psi, ri, r) = (vars["q"], vars["psi"], vars["ri"], vars["r"])
        (dpsi1, dq1, dr1) = (grads["psi"], grads["q"], grads["r"])
        (dpsi2, dq2, dr2) = (etas["psi"], etas["q"], etas["r"])
        (spsi, big_psi, gradF) = (reused["spsi"], reused["big_psi"], reused["gradF"])
        
        flg = (chunking_flg(vars.values()) 
               or chunking_flg(reused.values()) 
               or chunking_flg(grads.values()) 
               or chunking_flg(etas.values()))                 
        if flg:
            res = []
            for igpu in range(self.ngpus):
                with cp.cuda.Device(igpu):
                    res.append(cp.zeros([2], dtype="float32"))
        else:
            res = cp.zeros([2], dtype="float32")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _calc_alpha(self, res, ri, r, spsi, big_psi, gradF, dr1, dr2, d, q, dq2, psi, dpsi2):            
            # top part
            top = -redot(dr1, dr2)

            # frequencies
            xi1 = cp.fft.fftfreq(self.npatch).astype("float32")
            [xi2, xi1] = cp.meshgrid(xi1, xi1)

            # multipliers in frequencies
            dr = dr2[:, :, cp.newaxis, cp.newaxis]
            w = cp.exp(
                -2 * cp.pi * 1j * (xi2 * r[:, 1, None, None] + xi1 * r[:, 0, None, None])
            )
            w1 = xi1 * dr[:, 0] + xi2 * dr[:, 1]
            w2 = (
                xi1**2 * dr[:, 0] ** 2
                + 2 * xi1 * xi2 * (dr[:, 0] * dr[:, 1])
                + xi2**2 * dr[:, 1] ** 2
            )

            # DT,D2T terms, and spsi
            tmp = self.E(ri, dpsi2)
            tmp = cp.fft.fft2(tmp)
            sdpsi = cp.fft.ifft2(w * tmp)[:, self.ex : self.nq + self.ex, self.ex : self.nq + self.ex]
            dt2 = -2 * cp.pi * 1j * cp.fft.ifft2(w * w1 * tmp)[:, self.ex : self.nq + self.ex, self.ex : self.nq + self.ex]

            tmp = self.E(ri, psi)
            tmp = cp.fft.fft2(tmp)
            dt = -2 * cp.pi * 1j * cp.fft.ifft2(w * w1 * tmp)[:, self.ex : self.nq + self.ex, self.ex : self.nq + self.ex]
            d2t = -4 * cp.pi**2 * cp.fft.ifft2(w * w2 * tmp)[:, self.ex : self.nq + self.ex, self.ex : self.nq + self.ex]

            # DM and D2M terms
            d2m2 = q * (2 * dt2 + d2t) + 2 * dq2 * sdpsi + 2 * dq2 * dt
            dm = dq2 * spsi + q * (sdpsi + dt)

            # bottom part
            Ddm = self.D(dm)
            bottom = redot(gradF, d2m2) + self.hessian_F(big_psi, Ddm, Ddm, d)
            res[0] += top
            res[1] += bottom

        _calc_alpha(self, res, ri, r, spsi, big_psi, gradF, dr1, dr2, d, q, dq2, psi, dpsi2)
        if flg:
            for k in range(1,len(res)):
                res[0][0] += res[k][0]
                res[0][1] += res[k][1]                
            res = res[0]
        res[0] += -redot(dpsi1, dpsi2) - redot(dq1, dq2)

        # regularization and probe fitting terms,
        # fast on 1 GPU
        
        gq2 = self.G(dq2)        
        Dq = self.D(q[cp.newaxis])[0]
        Ddq2 = self.D(dq2[cp.newaxis])[0] 

        res[1] += 2*self.lam*redot(gq2,gq2) + self.hessianFq(Dq,Ddq2,Ddq2,dref)

        top = cp.float32((res[0]).get()) 
        bottom = cp.float32((res[1]).get()) 
        return top/bottom,top,bottom
    
    def fwd(self,ri,r,psi,q):
                
         # memory for result
        flg = chunking_flg(locals().values())        
        if flg:
            res = np.empty([len(ri), self.n, self.n], dtype="complex64")
        else:
            res = cp.empty([len(ri), self.n, self.n], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _fwd(self,res,ri,r,psi,q):
            res[:] = self.D(self.S(ri,r,psi) * q)
        
        _fwd(self,res,ri,r,psi,q)
        return res

    def plot_debug(self, vars, etas, top, bottom, alpha, d, dref, i):
        """Check the minimization functional behaviour"""
        if i % self.vis_step == 0 and self.vis_step != -1 and self.show:
            (q, psi, ri, r) = (vars["q"], vars["psi"], vars["ri"], vars["r"])
            (dq2, dpsi2, dr2) = (etas["q"], etas["psi"], etas["r"])
                        
            npp = 7
            errt = cp.zeros(npp * 2)
            errt2 = cp.zeros(npp * 2)
            for k in range(0, npp * 2):
                psit = psi + (alpha * k / (npp - 1)) * dpsi2
                qt = q + (alpha * k / (npp - 1)) * dq2
                rt = r + (alpha * k / (npp - 1)) * dr2
                tmp=self.fwd(ri,rt,psit,qt)
                errt[k] = self.minF(tmp, d) + \
                    self.lam*cp.linalg.norm(self.G(qt))**2 + \
                    cp.linalg.norm(cp.abs(self.D(qt[cp.newaxis]))-dref)**2

            t = alpha * (cp.arange(2 * npp)) / (npp - 1)
            tmp = self.fwd(ri,r,psi,q)
            errt2 = self.minF(tmp, d) + \
                self.lam*cp.linalg.norm(self.G(q))**2 + \
                cp.linalg.norm(cp.abs(self.D(q[cp.newaxis]))-dref)**2
            errt2 = errt2 - top * t + 0.5 * bottom * t**2

            plt.plot(
                alpha * cp.arange(2 * npp).get() / (npp - 1),
                errt.get(),
                ".",
                label="approximation",
            )
            plt.plot(
                alpha * cp.arange(2 * npp).get() / (npp - 1),
                errt2.get(),
                ".",
                label="real",
            )
            plt.legend()
            plt.grid()
            plt.show()
        

    def vis_debug(self, vars, i):
        """Visualization and data saving"""
        if i % self.vis_step == 0 and self.vis_step != -1:
            (q, psi) = (vars["q"], vars["psi"])
            mshow_polar(psi,self.show)
            mshow_polar(q,self.show)
            write_tiff(cp.angle(psi),f'{self.path_out}/rec_psi_angle/{i:04}')
            write_tiff(cp.abs(psi),f'{self.path_out}/rec_psi_abs/{i:04}')
            write_tiff(cp.angle(q),f'{self.path_out}/rec_prb_angle/{i:04}')
            write_tiff(cp.abs(q),f'{self.path_out}/rec_prb_abs/{i:04}')


    def error_debug(self,vars, reused, d, dref, i):
        """Visualization and data saving"""
        q = vars['q']
        if i % self.err_step == 0 and self.err_step != -1:
            err = self.minF(reused["big_psi"], d)+ self.lam*cp.linalg.norm(self.G(q))**2 + cp.linalg.norm(cp.abs(self.D(q[cp.newaxis]))-dref)**2
            print(f"{i}) {err=:1.5e}", flush=True)
            vars["table"].loc[len(vars["table"])] = [i, err.get(), time.time()]
            name = f'{self.path_out}/conv.csv'
            os.makedirs(os.path.dirname(name), exist_ok=True)
            vars["table"].to_csv(name, index=False)
    