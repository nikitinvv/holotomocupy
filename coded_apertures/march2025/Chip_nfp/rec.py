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
from holotomocupy.holo import *

cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)

class Rec:
    def __init__(self, args):
        ngpus = args.ngpus
        npsi = args.npsi
        nq = args.nq
        npatch = args.npatch
        n = args.n
        pad = args.pad
        npos = args.npos
        nchunk = args.nchunk
        ex = args.ex
        voxelsize = args.voxelsize
        wavelength = args.wavelength
        distance = args.distance
        eps = args.eps
        rho = args.rho
        crop = args.crop
        path_out = args.path_out
        show = args.show

        # allocate gpu and pinned memory buffers
        # calculate max buffer size
        nbytes = (
            10 * nchunk * npatch * npatch * np.dtype("complex64").itemsize            
        )

        # create CUDA streams and allocate pinned memory
        self.stream = [[[] for _ in range(3)] for _ in range(ngpus)]
        self.pinned_mem = [[] for _ in range(ngpus)]
        self.gpu_mem = [[] for _ in range(ngpus)]
        self.fker = [[] for _ in range(ngpus)]
        self.pool_inp = [[] for _ in range(ngpus)]
        self.pool_out = [[] for _ in range(ngpus)]
        self.pool = ThreadPoolExecutor(ngpus)    
        self.pad_type = 'symmetric'
        for igpu in range(ngpus):
            with cp.cuda.Device(igpu):
                self.pinned_mem[igpu] = cp.cuda.alloc_pinned_memory(nbytes)
                self.gpu_mem[igpu] = cp.cuda.alloc(nbytes)
                for k in range(3):
                    self.stream[igpu][k] = cp.cuda.Stream(non_blocking=False)
                fx = cp.fft.fftfreq(nq * 2, d=voxelsize).astype("float32")
                if self.pad_type=='nan':
                    fx = cp.fft.fftfreq(nq, d=voxelsize).astype("float32")
                [fx, fy] = cp.meshgrid(fx, fx)
                unimod = np.exp(1j * 2*np.pi* distance /wavelength)
                self.fker[igpu] = cp.exp(-1j * cp.pi * wavelength * distance * (fx**2 + fy**2))#*unimod
                self.pool_inp[igpu] = ThreadPoolExecutor(16 // ngpus)    
                self.pool_out[igpu] = ThreadPoolExecutor(16 // ngpus)
            
        self.npsi = npsi
        self.nq = nq
        self.npatch = npatch
        self.n = n
        self.npos = npos
        self.nchunk = nchunk
        self.ngpus = ngpus
        self.ex = ex
        self.pad = pad
        self.eps = eps
        self.rho = rho
        self.crop = crop
        self.path_out = path_out        
        self.niter = args.niter
        self.err_step = args.err_step
        self.vis_step = args.vis_step
        self.rho = args.rho
        self.method = args.method
        
        self.show = show

    def BH(self, d, vars):
        d = np.sqrt(d)
        
        alpha = 1
        rho = self.rho
        reused = {}

        for i in range(self.niter):
            # Calc reused variables and big_phi
            self.calc_reused(reused, vars)
            
            self.calc_phi(reused, d)
            
            # debug and visualization
            self.error_debug(vars, reused, d, i)
            
            self.vis_debug(vars, i)

            # gradients for each variable
            grads = self.gradients(vars, reused)
            
            if i == 0 or self.method == "BH-GD":
                etas = {}
                etas["psi"] = -grads["psi"] * rho[0] ** 2
                etas["q"] = -grads["q"] * rho[1] ** 2
                etas["r"] = -grads["r"] * rho[2] ** 2
            else:
                # conjugate direction
                beta = self.calc_beta(vars, grads, etas, reused, d)
                etas["psi"] = -grads["psi"] * rho[0] ** 2 + beta * etas["psi"]
                etas["q"] = -grads["q"] * rho[1] ** 2 + beta * etas["q"]
                etas["r"] = -grads["r"] * rho[2] ** 2 + beta * etas["r"]

            # step length
            alpha, top, bottom = self.calc_alpha(vars, grads, etas, reused, d)
            # print(alpha,top,bottom)
            # debug approxmation
            self.plot_debug(vars, etas, top, bottom, alpha, d, i)

            vars["psi"] += alpha * etas["psi"]
            vars["q"] += alpha * etas["q"]
            vars["r"] += alpha * etas["r"]

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

    def ET(self, patches, ri):
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
        def _ET(self, psi, patches, ri):
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

        _ET(self, psi, patches, ri)
        if flg:
            for k in range(1,len(psi)):
                psi[0] += psi[k]
            psi = psi[0]    
        return psi

    def S(self, ri, r, psi):
        # memory for result
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
            patches[:] = psir[
                :, self.ex : self.npatch - self.ex, self.ex : self.npatch - self.ex
            ]

        _S(self, patches, ri, r, psi)
        return patches

    def ST(self, patches, ri, r):
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
        def _ST(self, psi, patches , ri, r):
            """Adjont extract patches with subpixel shift"""
            psir = cp.pad(patches, ((0, 0), (self.ex, self.ex), (self.ex, self.ex)))

            x = cp.fft.fftfreq(self.npatch).astype("float32")
            [y, x] = cp.meshgrid(x, x)
            tmp = cp.exp(
                2 * cp.pi * 1j * (y * r[:, 1, None, None] + x * r[:, 0, None, None])
            ).astype("complex64")
            psir = cp.fft.ifft2(tmp * cp.fft.fft2(psir))

            psi[:] += self.ET(psir, ri)

        _ST(self, psi, patches, ri, r)
        if flg:
            for k in range(1,len(psi)):
                psi[0] += psi[k]
            psi = psi[0]    
        return psi


    def _fwd_pad(self,f):
        """Fwd data padding"""
        [ntheta, n] = f.shape[:2]
        fpad = cp.zeros([ntheta, 2*n, 2*n], dtype='complex64')
        pad_kernel((int(cp.ceil(2*n+2/32)), int(cp.ceil(2*n/32)), ntheta),
                (32, 32, 1), (fpad, f, n, ntheta, 0))
        return fpad/2
    
    def _fwd_pad_sym(self,f):
        """Fwd data padding"""
        fpad = cp.zeros([len(f), self.n+2*self.pad, self.n+2*self.pad], dtype='complex64')
        pad_sym_kernel((int(cp.ceil((self.n+2*self.pad)/32)), int(cp.ceil((self.n+2*self.pad)/32)), len(f)),
                (32, 32, 1), (fpad, f, self.pad, self.n, self.n, len(f), 0))
        return fpad/((self.n+2*self.pad)/self.n)


    def _adj_pad_sym(self, fpad):
        """Adj data padding"""
        f = cp.zeros([len(fpad), self.n, self.n], dtype='complex64')
        pad_sym_kernel((int(cp.ceil((self.n+2*self.pad)/32)), int(cp.ceil((self.n+2*self.pad)/32)), len(f)),
                (32, 32, 1), (fpad, f, self.pad, self.n, self.n, len(f), 1))
        return f/((self.n+2*self.pad)/self.n)
    
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
            if self.pad_type=='symmetric':
                ff = self._fwd_pad(ff)
                # v = cp.ones(2*self.nq,dtype='float32')
                # v[:self.nq//2] = cp.sin(cp.linspace(0,1,self.nq//2)*cp.pi/2)
                # v[-self.nq//2:] = cp.cos(cp.linspace(0,1,self.nq//2)*cp.pi/2)
                # v = cp.outer(v,v)
                ff *= 2        
            elif self.pad_type=='none':
                ff = cp.pad(ff,((0,0),(self.nq//2,self.nq//2),(self.nq//2,self.nq//2)))
            ff = cp.fft.ifft2(cp.fft.fft2(ff)*self.fker[cp.cuda.Device().id])
            if self.pad_type!='nan':
                ff = ff[:,self.nq//2:-self.nq//2,self.nq//2:-self.nq//2]
                            
            big_psi[:] = ff[:, self.pad : self.nq - self.pad, self.pad : self.nq - self.pad]            
            # big_psi[:] = self._adj_pad_sym(ff)#[:, self.pad : self.nq - self.pad, self.pad : self.nq - self.pad]
        
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
            # ff = self._fwd_pad_sym(big_psi)
            # ff = big_psi.copy
            if self.pad_type!='nan':
                ff = cp.pad(ff,((0,0),(self.nq//2,self.nq//2),(self.nq//2,self.nq//2)))
            ff = cp.fft.ifft2(cp.fft.fft2(ff)/self.fker[cp.cuda.Device().id])
            if self.pad_type=='symmetric':
                # v = cp.ones(2*self.nq,dtype='float32')
                # v[:self.nq//2] = cp.sin(cp.linspace(0,1,self.nq//2)*cp.pi/2)
                # v[-self.nq//2:] = cp.cos(cp.linspace(0,1,self.nq//2)*cp.pi/2)
                # v = cp.outer(v,v)
                ff *= 2        
                psi[:] = self._adj_pad(ff)
            elif self.pad_type=='none':
                psi[:] = ff[:,self.nq//2:-self.nq//2,self.nq//2:-self.nq//2]
            elif self.pad_type=='nan':
                psi[:] = ff[:]
                
        _DT(self, psi, big_psi)
        return psi 
    
    
    def calc_phi(self, reused, d):
        big_psi = reused["big_psi"]
        flg = chunking_flg(locals().values())        
        if flg:
            big_phi = np.empty([len(big_psi), self.nq, self.nq], dtype="complex64")
        else:
            big_phi = cp.empty([len(big_psi), self.nq, self.nq], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _calc_phi(self, big_phi, big_psi, d):
            td = d * (big_psi / (cp.abs(big_psi) + self.eps))
            big_phi[:] = self.DT(2 * (big_psi - td))
            
        _calc_phi(self, big_phi, big_psi, d)
        reused['big_phi'] = big_phi    

    def calc_phi_q(self, q, dref):
        tmp = self.D(q[cp.newaxis])        
        td = dref * (tmp / (cp.abs(tmp) + self.eps))
        return  self.DT(2 * (tmp - td))[0]
    
    def gradient_psi(self, ri, r, big_phi, psi, q):
        # memory for result
        flg = chunking_flg(locals().values())        
        if flg:
            gradpsi = []
            for igpu in range(self.ngpus):
                with cp.cuda.Device(igpu):
                    gradpsi.append(cp.zeros([self.npsi, self.npsi], dtype="complex64"))
        else:
            gradpsi = cp.zeros([self.npsi, self.npsi], dtype="complex64")
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)                
        def _gradient_psi(self, gradpsi, ri, r, big_phi, q):            
            gradpsi[:] += self.ST(cp.conj(q) * big_phi, ri, r)
        _gradient_psi(self, gradpsi, ri, r, big_phi, q)
        if flg:
            for k in range(1,len(gradpsi)):
                gradpsi[0] += gradpsi[k]
            gradpsi = gradpsi[0]    
        return gradpsi
    
    def gradient_q(self, spsi, big_phi, q):
        flg = chunking_flg(locals().values())        
        if flg:
            gradq = []
            for igpu in range(self.ngpus):
                with cp.cuda.Device(igpu):
                    gradq.append(cp.zeros([self.nq, self.nq], dtype="complex64"))
        else:
            gradq = cp.zeros([self.nq, self.nq], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)                
        def _gradient_q(self, gradq, spsi, big_phi):
            gradq[:] += cp.sum(cp.conj(spsi) * big_phi,axis=0)        
        _gradient_q(self, gradq, spsi, big_phi)

        if flg:
            for k in range(1,len(gradq)):
                gradq[0] += gradq[k]
            gradq = gradq[0]    

        # regularization and probe fitting terms,
        # fast on 1 GPU
        # gradq += 2*self.lam*self.GT(self.CT(self.C(self.G(q))))
        # gradq += self.calc_phi_q(q,dref)
            
        return gradq
   
    def gradient_r(self, ri, r, big_phi, psi, q):
        
        flg = chunking_flg(locals().values())        
        if flg:
            gradr = np.empty([len(ri), 2], dtype="float32")
        else:
            gradr = cp.empty([len(ri), 2], dtype="float32")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)                               
        def _gradient_r(self, gradr, ri, r, big_phi, psi, q):

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

            # inner product with big_phi
            
            gradr[:, 0] = redot(big_phi, q * dt1, axis=(1, 2))
            gradr[:, 1] = redot(big_phi, q * dt2, axis=(1, 2))
        
        _gradient_r(self, gradr, ri, r, big_phi, psi, q)
        return gradr
    
    def gradients(self, vars, reused):
        (q, psi, ri, r) = (vars["q"], vars["psi"], vars["ri"], vars["r"])
        (big_phi, spsi) = (reused["big_phi"], reused["spsi"])

        dpsi = self.gradient_psi(ri, r, big_phi, psi, q)
        dprb = self.gradient_q(spsi, big_phi, q)
        dr = self.gradient_r(ri, r, big_phi, psi, q)
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
                

    def hessian_Fq(self, q,dq1,dq2,dref):
        Dq = self.D(q[cp.newaxis])[0]
        Dq1 = self.D(dq1[cp.newaxis])[0]
        Dq2 = self.D(dq2[cp.newaxis])[0] 
        l0 = Dq/np.abs(Dq)
        d0 = dref/np.abs(Dq)
        v1 = np.sum((1-d0)*reprod(Dq1,Dq2))
        v2 = np.sum(d0*reprod(l0,Dq1)*reprod(l0,Dq2))        
        return 2*(v1+v2)
    
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

    def calc_beta(self, vars, grads, etas, reused, d):
        (q, psi, ri, r) = (vars["q"], vars["psi"], vars["ri"], vars["r"])
        (spsi, big_psi, big_phi) = (reused["spsi"], reused["big_psi"], reused["big_phi"])
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
        def _calc_beta(self, res, ri, r, spsi, big_psi, big_phi, dr1, dr2, d, q, dq1, dq2, psi,dpsi1, dpsi2):            
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
            tmp1 = self.E(ri, dpsi1)
            tmp1 = cp.fft.fft2(tmp1)
            sdpsi1 = cp.fft.ifft2(w * tmp1)[:, self.ex : self.nq + self.ex, self.ex : self.nq + self.ex]
            dt12 = -2 * cp.pi * 1j * cp.fft.ifft2(w * w2 * tmp1)[:, self.ex : self.nq + self.ex, self.ex : self.nq + self.ex]

            tmp2 = self.E(ri, dpsi2)
            tmp2 = cp.fft.fft2(tmp2)
            sdpsi2 = cp.fft.ifft2(w * tmp2)[:, self.ex : self.nq + self.ex, self.ex : self.nq + self.ex]
            dt21 = -2 * cp.pi * 1j * cp.fft.ifft2(w * w1 * tmp2)[:, self.ex : self.nq + self.ex, self.ex : self.nq + self.ex]
            dt22 = -2 * cp.pi * 1j * cp.fft.ifft2(w * w2 * tmp2)[:, self.ex : self.nq + self.ex, self.ex : self.nq + self.ex]

            tmp = self.E(ri, psi)
            tmp = cp.fft.fft2(tmp)
            dt1 = -2 * cp.pi * 1j * cp.fft.ifft2(w * w1 * tmp)[:, self.ex : self.nq + self.ex, self.ex : self.nq + self.ex]
            dt2 = -2 * cp.pi * 1j * cp.fft.ifft2(w * w2 * tmp)[:, self.ex : self.nq + self.ex, self.ex : self.nq + self.ex]
            d2t1 = -4 * cp.pi**2 * cp.fft.ifft2(w * w12 * tmp)[:, self.ex : self.nq + self.ex, self.ex : self.nq + self.ex]
            d2t2 = -4 * cp.pi**2 * cp.fft.ifft2(w * w22 * tmp)[:, self.ex : self.nq + self.ex, self.ex : self.nq + self.ex]

            # DM,D2M terms
            d2m1 = q * dt12 
            d2m1 += q * dt21 
            d2m1 += q * d2t1
            d2m1 += dq1 * sdpsi2 
            d2m1 += dq2 * sdpsi1
            d2m1 += dq1 * dt2 
            d2m1 += dq2 * dt1

            d2m2 = q * dt22 
            d2m2 += q * dt22 
            d2m2 += q * d2t2
            d2m2 += dq2 * sdpsi2 
            d2m2 += dq2 * sdpsi2
            d2m2 += dq2 * dt2 
            d2m2 += dq2 * dt2

            dm1 = dq1 * spsi
            dm1 += q * (sdpsi1 + dt1)
            dm2 = dq2 * spsi
            dm2 += q * (sdpsi2 + dt2)

            # top and bottom parts
            Ddm1 = self.D(dm1)
            Ddm2 = self.D(dm2)

            top = redot(big_phi, d2m1) + self.hessian_F(big_psi, Ddm1, Ddm2, d)
            bottom = redot(big_phi, d2m2) + self.hessian_F(big_psi, Ddm2, Ddm2, d)
            res[0] += top
            res[1] += bottom
            
        _calc_beta(self, res, ri, r, spsi, big_psi, big_phi, dr1, dr2, d, q, dq1, dq2, psi,dpsi1, dpsi2)           
        if flg:
            for k in range(1,len(res)):
                res[0][0] += res[k][0]
                res[0][1] += res[k][1]
            res = res[0]      
        
        
        top = cp.float32((res[0]).get()) 
        bottom = cp.float32((res[1]).get()) 
        return top/bottom
    

    def calc_alpha(self, vars, grads, etas, reused, d):
        (q, psi, ri, r) = (vars["q"], vars["psi"], vars["ri"], vars["r"])
        (dpsi1, dq1, dr1) = (grads["psi"], grads["q"], grads["r"])
        (dpsi2, dq2, dr2) = (etas["psi"], etas["q"], etas["r"])
        (spsi, big_psi, big_phi) = (reused["spsi"], reused["big_psi"], reused["big_phi"])
        
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
        def _calc_alpha(self, res, ri, r, spsi, big_psi, big_phi, dr1, dr2, d, q, dq2, psi, dpsi2):            
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
            d2m2 = q * (2 * dt2 + d2t) 
            d2m2 += 2 * dq2 * sdpsi 
            d2m2 += 2 * dq2 * dt
            dm = dq2 * spsi 
            dm += q * (sdpsi + dt)

            # bottom part
            Ddm = self.D(dm)
            bottom = redot(big_phi, d2m2) + self.hessian_F(big_psi, Ddm, Ddm, d)
            res[0] += top
            res[1] += bottom

        _calc_alpha(self, res, ri, r, spsi, big_psi, big_phi, dr1, dr2, d, q, dq2, psi, dpsi2)
        if flg:
            for k in range(1,len(res)):
                res[0][0] += res[k][0]
                res[0][1] += res[k][1]                
            res = res[0]
        res[0] += -redot(dpsi1, dpsi2) - redot(dq1, dq2)

        # regularization and probe fitting terms,
        # fast on 1 GPU
        
        
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

    def plot_debug(self, vars, etas, top, bottom, alpha, d, i):
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
                errt[k] = self.minF(tmp, d)

            t = alpha * (cp.arange(2 * npp)) / (npp - 1)
            errt2 = self.minF(self.fwd(ri,r,psi,q), d)
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
            diff = vars['r']-vars['r_init']
            plt.plot(diff[:,0],'.',label='y')
            plt.plot(diff[:,1],'.',label='x')
            plt.grid()
            plt.legend()
            plt.savefig(f'{self.path_out}/{i:04}')
            plt.close()
            # plt.show()
            write_tiff(cp.angle(psi),f'{self.path_out}/rec_psi_angle/{i:04}')
            write_tiff(cp.abs(psi),f'{self.path_out}/rec_psi_abs/{i:04}')
            write_tiff(cp.angle(q),f'{self.path_out}/rec_prb_angle/{i:04}')
            write_tiff(cp.abs(q),f'{self.path_out}/rec_prb_abs/{i:04}')
            np.save(f'{self.path_out}/{i:04}',vars['r'])


    def error_debug(self,vars, reused, d, i):
        """Visualization and data saving"""
        if i % self.err_step == 0 and self.err_step != -1:
            err = self.minF(reused["big_psi"], d)
            print(f"{i}) {err=:1.5e}", flush=True)
            vars["table"].loc[len(vars["table"])] = [i, err.get(), time.time()]
            name = f'{self.path_out}/conv.csv'
            os.makedirs(os.path.dirname(name), exist_ok=True)
            vars["table"].to_csv(name, index=False)
    