import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time
from concurrent.futures import ThreadPoolExecutor
import os
import sys
sys.path.insert(0, '..')

from cuda_kernels import *
from utils import *
from chunking import gpu_batch
import psutil

import cupyx.scipy.ndimage as ndimage

class Rec:
    def __init__(self, args):

        for key, value in vars(args).items():
            setattr(self, key, value)  
        nbytes = 8 * args.nchunk * args.npsi**2* np.dtype("complex64").itemsize            
        
        # create CUDA streams and allocate pinned memory
        self.stream = [[[] for _ in range(3)] for _ in range(args.ngpus)]
        self.pinned_mem = [[] for _ in range(args.ngpus)]
        self.gpu_mem = [[] for _ in range(args.ngpus)]
        self.fker = [[[] for _ in range(args.ndist)] for _ in range(args.ngpus)]
        self.pool_inp = [[] for _ in range(args.ngpus)]
        self.pool_out = [[] for _ in range(args.ngpus)]
        self.pool = ThreadPoolExecutor(args.ngpus)    
        
        for igpu in range(args.ngpus):
            with cp.cuda.Device(igpu):
                self.pinned_mem[igpu] = cp.cuda.alloc_pinned_memory(nbytes)
                self.gpu_mem[igpu] = cp.cuda.alloc(nbytes)
                for k in range(3):
                    self.stream[igpu][k] = cp.cuda.Stream(non_blocking=False)
                fx = cp.fft.fftfreq(args.nq * 2, d=args.voxelsize).astype("float32")
                [fx, fy] = cp.meshgrid(fx, fx)
                for j in range(args.ndist):
                    self.fker[igpu][j] = cp.exp(-1j * cp.pi * args.wavelength * args.distance[j] * (fx**2 + fy**2))
                self.pool_inp[igpu] = ThreadPoolExecutor(16 // args.ngpus)    
                self.pool_out[igpu] = ThreadPoolExecutor(16 // args.ngpus)
        
        self.pool_cpu = ThreadPoolExecutor(16)
    
    def linear_batch(self,x,y,a,b):
        # flg = chunking_flg(locals().values())        
        # if flg:
        #     res = np.empty_like(x)
        # else:
        #     res = cp.empty_like(x)
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _linear(self, res, x, y,a,b):
            res[:] = a*x[:]+b*y[:]
        _linear(self, x, x, y,a,b)
        return x
    
    def mulc_batch(self,x,a):
        flg = chunking_flg(locals().values())        
        if flg:
            res = np.empty_like(x)
        else:
            res = cp.empty_like(x)
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _mulc(self, res, x,a):
            res[:] = a*x[:]
        _mulc(self, res, x,a)
        return res
    
    @timer
    def BH(self, d, vars):
        
        for i in range(self.niter):
            self.error_debug(vars, d, i)
            self.vis_debug(vars, i)
            grads = self.gradients(vars, d)
            
            if i == 0:
                etas = {}
                etas["psi"] = self.mulc_batch(grads['psi'],-self.rho[0]**2)
                etas["q"] = self.mulc_batch(grads['q'],-self.rho[1]**2)
                etas["r"] = self.mulc_batch(grads['r'],-self.rho[2]**2)
            else:
                beta = self.calc_beta(vars, grads, etas, d); #print(f'{beta=}')
                self.linear_batch(etas['psi'],grads['psi'],beta, -self.rho[0]**2)                
                self.linear_batch(etas['q'],grads['q'],beta, -self.rho[1]**2)                                
                self.linear_batch(etas['r'],grads['r'],beta, -self.rho[2]**2)
            
            alpha, top, bottom = self.calc_alpha(vars, grads, etas, d); #print(f'{alpha=}')
            grads = []
            # debug approxmation
            # self.plot_debug(vars, etas, top, bottom, alpha, d, i)
            self.linear_batch(vars["psi"],etas['psi'],1,alpha)
            self.linear_batch(vars["q"],etas['q'],1,alpha)
            self.linear_batch(vars["r"],etas['r'],1,alpha)
            
            # os.system('echo micro*tomo |sudo -S /clear_cache3')
        return vars    

    def STa(self,  r, psi, mode, **kwargs):
        # memory for result
        flg = chunking_flg(locals().values())        
        n = psi.shape[-1]
        if flg:
            patches = np.empty([len(r), n, n], dtype="complex64")
        else:
            patches = cp.empty([len(r), n, n], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _STa(self, patches,  r, psi):
            """Extract patches with subpixel shift"""            
            x = cp.fft.fftfreq(2*n).astype("float32")
            [y, x] = cp.meshgrid(x, x)
            tmp = cp.exp(2 * cp.pi * 1j * (y * r[:, 1, None, None] + x * r[:, 0, None, None])).astype("complex64")
            psi = np.pad(psi,((0,0),(n//2,n//2),(n//2,n//2)),mode, **kwargs)            
            tmp = cp.fft.ifft2(tmp * cp.fft.fft2(psi))                        
            patches[:]=  tmp[:,n//2:-n//2,n//2:-n//2]

        _STa(self, patches,  r, psi)
        return patches


    def S(self,  r, psi):
        # memory for result
        flg = chunking_flg(locals().values())        
        if flg:
            patches = np.empty([len(r), psi.shape[-1], psi.shape[-1]], dtype="complex64")
        else:
            patches = cp.empty([len(r), psi.shape[-1], psi.shape[-1]], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _S(self, patches,  r, psi):
            """Extract patches with subpixel shift"""
            x = cp.fft.fftfreq(psi.shape[-1]).astype("float32")
            [y, x] = cp.meshgrid(x, x)
            tmp = cp.exp(-2 * cp.pi * 1j * (y * r[:, 1, None, None] + x * r[:, 0, None, None])).astype("complex64")
            patches[:] = cp.fft.ifft2(tmp * cp.fft.fft2(psi))                        

        _S(self, patches,  r, psi)
        return patches

    def ST(self, r, psi):
        """Place patches, note only on 1 gpu for now"""
        return self.S(-r,psi)
        
    def _fwd_pad(self,f):
        """Fwd data padding"""
        [ntheta, n] = f.shape[:2]
        fpad = cp.zeros([ntheta, 2*n, 2*n], dtype='complex64')
        pad_kernel((int(cp.ceil(2*n/32)), int(cp.ceil(2*n/32)), ntheta),
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
    
    def Da(self, psi,distance):
        """Forward propagator"""
        n = psi.shape[-1]
        # memory for result
        flg = chunking_flg(locals().values())        
        if flg:
            big_psi = np.empty([len(psi), n, n], dtype="complex64")
        else:
            big_psi = cp.empty([len(psi), n, n], dtype="complex64")
                
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _Da(self, big_psi, psi):
            ff = self._fwd_pad(psi)
            ff *= 2        
            fx = cp.fft.fftfreq(n * 2, d=self.voxelsize).astype("float32")
            [fx, fy] = cp.meshgrid(fx, fx)                
            fker = cp.exp(-1j * cp.pi * self.wavelength * distance * (fx**2 + fy**2))
            ff = cp.fft.ifft2(cp.fft.fft2(ff)*fker)
            ff = ff[:,n//2:-n//2,n//2:-n//2]            
            big_psi[:] = ff
                    
        _Da(self, big_psi, psi)
        return big_psi    

    def D(self, psi,j):
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
            ff = cp.fft.ifft2(cp.fft.fft2(ff)*self.fker[cp.cuda.Device().id][j])
            ff = ff[:,self.nq//2:-self.nq//2,self.nq//2:-self.nq//2]            
            big_psi[:] = ff[:, self.pad : self.nq - self.pad, self.pad : self.nq - self.pad]
                    
        _D(self, big_psi, psi)
        return big_psi


    def DT(self, big_psi,j):
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
            ff = cp.fft.ifft2(cp.fft.fft2(ff)/self.fker[cp.cuda.Device().id][j])
            ff *= 2        
            psi[:] = self._adj_pad(ff)
        _DT(self, psi, big_psi)
        return psi
    
    def init_mag(self, ne):
        # usfft parameters
        eps = 1e-3  # accuracy of usfft
        mu = -cp.log(eps) / (2 * ne * ne)
        m = int(cp.ceil(2 * ne * 1 / cp.pi * cp.sqrt(-mu *
                cp.log(eps) + (mu * ne) * (mu * ne) / 4)))
        # extra arrays
        # interpolation kernel
        t = cp.linspace(-1/2, 1/2, ne, endpoint=False).astype('float32')
        [dx, dy] = cp.meshgrid(t, t)
        phi = cp.exp((mu * (ne * ne) * (dx * dx + dy * dy)
                    ).astype('float32')) * (1-ne % 4)

        # (+1,-1) arrays for fftshift
        c2dtmp = 1-2*((cp.arange(1, 2*ne+1) % 2)).astype('int8')
        c2dfftshift = cp.outer(c2dtmp, c2dtmp)
        c2dtmp = 1-2*((cp.arange(1, ne+1) % 2)).astype('int8')
        c2dfftshift0 = cp.outer(c2dtmp, c2dtmp)
        return m, mu, phi, c2dfftshift, c2dfftshift0


    def M(self, psi, j):
        flg = chunking_flg(locals().values())        
        if flg:
            res = np.zeros([len(psi), self.nq, self.nq], dtype="complex64")
        else:
            res = cp.zeros([len(psi), self.nq, self.nq], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _M(self,res,f,magnification):
            m, mu, phi, c2dfftshift, c2dfftshift0 = self.init_mag(self.npsi)
            # FFT2D        
            fde = cp.fft.fft2(f*c2dfftshift0)*c2dfftshift0
            # adjoint USFFT2D
            fde = fde*phi
            
            fde = cp.pad(fde, ((0, 0), (self.npsi//2, self.npsi//2), (self.npsi//2, self.npsi//2)))
            fde = cp.fft.fft2(fde*c2dfftshift)*c2dfftshift
            fde = cp.pad(fde, ((0, 0), (m, m), (m, m)))
            wrap_kernel((int(cp.ceil((2 * self.npsi + 2 * m)/32)),
                        int(cp.ceil((2 * self.npsi + 2 * m)/32)), len(f)), (32, 32, 1), (fde, self.npsi, len(f), m))
            mua = cp.array([mu], dtype='float32')
            magnificationa = cp.array([magnification], dtype='float32')
            gather_mag_kernel((int(cp.ceil(self.nq/32)), int(cp.ceil(self.nq/32)), len(f)),
                            (32, 32, 1), (res, fde, magnificationa, m, mua, self.nq, self.npsi, len(f), 0))

            res[:]/=cp.float32(4*self.npsi**3)        

        _M(self, res, psi, self.norm_magnifications[j]*self.npsi/(self.nq))  
        return res

    def MT(self, psi,j):
        flg = chunking_flg(locals().values())        
        if flg:
            res = np.zeros([len(psi), self.npsi, self.npsi], dtype="complex64")
        else:
            res = cp.zeros([len(psi), self.npsi, self.npsi], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _MT(self,res,f,magnification):
            m, mu, phi, c2dfftshift, c2dfftshift0 = self.init_mag(self.npsi)

            mua = cp.array([mu], dtype='float32')
            magnificationa = cp.array([magnification], dtype='float32')
            fde = cp.zeros([len(f), 2*m+2*self.npsi, 2*m+2*self.npsi], dtype='complex64')
            f = cp.ascontiguousarray(f)
            gather_mag_kernel((int(cp.ceil(self.nq/32)), int(cp.ceil(self.nq/32)), len(f)),
                            (32, 32, 1), (f, fde, magnificationa, m, mua, self.nq, self.npsi, len(f), 1))
            wrapadj_kernel((int(cp.ceil((2 * self.npsi + 2 * m)/32)),
                            int(cp.ceil((2 * self.npsi + 2 * m)/32)), len(f)), (32, 32, 1), (fde, self.npsi, len(f), m))
            
            fde = fde[:, m:-m, m:-m]
            fde = cp.fft.ifft2(fde*c2dfftshift)*c2dfftshift
            
            fde = fde[:, self.npsi//2:3*self.npsi//2, self.npsi//2:3*self.npsi//2]*phi
            res[:] = cp.fft.ifft2(fde*c2dfftshift0)*c2dfftshift0
            
        _MT(self,res, psi, self.norm_magnifications[j]*self.npsi/(self.nq)) 
        return res
    
    @timer
    def gradient_psi(self, r, psi, d, q):

        flg = chunking_flg(locals().values())        
        if flg:
            gradpsi = np.empty([self.ntheta,self.npsi, self.npsi], dtype="complex64")
        else:
            gradpsi = cp.empty([self.ntheta,self.npsi, self.npsi], dtype="complex64")

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)                
        def _gradient_psi(self, gradpsi, r,  psi, d, q):            
            for j in range(self.ndist):
                mspsi = self.M(self.S(r[:,j], psi),j)  
                big_psi = self.D(mspsi*q[j],j) 
                td = d[:,j] * (big_psi / (cp.abs(big_psi)))
                gradF = self.DT(2 * (big_psi - td), j)
                tmp = self.MT(cp.conj(q[j]) * gradF,j)
                if j==0:
                    gradpsi[:] = self.ST(r[:,j],tmp)# * cp.conj(psi)*(-1j)
                else:
                    gradpsi[:] += self.ST(r[:,j],tmp)# * cp.conj(psi)*(-1j)
            
        _gradient_psi(self, gradpsi,  r, psi, d, q)
        if self.lam>0:
            gg =self.G(psi)
            gg = self.G(gg)
            self.linear_batch(gradpsi,gg,1,2*self.lam)
        # grad_u+=2*self.lam*self.G(self.G(u))

        return gradpsi
    
    @timer
    def gradient_q(self, r, psi, d, q):
        flg = chunking_flg(locals().values())        
        if flg:
            gradq = []
            for igpu in range(self.ngpus):
                with cp.cuda.Device(igpu):
                    gradq.append(cp.zeros([self.ndist, self.nq, self.nq], dtype="complex64"))
        else:
            gradq = cp.zeros([self.ndist, self.nq, self.nq], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)                
        def _gradient_q(self, gradq, r, psi, d, q):
            for j in range(self.ndist):# dirct writing in 
                mspsi = self.M(self.S(r[:,j], psi),j)  
                big_psi = self.D(mspsi*q[j],j) 
                td = d[:,j] * (big_psi / (cp.abs(big_psi)))
                gradF = self.DT(2 * (big_psi - td), j)
                gradq[j] += cp.sum(cp.conj(mspsi)*gradF,axis=0)        
        _gradient_q(self, gradq, r, psi, d, q)

        if flg:
            for k in range(1,len(gradq)):
                gradq[0] += gradq[k]
            gradq = gradq[0]    
            
        return gradq
   
    @timer
    def gradient_r(self,  r, psi, d, q):
        
        flg = chunking_flg(locals().values())        
        if flg:
            gradr = np.empty([len(r), self.ndist, 2], dtype="float32")
        else:
            gradr = cp.empty([len(r), self.ndist, 2], dtype="float32")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)                               
        def _gradient_r(self, gradr,  r, psi, d, q):

            # frequencies
            xi1 = cp.fft.fftfreq(self.npsi).astype("float32")
            xi2, xi1 = cp.meshgrid(xi1, xi1)

            for j in range(self.ndist):
                # multipliers in frequencies
                rj = r[:, j]
                w = cp.exp(
                    -2 * cp.pi * 1j * (xi2 * rj[:, 1, None, None] + xi1 * rj[:, 0, None, None])
                )

                # Gradient parts
                tmp = cp.fft.fft2(psi)
                dt1 = cp.fft.ifft2(w * xi1 * tmp)
                dt2 = cp.fft.ifft2(w * xi2 * tmp)
                dt1 = -2 * cp.pi * 1j * dt1#
                dt2 = -2 * cp.pi * 1j * dt2#

                # inner product with gradF
                dm1 = q[j]*self.M(dt1,j)
                dm2 = q[j]*self.M(dt2,j)
                mspsi = self.M(self.S(r[:,j], psi),j)  
                big_psi = self.D(mspsi*q[j],j) 
                td = d[:,j] * (big_psi / (cp.abs(big_psi)))
                gradF = self.DT(2 * (big_psi - td), j)
                gradr[:, j, 0] = redot(gradF, dm1, axis=(1, 2))
                gradr[:, j, 1] = redot(gradF, dm2, axis=(1, 2))
        
        _gradient_r(self, gradr,  r, psi, d, q)
        return gradr
    
    @timer
    def gradients(self, vars, d):
        (q, psi,  r) = (vars["q"], vars["psi"], vars["r"])
        dpsi = self.gradient_psi(r, psi, d, q)
        dq = self.gradient_q(r, psi, d, q)
        dr = self.gradient_r(r, psi, d, q)
        grads = {"psi": dpsi, "q": dq, "r": dr}
        
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
    
    def minFL(self, gu):
        flg = chunking_flg(locals().values())        
        if flg:
            res = []
            for igpu in range(self.ngpus):
                with cp.cuda.Device(igpu):
                    res.append(cp.zeros(1, dtype="float32"))
        else:
            res = cp.zeros(1, dtype="float32")

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)          
        def _minF(self, res, gu):            
            res[:]+=cp.linalg.norm(gu) ** 2
        _minF(self, res, gu)
        if flg:
            for k in range(1,len(res)):
                res[0] += res[k]
            res = res[0]    
        res = res[0]    
        return res
    
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
            l0 = big_psi / (cp.abs(big_psi))
            d0 = d / (cp.abs(big_psi))
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
    
    @timer
    def calc_beta(self, vars, grads, etas, d):
        (q, psi,  r) = (vars["q"], vars["psi"], vars["r"])
        (dpsi1, dq1, dr1) = (grads["psi"], grads["q"], grads["r"])
        (dpsi2, dq2, dr2) = (etas["psi"],  etas["q"], etas["r"])        

        flg = (chunking_flg(vars.values()) 
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
        def _calc_beta(self, res,  r, dr1, dr2, psi, dpsi1, dpsi2, d, q, dq1, dq2):            
            # note scaling with rho
            dpsi1 = dpsi1 * self.rho[0] ** 2
            dq1 = dq1 * self.rho[1] ** 2
            dr1 = dr1 * self.rho[2] ** 2
            # frequencies
            xi1 = cp.fft.fftfreq(self.npsi).astype("float32")
            [xi2, xi1] = cp.meshgrid(xi1, xi1)
            
            # multipliers in frequencies
            for j in range(self.ndist):
                dr1j = dr1[:, j, :, cp.newaxis, cp.newaxis]
                dr2j = dr2[:, j, :, cp.newaxis, cp.newaxis]
                rj = r[:, j]
                w = cp.exp(
                    -2 * cp.pi * 1j * (xi2 * rj[:, 1, None, None] + xi1 * rj[:, 0, None, None])
                )
                w1 = xi1 * dr1j[:, 0] + xi2 * dr1j[:, 1]
                w2 = xi1 * dr2j[:, 0] + xi2 * dr2j[:, 1]
                w12 = (
                    xi1**2 * dr1j[:, 0] * dr2j[:, 0]
                    + xi1 * xi2 * (dr1j[:, 0] * dr2j[:, 1] + dr1j[:, 1] * dr2j[:, 0])
                    + xi2**2 * dr1j[:, 1] * dr2j[:, 1]
                )
                w22 = (
                    xi1**2 * dr2j[:, 0] ** 2
                    + 2 * xi1 * xi2 * (dr2j[:, 0] * dr2j[:, 1])
                    + xi2**2 * dr2j[:, 1] ** 2
                )

                # dt1,dt2,d2t1,d2t2
                tmp = cp.fft.fft2(psi)
                dt1 = -2 * cp.pi * 1j * cp.fft.ifft2(w * w1 * tmp)
                dt2 = -2 * cp.pi * 1j * cp.fft.ifft2(w * w2 * tmp)
                d2t1 = -4 * cp.pi**2 * cp.fft.ifft2(w * w12 * tmp)
                d2t2 = -4 * cp.pi**2 * cp.fft.ifft2(w * w22 * tmp)

                tmp = dpsi1
                tmp = cp.fft.fft2(tmp)
                dt12 = -2 * cp.pi * 1j * cp.fft.ifft2(w * w2 * tmp)
                sdpsi1 = cp.fft.ifft2(w * tmp)
                
                tmp = dpsi2
                tmp = cp.fft.fft2(tmp)
                dt21 = -2 * cp.pi * 1j * cp.fft.ifft2(w * w1 * tmp)
                dt22 = -2 * cp.pi * 1j * cp.fft.ifft2(w * w2 * tmp)                                
                sdpsi2 = cp.fft.ifft2(w * tmp)
                

                mspsi = self.M(self.S(r[:,j], psi),j)  
                dv1 = q[j] * self.M(sdpsi1+dt1,j)+dq1[j]*mspsi#[:,j]
                dv2 = q[j] * self.M(sdpsi2+dt2,j)+dq2[j]*mspsi#[:,j]
                
                d2v1 = q[j]*self.M(dt12+dt21+d2t1,j)
                d2v1 += dq1[j]*self.M(sdpsi2+dt2,j)
                d2v1 += dq2[j]*self.M(sdpsi1+dt1,j)

                d2v2 = q[j]*self.M(2*dt22+d2t2,j)
                d2v2 += 2*dq2[j]*self.M(sdpsi2+dt2,j)
                                
                # top and bottom parts
                Ddv1 = self.D(dv1,j)
                Ddv2 = self.D(dv2,j)
                
                big_psi = self.D(mspsi*q[j],j) 
                td = d[:,j] * (big_psi / (cp.abs(big_psi)))
                gradF = self.DT(2 * (big_psi - td), j)

                top = redot(gradF, d2v1) + self.hessian_F(big_psi, Ddv1, Ddv2, d[:,j])
                bottom = redot(gradF, d2v2) + self.hessian_F(big_psi, Ddv2, Ddv2, d[:,j])
                res[0] += top
                res[1] += bottom
            
        _calc_beta(self, res,  r, dr1, dr2, psi,dpsi1,dpsi2, d, q, dq1, dq2)  
        if flg:
            for k in range(1,len(res)):
                res[0][0] += res[k][0]
                res[0][1] += res[k][1]
            res = res[0]      
        
        #to fit into mem
        if self.lam>0:
            gpsi1 = self.G(dpsi1)
            gpsi2 = self.G(dpsi2)
            res[0] += 2*self.lam*self.redot_batch(gpsi1,gpsi2)
            res[1] += 2*self.lam*self.redot_batch(gpsi2,gpsi2)        

        top = cp.float32((res[0]).get()) 
        bottom = cp.float32((res[1]).get()) 
        return top/bottom
    
    @timer
    def calc_alpha(self, vars, grads, etas, d):
        (q, psi,  r) = (vars["q"], vars["psi"], vars["r"])
        (dpsi1, dq1, dr1) = (grads["psi"], grads["q"], grads["r"])
        (dpsi2, dq2, dr2) = (etas["psi"],  etas["q"], etas["r"])   
        
        flg = (chunking_flg(vars.values()) 
               or chunking_flg(grads.values()) 
               or chunking_flg(etas.values()))                 
        if flg:
            res = []
            for igpu in range(self.ngpus):
                with cp.cuda.Device(igpu):
                    res.append(cp.zeros([2], dtype="float32"))
        else:
            res = cp.zeros([2], dtype="float32")
        # print(res.shape)
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _calc_alpha(self, res,  r, dr1, dr2, psi, dpsi2, d, q, dq2):            
            # print(res.shape)
            xi1 = cp.fft.fftfreq(self.npsi).astype("float32")
            [xi2, xi1] = cp.meshgrid(xi1, xi1)
            
            # multipliers in frequencies
            for j in range(self.ndist):
                # top part
                top = -redot(dr1[:,j], dr2[:,j])

                dr2j = dr2[:, j, :, cp.newaxis, cp.newaxis]
                rj = r[:, j]
                w = cp.exp(
                    -2 * cp.pi * 1j * (xi2 * rj[:, 1, None, None] + xi1 * rj[:, 0, None, None])
                )
                w2 = xi1 * dr2j[:, 0] + xi2 * dr2j[:, 1]
                w22 = (
                    xi1**2 * dr2j[:, 0] ** 2
                    + 2 * xi1 * xi2 * (dr2j[:, 0] * dr2j[:, 1])
                    + xi2**2 * dr2j[:, 1] ** 2
                )

                # dt1,dt2,d2t1,d2t2
                tmp = cp.fft.fft2(psi)
                dt2 = -2 * cp.pi * 1j * cp.fft.ifft2(w * w2 * tmp)
                d2t2 = -4 * cp.pi**2 * cp.fft.ifft2(w * w22 * tmp)
                
                tmp = cp.fft.fft2(dpsi2)
                dt22 = -2 * cp.pi * 1j * cp.fft.ifft2(w * w2 * tmp)                                
                sdpsi2 = cp.fft.ifft2(w * tmp)
                

                mspsi = self.M(self.S(r[:,j], psi),j)  
                dv2 = q[j] * self.M(sdpsi2+dt2,j)+dq2[j]*mspsi#[:,j]
                                
                d2v2 = q[j]*self.M(2*dt22+d2t2,j)
                d2v2 += 2*dq2[j]*self.M(sdpsi2+dt2,j)
                                
                # top and bottom parts
                Ddv2 = self.D(dv2,j)
                
                big_psi = self.D(mspsi*q[j],j) 
                td = d[:,j] * (big_psi / (cp.abs(big_psi)))
                gradF = self.DT(2 * (big_psi - td), j)

                bottom = redot(gradF, d2v2) + self.hessian_F(big_psi, Ddv2, Ddv2, d[:,j])
                res[0] += top
                res[1] += bottom                        

        _calc_alpha(self, res,  r, dr1, dr2, psi, dpsi2, d, q, dq2)
        if flg:
            for k in range(1,len(res)):
                res[0][0] += res[k][0]
                res[0][1] += res[k][1]                
            res = res[0]
        res[0] += -self.redot_batch(dpsi1, dpsi2) - self.redot_batch(dq1, dq2)
        
        if self.lam>0:
            gu2 = dpsi1#reuse memory
            gu2 = self.G(dpsi2,out=gu2)        
            res[1] += 2*self.lam*self.redot_batch(gu2,gu2)
            
        top = cp.float32((res[0]).get()) 
        bottom = cp.float32((res[1]).get()) 
        return top/bottom,top,bottom
    
    def fwd(self,r,psi,q):

        flg = chunking_flg(locals().values())        
        if flg:
            res = np.empty([len(r), self.ndist, self.n, self.n], dtype="complex64")
        else:
            res = cp.empty([len(r), self.ndist, self.n, self.n], dtype="complex64")
                
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _fwd(self,res,r,psi,q):            
            for j in range(self.ndist):
                mspsi = self.M(self.S(r[:,j],psi),j)
                res[:,j] = self.D(mspsi * q[j],j)
        _fwd(self,res,r,psi,q)
        return res
    
    def G(self,psi,out=None):

        if out is None:    
            flg = chunking_flg(locals().values())        
            if flg:
                res = np.empty(psi.shape, dtype="complex64")
            else:
                res = cp.empty(psi.shape, dtype="complex64")
        else:
            res=out    
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _G0(self, res, psi):
            stencil = cp.array([1, -2, 1]).astype('float32')            
            res[:] = ndimage.convolve1d(psi, stencil, axis=1)
            res[:] += ndimage.convolve1d(psi, stencil, axis=2)
                
        _G0(self,res,psi)
        return res    
    
    @timer
    def plot_debug(self, vars, etas, top, bottom, alpha, d, i):
        """Check the minimization functional behaviour"""
        if i % self.vis_step == 0 and self.vis_step != -1 and self.show:
            (q, psi,  r) = (vars["q"], vars["psi"], vars["r"])
            (dq2, dpsi2, dr2) = (etas["q"], etas["psi"], etas["r"])
            npp = 3
            errt = cp.zeros(npp * 2)
            errt2 = cp.zeros(npp * 2)
            for k in range(0, npp * 2):
                psit = psi + (alpha * k / (npp - 1)) * dpsi2
                qt = q + (alpha * k / (npp - 1)) * dq2
                rt = r + (alpha * k / (npp - 1)) * dr2
                tmp=self.fwd(rt,psit,qt)
                errt[k] = self.minF(tmp, d)
                if self.lam>0:
                    errt[k] += self.lam*self.minFL(self.G(psit))
                    # errt[k] += self.lam*np.linalg.norm(self.G(ut))**2
                
            t = alpha * (cp.arange(2 * npp)) / (npp - 1)
            tmp = self.fwd(r,psi,q)
            errt2 = self.minF(tmp, d) 
            if self.lam>0:
                errt2 += self.lam*self.minFL(self.G(psi)) 
                # errt2 += self.lam*np.linalg.norm(self.G(u))**2
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
        

    @timer
    def vis_debug(self, vars, i):
        """Visualization and data saving"""
        if i % self.vis_step == 0 and self.vis_step != -1:
            (q, psi, r) = (vars["q"], vars["psi"], vars["r"])
            
            mshow_polar(psi[0],self.show)
            mshow_polar(psi[-1],self.show)
            mshow_polar(q[0],self.show)
            for k in range(self.ndist):
                write_tiff(np.angle(q[k]),f'{self.path_out}/rec_prb_angle{k}/{i:04}')
                write_tiff(np.abs(q[k]),f'{self.path_out}/rec_prb_abs{k}/{i:04}')

                write_tiff(np.angle(psi[:,k]),f'{self.path_out}/rec_psi_angle{k}/{i:04}')
                write_tiff(np.abs(psi[:,k]),f'{self.path_out}/rec_psi_abs{k}/{i:04}')
            np.save(f'{self.path_out}/r{i:04}',r)
            
            fig,ax = plt.subplots(1,2, figsize=(15, 6))
            ax[0].plot(vars['r'][:,:,1]-vars['r_init'][:,:,1],'.')                            
            ax[1].plot(vars['r'][:,:,0]-vars['r_init'][:,:,0],'.')                
            plt.savefig(f'{self.path_out}/rerr{i:04}.png')
            
            if self.show:
                plt.show()
            plt.close()

    @timer
    def error_debug(self,vars, d, i):
        """Visualization and data saving"""
        if i % self.err_step == 0 and self.err_step != -1:
            (q, psi,  r) = (vars["q"], vars["psi"], vars["r"])
            big_psi=self.fwd(r,psi,q)
            err = self.minF(big_psi, d)
            if self.lam>0:
                err += self.lam*self.minFL(self.G(psi))
            print(f"{i}) {err=:1.5e}", flush=True)
            vars["table"].loc[len(vars["table"])] = [i, err.get(), time.time()]
            name = f'{self.path_out}/conv.csv'
            os.makedirs(os.path.dirname(name), exist_ok=True)
            vars["table"].to_csv(name, index=False)
    