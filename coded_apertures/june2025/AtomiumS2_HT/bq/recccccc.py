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

import cupyx.scipy.ndimage as ndimage

class Rec:
    def __init__(self, args):

        ngpus = args.ngpus
        ntheta = args.ntheta
        npsi = args.npsi
        nq = args.nq
        n = args.n
        pad = args.pad
        ndist = args.ndist
        nchunk = args.nchunk

        voxelsize = args.voxelsize
        wavelength = args.wavelength
        distance = args.distance
        
        # allocate gpu and pinned memory buffers
        # calculate max buffer size
        nbytes = 32 * nchunk * npsi * npsi * np.dtype("complex64").itemsize            

        # create CUDA streams and allocate pinned memory
        self.stream = [[[] for _ in range(3)] for _ in range(ngpus)]
        self.pinned_mem = [[] for _ in range(ngpus)]
        self.gpu_mem = [[] for _ in range(ngpus)]
        self.fker = [[[] for _ in range(ndist)] for _ in range(ngpus)]
        self.pool_inp = [[] for _ in range(ngpus)]
        self.pool_out = [[] for _ in range(ngpus)]
        self.pool = ThreadPoolExecutor(ngpus)    
        
        for igpu in range(ngpus):
            with cp.cuda.Device(igpu):
                self.pinned_mem[igpu] = cp.cuda.alloc_pinned_memory(nbytes)
                self.gpu_mem[igpu] = cp.cuda.alloc(nbytes)
                for k in range(3):
                    self.stream[igpu][k] = cp.cuda.Stream(non_blocking=False)
                fx = cp.fft.fftfreq(nq * 2, d=voxelsize).astype("float32")
                [fx, fy] = cp.meshgrid(fx, fx)
                for j in range(ndist):
                    self.fker[igpu][j] = cp.exp(-1j * cp.pi * wavelength * distance[j] * (fx**2 + fy**2))
                self.pool_inp[igpu] = ThreadPoolExecutor(16 // ngpus)    
                self.pool_out[igpu] = ThreadPoolExecutor(16 // ngpus)
        
        self.pool_cpu = ThreadPoolExecutor(16)

        self.npsi = npsi
        self.ntheta = ntheta
        self.nq = nq
        self.n = n
        self.ndist= ndist
        self.nchunk = nchunk
        self.ngpus = ngpus
        self.pad = pad

        self.eps = args.eps
        self.rho = args.rho
        self.theta = args.theta        
        self.rotation_axis = args.rotation_axis        
        self.norm_magnifications = args.norm_magnifications
        
        self.voxelsize = voxelsize
        self.wavelength = wavelength
        
        self.path_out = args.path_out                
        self.method = args.method
        self.show = args.show
        
        self.niter = args.niter
        self.err_step = args.err_step
        self.vis_step = args.vis_step
        self.lam = args.lam
        
    
    
    def rec_tomo(self,d,niter=1):
        def minf(Ru,d):
            return np.linalg.norm(Ru-d)**2

        u = np.zeros([d.shape[1],self.npsi,self.npsi],dtype='complex64')
        Ru = self.R(u)
        tmp = np.empty_like(d)
        for k in range(niter):
            linear(tmp,Ru,d,2,-2,self.pool_cpu)
            grad = self.RT(tmp)            
            Rgrad = self.R(grad)            
            if k==0:
                eta = mulc(grad,-1,self.pool_cpu)
                Reta = mulc(Rgrad,-1,self.pool_cpu)
            else:
                beta = self.redot_batch(Rgrad,Reta)/self.redot_batch(Reta,Reta)
                beta = beta.get()
                linear(eta,grad,eta,-1,beta,self.pool_cpu)
                linear(Reta,Rgrad,Reta,-1,beta,self.pool_cpu)
            alpha = -self.redot_batch(grad,eta)/(2*self.redot_batch(Reta,Reta))
            alpha = alpha.get()
            linear(u,u,eta,1,alpha,self.pool_cpu)
            linear(Ru,Ru,Reta,1,alpha,self.pool_cpu)            
        return u
    
    def linear_batch(self,x,y,a,b):
        flg = chunking_flg(locals().values())        
        if flg:
            res = np.empty_like(x)
        else:
            res = cp.empty_like(x)
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _linear(self, res, x,y,a,b):
            res[:] = a*x[:]+b*y[:]
        _linear(self, res, x,y,a,b)
        return res
    
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
            
            if i == 0 or self.method == "BH-GD":
                etas = {}
                etas["u"] = self.mulc_batch(grads['u'],-rho[0]**2)
                etas["Ru"] = self.mulc_batch(grads['Ru'],-rho[0]**2)
                etas["q"] = self.mulc_batch(grads['q'],-rho[1]**2)
                etas["r"] = self.mulc_batch(grads['r'],-rho[2]**2)
            else:
                # conjugate direction
                beta = self.calc_beta(vars, grads, etas, reused, d)
                etas['u'][:] = self.linear_batch(grads['u'],etas['u'],-rho[0]**2,beta)
                etas['Ru'][:] = self.linear_batch(grads['Ru'],etas['Ru'],-rho[0]**2,beta) 
                etas['q'][:] = self.linear_batch(grads['q'],etas['q'],-rho[1]**2,beta)                                
                etas['r'][:] = self.linear_batch(grads['r'],etas['r'],-rho[2]**2,beta)  
            alpha, top, bottom = self.calc_alpha(vars, grads, etas, reused, d)
            # debug approxmation
            self.plot_debug(vars, etas, top, bottom, alpha, d, i)
            vars["u"][:] = self.linear_batch(vars["u"],etas['u'],1,alpha)
            vars["q"][:] = self.linear_batch(vars["q"],etas['q'],1,alpha)
            vars["r"][:] = self.linear_batch(vars["r"],etas['r'],1,alpha)
            vars["Ru"][:] = self.linear_batch(vars["Ru"],etas['Ru'],1,alpha)
            vars["psi"][:] = self.expR(vars['Ru'])

        return vars
    

    def initR(self, n):
        # usfft parameters
        eps = 1e-3  # accuracy of usfft
        mu = -cp.log(eps) / (2 * n * n)
        m = int(cp.ceil(2 * n * 1 / cp.pi * cp.sqrt(-mu *
                cp.log(eps) + (mu * n) * (mu * n) / 4)))
        # extra arrays
        # interpolation kernel
        t = cp.linspace(-1/2, 1/2, n, endpoint=False).astype('float32')
        [dx, dy] = cp.meshgrid(t, t)
        phi = cp.exp((mu * (n * n) * (dx * dx + dy * dy)).astype('float32')) * (1-n % 4)

       
        # (+1,-1) arrays for fftshift
        c1dfftshift = (1-2*((cp.arange(1, n+1) % 2))).astype('int8')
        c2dtmp = 1-2*((cp.arange(1, 2*n+1) % 2)).astype('int8')
        c2dfftshift = cp.outer(c2dtmp, c2dtmp)
        return m, mu, phi, c1dfftshift, c2dfftshift
        
    def R(self, obj):
        
        flg = chunking_flg(locals().values())
        # memory for result
        if flg:
            data = np.empty([self.ntheta, obj.shape[0],self.npsi], dtype="complex64")
        else:
            data = cp.empty([self.ntheta, obj.shape[0],self.npsi], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=1, axis_inp=0)
        def _R(self, data, obj):
            
            [nz, n, n] = obj.shape
            theta = cp.array(self.theta, dtype='float32')
            m, mu, phi, c1dfftshift, c2dfftshift = self.initR(n)
            sino = cp.zeros([self.ntheta,nz,  n], dtype='complex64')
        
            # STEP0: multiplication by phi, padding
            fde = obj*phi
            fde = cp.pad(fde, ((0, 0), (n//2, n//2), (n//2, n//2)))
            # STEP1: fft 2d
            fde = cp.fft.fft2(fde*c2dfftshift)*c2dfftshift
            fde = cp.pad(fde, ((0, 0), (m, m), (m, m)))
            # STEP2: fft 2d
            wrap_kernel((int(cp.ceil((2 * n + 2 * m)/32)),
                        int(cp.ceil((2 * n + 2 * m)/32)), nz), (32, 32, 1), (fde, n, nz, m))
            
            mua = cp.array([mu], dtype='float32')
            
            gather_kernel((int(cp.ceil(n/32)), int(cp.ceil(self.ntheta/32)), nz),
                        (32, 32, 1), (sino, fde, theta, m, mua, n, self.ntheta, nz, 0))
            # STEP3: ifft 1d
            sino = cp.fft.ifft(c1dfftshift*sino)*c1dfftshift
            
            # STEP4: Shift based on the rotation axis
            t = cp.fft.fftfreq(n).astype('float32')
            w = cp.exp(-2*cp.pi*1j*t*(self.rotation_axis + n/2))
            sino = cp.fft.ifft(w*cp.fft.fft(sino))
            # normalization for the unity test
            sino /= cp.float32(4*n)#*np.sqrt(n*self.ntheta))                
            
            data[:] = sino
        
        _R(self, data, obj)
        return data
    
    def RT(self, data):
        
        flg = chunking_flg(locals().values())
        # memory for result
        if flg:
            obj = np.empty([data.shape[1], self.npsi,self.npsi], dtype="complex64")
        else:
            obj = cp.empty([data.shape[1], self.npsi,self.npsi], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=1)
        def _RT(self, obj, data):
            sino = data#.swapaxes(0,1)

            [ntheta,nz , n] = sino.shape
            theta = cp.array(self.theta, dtype='float32')

            m, mu, phi, c1dfftshift, c2dfftshift = self.initR(n)

            # STEP0: Shift based on the rotation axis
            t = cp.fft.fftfreq(n).astype('float32')
            w = cp.exp(-2*cp.pi*1j*t*(-self.rotation_axis + n/2))
            sino = cp.fft.ifft(w*cp.fft.fft(sino))

            # STEP1: fft 1d
            sino = cp.fft.fft(c1dfftshift*sino)*c1dfftshift

            # STEP2: interpolation (gathering) in the frequency domain
            # dont understand why RawKernel cant work with float, I have to send it as an array (TODO)
            mua = cp.array([mu], dtype='float32')
            fde = cp.zeros([nz, 2*m+2*n, 2*m+2*n], dtype='complex64')
            gather_kernel((int(cp.ceil(n/32)), int(cp.ceil(ntheta/32)), nz),
                        (32, 32, 1), (sino, fde, theta, m, mua, n, ntheta, nz, 1))
            wrapadj_kernel((int(cp.ceil((2 * n + 2 * m)/32)),
                            int(cp.ceil((2 * n + 2 * m)/32)), nz), (32, 32, 1), (fde, n, nz, m))
            # STEP3: ifft 2d
            fde = fde[:, m:-m, m:-m]
            fde = cp.fft.ifft2(fde*c2dfftshift)*c2dfftshift

            # STEP4: unpadding, multiplication by phi
            fde = fde[:, n//2:3*n//2, n//2:3*n//2]*phi
            fde /= cp.float32(n)#*np.sqrt(n*ntheta))  # normalization for the unity test
            
            obj[:] = fde
            
        _RT(self, obj,data)

        return obj

    def expR(self, psi):
        
        flg = chunking_flg(locals().values())
        # memory for result
        if flg:
            epsi = np.empty([self.ntheta, self.npsi,self.npsi], dtype="complex64")
        else:
            epsi = cp.empty([self.ntheta, self.npsi,self.npsi], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _expR(self, epsi, psi):
            epsi[:] = np.exp(1j*psi)
        _expR(self,epsi,psi)
        return epsi

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
        # return psi.copy()
        # memory for result
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
        # return psi.copy()    
        # memory for result
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
            # mshow_complex(fde[-1],True)
            # fde[:]=1
            wrapadj_kernel((int(cp.ceil((2 * self.npsi + 2 * m)/32)),
                            int(cp.ceil((2 * self.npsi + 2 * m)/32)), len(f)), (32, 32, 1), (fde, self.npsi, len(f), m))
            
            fde = fde[:, m:-m, m:-m]
            fde = cp.fft.ifft2(fde*c2dfftshift)*c2dfftshift
            
            fde = fde[:, self.npsi//2:3*self.npsi//2, self.npsi//2:3*self.npsi//2]*phi
            res[:] = cp.fft.ifft2(fde*c2dfftshift0)*c2dfftshift0
            
        _MT(self,res, psi, self.norm_magnifications[j]*self.npsi/(self.nq)) 
        return res

    def G(self,psi):
        if self.lam==0:
            return np.zeros_like(psi)
            
        flg = chunking_flg(locals().values())        
        if flg:
            res = np.empty([self.npsi, self.npsi, self.npsi], dtype="complex64")
        else:
            res = cp.empty([self.npsi, self.npsi, self.npsi], dtype="complex64")
        
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
            gradF = np.empty([len(big_psi),self.ndist, self.nq, self.nq], dtype="complex64")
        else:
            gradF = cp.empty([len(big_psi),self.ndist, self.nq, self.nq], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _gradientF(self, gradF, big_psi, d):
            for j in range(self.ndist):
                td = d[:,j] * (big_psi[:,j] / (cp.abs(big_psi[:,j]) + self.eps))
                gradF[:,j] = self.DT(2 * (big_psi[:,j] - td), j)
            
        _gradientF(self, gradF, big_psi, d)
        reused['gradF'] = gradF                
    
    def gradient_u(self, gradF, psi,  r, q, u):

        # memory for result
        flg = chunking_flg(locals().values())        
        if flg:
            gradpsi = np.empty([self.ntheta,self.npsi, self.npsi], dtype="complex64")
        else:
            gradpsi = cp.empty([self.ntheta,self.npsi, self.npsi], dtype="complex64")

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)                
        def _gradient_psi(self, gradpsi, gradF, psi,  r, q):            
            for j in range(self.ndist):
                tmp = self.MT(cp.conj(q[j]) * gradF[:,j],j)
                if j==0:
                    gradpsi[:] = self.ST(r[:,j],tmp) * cp.conj(psi)#*(-1j)
                else:
                    gradpsi[:] += self.ST(r[:,j],tmp) * cp.conj(psi)#*(-1j)
            
        _gradient_psi(self, gradpsi, gradF, psi,  r, q)
        grad_u = self.RT(gradpsi)
        
        # gg = self.G(self.G(u))
        # linear(grad_u,grad_u,gg,-1j,2*self.lam,self.pool_cpu)
        return grad_u
    
    def gradient_q(self, mspsi, gradF):
        flg = chunking_flg(locals().values())        
        if flg:
            gradq = []
            for igpu in range(self.ngpus):
                with cp.cuda.Device(igpu):
                    gradq.append(cp.zeros([self.ndist, self.nq, self.nq], dtype="complex64"))
        else:
            gradq = cp.zeros([self.ndist, self.nq, self.nq], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)                
        def _gradient_q(self, gradq, mspsi, gradF):
            for j in range(self.ndist):# dirct writing in 
                gradq[j] += cp.sum(cp.conj(mspsi[:,j])*gradF[:,j],axis=0)        
        _gradient_q(self, gradq, mspsi, gradF)

        if flg:
            for k in range(1,len(gradq)):
                gradq[0] += gradq[k]
            gradq = gradq[0]    
            
        return gradq
   
    def gradient_r(self,  r, psi, gradF, q):
        
        flg = chunking_flg(locals().values())        
        if flg:
            gradr = np.empty([len(r), self.ndist, 2], dtype="float32")
        else:
            gradr = cp.empty([len(r), self.ndist, 2], dtype="float32")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)                               
        def _gradient_r(self, gradr,  r, psi, gradF, q):

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

                gradr[:, j, 0] = redot(gradF[:,j], dm1, axis=(1, 2))
                gradr[:, j, 1] = redot(gradF[:,j], dm2, axis=(1, 2))
        
        _gradient_r(self, gradr,  r, psi, gradF, q)
        return gradr
    
    def gradients(self, vars, reused):
        (q, u, psi,  r) = (vars["q"], vars["u"], vars["psi"], vars["r"])
        (gradF, mspsi) = (reused["gradF"], reused["mspsi"])
        du = self.gradient_u(gradF, psi,  r, q, u)
        dq = self.gradient_q(mspsi,gradF)
        dr = self.gradient_r(r, psi, gradF, q)
        Rdu = self.R(du)
        grads = {"u": du, "Ru": Rdu, "q": dq, "r": dr}
        
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
        (q, psi,  r) = (vars["q"], vars["psi"], vars["r"])
        flg = chunking_flg(vars.values())   
        if flg:
            mspsi = np.zeros([*r.shape[:2], self.nq, self.nq], dtype="complex64")
            big_psi = np.zeros([*r.shape[:2], self.n, self.n], dtype="complex64")
        else:
            mspsi = cp.zeros([*r.shape[:2], self.nq, self.nq], dtype="complex64")
            big_psi = cp.zeros([*r.shape[:2], self.n, self.n], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _calc_reused1(self, mspsi, psi, r):
            for j in range(self.ndist):
                mspsi[:,j] = self.M(self.S(r[:,j], psi),j)                
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _calc_reused2(self, big_psi, mspsi, q):
            for j in range(self.ndist):
                big_psi[:,j] = self.D(mspsi[:,j]*q[j],j) 

        _calc_reused1(self, mspsi, psi, r)
        _calc_reused2(self, big_psi, mspsi, q)

        reused["mspsi"] = mspsi
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
    
    def calc_beta(self, vars, grads, etas, reused, d):
        (q, psi,  r) = (vars["q"], vars["psi"], vars["r"])
        (mspsi, big_psi, gradF) = (reused["mspsi"], reused["big_psi"], reused["gradF"])
        (Rdu1, du1, dq1, dr1) = (grads["Ru"], grads["u"], grads["q"], grads["r"])
        (Rdu2, du2, dq2, dr2) = (etas["Ru"], etas["u"],  etas["q"], etas["r"])        

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
        def _calc_beta(self, res,  r, mspsi, big_psi, gradF, dr1, dr2, psi, Rdu1, Rdu2, d, q, dq1, dq2):            
            # note scaling with rho
            Rdu1 = Rdu1 * self.rho[0] ** 2
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

                tmp = psi*1j*Rdu1
                tmp = cp.fft.fft2(tmp)
                dt12 = -2 * cp.pi * 1j * cp.fft.ifft2(w * w2 * tmp)
                sdpsi1R = cp.fft.ifft2(w * tmp)
                
                tmp = psi*1j*Rdu2
                tmp = cp.fft.fft2(tmp)
                dt1R = -2 * cp.pi * 1j * cp.fft.ifft2(w * w1 * tmp)
                dt2R = -2 * cp.pi * 1j * cp.fft.ifft2(w * w2 * tmp)                                
                sdpsi2R = cp.fft.ifft2(w * tmp)
                
                sdpsi1RR = self.S(rj,-0.5*psi*Rdu1*Rdu2)
                sdpsi2RR = self.S(rj,-0.5*psi*Rdu2*Rdu2)

                dv1 = q[j] * self.M(sdpsi1R+dt1,j)+dq1[j]*mspsi[:,j]
                dv2 = q[j] * self.M(sdpsi2R+dt2,j)+dq2[j]*mspsi[:,j]
                
                d2v1 = q[j]*self.M(2*sdpsi1RR+dt12+dt1R+d2t1,j)
                d2v1 += dq1[j]*self.M(sdpsi2R+dt2,j)
                d2v1 += dq2[j]*self.M(sdpsi1R+dt1,j)

                d2v2 = q[j]*self.M(2*sdpsi2RR+2*dt2R+d2t2,j)
                d2v2 += 2*dq2[j]*self.M(sdpsi2R+dt2,j)
                                
                # top and bottom parts
                Ddv1 = self.D(dv1,j)
                Ddv2 = self.D(dv2,j)
                top = redot(gradF[:,j], d2v1) + self.hessian_F(big_psi[:,j], Ddv1, Ddv2, d[:,j])
                bottom = redot(gradF[:,j], d2v2) + self.hessian_F(big_psi[:,j], Ddv2, Ddv2, d[:,j])
                res[0] += top
                res[1] += bottom
            
        _calc_beta(self, res,  r, mspsi, big_psi, gradF, dr1, dr2, psi,Rdu1, Rdu2, d, q, dq1, dq2)  
        if flg:
            for k in range(1,len(res)):
                res[0][0] += res[k][0]
                res[0][1] += res[k][1]
            res = res[0]      

        # gu1 = self.G(du1)
        # gu2 = self.G(du2)
        
        # res[0] += 2*self.lam*self.redot_batch(gu1,gu2)
        # res[1] += 2*self.lam*self.redot_batch(gu2,gu2)

        top = cp.float32((res[0]).get()) 
        bottom = cp.float32((res[1]).get()) 
        return top/bottom
    

    def calc_alpha(self, vars, grads, etas, reused, d):
        (q, psi,  r) = (vars["q"], vars["psi"], vars["r"])
        (mspsi, big_psi, gradF) = (reused["mspsi"], reused["big_psi"], reused["gradF"])
        (du1,Rdu1, dq1, dr1) = (grads["u"], grads["Ru"], grads["q"], grads["r"])
        (du2,Rdu2, dq2, dr2) = (etas["u"], etas["Ru"], etas["q"], etas["r"])   
        
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
        def _calc_alpha(self, res,  r, mspsi, big_psi, gradF, dr1, dr2, psi, Rdu2, d, q, dq2):            
            
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

                tmp = psi*1j*Rdu2
                tmp = cp.fft.fft2(tmp)
                dt2R = -2 * cp.pi * 1j * cp.fft.ifft2(w * w2 * tmp)                                
                sdpsi2R = cp.fft.ifft2(w * tmp)                                
                sdpsi2RR = self.S(rj,-0.5*psi*Rdu2*Rdu2)

                dv2 = q[j] * self.M(sdpsi2R+dt2,j)+dq2[j]*mspsi[:,j]
                                
                d2v2 = q[j]*self.M(2*sdpsi2RR+2*dt2R+d2t2,j)
                d2v2 += 2*dq2[j]*self.M(sdpsi2R+dt2,j)
                                           
                # top and bottom parts
                Ddv2 = self.D(dv2,j)
                bottom = redot(gradF[:,j], d2v2) + self.hessian_F(big_psi[:,j], Ddv2, Ddv2, d[:,j])
                res[0] += top
                res[1] += bottom            
            

        _calc_alpha(self, res,  r, mspsi, big_psi, gradF, dr1, dr2, psi, Rdu2, d, q, dq2)
        if flg:
            for k in range(1,len(res)):
                res[0][0] += res[k][0]
                res[0][1] += res[k][1]                
            res = res[0]
        res[0] += -self.redot_batch(du1, du2) - self.redot_batch(dq1, dq2)
        
        # gu2 = self.G(du2)        
        # res[1] += 2*self.lam*self.redot_batch(gu2,gu2)

        top = cp.float32((res[0]).get()) 
        bottom = cp.float32((res[1]).get()) 
        return top/bottom,top,bottom
    
    def fwd(self,r,u,q):
                
        # memory for result
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
        psi = self.expR(self.R(u))             
        _fwd(self,res,r,psi,q)
        return res

    def plot_debug(self, vars, etas, top, bottom, alpha, d, i):
        """Check the minimization functional behaviour"""
        if i % self.vis_step == 0 and self.vis_step != -1 and self.show:
            (q, u,  r) = (vars["q"], vars["u"], vars["r"])
            (dq2, du2, dr2) = (etas["q"], etas["u"], etas["r"])
            # print(alpha)
            # print(np.linalg.norm(du2))    
            # print(np.linalg.norm(dq2))    
            # print(np.linalg.norm(dr2))
            # return    
            npp = 3
            errt = cp.zeros(npp * 2)
            errt2 = cp.zeros(npp * 2)
            for k in range(0, npp * 2):
                ut = u + (alpha * k / (npp - 1)) * du2
                qt = q + (alpha * k / (npp - 1)) * dq2
                rt = r + (alpha * k / (npp - 1)) * dr2
                tmp=self.fwd(rt,ut,qt)
                errt[k] = self.minF(tmp, d)# + self.lam*np.linalg.norm(self.G(ut))**2 
                
            t = alpha * (cp.arange(2 * npp)) / (npp - 1)
            tmp = self.fwd(r,u,q)
            errt2 = self.minF(tmp, d)# + self.lam*np.linalg.norm(self.G(u))**2 
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
            (q, u, r) = (vars["q"], vars["u"], vars["r"])
            
            mshow_complex(u[u.shape[0]//2].real+1j*u[:,u.shape[1]//2].real,self.show)
            mshow_polar(q[0],self.show)
            # mshow_polar(q[-1],self.show)
            write_tiff(u.real,f'{self.path_out}/rec_u_real/{i:04}')
            # write_tiff(u.imag,f'{self.path_out}/rec_u_imag/{i:04}')
            write_tiff(u[self.npsi//2].real,f'{self.path_out}/rec_uz/{i:04}')
            write_tiff(u[:,self.npsi//2].real,f'{self.path_out}/rec_uy/{i:04}')            
            write_tiff(np.angle(q[0]),f'{self.path_out}/rec_prb_angle/{i:04}')
            write_tiff(np.abs(q[0]),f'{self.path_out}/rec_prb_abs/{i:04}')
            write_tiff(np.angle(q[-1]),f'{self.path_out}/rec_prb_angle1/{i:04}')
            write_tiff(np.abs(q[-1]),f'{self.path_out}/rec_prb_abs1/{i:04}')
            np.save(f'{self.path_out}/r{i:04}',r)
            
            fig,ax = plt.subplots(1,2, figsize=(15, 6))
            ax[0].plot(vars['r'][:,:,1]-vars['r_init'][:,:,1],'.')                            
            ax[1].plot(vars['r'][:,:,0]-vars['r_init'][:,:,0],'.')                
            plt.savefig(f'{self.path_out}/rerr{i:04}.png')
            
            if self.show:
                plt.show()
            plt.close()

    def error_debug(self,vars, reused, d, i):
        """Visualization and data saving"""
        u = vars['u']
        if i % self.err_step == 0 and self.err_step != -1:
            err = self.minF(reused["big_psi"], d)# + self.lam*np.linalg.norm(self.G(u))**2
            print(f"{i}) {err=:1.5e}", flush=True)
            vars["table"].loc[len(vars["table"])] = [i, err.get(), time.time()]
            name = f'{self.path_out}/conv.csv'
            os.makedirs(os.path.dirname(name), exist_ok=True)
            vars["table"].to_csv(name, index=False)
    