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
        npos = args.npos
        nchunk = args.nchunk
        ex = args.ex
        voxelsize = args.voxelsize
        wavelength = args.wavelength
        distance = args.distance
        distancec = args.distancec
        eps = args.eps
        rho = args.rho
        lam = args.lam
        crop = args.crop
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
                fx = cp.fft.fftfreq(nq * 2, d=voxelsize).astype("float32")
                [fx, fy] = cp.meshgrid(fx, fx)
                self.fker[igpu] = cp.exp(-1j * cp.pi * wavelength * distance * (fx**2 + fy**2))
                self.fkerc[igpu] = cp.exp(-1j * cp.pi * wavelength * distancec * (fx**2 + fy**2))
                self.pool_inp[igpu] = ThreadPoolExecutor(16 // ngpus)    
                self.pool_out[igpu] = ThreadPoolExecutor(16 // ngpus)
        
        self.pool_cpu = ThreadPoolExecutor(16)
        self.npsi = npsi
        self.ntheta = ntheta
        self.nq = nq
        self.npatch = npatch
        self.ncode = ncode
        self.n = n
        self.npos = npos
        self.nchunk = nchunk
        self.ngpus = ngpus
        self.ex = ex
        self.pad = pad
        self.eps = eps
        self.rho = rho
        self.lam = lam
        self.crop = crop
        self.path_out = path_out        
        self.niter = args.niter
        self.err_step = args.err_step
        self.vis_step = args.vis_step
        self.rho = args.rho
        self.method = args.method
        self.show = show
        self.theta = theta
        self.voxelsize = voxelsize
        self.wavelength = wavelength
        self.rotation_axis = rotation_axis
    
    def rec_tomo(self,d,rpsi,niter=1):
        def minf(Ru,d):
            return np.linalg.norm(Ru-d)**2

        u = np.zeros([self.npsi,self.npsi,self.npsi],dtype='complex64')
        Ru = self.Spsi(self.R(u),rpsi)
        tmp = np.empty_like(d)
        for k in range(niter):
            linear(tmp,Ru,d,2,-2,self.pool_cpu)
            # print(np.linalg.norm(tmp))
            grad = self.RT(self.STpsi(tmp,rpsi))
            Rgrad = self.Spsi(self.R(grad),rpsi)
                
            if k==0:
                #eta = -grad
                eta = mulc(grad,-1,self.pool_cpu)
                #Reta = -Rgrad
                Reta = mulc(Rgrad,-1,self.pool_cpu)

            else:
                beta = self.redot_batch(Rgrad,Reta)/self.redot_batch(Reta,Reta)
                beta = beta.get()
                linear(eta,grad,eta,-1,beta,self.pool_cpu)
                #eta = -grad + beta*eta
                linear(Reta,Rgrad,Reta,-1,beta,self.pool_cpu)
                #Reta = -Rgrad + beta*Reta

            alpha = -self.redot_batch(grad,eta)/(2*self.redot_batch(Reta,Reta))
            alpha = alpha.get()
            #u += alpha*eta
            linear(u,u,eta,1,alpha,self.pool_cpu)
            #Ru += alpha*Reta
            linear(Ru,Ru,Reta,1,alpha,self.pool_cpu)
            #print(minf(Ru,d))
        return u
    
    def BH(self, d, vars):
        d = np.sqrt(d)
        
        alpha = 1
        rho = self.rho
        reused = {}
        
        for i in range(self.niter):
            # Calc reused variables and gradF
            t = time.time()
            self.calc_reused(reused, vars)
            print('calc_reused',time.time()-t)
            
            t = time.time()
            self.gradientF(reused, d)
            print('gradientF',time.time()-t)

            # debug and visualization
            t = time.time()
            self.error_debug(vars, reused, d, i)
            print('error_debug',time.time()-t)

            t = time.time()
            self.vis_debug(vars, i)
            print('vis_debug',time.time()-t)
            # gradients for each variable
            
            t = time.time()
            grads = self.gradients(vars, reused)
            print('gradients',time.time()-t)
            
            t = time.time()
            if i == 0 or self.method == "BH-GD":
                etas = {}
                #etas["u"] = -grads["u"] * rho[0] ** 2
                etas["u"] = mulc(grads['u'],-rho[0]**2,self.pool_cpu)
                #etas["Ru"] = -grads["Ru"] * rho[0] ** 2
                etas["Ru"] = mulc(grads['Ru'],-rho[0]**2,self.pool_cpu)
                #etas["q"] = -grads["q"] * rho[1] ** 2
                etas["q"] = mulc(grads['q'],-rho[1]**2,self.pool_cpu)
                #etas["r"] = -grads["r"] * rho[2] ** 2
                etas["r"] = mulc(grads['r'],-rho[2]**2,self.pool_cpu)
            else:
                # conjugate direction
                beta = self.calc_beta(vars, grads, etas, reused, d)
                etas["u"] = -grads["u"] * rho[0] ** 2 + beta * etas["u"]
                #linear(etas['u'],grads['u'],etas['u'],-rho[0]**2,beta,self.pool_cpu)                                
                etas["Ru"] = -grads["Ru"] * rho[0] ** 2 + beta * etas["Ru"]
                #linear(etas['Ru'],grads['Ru'],etas['Ru'],-rho[0]**2,beta,self.pool_cpu)                                
                etas["q"] = -grads["q"] * rho[1] ** 2 + beta * etas["q"]
                #linear(etas['q'],grads['q'],etas['q'],-rho[1]**2,beta,self.pool_cpu)                                
                etas["r"] = -grads["r"] * rho[2] ** 2 + beta * etas["r"]
                #linear(etas['r'],grads['r'],etas['r'],-rho[2]**2,beta,self.pool_cpu)                                
            print('beta',time.time()-t)
            
            # step length
            t = time.time()
            alpha, top, bottom = self.calc_alpha(vars, grads, etas, reused, d)
            print('alpha',time.time()-t)
            # print(alpha,top,bottom)
            # print(np.linalg.norm(grads['u']))
            # print(np.linalg.norm(grads['q']))
            # print(np.linalg.norm(grads['r']))
            # debug approxmation
            self.plot_debug(vars, etas, top, bottom, alpha, d, i)
            t = time.time()
            # vars["u"] += alpha * etas["u"]
            linear(vars["u"],vars["u"],etas['u'],1,alpha,self.pool_cpu)                                
            # vars["q"] += alpha * etas["q"]
            linear(vars["q"],vars["q"],etas['q'],1,alpha,self.pool_cpu)                                
            # vars["r"] += alpha * etas["r"]
            linear(vars["r"],vars["r"],etas['r'],1,alpha,self.pool_cpu)                                
            # vars['Ru'] += alpha * etas["Ru"]
            linear(vars["Ru"],vars["Ru"],etas['Ru'],1,alpha,self.pool_cpu)                                
            vars["psi"] = self.expR(vars['Ru'])

            print('exp update',time.time()-t)
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
    
    def Rotate(self,u):
        flg = chunking_flg(locals().values())
        # memory for result
        if flg:
            res = np.empty([self.npsi, self.npsi,self.npsi], dtype="complex64")
        else:
            res = cp.empty([self.npsi, self.npsi,self.npsi], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _Rotate0(self,res,u):
            res[:]=ndimage.rotate(u,25,reshape=False,axes=(1,2))
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=2, axis_inp=2)
        def _Rotate2(self,res,u):            
            res[:]=ndimage.rotate(u,-1.5,reshape=False,axes=(0,1))
        _Rotate0(self,res,u)
        _Rotate2(self,res,res)
        
        return res 
        
    def R(self, obj):
        
        flg = chunking_flg(locals().values())
        # memory for result
        if flg:
            data = np.empty([self.ntheta, self.npsi,self.npsi], dtype="complex64")
        else:
            data = cp.empty([self.ntheta, self.npsi,self.npsi], dtype="complex64")
        
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
            obj = np.empty([self.npsi, self.npsi,self.npsi], dtype="complex64")
        else:
            obj = cp.empty([self.npsi, self.npsi,self.npsi], dtype="complex64")
        
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

    def Spsi(self, psi, r):
        # memory for result
        flg = chunking_flg(locals().values())        
        if flg:
            res = np.empty([len(r), self.nq, self.nq], dtype="complex64")
        else:
            res = cp.empty([len(r), self.nq, self.nq], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _Spsi(self,res, psi,r):
            """Extract patches with subpixel shift"""

            x = cp.fft.fftfreq(self.nq).astype("float32")
            [y, x] = cp.meshgrid(x, x)
            tmp = cp.exp(
                -2 * cp.pi * 1j * (y * r[:, 1, None, None] + x * r[:, 0, None, None])
            ).astype("complex64")
            res[:] = cp.fft.ifft2(tmp * cp.fft.fft2(psi))
            
        _Spsi(self, res, psi,r)
        return res

    def STpsi(self, psi, r):
        # memory for result
        flg = chunking_flg(locals().values())        
        if flg:
            res = np.empty([len(r), self.nq, self.nq], dtype="complex64")
        else:
            res = cp.empty([len(r), self.nq, self.nq], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _STpsi(self,res, psi,r):
            """Extract patches with subpixel shift"""

            x = cp.fft.fftfreq(self.nq).astype("float32")
            [y, x] = cp.meshgrid(x, x)
            tmp = cp.exp(
                2 * cp.pi * 1j * (y * r[:, 1, None, None] + x * r[:, 0, None, None])
            ).astype("complex64")
            res[:] = cp.fft.ifft2(tmp * cp.fft.fft2(psi))
            
        _STpsi(self, res, psi,r)
        return res

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
    #         ff = cp.pad(psi, ((0, 0), (self.nq // 2, self.nq // 2), (self.nq // 2, self.nq // 2)))            
    #         ff = cp.fft.ifft2(cp.fft.fft2(ff) * self.fker[cp.cuda.Device().id])            
    #         ff = ff[:, self.nq // 2 : -self.nq // 2, self.nq // 2 : -self.nq // 2]

    #         # crop to detector size
    #         big_psi[:] = ff[:, self.pad : self.nq - self.pad, self.pad : self.nq - self.pad]
        
    #     _D(self, big_psi, psi)
    #     return big_psi


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
    #         ff = cp.pad(psi, ((0, 0), (self.nq // 2, self.nq // 2), (self.nq // 2, self.nq // 2)))            
    #         ff = cp.fft.ifft2(cp.fft.fft2(ff) * self.fkerc[cp.cuda.Device().id])            
    #         ff = ff[:, self.nq // 2 : -self.nq // 2, self.nq // 2 : -self.nq // 2]
    #         big_psi[:] = ff
    #     _Dc(self, big_psi, psi)
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
    #         # convolution
    #         ff = cp.pad(ff, ((0, 0), (self.nq // 2, self.nq // 2), (self.nq // 2, self.nq // 2)))
    #         ff = cp.fft.ifft2(cp.fft.fft2(ff) / self.fker[cp.cuda.Device().id])
    #         psi[:] = ff[:, self.nq // 2 : -self.nq // 2, self.nq // 2 : -self.nq // 2]
    #     _DT(self, psi, big_psi)
    #     return psi    

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
    #         # convolution
    #         ff = cp.pad(ff, ((0, 0), (self.nq // 2, self.nq // 2), (self.nq // 2, self.nq // 2)))
    #         ff = cp.fft.ifft2(cp.fft.fft2(ff) / self.fkerc[cp.cuda.Device().id])
    #         psi[:] = ff[:, self.nq // 2 : -self.nq // 2, self.nq // 2 : -self.nq // 2]
    #     _DcT(self, psi, big_psi)
    #     return psi    
    

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
            ff = self._fwd_pad(ff)
            v = cp.ones(2*self.nq,dtype='float32')
            v[:self.nq//2] = cp.sin(cp.linspace(0,1,self.nq//2)*cp.pi/2)
            v[-self.nq//2:] = cp.cos(cp.linspace(0,1,self.nq//2)*cp.pi/2)
            v = cp.outer(v,v)
            ff *= v*2        
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
            v = cp.ones(2*self.nq,dtype='float32')
            v[:self.nq//2] = cp.sin(cp.linspace(0,1,self.nq//2)*cp.pi/2)
            v[-self.nq//2:] = cp.cos(cp.linspace(0,1,self.nq//2)*cp.pi/2)
            v = cp.outer(v,v)
            ff *= v*2        
            psi[:] = self._adj_pad(ff)
        _DT(self, psi, big_psi)
        return psi
    
    def Dc(self, psi):
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
            ff = self._fwd_pad(ff)
            v = cp.ones(2*self.nq,dtype='float32')
            v[:self.nq//2] = cp.sin(cp.linspace(0,1,self.nq//2)*cp.pi/2)
            v[-self.nq//2:] = cp.cos(cp.linspace(0,1,self.nq//2)*cp.pi/2)
            v = cp.outer(v,v)
            ff *= v*2        
            ff = cp.fft.ifft2(cp.fft.fft2(ff)*self.fkerc[cp.cuda.Device().id])
            ff = ff[:,self.nq//2:-self.nq//2,self.nq//2:-self.nq//2]
            big_psi[:] = ff

        _Dc(self, big_psi, psi)
        return big_psi

    def DcT(self, big_psi):
        """Adjoint propagator"""
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
            ff = cp.pad(ff,((0,0),(self.nq//2,self.nq//2),(self.nq//2,self.nq//2)))
            ff = cp.fft.ifft2(cp.fft.fft2(ff)/self.fkerc[cp.cuda.Device().id])
            v = cp.ones(2*self.nq,dtype='float32')
            v[:self.nq//2] = cp.sin(cp.linspace(0,1,self.nq//2)*cp.pi/2)
            v[-self.nq//2:] = cp.cos(cp.linspace(0,1,self.nq//2)*cp.pi/2)
            v = cp.outer(v,v)
            ff *= v*2        
            psi[:] = self._adj_pad(ff)
            
        _DcT(self, psi, big_psi)
        return psi    







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
    
    def gradient_u(self, scode, gradF, psi, rpsi, u, q):

        # memory for result
        flg = chunking_flg(locals().values())        
        if flg:
            gradpsi = np.empty([self.ntheta,self.npsi, self.npsi], dtype="complex64")
        else:
            gradpsi = cp.empty([self.ntheta,self.npsi, self.npsi], dtype="complex64")

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)                
        def _gradient_psi(self, gradpsi, scode, gradF, psi, rpsi, q):            
            gradpsi[:] = cp.conj(self.Dc(scode*q)) * gradF * cp.conj(psi)
            gradpsi[:] = self.STpsi(gradpsi,rpsi)
        _gradient_psi(self, gradpsi, scode, gradF, psi,rpsi, q)

        grad_u = self.RT(gradpsi)#*(-1j)
        gg = self.G(self.G(u))
        linear(grad_u,grad_u,gg,-1j,2*self.lam,self.pool_cpu)
        return grad_u
    
    def gradient_q(self, scode, gradF, psi):
        flg = chunking_flg(locals().values())        
        if flg:
            gradq = []
            for igpu in range(self.ngpus):
                with cp.cuda.Device(igpu):
                    gradq.append(cp.zeros([self.nq, self.nq], dtype="complex64"))
        else:
            gradq = cp.zeros([self.nq, self.nq], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)                
        def _gradient_q(self, gradq, scode, gradF, psi):
            gradq[:] += cp.sum(cp.conj(scode) * self.DcT(cp.conj(psi)*gradF),axis=0)        
        _gradient_q(self, gradq, scode, gradF,  psi)

        if flg:
            for k in range(1,len(gradq)):
                gradq[0] += gradq[k]
            gradq = gradq[0]    
            
        return gradq
   
    def gradient_r(self, ri, r, psi, gradF, code, q):
        
        flg = chunking_flg(locals().values())        
        if flg:
            gradr = np.empty([len(ri), 2], dtype="float32")
        else:
            gradr = cp.empty([len(ri), 2], dtype="float32")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)                               
        def _gradient_r(self, gradr, ri, r, psi, gradF, code, q):

            # frequencies
            xi1 = cp.fft.fftfreq(self.npatch).astype("float32")
            xi2, xi1 = cp.meshgrid(xi1, xi1)

            # multipliers in frequencies
            w = cp.exp(
                -2 * cp.pi * 1j * (xi2 * r[:, 1, None, None] + xi1 * r[:, 0, None, None])
            )

            # Gradient parts
            tmp = self.E(ri, code)
            tmp = cp.fft.fft2(tmp)
            
            dt1 = cp.fft.ifft2(w * xi1 * tmp)
            dt2 = cp.fft.ifft2(w * xi2 * tmp)
            dt1 = -2 * cp.pi * 1j * dt1[:, self.ex : self.nq + self.ex, self.ex : self.nq + self.ex]
            dt2 = -2 * cp.pi * 1j * dt2[:, self.ex : self.nq + self.ex, self.ex : self.nq + self.ex]

            # inner product with gradF

            dm1 = self.Dc(dt1*q)*psi
            dm2 = self.Dc(dt2*q)*psi

            gradr[:, 0] = redot(gradF, dm1, axis=(1, 2))
            gradr[:, 1] = redot(gradF, dm2, axis=(1, 2))
        
        _gradient_r(self, gradr, ri, r, psi, gradF, code, q)
        return gradr
    
    def gradients(self, vars, reused):
        (q, code, u, psi, ri, r, rpsi) = (vars["q"], vars["code"], vars["u"], vars["psi"], vars["ri"], vars["r"], vars["rpsi"])
        (gradF, scode) = (reused["gradF"], reused["scode"])
        t = time.time()
        du = self.gradient_u(scode, gradF, psi, rpsi, u, q)
        print('grad_u',time.time()-t)
        t = time.time()
        dprb = self.gradient_q(scode, gradF, psi)
        print('grad_q',time.time()-t)
        t = time.time()
        dr = self.gradient_r(ri, r, psi, gradF, code, q)
        print('grad_r',time.time()-t)
        t = time.time()
        Rdu = self.Spsi(self.R(du),rpsi)
        print('Rdu',time.time()-t)
        grads = {"u": du, "Ru": Rdu, "q": dprb, "r": dr}
        
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
        (q, code, u, psi, ri, r) = (vars["q"], vars['code'], vars["u"],vars["psi"], vars["ri"], vars["r"])
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
        def _calc_reused2(self, big_psi, scode, psi, q):
            big_psi[:] = self.D(self.Dc(scode*q)*psi)                    

        _calc_reused1(self, scode, ri, r, code)        
        _calc_reused2(self, big_psi, scode, psi, q)            
        
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
    
    def calc_beta(self, vars, grads, etas, reused, d):
        (q, code, psi, ri, r) = (vars["q"], vars["code"], vars["psi"], vars["ri"], vars["r"])
        (scode, big_psi, gradF) = (reused["scode"], reused["big_psi"], reused["gradF"])
        (du1, Rdu1, dq1, dr1) = (grads["u"], grads["Ru"], grads["q"], grads["r"])
        (du2, Rdu2, dq2, dr2) = (etas["u"], etas["Ru"], etas["q"], etas["r"])        

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
        def _calc_beta(self, res, ri, r, scode, big_psi, gradF, dr1, dr2, psi,Rdu1, Rdu2, d, q, dq1, dq2, code):            
            # note scaling with rho
            Rdu1 = Rdu1 * self.rho[0] ** 2
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

            tmp = self.E(ri, code)
            tmp = cp.fft.fft2(tmp)
            dt1 = -2 * cp.pi * 1j * cp.fft.ifft2(w * w1 * tmp)[:, self.ex : self.nq + self.ex, self.ex : self.nq + self.ex]
            dt2 = -2 * cp.pi * 1j * cp.fft.ifft2(w * w2 * tmp)[:, self.ex : self.nq + self.ex, self.ex : self.nq + self.ex]
            d2t1 = -4 * cp.pi**2 * cp.fft.ifft2(w * w12 * tmp)[:, self.ex : self.nq + self.ex, self.ex : self.nq + self.ex]
            d2t2 = -4 * cp.pi**2 * cp.fft.ifft2(w * w22 * tmp)[:, self.ex : self.nq + self.ex, self.ex : self.nq + self.ex]

            # D^2M|_{(q_0,c_0,{x}_0)}\big((\Delta q, \Delta c,\Delta{x}),(\Delta q, \Delta c,\Delta{x})\big)=
            # q_0\cdot D^2{T_c}|_{{{x}_{0,k}}}(\Delta{x}_k,\Delta{x}_k)
            # 2\Delta q\cdot DT_{c_0}|_{{{x}_{0,k}}}( \Delta {x}_k)\Big)
            d2m1 = q * d2t1
            d2m1 += dq1 * dt2 + dq2 * dt1

            d2m2 = q * d2t2
            d2m2 += dq2 * dt2 + dq2 * dt2

            # DM|_{(q_0,c_0,{x}_0)}(\Delta q, \Delta c,\Delta{x})= 
            # J(\Delta q)\cdot S_{{x}_{0}}(c_0)+  
            # \Big(q_0\cdot DT_{c_0}|_{{{x}_{0,k}}}( \Delta {x}_k) \Big)_{k=1}^K
            dm1 = dq1 * scode + q * dt1
            dm2 = dq2 * scode + q * dt2
            
            # psi(L_1(DM|_{q_0,x_0}(\Delta q,\Delta x)) + L_1\Big(M(q_0,x_0))\Big)(iR(\Delta u)))\\
            dn1 = psi*(self.Dc(dm1) + self.Dc(q*scode)*1j*Rdu1)
            dn2 = psi*(self.Dc(dm2) + self.Dc(q*scode)*1j*Rdu2)
        
            # psi(
            # \frac{1}{2}L_1(D^2M|_{q_0,x_0}\big((\Delta q,\Delta x),(\Delta q,\Delta x)\big))
            # L_1(DM|_{q_0,x_0}(\Delta q,\Delta x))(iR(\Delta u))
            # frac{1}{2}L_1\Big(M(q_0,x_0)\Big)(iR({\Delta u}))^2)
            d2n1 = psi*(self.Dc(d2m1) + self.Dc(dm1)*1j*Rdu2+ self.Dc(dm2)*1j*Rdu1-self.Dc(q*scode)*Rdu1*Rdu2)
            d2n2 = psi*(self.Dc(d2m2) + self.Dc(dm2)*1j*Rdu2+ self.Dc(dm2)*1j*Rdu2-self.Dc(q*scode)*Rdu2*Rdu2)
            
            # top and bottom parts
            Ddn1 = self.D(dn1)
            Ddn2 = self.D(dn2)
            top = redot(gradF, d2n1) + self.hessian_F(big_psi, Ddn1, Ddn2, d)
            bottom = redot(gradF, d2n2) + self.hessian_F(big_psi, Ddn2, Ddn2, d)
            res[0] += top
            res[1] += bottom
            
        _calc_beta(self, res, ri, r, scode, big_psi, gradF, dr1, dr2, psi,Rdu1, Rdu2, d, q, dq1, dq2, code)  
        if flg:
            for k in range(1,len(res)):
                res[0][0] += res[k][0]
                res[0][1] += res[k][1]
            res = res[0]      
        
        gu1 = self.G(du1)
        gu2 = self.G(du2)
        
        res[0] += 2*self.lam*self.redot_batch(gu1,gu2)
        res[1] += 2*self.lam*self.redot_batch(gu2,gu2)

        top = cp.float32((res[0]).get()) 
        bottom = cp.float32((res[1]).get()) 
        return top/bottom
    

    def calc_alpha(self, vars, grads, etas, reused, d):
        (q, code, psi, ri, r) = (vars["q"], vars["code"], vars["psi"], vars["ri"], vars["r"])
        (du1,Rdu1, dq1, dr1) = (grads['u'],grads["Ru"], grads["q"], grads["r"])
        (du2,Rdu2, dq2, dr2) = (etas['u'],etas["Ru"], etas["q"], etas["r"])       
        (scode, big_psi, gradF) = (reused["scode"], reused["big_psi"], reused["gradF"])
        
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
        def _calc_alpha(self, res, ri, r, scode, big_psi, gradF, dr1, dr2, psi, Rdu2, d, q, dq2, code):            
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

            # DT,D2T terms, and scode
            
            tmp = self.E(ri, code)
            tmp = cp.fft.fft2(tmp)
            dt = -2 * cp.pi * 1j * cp.fft.ifft2(w * w1 * tmp)[:, self.ex : self.nq + self.ex, self.ex : self.nq + self.ex]
            d2t = -4 * cp.pi**2 * cp.fft.ifft2(w * w2 * tmp)[:, self.ex : self.nq + self.ex, self.ex : self.nq + self.ex]

            # D^2M|_{(q_0,c_0,{x}_0)}\big((\Delta q, \Delta c,\Delta{x}),(\Delta q, \Delta c,\Delta{x})\big)=
            # q_0\cdot D^2{T_c}|_{{{x}_{0,k}}}(\Delta{x}_k,\Delta{x}_k)
            # 2\Delta q\cdot DT_{c_0}|_{{{x}_{0,k}}}( \Delta {x}_k)\Big)
            
            d2m2 = q * d2t
            d2m2 += dq2 * dt + dq2 * dt

            # DM|_{(q_0,c_0,{x}_0)}(\Delta q, \Delta c,\Delta{x})= 
            # J(\Delta q)\cdot S_{{x}_{0}}(c_0)+  
            # \Big(q_0\cdot DT_{c_0}|_{{{x}_{0,k}}}( \Delta {x}_k) \Big)_{k=1}^K
            dm2 = dq2 * scode + q * dt
            
            # psi(L_1(DM|_{q_0,x_0}(\Delta q,\Delta x)) + L_1\Big(M(q_0,x_0))\Big)(iR(\Delta u)))\\
            dn2 = psi*(self.Dc(dm2) + self.Dc(q*scode)*1j*Rdu2)
        
            # psi(
            # \frac{1}{2}L_1(D^2M|_{q_0,x_0}\big((\Delta q,\Delta x),(\Delta q,\Delta x)\big))
            # L_1(DM|_{q_0,x_0}(\Delta q,\Delta x))(iR(\Delta u))
            # frac{1}{2}L_1\Big(M(q_0,x_0)\Big)(iR({\Delta u}))^2)
            d2n2 = psi*(self.Dc(d2m2) + self.Dc(dm2)*1j*Rdu2+ self.Dc(dm2)*1j*Rdu2-self.Dc(q*scode)*Rdu2*Rdu2)
            
            # top and bottom parts
            Ddn2 = self.D(dn2)
            bottom = redot(gradF, d2n2) + self.hessian_F(big_psi, Ddn2, Ddn2, d)

            res[0] += top
            res[1] += bottom
            

        _calc_alpha(self, res, ri, r, scode, big_psi, gradF, dr1, dr2, psi, Rdu2, d, q, dq2, code)
        if flg:
            for k in range(1,len(res)):
                res[0][0] += res[k][0]
                res[0][1] += res[k][1]                
            res = res[0]
        res[0] += -redot(du1, du2) - redot(dq1, dq2)
      
        gu2 = self.G(du2)        
        res[1] += 2*self.lam*self.redot_batch(gu2,gu2)

        top = cp.float32((res[0]).get()) 
        bottom = cp.float32((res[1]).get()) 
        return top/bottom,top,bottom
    
    def fwd(self,ri,r,rpsi,code, u,q):
                
        # memory for result
        flg = chunking_flg(locals().values())        
        if flg:
            res = np.empty([len(ri), self.n, self.n], dtype="complex64")
        else:
            res = cp.empty([len(ri), self.n, self.n], dtype="complex64")
                
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _fwd(self,res,ri,r,psi,code,q):            
            scode = self.S(ri,r,code)
            res[:] = self.D(self.Dc(scode * q)*psi)
        psi = self.expR(self.Spsi(self.R(u),rpsi))     
        
        _fwd(self,res,ri,r,psi,code,q)
        return res

    def plot_debug(self, vars, etas, top, bottom, alpha, d, i):
        """Check the minimization functional behaviour"""
        if i % self.vis_step == 0 and self.vis_step != -1 and self.show:
            (q, code, u, ri, r,rpsi) = (vars["q"], vars["code"], vars["u"], vars["ri"], vars["r"], vars["rpsi"])
            (dq2, du2, dr2) = (etas["q"], etas["u"], etas["r"])
                        
            npp = 3
            errt = cp.zeros(npp * 2)
            errt2 = cp.zeros(npp * 2)
            for k in range(0, npp * 2):
                ut = u + (alpha * k / (npp - 1)) * du2
                qt = q + (alpha * k / (npp - 1)) * dq2
                rt = r + (alpha * k / (npp - 1)) * dr2
                tmp=self.fwd(ri,rt,rpsi,code,ut,qt)
                errt[k] = self.minF(tmp, d) + self.lam*np.linalg.norm(self.G(ut))**2 
                
            t = alpha * (cp.arange(2 * npp)) / (npp - 1)
            tmp = self.fwd(ri,r,rpsi,code,u,q)
            errt2 = self.minF(tmp, d) + self.lam*np.linalg.norm(self.G(u))**2            
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
            (q, u) = (vars["q"], vars["u"])
            
            ur = u#self.Rotate(u)
            mshow_complex(ur[u.shape[0]//2].real+1j*ur[:,u.shape[1]//2].real,self.show)
            mshow_polar(q,self.show)
            write_tiff(ur.real,f'{self.path_out}/rec_u_real/{i:04}')
            # write_tiff(u.imag,f'{self.path_out}/rec_u_imag/{i:04}')
            write_tiff(ur[self.npsi//2].real,f'{self.path_out}/rec_uz/{i:04}')
            write_tiff(ur[:,self.npsi//2].real,f'{self.path_out}/rec_uy/{i:04}')            
            write_tiff(cp.angle(q),f'{self.path_out}/rec_prb_angle/{i:04}')
            write_tiff(cp.abs(q),f'{self.path_out}/rec_prb_abs/{i:04}')
            if self.show:
                plt.plot(vars['r'][:,1],'.')
                plt.plot(vars['r'][:,0],'.')
                plt.show()

    def error_debug(self,vars, reused, d, i):
        """Visualization and data saving"""
        u = vars['u']
        if i % self.err_step == 0 and self.err_step != -1:
            err = self.minF(reused["big_psi"], d) + self.lam*np.linalg.norm(self.G(u))**2 
            print(f"{i}) {err=:1.5e}", flush=True)
            vars["table"].loc[len(vars["table"])] = [i, err.get(), time.time()]
            name = f'{self.path_out}/conv.csv'
            os.makedirs(os.path.dirname(name), exist_ok=True)
            vars["table"].to_csv(name, index=False)
    