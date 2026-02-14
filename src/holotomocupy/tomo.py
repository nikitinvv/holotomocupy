import numpy as np
import cupy as cp
import math
from .cuda_kernels import *
from .utils import *


class Tomo:
    """Functionality for Radon transforms and exp"""

    def __init__(self, n, theta, mask_r):
        """Usfft parameters"""
        eps = 1e-3  # accuracy of usfft
        mu = -cp.log(eps) / (2 * n * n)
        m = math.ceil(2 * n * 1 / cp.pi * cp.sqrt(-mu * cp.log(eps) + (mu * n) * (mu * n) / 4))
        # extra arrays
        # interpolation kernel
        t = cp.linspace(-1 / 2, 1 / 2, n, endpoint=False).astype("float32")
        [dx, dy] = cp.meshgrid(t, t)
        phi = cp.exp((mu * (n * n) * (dx * dx + dy * dy)).astype("float32")) * (1 - n % 4)

        # (+1,-1) arrays for fftshift
        c1dfftshift = (1 - 2 * ((cp.arange(1, n + 1) % 2))).astype("int8")
        c2dtmp = 1 - 2 * ((cp.arange(1, 2 * n + 1) % 2)).astype("int8")
        c2dfftshift = cp.outer(c2dtmp, c2dtmp)

        mua = cp.array([mu], dtype="float32")        
        
        self.n = n
        self.ntheta = len(theta)
        self.theta = cp.array(theta.astype("float32"))

        if mask_r > 0:
            x = np.linspace(-1, 1, self.n)
            [x, y] = np.meshgrid(x, x)
            circ = (x**2 + y**2 < mask_r).astype("float32")
            g = np.exp(-(20**2) * (x**2 + y**2))
            # g/=np.linalg.norm(g)
            fcirc = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(circ)))
            fg = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(g)))
            self.mask = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fcirc * fg))).real.astype("float32")
            self.mask /= np.amax(self.mask)
        else:
            self.mask = 1

        phi *= cp.array(self.mask/(n * np.sqrt(n * self.ntheta)))
        self.pars = m, mua, phi, c1dfftshift, c2dfftshift

    def R(self, obj):
        """Radon transform"""
        [nz, n, n] = obj.shape
                
        m, mua, phi, c1dfftshift, c2dfftshift = self.pars
        sino = cp.zeros([self.ntheta, nz, n], dtype="complex64")

        # STEP0: multiplication by phi, padding
        fde = obj * phi
        fde = cp.pad(fde, ((0, 0), (n // 2, n // 2), (n // 2, n // 2)))
        # STEP1: fft 2d
        fde*=c2dfftshift
        fde = cp.fft.fft2(fde)
        fde *= c2dfftshift

        gather_kernel(
            (math.ceil(n / 32), math.ceil(self.ntheta / 32), nz),
            (32, 32, 1),
            (sino, fde, self.theta, m, mua, n, self.ntheta, nz, 0),
        )
        # STEP3: ifft 1d
        sino*=c1dfftshift
        sino= cp.fft.ifft(sino) 
        sino*= c1dfftshift

        # STEP4: Shift based on the rotation axis
        # t = cp.fft.fftfreq(n).astype("float32")
        # w = cp.exp(2 * cp.pi * 1j * t * (-self.rotation_axis + n / 2))
        # sino = cp.fft.ifft(w * cp.fft.fft(sino))
        # normalization for the unity test
        sino /= 4

        ### convert the result back
        if obj.dtype=='float32':
            sino=sino.real
        return cp.ascontiguousarray(sino)

    def RT(self, data):
        """Adjoint Radon transform"""

        [ntheta, nz, n] = data.shape
        
        m, mua, phi, c1dfftshift, c2dfftshift = self.pars

        # STEP0: Shift based on the rotation axis
        # t = cp.fft.fftfreq(n).astype("float32")
        # w = cp.exp(-2 * cp.pi * 1j * t * (-self.rotation_axis + n / 2))
        # sino = cp.fft.ifft(w * cp.fft.fft(sino))

        # STEP1: fft 1d
        sino = data.copy()
        sino*=c1dfftshift
        sino = cp.fft.fft(sino) 
        sino*= c1dfftshift

        # STEP2: interpolation (gathering) in the frequency domain
        fde = cp.zeros([nz, 2 * n, 2 * n], dtype="complex64")
        gather_kernel(
            (math.ceil(n / 32), math.ceil(self.ntheta / 32), nz),
            (32, 32, 1),
            (sino, fde, self.theta, m, mua, n, self.ntheta, nz, 1),
        )
        # STEP3: ifft 2d
        fde*=c2dfftshift
        fde[:] = cp.fft.ifft2(fde)
        fde *= c2dfftshift

        # STEP4: unpadding, multiplication by phi
        fde = fde[:, n // 2 : 3 * n // 2, n // 2 : 3 * n // 2] * phi
        ## convert the result back
        if data.dtype=='float32':
            fde=fde.real
        return cp.ascontiguousarray(fde)

    def rec_tomo(self, d, niter=1):
        """Regular tomography reconstrution for initial guess"""

        def minf(Ru, d):
            return np.linalg.norm(Ru - d) ** 2

        u = cp.zeros([d.shape[1], self.n, self.n], dtype=d.dtype)
        Ru = self.R(u)
        tmp = np.empty_like(d)
        for k in range(niter):
            if k % 32 == 0:
                print(f"{k} err={minf(Ru,d)}")
            tmp = 2 * (Ru - d)
            grad = self.RT(tmp)
            Rgrad = self.R(grad)
            if k == 0:
                eta = -grad
                Reta = -Rgrad
            else:
                beta = redot(Rgrad, Reta) / redot(Reta, Reta)
                eta = beta * eta - grad
                Reta = beta * Reta - Rgrad
            alpha = -redot(grad, eta) / (2 * redot(Reta, Reta))
            u += alpha * eta
            Ru += alpha * Reta

        return u
