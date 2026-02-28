import math
import numpy as np
import cupy as cp
from .cuda_kernels import gather_kernel
from .utils import redot, logger


class Tomo:
    """Functionality for Radon transforms and exp"""

    def __init__(self, n, theta, mask_r):
        """Usfft parameters"""
        eps = 1e-3  # accuracy of usfft
        mu = -math.log(eps) / (2 * n * n)
        m  = math.ceil(2 * n / math.pi * math.sqrt(-mu * math.log(eps) + (mu * n) ** 2 / 4))

        # interpolation kernel
        t = cp.linspace(-1 / 2, 1 / 2, n, endpoint=False).astype("float32")
        dx, dy = cp.meshgrid(t, t)
        phi = cp.exp((mu * (n * n) * (dx * dx + dy * dy)).astype("float32")) * (1 - n % 4)

        # (+1,-1) sign arrays for fftshift-via-multiply
        c1dfftshift = (1 - 2 * ((cp.arange(1, n + 1) % 2))).astype("int8")
        c2dtmp      = (1 - 2 * ((cp.arange(1, 2 * n + 1) % 2))).astype("int8")
        c2dfftshift = cp.outer(c2dtmp, c2dtmp)

        mua = cp.array([mu], dtype="float32")

        self.n      = n
        self.ntheta = len(theta)
        self.theta  = cp.array(theta.astype("float32"))

        if mask_r > 0:
            t1d = np.linspace(-1, 1, self.n)
            x, y = np.meshgrid(t1d, t1d)
            circ  = (x**2 + y**2 < mask_r).astype("float32")
            g     = np.exp(-(20**2) * (x**2 + y**2))
            fcirc = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(circ)))
            fg    = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(g)))
            mask  = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fcirc * fg))).real.astype("float32")
            mask /= np.amax(mask)
        else:
            mask = 1.0

        self.mask = mask
        phi *= cp.array(mask / (n * np.sqrt(n * self.ntheta)))
        self.pars = m, mua, phi, c1dfftshift, c2dfftshift

    def R(self, obj):
        """Radon transform"""
        nz = obj.shape[0]
        n  = self.n
        m, mua, phi, c1dfftshift, c2dfftshift = self.pars
        sino = cp.zeros([self.ntheta, nz, n], dtype="complex64")

        # STEP0: multiplication by phi, padding
        fde = obj * phi
        fde = cp.pad(fde, ((0, 0), (n // 2, n // 2), (n // 2, n // 2)))
        # STEP1: 2D FFT
        fde *= c2dfftshift
        fde  = cp.fft.fft2(fde)
        fde *= c2dfftshift
        # STEP2: NUFFT gather (Cartesian -> polar)
        gather_kernel(
            (math.ceil(n / 32), math.ceil(self.ntheta / 32), nz),
            (32, 32, 1),
            (sino, fde, self.theta, m, mua, n, self.ntheta, nz, 0),
        )
        # STEP3: 1D IFFT along detector axis
        sino *= c1dfftshift
        sino  = cp.fft.ifft(sino)
        sino *= c1dfftshift
        # STEP4: normalization
        sino /= 4

        if obj.dtype == 'float32':
            sino = sino.real
        return cp.ascontiguousarray(sino)

    def RT(self, data):
        """Adjoint Radon transform"""
        nz = data.shape[1]
        n  = self.n
        m, mua, phi, c1dfftshift, c2dfftshift = self.pars

        # STEP1: 1D FFT along detector axis
        sino  = data * c1dfftshift
        sino  = cp.fft.fft(sino)
        sino *= c1dfftshift
        # STEP2: NUFFT scatter (polar -> Cartesian)
        fde = cp.zeros([nz, 2 * n, 2 * n], dtype="complex64")
        gather_kernel(
            (math.ceil(n / 32), math.ceil(self.ntheta / 32), nz),
            (32, 32, 1),
            (sino, fde, self.theta, m, mua, n, self.ntheta, nz, 1),
        )
        # STEP3: 2D IFFT
        fde *= c2dfftshift
        fde[:] = cp.fft.ifft2(fde)
        fde *= c2dfftshift
        # STEP4: unpadding and multiplication by phi
        fde = fde[:, n // 2 : 3 * n // 2, n // 2 : 3 * n // 2] * phi

        if data.dtype == 'float32':
            fde = fde.real
        return cp.ascontiguousarray(fde)

    def rec_tomo(self, d, niter=1):
        """Regular tomography reconstruction for initial guess"""

        def minf(Ru, d):
            return np.linalg.norm(Ru - d) ** 2

        u  = cp.zeros([d.shape[1], self.n, self.n], dtype=d.dtype)
        Ru = self.R(u)
        for k in range(niter):
            if k % 32 == 0:
                logger.info(f"rec_tomo iter {k}: err={minf(Ru, d):.6e}")
            tmp   = 2 * (Ru - d)
            grad  = self.RT(tmp)
            Rgrad = self.R(grad)
            if k == 0:
                eta  = -grad
                Reta = -Rgrad
            else:
                beta = redot(Rgrad, Reta) / redot(Reta, Reta)
                eta  = beta * eta  - grad
                Reta = beta * Reta - Rgrad
            alpha = -redot(grad, eta) / (2 * redot(Reta, Reta))
            u  += alpha * eta
            Ru += alpha * Reta

        return u
