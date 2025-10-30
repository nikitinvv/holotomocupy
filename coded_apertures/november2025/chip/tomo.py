import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import os

from cuda_kernels import *
from utils import *


class Tomo:
    ######################### Functionality for Radon transforms and exp #########################
    def __init__(self, n, theta, rotation_axis):
        """Usfft parameters"""
        eps = 1e-3  # accuracy of usfft
        mu = -cp.log(eps) / (2 * n * n)
        m = int(cp.ceil(2 * n * 1 / cp.pi * cp.sqrt(-mu * cp.log(eps) + (mu * n) * (mu * n) / 4)))
        # extra arrays
        # interpolation kernel
        t = cp.linspace(-1 / 2, 1 / 2, n, endpoint=False).astype("float32")
        [dx, dy] = cp.meshgrid(t, t)
        phi = cp.exp((mu * (n * n) * (dx * dx + dy * dy)).astype("float32")) * (1 - n % 4)

        # (+1,-1) arrays for fftshift
        c1dfftshift = (1 - 2 * ((cp.arange(1, n + 1) % 2))).astype("int8")
        c2dtmp = 1 - 2 * ((cp.arange(1, 2 * n + 1) % 2)).astype("int8")
        c2dfftshift = cp.outer(c2dtmp, c2dtmp)

        self.pars = m, mu, phi, c1dfftshift, c2dfftshift
        self.n = n
        self.rotation_axis = rotation_axis
        self.ntheta = len(theta)
        self.theta = cp.array(theta.astype("float32"))

    def R(self, obj, out=None):
        """Radon transform"""

        [nz, n, n] = obj.shape
        theta = cp.array(self.theta, dtype="float32")

        m, mu, phi, c1dfftshift, c2dfftshift = self.pars
        sino = cp.zeros([self.ntheta, nz, n], dtype="complex64")

        # STEP0: multiplication by phi, padding
        fde = obj * phi
        fde = cp.pad(fde, ((0, 0), (n // 2, n // 2), (n // 2, n // 2)))
        # STEP1: fft 2d
        fde = cp.fft.fft2(fde * c2dfftshift) * c2dfftshift

        mua = cp.array([mu], dtype="float32")

        gather_kernel(
            (int(cp.ceil(n / 32)), int(cp.ceil(self.ntheta / 32)), nz),
            (32, 32, 1),
            (sino, fde, theta, m, mua, n, self.ntheta, nz, 0),
        )
        # STEP3: ifft 1d
        sino = cp.fft.ifft(c1dfftshift * sino) * c1dfftshift

        # STEP4: Shift based on the rotation axis
        # t = cp.fft.fftfreq(n).astype("float32")
        # w = cp.exp(2 * cp.pi * 1j * t * (-self.rotation_axis + n / 2))
        # sino = cp.fft.ifft(w * cp.fft.fft(sino))
        # normalization for the unity test
        sino /= cp.float32(4 * n) * np.sqrt(n * self.ntheta)

        return cp.ascontiguousarray(sino)

    def RT(self, data):
        """Adjoint Radon transform"""
        sino = data.copy()  # .swapaxes(0,1)

        [ntheta, nz, n] = sino.shape
        theta = cp.array(self.theta, dtype="float32")

        m, mu, phi, c1dfftshift, c2dfftshift = self.pars

        # STEP0: Shift based on the rotation axis
        # t = cp.fft.fftfreq(n).astype("float32")
        # w = cp.exp(-2 * cp.pi * 1j * t * (-self.rotation_axis + n / 2))
        # sino = cp.fft.ifft(w * cp.fft.fft(sino))

        # STEP1: fft 1d
        sino = cp.fft.fft(c1dfftshift * sino) * c1dfftshift

        # STEP2: interpolation (gathering) in the frequency domain
        # dont understand why RawKernel cant work with float, I have to send it as an array (TODO)
        mua = cp.array([mu], dtype="float32")
        fde = cp.zeros([nz, 2 * n, 2 * n], dtype="complex64")
        gather_kernel(
            (int(cp.ceil(n / 32)), int(cp.ceil(self.ntheta / 32)), nz),
            (32, 32, 1),
            (sino, fde, theta, m, mua, n, self.ntheta, nz, 1),
        )
        # STEP3: ifft 2d
        fde = cp.fft.ifft2(fde * c2dfftshift) * c2dfftshift

        # STEP4: unpadding, multiplication by phi
        fde = fde[:, n // 2 : 3 * n // 2, n // 2 : 3 * n // 2] * phi
        fde /= cp.float32(n) * np.sqrt(n * ntheta)  # normalization for the unity test

        return cp.ascontiguousarray(fde)

    def expR(self, psi, out=None):
        if out is None:
            epsi = cp.empty([self.ntheta, self.n, self.n], dtype="complex64")
        else:
            epsi = out

        epsi[:] = np.exp(1j * psi)
        return epsi

    def rec_tomo(self, d, niter=1):
        """Regular tomography reconstrution for initial guess"""

        def minf(Ru, d):
            return np.linalg.norm(Ru - d) ** 2

        u = cp.zeros([d.shape[1], self.npsi, self.npsi], dtype="complex64")
        Ru = self.R(u)
        tmp = np.empty_like(d)
        for k in range(niter):
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
            print(f"{k} err={minf(Ru,d)}")
        return u
