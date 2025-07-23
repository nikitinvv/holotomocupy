import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import os

from cuda_kernels import *
from utils import *

class Magnification():    
    
    ############################## Functionality for  Magnifications #########################
    def __init__(self, n, ne, norm_magnifications):
        # usfft parameters
        eps = 1e-3  # accuracy of usfft
        mu = -cp.log(eps) / (2 * ne * ne)
        m = int(cp.ceil(2 * ne * 1 / cp.pi * cp.sqrt(-mu * cp.log(eps) + (mu * ne) * (mu * ne) / 4)))
        # extra arrays
        # interpolation kernel
        t = cp.linspace(-1 / 2, 1 / 2, ne, endpoint=False).astype("float32")
        [dx, dy] = cp.meshgrid(t, t)
        phi = cp.exp((mu * (ne * ne) * (dx * dx + dy * dy)).astype("float32")) * (1 - ne % 4)

        # (+1,-1) arrays for fftshift
        c2dtmp = 1 - 2 * ((cp.arange(1, 2 * ne + 1) % 2)).astype("int8")
        c2dfftshift = cp.outer(c2dtmp, c2dtmp)
        c2dtmp = 1 - 2 * ((cp.arange(1, ne + 1) % 2)).astype("int8")
        c2dfftshift0 = cp.outer(c2dtmp, c2dtmp)
        self.pars = m, mu, phi, c2dfftshift, c2dfftshift0
        self.n = n
        self.ne = ne
        self.norm_magnifications = norm_magnifications

    def M(self, psi, j):
        res = cp.zeros([len(psi), self.n, self.n], dtype="complex64")

        magnification = self.norm_magnifications[j] * self.ne / (self.n)
        m, mu, phi, c2dfftshift, c2dfftshift0 = self.pars
        # FFT2D
        fde = cp.fft.fft2(psi * c2dfftshift0) * c2dfftshift0
        # adjoint USFFT2D
        fde = fde * phi

        fde = cp.pad(fde, ((0, 0), (self.ne // 2, self.ne // 2), (self.ne // 2, self.ne // 2)))
        fde = cp.fft.fft2(fde * c2dfftshift) * c2dfftshift
        fde = cp.pad(fde, ((0, 0), (m, m), (m, m)))
        wrap_kernel(
            (int(cp.ceil((2 * self.ne + 2 * m) / 32)), int(cp.ceil((2 * self.ne + 2 * m) / 32)), len(psi)),
            (32, 32, 1),
            (fde, self.ne, len(psi), m),
        )
        mua = cp.array([mu], dtype="float32")
        magnificationa = cp.array([magnification], dtype="float32")
        gather_mag_kernel(
            (int(cp.ceil(self.n / 32)), int(cp.ceil(self.n / 32)), len(psi)),
            (32, 32, 1),
            (res, fde, magnificationa, m, mua, self.n, self.ne, len(psi), 0),
        )

        res[:] /= cp.float32(4 * self.ne**3)

        return res

    def MT(self, psi, j):
        res = cp.zeros([len(psi), self.ne, self.ne], dtype="complex64")

        magnification = self.norm_magnifications[j] * self.ne / (self.n)
        m, mu, phi, c2dfftshift, c2dfftshift0 = self.pars

        mua = cp.array([mu], dtype="float32")
        magnificationa = cp.array([magnification], dtype="float32")
        fde = cp.zeros([len(psi), 2 * m + 2 * self.ne, 2 * m + 2 * self.ne], dtype="complex64")
        psi = cp.ascontiguousarray(psi)
        gather_mag_kernel(
            (int(cp.ceil(self.n / 32)), int(cp.ceil(self.n / 32)), len(psi)),
            (32, 32, 1),
            (psi, fde, magnificationa, m, mua, self.n, self.ne, len(psi), 1),
        )
        wrapadj_kernel(
            (int(cp.ceil((2 * self.ne + 2 * m) / 32)), int(cp.ceil((2 * self.ne + 2 * m) / 32)), len(psi)),
            (32, 32, 1),
            (fde, self.ne, len(psi), m),
        )

        fde = fde[:, m:-m, m:-m]
        fde = cp.fft.ifft2(fde * c2dfftshift) * c2dfftshift

        fde = fde[:, self.ne // 2 : 3 * self.ne // 2, self.ne // 2 : 3 * self.ne // 2] * phi
        res[:] = cp.fft.ifft2(fde * c2dfftshift0) * c2dfftshift0

        return res