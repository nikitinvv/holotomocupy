import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import os

from cuda_kernels import *
from utils import *

class Shift():    
    #########################  Functionality for Shifts #########################    
    def __init__(self):
        pass
    
    def S(self, r, psi):
        """Shift operator"""
        patches = cp.empty([len(r), psi.shape[-1], psi.shape[-1]], dtype="complex64")
        x = cp.fft.fftfreq(psi.shape[-1]).astype("float32")
        [y, x] = cp.meshgrid(x, x)
        tmp = cp.exp(-2 * cp.pi * 1j * (y * r[:, 1, None, None] + x * r[:, 0, None, None])).astype("complex64")
        patches[:] = cp.fft.ifft2(tmp * cp.fft.fft2(psi))

        return patches

    def ST(self, r, psi):
        """Shift adjoint"""
        return self.S(-r, psi)

    def STa(self, r, psi, mode, **kwargs):
        """Shift operator with a given shift (for data preprocessing)"""
        n = psi.shape[-1]
        patches = cp.empty([len(r), n, n], dtype="complex64")

        x = cp.fft.fftfreq(2 * n).astype("float32")
        [y, x] = cp.meshgrid(x, x)
        tmp = cp.exp(2 * cp.pi * 1j * (y * r[:, 1, None, None] + x * r[:, 0, None, None])).astype("complex64")
        psi = np.pad(psi, ((0, 0), (n // 2, n // 2), (n // 2, n // 2)), mode, **kwargs)
        tmp = cp.fft.ifft2(tmp * cp.fft.fft2(psi))
        patches[:] = tmp[:, n // 2 : -n // 2, n // 2 : -n // 2]

        return patches

    