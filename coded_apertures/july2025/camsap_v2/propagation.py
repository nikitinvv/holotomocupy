import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import os

from cuda_kernels import *
from utils import *

class Propagation():    
    ############################# Functionality for Propagation #########################
    def __init__(self, n, ndist, wavelength, voxelsize, distance, distancep):
        # Fresnel kernels
        fx = cp.fft.fftfreq(n * 2, d=voxelsize).astype("float32")
        [fx, fy] = cp.meshgrid(fx, fx)
        self.fker = cp.empty([ndist, 2 * n, 2 * n], dtype="complex64")
        self.fkerp = cp.empty([ndist, 2 * n, 2 * n], dtype="complex64")
        for j in range(ndist):
            # propagation sample-detector
            self.fker[j] = cp.exp(-1j * cp.pi * wavelength * distance[j] * (fx**2 + fy**2))
            # propagation probe-sample
            self.fkerp[j] = cp.exp(-1j * cp.pi * wavelength * distancep[j] * (fx**2 + fy**2))
        self.n = n
    
    def _fwd_pad(self, f):
        """Fwd data padding"""
        [ntheta, n] = f.shape[:2]
        fpad = cp.zeros([ntheta, 2 * n, 2 * n], dtype="complex64")
        pad_kernel((int(cp.ceil(2 * n / 32)), int(cp.ceil(2 * n / 32)), ntheta), (32, 32, 1), (fpad, f, n, ntheta, 0))
        return fpad / 2

    def _adj_pad(self, fpad):
        """Adj data padding"""
        [ntheta, n] = fpad.shape[:2]
        n //= 2
        f = cp.zeros([ntheta, n, n], dtype="complex64")
        pad_kernel((int(cp.ceil(2 * n / 32)), int(cp.ceil(2 * n / 32)), ntheta), (32, 32, 1), (fpad, f, n, ntheta, 1))
        return f / 2

    def D(self, psi, j):
        """Forward propagator"""
        big_psi = cp.empty([len(psi), self.n, self.n], dtype="complex64")

        ff = self._fwd_pad(psi)
        ff *= 2
        ff = cp.fft.ifft2(cp.fft.fft2(ff) * self.fker[j])
        ff = ff[:, self.n // 2 : -self.n // 2, self.n // 2 : -self.n // 2]
        big_psi[:] = ff

        return big_psi
    
    # def D1(self, psi, j):
    #     """Forward propagator"""
    #     big_psi = cp.empty([len(psi), self.n, self.n], dtype="complex64")

    #     ff = cp.pad(psi,((0,0),(self.n//2,self.n//2),(self.n//2,self.n//2)))        
    #     ff = cp.fft.ifft2(cp.fft.fft2(ff) * self.fker[j])
    #     ff = ff[:, self.n // 2 : -self.n // 2, self.n // 2 : -self.n // 2]
    #     big_psi[:] = ff

    #     return big_psi
    
    # def D2(self, psi, j):
    #     """Forward propagator"""
    #     big_psi = cp.empty([len(psi), self.n, self.n], dtype="complex64")

    #     ff = cp.pad(psi,((0,0),(3*self.n//2,3*self.n//2),(3*self.n//2,3*self.n//2)))        
    #      # Fresnel kernels
    #     fx = cp.fft.fftfreq(self.n * 4, d=self.voxelsize).astype("float32")
    #     [fx, fy] = cp.meshgrid(fx, fx)
    #     fker = cp.empty([self.ndist, 4 * self.n, 4* self.n], dtype="complex64")
    #     for j in range(self.ndist):
    #         fker[j] = cp.exp(-1j * cp.pi * self.wavelength * self.distance[j] * (fx**2 + fy**2))
       
    #     ff = cp.fft.ifft2(cp.fft.fft2(ff) * fker)
    #     ff = ff[:, 3*self.n // 2 : -3*self.n // 2, 3*self.n // 2 : -3*self.n // 2]
    #     big_psi[:] = ff

    #     return big_psi

    def DT(self, big_psi, j):
        """Adjoint propagator"""
        psi = cp.empty([len(big_psi), self.n, self.n], dtype="complex64")

        # pad to the probe size
        ff = big_psi
        ff = cp.pad(ff, ((0, 0), (self.n // 2, self.n // 2), (self.n // 2, self.n // 2)))
        ff = cp.fft.ifft2(cp.fft.fft2(ff) / self.fker[j])
        ff *= 2
        psi[:] = self._adj_pad(ff)
        return psi
    def Dp(self, psi, j):
        """Forward propagator"""
        big_psi = cp.empty([len(psi), self.n, self.n], dtype="complex64")

        ff = self._fwd_pad(psi)
        ff *= 2
        ff = cp.fft.ifft2(cp.fft.fft2(ff) * self.fkerp[j])
        ff = ff[:, self.n // 2 : -self.n // 2, self.n // 2 : -self.n // 2]
        big_psi[:] = ff

        return big_psi

    def DpT(self, big_psi, j):
        """Adjoint propagator"""
        psi = cp.empty([len(big_psi), self.n, self.n], dtype="complex64")

        # pad to the probe size
        ff = big_psi
        ff = cp.pad(ff, ((0, 0), (self.n // 2, self.n // 2), (self.n // 2, self.n // 2)))
        ff = cp.fft.ifft2(cp.fft.fft2(ff) / self.fkerp[j])
        ff *= 2
        psi[:] = self._adj_pad(ff)
        return psi