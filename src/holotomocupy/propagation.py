import math
import cupy as cp
from .cuda_kernels import *
from .utils import *


class Propagation:
    """Functionality for Propagation"""

    def __init__(self, n, nz, ndist, wavelength, voxelsize, distance):
        # Fresnel kernels
        fx = cp.fft.fftfreq(2*n, d=voxelsize).astype("float32")
        fy = cp.fft.fftfreq(2*nz, d=voxelsize).astype("float32")
        [fx, fy] = cp.meshgrid(fx, fy)
        self.fker = cp.empty([ndist, 2*nz, 2*n], dtype="complex64")
        # self.ifker = cp.empty([ndist, 2*nz, 2*n], dtype="complex64")
        for j in range(ndist):
            # propagation sample-detector
            self.fker[j] = cp.exp(-1j * cp.pi * wavelength * distance[j] * (fx**2 + fy**2))
            # self.ifker[j] = cp.exp(1j * cp.pi * wavelength * distance[j] * (fx**2 + fy**2))

        self.n = n
        self.nz = nz

    def _fwd_pad(self, f):
        """Fwd data padding"""
        [ntheta, nz, n] = f.shape
        fpad = cp.zeros([ntheta, 2 * nz, 2 * n], dtype="complex64")
        # f=cp.ascontiguousarray(f)
        pad_kernel(
            (math.ceil(2 * n / 32), math.ceil(2 * nz / 32), ntheta),
            (32, 32, 1),
            (fpad, f, n, nz, ntheta, 0),
        )
        return fpad 

    def _adj_pad(self, fpad):
        """Adj data padding"""
        [ntheta, nz, n] = fpad.shape
        n //= 2
        nz //=2
        f = cp.zeros([ntheta, nz, n], dtype="complex64")
        # fpad=cp.ascontiguousarray(fpad)
        pad_kernel(
            (math.ceil(2 * n / 32), math.ceil(2 * nz / 32), ntheta),
            (32, 32, 1),
            (fpad, f, n, nz, ntheta, 1),
        )
        return f 

    def D(self, psi, j):
        """Forward propagator"""
        if len(psi.shape) == 2:
            psi = psi[cp.newaxis]
            flg = 1
        else:
            flg = 0
        
        ff = psi.copy()
        ff = self._fwd_pad(ff)
        ff = cp.fft.ifft2(cp.fft.fft2(ff) * self.fker[j])
        ff = ff[:, self.nz // 2 : -self.nz // 2, self.n // 2 : -self.n // 2]
        
        # big_psi[:] = cp.fft.ifft2(cp.fft.fft2(psi) * self.fker[j])
        if flg == 1:
            ff = ff[0]

        return ff

    def DT(self, big_psi, j):
        """Adjoint propagator"""
        if len(big_psi.shape) == 2:
            big_psi = big_psi[cp.newaxis]
            flg = 1
        else:
            flg = 0
        
        # pad to the probe size
        ff = big_psi.copy()
        ff = cp.pad(ff, ((0, 0), (self.nz // 2, self.nz // 2), (self.n // 2, self.n // 2)))
        ff = cp.fft.ifft2(cp.fft.fft2(ff) / self.fker[j])
        ff = self._adj_pad(ff)

        # ff[:] = cp.fft.ifft2(cp.fft.fft2(big_psi) * 1 / self.fker[j])
        if flg == 1:
            ff = ff[0]
        return ff
