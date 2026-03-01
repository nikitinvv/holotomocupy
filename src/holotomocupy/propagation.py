import math
import cupy as cp
from .cuda_kernels import pad_kernel


class Propagation:
    """Functionality for Propagation"""

    def __init__(self, n, nz, ndist, wavelength, voxelsize, distance):
        self.n  = n
        self.nz = nz

        # Fresnel kernels on the padded (2n × 2nz) grid
        fx = cp.fft.fftfreq(2 * n,  d=voxelsize).astype("float32")
        fy = cp.fft.fftfreq(2 * nz, d=voxelsize).astype("float32")
        fx, fy = cp.meshgrid(fx, fy)
        f2 = fx ** 2 + fy ** 2  # hoisted outside the distance loop

        self.fker      = cp.empty([ndist, 2 * nz, 2 * n], dtype="complex64")
        for j in range(ndist):
            self.fker[j]      = cp.exp(-1j * cp.pi * wavelength * distance[j] * f2)
            
    def _fwd_pad(self, f):
        """Symmetric padding: (ntheta, nz, n) -> (ntheta, 2nz, 2n)"""
        ntheta, nz, n = f.shape
        fpad = cp.zeros([ntheta, 2 * nz, 2 * n], dtype="complex64")
        f = cp.ascontiguousarray(f)
        pad_kernel(
            (math.ceil(2 * n / 32), math.ceil(2 * nz / 32), ntheta),
            (32, 32, 1),
            (fpad, f, n, nz, ntheta, 0),
        )
        return fpad 

    def _adj_pad(self, fpad):
        """Adjoint padding: (ntheta, 2nz, 2n) -> (ntheta, nz, n)"""
        ntheta = fpad.shape[0]
        nz     = fpad.shape[1] // 2
        n      = fpad.shape[2] // 2
        f = cp.zeros([ntheta, nz, n], dtype="complex64")
        fpad = cp.ascontiguousarray(fpad)
        pad_kernel(
            (math.ceil(2 * n / 32), math.ceil(2 * nz / 32), ntheta),
            (32, 32, 1),
            (fpad, f, n, nz, ntheta, 1),
        )
        return f 

    def D(self, psi, j):
        """Forward propagator"""
        added_dim = psi.ndim == 2
        if added_dim:
            psi = psi[cp.newaxis]

        ff = self._fwd_pad(psi)
        ff = cp.fft.ifft2(cp.fft.fft2(ff) * self.fker[j])
        ff = ff[:, self.nz // 2 : -self.nz // 2, self.n // 2 : -self.n // 2]

        return ff[0] if added_dim else ff

    def DT(self, big_psi, j):
        """Adjoint propagator"""
        added_dim = big_psi.ndim == 2
        if added_dim:
            big_psi = big_psi[cp.newaxis]

        ff = cp.pad(big_psi, ((0, 0), (self.nz // 2, self.nz // 2), (self.n // 2, self.n // 2)))
        ff = cp.fft.ifft2(cp.fft.fft2(ff)/ self.fker[j])
        ff = self._adj_pad(ff)

        return ff[0] if added_dim else ff
