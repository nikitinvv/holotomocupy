import math
import numpy as np
import cupy as cp
from cuda_kernels import (
    s_kernel, sf_kernel,
)


class Shift():
    """Functionality for Shifts"""

    def __init__(self, n, npsi, nz, nzpsi, mag, obj_dtype):
        self.n = n
        self.npsi = npsi
        self.nz = nz
        self.nzpsi = nzpsi
        self.mag = cp.array(mag).astype('float32')
        self.obj_dtype = obj_dtype

        # Forward B-spline denominator (unit magnification, k=0,1)
        x = cp.linspace(-1/2, 1/2 - 1/npsi,  npsi ).astype('float32')
        y = cp.linspace(-1/2, 1/2 - 1/nzpsi, nzpsi).astype('float32')
        divx = (self.phi(0) + 2 * self.phi(1) * cp.cos(2 * cp.pi * x)).astype('float32')
        divy = (self.phi(0) + 2 * self.phi(1) * cp.cos(2 * cp.pi * y)).astype('float32')
        self.ifB3 = 1 / cp.fft.fftshift(cp.outer(divy, divx), axes=(-1, -2))
        # # def _tukey(n, alpha=1):
        # #     """1D Tukey window, symmetric about centre (DC), in natural order."""
        # #     t = cp.linspace(0.0, 1.0, n, dtype='float32')
        # #     lo, hi = alpha / 2, 1.0 - alpha / 2
        # #     return cp.where(t < lo,
        # #                     0.5 * (1 + cp.cos(cp.pi * (2 * t / alpha - 1))),
        # #                     cp.where(t > hi,
        # #                             0.5 * (1 + cp.cos(cp.pi * (2 * t / alpha - 2 / alpha + 1))),
        # #                             cp.ones(n, dtype='float32'))).astype('float32')


        # # # Tukey window: suppresses Nyquist amplification in the inverse B-spline
        # # # prefilter, which otherwise causes ringing near sharp edges.
        # # wx = _tukey(npsi)
        # # wy = _tukey(nzpsi)
        # self.ifB3 *= cp.fft.fftshift(cp.outer(wy, wx))



    # ------------------------------------------------------------------
    # B-spline basis and its derivatives
    # ------------------------------------------------------------------

    def phi(self, t):
        return (
            (-2 < t) * (t <= -1) * (t + 2)**3
            + (-1 < t) * (t <=  1) * (4 - 6*t**2 + 3*t**3 * cp.sign(t))
            + ( 1 < t) * (t <=  2) * (2 - t)**3
        )

    # ------------------------------------------------------------------
    # Internal kernel launcher — eliminates repeated if/else dispatch
    # ------------------------------------------------------------------

    def _launch(self, kernel_c, kernel_f, ntheta, args):
        grid = (math.ceil(self.n / 16), math.ceil(self.nz / 16), ntheta)
        kernel = kernel_c if self.obj_dtype == 'complex64' else kernel_f
        kernel(grid, (16, 16, 1), args)

    # ------------------------------------------------------------------
    # B-spline coefficient computation
    # ------------------------------------------------------------------

    def coeff(self, psi):
        out = cp.fft.ifft2(cp.fft.fft2(psi) * self.ifB3)
        if self.obj_dtype == 'float32':
            out = out.real
        return out

    # ------------------------------------------------------------------
    # Forward / adjoint shift  S / S*
    # ------------------------------------------------------------------

    def S(self, c, r, imagn):
        ntheta = c.shape[0]
        spsi = cp.zeros([ntheta, self.nz, self.n], dtype=self.obj_dtype)
        c = cp.ascontiguousarray(c)
        r = cp.ascontiguousarray(r)
        self._launch(s_kernel, sf_kernel, ntheta,
                     (spsi, c, r, self.mag[imagn:imagn+1],
                      self.n, self.npsi, self.nz, self.nzpsi, ntheta, 0))
        return spsi

    def Sadj(self, spsi, r, imagn):
        ntheta = spsi.shape[0]
        c = cp.zeros([ntheta, self.nzpsi, self.npsi], dtype=self.obj_dtype)
        spsi = cp.ascontiguousarray(spsi)
        r = cp.ascontiguousarray(r)
        self._launch(s_kernel, sf_kernel, ntheta,
                     (spsi, c, r, self.mag[imagn:imagn+1],
                      self.n, self.npsi, self.nz, self.nzpsi, ntheta, 1))
        return c

    # ------------------------------------------------------------------
    # Composed operators  curlyS = S ∘ coeff
    # ------------------------------------------------------------------

    def curlyS(self, psi, r, imagn):
        out = self.S(self.coeff(psi), r, imagn)
        if self.obj_dtype == 'float32':
            out = out.real
        return out

