import math
import numpy as np
import cupy as cp
from .cuda_kernels import (
    s_kernel, sf_kernel,
    sback_kernel, 
    ds_kernel, dsf_kernel,
    d2s_kernel, d2sf_kernel,
    dsadj_kernel, dsadjf_kernel
)
from .utils import redot


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

        # Per-distance back-projection denominators (magnification-aware, k=0..4)
        # Cached here so coeffback() doesn't recompute them on every call.
        self._ifB3back = []
        for m in self.mag:
            dx = self.phi(cp.zeros(1, dtype='float32')[0]).astype('float32')
            dy = self.phi(cp.zeros(1, dtype='float32')[0]).astype('float32')
            for k in range(1, 5):
                dx = (dx + 2 * self.phi(k / m) * cp.cos(2 * cp.pi * k * x)).astype('float32')
                dy = (dy + 2 * self.phi(k / m) * cp.cos(2 * cp.pi * k * y)).astype('float32')
            self._ifB3back.append(1 / cp.fft.fftshift(cp.outer(dy, dx), axes=(-1, -2)))

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

    def coeffback(self, psi, imagn):
        return cp.fft.ifft2(cp.fft.fft2(psi) * self._ifB3back[imagn])

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
    # Composed operator  curlyS = S ∘ coeff
    # ------------------------------------------------------------------

    def curlyS(self, psi, r, imagn):
        out = self.S(self.coeff(psi), r, imagn)
        if self.obj_dtype == 'float32':
            out = out.real
        return out

    # ------------------------------------------------------------------
    # Back-projection shift for Paganin initial guess
    # ------------------------------------------------------------------

    def Sback(self, spsi, r, imagn):
        ntheta = spsi.shape[0]
        c    = cp.zeros([ntheta, self.nzpsi, self.npsi], dtype='complex64')
        spsi = cp.ascontiguousarray(spsi)
        r    = cp.ascontiguousarray(r)
        sback_kernel(
            (math.ceil(self.n / 32), math.ceil(self.nz / 32), ntheta),
            (32, 32, 1),
            (spsi, c, r, self.mag[imagn:imagn+1],
             self.n, self.npsi, self.nz, self.nzpsi, ntheta, 1),
        )
        return c

    def curlySback(self, psi, r, imagn):
        return self.coeffback(self.Sback(psi, r, imagn), imagn)

    # ------------------------------------------------------------------
    # Optimized coefficient-space variants  (operate on pre-computed coefficients)
    # ------------------------------------------------------------------

    def curlySc(self, c, r, imagn):
        ntheta = c.shape[0]
        spsi = cp.zeros([ntheta, self.nz, self.n], dtype=self.obj_dtype)
        c = cp.ascontiguousarray(c)
        r = cp.ascontiguousarray(r)
        self._launch(s_kernel, sf_kernel, ntheta,
                     (spsi, c, r, self.mag[imagn:imagn+1],
                      self.n, self.npsi, self.nz, self.nzpsi, ntheta, 0))
        return spsi

    def dcurlySc(self, c, r, imagn, c1, Deltar):
        ntheta = c.shape[0]
        res     = cp.zeros([ntheta, self.nz, self.n], self.obj_dtype)
        c       = cp.ascontiguousarray(c)
        c1      = cp.ascontiguousarray(c1)
        r       = cp.ascontiguousarray(r)
        Deltar = cp.ascontiguousarray(Deltar)
        
        self._launch(ds_kernel, dsf_kernel, ntheta,
                     (res, c, c1, r, self.mag[imagn:imagn+1], Deltar,
                      self.n, self.npsi, self.nz, self.nzpsi, ntheta))
        
        return res
        
    def dcurlySadjc(self, c, r, imagn, Deltaphi):
        
        ntheta = c.shape[0]
        out1 = cp.zeros([ntheta, self.nzpsi, self.npsi], dtype=self.obj_dtype)        
        out2  = cp.zeros(r.shape, dtype='float32')
        dt1  = cp.zeros(Deltaphi.shape, self.obj_dtype)
        dt2  = cp.zeros(Deltaphi.shape, self.obj_dtype)
        c        = cp.ascontiguousarray(c)
        r        = cp.ascontiguousarray(r)
        Deltaphi = cp.ascontiguousarray(Deltaphi)

        self._launch(dsadj_kernel, dsadjf_kernel, ntheta,
                     (out1, dt1, dt2, c, Deltaphi, r, self.mag[imagn:imagn+1],
                      self.n, self.npsi, self.nz, self.nzpsi, ntheta))

        out2[:, 0] = redot(Deltaphi, dt1, axis=(1, 2))
        out2[:, 1] = redot(Deltaphi, dt2, axis=(1, 2))
        
        return [out1, out2]

    def d2curlySc(self, c, r, imagn, c1, Deltar1, c2, Deltar2):

        ntheta = c.shape[0]
        res     = cp.zeros([ntheta, self.nz, self.n], self.obj_dtype)
        c       = cp.ascontiguousarray(c)
        c1      = cp.ascontiguousarray(c1)
        c2      = cp.ascontiguousarray(c2)
        r       = cp.ascontiguousarray(r)
        Deltar1 = cp.ascontiguousarray(Deltar1)
        Deltar2 = cp.ascontiguousarray(Deltar2)

        self._launch(d2s_kernel, d2sf_kernel, ntheta,
                     (res, c, c1, c2, r, self.mag[imagn:imagn+1], Deltar1, Deltar2,
                      self.n, self.npsi, self.nz, self.nzpsi, ntheta))
        return res
    