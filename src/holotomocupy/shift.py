import math
import numpy as np
import cupy as cp
import cupyx.scipy.fft as cufft
from .cuda_kernels import (
    s_kernel, sf_kernel,
    sback_kernel,
    ds_kernel, dsf_kernel,
    d2s_kernel, d2sf_kernel,
    dsadj_kernel, dsadjf_kernel,
)
from .utils import redot


def _ascontig(x):
    """cp.ascontiguousarray that also accepts numpy/pinned input (auto-uploads).
    Newer cupy rejects numpy inputs to ascontiguousarray; we go through cp.asarray
    first so callers can pass pinned-numpy slices (e.g. vars['pos'][:, k]) directly."""
    return cp.ascontiguousarray(cp.asarray(x))


class Shift():
    """Cubic B-spline shift operator (requires coeff() prefilter, C2 smooth)."""

    def __init__(self, n, npsi, nz, nzpsi, obj_dtype, nchunk=None):
        self.n = n
        self.npsi = npsi
        self.nz = nz
        self.nzpsi = nzpsi
        self.obj_dtype = obj_dtype

        # Forward B-spline denominator (unit magnification, k=0,1)
        x = cp.linspace(-1/2, 1/2 - 1/npsi,  npsi ).astype('float32')
        y = cp.linspace(-1/2, 1/2 - 1/nzpsi, nzpsi).astype('float32')
        divx = (self.phi(0) + 2 * self.phi(1) * cp.cos(2 * cp.pi * x)).astype('float32')
        divy = (self.phi(0) + 2 * self.phi(1) * cp.cos(2 * cp.pi * y)).astype('float32')
        self.ifB3 = 1 / cp.fft.fftshift(cp.outer(divy, divx), axes=(-1, -2))


        if nchunk is not None:
            _tmp = cp.empty([nchunk, nzpsi, npsi], dtype='complex64')
            self._plan_coeff       = cufft.get_fft_plan(_tmp, axes=(-2, -1), value_type='C2C')
            self._plan_coeff_batch = nchunk
            del _tmp
        else:
            self._plan_coeff = None

        self._coeff_cache  = {}
        self._coeff_hits   = 0
        self._coeff_misses = 0

        self._sk      = s_kernel
        self._sfk     = sf_kernel
        self._dsk     = ds_kernel
        self._dsfk    = dsf_kernel
        self._d2sk    = d2s_kernel
        self._d2sfk   = d2sf_kernel
        self._dsadjk  = dsadj_kernel
        self._dsadjfk = dsadjf_kernel

    # ------------------------------------------------------------------
    # B-spline basis
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
        if self._plan_coeff is not None and psi.shape[0] == self._plan_coeff_batch:
            with self._plan_coeff:
                out = cufft.ifft2(cufft.fft2(psi) * self.ifB3)
        else:
            out = cp.fft.ifft2(cp.fft.fft2(psi) * self.ifB3)
        if self.obj_dtype == 'float32':
            out = out.real
        return out

    def coeff_cached(self, psi):
        """coeff(psi) memoized by id(psi). MUST be paired with explicit
        coeff_cache_reset() at safe lifetime boundaries (e.g., per chunk):
        id() values can be reused across distinct Python objects once an
        earlier one is garbage-collected. Hit/miss counters are exposed for
        verification; reset along with the cache."""
        key = id(psi)
        cached = self._coeff_cache.get(key)
        if cached is None:
            self._coeff_misses += 1
            cached = self.coeff(psi)
            self._coeff_cache[key] = cached
        else:
            self._coeff_hits += 1
        return cached

    def coeff_cache_reset(self):
        self._coeff_cache = {}

    def coeff_cache_stats(self, reset=False):
        """Return (hits, misses) accumulated since the last stats reset.
        Set reset=True to zero the counters."""
        stats = (self._coeff_hits, self._coeff_misses)
        if reset:
            self._coeff_hits = 0
            self._coeff_misses = 0
        return stats

    # ------------------------------------------------------------------
    # Forward / adjoint shift  S / S*
    # ------------------------------------------------------------------

    def S(self, c, r, m):
        ntheta = c.shape[0]
        # Kernel writes every (t, k, i) — no need to zero first.
        spsi = cp.empty([ntheta, self.nz, self.n], dtype=self.obj_dtype)
        c = _ascontig(c)
        r = _ascontig(r)
        m = _ascontig(m)
        self._launch(self._sk, self._sfk, ntheta,
                     (spsi, c, r, m,
                      self.n, self.npsi, self.nz, self.nzpsi, ntheta, 0))
        return spsi

    def Sadj(self, spsi, r, m):
        ntheta = spsi.shape[0]
        c = cp.zeros([ntheta, self.nzpsi, self.npsi], dtype=self.obj_dtype)
        spsi = _ascontig(spsi)
        r = _ascontig(r)
        m = _ascontig(m)
        self._launch(self._sk, self._sfk, ntheta,
                     (spsi, c, r, m,
                      self.n, self.npsi, self.nz, self.nzpsi, ntheta, 1))
        return c

    # ------------------------------------------------------------------
    # Composed operator  curlyS = S ∘ coeff
    # ------------------------------------------------------------------

    def curlyS(self, psi, r, m):
        out = self.S(self.coeff(psi), r, m)
        if self.obj_dtype == 'float32':
            out = out.real
        return out

    # ------------------------------------------------------------------
    # Back-projection shift for Paganin initial guess
    # ------------------------------------------------------------------

    def coeff_back(self, psi):
        """B-spline prefilter on the small (n x nz) input grid for back-interpolation."""
        xs = cp.linspace(-1/2, 1/2 - 1/self.n,  self.n ).astype('float32')
        ys = cp.linspace(-1/2, 1/2 - 1/self.nz, self.nz).astype('float32')
        divx = (self.phi(0) + 2 * self.phi(1) * cp.cos(2 * cp.pi * xs)).astype('float32')
        divy = (self.phi(0) + 2 * self.phi(1) * cp.cos(2 * cp.pi * ys)).astype('float32')
        ifB3 = 1 / cp.fft.fftshift(cp.outer(divy, divx), axes=(-1, -2))
        return cp.fft.ifft2(cp.fft.fft2(psi) * ifB3)

    def Sback(self, c, r, m):
        """Gather-interpolate from small (n x nz) B-spline coefficients to large (npsi x nzpsi) grid."""
        ntheta = c.shape[0]
        # Kernel writes every (t, k, i) — no zero-fill needed.
        g    = cp.empty([ntheta, self.nzpsi, self.npsi], dtype='complex64')
        c    = _ascontig(c)
        r    = _ascontig(r)
        m    = _ascontig(cp.asarray(m, dtype='float32'))
        sback_kernel(
            (math.ceil(self.npsi / 32), math.ceil(self.nzpsi / 32), ntheta),
            (32, 32, 1),
            (g, c, r, m,
             self.n, self.npsi, self.nz, self.nzpsi, ntheta),
        )
        return g

    def curlySback(self, psi, r, m):
        """Interpolate from small (n x nz) grid to large (npsi x nzpsi) with shift+magnification."""
        return self.Sback(self.coeff_back(psi), r, m)

    # ------------------------------------------------------------------
    # Optimized coefficient-space variants  (operate on pre-computed coefficients)
    # ------------------------------------------------------------------

    def curlySc(self, c, r, m):
        ntheta = c.shape[0]
        spsi = cp.empty([ntheta, self.nz, self.n], dtype=self.obj_dtype)
        c = _ascontig(c)
        r = _ascontig(r)
        m = _ascontig(m)
        self._launch(self._sk, self._sfk, ntheta,
                     (spsi, c, r, m,
                      self.n, self.npsi, self.nz, self.nzpsi, ntheta, 0))
        return spsi

    def dcurlySc(self, c, r, m, c1, Deltar):
        ntheta = c.shape[0]
        res     = cp.empty([ntheta, self.nz, self.n], self.obj_dtype)
        c       = _ascontig(c)
        c1      = _ascontig(c1)
        r       = _ascontig(r)
        Deltar  = _ascontig(Deltar)
        m       = _ascontig(m)

        self._launch(self._dsk, self._dsfk, ntheta,
                     (res, c, c1, r, m, Deltar,
                      self.n, self.npsi, self.nz, self.nzpsi, ntheta))

        return res

    def dcurlySadjc(self, c, r, m, Deltaphi):

        ntheta = c.shape[0]
        # out1 is an atomicAdd target -> MUST be zeroed. dt1/dt2 are written every
        # position by the kernel; out2 is overwritten by the redot() lines below.
        out1 = cp.zeros([ntheta, self.nzpsi, self.npsi], dtype=self.obj_dtype)
        out2  = cp.empty(r.shape, dtype='float32')
        dt1  = cp.empty(Deltaphi.shape, self.obj_dtype)
        dt2  = cp.empty(Deltaphi.shape, self.obj_dtype)
        c        = _ascontig(c)
        r        = _ascontig(r)
        Deltaphi = _ascontig(Deltaphi)
        m        = _ascontig(m)

        self._launch(self._dsadjk, self._dsadjfk, ntheta,
                     (out1, dt1, dt2, c, Deltaphi, r, m,
                      self.n, self.npsi, self.nz, self.nzpsi, ntheta))

        out2[:, 0] = redot(Deltaphi, dt1, axis=(1, 2))
        out2[:, 1] = redot(Deltaphi, dt2, axis=(1, 2))

        return [out1, out2]

    def d2curlySc(self, c, r, m, c1, Deltar1, c2, Deltar2):

        ntheta = c.shape[0]
        res     = cp.empty([ntheta, self.nz, self.n], self.obj_dtype)
        c       = _ascontig(c)
        c1      = _ascontig(c1)
        c2      = _ascontig(c2)
        r       = _ascontig(r)
        Deltar1 = _ascontig(Deltar1)
        Deltar2 = _ascontig(Deltar2)
        m       = _ascontig(m)

        self._launch(self._d2sk, self._d2sfk, ntheta,
                     (res, c, c1, c2, r, m, Deltar1, Deltar2,
                      self.n, self.npsi, self.nz, self.nzpsi, ntheta))
        return res

