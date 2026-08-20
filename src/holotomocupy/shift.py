import math
import numpy as np
import cupy as cp
import cupyx.scipy.fft as cufft
from .cuda_kernels import (
    s_kernel,
    sback_kernel,
    ds_kernel,
    d2s_kernel,
    dsadj_kernel,
)
from .utils import redot


def _ascontig(x):
    """cp.ascontiguousarray that also accepts numpy/pinned input (auto-uploads).
    Newer cupy rejects numpy inputs to ascontiguousarray; we go through cp.asarray
    first so callers can pass pinned-numpy slices (e.g. vars['pos'][:, k]) directly."""
    return cp.ascontiguousarray(cp.asarray(x))


class Shift():
    """Cubic B-spline shift operator (requires coeff() prefilter, C2 smooth)."""

    def __init__(self, n, npsi, nz, nzpsi, nchunk=None):
        self.n = n
        self.npsi = npsi
        self.nz = nz
        self.nzpsi = nzpsi

        # Forward B-spline denominator (unit magnification, k=0,1)
        x = cp.linspace(-1/2, 1/2 - 1/npsi,  npsi ).astype('float32')
        y = cp.linspace(-1/2, 1/2 - 1/nzpsi, nzpsi).astype('float32')
        divx = (self.phi(0) + 2 * self.phi(1) * cp.cos(2 * cp.pi * x)).astype('float32')
        divy = (self.phi(0) + 2 * self.phi(1) * cp.cos(2 * cp.pi * y)).astype('float32')
        self.ifB3 = 1 / cp.fft.fftshift(cp.outer(divy, divx), axes=(-1, -2))


        # cuFFT C2C plans for coeff(), memoized by input shape. The global cupy
        # plan cache is disabled (rec_mpi sets its size to 0) to keep GPU memory
        # down, so without this every short trailing chunk of a Chunking.run --
        # and every 2-D call from the near-field-ptycho variant -- would rebuild
        # a plan + work area on each call. The shape set is tiny and fixed
        # ([nchunk, ...], [tail, ...], [nzpsi, npsi]) and every extra plan is
        # smaller than the nchunk one, so the added GPU memory is minor.
        self._coeff_plans = {}
        if nchunk is not None:
            self._coeff_plan((nchunk, nzpsi, npsi))   # warm up the common shape

        self._coeff_cache  = {}
        self._coeff_hits   = 0
        self._coeff_misses = 0

        self._sk      = s_kernel
        self._dsk     = ds_kernel
        self._d2sk    = d2s_kernel
        self._dsadjk  = dsadj_kernel

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

    def _launch(self, kernel, ntheta, args):
        grid = (math.ceil(self.n / 16), math.ceil(self.nz / 16), ntheta)
        kernel(grid, (16, 16, 1), args)

    # ------------------------------------------------------------------
    # B-spline coefficient computation
    # ------------------------------------------------------------------

    def _coeff_plan(self, shape):
        """cuFFT C2C plan for a `shape` transform over the last two axes, memoized."""
        plan = self._coeff_plans.get(shape)
        if plan is None:
            _tmp = cp.empty(shape, dtype='complex64')
            plan = cufft.get_fft_plan(_tmp, axes=(-2, -1), value_type='C2C')
            self._coeff_plans[shape] = plan
            del _tmp
        return plan

    def coeff(self, psi):
        """B-spline prefilter  ifft2(fft2(psi) * ifB3).

        One object-plane temporary instead of three: fft2 allocates it, the ifB3
        multiply is in place, and ifft2 transforms in place -- the same
        overwrite_x pattern already used in Tomo.R/RT.
        """
        with self._coeff_plan(tuple(psi.shape)):
            out = cufft.fft2(psi)
            out *= self.ifB3
            out = cufft.ifft2(out, overwrite_x=True)
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
        spsi = cp.empty([ntheta, self.nz, self.n], dtype='complex64')
        c = _ascontig(c)
        r = _ascontig(r)
        m = _ascontig(m)
        self._launch(self._sk, ntheta,
                     (spsi, c, r, m,
                      self.n, self.npsi, self.nz, self.nzpsi, ntheta, 0))
        return spsi

    def Sadj(self, spsi, r, m):
        ntheta = spsi.shape[0]
        c = cp.zeros([ntheta, self.nzpsi, self.npsi], dtype='complex64')
        spsi = _ascontig(spsi)
        r = _ascontig(r)
        m = _ascontig(m)
        self._launch(self._sk, ntheta,
                     (spsi, c, r, m,
                      self.n, self.npsi, self.nz, self.nzpsi, ntheta, 1))
        return c

    # ------------------------------------------------------------------
    # Composed operator  curlyS = S ∘ coeff
    # ------------------------------------------------------------------

    def curlyS(self, psi, r, m):
        return self.S(self.coeff(psi), r, m)

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
        spsi = cp.empty([ntheta, self.nz, self.n], dtype='complex64')
        c = _ascontig(c)
        r = _ascontig(r)
        m = _ascontig(m)
        self._launch(self._sk, ntheta,
                     (spsi, c, r, m,
                      self.n, self.npsi, self.nz, self.nzpsi, ntheta, 0))
        return spsi

    def dcurlySc(self, c, r, m, c1, Deltar):
        ntheta = c.shape[0]
        res     = cp.empty([ntheta, self.nz, self.n], 'complex64')
        c       = _ascontig(c)
        c1      = _ascontig(c1)
        r       = _ascontig(r)
        Deltar  = _ascontig(Deltar)
        m       = _ascontig(m)

        self._launch(self._dsk, ntheta,
                     (res, c, c1, r, m, Deltar,
                      self.n, self.npsi, self.nz, self.nzpsi, ntheta))
        
        return res
        
    def dcurlySadjc(self, c, r, m, Deltaphi):
        
        ntheta = c.shape[0]
        # out1 is an atomicAdd target -> MUST be zeroed. dt1/dt2 are written every
        # position by the kernel; out2 is overwritten by the redot() lines below.
        out1 = cp.zeros([ntheta, self.nzpsi, self.npsi], dtype='complex64')
        out2  = cp.empty(r.shape, dtype='float32')
        dt1  = cp.empty(Deltaphi.shape, 'complex64')
        dt2  = cp.empty(Deltaphi.shape, 'complex64')
        c        = _ascontig(c)
        r        = _ascontig(r)
        Deltaphi = _ascontig(Deltaphi)
        m        = _ascontig(m)

        self._launch(self._dsadjk, ntheta,
                     (out1, dt1, dt2, c, Deltaphi, r, m,
                      self.n, self.npsi, self.nz, self.nzpsi, ntheta))

        out2[:, 0] = redot(Deltaphi, dt1, axis=(1, 2))
        out2[:, 1] = redot(Deltaphi, dt2, axis=(1, 2))
        
        return [out1, out2]

    def d2curlySc(self, c, r, m, c1, Deltar1, c2, Deltar2):

        ntheta = c.shape[0]
        res     = cp.empty([ntheta, self.nz, self.n], 'complex64')
        c       = _ascontig(c)
        c1      = _ascontig(c1)
        c2      = _ascontig(c2)
        r       = _ascontig(r)
        Deltar1 = _ascontig(Deltar1)
        Deltar2 = _ascontig(Deltar2)
        m       = _ascontig(m)

        self._launch(self._d2sk, ntheta,
                     (res, c, c1, c2, r, m, Deltar1, Deltar2,
                      self.n, self.npsi, self.nz, self.nzpsi, ntheta))
        return res
    