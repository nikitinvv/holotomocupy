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


def ascontig(x):
    """cp.ascontiguousarray that also accepts numpy/pinned input (auto-uploads).
    Newer cupy rejects numpy inputs to ascontiguousarray; we go through cp.asarray
    first so callers can pass pinned-numpy slices (e.g. vars['pos'][:, k]) directly."""
    return cp.ascontiguousarray(cp.asarray(x))


class Shift():
    """Cubic B-spline shift operator (requires coeff() prefilter, C2 smooth).

    When symmetric=True, the FFT-based B-spline prefilter operates on a 2×
    mirror-padded grid: input psi is half-sample-reflected at each boundary
    into a (2·nzpsi, 2·npsi) array, prefilter runs on that bigger grid, and
    the resulting coefficients c live on the bigger grid. The shift kernel
    is invoked with the bigger nzpsi/npsi so the interpolation correctly
    reads from the mirror-padded c. coeff() is shape-polymorphic:
      - small input  [..., nzpsi, npsi]    → forward (mirror-pad + prefilter) → big
      - big   input  [..., 2·nzpsi, 2·npsi] → adjoint (prefilter + fold-sum)   → small
    This matches what ShiftFFT does internally and eliminates the periodic-BC
    artifacts the unpadded FFT prefilter would otherwise introduce near the
    boundary. Costs 4× in memory and prefilter FFT work.
    """

    def __init__(self, n, npsi, nz, nzpsi, obj_dtype, nchunk=None, symmetric=False):
        self.n = n
        self.npsi = npsi
        self.nz = nz
        self.nzpsi = nzpsi
        self.obj_dtype = obj_dtype
        self.symmetric = symmetric

        # Effective coefficient-grid sizes that all shift kernels see. With
        # symmetric=True the grid is doubled and inputs are mirror-placed at
        # offset (sy, sx); with symmetric=False everything collapses to the
        # original sizes (sy = sx = 0).
        if symmetric:
            self.sy = nzpsi // 2
            self.sx = npsi  // 2
            self.nzpsi_eff = nzpsi + 2 * self.sy
            self.npsi_eff  = npsi  + 2 * self.sx
        else:
            self.sy = 0
            self.sx = 0
            self.nzpsi_eff = nzpsi
            self.npsi_eff  = npsi

        # Forward B-spline denominator at the EFFECTIVE grid size (unit
        # magnification, k=0,1). Different N → different denominator, so the
        # symmetric path needs its own filter sized for the bigger grid.
        x = cp.linspace(-1/2, 1/2 - 1/self.npsi_eff,  self.npsi_eff ).astype('float32')
        y = cp.linspace(-1/2, 1/2 - 1/self.nzpsi_eff, self.nzpsi_eff).astype('float32')
        divx = (self.phi(0) + 2 * self.phi(1) * cp.cos(2 * cp.pi * x)).astype('float32')
        divy = (self.phi(0) + 2 * self.phi(1) * cp.cos(2 * cp.pi * y)).astype('float32')
        self.ifB3 = 1 / cp.fft.fftshift(cp.outer(divy, divx), axes=(-1, -2))

        if nchunk is not None:
            tmp = cp.empty([nchunk, self.nzpsi_eff, self.npsi_eff], dtype='complex64')
            self.plan_coeff       = cufft.get_fft_plan(tmp, axes=(-2, -1), value_type='C2C')
            self.plan_coeff_batch = nchunk
            del tmp
        else:
            self.plan_coeff       = None
            self.plan_coeff_batch = None

        self.coeff_cache  = {}
        self.coeff_hits   = 0
        self.coeff_misses = 0

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

    def launch(self, kernel_c, kernel_f, ntheta, args):
        grid = (math.ceil(self.n / 16), math.ceil(self.nz / 16), ntheta)
        kernel = kernel_c if self.obj_dtype == 'complex64' else kernel_f
        kernel(grid, (16, 16, 1), args)

    # ------------------------------------------------------------------
    # B-spline coefficient computation
    # ------------------------------------------------------------------

    def prefilter(self, x):
        """FFT B-spline prefilter on the effective grid; x must already have
        last two dims (nzpsi_eff, npsi_eff)."""
        if (self.plan_coeff is not None and x.ndim == 3
                and x.shape[0] == self.plan_coeff_batch):
            with self.plan_coeff:
                return cufft.ifft2(cufft.fft2(x) * self.ifB3)
        return cp.fft.ifft2(cp.fft.fft2(x) * self.ifB3)

    def mirror_pad(self, x):
        """Whole-sample mirror reflection (cp.pad mode='reflect') of the last
        two axes from (nzpsi, npsi) to (nzpsi_eff, npsi_eff)."""
        n_extra = x.ndim - 2
        pad = [(0, 0)] * n_extra + [(self.sy, self.sy), (self.sx, self.sx)]
        return cp.pad(x, pad, mode='reflect')

    def fold_to_small(self, big):
        """Adjoint of mirror_pad: fold-and-sum reflected bands of the last
        two axes back into the central (nzpsi, npsi) region. With 'reflect'
        BC, the reflection axes (rows 0, N-1 and cols 0, N-1) are NOT in any
        side band, so they receive only the central contribution; other
        rows/cols pick up one extra contribution from the matching mirror."""
        sy, sx       = self.sy, self.sx
        nzpsi, npsi  = self.nzpsi, self.npsi
        # Fold y-axis: central + reversed left/right bands.
        out_y = big[..., sy:sy + nzpsi, :].copy()
        out_y[..., 1:sy + 1, :]                 += big[..., :sy, :][..., ::-1, :]
        out_y[..., nzpsi - 1 - sy:nzpsi - 1, :] += big[..., sy + nzpsi:sy + nzpsi + sy, :][..., ::-1, :]
        # Fold x-axis on out_y.
        out = out_y[..., sx:sx + npsi].copy()
        out[..., 1:sx + 1]               += out_y[..., :sx][..., ::-1]
        out[..., npsi - 1 - sx:npsi - 1] += out_y[..., sx + npsi:sx + npsi + sx][..., ::-1]
        return out

    def coeff(self, psi):
        """B-spline prefilter.
        symmetric=False: same-shape FFT prefilter (self-adjoint).
        symmetric=True: shape-polymorphic —
          - psi.shape[-1] == npsi      → forward (mirror-pad + prefilter), output big
          - psi.shape[-1] == npsi_eff  → adjoint (prefilter + fold-sum),    output small
        Either branch passes psi through one big-grid FFT pair."""
        if not self.symmetric:
            out = self.prefilter(psi)
        else:
            last = psi.shape[-1]
            if last == self.npsi:
                out = self.prefilter(self.mirror_pad(psi))
            elif last == self.npsi_eff:
                out = self.fold_to_small(self.prefilter(psi))
            else:
                raise ValueError(
                    f"Shift.coeff(symmetric=True): expected last dim "
                    f"{self.npsi} (forward) or {self.npsi_eff} (adjoint), "
                    f"got shape {psi.shape}"
                )
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
        cached = self.coeff_cache.get(key)
        if cached is None:
            self.coeff_misses += 1
            cached = self.coeff(psi)
            self.coeff_cache[key] = cached
        else:
            self.coeff_hits += 1
        return cached

    def coeff_cache_reset(self):
        self.coeff_cache = {}

    def coeff_cache_stats(self, reset=False):
        """Return (hits, misses) accumulated since the last stats reset.
        Set reset=True to zero the counters."""
        stats = (self.coeff_hits, self.coeff_misses)
        if reset:
            self.coeff_hits = 0
            self.coeff_misses = 0
        return stats

    # ------------------------------------------------------------------
    # Forward / adjoint shift  S / S*
    # ------------------------------------------------------------------

    def S(self, c, r, m):
        ntheta = c.shape[0]
        # Kernel writes every (t, k, i) — no need to zero first.
        spsi = cp.empty([ntheta, self.nz, self.n], dtype=self.obj_dtype)
        c = ascontig(c)
        r = ascontig(r)
        m = ascontig(m)
        self.launch(s_kernel, sf_kernel, ntheta,
                    (spsi, c, r, m,
                     self.n, self.npsi_eff, self.nz, self.nzpsi_eff, ntheta, 0))
        return spsi

    def Sadj(self, spsi, r, m):
        ntheta = spsi.shape[0]
        c = cp.zeros([ntheta, self.nzpsi_eff, self.npsi_eff], dtype=self.obj_dtype)
        spsi = ascontig(spsi)
        r = ascontig(r)
        m = ascontig(m)
        self.launch(s_kernel, sf_kernel, ntheta,
                    (spsi, c, r, m,
                     self.n, self.npsi_eff, self.nz, self.nzpsi_eff, ntheta, 1))
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
        c    = ascontig(c)
        r    = ascontig(r)
        m    = ascontig(cp.asarray(m, dtype='float32'))
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
        c = ascontig(c)
        r = ascontig(r)
        m = ascontig(m)
        self.launch(s_kernel, sf_kernel, ntheta,
                    (spsi, c, r, m,
                     self.n, self.npsi_eff, self.nz, self.nzpsi_eff, ntheta, 0))
        return spsi

    def dcurlySc(self, c, r, m, c1, Deltar):
        ntheta = c.shape[0]
        res     = cp.empty([ntheta, self.nz, self.n], self.obj_dtype)
        c       = ascontig(c)
        c1      = ascontig(c1)
        r       = ascontig(r)
        Deltar  = ascontig(Deltar)
        m       = ascontig(m)

        self.launch(ds_kernel, dsf_kernel, ntheta,
                    (res, c, c1, r, m, Deltar,
                     self.n, self.npsi_eff, self.nz, self.nzpsi_eff, ntheta))

        return res

    def dcurlySadjc(self, c, r, m, Deltaphi):

        ntheta = c.shape[0]
        # out1 is an atomicAdd target -> MUST be zeroed. dt1/dt2 are written every
        # position by the kernel; out2 is overwritten by the redot() lines below.
        out1 = cp.zeros([ntheta, self.nzpsi_eff, self.npsi_eff], dtype=self.obj_dtype)
        out2 = cp.empty(r.shape, dtype='float32')
        dt1  = cp.empty(Deltaphi.shape, self.obj_dtype)
        dt2  = cp.empty(Deltaphi.shape, self.obj_dtype)
        c        = ascontig(c)
        r        = ascontig(r)
        Deltaphi = ascontig(Deltaphi)
        m        = ascontig(m)

        self.launch(dsadj_kernel, dsadjf_kernel, ntheta,
                    (out1, dt1, dt2, c, Deltaphi, r, m,
                     self.n, self.npsi_eff, self.nz, self.nzpsi_eff, ntheta))

        out2[:, 0] = redot(Deltaphi, dt1, axis=(1, 2))
        out2[:, 1] = redot(Deltaphi, dt2, axis=(1, 2))

        return [out1, out2]

    def d2curlySc(self, c, r, m, c1, Deltar1, c2, Deltar2):

        ntheta = c.shape[0]
        res     = cp.empty([ntheta, self.nz, self.n], self.obj_dtype)
        c       = ascontig(c)
        c1      = ascontig(c1)
        c2      = ascontig(c2)
        r       = ascontig(r)
        Deltar1 = ascontig(Deltar1)
        Deltar2 = ascontig(Deltar2)
        m       = ascontig(m)

        self.launch(d2s_kernel, d2sf_kernel, ntheta,
                    (res, c, c1, c2, r, m, Deltar1, Deltar2,
                     self.n, self.npsi_eff, self.nz, self.nzpsi_eff, ntheta))
        return res
