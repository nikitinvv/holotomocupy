import math
import numpy as np
import cupy as cp
import cupyx.scipy.fft as cufft
from .cuda_kernels import (
    s_kernel, sf_kernel,
    sback_kernel,
    ds_kernel, dsf_kernel,
    dsm_kernel, dsmf_kernel,
    d2s_kernel, d2sf_kernel,
    d2sm_kernel, d2smf_kernel,
    dsadj_kernel, dsadjf_kernel,
    dsmadj_kernel, dsmadjf_kernel,
)
from .utils import redot, CoeffCacheMixin


def ascontig(x):
    """cp.ascontiguousarray that also accepts numpy/pinned input (auto-uploads).
    Newer cupy rejects numpy inputs to ascontiguousarray; we go through cp.asarray
    first so callers can pass pinned-numpy slices (e.g. vars['pos'][:, k]) directly."""
    return cp.ascontiguousarray(cp.asarray(x))


class Shift(CoeffCacheMixin):
    """Cubic B-spline shift operator (requires coeff() prefilter, C2 smooth).

    coeff() is the FFT B-spline prefilter on the input grid (periodic BC).
    Shape-preserving and self-adjoint.
    """

    def __init__(self, n, npsi, nz, nzpsi, obj_dtype, nchunk=None):
        self.n = n
        self.npsi = npsi
        self.nz = nz
        self.nzpsi = nzpsi
        self.obj_dtype = obj_dtype

        # Forward B-spline denominator on the input grid (unit magnification, k=0,1).
        x = cp.linspace(-1/2, 1/2 - 1/npsi,  npsi ).astype('float32')
        y = cp.linspace(-1/2, 1/2 - 1/nzpsi, nzpsi).astype('float32')
        divx = (self.phi(0) + 2 * self.phi(1) * cp.cos(2 * cp.pi * x)).astype('float32')
        divy = (self.phi(0) + 2 * self.phi(1) * cp.cos(2 * cp.pi * y)).astype('float32')
        self.ifB3 = 1 / cp.fft.fftshift(cp.outer(divy, divx), axes=(-1, -2))

        if nchunk is not None:
            tmp = cp.empty([nchunk, nzpsi, npsi], dtype='complex64')
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
        """FFT B-spline prefilter; x has last two dims (nzpsi, npsi)."""
        if (self.plan_coeff is not None and x.ndim == 3
                and x.shape[0] == self.plan_coeff_batch):
            with self.plan_coeff:
                return cufft.ifft2(cufft.fft2(x) * self.ifB3)
        return cp.fft.ifft2(cp.fft.fft2(x) * self.ifB3)

    def coeff(self, psi):
        """B-spline prefilter — same-shape FFT (periodic BC), self-adjoint."""
        out = self.prefilter(psi)
        if self.obj_dtype == 'float32':
            out = out.real
        return out

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
                     self.n, self.npsi, self.nz, self.nzpsi, ntheta, 0))
        return spsi

    def Sadj(self, spsi, r, m):
        ntheta = spsi.shape[0]
        c = cp.zeros([ntheta, self.nzpsi, self.npsi], dtype=self.obj_dtype)
        spsi = ascontig(spsi)
        r = ascontig(r)
        m = ascontig(m)
        self.launch(s_kernel, sf_kernel, ntheta,
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
        c    = ascontig(c)
        r    = ascontig(r)
        m    = ascontig(m)
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
                     self.n, self.npsi, self.nz, self.nzpsi, ntheta, 0))
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
                     self.n, self.npsi, self.nz, self.nzpsi, ntheta))

        return res

    def dcurlySmc(self, c, r, m, c1, Deltar, Deltam):
        """Single-pass (c, r, m) directional derivative:
            curlySc(c1, r, m)
          + d/dr curlySc(c, r, m) * Deltar
          + d/dm curlySc(c, r, m) * Deltam

        Same signature as dcurlySc with an extra Deltam (chunk, 2). Uses
        the identity d/dm_axis = -tau_axis(pixel) * d/dr_axis to fold Deltam
        into a per-pixel effective r-direction Deltar - tau * Deltam inside
        the CUDA kernel, so this is exactly one kernel launch (same cost as
        dcurlySc) instead of dcurlySc + two extra weighted grad-r calls.
        """
        ntheta = c.shape[0]
        res     = cp.empty([ntheta, self.nz, self.n], self.obj_dtype)
        c       = ascontig(c)
        c1      = ascontig(c1)
        r       = ascontig(r)
        Deltar  = ascontig(Deltar)
        Deltam  = ascontig(Deltam)
        m       = ascontig(m)

        self.launch(dsm_kernel, dsmf_kernel, ntheta,
                    (res, c, c1, r, m, Deltar, Deltam,
                     self.n, self.npsi, self.nz, self.nzpsi, ntheta))

        return res

    def dcurlySadjc(self, c, r, m, Deltaphi):

        ntheta = c.shape[0]
        # out1 is an atomicAdd target -> MUST be zeroed. dt1/dt2 are written every
        # position by the kernel; out2 is overwritten by the redot() lines below.
        out1 = cp.zeros([ntheta, self.nzpsi, self.npsi], dtype=self.obj_dtype)
        out2 = cp.empty(r.shape, dtype='float32')
        dt1  = cp.empty(Deltaphi.shape, self.obj_dtype)
        dt2  = cp.empty(Deltaphi.shape, self.obj_dtype)
        c        = ascontig(c)
        r        = ascontig(r)
        Deltaphi = ascontig(Deltaphi)
        m        = ascontig(m)

        self.launch(dsadj_kernel, dsadjf_kernel, ntheta,
                    (out1, dt1, dt2, c, Deltaphi, r, m,
                     self.n, self.npsi, self.nz, self.nzpsi, ntheta))

        out2[:, 0] = redot(Deltaphi, dt1, axis=(1, 2))
        out2[:, 1] = redot(Deltaphi, dt2, axis=(1, 2))

        return [out1, out2]

    def dcurlySadjmc(self, c, r, m, Deltaphi):
        """Adjoint of dcurlySmc. Returns [out1, out2_r, out2_m] such that
            <dcurlySmc(c, r, m, c1, Deltar, Deltam), g>
              = <c1, out1> + <Deltar, out2_r> + <Deltam, out2_m>
        for the standard inner products (complex for c1, real for the r/m
        direction vectors).

        dsmadj_kernel writes four per-pixel fields:
            dt1  = ∂curlySc/∂r_y,  dt2  = ∂curlySc/∂r_x
            dtm1 = ∂curlySc/∂m_y,  dtm2 = ∂curlySc/∂m_x
        so all four adjoint reductions become a plain redot with Deltaphi —
        no big broadcast multiply of a (chunk, nz, n) array by a τ vector.
        """
        ntheta = c.shape[0]
        out1   = cp.zeros([ntheta, self.nzpsi, self.npsi], dtype=self.obj_dtype)
        out2_r = cp.empty(r.shape, dtype='float32')
        out2_m = cp.empty(r.shape, dtype='float32')
        dt1    = cp.empty(Deltaphi.shape, self.obj_dtype)
        dt2    = cp.empty(Deltaphi.shape, self.obj_dtype)
        dtm1   = cp.empty(Deltaphi.shape, self.obj_dtype)
        dtm2   = cp.empty(Deltaphi.shape, self.obj_dtype)
        c        = ascontig(c)
        r        = ascontig(r)
        Deltaphi = ascontig(Deltaphi)
        m        = ascontig(m)

        self.launch(dsmadj_kernel, dsmadjf_kernel, ntheta,
                    (out1, dt1, dt2, dtm1, dtm2, c, Deltaphi, r, m,
                     self.n, self.npsi, self.nz, self.nzpsi, ntheta))

        out2_r[:, 0] = redot(Deltaphi, dt1,  axis=(1, 2))
        out2_r[:, 1] = redot(Deltaphi, dt2,  axis=(1, 2))
        out2_m[:, 0] = redot(Deltaphi, dtm1, axis=(1, 2))
        out2_m[:, 1] = redot(Deltaphi, dtm2, axis=(1, 2))

        return [out1, out2_r, out2_m]

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
                     self.n, self.npsi, self.nz, self.nzpsi, ntheta))
        return res

    def d2curlySmc(self, c, r, m, c1, Deltar1, Deltam1, c2, Deltar2, Deltam2):
        """Single-pass 2nd directional derivative on (c, r, m).

        Bilinear form on directions (c1, Δr1, Δm1) and (c2, Δr2, Δm2).
        Returns (with c-linearity ∂²/∂c² = 0):
            ∂²/∂r²   · Δr1·Δr2
          + ∂²/∂m²   · Δm1·Δm2
          + ∂²/∂r∂m  · (Δr1·Δm2 + Δr2·Δm1)
          + ∂²/∂c∂r  · (c1·Δr2 + c2·Δr1)
          + ∂²/∂c∂m  · (c1·Δm2 + c2·Δm1)

        Uses d/dm_axis = -tau_axis(pixel) * d/dr_axis to fold Δm into a
        per-pixel effective Δr = Δr − tau · Δm for BOTH direction slots
        inside the CUDA kernel — one kernel launch, same cost as d2curlySc.
        """
        ntheta = c.shape[0]
        res     = cp.empty([ntheta, self.nz, self.n], self.obj_dtype)
        c       = ascontig(c)
        c1      = ascontig(c1)
        c2      = ascontig(c2)
        r       = ascontig(r)
        Deltar1 = ascontig(Deltar1)
        Deltam1 = ascontig(Deltam1)
        Deltar2 = ascontig(Deltar2)
        Deltam2 = ascontig(Deltam2)
        m       = ascontig(m)

        self.launch(d2sm_kernel, d2smf_kernel, ntheta,
                    (res, c, c1, c2, r, m, Deltar1, Deltam1, Deltar2, Deltam2,
                     self.n, self.npsi, self.nz, self.nzpsi, ntheta))
        return res
