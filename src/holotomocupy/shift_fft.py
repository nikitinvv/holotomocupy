"""FFT-based shift operator (assumes magnification m = 1).

Drop-in alternative for `holotomocupy.shift.Shift`: same public API
(`coeff`, `S`, `Sadj`, `curlyS`, `curlySc`, `dcurlySc`, `dcurlySadjc`,
`d2curlySc`, plus the `coeff_cached`/`_reset`/`_stats` triad), but every
shift is implemented via the Fourier shift theorem instead of cubic
B-spline interpolation.

Runs the shift directly on the input (nzpsi × npsi) grid — periodic BC, so
the input should vanish near the boundary or shifts should stay well inside
the grid; otherwise the periodic wrap shows up in the result.

The pad-offset convention `x_in = tx - rx + (npsi - n)/2` of the cubic
kernel is folded into the per-theta phase, so the output is the top-left
`[:nz, :n]` slice of the inverse FFT — no extra cropping arithmetic.

`coeff(psi)` is the identity: FFT shifts need no B-spline prefilter.
`Sback`/`curlySback`/`coeff_back` (used for the Paganin initial guess in
`rec_mpi`) are intentionally omitted; callers needing them should hold a
`Shift` instance alongside.
"""

import cupy as cp
import cupyx.scipy.fft as cufft


def ascontig(x):
    """cp.ascontiguousarray that also handles numpy/pinned inputs."""
    return cp.ascontiguousarray(cp.asarray(x))


# Fused elementwise kernels — each one is a single CUDA pass with no
# intermediate memory traffic, replacing chains of broadcast multiplies.
apply_sep_phase = cp.ElementwiseKernel(
    'complex64 c, complex64 py, complex64 px',
    'complex64 out',
    'out = c * py * px',
    'shift_fft_apply_sep_phase',
)

# imag(conj(a) * b) = a.real * b.imag - a.imag * b.real, written as a single
# pass over float32 lanes — avoids the big complex intermediate
# `cp.conj(a) * b` would otherwise materialize.
imag_conj_prod = cp.ElementwiseKernel(
    'complex64 a, complex64 b',
    'float32 out',
    'out = a.real() * b.imag() - a.imag() * b.real()',
    'shift_fft_imag_conj_prod',
)

# Combine the four big-array ops of dcurlySc into one pass:
#   out = (c * d + c1) * py * px
combine_dcurlySc = cp.ElementwiseKernel(
    'complex64 c, complex64 d, complex64 c1, complex64 py, complex64 px',
    'complex64 out',
    'out = (c * d + c1) * py * px',
    'shift_fft_combine_dcurlySc',
)

# Combine the d2curlySc terms into one pass:
#   out = (c * d1 * d2 + c1 * d2 + c2 * d1) * py * px
combine_d2curlySc = cp.ElementwiseKernel(
    'complex64 c, complex64 c1, complex64 c2, '
    'complex64 d1, complex64 d2, complex64 py, complex64 px',
    'complex64 out',
    'out = (c * d1 * d2 + c1 * d2 + c2 * d1) * py * px',
    'shift_fft_combine_d2curlySc',
)


class ShiftFFT():
    """Fourier-shift-theorem shift operator (assumes magnification m = 1).

    Runs the FFT shift directly on the input (nzpsi × npsi) grid — periodic
    BC, so the input should vanish near the boundary or shifts should stay
    well inside the grid; otherwise wrap-around artefacts appear.

    coeff() is the identity (no B-spline prefilter needed for FFT shifts).
    """

    def __init__(self, n, npsi, nz, nzpsi, obj_dtype, nchunk=None):
        self.n = n
        self.npsi = npsi
        self.nz = nz
        self.nzpsi = nzpsi
        self.obj_dtype = obj_dtype

        # cuFFT plan reuse for the batch.
        if nchunk is not None:
            tmp = cp.empty([nchunk, nzpsi, npsi], dtype='complex64')
            self.plan       = cufft.get_fft_plan(tmp, axes=(-2, -1), value_type='C2C')
            self.plan_batch = nchunk
            del tmp
        else:
            self.plan       = None
            self.plan_batch = None

        # 2π·k/N per axis on the input grid (fftfreq order: DC, +, -).
        # The output-region offset (nzpsi-nz)/2, (npsi-n)/2 is folded into
        # the per-call phase so the output is the top-left [:nz, :n] slice
        # of the ifft result.
        self.fy = (2 * cp.pi * cp.fft.fftfreq(nzpsi)).astype('float32')
        self.fx = (2 * cp.pi * cp.fft.fftfreq(npsi )).astype('float32')
        # Precomputed -i·fy and -i·fx as complex64 — used by deriv_factor
        # to skip the per-call (-1j)·real cast.
        self.negi_fy = ((-1j) * self.fy).astype('complex64')
        self.negi_fx = ((-1j) * self.fx).astype('complex64')
        self.eff_dy = (nzpsi - nz) * 0.5
        self.eff_dx = (npsi  - n ) * 0.5
        self.inv_N = cp.float32(1.0 / (nzpsi * npsi))

        # Match Shift's coeff cache surface so this class is drop-in.
        self.coeff_cache  = {}
        self.coeff_hits   = 0
        self.coeff_misses = 0

    # ------------------------------------------------------------------
    # FFT helpers
    # ------------------------------------------------------------------

    def fft2(self, x):
        if self.plan is not None and x.shape[0] == self.plan_batch:
            with self.plan:
                return cufft.fft2(x)
        return cp.fft.fft2(x)

    def ifft2(self, x):
        if self.plan is not None and x.shape[0] == self.plan_batch:
            with self.plan:
                return cufft.ifft2(x)
        return cp.fft.ifft2(x)

    def phase_separable(self, r):
        """Per-theta phase exp(-i (fy*(ry - eff_dy) + fx*(rx - eff_dx))) as
        separable factors py[ntheta, nzpsi] and px[ntheta, npsi]."""
        r = cp.asarray(r)
        ry_eff = (r[:, 0] - self.eff_dy).astype('float32')
        rx_eff = (r[:, 1] - self.eff_dx).astype('float32')
        py = cp.exp((-1j) * self.fy[cp.newaxis, :] * ry_eff[:, cp.newaxis]).astype('complex64')
        px = cp.exp((-1j) * self.fx[cp.newaxis, :] * rx_eff[:, cp.newaxis]).astype('complex64')
        return py, px

    def to_complex(self, x):
        return x if x.dtype == cp.complex64 else x.astype('complex64')

    def from_complex(self, x):
        # Adjoint of "cast real→complex" is "take real part"; preserves
        # exact adjointness in obj_dtype='float32' mode.
        # Both branches return a fresh contig array — important so the caller
        # doesn't hold the parent ifft buffer alive via a view.
        if self.obj_dtype == 'float32':
            return x.real.astype('float32')
        return cp.ascontiguousarray(x)

    def pad_output(self, g):
        """Place output g (nz, n) at the top-left of the FFT grid. Used by
        adjoint paths; phase folds in the eff_d offsets so the forward output
        appears at top-left."""
        gc = self.to_complex(ascontig(g))
        padded = cp.zeros([gc.shape[0], self.nzpsi, self.npsi], dtype='complex64')
        padded[:, :self.nz, :self.n] = gc
        return padded

    def deriv_factor(self, Delta):
        """D[t,y,x] = -i (fy[y]·Δy[t] + fx[x]·Δx[t]) on the FFT grid.
        Uses precomputed -i·fy / -i·fx; the sum is built in one expression
        so the full [ntheta, nzpsi, npsi] tensor materializes once
        (in-place += would fail because the (1, nzpsi, 1)+(1, 1, npsi)
        broadcast target shape doesn't match either operand's allocation)."""
        Delta = cp.asarray(Delta)
        Dy = Delta[:, 0].astype('complex64')      # [ntheta]
        Dx = Delta[:, 1].astype('complex64')
        return (self.negi_fy[cp.newaxis, :, cp.newaxis] * Dy[:, cp.newaxis, cp.newaxis]
              + self.negi_fx[cp.newaxis, cp.newaxis, :] * Dx[:, cp.newaxis, cp.newaxis])

    # ------------------------------------------------------------------
    # Coefficient-space cache (identity coeff, but API-compatible)
    # ------------------------------------------------------------------

    def coeff(self, psi):
        """FFT shift needs no B-spline prefilter — identity."""
        return psi

    def coeff_cached(self, psi):
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
        stats = (self.coeff_hits, self.coeff_misses)
        if reset:
            self.coeff_hits = 0
            self.coeff_misses = 0
        return stats

    # ------------------------------------------------------------------
    # Forward / adjoint shift  S / S*
    # ------------------------------------------------------------------

    def S(self, c, r, m):
        py, px = self.phase_separable(r)
        C = self.fft2(self.to_complex(ascontig(c)))
        # Fused: C = C · py · px (single elementwise kernel, in-place).
        apply_sep_phase(C, py[:, :, cp.newaxis], px[:, cp.newaxis, :], C)
        s = self.ifft2(C)
        del C
        return self.from_complex(s[:, :self.nz, :self.n])

    def Sadj(self, spsi, r, m):
        py, px = self.phase_separable(r)
        S = self.fft2(self.pad_output(spsi))
        # Fused: S = S · conj(py) · conj(px), in-place.
        apply_sep_phase(S, cp.conj(py)[:, :, cp.newaxis],
                           cp.conj(px)[:, cp.newaxis, :], S)
        return self.from_complex(self.ifft2(S))

    def curlyS(self, psi, r, m):
        return self.S(psi, r, m)   # coeff is identity in FFT mode

    # ------------------------------------------------------------------
    # Coefficient-space variants
    # ------------------------------------------------------------------

    def curlySc(self, c, r, m):
        return self.S(c, r, m)

    def dcurlySc(self, c, r, m, c1, Deltar):
        """∂S(c,r)/∂c · c1 + ∂S(c,r)/∂r · Δr."""
        py, px = self.phase_separable(r)
        C  = self.fft2(self.to_complex(ascontig(c)))
        C1 = self.fft2(self.to_complex(ascontig(c1)))
        D  = self.deriv_factor(Deltar)
        # Fused: out = (C·D + C1)·py·px, in-place into C, no intermediates.
        combine_dcurlySc(C, D, C1,
                         py[:, :, cp.newaxis], px[:, cp.newaxis, :], C)
        del D, C1
        s = self.ifft2(C)
        del C
        return self.from_complex(s[:, :self.nz, :self.n])

    def dcurlySadjc(self, c, r, m, Deltaphi):
        """Adjoint of (c1, Δr) → dcurlySc(c, r, m, c1, Δr) applied to Δφ.
        Returns [out1, out2] where out1 = Sadj(Δφ) and
        out2[t, 0/1] = redot(Δφ, ∂S/∂(ry/rx)·c)."""
        ntheta = c.shape[0]
        py, px = self.phase_separable(r)
        py_b = py[:, :, cp.newaxis]
        px_b = px[:, cp.newaxis, :]

        # fft(ZeroPad(Δφ)) on the internal grid — reused for out1 (Sadj) and
        # out2 (Parseval redots). Kept around past out1 since out2 needs it.
        Phat = self.fft2(self.pad_output(Deltaphi))

        # out1 = Sadj(Δφ). Copy Phat into Sbuf so the original survives.
        Sbuf = Phat.copy()
        apply_sep_phase(Sbuf, cp.conj(py_b), cp.conj(px_b), Sbuf)
        out1 = self.from_complex(self.ifft2(Sbuf))
        del Sbuf

        # Cshift = fft(c) · phase, in-place
        Cshift = self.fft2(self.to_complex(ascontig(c)))
        apply_sep_phase(Cshift, py_b, px_b, Cshift)

        # PhatC_im = Im(conj(Phat) · Cshift), via fused float-arithmetic
        # kernel — avoids the big complex intermediate cp.conj(Phat)*Cshift.
        PhatC_im = imag_conj_prod(Phat, Cshift)
        del Phat, Cshift

        # out2 via Parseval, separable sum-then-dot to avoid the huge
        # fy[None,:,None]*PhatC_im broadcast temp the naive form would build.
        #   out2[t,0] = inv_N · Σ_y fy[y] · Σ_x PhatC_im[t,y,x]
        #   out2[t,1] = inv_N · Σ_x fx[x] · Σ_y PhatC_im[t,y,x]
        sum_x = PhatC_im.sum(axis=2)   # [ntheta, nzpsi]
        sum_y = PhatC_im.sum(axis=1)   # [ntheta, npsi]
        del PhatC_im

        out2 = cp.empty([ntheta, 2], dtype='float32')
        out2[:, 0] = self.inv_N * (sum_x @ self.fy)
        out2[:, 1] = self.inv_N * (sum_y @ self.fx)
        return [out1, out2]

    def d2curlySc(self, c, r, m, c1, Deltar1, c2, Deltar2):
        """∂²S · ((c1, Δ1), (c2, Δ2)) =
            ∂²S/∂c²·(c1,c2) + ∂²S/∂c∂r·(c1,Δ2) + ∂²S/∂r∂c·(Δ1,c2) + ∂²S/∂r²·(Δ1,Δ2)
          = Crop(ifft((C·D1·D2 + C1·D2 + C2·D1)·P))    (∂²S/∂c² = 0)"""
        py, px = self.phase_separable(r)
        D1 = self.deriv_factor(Deltar1)
        D2 = self.deriv_factor(Deltar2)
        C  = self.fft2(self.to_complex(ascontig(c)))
        C1 = self.fft2(self.to_complex(ascontig(c1)))
        C2 = self.fft2(self.to_complex(ascontig(c2)))
        # Fused: out = (C·D1·D2 + C1·D2 + C2·D1)·py·px, in-place into C.
        combine_d2curlySc(C, C1, C2, D1, D2,
                          py[:, :, cp.newaxis], px[:, cp.newaxis, :], C)
        del D1, D2, C1, C2
        s = self.ifft2(C)
        del C
        return self.from_complex(s[:, :self.nz, :self.n])
