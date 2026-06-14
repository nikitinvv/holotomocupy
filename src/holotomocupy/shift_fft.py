"""FFT-based shift operator (assumes magnification m = 1).

Drop-in alternative for `holotomocupy.shift.Shift`: same public API
(`coeff`, `S`, `Sadj`, `curlyS`, `curlySc`, `dcurlySc`, `dcurlySadjc`,
`d2curlySc`, plus the `coeff_cached`/`_reset`/`_stats` triad), but every
shift is implemented via the Fourier shift theorem instead of cubic
B-spline interpolation.

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


def _ascontig(x):
    """Mirror of `shift._ascontig`: also handles numpy/pinned inputs."""
    return cp.ascontiguousarray(cp.asarray(x))


class ShiftFFT():
    """Fourier-shift-theorem shift operator (assumes magnification m = 1).

    Always uses symmetric (mirror-reflection) padding to double the FFT grid:
    the input is half-sample-reflected at each boundary into a (2·nzpsi,
    2·npsi) array, the shift happens on that larger grid, and the original
    region is recovered by fold-and-sum — the exact transpose of mirror-pad
    — so adjointness is preserved. Mirror reflection makes the signal
    smoothly periodic across the FFT boundary, so non-vanishing edges no
    longer wrap into the interior. Costs 4× in both memory and FFT work
    versus an un-padded shift.
    """

    def __init__(self, n, npsi, nz, nzpsi, obj_dtype, nchunk=None):
        self.n = n
        self.npsi = npsi
        self.nz = nz
        self.nzpsi = nzpsi
        self.obj_dtype = obj_dtype

        # Mirror-padded internal FFT grid: 2× the input grid on each axis,
        # input placed centered at offset (sy, sx) = (nzpsi // 2, npsi // 2).
        self._sy = nzpsi // 2
        self._sx = npsi  // 2
        self._nzpsi_int = nzpsi + 2 * self._sy
        self._npsi_int  = npsi  + 2 * self._sx

        # cuFFT plan reuse for the internal-grid batch.
        if nchunk is not None:
            _tmp = cp.empty([nchunk, self._nzpsi_int, self._npsi_int], dtype='complex64')
            self._plan       = cufft.get_fft_plan(_tmp, axes=(-2, -1), value_type='C2C')
            self._plan_batch = nchunk
            del _tmp
        else:
            self._plan       = None
            self._plan_batch = None

        # 2π·k/N per axis on the internal grid, in fftfreq order (DC, +, -).
        # The output-region offset (sy + (nzpsi-nz)/2, sx + (npsi-n)/2) is
        # folded into the per-call phase so the output is the top-left
        # [:nz, :n] slice of the (padded) ifft result.
        self._fy = (2 * cp.pi * cp.fft.fftfreq(self._nzpsi_int)).astype('float32')
        self._fx = (2 * cp.pi * cp.fft.fftfreq(self._npsi_int )).astype('float32')
        self._eff_dy = self._sy + (nzpsi - nz) * 0.5
        self._eff_dx = self._sx + (npsi  - n ) * 0.5

        # Match Shift's coeff cache surface so this class is drop-in.
        self._coeff_cache  = {}
        self._coeff_hits   = 0
        self._coeff_misses = 0

    # ------------------------------------------------------------------
    # FFT helpers
    # ------------------------------------------------------------------

    def _fft_big(self, x):
        if self._plan is not None and x.shape[0] == self._plan_batch:
            with self._plan:
                return cufft.fft2(x)
        return cp.fft.fft2(x)

    def _ifft_big(self, x):
        if self._plan is not None and x.shape[0] == self._plan_batch:
            with self._plan:
                return cufft.ifft2(x)
        return cp.fft.ifft2(x)

    def _phase_separable(self, r):
        """Per-theta phase exp(-i (fy*(ry - eff_dy) + fx*(rx - eff_dx))) as
        separable factors py[ntheta, nzpsi_int] and px[ntheta, npsi_int].
        Broadcast-multiplied so the full [ntheta, nzpsi_int, npsi_int]
        phase tensor is never materialized."""
        r = cp.asarray(r)
        ry_eff = (r[:, 0] - self._eff_dy).astype('float32')
        rx_eff = (r[:, 1] - self._eff_dx).astype('float32')
        py = cp.exp((-1j) * self._fy[cp.newaxis, :] * ry_eff[:, cp.newaxis]).astype('complex64')
        px = cp.exp((-1j) * self._fx[cp.newaxis, :] * rx_eff[:, cp.newaxis]).astype('complex64')
        return py, px

    def _to_complex(self, x):
        return x if x.dtype == cp.complex64 else x.astype('complex64')

    def _from_complex(self, x):
        # Adjoint of "cast real→complex" is "take real part"; preserves
        # exact adjointness in obj_dtype='float32' mode.
        return x.real.astype('float32') if self.obj_dtype == 'float32' else x

    def _pad_input(self, c):
        """Whole-sample mirror reflection of c into the internal FFT grid:
        c[-k] = c[k] (axis through index 0), matching the cubic kernel's
        sym_idx convention. Equivalent to numpy.pad(mode='reflect').
        Requires nzpsi, npsi ≥ 2 (always true in practice)."""
        cc = self._to_complex(_ascontig(c))
        return cp.pad(cc, ((0, 0), (self._sy, self._sy), (self._sx, self._sx)),
                      mode='reflect')

    def _crop_input(self, padded):
        """Adjoint of _pad_input (fold-and-sum). With 'reflect' BC, each
        c[i] in the central region appears at:
          - padded[sy + i]              (always)
          - padded[sy - i]              if 1 ≤ i ≤ sy        (left mirror)
          - padded[sy + 2N - 2 - i]     if N-1-sy ≤ i ≤ N-2  (right mirror)
        i=0 and i=N-1 receive only the central contribution (they ARE the
        reflection axes, not reflected). 2D adjoint = separable 1D folds."""
        sy, sx       = self._sy, self._sx
        nzpsi, npsi  = self.nzpsi, self.npsi

        # Fold y-axis: take central rows, add reversed left/right bands.
        out_y = padded[:, sy:sy + nzpsi, :].copy()
        # left mirror lands on central rows 1..sy (NOT row 0)
        out_y[:, 1:sy + 1, :]              += padded[:, :sy, :][:, ::-1, :]
        # right mirror lands on central rows (N-1-sy)..(N-2) (NOT row N-1)
        out_y[:, nzpsi - 1 - sy:nzpsi - 1, :] += padded[:, sy + nzpsi:sy + nzpsi + sy, :][:, ::-1, :]

        # Fold x-axis on out_y.
        out = out_y[:, :, sx:sx + npsi].copy()
        out[:, :, 1:sx + 1]                += out_y[:, :, :sx][:, :, ::-1]
        out[:, :, npsi - 1 - sx:npsi - 1]  += out_y[:, :, sx + npsi:sx + npsi + sx][:, :, ::-1]
        return out

    def _pad_output(self, g):
        """Place output g (nz, n) at the top-left of the internal grid.
        Used by adjoint paths; phase folds in the eff_d offsets so the
        forward output appears at top-left."""
        gc = self._to_complex(_ascontig(g))
        padded = cp.zeros([gc.shape[0], self._nzpsi_int, self._npsi_int], dtype='complex64')
        padded[:, :self.nz, :self.n] = gc
        return padded

    def _deriv_factor(self, Delta):
        """D[t,y,x] = -i (fy[y]*Δy[t] + fx[x]*Δx[t]) on the internal grid."""
        Delta = cp.asarray(Delta)
        Dy = Delta[:, 0].astype('float32')
        Dx = Delta[:, 1].astype('float32')
        D = (self._fy[cp.newaxis, :, cp.newaxis] * Dy[:, cp.newaxis, cp.newaxis]
             + self._fx[cp.newaxis, cp.newaxis, :] * Dx[:, cp.newaxis, cp.newaxis])
        return ((-1j) * D).astype('complex64')

    # ------------------------------------------------------------------
    # Coefficient-space cache (identity coeff, but API-compatible)
    # ------------------------------------------------------------------

    def coeff(self, psi):
        """FFT shift needs no B-spline prefilter — identity."""
        return psi

    def coeff_cached(self, psi):
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
        stats = (self._coeff_hits, self._coeff_misses)
        if reset:
            self._coeff_hits = 0
            self._coeff_misses = 0
        return stats

    # ------------------------------------------------------------------
    # Forward / adjoint shift  S / S*
    # ------------------------------------------------------------------

    def S(self, c, r, m):
        py, px = self._phase_separable(r)
        C = self._fft_big(self._pad_input(c))
        C *= py[:, :, cp.newaxis]
        C *= px[:, cp.newaxis, :]
        s = self._ifft_big(C)
        return self._from_complex(s[:, :self.nz, :self.n])

    def Sadj(self, spsi, r, m):
        py, px = self._phase_separable(r)
        S = self._fft_big(self._pad_output(spsi))
        S *= cp.conj(py)[:, :, cp.newaxis]
        S *= cp.conj(px)[:, cp.newaxis, :]
        return self._from_complex(self._crop_input(self._ifft_big(S)))

    def curlyS(self, psi, r, m):
        return self.S(psi, r, m)   # coeff is identity in FFT mode

    # ------------------------------------------------------------------
    # Coefficient-space variants
    # ------------------------------------------------------------------

    def curlySc(self, c, r, m):
        return self.S(c, r, m)

    def dcurlySc(self, c, r, m, c1, Deltar):
        """∂S(c,r)/∂c · c1 + ∂S(c,r)/∂r · Δr."""
        py, px = self._phase_separable(r)
        C  = self._fft_big(self._pad_input(c))
        C1 = self._fft_big(self._pad_input(c1))
        D  = self._deriv_factor(Deltar)
        combined = C * D + C1
        combined *= py[:, :, cp.newaxis]
        combined *= px[:, cp.newaxis, :]
        s = self._ifft_big(combined)
        return self._from_complex(s[:, :self.nz, :self.n])

    def dcurlySadjc(self, c, r, m, Deltaphi):
        """Adjoint of (c1, Δr) → dcurlySc(c, r, m, c1, Δr) applied to Δφ.
        Returns [out1, out2] where out1 = Sadj(Δφ) and
        out2[t, 0/1] = redot(Δφ, ∂S/∂(ry/rx)·c)."""
        ntheta = c.shape[0]
        py, px = self._phase_separable(r)
        py_b = py[:, :, cp.newaxis]
        px_b = px[:, cp.newaxis, :]

        # fft(ZeroPad(Δφ)) on the internal grid — reused for out1 (Sadj) and
        # out2 (Parseval redots).
        Phat = self._fft_big(self._pad_output(Deltaphi))

        # out1 = Sadj(Δφ)
        tmp = Phat * cp.conj(py_b)
        tmp *= cp.conj(px_b)
        out1 = self._from_complex(self._crop_input(self._ifft_big(tmp)))

        # out2 via Parseval:
        #   redot(Δφ, ∂S/∂r_k · c) = Re<ZeroPad(Δφ), ifft(fft(c)*P*(-i f_k))>
        #                          = (1/N_int) Σ f_k Im(conj(Phat) · C·P)
        Cshift = self._fft_big(self._pad_input(c))
        Cshift *= py_b
        Cshift *= px_b
        PhatC_im = cp.imag(cp.conj(Phat) * Cshift)
        inv_N = cp.float32(1.0 / (self._nzpsi_int * self._npsi_int))

        out2 = cp.empty([ntheta, 2], dtype='float32')
        out2[:, 0] = inv_N * cp.sum(self._fy[cp.newaxis, :, cp.newaxis] * PhatC_im, axis=(1, 2))
        out2[:, 1] = inv_N * cp.sum(self._fx[cp.newaxis, cp.newaxis, :] * PhatC_im, axis=(1, 2))
        return [out1, out2]

    def d2curlySc(self, c, r, m, c1, Deltar1, c2, Deltar2):
        """∂²S · ((c1, Δ1), (c2, Δ2)) =
            ∂²S/∂c²·(c1,c2) + ∂²S/∂c∂r·(c1,Δ2) + ∂²S/∂r∂c·(Δ1,c2) + ∂²S/∂r²·(Δ1,Δ2)
          = Crop(ifft((C·D1·D2 + C1·D2 + C2·D1)·P))    (∂²S/∂c² = 0)"""
        py, px = self._phase_separable(r)
        D1 = self._deriv_factor(Deltar1)
        D2 = self._deriv_factor(Deltar2)
        C  = self._fft_big(self._pad_input(c))
        C1 = self._fft_big(self._pad_input(c1))
        C2 = self._fft_big(self._pad_input(c2))
        combined = C * (D1 * D2) + C1 * D2 + C2 * D1
        combined *= py[:, :, cp.newaxis]
        combined *= px[:, cp.newaxis, :]
        s = self._ifft_big(combined)
        return self._from_complex(s[:, :self.nz, :self.n])
