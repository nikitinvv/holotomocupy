"""FFT-based shift operator with chirp-z (Bluestein) magnification.

Drop-in alternative for `holotomocupy.shift.Shift`: same public API
(`coeff`, `S`, `Sadj`, `curlyS`, `curlySc`, `dcurlySc`, `dcurlySadjc`,
`d2curlySc`, plus the `coeff_cached`/`_reset`/`_stats` triad), but every
shift is implemented via the Fourier shift theorem (m=1) or a separable
chirp-z transform (arbitrary per-projection m).

When all `m[t] == 1`, `S` / `Sadj` take the fast path: a single batched
2-D FFT pair on the input grid with a separable linear-phase multiply.
When any `m[t] != 1`, the chirp-z path runs per axis — one Bluestein
correlation each, evaluating the Fourier interpolant at the m-scaled,
r-shifted output positions. The chirp-z path matches the `s_kernel`
convention `x_in = m·(tx - (n-1)/2) - r_x + (npsi-1)/2`.

The Bluestein convolution length L is the next power of two of
`N_in + N_out - 1`, so chirp-z costs ≈ 3 batched FFTs of length L per
axis (vs 1 of length N for the m=1 fast path).

Runs the shift directly on the input (nzpsi × npsi) grid — periodic BC,
so the input should vanish near the boundary or shifts should stay well
inside the grid; otherwise the periodic wrap shows up in the result.

`coeff(psi)` is the identity: FFT shifts need no B-spline prefilter.
The derivative methods (`dcurlySc`, `dcurlySadjc`, `d2curlySc`) also
dispatch automatically: at `m=1` they use the existing fast-path
algebra; at `m≠1` they exploit the linearity of `S` in its input —
e.g. `dcurlySc = S(c1 + Δry·∂c/∂y + Δrx·∂c/∂x, r, m)` with the
spatial derivatives computed via FFT differentiation — then call
the same `S`/`Sadj` chirp-z path used by the forward operator.

`Sback`/`curlySback`/`coeff_back` (used for the Paganin initial guess in
`rec_mpi`) are intentionally omitted; callers needing them should hold a
`Shift` instance alongside.
"""

import cupy as cp
import cupyx.scipy.fft as cufft
import numpy as np
from .utils import redot


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

        # ---- Chirp-z (Bluestein) precomputations -----------------------
        # Used when any m[t] != 1. Per axis:
        #   κ        — signed fftfreq integer    (length N_in)
        #   κ²       — for the pre-chirp
        #   ty²      — for the post-chirp        (length N_out)
        #   (j-N//2)² — for the kernel h         (length N_in + N_out - 1)
        # L is the Bluestein FFT length, next power of two of N_in+N_out-1.
        self.L_x = self._next_pow2(npsi + n  - 1)
        self.L_y = self._next_pow2(nzpsi + nz - 1)
        self.k_signed_x = (cp.fft.fftfreq(npsi ) * cp.float32(npsi )).astype('float32')
        self.k_signed_y = (cp.fft.fftfreq(nzpsi) * cp.float32(nzpsi)).astype('float32')
        self.k_sq_x  = (self.k_signed_x ** 2).astype('float32')
        self.k_sq_y  = (self.k_signed_y ** 2).astype('float32')
        self.ty_sq_x = (cp.arange(n , dtype='float32') ** 2).astype('float32')
        self.ty_sq_y = (cp.arange(nz, dtype='float32') ** 2).astype('float32')
        self.j_sq_x  = ((cp.arange(npsi  + n  - 1, dtype='float32') - npsi  // 2) ** 2).astype('float32')
        self.j_sq_y  = ((cp.arange(nzpsi + nz - 1, dtype='float32') - nzpsi // 2) ** 2).astype('float32')

        # Match Shift's coeff cache surface so this class is drop-in.
        self.coeff_cache  = {}
        self.coeff_hits   = 0
        self.coeff_misses = 0

    @staticmethod
    def _next_pow2(n):
        p = 1
        while p < n:
            p <<= 1
        return p

    @staticmethod
    def _is_unit_mag(m):
        """All entries of m equal to 1 → fast (FFT-shift) path; else chirp-z.

        m is shape (ntheta, 2) — axis 1 is (my, mx).
        """
        m_arr = cp.asarray(m)
        # Tolerant comparison so float32 1.0 hits the fast path even if the
        # caller built m as numpy float64.
        return bool(cp.all(cp.abs(m_arr - 1) < 1e-7))

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
    # Chirp-z (Bluestein) magnification — separable per axis
    # ------------------------------------------------------------------
    # All four helpers below evaluate / adjoint-evaluate the band-limited
    # Fourier interpolant of the input grid at the m-scaled, r-shifted
    # sample positions
    #
    #     y_{t,ty} = m[t] · ty + b[t],     ty = 0 … N_out − 1
    #
    # which, after folding the s_kernel center convention into b[t],
    # equals  m[t]·(ty − (N_out−1)/2) − r[t] + (N_in−1)/2.
    #
    # Math:
    #   g[t,ty] = (1/N_in) Σ_κ C[t,κ] · exp(2πi κ (m·ty + b)/N_in)
    #   Identity 2κ·ty = (κ+ty)² − κ² − ty² gives, with β = m/N_in:
    #     g[t,ty] = (1/N_in) e^(−iπ β ty²) · Σ_κ [C[t,κ] · e^(2πi κ b/N_in)
    #                                              · e^(−iπ β κ²)] · e^(iπ β (κ+ty)²)
    #
    # The inner sum is a cross-correlation with the chirp kernel
    # h[j] = exp(iπ β j²), j = (κ+ty), evaluated via FFT-convolution of
    # length L = next_pow2(N_in + N_out − 1).

    def _chirpz_lastaxis(self, X, m, b, axis_xy, adjoint):
        """Chirp-z transform along the last axis.

        X:        complex64 array, shape [B, K, N_in] for adjoint=False
                  or [B, K, N_out] for adjoint=True (B = ntheta, K may be 1).
        m:        [B] float32 — per-projection magnification.
        b:        [B] float32 — per-projection linear shift offset
                  (= (N_in−1)/2 − r − m·(N_out−1)/2).
        axis_xy:  'x' or 'y' — picks which precomputed index arrays to use.
        adjoint:  True for the L²-adjoint of the forward chirp-z.

        Returns [B, K, N_out] (forward) or [B, K, N_in] (adjoint).
        """
        if axis_xy == 'x':
            N_in, N_out, L = self.npsi,  self.n,  self.L_x
            k_signed = self.k_signed_x
            k_sq, ty_sq, j_sq = self.k_sq_x, self.ty_sq_x, self.j_sq_x
        else:
            N_in, N_out, L = self.nzpsi, self.nz, self.L_y
            k_signed = self.k_signed_y
            k_sq, ty_sq, j_sq = self.k_sq_y, self.ty_sq_y, self.j_sq_y

        B = X.shape[0]
        K = X.shape[1]
        beta = (cp.asarray(m).astype('float32') / cp.float32(N_in))   # [B]
        b    = cp.asarray(b).astype('float32')                        # [B]

        # Pre-twist (fftfreq order):
        #   pre[t,k] = exp(2πi κ b/N_in − iπ β κ²)
        # combined into one elementwise exp.
        phase_pre = ((2.0 * cp.pi / N_in) * b[:, None] * k_signed[None, :]
                     - cp.pi * beta[:, None] * k_sq[None, :]).astype('float32')   # [B, N_in]
        pre_twist = cp.exp(1j * phase_pre).astype('complex64')                    # [B, N_in]

        # Post-chirp at output sample positions ty = 0 … N_out − 1:
        #   post[t,ty] = exp(−iπ β ty²)
        post = cp.exp(-1j * cp.pi * beta[:, None] * ty_sq[None, :]).astype('complex64')  # [B, N_out]

        # Bluestein kernel h on physical index j_phys = idx − N_in//2:
        #   h[t,idx] = exp(+iπ β j_phys²),  idx = 0 … N_in+N_out−2
        # zero-padded to length L for circular FFT-correlation.
        h = cp.exp(1j * cp.pi * beta[:, None] * j_sq[None, :]).astype('complex64')   # [B, J]
        h_pad = cp.zeros((B, L), dtype='complex64')
        h_pad[:, : N_in + N_out - 1] = h
        H_hat = cp.fft.fft(h_pad, axis=-1)                                            # [B, L]

        if not adjoint:
            # Forward: X[B,K,N_in] → g[B,K,N_out]
            # 1) FFT along last axis.
            C = cp.fft.fft(X, axis=-1)                                                # [B, K, N_in]
            # 2) Pre-twist (shift phase + pre-chirp).
            a = C * pre_twist[:, None, :]
            # 3) fftshift along last axis so κ = idx − N_in//2 in centered storage.
            a = cp.fft.fftshift(a, axes=-1)
            # 4) Zero-pad to L.
            a_pad = cp.zeros((B, K, L), dtype='complex64')
            a_pad[:, :, :N_in] = a
            # 5) Compute Σ_n a_pad[n] · h_pad[n+ty] for ty=0..L−1 via FFT.
            #    Derivation (with ω = exp(2πi/L)):
            #      h[m] = (1/L) Σ_j H_hat[j] ω^(jm)  ⇒
            #      Σ_n a_pad[n] · h[n+ty]
            #         = (1/L) Σ_j H_hat[j] · ω^(jty) · Σ_n a_pad[n] ω^(jn)
            #         = ifft( H_hat · A_pos )[ty],  A_pos[j] = Σ_n a_pad[n] ω^(jn).
            #    A_pos is cupy's IFFT with norm='forward' (no 1/L, +sign).
            #    NOTE: the apparent "ifft(conj(FFT(a))·FFT(h))" identity is
            #    only correct for REAL a — for complex a the conj sticks on
            #    the data instead of doing nothing.
            A_pos = cp.fft.ifft(a_pad, axis=-1, norm='forward')
            corr  = cp.fft.ifft(A_pos * H_hat[:, None, :], axis=-1)                   # [B, K, L]
            # 6) Extract first N_out, post-multiply, scale by 1/N_in.
            g = corr[:, :, :N_out] * post[:, None, :] * cp.float32(1.0 / N_in)
            return g

        # Adjoint: g[B,K,N_out] → X[B,K,N_in]
        # Apply the L²-adjoint of each forward step in reverse order.
        # cupy adjoints (sum-of-conj·· inner product):
        #     (fft)*    = ifft(·, norm='forward'),
        #     (ifft)*   = fft(·,  norm='forward'),
        #     (ifft_fwd)* = fft(·)  (since ifft_fwd = N·ifft, adjoint cancels N).
        g = X
        # 8*) g · conj(post) · (1/N_in)
        scaled = g * cp.conj(post)[:, None, :] * cp.float32(1.0 / N_in)               # [B, K, N_out]
        # 7*) Zero-pad to length L (place at the front).
        corr_adj = cp.zeros((B, K, L), dtype='complex64')
        corr_adj[:, :, :N_out] = scaled
        # 6*) Adjoint of  corr = ifft(A_pos · H_hat) :
        #     A_pos_adj = conj(H_hat) · fft(corr_adj, norm='forward')
        A_pos_adj = (cp.conj(H_hat)[:, None, :]
                     * cp.fft.fft(corr_adj, axis=-1, norm='forward'))
        # 5*) Adjoint of  A_pos = ifft(a_pad, norm='forward'):
        a_pad_adj = cp.fft.fft(A_pos_adj, axis=-1)
        # 4*) Adjoint of zero-pad: truncate to N_in.
        a_centered_adj = a_pad_adj[:, :, :N_in]                                       # [B, K, N_in]
        # 3*) ifftshift (adjoint of fftshift).
        a_adj = cp.fft.ifftshift(a_centered_adj, axes=-1)
        # 2*) Adjoint of multiply by twist: multiply by conj(twist).
        C_adj = a_adj * cp.conj(pre_twist)[:, None, :]
        # 1*) Adjoint of FFT = ifft with norm='forward'.
        X_adj = cp.fft.ifft(C_adj, axis=-1, norm='forward')
        return X_adj

    def _chirpz_2d(self, X, m, ry, rx, adjoint):
        """Separable 2-D chirp-z. Calls the last-axis helper twice — once
        along x, once along y — with appropriate axis swapping to keep
        FFTs on the contiguous last axis.

        m is (ntheta, 2) — axis 1 is (my, mx). Each axis's chirp-z receives
        its own magnification.
        """
        m  = cp.asarray(m).astype('float32')
        my = cp.ascontiguousarray(m[:, 0])
        mx = cp.ascontiguousarray(m[:, 1])
        ry = cp.asarray(ry).astype('float32')
        rx = cp.asarray(rx).astype('float32')
        b_x = (cp.float32((self.npsi  - 1) * 0.5) - rx - mx * cp.float32((self.n  - 1) * 0.5))
        b_y = (cp.float32((self.nzpsi - 1) * 0.5) - ry - my * cp.float32((self.nz - 1) * 0.5))

        if not adjoint:
            # Forward order: x first (last axis), then y (last axis after swap).
            X = self.to_complex(ascontig(X))                       # [B, nzpsi, npsi]
            Y = self._chirpz_lastaxis(X, mx, b_x, 'x', adjoint=False)   # [B, nzpsi, n]
            Y = cp.ascontiguousarray(cp.swapaxes(Y, -2, -1))            # [B, n, nzpsi]
            Y = self._chirpz_lastaxis(Y, my, b_y, 'y', adjoint=False)   # [B, n, nz]
            Y = cp.ascontiguousarray(cp.swapaxes(Y, -2, -1))            # [B, nz, n]
            return Y

        # Adjoint: reverse the order — y_adj first (on a swapped view),
        # then x_adj.
        X = self.to_complex(ascontig(X))                                # [B, nz, n]
        Y = cp.ascontiguousarray(cp.swapaxes(X, -2, -1))                # [B, n, nz]
        Y = self._chirpz_lastaxis(Y, my, b_y, 'y', adjoint=True)         # [B, n, nzpsi]
        Y = cp.ascontiguousarray(cp.swapaxes(Y, -2, -1))                # [B, nzpsi, n]
        Y = self._chirpz_lastaxis(Y, mx, b_x, 'x', adjoint=True)         # [B, nzpsi, npsi]
        return Y

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
        if not self._is_unit_mag(m):
            # Chirp-z magnification path.
            r = cp.asarray(r)
            out = self._chirpz_2d(c, m, r[:, 0], r[:, 1], adjoint=False)
            return self.from_complex(out)
        py, px = self.phase_separable(r)
        C = self.fft2(self.to_complex(ascontig(c)))
        # Fused: C = C · py · px (single elementwise kernel, in-place).
        apply_sep_phase(C, py[:, :, cp.newaxis], px[:, cp.newaxis, :], C)
        s = self.ifft2(C)
        del C
        return self.from_complex(s[:, :self.nz, :self.n])

    def Sadj(self, spsi, r, m):
        if not self._is_unit_mag(m):
            # Chirp-z magnification path (adjoint).
            r = cp.asarray(r)
            out = self._chirpz_2d(spsi, m, r[:, 0], r[:, 1], adjoint=True)
            return self.from_complex(out)
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
        if not self._is_unit_mag(m):
            # Chirp-z magnification path. By linearity of S in its input,
            #   dcurlySc = S(c1 + Δry·∂c/∂y + Δrx·∂c/∂x, r, m)
            # and the spatial derivatives ∂c/∂(y,x) are computed via FFT
            # differentiation:  ∂c/∂y = ifft2(fft2(c) · −i·fy), and the same
            # for x. Both directional contributions fuse into one ifft2 by
            # using the existing `deriv_factor` (Δry·−i·fy + Δrx·−i·fx).
            c  = self.to_complex(ascontig(c))
            c1 = self.to_complex(ascontig(c1))
            D  = self.deriv_factor(Deltar)
            combined = c1 + self.ifft2(self.fft2(c) * D)
            return self.S(combined, r, m)

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
        if not self._is_unit_mag(m):
            # Chirp-z magnification path.
            #   out1 = Sadj(Δφ).
            #   out2[t,i] = redot(Δφ, S(∂c/∂rᵢ, r, m))  with the spatial
            # derivatives ∂c/∂y, ∂c/∂x computed by FFT differentiation.
            ntheta_loc = c.shape[0]
            Deltaphi_c = self.to_complex(ascontig(Deltaphi))
            out1 = self.Sadj(Deltaphi_c, r, m)

            C = self.fft2(self.to_complex(ascontig(c)))
            d_y_c = self.ifft2(C * self.negi_fy[None, :, None])
            d_x_c = self.ifft2(C * self.negi_fx[None, None, :])
            del C
            dy = self.S(d_y_c, r, m)
            dx = self.S(d_x_c, r, m)

            out2 = cp.empty([ntheta_loc, 2], dtype='float32')
            out2[:, 0] = redot(Deltaphi_c, dy, axis=(1, 2))
            out2[:, 1] = redot(Deltaphi_c, dx, axis=(1, 2))
            return [out1, out2]

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
        if not self._is_unit_mag(m):
            # Chirp-z magnification path. By linearity of S, the same algebra
            # as the m=1 spectral combiner holds, applied to a single
            # spatial "combined input":
            #   d2curlySc = S(ifft2(C·D1·D2 + C1·D2 + C2·D1), r, m).
            C  = self.fft2(self.to_complex(ascontig(c )))
            C1 = self.fft2(self.to_complex(ascontig(c1)))
            C2 = self.fft2(self.to_complex(ascontig(c2)))
            D1 = self.deriv_factor(Deltar1)
            D2 = self.deriv_factor(Deltar2)
            combined = self.ifft2(C * D1 * D2 + C1 * D2 + C2 * D1)
            del C, C1, C2, D1, D2
            return self.S(combined, r, m)

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

    # ==================================================================
    # Magnification-derivative extensions (Option 3): dcurlySmc / dcurlySadjmc
    # See docs/shift_fft_m_derivative.md for derivations.
    # ==================================================================

    def _chirpz_lastaxis_dm(self, X, m, b, axis_xy):
        """Forward chirp-z along the last axis PLUS its m-derivative, in one
        pass. Returns (g, dg_dm) with the same shape as _chirpz_lastaxis(...).

        m appears through beta = m/N_in (in pre_twist, h, post) AND through b
        (in pre_twist only, since the caller passes b = (N_in-1)/2 - r -
        m*(N_out-1)/2 with a linear m-dependence). Both are linear in m, so
        every phase's m-derivative is a constant (per index) multiplier
        (see docs/shift_fft_m_derivative.md).

        Only the forward direction is exposed here (adjoint = False in the
        original _chirpz_lastaxis). The adjoint is not needed: dS/dm images
        become adjoint outputs via a redot in dcurlySadjmc.
        """
        if axis_xy == 'x':
            N_in, N_out, L = self.npsi,  self.n,  self.L_x
            k_signed = self.k_signed_x
            k_sq, ty_sq, j_sq = self.k_sq_x, self.ty_sq_x, self.j_sq_x
        else:
            N_in, N_out, L = self.nzpsi, self.nz, self.L_y
            k_signed = self.k_signed_y
            k_sq, ty_sq, j_sq = self.k_sq_y, self.ty_sq_y, self.j_sq_y

        B = X.shape[0]
        K = X.shape[1]
        beta = (cp.asarray(m).astype('float32') / cp.float32(N_in))   # [B]
        b    = cp.asarray(b).astype('float32')                        # [B]

        # phi_pre(k, t) and its m-derivative (both linear in m, so d/dm is a scalar-per-k).
        #   phi_pre         = (2π/N_in) * b * k_signed - π * β * k_sq
        #   d phi_pre / d m = -(π*(N_out-1)/N_in) * k_signed - (π/N_in) * k_sq
        two_pi_over_Nin = cp.float32(2.0 * cp.pi / N_in)
        pi_over_Nin     = cp.float32(cp.pi / N_in)
        pi_Nout_over_Nin = cp.float32(cp.pi * (N_out - 1) / N_in)

        phase_pre    = (two_pi_over_Nin * b[:, None] * k_signed[None, :]
                        - cp.pi * beta[:, None] * k_sq[None, :]).astype('float32')  # [B, N_in]
        pre_twist    = cp.exp(1j * phase_pre).astype('complex64')                     # [B, N_in]
        # d phi_pre / d m (does NOT depend on t — same k-profile for all t):
        phi_pre_dm   = (-pi_Nout_over_Nin * k_signed - pi_over_Nin * k_sq).astype('float32')  # [N_in]
        # d pre_twist / d m = 1j * phi_pre_dm * pre_twist   (broadcast over t)
        dpre_twist_dm = (1j * phi_pre_dm[None, :]).astype('complex64') * pre_twist    # [B, N_in]

        # post(ty, t) = exp(-1j π β ty²), d/dm = -1j π (1/N_in) ty² post
        post          = cp.exp(-1j * cp.pi * beta[:, None] * ty_sq[None, :]).astype('complex64')  # [B, N_out]
        phi_post_dm   = (-pi_over_Nin * ty_sq).astype('float32')                                   # [N_out]
        dpost_dm      = (1j * phi_post_dm[None, :]).astype('complex64') * post                     # [B, N_out]

        # h(j, t) = exp(+1j π β j²), d/dm = +1j π (1/N_in) j² h
        h             = cp.exp(1j * cp.pi * beta[:, None] * j_sq[None, :]).astype('complex64')    # [B, N_in+N_out-1]
        phi_h_dm      = (pi_over_Nin * j_sq).astype('float32')                                     # [N_in+N_out-1]
        dh_dm         = (1j * phi_h_dm[None, :]).astype('complex64') * h                           # [B, N_in+N_out-1]

        # Zero-pad h and dh_dm to length L, then take FFT to get H_hat and dH_hat_dm.
        h_pad         = cp.zeros((B, L), dtype='complex64')
        h_pad[:, :N_in + N_out - 1] = h
        dh_pad_dm     = cp.zeros((B, L), dtype='complex64')
        dh_pad_dm[:, :N_in + N_out - 1] = dh_dm
        H_hat         = cp.fft.fft(h_pad,     axis=-1)
        dH_hat_dm     = cp.fft.fft(dh_pad_dm, axis=-1)

        # -- Forward stage: FFT the input.
        C = cp.fft.fft(X, axis=-1)                                                # [B, K, N_in]

        # a and da/dm share C (X is independent of m).
        a     = C * pre_twist   [:, None, :]
        da_dm = C * dpre_twist_dm[:, None, :]

        a     = cp.fft.fftshift(a,     axes=-1)
        da_dm = cp.fft.fftshift(da_dm, axes=-1)

        # Zero-pad to length L.
        a_pad      = cp.zeros((B, K, L), dtype='complex64')
        a_pad[:, :, :N_in] = a
        da_pad_dm  = cp.zeros((B, K, L), dtype='complex64')
        da_pad_dm[:, :, :N_in] = da_dm

        # A_pos = ifft(a_pad, norm='forward'), same for derivative.
        A_pos      = cp.fft.ifft(a_pad,     axis=-1, norm='forward')
        dA_pos_dm  = cp.fft.ifft(da_pad_dm, axis=-1, norm='forward')

        # corr = ifft(A_pos * H_hat), and by Leibniz:
        # dcorr/dm = ifft(dA_pos/dm * H_hat + A_pos * dH_hat/dm)
        corr       = cp.fft.ifft(A_pos * H_hat[:, None, :], axis=-1)              # [B, K, L]
        dcorr_dm   = cp.fft.ifft(dA_pos_dm * H_hat[:, None, :]
                                 + A_pos * dH_hat_dm[:, None, :], axis=-1)

        inv_Nin = cp.float32(1.0 / N_in)
        g      = corr    [:, :, :N_out] * post[:, None, :] * inv_Nin
        dg_dm  = (dcorr_dm[:, :, :N_out] * post   [:, None, :]
                  + corr   [:, :, :N_out] * dpost_dm[:, None, :]) * inv_Nin
        return g, dg_dm

    def _chirpz_2d_dm(self, X, m, ry, rx):
        """Separable 2-D chirp-z forward plus per-axis m-derivatives.
        Returns (g, dg_dmy, dg_dmx), each same shape as `_chirpz_2d(..., adjoint=False)`.

        Composition (forward): S_2D(X) = chirpZ_y( chirpZ_x(X, mx), my ).
        By chain rule (chirpZ_y linear in its first arg):
          dS_2D / d mx = chirpZ_y( d chirpZ_x / d mx (X), my )
          dS_2D / d my = d chirpZ_y / d my ( chirpZ_x(X, mx) )
        """
        m  = cp.asarray(m).astype('float32')
        my = cp.ascontiguousarray(m[:, 0])
        mx = cp.ascontiguousarray(m[:, 1])
        ry = cp.asarray(ry).astype('float32')
        rx = cp.asarray(rx).astype('float32')
        b_x = (cp.float32((self.npsi  - 1) * 0.5) - rx - mx * cp.float32((self.n  - 1) * 0.5))
        b_y = (cp.float32((self.nzpsi - 1) * 0.5) - ry - my * cp.float32((self.nz - 1) * 0.5))

        X = self.to_complex(ascontig(X))                                # [B, nzpsi, npsi]

        # X-axis pass with m-derivative.
        Y_x, dY_dmx = self._chirpz_lastaxis_dm(X, mx, b_x, 'x')          # both [B, nzpsi, n]

        # Swap for y-axis pass (last axis becomes y).
        Y_x_swap   = cp.ascontiguousarray(cp.swapaxes(Y_x,   -2, -1))    # [B, n, nzpsi]
        dY_dmx_swap = cp.ascontiguousarray(cp.swapaxes(dY_dmx, -2, -1))

        # Y-axis pass on the value: get final value + dS/dmy.
        Y_val_swap, dY_dmy_swap = self._chirpz_lastaxis_dm(Y_x_swap, my, b_y, 'y')  # [B, n, nz]

        # Y-axis forward on the x-derivative branch → completes dS/dmx.
        dY_dmx_final_swap = self._chirpz_lastaxis(dY_dmx_swap, my, b_y, 'y', adjoint=False)

        # Swap back to [B, nz, n].
        g       = cp.ascontiguousarray(cp.swapaxes(Y_val_swap,        -2, -1))
        dg_dmy  = cp.ascontiguousarray(cp.swapaxes(dY_dmy_swap,       -2, -1))
        dg_dmx  = cp.ascontiguousarray(cp.swapaxes(dY_dmx_final_swap, -2, -1))
        return g, dg_dmy, dg_dmx

    def _dS_dm_images(self, c, r, m):
        """Return (dS/dmy, dS/dmx) — full images of shape [ntheta, nz, n],
        computed at the base (c, r, m). Handles both the chirp-z and
        m=1 (unit-mag) paths.

        For the m=1 fast path, uses the identity  dS/dm_axis = -tau_axis · dS/dr_axis
        with tau at the OUTPUT (per-pixel), reducing the m-derivative to two
        extra shifts on spatial derivatives of c.
        """
        if not self._is_unit_mag(m):
            _val, dg_dmy, dg_dmx = self._chirpz_2d_dm(c, m, r[:, 0], r[:, 1])
            return self.from_complex(dg_dmy), self.from_complex(dg_dmx)

        # m == 1 path: dS/dm_axis at output pixel = -tau_axis[pixel] * dS/dr_axis
        # tau_axis[ty] = ty - (N_out - 1)/2 (output-space centred coordinate).
        py, px = self.phase_separable(r)
        C = self.fft2(self.to_complex(ascontig(c)))
        # Spatial derivatives of c: ∂c/∂y = ifft(C · -i·fy), same for x.
        Cy = C * self.negi_fy[None, :, None]
        Cx = C * self.negi_fx[None, None, :]
        # Apply the same phase multiplier as S (in-place fuse).
        apply_sep_phase(Cy, py[:, :, cp.newaxis], px[:, cp.newaxis, :], Cy)
        apply_sep_phase(Cx, py[:, :, cp.newaxis], px[:, cp.newaxis, :], Cx)
        s_dy = self.ifft2(Cy)[:, :self.nz, :self.n]
        s_dx = self.ifft2(Cx)[:, :self.nz, :self.n]
        del Cy, Cx, C
        # Multiply by -tau (per output pixel).
        tau_y = (cp.arange(self.nz, dtype='float32') - cp.float32(0.5 * (self.nz - 1)))
        tau_x = (cp.arange(self.n , dtype='float32') - cp.float32(0.5 * (self.n  - 1)))
        dS_dmy = -tau_y[None, :, None] * s_dy
        dS_dmx = -tau_x[None, None, :] * s_dx
        return self.from_complex(dS_dmy), self.from_complex(dS_dmx)

    def dcurlySmc(self, c, r, m, c1, Deltar, Deltam):
        """1st directional derivative on (c, r, m). Matches Shift.dcurlySmc:
            curlySc(c1, r, m)
          + ∂/∂r curlySc(c, r, m) · Δr
          + ∂/∂m curlySc(c, r, m) · Δm
        """
        # r-block + c1: reuse existing dcurlySc for both branches.
        r_part = self.dcurlySc(c, r, m, c1, Deltar)

        # m-block: dS/dm_y * Δm_y + dS/dm_x * Δm_x.
        Deltam = cp.asarray(Deltam).astype('float32')
        # Short-circuit if Δm is all zero (avoids the extra chirp-z / spatial-diff work).
        if bool(cp.all(Deltam == 0)):
            return r_part
        dSy, dSx = self._dS_dm_images(c, r, m)
        m_part = (dSy * Deltam[:, 0, None, None]
                  + dSx * Deltam[:, 1, None, None])
        # dSy/dSx already come back at obj_dtype (from_complex applied inside
        # _dS_dm_images); Deltam is float32 → product keeps obj_dtype.
        return r_part + m_part

    def dcurlySadjmc(self, c, r, m, Deltaphi):
        """Adjoint of dcurlySmc. Returns [out1_c, out2_r, out2_m] such that
            <dcurlySmc(c, r, m, c1, Δr, Δm), Δφ>
              = <c1, out1_c> + <Δr, out2_r> + <Δm, out2_m>.

        out1_c, out2_r come from the existing dcurlySadjc; out2_m is added
        here by redotting Δφ with dS/dm_y and dS/dm_x images.
        """
        out1, out2_r = self.dcurlySadjc(c, r, m, Deltaphi)
        dSy, dSx = self._dS_dm_images(c, r, m)
        Dphi_c = self.to_complex(ascontig(Deltaphi))
        # dS/dm_i comes back at the output-real dtype from _dS_dm_images already.
        dSy_c = self.to_complex(ascontig(dSy))
        dSx_c = self.to_complex(ascontig(dSx))
        ntheta_loc = c.shape[0]
        out2_m = cp.empty([ntheta_loc, 2], dtype='float32')
        out2_m[:, 0] = redot(Dphi_c, dSy_c, axis=(1, 2))
        out2_m[:, 1] = redot(Dphi_c, dSx_c, axis=(1, 2))
        return [out1, out2_r, out2_m]

    def d2curlySmc(self, c, r, m, c1, Deltar1, Deltam1, c2, Deltar2, Deltam2):
        """Bilinear 2nd derivative on (c, r, m). NOT YET IMPLEMENTED.

        The native chirp-z 2nd-order derivation is a 9-term Leibniz expansion
        over the (pre_twist, h, post) branches (see
        docs/shift_fft_m_derivative.md). It's substantial work, so this
        stub raises to force a decision. Options:
          - shift_type='cubic' — the cubic Shift class already has a native
            single-kernel implementation and is production-ready today.
          - Wait for Option 3 (analytic chirp-z 2nd-order) to land here.
        """
        raise NotImplementedError(
            "ShiftFFT.d2curlySmc is not implemented — the analytic chirp-z "
            "2nd-order derivation (Option 3 in docs/shift_fft_m_derivative.md) "
            "hasn't been written yet. Use shift_type='cubic' when args.rho[3] "
            "> 0 (tp optimization) until it lands.")
