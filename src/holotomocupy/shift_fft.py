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
    # Magnification-derivative extensions (Option 3):
    #   dcurlySmc / dcurlySadjmc / d2curlySmc
    # All phases (pre_twist, h, post) are LINEAR in r and m, so every phase's
    # derivative is a constant multiplier of the phase itself; every 2nd
    # phase-derivative is zero. Everything downstream (fftshift/pad/ifft/fft)
    # is LINEAR in its input, so Leibniz gives a finite closed-form expansion.
    # See docs/shift_fft_m_derivative.md for the derivation.
    # ==================================================================

    def _chirpz_lastaxis_all_derivs(self, X, m, b, axis_xy):
        """Forward chirp-z + every 1st- and 2nd-order (r, m) derivative in
        a single shared pipeline. Returns a dict with 6 images (each
        shape [B, K, N_out]), keyed by:
            'val'  : chirp-z forward value          G(pre, h, post)
            'dr'   : image such that ∂S/∂r · Δr = out['dr'] * Δr[t]
            'dm'   : image such that ∂S/∂m · Δm = out['dm'] * Δm[t]
            'd2r'  : image such that ∂²S/∂r² · Δr1·Δr2 = out['d2r'] * Δr1·Δr2
            'd2m'  : image such that ∂²S/∂m² · Δm1·Δm2 = out['d2m'] * Δm1·Δm2
            'drdm' : image such that ∂²S/∂r∂m · (Δr1·Δm2 + Δr2·Δm1)
                                      = out['drdm'] * (Δr1·Δm2 + Δr2·Δm1)

        FFT count (per axis): 1 (fft(X)) + 6 (A_pos ifft per pre variant)
        + 3 (H_hat fft per h variant) + 10 (unique corr ifft calls) = ~20.
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

        two_pi_over_Nin  = cp.float32(2.0 * cp.pi / N_in)
        pi_over_Nin      = cp.float32(cp.pi / N_in)
        pi_Nout_over_Nin = cp.float32(cp.pi * (N_out - 1) / N_in)

        # ---- 1st-order phase derivatives (per-index scalars, no t dep). --
        # phi_pre = (2π/N_in)·b·k_signed − π·β·k_sq
        # phi_h   = π·β·j_sq
        # phi_post = −π·β·ty_sq
        phi_pre_dr = (-two_pi_over_Nin * k_signed).astype('float32')                 # [N_in]
        phi_pre_dm = (-pi_Nout_over_Nin * k_signed - pi_over_Nin * k_sq).astype('float32')
        phi_h_dm   = (pi_over_Nin * j_sq).astype('float32')                          # [N_in+N_out-1]
        phi_post_dm = (-pi_over_Nin * ty_sq).astype('float32')                       # [N_out]

        # ---- Elementary phase-only quantities (per (t, index)). ----------
        phase_pre = (two_pi_over_Nin * b[:, None] * k_signed[None, :]
                     - cp.pi * beta[:, None] * k_sq[None, :]).astype('float32')
        pre_twist = cp.exp(1j * phase_pre).astype('complex64')                       # [B, N_in]

        post = cp.exp(-1j * cp.pi * beta[:, None] * ty_sq[None, :]).astype('complex64')  # [B, N_out]
        h    = cp.exp( 1j * cp.pi * beta[:, None] * j_sq[None, :]).astype('complex64')   # [B, N_in+N_out-1]

        # ---- 6 pre variants (all differ by a per-k float multiplier).
        # For each we materialize C*pre_variant → fftshift → pad → ifft-forward
        # to get A_pos.
        C = cp.fft.fft(X, axis=-1)                                                   # [B, K, N_in]

        def _apos_from_multiplier(pre_mult_1d):
            """A_pos = ifft(pad(shift(C * pre_twist * pre_mult_1d))), norm='forward'."""
            a = (C * pre_twist[:, None, :]) * pre_mult_1d[None, None, :]
            a = cp.fft.fftshift(a, axes=-1)
            a_pad = cp.zeros((B, K, L), dtype='complex64')
            a_pad[:, :, :N_in] = a
            return cp.fft.ifft(a_pad, axis=-1, norm='forward')

        ones_Nin = cp.ones(N_in, dtype='complex64')
        # A_pos variants (each ~ 1 length-L ifft).
        Ap_val  = _apos_from_multiplier(ones_Nin)
        Ap_r    = _apos_from_multiplier(phi_pre_dr.astype('complex64'))
        Ap_m    = _apos_from_multiplier(phi_pre_dm.astype('complex64'))
        Ap_r2   = _apos_from_multiplier((phi_pre_dr * phi_pre_dr).astype('complex64'))
        Ap_m2   = _apos_from_multiplier((phi_pre_dm * phi_pre_dm).astype('complex64'))
        Ap_rm   = _apos_from_multiplier((phi_pre_dr * phi_pre_dm).astype('complex64'))

        # ---- 3 H_hat variants (each ~ 1 length-L fft).
        def _hhat(h_mult_1d):
            h_pad = cp.zeros((B, L), dtype='complex64')
            h_pad[:, :N_in + N_out - 1] = h * h_mult_1d[None, :]
            return cp.fft.fft(h_pad, axis=-1)
        ones_h = cp.ones(N_in + N_out - 1, dtype='complex64')
        Hh_val = _hhat(ones_h)
        Hh_m   = _hhat(phi_h_dm.astype('complex64'))
        Hh_m2  = _hhat((phi_h_dm * phi_h_dm).astype('complex64'))

        # ---- 3 post variants (multiplicative, no FFT).
        Po_val = post
        Po_m   = post * phi_post_dm[None, :].astype('complex64')
        Po_m2  = post * (phi_post_dm * phi_post_dm)[None, :].astype('complex64')

        # ---- corr(A, H) helper: ifft(A_pos * H_hat_broadcast) along last axis.
        # We compute each unique (Ap, Hh) combination exactly once.
        def _corr(Ap, Hh):
            return cp.fft.ifft(Ap * Hh[:, None, :], axis=-1)                          # [B, K, L]

        # Terms we need (kept as scaled slices [B, K, N_out]):
        inv_Nin = cp.float32(1.0 / N_in)

        def _finish(corr_arr, post_arr):
            """Truncate to N_out and apply the given post multiplier + 1/N_in."""
            return corr_arr[:, :, :N_out] * post_arr[:, None, :] * inv_Nin

        # ----- val = G(pre, h, post) -----
        val_img = _finish(_corr(Ap_val, Hh_val), Po_val)

        # ----- dr = 1j · G(phi_pre_dr·pre, h, post)  (only pre depends on r) -----
        dr_img = (1j) * _finish(_corr(Ap_r, Hh_val), Po_val)

        # ----- dm = 1j · [G(phi_pre_dm·pre, h, post) + G(pre, phi_h_dm·h, post)
        #                 + G(pre, h, phi_post_dm·post)] -----
        dm_img = (1j) * (
            _finish(_corr(Ap_m, Hh_val), Po_val)
            + _finish(_corr(Ap_val, Hh_m), Po_val)
            + _finish(_corr(Ap_val, Hh_val), Po_m)
        )

        # ----- d2r = -G(phi_pre_dr²·pre, h, post) -----
        d2r_img = -_finish(_corr(Ap_r2, Hh_val), Po_val)

        # ----- d2m = -[6 Leibniz terms] -----
        # pure squares:
        term_pp = _finish(_corr(Ap_m2, Hh_val), Po_val)   # G(phi_pre_dm²·pre, h, post)
        term_hh = _finish(_corr(Ap_val, Hh_m2), Po_val)   # G(pre, phi_h_dm²·h, post)
        term_op = _finish(_corr(Ap_val, Hh_val), Po_m2)   # G(pre, h, phi_post_dm²·post)
        # cross doubles:
        term_ph = _finish(_corr(Ap_m, Hh_m), Po_val)      # G(phi_pre_dm·pre, phi_h_dm·h, post)
        term_pp_op = _finish(_corr(Ap_m, Hh_val), Po_m)   # G(phi_pre_dm·pre, h, phi_post_dm·post)
        term_h_op = _finish(_corr(Ap_val, Hh_m), Po_m)    # G(pre, phi_h_dm·h, phi_post_dm·post)
        d2m_img = -(term_pp + term_hh + term_op
                    + 2 * term_ph + 2 * term_pp_op + 2 * term_h_op)

        # ----- drdm = -[3 Leibniz terms] (r-dep only lives in pre) -----
        # pre-pre cross:
        cross_pp = _finish(_corr(Ap_rm, Hh_val), Po_val)  # G(phi_pre_dr·phi_pre_dm·pre, h, post)
        # pre-h cross (r on pre, m on h):
        cross_ph = _finish(_corr(Ap_r, Hh_m), Po_val)     # G(phi_pre_dr·pre, phi_h_dm·h, post)
        # pre-post cross (r on pre, m on post):
        cross_pop = _finish(_corr(Ap_r, Hh_val), Po_m)    # G(phi_pre_dr·pre, h, phi_post_dm·post)
        drdm_img = -(cross_pp + cross_ph + cross_pop)

        return {'val': val_img, 'dr': dr_img, 'dm': dm_img,
                'd2r': d2r_img, 'd2m': d2m_img, 'drdm': drdm_img}

    def _chirpz_2d_apply_axis_val_only(self, X, m, b, axis_xy):
        """Shortcut: chirp-z forward only, same signature as
        _chirpz_lastaxis_all_derivs but returns just the value image (used
        as an internal building block when a specific x-branch only needs
        the y-forward)."""
        return self._chirpz_lastaxis(X, m, b, axis_xy, adjoint=False)

    def _dS_dm_images(self, c, r, m):
        """Return (dS/dmy, dS/dmx) — full images of shape [ntheta, nz, n],
        computed at the base (c, r, m). Handles both the chirp-z and
        m=1 (unit-mag) paths.

        Chirp-z path: uses the all-derivs helper on x-axis for {val, dm},
        then y-axis for the y-dm image, and forward-only y on x_dm for the
        cross image (dS/dmx after y-forward). Two chirp axes per image.

        Unit-mag path: uses the identity dS/dm_axis = -tau_axis · dS/dr_axis
        with tau at the OUTPUT (per-pixel), reducing the m-derivative to two
        extra shifts on spatial derivatives of c.
        """
        if not self._is_unit_mag(m):
            m_np = cp.asarray(m).astype('float32')
            my = cp.ascontiguousarray(m_np[:, 0])
            mx = cp.ascontiguousarray(m_np[:, 1])
            ry = cp.asarray(r[:, 0]).astype('float32')
            rx = cp.asarray(r[:, 1]).astype('float32')
            b_x = (cp.float32((self.npsi  - 1) * 0.5) - rx - mx * cp.float32((self.n  - 1) * 0.5))
            b_y = (cp.float32((self.nzpsi - 1) * 0.5) - ry - my * cp.float32((self.nz - 1) * 0.5))

            X = self.to_complex(ascontig(c))                                          # [B, nzpsi, npsi]
            # x-axis: need value + dm (for cross-axis dS/dmx = y_val(x_dm)).
            x_out = self._chirpz_lastaxis_all_derivs(X, mx, b_x, 'x')
            x_val = x_out['val']                                                       # [B, nzpsi, n]
            x_dm  = x_out['dm']

            # swap to [B, n, nzpsi] for y-axis chirp along last axis.
            x_val_s = cp.ascontiguousarray(cp.swapaxes(x_val, -2, -1))
            x_dm_s  = cp.ascontiguousarray(cp.swapaxes(x_dm , -2, -1))

            # y-axis on x_val: need value + dm (dm gives dS/dmy).
            y_out    = self._chirpz_lastaxis_all_derivs(x_val_s, my, b_y, 'y')
            g_dmy_s  = y_out['dm']                                                     # [B, n, nz]
            # y-axis forward on x_dm: gives dS/dmx (cross axis).
            g_dmx_s  = self._chirpz_lastaxis(x_dm_s, my, b_y, 'y', adjoint=False)

            g_dmy = cp.ascontiguousarray(cp.swapaxes(g_dmy_s, -2, -1))                 # [B, nz, n]
            g_dmx = cp.ascontiguousarray(cp.swapaxes(g_dmx_s, -2, -1))
            return self.from_complex(g_dmy), self.from_complex(g_dmx)

        # m == 1 fast path: use the identity dS/dm = -tau · dS/dr at output.
        py, px = self.phase_separable(r)
        C = self.fft2(self.to_complex(ascontig(c)))
        Cy = C * self.negi_fy[None, :, None]
        Cx = C * self.negi_fx[None, None, :]
        apply_sep_phase(Cy, py[:, :, cp.newaxis], px[:, cp.newaxis, :], Cy)
        apply_sep_phase(Cx, py[:, :, cp.newaxis], px[:, cp.newaxis, :], Cx)
        s_dy = self.ifft2(Cy)[:, :self.nz, :self.n]
        s_dx = self.ifft2(Cx)[:, :self.nz, :self.n]
        del Cy, Cx, C
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
        # Short-circuit if Δm is all zero (avoids the extra chirp-z work).
        if bool(cp.all(Deltam == 0)):
            return r_part
        dSy, dSx = self._dS_dm_images(c, r, m)
        m_part = (dSy * Deltam[:, 0, None, None]
                  + dSx * Deltam[:, 1, None, None])
        return r_part + m_part

    def dcurlySadjmc(self, c, r, m, Deltaphi):
        """Adjoint of dcurlySmc. Returns [out1_c, out2_r, out2_m]."""
        out1, out2_r = self.dcurlySadjc(c, r, m, Deltaphi)
        dSy, dSx = self._dS_dm_images(c, r, m)
        Dphi_c = self.to_complex(ascontig(Deltaphi))
        dSy_c = self.to_complex(ascontig(dSy))
        dSx_c = self.to_complex(ascontig(dSx))
        ntheta_loc = c.shape[0]
        out2_m = cp.empty([ntheta_loc, 2], dtype='float32')
        out2_m[:, 0] = redot(Dphi_c, dSy_c, axis=(1, 2))
        out2_m[:, 1] = redot(Dphi_c, dSx_c, axis=(1, 2))
        return [out1, out2_r, out2_m]

    def d2curlySmc(self, c, r, m, c1, Deltar1, Deltam1, c2, Deltar2, Deltam2):
        """Bilinear 2nd derivative on (c, r, m). Native chirp-z, no FD.

        Decomposition (S linear in c → all ∂²/∂c² terms vanish):
            d2curlySmc = rr + cr₁ + cr₂                    ← d2curlySc handles these
                       + cm₁ + cm₂                          ← linearity in c: dS/dm(c1), dS/dm(c2)
                       + mm  (dm1, dm2)                     ← native chirp 2D 2nd-order
                       + rm₁ + rm₂ (dr1·dm2, dr2·dm1)       ← native chirp 2D mixed 2nd-order

        The rm/mm block is 7 images (3 pure mm axes + 4 cross rm axes),
        each a specific composition of x-axis and y-axis chirp derivatives
        via `_chirpz_lastaxis_all_derivs`.
        """
        # Part 1: rr + cr₁ + cr₂ via existing d2curlySc (chirp-z internal path).
        result = self.d2curlySc(c, r, m, c1, Deltar1, c2, Deltar2)

        # Part 2: cm₁ + cm₂ from dS/dm applied to c1 and c2 (linearity in c).
        Deltam1 = cp.asarray(Deltam1).astype('float32')
        Deltam2 = cp.asarray(Deltam2).astype('float32')
        Deltar1 = cp.asarray(Deltar1).astype('float32')
        Deltar2 = cp.asarray(Deltar2).astype('float32')
        dSy_c1, dSx_c1 = self._dS_dm_images(c1, r, m)
        dSy_c2, dSx_c2 = self._dS_dm_images(c2, r, m)
        result = (result
                  + dSy_c1 * Deltam2[:, 0, None, None] + dSx_c1 * Deltam2[:, 1, None, None]
                  + dSy_c2 * Deltam1[:, 0, None, None] + dSx_c2 * Deltam1[:, 1, None, None])

        # Part 3: mm + rm blocks. Fall back on a "shortcut zero" if all m directions
        # are zero (no tp motion → no mm/rm contribution).
        if bool(cp.all(Deltam1 == 0)) and bool(cp.all(Deltam2 == 0)):
            return result

        # For the chirp-z path only: assemble 7 images by composing per-axis derivs.
        # (The unit-mag path is not exercised when tp is being optimized — force chirp.)
        m_np = cp.asarray(m).astype('float32')
        my = cp.ascontiguousarray(m_np[:, 0])
        mx = cp.ascontiguousarray(m_np[:, 1])
        r_a = cp.asarray(r).astype('float32')
        ry = cp.ascontiguousarray(r_a[:, 0])
        rx = cp.ascontiguousarray(r_a[:, 1])
        b_x = (cp.float32((self.npsi  - 1) * 0.5) - rx - mx * cp.float32((self.n  - 1) * 0.5))
        b_y = (cp.float32((self.nzpsi - 1) * 0.5) - ry - my * cp.float32((self.nz - 1) * 0.5))

        X = self.to_complex(ascontig(c))                                              # [B, nzpsi, npsi]

        # x-axis all-derivs (6 images on the x-output grid).
        x_all = self._chirpz_lastaxis_all_derivs(X, mx, b_x, 'x')
        x_val   = x_all['val']
        x_dr    = x_all['dr']
        x_dm    = x_all['dm']
        x_d2m   = x_all['d2m']
        x_drdm  = x_all['drdm']

        # Swap last two axes for y-axis chirp (last axis becomes y).
        def _swap(a): return cp.ascontiguousarray(cp.swapaxes(a, -2, -1))
        x_val_s   = _swap(x_val)
        x_dr_s    = _swap(x_dr)
        x_dm_s    = _swap(x_dm)
        x_d2m_s   = _swap(x_d2m)
        x_drdm_s  = _swap(x_drdm)

        # y-axis all-derivs on x_val → gives y_val, y_dm, y_d2m, y_drdm (all needed
        # for pure-y-axis 2nd derivs). y_val is the 2D forward value (unused here).
        y_val_all = self._chirpz_lastaxis_all_derivs(x_val_s, my, b_y, 'y')
        y_d2m_of_xval  = y_val_all['d2m']         # ∂²S/∂my²
        y_drdm_of_xval = y_val_all['drdm']        # ∂²S/∂ry∂my

        # y-axis on x_d2m: only need value → ∂²S/∂mx²
        y_val_of_xd2m = self._chirpz_lastaxis(x_d2m_s, my, b_y, 'y', adjoint=False)
        # y-axis on x_drdm: only need value → ∂²S/∂rx∂mx
        y_val_of_xdrdm = self._chirpz_lastaxis(x_drdm_s, my, b_y, 'y', adjoint=False)
        # y-axis on x_dm: need dm (for ∂²S/∂mx∂my) and dr (for ∂²S/∂ry∂mx)
        y_dm_all_of_xdm = self._chirpz_lastaxis_all_derivs(x_dm_s, my, b_y, 'y')
        y_dm_of_xdm = y_dm_all_of_xdm['dm']       # ∂²S/∂mx∂my (cross axes)
        y_dr_of_xdm = y_dm_all_of_xdm['dr']       # ∂²S/∂ry∂mx (cross axes)
        # y-axis on x_dr: need dm → ∂²S/∂rx∂my
        y_dm_all_of_xdr = self._chirpz_lastaxis_all_derivs(x_dr_s, my, b_y, 'y')
        y_dm_of_xdr = y_dm_all_of_xdr['dm']       # ∂²S/∂rx∂my (cross axes)

        # Swap back to [B, nz, n]. Each image is the "bilinear coefficient" — the
        # caller multiplies by the direction scalars to get the actual contribution.
        d2S_dmy2   = _swap(y_d2m_of_xval)         # ∂²S/∂my²  ← dm1_y·dm2_y
        d2S_dmx2   = _swap(y_val_of_xd2m)         # ∂²S/∂mx²  ← dm1_x·dm2_x
        d2S_dmydmx = _swap(y_dm_of_xdm)           # ∂²S/∂my∂mx ← (dm1_y·dm2_x + dm1_x·dm2_y)
        d2S_dry_dmy = _swap(y_drdm_of_xval)       # ∂²S/∂ry∂my ← (dr1_y·dm2_y + dr2_y·dm1_y)
        d2S_drx_dmx = _swap(y_val_of_xdrdm)       # ∂²S/∂rx∂mx ← (dr1_x·dm2_x + dr2_x·dm1_x)
        d2S_drx_dmy = _swap(y_dm_of_xdr)          # ∂²S/∂rx∂my ← (dr1_x·dm2_y + dr2_x·dm1_y)
        d2S_dry_dmx = _swap(y_dr_of_xdm)          # ∂²S/∂ry∂mx ← (dr1_y·dm2_x + dr2_y·dm1_x)

        # Combine with direction scalars (per-t; broadcast over pixels).
        dm1y = Deltam1[:, 0, None, None]; dm1x = Deltam1[:, 1, None, None]
        dm2y = Deltam2[:, 0, None, None]; dm2x = Deltam2[:, 1, None, None]
        dr1y = Deltar1[:, 0, None, None]; dr1x = Deltar1[:, 1, None, None]
        dr2y = Deltar2[:, 0, None, None]; dr2x = Deltar2[:, 1, None, None]

        mm_rm = (
            d2S_dmy2   * (dm1y * dm2y)
          + d2S_dmx2   * (dm1x * dm2x)
          + d2S_dmydmx * (dm1y * dm2x + dm1x * dm2y)
          + d2S_dry_dmy * (dr1y * dm2y + dr2y * dm1y)
          + d2S_drx_dmx * (dr1x * dm2x + dr2x * dm1x)
          + d2S_drx_dmy * (dr1x * dm2y + dr2x * dm1y)
          + d2S_dry_dmx * (dr1y * dm2x + dr2y * dm1x)
        )
        return result + self.from_complex(mm_rm)
