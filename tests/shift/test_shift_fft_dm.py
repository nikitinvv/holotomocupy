"""Taylor + adjoint tests for ShiftFFT.dcurlySmc / dcurlySadjmc / d2curlySmc
(magnification-derivative extensions — Option 3, native chirp-z path).

Run:
    python tests/shift/test_shift_fft_dm.py
"""

import os
import sys
import numpy as np
import cupy as cp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from holotomocupy.shift_fft import ShiftFFT


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------
NTHETA = 4
N      = 64
NZ     = 48
NPSI   = 96
NZPSI  = 80
SEED   = 15

# Non-unit m so the chirp-z path fires.
BASE_M = np.array([[1.10, 0.95],
                   [0.85, 1.05],
                   [0.72, 1.18],
                   [1.30, 0.78]], dtype='float32')

cl = ShiftFFT(N, NPSI, NZ, NZPSI, obj_dtype='complex64', nchunk=None)


def _rand_c(rng):
    re = rng.standard_normal((NTHETA, NZPSI, NPSI), dtype='float32')
    im = rng.standard_normal((NTHETA, NZPSI, NPSI), dtype='float32')
    x = (re + 1j * im).astype('complex64')
    ay = np.hanning(NZPSI).astype('float32')
    ax = np.hanning(NPSI).astype('float32')
    return cp.asarray(x * ay[None, :, None] * ax[None, None, :])


def _rand_r(rng, scale=1.5):
    return cp.asarray(scale * rng.standard_normal((NTHETA, 2)).astype('float32'))


def _rand_m(rng, scale=0.05):
    return cp.asarray((BASE_M + scale * rng.standard_normal(BASE_M.shape)).astype('float32'))


def _rand_direction(rng, shape, dtype='float32'):
    if dtype == 'complex64':
        re = rng.standard_normal(shape, dtype='float32')
        im = rng.standard_normal(shape, dtype='float32')
        return cp.asarray((re + 1j * im).astype('complex64'))
    return cp.asarray(rng.standard_normal(shape).astype(dtype))


# ---------------------------------------------------------------------------
# 1) Taylor test for dcurlySmc  (1st order)
# ---------------------------------------------------------------------------
def taylor_test():
    rng = np.random.default_rng(SEED)
    c   = _rand_c(rng)
    r   = _rand_r(rng)
    m   = _rand_m(rng)
    c1  = _rand_c(rng)
    dr  = _rand_direction(rng, (NTHETA, 2))
    dm  = _rand_direction(rng, (NTHETA, 2))

    S0     = cl.curlySc(c, r, m)
    dS     = cl.dcurlySmc(c, r, m, c1, dr, dm)
    S0_norm = float(cp.linalg.norm(S0.astype('complex64')).get()) + 1e-30

    print('\n=== Taylor test: dcurlySmc ===')
    print(f'{"eps":>10}  {"|residual| / |S0|":>22}  {"|no-corr| / |S0|":>22}')
    print('-' * 60)
    for eps in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
        eps_f = np.float32(eps)
        S_pert = cl.curlySc(c + eps_f * c1, r + eps_f * dr, m + eps_f * dm)
        resid = S_pert - S0 - eps_f * dS
        resid_norm = float(cp.linalg.norm(resid.astype('complex64')).get())
        base = S_pert - S0
        base_norm = float(cp.linalg.norm(base.astype('complex64')).get())
        print(f'{eps:>10.1e}  {resid_norm / S0_norm:>22.3e}  '
              f'{base_norm / S0_norm:>22.3e}')
    print('  → residuals should shrink like ε², baseline like ε.')


# ---------------------------------------------------------------------------
# 2) Adjoint test for dcurlySadjmc
# ---------------------------------------------------------------------------
def adjoint_test():
    rng = np.random.default_rng(SEED + 1)
    c    = _rand_c(rng)
    r    = _rand_r(rng)
    m    = _rand_m(rng)
    c1   = _rand_c(rng)
    dr   = _rand_direction(rng, (NTHETA, 2))
    dm   = _rand_direction(rng, (NTHETA, 2))
    re   = rng.standard_normal((NTHETA, NZ, N), dtype='float32')
    im   = rng.standard_normal((NTHETA, NZ, N), dtype='float32')
    dphi = cp.asarray((re + 1j * im).astype('complex64'))

    lhs_img = cl.dcurlySmc(c, r, m, c1, dr, dm).astype('complex64')
    lhs = float(cp.real(cp.sum(cp.conj(dphi) * lhs_img)).get())

    out1, out2_r, out2_m = cl.dcurlySadjmc(c, r, m, dphi)
    out1 = out1.astype('complex64')
    rhs_c1 = float(cp.real(cp.sum(cp.conj(c1) * out1)).get())
    rhs_r  = float(cp.sum(dr * out2_r).get())
    rhs_m  = float(cp.sum(dm * out2_m).get())
    rhs    = rhs_c1 + rhs_r + rhs_m

    print('\n=== Adjoint test: dcurlySmc vs dcurlySadjmc ===')
    print(f'  lhs = <dcurlySmc, Δφ>            = {lhs:+.6e}')
    print(f'  rhs = <c1, out1> + <Δr, out_r> + <Δm, out_m>  = {rhs:+.6e}')
    print(f'    <c1, out1_c>                   = {rhs_c1:+.6e}')
    print(f'    <Δr, out2_r>                   = {rhs_r:+.6e}')
    print(f'    <Δm, out2_m>                   = {rhs_m:+.6e}')
    print(f'  relative error                   = {abs(lhs - rhs) / (abs(lhs) + 1e-30):.3e}')


# ---------------------------------------------------------------------------
# 3) Sanity check for dS/dm via central FD (dcurlySmc "m only")
# ---------------------------------------------------------------------------
def m_deriv_fd_sanity():
    rng = np.random.default_rng(SEED + 2)
    c   = _rand_c(rng)
    r   = _rand_r(rng)
    m   = _rand_m(rng)
    dm  = _rand_direction(rng, (NTHETA, 2))

    ds_analytic = cl.dcurlySmc(c, r, m, cp.zeros_like(c), cp.zeros_like(dm), dm)

    print('\n=== FD sanity: dS/dm via dcurlySmc ===')
    print(f'{"h":>10}  {"|analytic - FD| / |analytic|":>32}')
    print('-' * 46)
    for h_val in (1e-2, 1e-3, 1e-4, 1e-5):
        h_f = np.float32(h_val)
        S_plus  = cl.curlySc(c, r, m + h_f * dm).astype('complex64')
        S_minus = cl.curlySc(c, r, m - h_f * dm).astype('complex64')
        ds_fd   = (S_plus - S_minus) / (2 * h_f)
        num = float(cp.linalg.norm((ds_analytic.astype('complex64') - ds_fd)).get())
        den = float(cp.linalg.norm(ds_analytic.astype('complex64')).get()) + 1e-30
        print(f'{h_val:>10.1e}  {num / den:>32.3e}')


# ---------------------------------------------------------------------------
# 4) Taylor test for d2curlySmc  (2nd order)
# ---------------------------------------------------------------------------
def taylor_test_d2():
    """S(v + ε·v1) − S(v) − ε·dcurlySmc(v1) − 0.5·ε²·d2curlySmc(v1, v1) = O(ε³).
    Should shrink like ε³ (slope 3 on log-log) until float32 noise (~1e-6).
    The 'no 2nd' column subtracts only 1st order and should shrink like ε².
    """
    rng = np.random.default_rng(SEED + 3)
    c   = _rand_c(rng)
    r   = _rand_r(rng)
    m   = _rand_m(rng)
    c1  = _rand_c(rng)
    dr  = _rand_direction(rng, (NTHETA, 2))
    dm  = _rand_direction(rng, (NTHETA, 2))

    S0  = cl.curlySc(c, r, m)
    dS  = cl.dcurlySmc(c, r, m, c1, dr, dm)
    d2S = cl.d2curlySmc(c, r, m, c1, dr, dm, c1, dr, dm)   # diagonal (v1, v1)
    S0_norm = float(cp.linalg.norm(S0.astype('complex64')).get()) + 1e-30

    print('\n=== Taylor test: d2curlySmc (diagonal (v1, v1)) ===')
    print(f'{"eps":>10}  {"|3rd-order resid|":>22}  {"|no 2nd|":>18}')
    print('-' * 58)
    for eps in (1e-1, 5e-2, 2e-2, 1e-2, 5e-3):
        eps_f  = np.float32(eps)
        S_pert = cl.curlySc(c + eps_f * c1, r + eps_f * dr, m + eps_f * dm)
        resid  = S_pert - S0 - eps_f * dS - np.float32(0.5) * (eps_f ** 2) * d2S
        resid_norm = float(cp.linalg.norm(resid.astype('complex64')).get())
        no2    = S_pert - S0 - eps_f * dS
        no2_norm = float(cp.linalg.norm(no2.astype('complex64')).get())
        print(f'{eps:>10.1e}  {resid_norm / S0_norm:>22.3e}  '
              f'{no2_norm / S0_norm:>18.3e}')
    print('  → 3rd-order residual should scale like ε³; "no 2nd" like ε².')


# ---------------------------------------------------------------------------
# 5) Symmetry check for d2curlySmc  (bilinear form must be symmetric)
# ---------------------------------------------------------------------------
def symmetry_test_d2():
    rng = np.random.default_rng(SEED + 4)
    c   = _rand_c(rng)
    r   = _rand_r(rng)
    m   = _rand_m(rng)
    c1  = _rand_c(rng); dr1 = _rand_direction(rng, (NTHETA, 2)); dm1 = _rand_direction(rng, (NTHETA, 2))
    c2  = _rand_c(rng); dr2 = _rand_direction(rng, (NTHETA, 2)); dm2 = _rand_direction(rng, (NTHETA, 2))

    a = cl.d2curlySmc(c, r, m, c1, dr1, dm1, c2, dr2, dm2)
    b = cl.d2curlySmc(c, r, m, c2, dr2, dm2, c1, dr1, dm1)
    diff = float(cp.linalg.norm((a - b).astype('complex64')).get())
    ref  = float(cp.linalg.norm(a.astype('complex64')).get()) + 1e-30
    print('\n=== Symmetry test: d2curlySmc(v1,v2) vs d2curlySmc(v2,v1) ===')
    print(f'  relative asymmetry = {diff / ref:.3e}  (should be float32 noise)')


if __name__ == '__main__':
    taylor_test()
    adjoint_test()
    m_deriv_fd_sanity()
    taylor_test_d2()
    symmetry_test_d2()
