"""Tests for Shift(symmetric=True) — the mirror-padded cubic prefilter path.

In symmetric mode:
  - coeff(small_psi) → big c              (mirror_pad + prefilter on 2× grid)
  - coeff(big_c)     → small psi          (prefilter + fold-and-sum, the adjoint)
  - S / Sadj / curlySc / dcurlySc / dcurlySadjc / d2curlySc all operate on
    big-grid c with the kernel sizes passed as 2·nzpsi / 2·npsi.

These tests verify the math: the new adjoint chain still satisfies
<S(c), b> = <c, Sadj(b)> and <dcurlySc(c, r, c1, Δr), Δφ> = <c1, out1> +
<Δr, out2>, plus that the full curlyS / curlyS_adjoint pair is correct
where curlyS_adjoint(Δφ) = coeff(Sadj(Δφ)) (using coeff in its
shape-polymorphic adjoint mode).
"""

import numpy as np
import cupy as cp

from holotomocupy.shift import Shift


def _rand_complex(shape, seed):
    rng = cp.random.default_rng(seed)
    return (rng.standard_normal(shape, dtype='float32')
            + 1j * rng.standard_normal(shape, dtype='float32')).astype('complex64')


def _inner(a, b):
    return float(cp.real(cp.sum(cp.conj(a) * b)))


# ----------------------------------------------------------------------
# 1. nzpsi_eff / npsi_eff exposed correctly
# ----------------------------------------------------------------------
def test_eff_shapes(n=48, npsi=64):
    cl_sym = Shift(n=n, npsi=npsi, nz=n, nzpsi=npsi, obj_dtype='complex64', symmetric=True)
    cl_no  = Shift(n=n, npsi=npsi, nz=n, nzpsi=npsi, obj_dtype='complex64', symmetric=False)
    assert (cl_sym.nzpsi_eff, cl_sym.npsi_eff) == (2 * npsi, 2 * npsi)
    assert (cl_no.nzpsi_eff,  cl_no.npsi_eff)  == (npsi, npsi)
    print(f"[1] eff shapes: sym {cl_sym.nzpsi_eff}×{cl_sym.npsi_eff}  "
          f"non-sym {cl_no.nzpsi_eff}×{cl_no.npsi_eff}")


# ----------------------------------------------------------------------
# 2. coeff polymorphism: small → big → small round-trip shape
# ----------------------------------------------------------------------
def test_coeff_polymorphic_shapes(n=48, npsi=64, ntheta=2):
    cl = Shift(n=n, npsi=npsi, nz=n, nzpsi=npsi, obj_dtype='complex64', symmetric=True)
    psi = _rand_complex([ntheta, npsi, npsi], seed=0)
    c_big = cl.coeff(psi)
    assert c_big.shape == (ntheta, 2 * npsi, 2 * npsi), c_big.shape
    psi_back = cl.coeff(c_big)
    assert psi_back.shape == (ntheta, npsi, npsi), psi_back.shape
    print(f"[2] coeff(small)={c_big.shape}, coeff(big)={psi_back.shape}")


# ----------------------------------------------------------------------
# 3. Adjoint of coeff (cubic, symmetric): <coeff(small_psi), big_grad> = <small_psi, coeff(big_grad)>
# ----------------------------------------------------------------------
def test_coeff_adjoint(n=48, npsi=64, ntheta=3):
    cl = Shift(n=n, npsi=npsi, nz=n, nzpsi=npsi, obj_dtype='complex64', symmetric=True)
    psi  = _rand_complex([ntheta, npsi, npsi], seed=1)
    g    = _rand_complex([ntheta, 2 * npsi, 2 * npsi], seed=2)
    lhs = _inner(cl.coeff(psi), g)
    rhs = _inner(psi, cl.coeff(g))
    rel = abs(lhs - rhs) / (abs(lhs) + 1e-30)
    print(f"[3] coeff adjoint (sym): lhs={lhs:.6e} rhs={rhs:.6e} rel={rel:.3e}")
    assert rel < 1e-4


# ----------------------------------------------------------------------
# 4. Adjoint test S/Sadj on big coefficient grid
# ----------------------------------------------------------------------
def test_S_adjoint_symmetric(n=48, npsi=64, ntheta=4):
    cl = Shift(n=n, npsi=npsi, nz=n, nzpsi=npsi, obj_dtype='complex64', symmetric=True)
    rng = cp.random.default_rng(3)
    r = (rng.random([ntheta, 2], dtype='float32') - 0.5) * 10
    m = cp.ones(ntheta, dtype='float32')
    c = _rand_complex([ntheta, 2 * npsi, 2 * npsi], seed=4)   # on big grid
    b = _rand_complex([ntheta, n, n], seed=5)
    lhs = _inner(cl.S(c, r, m),    b)
    rhs = _inner(c, cl.Sadj(b, r, m))
    rel = abs(lhs - rhs) / (abs(lhs) + 1e-30)
    print(f"[4] S/Sadj adjoint (sym, big c): lhs={lhs:.6e} rhs={rhs:.6e} rel={rel:.3e}")
    assert rel < 1e-4


# ----------------------------------------------------------------------
# 5. Adjoint test for curlyS chain: <curlyS(psi), Δφ> = <psi, coeff(Sadj(Δφ))>
# ----------------------------------------------------------------------
def test_curlyS_adjoint_symmetric(n=48, npsi=64, ntheta=3):
    cl = Shift(n=n, npsi=npsi, nz=n, nzpsi=npsi, obj_dtype='complex64', symmetric=True)
    rng = cp.random.default_rng(6)
    r = (rng.random([ntheta, 2], dtype='float32') - 0.5) * 10
    m = cp.ones(ntheta, dtype='float32')
    psi = _rand_complex([ntheta, npsi, npsi], seed=7)
    dphi = _rand_complex([ntheta, n, n], seed=8)
    lhs = _inner(cl.curlyS(psi, r, m), dphi)
    rhs = _inner(psi, cl.coeff(cl.Sadj(dphi, r, m)))   # big Sadj → small via coeff adjoint
    rel = abs(lhs - rhs) / (abs(lhs) + 1e-30)
    print(f"[5] curlyS adjoint (sym, full chain): lhs={lhs:.6e} rhs={rhs:.6e} rel={rel:.3e}")
    assert rel < 1e-4


# ----------------------------------------------------------------------
# 6. dcurlySc / dcurlySadjc adjoint on big coeff grid
# ----------------------------------------------------------------------
def test_dcurlyS_adjoint_symmetric(n=48, npsi=64, ntheta=4):
    cl = Shift(n=n, npsi=npsi, nz=n, nzpsi=npsi, obj_dtype='complex64', symmetric=True)
    rng = cp.random.default_rng(9)
    r  = (rng.random([ntheta, 2], dtype='float32') - 0.5) * 10
    m  = cp.ones(ntheta, dtype='float32')
    c  = _rand_complex([ntheta, 2 * npsi, 2 * npsi], seed=10)
    c1 = _rand_complex([ntheta, 2 * npsi, 2 * npsi], seed=11)
    Dr = (rng.random([ntheta, 2], dtype='float32') - 0.5) * 1.0
    Dp = _rand_complex([ntheta, n, n], seed=12)
    fwd = cl.dcurlySc(c, r, m, c1, Dr)
    out1, out2 = cl.dcurlySadjc(c, r, m, Dp)
    assert out1.shape == c.shape, (out1.shape, c.shape)
    lhs = _inner(fwd, Dp)
    rhs = _inner(c1, out1) + float(cp.sum(Dr * out2))
    rel = abs(lhs - rhs) / (abs(lhs) + 1e-30)
    print(f"[6] dcurlyS adjoint (sym): lhs={lhs:.6e} rhs={rhs:.6e} rel={rel:.3e}")
    assert rel < 1e-4


# ----------------------------------------------------------------------
# 7. Finite-difference vs d2curlySc on big coeff grid  (y == z case)
#
# The cubic d2s_kernel pairs (c1, Δ1) and (c2, Δ2) within-pair, while the
# standard Hessian uses the cross pairing (c1, Δ2) + (c2, Δ1). The two
# agree only when c1 = c2 and Δ1 = Δ2 — which is precisely the path the
# BH cascade in rec_mpi.py exercises (via y_is_z). Test that path here.
# ----------------------------------------------------------------------
def test_fd_d2curlySc_symmetric(n=48, npsi=64, ntheta=2, eps=1e-3):
    cl = Shift(n=n, npsi=npsi, nz=n, nzpsi=npsi, obj_dtype='complex64', symmetric=True)
    rng = cp.random.default_rng(13)
    r  = (rng.random([ntheta, 2], dtype='float32') - 0.5) * 5
    m  = cp.ones(ntheta, dtype='float32')
    c  = _rand_complex([ntheta, 2 * npsi, 2 * npsi], seed=14)
    c1 = _rand_complex([ntheta, 2 * npsi, 2 * npsi], seed=15)
    D1 = (rng.random([ntheta, 2], dtype='float32') - 0.5) * 1.0
    analytic = cl.d2curlySc(c, r, m, c1, D1, c1, D1)
    plus  = cl.dcurlySc(c + eps * c1, r + eps * D1, m, c1, D1)
    minus = cl.dcurlySc(c - eps * c1, r - eps * D1, m, c1, D1)
    fd = (plus - minus) / (2 * eps)
    err = float(cp.max(cp.abs(analytic - fd)))
    ref = float(cp.max(cp.abs(analytic))) + 1e-30
    print(f"[7] FD vs analytic d2curlySc (sym, y==z): rel={err/ref:.3e}")
    assert err / ref < 5e-2


if __name__ == '__main__':
    test_eff_shapes()
    test_coeff_polymorphic_shapes()
    test_coeff_adjoint()
    test_S_adjoint_symmetric()
    test_curlyS_adjoint_symmetric()
    test_dcurlyS_adjoint_symmetric()
    test_fd_d2curlySc_symmetric()
    print("\nAll cubic-symmetric tests passed.")
