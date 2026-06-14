"""Tests for the FFT-based shift path (Shift(..., method='fft')).

Covers:
1. Forward consistency: FFT vs cubic agree on interior of integer shifts
   when m=1 and npsi=n, nzpsi=nz.
2. Adjoint test S/Sadj.
3. dcurlySc / dcurlySadjc adjoint test.
4. Finite-difference check on dcurlySc (∂S/∂r part).
5. Finite-difference check on d2curlySc.
6. The npsi > n case (center-pad offset).
"""

import numpy as np
import cupy as cp

from holotomocupy.shift import Shift
from holotomocupy.shift_fft import ShiftFFT


def _rand_complex(shape, seed):
    rng = cp.random.default_rng(seed)
    return (rng.standard_normal(shape, dtype='float32')
            + 1j * rng.standard_normal(shape, dtype='float32')).astype('complex64')


def _inner(a, b):
    """Re<a, b> over the entire array (sum_i Re(conj(a_i)*b_i))."""
    return float(cp.real(cp.sum(cp.conj(a) * b)))


# ----------------------------------------------------------------------
# 1. Forward consistency (interior only — boundary BCs differ)
# ----------------------------------------------------------------------
def _smooth_periodic_complex(n, ntheta):
    """Sum of integer-frequency sinusoids — exactly periodic on the n×n grid
    and band-limited to k≤5. Both cubic-spline and sinc (FFT) interpolation
    are near-exact in this regime, so cubic vs FFT can be compared
    quantitatively."""
    yy, xx = cp.mgrid[0:n, 0:n].astype('float32')
    re = (cp.sin(2*cp.pi*xx/n * 3) * cp.cos(2*cp.pi*yy/n * 2)
          + 0.5 * cp.sin(2*cp.pi*xx/n * 5) * cp.cos(2*cp.pi*yy/n * 4))
    im = (cp.cos(2*cp.pi*xx/n * 4) * cp.sin(2*cp.pi*yy/n * 5)
          + 0.5 * cp.cos(2*cp.pi*xx/n * 2) * cp.sin(2*cp.pi*yy/n * 3))
    c = (re + 1j*im).astype('complex64')
    return cp.broadcast_to(c[cp.newaxis], (ntheta, n, n)).copy()


def test_fft_vs_cubic_interior(n=64, ntheta=3, margin=10):
    """Cubic and FFT must agree on the interior for a smooth periodic signal.
    Random noise wouldn't work — sinc and cubic-spline interpolation differ
    substantially on broadband data."""
    cubic = Shift   (n=n, npsi=n, nz=n, nzpsi=n, obj_dtype='complex64')
    fft   = ShiftFFT(n=n, npsi=n, nz=n, nzpsi=n, obj_dtype='complex64')

    c = _smooth_periodic_complex(n, ntheta)
    r = cp.asarray(np.array([[1.3, -2.7], [-3.4, 4.1], [0.5, 0.5]], dtype='float32'))
    m = cp.ones(ntheta, dtype='float32')

    # Cubic needs the B-spline prefilter; FFT does not.
    a = cubic.curlyS(c, r, m).get()
    b = fft.curlyS(c, r, m).get()

    s = (slice(None), slice(margin, -margin), slice(margin, -margin))
    err = np.max(np.abs(a[s] - b[s]))
    ref = np.max(np.abs(a[s])) + 1e-30
    print(f"[1] fft vs cubic interior (smooth): max|err|={err:.3e}  rel={err/ref:.3e}")
    assert err / ref < 5e-2, "FFT and cubic should agree on interior for smooth input"


# ----------------------------------------------------------------------
# 2. Adjoint test for S
# ----------------------------------------------------------------------
def test_adjoint_S(n=64, ntheta=4):
    cl = ShiftFFT(n=n, npsi=n, nz=n, nzpsi=n, obj_dtype='complex64')
    rng = cp.random.default_rng(1)
    r = (rng.random([ntheta, 2], dtype='float32') - 0.5) * 20
    m = cp.ones(ntheta, dtype='float32')

    c = _rand_complex([ntheta, n, n], seed=2)
    b = _rand_complex([ntheta, n, n], seed=3)

    Sc    = cl.S(c, r, m)
    Sadjb = cl.Sadj(b, r, m)
    lhs = _inner(Sc, b)
    rhs = _inner(c, Sadjb)
    rel = abs(lhs - rhs) / (abs(lhs) + 1e-30)
    print(f"[2] S/Sadj adjoint: lhs={lhs:.6e} rhs={rhs:.6e} rel={rel:.3e}")
    assert rel < 1e-4, "S/Sadj must be an exact adjoint pair"


# ----------------------------------------------------------------------
# 3. dcurlySc / dcurlySadjc adjoint test
# ----------------------------------------------------------------------
def test_adjoint_dcurlyS(n=64, ntheta=4):
    cl = ShiftFFT(n=n, npsi=n, nz=n, nzpsi=n, obj_dtype='complex64')
    rng = cp.random.default_rng(4)
    r  = (rng.random([ntheta, 2], dtype='float32') - 0.5) * 20
    m  = cp.ones(ntheta, dtype='float32')
    c  = _rand_complex([ntheta, n, n], seed=5)
    c1 = _rand_complex([ntheta, n, n], seed=6)
    Dr = (rng.random([ntheta, 2], dtype='float32') - 0.5) * 2
    Dp = _rand_complex([ntheta, n, n], seed=7)

    forward = cl.dcurlySc(c, r, m, c1, Dr)
    out1, out2 = cl.dcurlySadjc(c, r, m, Dp)

    lhs = _inner(forward, Dp)
    rhs = _inner(c1, out1) + float(cp.sum(Dr * out2))
    rel = abs(lhs - rhs) / (abs(lhs) + 1e-30)
    print(f"[3] dcurlyS adjoint: lhs={lhs:.6e} rhs={rhs:.6e} rel={rel:.3e}")
    assert rel < 1e-4, "dcurlySc and dcurlySadjc must be adjoint"


# ----------------------------------------------------------------------
# 4. Finite-difference check on dcurlySc (∂S/∂r component)
# ----------------------------------------------------------------------
def test_fd_dcurlySc_r(n=64, ntheta=3, eps=1e-3):
    cl = ShiftFFT(n=n, npsi=n, nz=n, nzpsi=n, obj_dtype='complex64')
    rng = cp.random.default_rng(8)
    r  = (rng.random([ntheta, 2], dtype='float32') - 0.5) * 10
    m  = cp.ones(ntheta, dtype='float32')
    c  = _rand_complex([ntheta, n, n], seed=9)
    Dr = (rng.random([ntheta, 2], dtype='float32') - 0.5) * 2

    zero = cp.zeros_like(c)
    analytic = cl.dcurlySc(c, r, m, zero, Dr)
    fd = (cl.S(c, r + eps * Dr, m) - cl.S(c, r - eps * Dr, m)) / (2 * eps)
    err = float(cp.max(cp.abs(analytic - fd)))
    ref = float(cp.max(cp.abs(analytic))) + 1e-30
    print(f"[4] FD vs analytic dcurlySc(r): max|err|={err:.3e}  rel={err/ref:.3e}")
    assert err / ref < 1e-3, "Analytic dS/dr must match finite difference"


# ----------------------------------------------------------------------
# 5. Finite-difference check on d2curlySc
# ----------------------------------------------------------------------
def test_fd_d2curlySc(n=64, ntheta=3, eps=1e-3):
    cl = ShiftFFT(n=n, npsi=n, nz=n, nzpsi=n, obj_dtype='complex64')
    rng = cp.random.default_rng(10)
    r   = (rng.random([ntheta, 2], dtype='float32') - 0.5) * 10
    m   = cp.ones(ntheta, dtype='float32')
    c   = _rand_complex([ntheta, n, n], seed=11)
    c1  = _rand_complex([ntheta, n, n], seed=12)
    c2  = _rand_complex([ntheta, n, n], seed=13)
    D1  = (rng.random([ntheta, 2], dtype='float32') - 0.5) * 2
    D2  = (rng.random([ntheta, 2], dtype='float32') - 0.5) * 2

    analytic = cl.d2curlySc(c, r, m, c1, D1, c2, D2)
    # FD: derivative of dcurlySc(c2, D2) at (c, r) along (c1, D1)
    plus  = cl.dcurlySc(c + eps * c1, r + eps * D1, m, c2, D2)
    minus = cl.dcurlySc(c - eps * c1, r - eps * D1, m, c2, D2)
    fd = (plus - minus) / (2 * eps)
    err = float(cp.max(cp.abs(analytic - fd)))
    ref = float(cp.max(cp.abs(analytic))) + 1e-30
    print(f"[5] FD vs analytic d2curlySc:  max|err|={err:.3e}  rel={err/ref:.3e}")
    assert err / ref < 1e-2, "Analytic d2curlySc must match finite difference"


# ----------------------------------------------------------------------
# 6. npsi > n (center-pad case) — adjoint test
# ----------------------------------------------------------------------
def test_adjoint_S_padded(n=48, npsi=64, ntheta=3):
    cl = ShiftFFT(n=n, npsi=npsi, nz=n, nzpsi=npsi, obj_dtype='complex64')
    rng = cp.random.default_rng(14)
    r = (rng.random([ntheta, 2], dtype='float32') - 0.5) * 10
    m = cp.ones(ntheta, dtype='float32')
    c = _rand_complex([ntheta, npsi, npsi], seed=15)
    b = _rand_complex([ntheta, n,    n   ], seed=16)
    lhs = _inner(cl.S(c, r, m),    b)
    rhs = _inner(c, cl.Sadj(b, r, m))
    rel = abs(lhs - rhs) / (abs(lhs) + 1e-30)
    print(f"[6] S/Sadj adjoint (padded n={n}, npsi={npsi}): "
          f"lhs={lhs:.6e} rhs={rhs:.6e} rel={rel:.3e}")
    assert rel < 1e-4


# ----------------------------------------------------------------------
# 7. float32 obj_dtype path
# ----------------------------------------------------------------------
def test_adjoint_S_float(n=64, ntheta=4):
    cl = ShiftFFT(n=n, npsi=n, nz=n, nzpsi=n, obj_dtype='float32')
    rng = cp.random.default_rng(17)
    r = (rng.random([ntheta, 2], dtype='float32') - 0.5) * 20
    m = cp.ones(ntheta, dtype='float32')
    c = rng.standard_normal([ntheta, n, n], dtype='float32')
    b = rng.standard_normal([ntheta, n, n], dtype='float32')
    Sc    = cl.S(c, r, m)
    Sadjb = cl.Sadj(b, r, m)
    assert Sc.dtype == cp.float32
    assert Sadjb.dtype == cp.float32
    lhs = float(cp.sum(Sc * b))
    rhs = float(cp.sum(c * Sadjb))
    rel = abs(lhs - rhs) / (abs(lhs) + 1e-30)
    print(f"[7] S/Sadj adjoint (float32):  lhs={lhs:.6e} rhs={rhs:.6e} rel={rel:.3e}")
    assert rel < 1e-3


if __name__ == '__main__':
    test_fft_vs_cubic_interior()
    test_adjoint_S()
    test_adjoint_dcurlyS()
    test_fd_dcurlySc_r()
    test_fd_d2curlySc()
    test_adjoint_S_padded()
    test_adjoint_S_float()
    print("\nAll FFT-mode tests passed.")
