"""Taylor approximation tests: F0 - F4 in the Rec cascade.

Verifies the second-order Taylor expansion for each functional:
    f(x + dx) = f(x) + df(x, dx) + 1/2 d2f(x, dx, dx) + O(|dx|^3)

Expected: e2/e1 ratio ~ 4, e3/e2 ratio ~ 8 when |dx| is halved.

| Functional | Nonlinear component tested                       | Note        |
|------------|--------------------------------------------------|-------------|
| F4         | x34 = demag(tp) -- affine, so e2 ~ 0             | e2 ~ 0      |
| F3         | x22 = S_{x33,x34}(x32) -- B-spline shift         | e2 ~ O(h^3) |
| F2         | x12 = exp(i*x22) -- phase encoding               | e2 ~ O(h^3) |
| F1         | x0  = D(x11*x12) -- bilinear                     | e3 ~ 0      |
| F0         | (1/N) ||x0| - d||^2                              | e2 ~ O(h^3) |

F3 additionally gets a polarization check: the diagonal Taylor test above uses
d2f(x, dx, dx) and so cannot see which argument slot each coefficient is
contracted with, while B(y+z, y+z) = B(y,y) + 2B(y,z) + B(z,z) can.

Saves a single 4-panel figure (one subplot per Fk) instead of pop-up windows.
"""

import os

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from mpi4py import MPI
from types import SimpleNamespace

from holotomocupy.rec_mpi import Rec


# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
# cuFFTDx JIT needs cufft.h, which is missing in some installs; there only
# the prebuilt 512/1024/2048/4096 kernels load, i.e. n in {256, 512, 1024,
# 2048}. Run `N=256 python test_approximation.py` on such a box.
n      = int(os.environ.get('N', 128))
ntheta = 4
ndist  = 2
nz     = n
nzobj  = n
nobj   = n

args = SimpleNamespace(
    energy                  = 17.1,
    detector_pixelsize      = 1.4760147601476e-6 * 16,
    focustodetectordistance = 1.217,
    z1                      = np.array([5.110, 5.464]) * 1e-3,
    theta                   = np.linspace(0, np.pi, ntheta, dtype='float32'),
    ndist                   = ndist,
    ntheta                  = ntheta,
    nz                      = nz,
    n                       = n,
    nzobj                   = nzobj,
    nobj                    = nobj,
    mask                    = 0.9,
    lam_prbfit              = 0.0,
    rho                     = [1, 0.05, 0.02, 1e-4],
    niter                   = 1,
    nchunk                  = ntheta,
    vis_step                = -1,
    err_step                = -1,
    start_iter              = 0,
    comm                    = MPI.COMM_WORLD,
)

cl = Rec(args)

# F4 reads self._t_chunk and self._dist_idx, normally set inside the batch loops
# in BH() / gen_sqrt_data(). Mirror that setup once here so the functionals can
# be called standalone. The whole (single-rank) theta range is one chunk.
cl._dist_idx = 0
cl._t_chunk  = cl.t_local              # (ntheta, 1)


# ----------------------------------------------------------------------------
# Random test inputs (same RNG seed → same arrays as the notebook)
# ----------------------------------------------------------------------------
rng = np.random.default_rng(42)


def rc(shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype('complex64')


def rf(shape, scale=1.0):
    return (rng.standard_normal(shape) * scale).astype('float32')


prb_g   = cp.array(rc((nz, n)))            # one distance: F1..F3 are per-dist
proj_g  = cp.array(rf((ntheta, nzobj, nobj), 0.1))
pos_g   = cp.array(rf((ntheta, ndist, 2), 0.5))
data_g  = cp.array(np.abs(rf((ntheta, nz, n))) + 0.1).astype('float32')

dprb_g  = cp.array(rc((nz, n)))
dproj_g = cp.array(rf((ntheta, nzobj, nobj), 0.1))
dpos_g  = cp.array(rf((ntheta, ndist, 2), 0.5))

L = np.linspace(0, 0.1, 20, dtype='float32')


# ----------------------------------------------------------------------------
# Helper: print expected ratio summary and return arrays for plotting
# ----------------------------------------------------------------------------
def _print_summary(name, err1, err2, err3, bilinear=False):
    print(f'\n--- {name} ---')
    print('e2/e1 ratios (expect ~4):', np.round(err1[10:15] / err2[10:15], 2))
    if bilinear:
        print('e3 (expect ~0):         ', np.round(err3[10:15], 6))
    else:
        print('e3/e2 ratios (expect ~8):', np.round(err2[10:15] / err3[10:15], 2))


def _plot(ax, title, L, err1, err2, err3, bilinear=False):
    ax.semilogy(L,    err1, label='O(h)')
    ax.semilogy(L,    err2, label='O(h²)')
    if bilinear:
        ax.semilogy(L[1:], err3[1:], label='O(h³) ~ 0')
    else:
        ax.semilogy(L,    err3, label='O(h³)')
    ax.set_title(title)
    ax.set_xlabel('|dx|')
    ax.set_ylabel('error')
    ax.legend()
    ax.grid(True)


fig, axs = plt.subplots(2, 2, figsize=(12, 9))


# ----------------------------------------------------------------------------
# F3: (prb, proj, pos) → (prb, S_pos(proj))
# ----------------------------------------------------------------------------
# x34 is the effective demagnification, now a differentiable input of F3; the
# perturbation direction dmag_g is what exercises the *m* half of the kernels.
mag_g  = cp.array(rf((ntheta, 2), 0.02)) + cp.asarray(
    1.0 / cl.norm_magnifications_gpu[cl._dist_idx])
dmag_g = cp.array(rf((ntheta, 2), 0.02))

x3    = [prb_g, proj_g, pos_g, mag_g]
dw0_3 = [dprb_g, dproj_g, dpos_g, dmag_g]

err1, err2, err3 = np.zeros(20), np.zeros(20), np.zeros(20)
f_w = cl.F3(x3)[1]
for k, l in enumerate(L):
    # coeff_cached is keyed by id(); the perturbed projections below are freed
    # and reallocated every iteration, so CuPy hands out the same ids again and
    # a stale entry would silently feed the unperturbed coefficients back in.
    # BH()/gen_sqrt_data() reset per chunk for the same reason.
    cl.cl_shift.coeff_cache_reset()
    dx3 = [l * dv for dv in dw0_3]
    a   = cl.F3([v + dv for v, dv in zip(x3, dx3)])[1]
    df  = cl.dF3(x3, dx3, return_x=False)[1]
    d2f = cl.d2F_dF3(x3, dx3, dx3, [None] * 4)[1]
    err1[k] = float(cp.linalg.norm(f_w - a))
    err2[k] = float(cp.linalg.norm(f_w + df - a))
    err3[k] = float(cp.linalg.norm(f_w + df + 0.5 * d2f - a))

_print_summary('F3: S_{pos,mag}(proj)', err1, err2, err3)
# F3 is the level the shrinkage variable added a direction to (x34 = demag),
# so check the actual convergence RATES, not just the magnitudes: a log-log
# slope of 1 / 2 / 3 is what makes dcurlySmc and d2curlySmc the first and
# second differentials in (proj, pos, demag) jointly.
_sl = slice(4, 12)
_slope = lambda e: float(np.polyfit(np.log(L[_sl]), np.log(e[_sl]), 1)[0])
_s1, _s2, _s3 = _slope(err1), _slope(err2), _slope(err3)
print(f'log-log slopes: O(h) {_s1:.2f} (want 1)  O(h^2) {_s2:.2f} (want 2)  '
      f'O(h^3) {_s3:.2f} (want 3)')
assert _s2 > 1.8, f'dF3 is not the first differential (slope {_s2:.2f})'
assert _s3 > 2.6, f'd2F_dF3 is not the second differential (slope {_s3:.2f})'
_plot(axs[0, 0], r'F3: $\mathcal{S}_{pos,mag}(proj)$', L, err1, err2, err3)

# The Taylor loop above only ever evaluates d2F_dF3 on the diagonal (dx, dx),
# where crossing the two coefficient slots is invisible. The polarization
# identity is the test that sees it -- see the slot-pairing note in
# Shift.d2curlySc.
ydir = [0.010 * v for v in dw0_3]
zdir = [0.013 * cp.array(rc(v.shape) if v.dtype.kind == 'c' else rf(v.shape, 0.02))
        for v in dw0_3]


def _bilin(u, v):
    cl.cl_shift.coeff_cache_reset()
    return cl.d2F_dF3(x3, u, v, [None] * 4)[1]


_uv = [a + b for a, b in zip(ydir, zdir)]
_lhs = _bilin(_uv, _uv)
_rhs = _bilin(ydir, ydir) + 2 * _bilin(ydir, zdir) + _bilin(zdir, zdir)
_e_pol = float(cp.linalg.norm(_lhs - _rhs) / cp.linalg.norm(_lhs))
print(f'polarization B(y+z,y+z) vs B(y,y)+2B(y,z)+B(z,z): rel={_e_pol:.3e}')
assert _e_pol < 1e-4, f'd2F_dF3 is not the bilinear form of the Hessian ' \
                      f'(rel={_e_pol:.3e}); check the c1/c2 slot pairing'



# ----------------------------------------------------------------------------
# F4: tp = [[A_y, A_x], [B_y, B_x]] -> demag(theta, axis)  (affine: e2 at
# machine precision, so the second-order column is the one to read)
# ----------------------------------------------------------------------------
tp_g  = cp.array(rf((2, 2), 1e-3))
dtp_g = cp.array(rf((2, 2), 1e-3))

x4    = [prb_g, proj_g, pos_g, tp_g]
dw0_4 = [dprb_g, dproj_g, dpos_g, dtp_g]

err1, err2, err3 = np.zeros(20), np.zeros(20), np.zeros(20)
f_w = cl.F4(x4)[3]
for k, l in enumerate(L):
    dx4 = [l * dv for dv in dw0_4]
    a   = cl.F4([v + dv for v, dv in zip(x4, dx4)])[3]
    df  = cl.dF4(x4, dx4, return_x=False)[3]
    d2f = cl.d2F_dF4(x4, dx4, dx4, [None] * 4)[3]
    d2f = 0.0 if d2f is None else d2f
    err1[k] = float(cp.linalg.norm(f_w - a))
    err2[k] = float(cp.linalg.norm(f_w + df - a))
    err3[k] = float(cp.linalg.norm(f_w + df + 0.5 * d2f - a))

print('\n--- F4: demag(tp) [affine] ---')
print('e1 (expect ~ l):', np.round(err1[10:15], 8))
print('e2 (expect ~0): ', np.round(err2[10:15], 12))
# demag is O(1), so the affine residual bottoms out at float32 round-off on
# f_w; a bound relative to err1 (itself only ~1e-4 here) would sit below eps.
_tol4 = 1e-5 * float(cp.linalg.norm(f_w))
assert err2[1:].max() < _tol4, \
    f'F4 is not affine in tp ({err2[1:].max():.3e} >= {_tol4:.3e})'


# ----------------------------------------------------------------------------
# F2: (prb, shifted_proj) → (prb, exp(i·shifted_proj))
# ----------------------------------------------------------------------------
x22_f2  = cp.array(rf((ntheta, nz, n), 0.1))
dx22_f2 = cp.array(rf((ntheta, nz, n), 0.1))

x2    = [prb_g, x22_f2]
dw0_2 = [dprb_g, dx22_f2]

err1, err2, err3 = np.zeros(20), np.zeros(20), np.zeros(20)
f_w = cl.F2(x2)[1]
for k, l in enumerate(L):
    dx2 = [l * dv for dv in dw0_2]
    a   = cl.F2([v + dv for v, dv in zip(x2, dx2)])[1]
    df  = cl.dF2(x2, dx2, return_x=False)[1]
    d2f = cl.d2F_dF2(x2, dx2, dx2, [None, None])[1]
    err1[k] = float(cp.linalg.norm(f_w - a))
    err2[k] = float(cp.linalg.norm(f_w + df - a))
    err3[k] = float(cp.linalg.norm(f_w + df + 0.5 * d2f - a))

_print_summary('F2: exp(i·x22)', err1, err2, err3)
_plot(axs[0, 1], r'F2: $e^{i\cdot x_{22}}$', L, err1, err2, err3)


# ----------------------------------------------------------------------------
# F1: (prb, exp_proj) → D(prb · exp_proj)  (bilinear: e3 at machine precision)
# ----------------------------------------------------------------------------
x11_f1  = cp.array(rc((nz, n)))
x12_f1  = cp.array(rc((ntheta, nz, n)))
dx11_f1 = cp.array(rc((nz, n)))
dx12_f1 = cp.array(rc((ntheta, nz, n)) * 0.1)

x1    = [x11_f1, x12_f1]
dw0_1 = [dx11_f1, dx12_f1]

err1, err2, err3 = np.zeros(20), np.zeros(20), np.zeros(20)
f_w = cl.F1(x1)
for k, l in enumerate(L):
    dx1 = [l * dv for dv in dw0_1]
    a   = cl.F1([v + dv for v, dv in zip(x1, dx1)])
    df  = cl.dF1(x1, dx1, return_x=False)
    d2f = cl.d2F_dF1(x1, dx1, dx1, [None, None])
    err1[k] = float(cp.linalg.norm(f_w - a))
    err2[k] = float(cp.linalg.norm(f_w + df - a))
    err3[k] = float(cp.linalg.norm(f_w + df + 0.5 * d2f - a))

_print_summary('F1: D(prb · exp_proj) [bilinear]', err1, err2, err3, bilinear=True)
_plot(axs[1, 0], r'F1: $D(prb \cdot exp\_proj)$ (bilinear: $e_3 \approx 0$)',
      L, err1, err2, err3, bilinear=True)


# ----------------------------------------------------------------------------
# F0: (1/N) || |x0| - d ||²
# ----------------------------------------------------------------------------
# +0.5 keeps |x0| away from zero, as the real x0 is: F0 differentiates
# |x|, whose curvature is 1/|x|, so a few near-zero pixels would otherwise
# dominate the residual and hide the actual Taylor behaviour.
x0_f0  = cp.array(rc((ntheta, nz, n))) + 0.5
dx0_f0 = cp.array(rc((ntheta, nz, n)) * 0.1)

err1, err2, err3 = np.zeros(20), np.zeros(20), np.zeros(20)
f_w = float(cl.F0(x0_f0, data_g))
for k, l in enumerate(L):
    dx  = l * dx0_f0
    a   = float(cl.F0(x0_f0 + dx, data_g))
    df  = float(cl.dF0(x0_f0, dx, data_g))
    d2f = float(cl.d2F_dF0(x0_f0, dx, dx, None, data_g))
    err1[k] = abs(f_w - a)
    err2[k] = abs(f_w + df - a)
    err3[k] = abs(f_w + df + 0.5 * d2f - a)

_print_summary('F0: ||x| - d||²', err1, err2, err3)
_plot(axs[1, 1], r'F0: $\||x| - d\|_2^2$', L, err1, err2, err3)


# ----------------------------------------------------------------------------
# Save combined figure
# ----------------------------------------------------------------------------
fig.suptitle('Taylor approximation tests — F0..F4', fontsize=14)
fig.tight_layout()
out_png = 'test_approximation.png'
fig.savefig(out_png, dpi=110)
print(f'\nSaved figure: {out_png}')
