"""Taylor approximation tests: F0 – F3 in the Rec cascade.

Verifies the second-order Taylor expansion for each functional:
    f(x + dx) = f(x) + df(x, dx) + 1/2 d²f(x, dx, dx) + O(|dx|³)

Expected: e2/e1 ratio ≈ 4, e3/e2 ratio ≈ 8 when |dx| is halved.

| Functional | Nonlinear component tested                            | Note          |
|------------|-------------------------------------------------------|---------------|
| F3         | x22 = S_x33(x32) — B-spline shift of proj by pos      | e2 ~ O(h³)    |
| F2         | x12 = exp(i·x22) — phase encoding                     | e2 ~ O(h³)    |
| F1         | x0  = D(x11·x12) — bilinear                           | e3 ≈ 0        |
| F0         | (1/N) ||x0| - d||²                                    | e2 ~ O(h³)    |

Saves a single 4-panel figure (one subplot per Fk) instead of pop-up windows.
"""

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from mpi4py import MPI
from types import SimpleNamespace

from holotomocupy.rec_mpi import Rec


# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
n      = 128
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
    obj_dtype               = 'complex64',
    mask                    = 0.9,
    lam_prbfit              = 0.0,
    rho                     = [1, 0.05, 0.02],
    niter                   = 1,
    nchunk                  = ntheta,
    vis_step                = -1,
    err_step                = -1,
    start_iter              = 0,
    comm                    = MPI.COMM_WORLD,
)

cl = Rec(args)

# F3/dF3/d2F_dF3 read self._eff_demag_chunk, which is normally set inside the
# batch loops in BH() / gen_sqrt_data(). Mirror that setup once here so the
# functionals can be called standalone.
cl.eff_demagnifications[:] = (1 + cl.shrink_nd) / cp.array(cl.norm_magnifications[None, :])
cl._eff_demag_chunk = cl.eff_demagnifications


# ----------------------------------------------------------------------------
# Random test inputs (same RNG seed → same arrays as the notebook)
# ----------------------------------------------------------------------------
rng = np.random.default_rng(42)


def rc(shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype('complex64')


def rf(shape, scale=1.0):
    return (rng.standard_normal(shape) * scale).astype('float32')


prb_g   = cp.array(rc((ndist, nz, n)))
proj_g  = cp.array(rf((ntheta, nzobj, nobj), 0.1))
pos_g   = cp.array(rf((ntheta, ndist, 2), 0.5))
data_g  = cp.array(np.abs(rf((ntheta, ndist, nz, n))) + 0.1).astype('float32')

dprb_g  = cp.array(rc((ndist, nz, n)))
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
x3    = [prb_g, proj_g, pos_g]
dw0_3 = [dprb_g, dproj_g, dpos_g]

err1, err2, err3 = np.zeros(20), np.zeros(20), np.zeros(20)
f_w = cl.F3(x3)[1]
for k, l in enumerate(L):
    dx3 = [l * dv for dv in dw0_3]
    a   = cl.F3([v + dv for v, dv in zip(x3, dx3)])[1]
    df  = cl.dF3(x3, dx3, return_x=False)[1]
    d2f = cl.d2F_dF3(x3, dx3, dx3, [None] * 3)[1]
    err1[k] = float(cp.linalg.norm(f_w - a))
    err2[k] = float(cp.linalg.norm(f_w + df - a))
    err3[k] = float(cp.linalg.norm(f_w + df + 0.5 * d2f - a))

_print_summary('F3: S_pos(proj)', err1, err2, err3)
_plot(axs[0, 0], r'F3: $\mathcal{S}_{pos}(proj)$', L, err1, err2, err3)


# ----------------------------------------------------------------------------
# F2: (prb, shifted_proj) → (prb, exp(i·shifted_proj))
# ----------------------------------------------------------------------------
x22_f2  = cp.array(rf((ntheta, ndist, nz, n), 0.1))
dx22_f2 = cp.array(rf((ntheta, ndist, nz, n), 0.1))

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
x11_f1  = cp.array(rc((ndist, nz, n)))
x12_f1  = cp.array(rc((ntheta, ndist, nz, n)))
dx11_f1 = cp.array(rc((ndist, nz, n)))
dx12_f1 = cp.array(rc((ntheta, ndist, nz, n)) * 0.1)

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
x0_f0  = cp.array(rc((ntheta, ndist, nz, n)))
dx0_f0 = cp.array(rc((ntheta, ndist, nz, n)) * 0.1)

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
fig.suptitle('Taylor approximation tests — F0..F3', fontsize=14)
fig.tight_layout()
out_png = 'test_approximation.png'
fig.savefig(out_png, dpi=110)
print(f'\nSaved figure: {out_png}')
