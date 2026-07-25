"""Standalone F3_new / dF3_new / d2F_dF3_new — like Rec.F3 but with
demag (m) as a fourth variable. Includes a Taylor approximation test
in the same style as test_approximation.py.

Motivation
----------
Current Rec.F3 treats the per-projection magnification m = demag as a
FIXED PARAMETER (read from self._demag_chunk). We want to make m a
VARIABLE that can be optimized alongside (prb, proj, pos). That means F3
becomes a function of 4 inputs and dF3 / d2F_dF3 have to include the m
direction.

Math
----
Let s(c, r, m) := curlySc(c, r, m). The B-spline evaluation samples at
    sample_y(ty) = m_y · (ty - (nz-1)/2) - r_y + (nzpsi-1)/2
so ∂sample_y/∂r_y = -1 and ∂sample_y/∂m_y = ty - (nz-1)/2 =: t_y(ty).
Hence every derivative in m reduces to a weighted derivative in r:

    ∂/∂m_axis    s = -t_axis * ∂/∂r_axis s
    ∂²/∂m_a ∂m_b s =  t_a t_b * ∂²/∂r_a ∂r_b s
    ∂²/∂r_a ∂m_b s = -t_b     * ∂²/∂r_a ∂r_b s

These identities let us reuse the existing dcurlySc / d2curlySc kernels
(which already provide the r-direction derivatives) without adding new
CUDA code. The test below confirms the resulting Taylor expansion of
F3_new works out to the expected O(h) / O(h²) / O(h³) ratios (4 and 8).
"""

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

from holotomocupy.shift import Shift


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
n      = 128
nz     = n
npsi   = n
nzpsi  = nz
ntheta = 4
dtype  = 'complex64'

cl_shift = Shift(n=n, npsi=npsi, nz=nz, nzpsi=nzpsi, obj_dtype=dtype)


# ---------------------------------------------------------------------------
# F3_new / dF3_new / d2F_dF3_new — same layout as Rec.F3, one extra input
# ---------------------------------------------------------------------------
def F3_new(x):
    """x = [prb, proj, pos, ed]. Returns [prb, curlySc(coeff(proj), pos, ed)]."""
    prb, proj, pos, ed = x
    c = cl_shift.coeff(proj)
    return [prb, cl_shift.curlySc(c, pos, ed)]


def dF3_new(x, y):
    """Directional derivative in y = [dprb, dproj, dpos, ded]. Returns [dprb, ...]."""
    _, proj, pos, ed = x
    dprb, dproj, dpos, ded = y
    c  = cl_shift.coeff(proj)
    dc = cl_shift.coeff(dproj)
    # single-kernel (c, r, m) directional derivative — folds Deltam into a
    # per-pixel effective Deltar inside the CUDA kernel.
    dS = cl_shift.dcurlySmc(c, pos, ed, dc, dpos, ded)
    return [dprb, dS]


def d2F_dF3_new(x, y, z):
    """Bilinear 2nd derivative on directions (y, z). Returns [None, ...].

    Two pieces, each a single kernel call:
      d_rrmm = d2curlySmc(c, pos, ed, 0, dpos_y, ded_y, 0, dpos_z, ded_z)
               → (r,r) + (m,m) + (r,m) — no c-mixed since c1 = c2 = 0.
      d_c_mixed = dcurlySmc(dcy, ..., dpos_z, ded_z) + dcurlySmc(dcz, ..., dpos_y, ded_y)
               → (c,r) + (c,m) for both direction orderings.
    The (c,c) part is 0 because curlySc is linear in c.
    """
    _, proj, pos, ed = x
    _dprb_y, dproj_y, dpos_y, ded_y = y
    _dprb_z, dproj_z, dpos_z, ded_z = z
    c   = cl_shift.coeff(proj)
    dcy = cl_shift.coeff(dproj_y)
    dcz = cl_shift.coeff(dproj_z)
    zero_c = cp.zeros_like(c)

    d_rrmm    = cl_shift.d2curlySmc(c, pos, ed,
                                    zero_c, dpos_y, ded_y,
                                    zero_c, dpos_z, ded_z)
    d_c_mixed = (cl_shift.dcurlySmc(dcy, pos, ed, zero_c, dpos_z, ded_z)
               + cl_shift.dcurlySmc(dcz, pos, ed, zero_c, dpos_y, ded_y))

    return [None, d_rrmm + d_c_mixed]


# ---------------------------------------------------------------------------
# Random inputs — same RNG seed as test_approximation.py for reproducibility
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)


def rc(shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype('complex64')


def rf(shape, scale=1.0):
    return (rng.standard_normal(shape) * scale).astype('float32')


prb_g   = cp.array(rc((1, nz, n)))                                # passthrough (unused in norm)
proj_g  = cp.array(rf((ntheta, nzpsi, npsi), 0.1)).astype('complex64')
pos_g   = cp.array(rf((ntheta, 2), 0.5))
ed_g    = cp.array(rf((ntheta, 2), 0.02) + 1.0)                   # m near 1

# perturbation directions (mirror layout)
dprb_g  = cp.array(rc((1, nz, n)))
dproj_g = cp.array(rf((ntheta, nzpsi, npsi), 0.1)).astype('complex64')
dpos_g  = cp.array(rf((ntheta, 2), 0.5))
ded_g   = cp.array(rf((ntheta, 2), 0.02))

x_g     = [prb_g,  proj_g,  pos_g,  ed_g ]
dw0_g   = [dprb_g, dproj_g, dpos_g, ded_g]

L = np.linspace(0, 0.1, 20, dtype='float32')


# ---------------------------------------------------------------------------
# Taylor loop
# ---------------------------------------------------------------------------
err1 = np.zeros(len(L))
err2 = np.zeros(len(L))
err3 = np.zeros(len(L))

f_w = F3_new(x_g)[1]
for k, l in enumerate(L):
    dx = [l * dv for dv in dw0_g]
    a  = F3_new([v + dv for v, dv in zip(x_g, dx)])[1]
    df = dF3_new(x_g, dx)[1]
    d2 = d2F_dF3_new(x_g, dx, dx)[1]
    err1[k] = float(cp.linalg.norm(f_w - a))
    err2[k] = float(cp.linalg.norm(f_w + df - a))
    err3[k] = float(cp.linalg.norm(f_w + df + 0.5 * d2 - a))

print('\n--- F3_new: curlySc(coeff(proj), pos, ed) with ed as variable ---')
print('e2/e1 ratios (expect ~4):', np.round(err1[10:15] / err2[10:15], 2))
print('e3/e2 ratios (expect ~8):', np.round(err2[10:15] / err3[10:15], 2))


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
ax.semilogy(L, err1, label='O(h)')
ax.semilogy(L, err2, label='O(h^2)')
ax.semilogy(L, err3, label='O(h^3)')
ax.set_title(r'F3_new Taylor test (prb, proj, pos, ed)')
ax.set_xlabel('|dx|')
ax.set_ylabel('error')
ax.legend()
ax.grid(True)
fig.tight_layout()
out_png = 'test_approximation_F3_new.png'
fig.savefig(out_png, dpi=110)
print(f'\nSaved figure: {out_png}')


# ---------------------------------------------------------------------------
# Adjoint test for dcurlySmc <-> dcurlySadjmc
# ---------------------------------------------------------------------------
# For the linear operator A = dcurlySmc(c, r, m, .) mapping (c1, Deltar, Deltam)
# to an output field g_shape, the adjoint identity is
#     <A(c1, dr, dm), g>   ==   <c1, out1> + <dr, out2_r> + <dm, out2_m>
# where (out1, out2_r, out2_m) = dcurlySadjmc(c, r, m, g). We verify this
# holds numerically with a random input and random g.
from holotomocupy.utils import redot

# Fresh direction inputs and a random output-space vector g.
c_g   = cl_shift.coeff(proj_g)
c1_g  = cl_shift.coeff(dproj_g)
dr_g  = cp.array(rf((ntheta, 2), 0.5))
dm_g  = cp.array(rf((ntheta, 2), 0.02))
g_out = cp.array(rc((ntheta, nz, n)))

# Forward direction: dcurlySmc as a LINEAR map of (c1, dr, dm).
fwd = cl_shift.dcurlySmc(c_g, pos_g, ed_g, c1_g, dr_g, dm_g)

# Adjoint outputs at the same base point.
adj_c1, adj_dr, adj_dm = cl_shift.dcurlySadjmc(c_g, pos_g, ed_g, g_out)

# LHS: <A(c1, dr, dm), g>  (real inner product on complex64, via redot).
lhs = float(redot(fwd, g_out))

# RHS: <c1, adj_c1> (complex real-inner product) + <dr, adj_dr> + <dm, adj_dm>.
rhs = float(redot(c1_g, adj_c1)) + float(cp.sum(dr_g * adj_dr)) + float(cp.sum(dm_g * adj_dm))

rel_err = abs(lhs - rhs) / max(abs(lhs), abs(rhs))
print('\n--- Adjoint test for dcurlySmc <-> dcurlySadjmc ---')
print(f'  <A(c1, dr, dm), g>          = {lhs: .6e}')
print(f'  <c1, adj_c1> + <dr, adj_dr> + <dm, adj_dm> = {rhs: .6e}')
print(f'  |LHS - RHS| / max(|LHS|,|RHS|) = {rel_err:.2e}   (expect < 1e-5)')
