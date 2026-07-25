"""Taylor approximation test for the composed shift operator curlySc.

Same style as test_approximation.py (F0..F3), but targets the primitive
Shift operator in isolation rather than through the Rec cascade.

For f(c, r) = curlySc(c, r, m) (with m fixed):
    e1 = ||f(c+dc, r+dr) - f(c, r)||                          ~ O(h)
    e2 = ||f(c+dc, r+dr) - f(c, r) - df(c, r; dc, dr)||       ~ O(h^2)
    e3 = ||f(c+dc, r+dr) - f(c, r) - df - 1/2 d2f(...)||      ~ O(h^3)

Expected: e2/e1 -> 4, e3/e2 -> 8 when |dx| is halved.

Notes
-----
coeff() is linear in psi, so we perturb the *coefficients* c directly by
`l * dc` and (via linearity) equivalently `curlySc` sees `coeff(psi + l*dpsi)`.
The composite first derivative dcurlySc(c, r, m, dc, dr) is the directional
derivative in the direction (dc, dr); d2curlySc(c, r, m, dc, dr, dc, dr) is
the quadratic form on the same direction.
"""

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

from holotomocupy.shift import Shift


# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
n       = 128           # data-grid size (output of curlySc)
nz      = n
npsi    = n             # input/coeff grid size (nobj in Rec)
nzpsi   = nz
ntheta  = 32            # angles per distance
dtype   = 'complex64'   # matches rec_mpi.obj_dtype default

cl_shift = Shift(n=n, npsi=npsi, nz=nz, nzpsi=nzpsi, obj_dtype=dtype)


# ----------------------------------------------------------------------------
# Random inputs — same seed as test_approximation.py so runs are reproducible.
# ----------------------------------------------------------------------------
rng = np.random.default_rng(42)


def rc(shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype('complex64')


def rf(shape, scale=1.0):
    return (rng.standard_normal(shape) * scale).astype('float32')


# psi lives on the (nzpsi, npsi) grid; r is a per-projection (y, x) shift;
# m is a per-projection (y, x) magnification. Match the (chunk, 2) layout
# demagnifications[:, dist_idx] uses in F3.
psi_g  = cp.array(rc((ntheta, nzpsi, npsi)))
r_g    = cp.array(rf((ntheta, 2), 0.5))

# Shrink grows sqrt-like within each distance and continues cumulatively into
# the next distance (matches the shrink_list convention on real data).
# Y and X have different per-distance increments; each successive distance is
# slower than the previous (decay factor `dist_decay` per distance).
ndist        = 4
dist_idx     = 0
axis_inc     = np.array([0.020, 0.012], dtype='float32')  # per-dist end incr. (y, x)
dist_decay   = 0.7                                        # dist k inc = axis_inc * dist_decay**k
inc_per_dist = axis_inc[None, :] * (dist_decay ** np.arange(ndist)[:, None])  # (ndist, 2)
end_per_dist = np.cumsum(inc_per_dist, axis=0)             # (ndist, 2), end shrink per (k, axis)
start_per_dist = np.vstack(
    [np.zeros((1, 2), dtype='float32'), end_per_dist[:-1]]
)                                                          # (ndist, 2)

sqrt_ramp = np.sqrt(np.linspace(0, 1, ntheta, dtype='float32'))  # 0 -> 1 sqrt shape
shrink_nd = np.zeros((ntheta, ndist, 2), dtype='float32')
for k in range(ndist):
    shrink_nd[:, k, :] = start_per_dist[k][None, :] + \
                         (end_per_dist[k] - start_per_dist[k])[None, :] * sqrt_ramp[:, None]
shrink_nd = cp.asarray(shrink_nd)

# For a single-distance shift test, m = (1 + shrink) / norm_mag, with
# norm_mag=1 (no relative demagnification between distances here).
m_g = (1.0 + shrink_nd[:, dist_idx, :])   # (ntheta, 2)

# perturbation directions
dpsi_g = cp.array(rc((ntheta, nzpsi, npsi)))
dr_g   = cp.array(rf((ntheta, 2), 0.5))

# coeff() is linear, so we pre-compute coefficients of psi and dpsi and step
# c by l * c_dpsi to match `coeff(psi + l * dpsi)`.
c_g       = cl_shift.coeff(psi_g)
c_dpsi_g  = cl_shift.coeff(dpsi_g)

L = np.linspace(0, 0.1, 20, dtype='float32')


# ----------------------------------------------------------------------------
# Taylor loop
# ----------------------------------------------------------------------------
err1 = np.zeros(len(L))
err2 = np.zeros(len(L))
err3 = np.zeros(len(L))

f_w = cl_shift.curlySc(c_g, r_g, m_g)
for k, l in enumerate(L):
    dc_k = l * c_dpsi_g
    dr_k = l * dr_g

    a   = cl_shift.curlySc(c_g + dc_k, r_g + dr_k, m_g)
    df  = cl_shift.dcurlySc(c_g,  r_g, m_g, dc_k, dr_k)
    d2f = cl_shift.d2curlySc(c_g, r_g, m_g, dc_k, dr_k, dc_k, dr_k)

    err1[k] = float(cp.linalg.norm(f_w - a))
    err2[k] = float(cp.linalg.norm(f_w + df - a))
    err3[k] = float(cp.linalg.norm(f_w + df + 0.5 * d2f - a))


# ----------------------------------------------------------------------------
# Summary + plot
# ----------------------------------------------------------------------------
print('\n--- Shift: curlySc(c, r, m) ---')
print('e2/e1 ratios (expect ~4):', np.round(err1[10:15] / err2[10:15], 2))
print('e3/e2 ratios (expect ~8):', np.round(err2[10:15] / err3[10:15], 2))

fig, ax = plt.subplots(figsize=(7, 5))
ax.semilogy(L, err1, label='O(h)')
ax.semilogy(L, err2, label='O(h^2)')
ax.semilogy(L, err3, label='O(h^3)')
ax.set_title(r'Shift approximation: $\mathcal{S}_c(c, r, m)$')
ax.set_xlabel('|dx|')
ax.set_ylabel('error')
ax.legend()
ax.grid(True)
fig.tight_layout()
out_png = 'test_approximation_shift.png'
fig.savefig(out_png, dpi=110)
print(f'\nSaved figure: {out_png}')

# Plot the shrinkage as a sanity check: one line per distance, per axis.
_sh_np    = cp.asnumpy(shrink_nd)
_fig_sh, _ax_sh = plt.subplots(figsize=(7, 4))
_theta_ax = np.arange(ntheta)
for _k in range(ndist):
    _ax_sh.plot(_theta_ax, _sh_np[:, _k, 0], label=f'dist {_k}, y', linestyle='-')
    _ax_sh.plot(_theta_ax, _sh_np[:, _k, 1], label=f'dist {_k}, x', linestyle='--')
_ax_sh.set_xlabel('projection index')
_ax_sh.set_ylabel('shrink_nd')
_ax_sh.set_title('cumulative sqrt-ramp shrinkage per distance')
_ax_sh.grid(True)
_ax_sh.legend(fontsize=8)
_fig_sh.tight_layout()
_fig_sh.savefig('test_approximation_shift_shrink.png', dpi=110)
print('Saved figure: test_approximation_shift_shrink.png')
