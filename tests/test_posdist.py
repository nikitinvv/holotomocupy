"""Cross-check rec_mpi_shrink_posdist.Rec against rec_mpi_shrink.Rec.

The posdist variant refines one (y, x) shift per (tile x distance) instead of
one per projection, holding the measured per-angle shifts fixed in
`cl.pos_base`. Since  r[theta, k] = pos_base[theta, k] + pos[k]  is affine in
pos with unit Jacobian, everything must line up with the parent exactly:

  1. functional     min_new(pos_base=P, pos=q)  ==  min_parent(pos=P + q)
  2. gradient       grad_new['pos'][k]          ==  sum_theta grad_parent['pos'][:,k]
                    grad_new[obj/prb/tp/proj]   ==  grad_parent[...]
  3. Hessian        H_new(eta_pos=v)            ==  H_parent(eta_pos=broadcast v)
  4. finite diff    grad_new['pos']             ==  d min_new / d pos   (central diff)

Run (single rank is enough; ndist must differ from local_ntheta):

    python tests/test_posdist.py [cubic|fft]
"""

import sys
import numpy as np
import cupy as cp
from types import SimpleNamespace
from mpi4py import MPI

from holotomocupy.rec_mpi_shrink import Rec as RecParent
from holotomocupy.rec_mpi_shrink_posdist import Rec as RecPosDist
from holotomocupy.logger_config import set_log_level

set_log_level('ERROR')

SHIFT_TYPE = sys.argv[1] if len(sys.argv) > 1 else 'cubic'

# --- tiny problem --------------------------------------------------------
n, nz   = 32, 32
ntheta  = 12
ndist   = 3          # != local_ntheta, as the posdist class requires
nobj    = 48
rng     = np.random.default_rng(0)


def make_args():
    a = SimpleNamespace()
    a.energy                  = 17.1
    a.detector_pixelsize      = 1.4760147601476e-6 * 2 * 8
    a.focustodetectordistance = 1.217
    a.z1     = np.array([5.110, 5.464, 6.879])[:ndist] * 1e-3
    a.theta  = np.linspace(0, np.pi, ntheta, endpoint=False).astype('float32')
    a.ndist, a.ntheta = ndist, ntheta
    a.nz, a.n         = nz, n
    a.nzobj, a.nobj   = nobj, nobj
    a.obj_dtype       = 'float32'
    a.mask            = 0.9
    a.lam_prbfit      = 0.0        # keeps the comparison to the pos machinery
    a.lam_laplacian   = 0.0
    a.rho             = [1.0, 1.0, 1.0, 1.0]   # no scaling: grads are raw
    a.niter           = 1
    a.nchunk          = 5          # < ntheta so chunk accumulation is exercised
    a.checkpoint_step = -1
    a.error_step      = -1
    a.start_iter      = 0
    a.shift_type      = SHIFT_TYPE
    a.comm            = MPI.COMM_WORLD
    return a


# --- shared ground truth -------------------------------------------------
obj_gt = rng.standard_normal((nobj, nobj, nobj)).astype('float32') * 0.02
prb_gt = (1 + 0.05 * rng.standard_normal((ndist, nz, n))
          + 0.05j * rng.standard_normal((ndist, nz, n))).astype('complex64')
# measured per-angle base shifts, and the global per-(tile, dist) correction
pos_base_gt = (rng.standard_normal((ntheta, ndist, 2)) * 1.5).astype('float32')
pos_glob_gt = (rng.standard_normal((ndist, 2)) * 0.7).astype('float32')
tp_gt       = np.zeros((ndist, 2, 2), dtype='float32')
tp_gt[:, 0, :] = 0.01 * (1 + np.arange(ndist))[:, None]    # A (slope)
tp_gt[:, 1, :] = 0.005                                     # B (intercept)

cl_p = RecParent(make_args())
cl_n = RecPosDist(make_args())
st, end = cl_p.st_theta, cl_p.end_theta

# data: generated once with the parent at the *total* shift, shared by both
cl_p.vars['obj'][:] = obj_gt[cl_p.st_obj:cl_p.end_obj]
cl_p.vars['prb'][:] = cp.asarray(prb_gt)
cl_p.vars['tp'][:]  = cp.asarray(tp_gt)
cl_p.vars['pos'][:] = cp.asarray(pos_base_gt[st:end] + pos_glob_gt[None])
cl_p.gen_sqrt_data(cl_p.vars, cl_p.data)
cl_n.data[:] = cl_p.data

# --- put both solvers at the same point ---------------------------------
# a point away from the truth, so nothing is evaluated at a stationary point
pos_glob = (pos_glob_gt + 0.4 * rng.standard_normal((ndist, 2))).astype('float32')
obj0     = obj_gt + 0.01 * rng.standard_normal(obj_gt.shape).astype('float32')
prb0     = prb_gt * np.complex64(1.02)
tp0      = tp_gt * np.float32(1.1)

for cl in (cl_p, cl_n):
    cl.vars['obj'][:] = obj0[cl.st_obj:cl.end_obj]
    cl.vars['prb'][:] = cp.asarray(prb0)
    cl.vars['tp'][:]  = cp.asarray(tp0)
cl_p.vars['pos'][:] = cp.asarray(pos_base_gt[st:end] + pos_glob[None])
cl_n.pos_base[:]    = cp.asarray(pos_base_gt[st:end])
cl_n.vars['pos'][:] = cp.asarray(pos_glob)

cl_p.precalc(cl_p.vars)
cl_n.precalc(cl_n.vars)

ok = True


def check(name, a, b, tol):
    global ok
    a = np.asarray(a, dtype='float64')
    b = np.asarray(b, dtype='float64')
    scale = max(np.abs(a).max(), np.abs(b).max(), 1e-30)
    rel   = np.abs(a - b).max() / scale
    good  = rel < tol
    ok   &= good
    print(f"  [{'ok ' if good else 'FAIL'}] {name:<42s} rel={rel:.3e} (tol {tol:.0e})")


print(f"=== rec_mpi_shrink_posdist vs rec_mpi_shrink  (shift_type={SHIFT_TYPE}) ===")

# --- 1. functional -------------------------------------------------------
f_p = cl_p.min(cl_p.vars['prb'], cl_p.vars['obj'], cl_p.vars['pos'],
               cl_p.vars['proj'], cl_p.vars['tp'])
f_n = cl_n.min(cl_n.vars['prb'], cl_n.vars['obj'], cl_n.vars['pos'],
               cl_n.vars['proj'], cl_n.vars['tp'])
check('min()', f_p, f_n, 1e-6)

# --- 2. gradient ---------------------------------------------------------
cl_p.compute_gradient(cl_p.vars, cl_p.grads)
cl_n.compute_gradient(cl_n.vars, cl_n.grads)

g_p_pos = cp.asnumpy(cl_p.grads['pos'])            # (local_ntheta, ndist, 2)
g_n_pos = cp.asnumpy(cl_n.grads['pos'])            # (ndist, 2)
check('grad pos  (sum over theta)', g_p_pos.sum(axis=0), g_n_pos, 2e-5)
check('grad prb',  cp.asnumpy(cl_p.grads['prb']),  cp.asnumpy(cl_n.grads['prb']),  1e-5)
check('grad tp',   cp.asnumpy(cl_p.grads['tp']),   cp.asnumpy(cl_n.grads['tp']),   1e-5)
check('grad obj',  np.asarray(cl_p.grads['obj']),  np.asarray(cl_n.grads['obj']),  1e-5)
check('grad proj', np.asarray(cl_p.grads['proj']), np.asarray(cl_n.grads['proj']), 1e-5)

# --- 3. Hessian ----------------------------------------------------------
# Identical direction in every block; the pos direction is constant over theta
# for the parent, which is exactly what a global (ndist, 2) direction means.
eta_obj = (0.01 * rng.standard_normal(cl_p.etas['obj'].shape)).astype(cl_p.obj_dtype)
eta_prb = (0.01 * rng.standard_normal((ndist, nz, n))
           + 0.01j * rng.standard_normal((ndist, nz, n))).astype('complex64')
eta_tp  = (0.001 * rng.standard_normal((ndist, 2, 2))).astype('float32')
eta_pos = (0.3 * rng.standard_normal((ndist, 2))).astype('float32')

for cl in (cl_p, cl_n):
    cl.etas['obj'][:] = eta_obj
    cl.etas['prb'][:] = cp.asarray(eta_prb)
    cl.etas['tp'][:]  = cp.asarray(eta_tp)
    cl.fwd_tomo(cl.etas['obj'], out=cl.proj_tmp)
    cl.redist(cl.proj_tmp, cl.etas['proj'])
cl_p.etas['pos'][:] = cp.asarray(np.broadcast_to(eta_pos, (end - st, ndist, 2)).copy())
cl_n.etas['pos'][:] = cp.asarray(eta_pos)

H_p = cl_p.hessian(cl_p.vars, cl_p.etas, cl_p.etas)
H_n = cl_n.hessian(cl_n.vars, cl_n.etas, cl_n.etas)
check('hessian(eta, eta)', H_p, H_n, 1e-5)

# Mixed form (grads, etas) — exercises the y_is_z=False branch. grads is only
# scratch here; the parent's pos block is replaced by the broadcast global
# gradient so the two calls see the same direction (its other blocks already
# match, per the checks above).
cl_p.grads['pos'][:] = cp.asarray(np.broadcast_to(g_n_pos.astype('float32'),
                                                  (end - st, ndist, 2)).copy())
H_p2 = cl_p.hessian(cl_p.vars, cl_p.grads, cl_p.etas)
H_n2 = cl_n.hessian(cl_n.vars, cl_n.grads, cl_n.etas)
check('hessian(grad, eta)  [y is not z]', H_p2, H_n2, 1e-5)

# --- 4. finite-difference gradient ---------------------------------------
# Central difference of min() w.r.t. each entry of the global pos, over a range
# of eps. min() is a float32 reduction over ntheta*ndist*nz*n samples, so the
# difference of two O(1) values loses digits fast: too small an eps is round-off,
# too large is truncation. The parent is differenced the same way (broadcasting
# the same perturbation over theta) as the reference for "how well the cascade
# gradient matches finite differences at all" — the posdist number should track
# it, since both differentiate the very same composite.
p0_n = cp.asnumpy(cl_n.vars['pos']).copy()
p0_p = cp.asnumpy(cl_p.vars['pos']).copy()


def fd_new(eps):
    g = np.zeros((ndist, 2))
    for k in range(ndist):
        for ax in range(2):
            for sgn in (+1, -1):
                p = p0_n.copy(); p[k, ax] += sgn * eps
                cl_n.vars['pos'][:] = cp.asarray(p)
                g[k, ax] += sgn * float(cl_n.min(
                    cl_n.vars['prb'], cl_n.vars['obj'], cl_n.vars['pos'],
                    cl_n.vars['proj'], cl_n.vars['tp']))
            g[k, ax] /= 2 * eps
    cl_n.vars['pos'][:] = cp.asarray(p0_n)
    return g


def fd_parent(eps):
    g = np.zeros((ndist, 2))
    for k in range(ndist):
        for ax in range(2):
            for sgn in (+1, -1):
                p = p0_p.copy(); p[:, k, ax] += sgn * eps
                cl_p.vars['pos'][:] = cp.asarray(p)
                g[k, ax] += sgn * float(cl_p.min(
                    cl_p.vars['prb'], cl_p.vars['obj'], cl_p.vars['pos'],
                    cl_p.vars['proj'], cl_p.vars['tp']))
            g[k, ax] /= 2 * eps
    cl_p.vars['pos'][:] = cp.asarray(p0_p)
    return g


def relerr(a, b):
    return np.abs(a - b).max() / max(np.abs(a).max(), np.abs(b).max(), 1e-30)


print('  finite differences (analytic grad vs central diff), per eps:')
best = np.inf
for eps in (1e-1, 3e-2, 1e-2, 3e-3):
    r_n = relerr(fd_new(eps),    g_n_pos)
    r_p = relerr(fd_parent(eps), g_p_pos.sum(axis=0))
    best = min(best, r_n)
    print(f"      eps={eps:<7.0e} posdist rel={r_n:.3e}   parent rel={r_p:.3e}")
# Loose: this bounds float32 FD noise + the cascade's own approximation, not the
# posdist reduction (which checks 1 and 2 above already pinned to ~1e-7).
ok &= best < 3e-2
print(f"  [{'ok ' if best < 3e-2 else 'FAIL'}] best FD agreement over eps sweep    "
      f"rel={best:.3e} (tol 3e-02)")

# --- 5. rho scaling law ---------------------------------------------------
# rho_sq is a Cauchy-step metric, rho_sq[v] = <g,g> / <g, H_vv g>. Summing the
# per-projection adjoint over theta turns the pos block into a single direction
# shared by all ntheta projections, so its curvature picks up a factor ntheta
# while the Rayleigh quotient's numerator does not:
#
#     rho_sq_posdist / rho_sq_parent  ==  1 / ntheta
#     rho_posdist                     ==  rho_parent / sqrt(ntheta)
#
# estimate_rho_from_hessian computes exactly that quotient with the true
# Hessian, so the two classes measure the ratio directly. This is where the
# rho= line in the posdist configs comes from.
cl_p.estimate_rho_from_hessian(cl_p.vars, cl_p.grads, cl_p.etas)
rho_pos_p = float(np.sqrt(cl_p.rho_sq['pos']))
cl_n.estimate_rho_from_hessian(cl_n.vars, cl_n.grads, cl_n.etas)
rho_pos_n = float(np.sqrt(cl_n.rho_sq['pos']))

ratio    = rho_pos_p / rho_pos_n
expected = np.sqrt(ntheta)
print(f"  Cauchy rho_pos (obj-normalized): parent={rho_pos_p:.4e}  "
      f"posdist={rho_pos_n:.4e}")
print(f"      ratio={ratio:.3f}   sqrt(ntheta)={expected:.3f}   "
      f"→ rho_pos_new = rho_pos_old / {ratio:.2f}")
law_ok = abs(ratio / expected - 1) < 0.15
ok &= law_ok
print(f"  [{'ok ' if law_ok else 'FAIL'}] rho_pos scales as 1/sqrt(ntheta)")

# --- 6. end-to-end BH ----------------------------------------------------
# Plumbing smoke test for everything the checks above do not touch:
# compute_alpha's rank-0-only global group, apply_step / linear_batch on the
# (ndist, 2) buffers, estimate_rho_coord's snapshot-restore, error_debug's
# 40-offset print, and check_approximation's cp.empty_like(vars['pos']).
cl_n.postcalc(cl_n.vars)          # undo the manual precalc; BH does its own
cl_n.estimate_rho       = True
cl_n.rho_estimate_niter = 3
cl_n.niter              = 8
cl_n.error_step         = 1
cl_n.checkpoint_step    = -1
cl_n.vars['pos'][:] = 0           # start the correction from zero, as a real run does
cl_n.pos_base[:]    = cp.asarray(pos_base_gt[st:end] + pos_glob)

cl_n.BH(writer=None)

err = cl_n.table['err'].to_numpy(dtype='float64')
print(f"  BH errors: {err[0]:.5e} -> {err[-1]:.5e}  "
      f"({len(err)} entries, rho_pos={np.sqrt(cl_n.rho_sq['pos']):.3e})")
converged = np.isfinite(err).all() and err[-1] < err[0]
ok &= converged
print(f"  [{'ok ' if converged else 'FAIL'}] BH runs and reduces the error")

# The refined offset should move back toward -pos_glob' (data was generated at
# pos_glob_gt, pos_base was seeded with the perturbed pos_glob), i.e. toward
# pos_glob_gt - pos_glob. Only report it: 8 iterations is not convergence.
want = pos_glob_gt - pos_glob
got  = cp.asnumpy(cl_n.vars['pos'])
print(f"  refined offset after {cl_n.niter} iters vs the seeded error "
      f"(|want|max={np.abs(want).max():.3f}):")
for k in range(ndist):
    print(f"      d{k}: got ({got[k,0]:+.4f}, {got[k,1]:+.4f})   "
          f"want ({want[k,0]:+.4f}, {want[k,1]:+.4f})")

print('=== ALL OK ===' if ok else '=== FAILURES ===')
sys.exit(0 if ok else 1)
