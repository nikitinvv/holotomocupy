"""
test_data_mask.py — unit checks for the out-of-grid detector mask.

    mpirun -np 1 python test_data_mask.py
    mpirun -np 4 python test_data_mask.py      # exercises the global MPI max

eff_demag = (1+shrink)/norm_magnification is > 1 for every plane but the
reference one, so a detector pixel back-maps to an object-grid coordinate that
falls outside the grid as soon as nobj < n*max(eff_demag). Rec._build_data_mask
gives those pixels zero weight in the data fit; the F0 family then carries that
weight through F0, dF0, d2F_dF0 and gF0.

Three things are checked:

  1. cp.fuse broadcasts the [nz, n] mask against the [chunk, nz, n] arrays the
     cascade kernels carry, and the masked-out pixels come back exactly zero
     from all four kernels -- not merely small.
  2. dF0 and d2F_dF0 are still the first and second derivatives of the MASKED
     F0 (central differences, float64 accumulation).
  3. The mask geometry is a correct *sufficient* condition: it never keeps a
     detector pixel whose four cubic B-spline taps are not all inside the
     object grid for every angle, checked against a brute-force per-angle
     evaluation of the sampling formula in s_kernel.
"""

import sys
import numpy as np
import cupy as cp
from mpi4py import MPI

sys.path.insert(0, '../..')
from holotomocupy.rec_mpi import Rec
from holotomocupy.utils import redot


def check_kernels():
    """1. broadcasting + exact zeros on masked pixels; 2. derivatives."""
    chunk, nz, n = 3, 16, 20
    rng = cp.random.RandomState(0)
    def c(): return (rng.rand(chunk, nz, n) + 1j * rng.rand(chunk, nz, n)).astype('complex64')
    x = c() + 0.5          # +0.5: keep |x| away from 0, as the real x0 is
    y, z, w = c(), c(), c()
    d = rng.rand(chunk, nz, n).astype('float32') + 0.5
    m = (rng.rand(nz, n) > 0.4).astype('float32')      # [nz, n] -> broadcasts

    f  = Rec._F0_fused(x, d, m)
    g  = Rec._dF0_fused(x, d, m)
    gg = Rec._gF0_fused(x, y, m, np.float32(2.0))
    h  = Rec._d2F_dF0_fused(x, y, z, w, d, m)
    assert f.shape == g.shape == gg.shape == h.shape == x.shape

    e_f  = float(cp.abs(f  - m[None] * (cp.abs(x) - d)**2).max())
    e_g  = float(cp.abs(g  - m[None] * (x - d * (x / cp.abs(x)))).max())
    e_gg = float(cp.abs(gg - 2.0 * m[None] * (x - y * (x / cp.abs(x)))).max())
    print(f"broadcast vs explicit: F0 {e_f:.3g}  dF0 {e_g:.3g}  gF0 {e_gg:.3g}")
    assert e_f < 1e-6 and e_g == 0.0 and e_gg == 0.0

    off = cp.broadcast_to((m == 0)[None], x.shape)
    for name, arr in (("F0", f), ("dF0", g), ("gF0", gg), ("d2F_dF0", h)):
        bad = float(cp.abs(arr[off]).max())
        print(f"{name:>9}: max|value| on masked-out pixels = {bad:g}")
        assert bad == 0.0

    # float64 accumulation: the second difference cancels ~6 digits
    def F(t):
        return float(cp.sum(Rec._F0_fused(x + np.float32(t) * y, d, m),
                            dtype=cp.float64))
    t = 1e-2
    num1 = (F(t) - F(-t)) / (2 * t)
    ana1 = 2 * float(redot(Rec._dF0_fused(x, d, m), y))
    num2 = (F(t) - 2 * F(0) + F(-t)) / t**2
    ana2 = 2 * float(cp.sum(Rec._d2F_dF0_fused(x, y, y, None, d, m),
                            dtype=cp.float64))
    r1, r2 = abs(num1 - ana1) / abs(ana1), abs(num2 - ana2) / abs(ana2)
    print(f"  dF0: numeric={num1:.8e} analytic={ana1:.8e} rel={r1:.2e}")
    print(f"d2F0 : numeric={num2:.8e} analytic={ana2:.8e} rel={r2:.2e}")
    assert r1 < 1e-3 and r2 < 1e-3


class _Stub:
    """Enough of Rec for _build_data_mask to run unmodified."""
    _build_data_mask = Rec._build_data_mask

    def __init__(self, n, nz, nobj, nzobj, ed, pos, margin, comm):
        self.n, self.nz, self.nobj, self.nzobj = n, nz, nobj, nzobj
        self.ndist, self.local_ntheta = ed.shape
        self.mask_oob, self.mask_oob_margin = True, margin
        self.eff_demag = cp.asarray(ed.astype('float32'))
        self.data_mask = cp.ones((self.ndist, nz, n), dtype='float32')
        self.rank = comm.rank
        self.cl_mpi = type('m', (), {'comm': comm})()


def exact_mask(n, nz, nobj, nzobj, ed, pos):
    """Brute force: keep a pixel only if all four taps of the cubic B-spline
    are in-grid at EVERY angle. Mirrors the index math in s_kernel."""
    out = np.ones((ed.shape[0], nz, n), bool)
    tx, ty = np.arange(n), np.arange(nz)
    for k in range(ed.shape[0]):
        for t in range(ed.shape[1]):
            xx = ed[k, t] * (tx - (n - 1) * 0.5) - pos[k, t, 1] + (nobj - 1) * 0.5
            yy = ed[k, t] * (ty - (nz - 1) * 0.5) - pos[k, t, 0] + (nzobj - 1) * 0.5
            ix, iy = np.floor(xx).astype(int), np.floor(yy).astype(int)
            out[k] &= np.outer((iy - 1 >= 0) & (iy + 2 <= nzobj - 1),
                               (ix - 1 >= 0) & (ix + 2 <= nobj - 1))
    return out


def check_geometry(comm):
    """3. the built mask never keeps a pixel the brute force rejects."""
    # AtomiumL1_HT geometry, scaled down; the angles are split over the ranks
    # so the global MPI max is what has to reconstruct the full worst case.
    rng = np.random.default_rng(0)
    nm = np.array([1, 0.95890, 0.82341, 0.63664])
    N, T = 128, 32
    ed_all  = np.tile((1.0 / nm)[:, None], (1, T)) * (1 + rng.normal(0, 0.002, (4, T)))
    pos_all = rng.uniform(-12.5, 12.5, (4, T, 2))
    lo = T * comm.rank // comm.size
    hi = T * (comm.rank + 1) // comm.size

    for nobj, tag in ((N, "nobj == n  (undersized grid)"),
                      (int(np.ceil(N / nm[-1] / 64) * 64), "nobj == auto-computed")):
        st = _Stub(N, N, nobj, nobj, ed_all[:, lo:hi], pos_all[:, lo:hi], 2.0, comm)
        st._build_data_mask({'pos': pos_all[:, lo:hi].astype('float32')})
        built = cp.asnumpy(st.data_mask) > 0
        ref   = exact_mask(N, N, nobj, nobj, ed_all, pos_all)
        wrong = int((built & ~ref).sum())
        extra = int((ref & ~built).sum())
        if comm.rank == 0:
            print(f"{tag}: nobj={nobj} kept={built.mean():.4f} "
                  f"exact={ref.mean():.4f} wrongly-kept={wrong} "
                  f"extra-discarded={extra} ({100*extra/max(ref.sum(),1):.2f}%)")
        assert wrong == 0, "mask keeps a pixel whose B-spline support is out of grid"


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        check_kernels()
    check_geometry(comm)
    comm.Barrier()
    if comm.rank == 0:
        print("ALL OK")
