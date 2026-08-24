"""
test_data_mask.py — unit checks for the out-of-grid detector mask.

    mpirun -np 1 python test_data_mask.py
    mpirun -np 4 python test_data_mask.py      # exercises the theta split

eff_demag = (1+shrink)/norm_magnification is > 1 for every plane but the
reference one, so a detector pixel back-maps to an object-grid coordinate that
falls outside the grid as soon as nobj < n*max(eff_demag) -- and the sample
sliding by -r moves that footprint further off the grid on one side. Shrinkage
is fitted per axis, so eff_demag is [ndist, ntheta, 2] with (y, x) differing
slightly and the y and x bounds computed independently.
Rec._build_data_mask gives the unsupported pixels zero weight in the data fit;
the F0 family then carries that weight through F0, dF0, d2F_dF0 and gF0.

The mask is per (distance, angle) and axis-separable, so it is stored as the
two 1-D factors (mask_1d) plus the box they came from (mask_box), never as a
dense [ndist, ntheta, nz, n] array.

Four things are checked:

  1. cp.fuse broadcasts the [chunk, nz, 1] and [chunk, 1, n] mask factors
     against the [chunk, nz, n] arrays the cascade kernels carry, and the
     masked-out pixels come back exactly zero from all four kernels -- not
     merely small.
  2. dF0 and d2F_dF0 are still the first and second derivatives of the MASKED
     F0 (central differences, float64 accumulation).
  3. The geometry, per angle, against a brute-force evaluation of the sampling
     formula in s_kernel: at margin 0 it is EXACT, and with a margin it is
     strictly conservative -- it never keeps a pixel whose four cubic B-spline
     taps are not all inside the object grid. mask_1d and mask_box agree, and
     each kept set is one contiguous interval.
  4. The boxes do not depend on how the angles are split over ranks: under
     -np 4 each rank reproduces the -np 1 boxes for the angles it owns.
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
    # separable per-angle mask: a random box per chunk element
    my = cp.zeros((chunk, nz, 1), dtype='float32')
    mx = cp.zeros((chunk, 1, n), dtype='float32')
    for j, (y0, y1, x0, x1) in enumerate([(0, nz, 0, n), (3, 12, 2, 17), (5, 6, 0, 9)]):
        my[j, y0:y1, 0] = 1
        mx[j, 0, x0:x1] = 1
    m = my * mx                                        # [chunk, nz, n]

    f  = Rec._F0_fused(x, d, my, mx)
    g  = Rec._dF0_fused(x, d, my, mx)
    gg = Rec._gF0_fused(x, y, my, mx, np.float32(2.0))
    h  = Rec._d2F_dF0_fused(x, y, z, w, d, my, mx)
    assert f.shape == g.shape == gg.shape == h.shape == x.shape

    e_f  = float(cp.abs(f  - m * (cp.abs(x) - d)**2).max())
    e_g  = float(cp.abs(g  - m * (x - d * (x / cp.abs(x)))).max())
    e_gg = float(cp.abs(gg - 2.0 * m * (x - y * (x / cp.abs(x)))).max())
    print(f"broadcast vs explicit: F0 {e_f:.3g}  dF0 {e_g:.3g}  gF0 {e_gg:.3g}")
    assert e_f < 1e-6 and e_g == 0.0 and e_gg == 0.0

    off = cp.broadcast_to((m == 0), x.shape)
    for name, arr in (("F0", f), ("dF0", g), ("gF0", gg), ("d2F_dF0", h)):
        bad = float(cp.abs(arr[off]).max())
        print(f"{name:>9}: max|value| on masked-out pixels = {bad:g}")
        assert bad == 0.0

    # float64 accumulation: the second difference cancels ~6 digits
    def F(t):
        return float(cp.sum(Rec._F0_fused(x + np.float32(t) * y, d, my, mx),
                            dtype=cp.float64))
    t = 1e-2
    num1 = (F(t) - F(-t)) / (2 * t)
    ana1 = 2 * float(redot(Rec._dF0_fused(x, d, my, mx), y))
    num2 = (F(t) - 2 * F(0) + F(-t)) / t**2
    ana2 = 2 * float(cp.sum(Rec._d2F_dF0_fused(x, y, y, None, d, my, mx),
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
        self.ndist, self.local_ntheta = ed.shape[:2]   # ed is [ndist, ntheta, 2]
        self.mask_oob, self.mask_oob_margin = True, margin
        self.eff_demag = cp.asarray(ed.astype('float32'))
        self.mask_1d  = np.ones((self.ndist, self.local_ntheta, nz + n), dtype='float32')
        self.mask_box = np.zeros((self.ndist, self.local_ntheta, 4), dtype='int32')
        self.rank = comm.rank
        self.cl_mpi = type('m', (), {'comm': comm})()

    def dense(self):
        """Materialize the per-angle mask the kernels see: [ndist, ntheta, nz, n]."""
        my = self.mask_1d[:, :, :self.nz]
        mx = self.mask_1d[:, :, self.nz:]
        return (my[..., :, None] * mx[..., None, :]) > 0


def exact_mask(n, nz, nobj, nzobj, ed, pos):
    """Brute force, PER ANGLE: keep a pixel only if all four taps of the cubic
    B-spline are in-grid. Mirrors the index math in s_kernel."""
    ndist, T = ed.shape[:2]
    out = np.zeros((ndist, T, nz, n), bool)
    tx, ty = np.arange(n), np.arange(nz)
    for k in range(ndist):
        for t in range(T):
            xx = ed[k, t, 1] * (tx - (n - 1) * 0.5) - pos[k, t, 1] + (nobj - 1) * 0.5
            yy = ed[k, t, 0] * (ty - (nz - 1) * 0.5) - pos[k, t, 0] + (nzobj - 1) * 0.5
            ix, iy = np.floor(xx).astype(int), np.floor(yy).astype(int)
            out[k, t] = np.outer((iy - 1 >= 0) & (iy + 2 <= nzobj - 1),
                                 (ix - 1 >= 0) & (ix + 2 <= nobj - 1))
    return out


def _geometry():
    """AtomiumL1_HT geometry scaled down, plus a large-displacement distance
    (eff_demag == 1, sample sliding most of a frame) -- the case the per-angle
    box exists for."""
    rng = np.random.default_rng(0)
    nm = np.array([1, 0.95890, 0.82341, 0.63664, 1.0])
    N, T = 128, 32
    # per-axis shrink: y and x drift independently, so ed is [ndist, T, 2]
    ed = (np.tile((1.0 / nm)[:, None, None], (1, T, 2))
          * (1 + rng.normal(0, 0.002, (len(nm), T, 2))))
    pos = rng.uniform(-12.5, 12.5, (len(nm), T, 2))
    pos[-1] = rng.uniform(-19.0, 19.0, (T, 2))          # the large-disp plane
    return N, T, nm, ed, pos


def check_geometry(comm):
    """3. exact at margin 0, conservative with a margin; boxes consistent."""
    N, T, nm, ed_all, pos_all = _geometry()
    lo = T * comm.rank // comm.size
    hi = T * (comm.rank + 1) // comm.size

    for nobj, tag in ((N, "nobj == n  (undersized grid)"),
                      (int(np.ceil(N / nm[1:-1].min() / 64) * 64), "nobj == auto-computed")):
        for margin in (0.0, 2.0):
            st = _Stub(N, N, nobj, nobj, ed_all[:, lo:hi], pos_all[:, lo:hi], margin, comm)
            st._build_data_mask({'pos': pos_all[:, lo:hi].astype('float32')})
            built = st.dense()
            ref   = exact_mask(N, N, nobj, nobj, ed_all[:, lo:hi], pos_all[:, lo:hi])
            wrong = int((built & ~ref).sum())
            extra = int((ref & ~built).sum())

            # mask_1d is exactly the box, i.e. each kept set is one interval
            for k in range(st.ndist):
                for t in range(st.local_ntheta):
                    y0, y1, x0, x1 = st.mask_box[k, t]
                    want = np.zeros((N, N), bool)
                    want[y0:y1, x0:x1] = True
                    assert np.array_equal(want, built[k, t]), \
                        f"mask_1d/mask_box disagree at dist {k}, angle {t}"

            if comm.rank == 0:
                print(f"{tag}, margin={margin:g}: nobj={nobj} "
                      f"kept={built.mean():.4f} exact={ref.mean():.4f} "
                      f"wrongly-kept={wrong} extra-discarded={extra} "
                      f"({100 * extra / max(ref.sum(), 1):.2f}%)")
            assert wrong == 0, "mask keeps a pixel whose B-spline support is out of grid"
            if margin == 0.0:
                assert extra == 0, "at margin 0 the box must be the exact support"

    # the per-angle box must beat one shared centred rectangle on the
    # large-displacement plane -- that is the entire point of the change
    st = _Stub(N, N, N, N, ed_all[:, lo:hi], pos_all[:, lo:hi], 2.0, comm)
    st._build_data_mask({'pos': pos_all[:, lo:hi].astype('float32')})
    loc = float(st.dense()[-1].mean()) * st.local_ntheta
    tot = comm.allreduce(loc, op=MPI.SUM) / T
    r = np.abs(pos_all[-1]).max(axis=0)
    shared = ((N - 2 * (r[0] + 2.0)) / N) * ((N - 2 * (r[1] + 2.0)) / N)
    if comm.rank == 0:
        print(f"large-disp plane: per-angle keeps {100*tot:.1f}%, "
              f"one shared centred box would keep {100*shared:.1f}%")
    assert tot > shared + 0.02


def check_mpi_invariance(comm):
    """4. the boxes do not depend on the theta split over ranks."""
    N, T, nm, ed_all, pos_all = _geometry()
    lo = T * comm.rank // comm.size
    hi = T * (comm.rank + 1) // comm.size

    ser = _Stub(N, N, N, N, ed_all, pos_all, 2.0, MPI.COMM_SELF)
    ser._build_data_mask({'pos': pos_all.astype('float32')})
    par = _Stub(N, N, N, N, ed_all[:, lo:hi], pos_all[:, lo:hi], 2.0, comm)
    par._build_data_mask({'pos': pos_all[:, lo:hi].astype('float32')})

    assert np.array_equal(ser.mask_box[:, lo:hi], par.mask_box), \
        f"rank {comm.rank}: boxes differ from the serial build"
    if comm.rank == 0:
        print(f"mpi invariance: boxes identical for a {comm.size}-way theta split")


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        check_kernels()
    check_geometry(comm)
    check_mpi_invariance(comm)
    comm.Barrier()
    if comm.rank == 0:
        print("ALL OK")
