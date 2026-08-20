"""
test_hessian3.py — unit checks for the fused single-sweep Hessian.

Covers the pieces that no existing config exercises: the two-haloed-input
chunking path, LaplacianTerm's padded grads buffer, and the three-bilinear-form
routines that replaced the three separate Hessian sweeps.

    mpirun -np 1 python test_hessian3.py
    mpirun -np 4 python test_hessian3.py      # exercises ghost exchange

Every check compares the new fused routine against the pre-existing route it
replaced, so a pass means "identical to what the old code computed", not
"matches an independent re-derivation of the stencil".

The cascade term (hessian_cascade3) is NOT covered here — it needs a real
dataset. Verify it end-to-end with check_fused_hessian=1 on a real config; see
the Readme section this file is referenced from.
"""

import sys
import numpy as np
import cupy as cp
from mpi4py import MPI

from holotomocupy.chunking import Chunking
from holotomocupy.extra_terms import LaplacianTerm, PrbfitTerm, biharm
from holotomocupy.mpi_functions import MPIClass
from holotomocupy.propagation import Propagation
from holotomocupy.utils import redot, reprod, make_pinned

comm = MPI.COMM_WORLD
rank, nranks = comm.Get_rank(), comm.Get_size()

# float32 accumulation over nobj^2 * nzobj elements; 3e-5 is round-off, not slack
RTOL = 3e-5
_failures = []


def check(name, got, ref, rtol=RTOL):
    got, ref = np.atleast_1d(np.asarray(got, 'float64')), np.atleast_1d(np.asarray(ref, 'float64'))
    denom = np.where(np.abs(ref) > 0, np.abs(ref), 1.0)
    rel = np.abs(got - ref) / denom
    ok = bool(np.all(rel <= rtol))
    if rank == 0:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<44} rel={np.max(rel):.3e}")
        if not ok:
            print(f"        got={got}\n        ref={ref}")
    if not ok:
        _failures.append(name)


def fill(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    a = make_pinned(list(shape), dtype)
    if np.dtype(dtype).kind == 'c':
        a[:] = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)
    else:
        a[:] = rng.standard_normal(shape).astype(dtype)
    return a


# ======================================================================== 1
def test_chunking_two_halos():
    """gpu_batch must give BOTH size+inp_pad inputs their halo, and must not
    have changed what it does with a single one."""
    nz, ny, nx, chunk = 11, 6, 7, 4          # nz % chunk != 0: exercise the ragged tail
    cl = Chunking(200 * 1024**2, chunk)
    gp, ep = fill((nz + 4, ny, nx), 'complex64', 1), fill((nz + 4, ny, nx), 'complex64', 2)
    d1 = fill((nz, ny, nx), 'complex64', 3)

    acc3 = cp.zeros(3, dtype='float32')

    @cl.gpu_batch(axis_out=0, axis_inp=0, nout=1, inp_pad=4)
    def _three(self, acc, g_pad_chunk, e_pad_chunk):
        bg, be = biharm(g_pad_chunk), biharm(e_pad_chunk)
        g = g_pad_chunk[2:-2]
        acc[0:1] += redot(g, bg)
        acc[1:2] += redot(g, be)
        acc[2:3] += redot(e_pad_chunk[2:-2], be)

    _three(None, acc3, gp, ep)

    acc1 = cp.zeros(1, dtype='float32')

    @cl.gpu_batch(axis_out=0, axis_inp=0, nout=1, inp_pad=4)
    def _one(self, acc, e_pad_chunk, d):
        acc[:] += redot(d, biharm(e_pad_chunk))

    _one(None, acc1, ep, d1)

    # reference: no chunking at all, whole slab resident
    G, E = cp.asarray(gp), cp.asarray(ep)
    BG, BE = biharm(G), biharm(E)
    check("chunked 2-halo == unchunked",
          acc3.get(), [float(redot(G[2:-2], BG)), float(redot(G[2:-2], BE)),
                       float(redot(E[2:-2], BE))])
    check("chunked 1-halo == unchunked (regression)",
          float(acc1[0]), float(redot(cp.asarray(d1), BE)))


# ======================================================================== 2
def test_laplacian_hessian3():
    """LaplacianTerm.hessian3() must equal the three forms the old two-call
    route produced, and must stay symmetric and local."""
    nzobj, nobj, chunk, lam = 4 * nranks + 3, 24, 3, 7.5e-4
    cl_mpi = MPIClass(comm, nzobj, nranks, nobj, 'complex64')
    cl = Chunking(400 * 1024**2, chunk)
    local_nzobj = cl_mpi.local_nzobj
    obj_size = nzobj * nobj**2

    lt = LaplacianTerm(lam, obj_size, local_nzobj, nobj, 'complex64',
                       cl_mpi, cl.gpu_batch, grad_pad=True)
    # write through the views exactly as alloc_arrays wires them up
    lt.grads_view[:] = fill((local_nzobj, nobj, nobj), 'complex64', 10 + rank)
    lt.etas_view[:] = fill((local_nzobj, nobj, nobj), 'complex64', 50 + rank)

    gg, ge, ee = lt.hessian3()

    # Reference: hessian(d) contracts d against (grad^2)^2 of e_pad, which is
    # what the pre-change code called for the ge and ee forms. It re-exchanges
    # ghosts itself, so call order does not matter.
    ref_ge = lt.hessian(lt.grads_view)
    ref_ee = lt.hessian(lt.etas_view)
    check("lap hessian3 ge == hessian(g)", ge, ref_ge)
    check("lap hessian3 ee == hessian(e)", ee, ref_ee)

    # gg has no old-code equivalent (that is why it needed g_pad), so check it
    # against the same routine with the roles of the two buffers swapped.
    lt.e_pad, lt.g_pad = lt.g_pad, lt.e_pad
    sgg, sge, see = lt.hessian3()
    lt.e_pad, lt.g_pad = lt.g_pad, lt.e_pad
    check("lap hessian3 gg == swapped ee", gg, see)

    # Symmetry holds only AFTER the allreduce: the biharmonic couples across the
    # rank boundary, so this rank's <g, (grad^2)^2 e> draws on its neighbours' e
    # while <e, (grad^2)^2 g> draws on their g. The two differ per rank by a
    # boundary flux that cancels in the global sum. This is why the fused step
    # reduces first and expands `bottom` second.
    check("lap hessian3 symmetric B(g,e)==B(e,g) after allreduce",
          comm.allreduce(ge, op=MPI.SUM), comm.allreduce(sge, op=MPI.SUM))
    if nranks > 1:
        local_asym = abs(ge - sge) / max(abs(ge), 1e-30)
        check("lap hessian3 per-rank forms are NOT symmetric (flux is real)",
              float(local_asym > 1e-4), 1.0, rtol=0)

    # Consistency with the gradient: the Hessian operator must be the one the
    # gradient applies. gradient() adds scale*(grad^2)^2 u to its output, so
    # feeding e through u_pad and contracting by hand reproduces B(g,e)/B(e,e).
    lt.u_pad[:] = lt.e_pad
    zero = make_pinned([local_nzobj, nobj, nobj], 'complex64'); zero[:] = 0
    biharm_e = make_pinned([local_nzobj, nobj, nobj], 'complex64')
    biharm_e[:] = 0
    lt.gradient(biharm_e)                      # biharm_e = 0 + scale*(grad^2)^2 e
    Be = cp.asarray(biharm_e)
    check("lap hessian3 ge == <g, grad-operator e>",
          ge, float(redot(cp.asarray(np.ascontiguousarray(lt.grads_view)), Be)))
    check("lap hessian3 ee == <e, grad-operator e>",
          ee, float(redot(cp.asarray(np.ascontiguousarray(lt.etas_view)), Be)))
    del zero

    # Locality: hessian3 must return this rank's slab only, so that Rec's single
    # allreduce is correct. (The old hessian() allreduced internally and was
    # then allreduced again by the caller -- counted nranks times.)
    if nranks > 1:
        spread = comm.allreduce(ee, op=MPI.MAX) - comm.allreduce(ee, op=MPI.MIN)
        check("lap hessian3 is LOCAL (ranks report different values)",
              float(spread > 0), 1.0, rtol=0)

    lt.u_pad = lt.e_pad = lt.g_pad = None


# ======================================================================== 3
def test_prbfit_hessian3():
    """PrbfitTerm.hessian3 must equal three PrbfitTerm.hessian calls."""
    ndist, nz, n, lam = 3, 16, 16, 3.1e-3
    prop = Propagation(n, nz, 2, ndist, 1.24e-9 / 33.35,
                       np.float32(1e-8), np.float32(np.linspace(1e-3, 4e-3, ndist)))
    pt = PrbfitTerm(lam, ndist * nz * n, ndist, nz, n, prop)
    pt.ref[:] = cp.asarray(np.abs(np.random.default_rng(7).standard_normal((ndist, nz, n))).astype('float32'))
    prb = fill((ndist, nz, n), 'complex64', 11)
    g = fill((ndist, nz, n), 'complex64', 12)
    e = fill((ndist, nz, n), 'complex64', 13)

    gg, ge, ee = pt.hessian3(prb, g, e)
    check("prb hessian3 gg", gg, pt.hessian(prb, g, g))
    check("prb hessian3 ge", ge, pt.hessian(prb, g, e))
    check("prb hessian3 ee", ee, pt.hessian(prb, e, e))
    check("prb hessian3 symmetric", ge, pt.hessian(prb, e, g))


# ======================================================================== 4
def test_step_algebra():
    """The identity the fused step relies on:
           B(b*e - g, b*e - g) == b^2 B(e,e) - 2b B(g,e) + B(g,g)
    checked on the only term available without a dataset (probe fit), which is
    the same bilinear contraction shape the cascade uses."""
    ndist, nz, n, lam = 3, 16, 16, 3.1e-3
    prop = Propagation(n, nz, 2, ndist, 1.24e-9 / 33.35,
                       np.float32(1e-8), np.float32(np.linspace(1e-3, 4e-3, ndist)))
    pt = PrbfitTerm(lam, ndist * nz * n, ndist, nz, n, prop)
    pt.ref[:] = cp.asarray(np.abs(np.random.default_rng(7).standard_normal((ndist, nz, n))).astype('float32'))
    prb = fill((ndist, nz, n), 'complex64', 21)
    g = fill((ndist, nz, n), 'complex64', 22)
    e = fill((ndist, nz, n), 'complex64', 23)

    gg, ge, ee = pt.hessian3(prb, g, e)
    for beta in (0.0, 0.37, -2.4, 11.0):
        d = make_pinned([ndist, nz, n], 'complex64')
        d[:] = beta * e - g
        check(f"expansion of bottom at beta={beta:g}",
              beta * beta * ee - 2.0 * beta * ge + gg, pt.hessian(prb, d, d),
              rtol=2e-4 if abs(beta) > 5 else RTOL)   # Schur cancellation at large beta


if __name__ == "__main__":
    if rank == 0:
        print(f"\nfused-Hessian unit checks, {nranks} rank(s)\n" + "=" * 62)
    for t in (test_chunking_two_halos, test_laplacian_hessian3,
              test_prbfit_hessian3, test_step_algebra):
        if rank == 0:
            print(f"\n{t.__name__}:")
        t()
        comm.Barrier()
    nfail = comm.allreduce(len(_failures), op=MPI.SUM)
    if rank == 0:
        print("\n" + "=" * 62)
        print(f"{'ALL CHECKS PASSED' if nfail == 0 else str(nfail) + ' CHECK(S) FAILED'}\n")
    sys.exit(1 if nfail else 0)
