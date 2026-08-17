"""Variant of rec_mpi_shrink in which the OPTIMIZED position variable is one
shift per (tile, distance) instead of one shift per (projection, distance).

Motivation
----------
``rec_mpi_shrink.Rec`` refines ``pos`` with shape (local_ntheta, ndist, 2) —
ntheta*ndist*2 unknowns, one per projection. On a mosaic scan the encoder
already measures the per-angle jitter well; what is not known is a single
rigid offset per tile and per distance (detector/optics placement, tile
placement on the mosaic). This class optimizes exactly those:

    vars['pos']  shape (ndist, 2)   float32, GLOBAL (identical on every rank)

On a mosaic run ``ndist`` is the flattened ``ntiles * ndist_tile`` axis that
``MosaicReader`` builds (tile-major), so (ndist, 2) is precisely "one (y, x)
shift per tile and distance". On a single-tile run it degenerates to one shift
per distance.

The per-angle measured shifts are NOT thrown away: they stay as a fixed,
non-optimized base

    self.pos_base   shape (local_ntheta, ndist, 2)   float32, cupy

and the shift actually handed to the shift operator is

    r[theta, dist] = pos_base[theta, dist] + pos[dist]

Because r is affine in pos with unit Jacobian, nothing in the F0..F4 cascade
changes: the wrappers that enter the cascade expand pos to the per-projection
grid, and the gradient wrapper sums the per-projection adjoint back down over
theta (and, since theta is distributed, over MPI ranks). ``pos`` therefore
behaves exactly like ``prb`` and ``tp``: global, allreduced, and counted only
once (on rank 0) in the CG numerator.

Driver contract
---------------
Fill ``cl.pos_base`` — not ``cl.vars['pos']`` — with the measured positions:

    reader.read_pos(out=cl.pos_base)                       # fresh start
    reader.read_checkpoint(ckpt, ..., out_pos=cl.pos_base) # resume

Checkpoints keep the usual (ntheta, ndist, 2) ``/pos`` dataset holding the
TOTAL shift (base + correction), so they stay readable by every existing tool
and by ``rec_mpi_shrink`` itself. A resumed run therefore loads the total into
``pos_base`` and restarts the correction from zero.

Requires args.rho to have 4 entries, as rec_mpi_shrink does:
[rho_obj, rho_prb, rho_pos, rho_demag]."""

import os

import numpy as np
import cupy as cp
import nvtx

from .rec_mpi_shrink import Rec as RecShrink
from .utils import *
from .logger_config import logger


class Rec(RecShrink):
    """rec_mpi_shrink.Rec with vars['pos'] reduced to (ndist, 2), global."""

    # Variables that are replicated on every rank rather than theta-distributed.
    # Their gradients are allreduced and they contribute to the CG numerator
    # from rank 0 only, so the allreduce does not count them ndist times.
    GLOBAL_VARS = ('prb', 'pos', 'tp')

    def __init__(self, args):
        super().__init__(args)
        # gpu_batch tells a "proper" (theta-chunked) argument from a global one
        # by comparing shape[axis] against the chunking length. A global
        # (ndist, 2) array would be mistaken for a proper one if ndist happened
        # to equal this rank's theta count, silently shifting every argument.
        # Same latent trap as tp (ndist, 2, 2) in the parent class; here it is
        # checked instead of hoped for.
        if self.ndist == self.local_ntheta:
            raise ValueError(
                f'ndist == local_ntheta == {self.ndist}: the chunking layer '
                f'cannot tell the global (ndist, 2) position array from a '
                f'per-projection one. Change the rank count so the two differ.')

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------
    def alloc_arrays(self):
        """Parent allocation, then swap the pos buffers for global (ndist, 2)
        ones and add the fixed per-projection base."""
        super().alloc_arrays()

        # Fixed, never optimized: the measured per-angle shifts. The driver
        # fills this (reader.read_pos / read_checkpoint) before BH starts.
        self.pos_base = cp.zeros([self.local_ntheta, self.ndist, 2], dtype='float32')

        pos_shape = [self.ndist, 2]
        self.vars['pos'] = cp.zeros(pos_shape, dtype='float32')
        for ge in (self.grads, self.etas):
            ge['pos'] = cp.zeros(pos_shape, dtype='float32')

    def pos_full(self, pos=None):
        """Per-projection shifts actually used: pos_base + pos[None].
        Shape (local_ntheta, ndist, 2). `pos` defaults to vars['pos']."""
        if pos is None:
            pos = self.vars['pos']
        return self.pos_base + pos[None]

    # ------------------------------------------------------------------
    # BH plumbing that has to know pos is global now
    # ------------------------------------------------------------------
    def compute_alpha(self, vars, grads, etas, beta):
        """As the parent, but pos joins prb/tp in the rank-0-only group.

        linear_redot_batch also performs eta <- beta*eta - grad, so it must run
        on every rank; only its dot contribution is dropped off rank 0.
        """
        with nvtx.annotate(":::BH:calc_alpha"):
            top = -self.linear_redot_batch(etas['obj'], grads['obj'], beta, -1) \
                  / (self.rho_sq['obj'] + 1e-14)
            dots = {v: self.linear_redot_batch(etas[v], grads[v], beta, -1)
                    for v in self.GLOBAL_VARS}
            if self.rank == 0:
                for v, d in dots.items():
                    top -= d / (self.rho_sq[v] + 1e-14)
            self.linear_batch(etas['proj'], grads['proj'], beta, -1)
            bottom = self.hessian(vars, etas, etas)
            top, bottom = self.allreduce2(top, bottom)
            alpha = top / bottom
        return alpha, top, bottom

    def gradients(self, vars, grads):
        """Parent gradient (which allreduces prb and tp), plus pos — now global,
        so each rank holds only the sum over its own projections."""
        super().gradients(vars, grads)
        grads['pos'][:] = cp.array(self.allreduce(grads['pos'].get()))
        # Gauge fix: pin the centre tile's distance 0. A shift common to all
        # (tile, distance) is degenerate with shifting the object, so one entry
        # has to stay put. Zeroing the gradient is enough — etas['pos'] starts
        # at zero and is only ever updated as eta <- beta*eta - grad, so this
        # row's eta stays zero and apply_step never moves it.
        grads['pos'][self.tiles.index('center') * self.ndist_tile] = 0

    @timer
    def gradients_cascade(self, vars, grads):
        """Cascade gradient with pos as a global (nonproper) output.

        The cascade itself is untouched: pos is expanded to the per-projection
        grid on entry, and gF3's per-projection adjoint is summed back over the
        chunk's theta axis (chain rule for r = pos_base + pos, dr/dpos = 1).
        """
        grads['prb'][:] = 0
        grads['tp'][:]  = 0
        grads['pos'][:] = 0

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=4)
        def _gradients_cascade(self,
                               gradproj,                      # 1 proper output
                               gradpos, gradprb, gradtp,      # 3 nonproper outputs
                               d, proj, pos_base, t_chunk,    # 4 proper inputs
                               pos, tp, prb):                 # 3 nonproper inputs
            self._t_chunk = t_chunk
            self.cl_shift.coeff_cache_reset()
            r = pos_base + pos[None]                          # (chunk, ndist, 2)
            gradproj[:] = 0
            for k in range(self.ndist):
                self._dist_idx = k
                x = [prb[k], proj, r[:, k], tp[k]]
                y = d[:, k]
                for id in range(len(self.gF)):
                    y = self.gF[id](x, y)
                gradprb[k]  += y[0] * self.rho_sq['prb']
                gradproj    += y[1] * self.rho_sq['obj']
                gradpos[k]  += cp.sum(y[2], axis=0) * self.rho_sq['pos']
                gradtp[k]   += y[3] * self.rho_sq['tp']
            gradproj[:] = self.cl_shift.coeff(gradproj)

        _gradients_cascade(self,
                           grads['proj'],                                     # proper out
                           grads['pos'], grads['prb'], grads['tp'],           # nonproper out
                           self.data, vars['proj'], self.pos_base, self.t_local,  # proper in
                           vars['pos'], vars['tp'], vars['prb'])              # nonproper in

    @timer
    def hessian_cascade(self, vars, grads, etas):
        """As the parent, with pos (value and both directions) global.

        The (y, z) directions are constant over theta, so they are materialized
        once per chunk onto the per-projection grid — a few kB — and the F3
        kernels see exactly the shapes they always did.
        """
        out = cp.zeros(1, dtype="float32")

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _hessian_cascade(
            self, out,
            d, pos_base, x1, y1, z1, t_chunk,   # proper inputs (chunked on axis 0)
            x2, y2, z2,                          # pos (ndist, 2)   — nonproper (global)
            x3, y3, z3,                          # tp  (ndist, 2, 2) — nonproper (global)
            x0, y0, z0,                          # prb (ndist, ...)  — nonproper (global)
        ):
            self._t_chunk = t_chunk
            self.cl_shift.coeff_cache_reset()
            # Identity check must happen on the un-sliced arrays; arr[k] views
            # are never `is`-equal even when their backing memory is identical.
            y_is_z = y0 is z0

            r  = pos_base + x2[None]                 # (chunk, ndist, 2)
            dy = cp.empty_like(r); dy[:] = y2[None]
            if not y_is_z:
                dz = cp.empty_like(r); dz[:] = z2[None]

            for k in range(self.ndist):
                self._dist_idx = k
                x = [x0[k], x1, r[:, k],  x3[k]]
                y = [y0[k], y1, dy[:, k], y3[k]]
                z = y if y_is_z else [z0[k], z1, dz[:, k], z3[k]]
                w = [None, None, None, None]

                for id in range(1, len(self.F))[::-1]:
                    # d2F(dFy,dFz) + dF(d2F(y,z))
                    w = self.d2F_dF[id](x, y, z, w)
                    fx, y = self.dF[id](x, y)
                    z = y if y_is_z else self.dF[id](x, z, return_x=False)
                    x = fx

                out[:] += self.d2F_dF[0](x, y, z, w, d[:, k])

        _hessian_cascade(
            self, out,
            self.data, self.pos_base,
            vars["proj"], grads["proj"], etas["proj"], self.t_local,
            vars["pos"], grads["pos"], etas["pos"],
            vars["tp"],  grads["tp"],  etas["tp"],
            vars["prb"], grads["prb"], etas["prb"],
        )

        return out[0].get()

    def estimate_rho_from_hessian(self, vars, grads, etas):
        """Cauchy-step per-variable rho estimate, normalized to rho_sq['obj']=1.

        Identical to the parent's, except pos is in GLOBAL_VARS and so is
        excluded from the <g, g> sum on every rank but 0.
        """
        self.rho_sq = {'obj': 1.0, 'prb': 1.0, 'pos': 1.0, 'tp': 1.0}
        self.compute_gradient(vars, grads)

        new_rho_sq = {}
        for v in ('obj', 'prb', 'pos', 'tp'):
            for buf in etas.values():
                buf[:] = 0
            etas[v][:] = grads[v]
            if v == 'obj':
                self.fwd_tomo(etas['obj'], out=self.proj_tmp)
                self.redist(self.proj_tmp, etas['proj'])

            H_vv = self.hessian(vars, etas, etas)
            g2   = self.redot_batch(grads[v], grads[v])
            if v in self.GLOBAL_VARS and self.rank != 0:
                g2 = 0.0
            g2, H_vv = self.allreduce2(g2, H_vv)

            if H_vv > 0 and g2 > 0:
                new_rho_sq[v] = g2 / H_vv
            else:
                new_rho_sq[v] = 1.0
                if self.rank == 0:
                    logger.warning(f'estimate_rho_from_hessian: {v} '
                                   f'H_vv={H_vv:.3e} g2={g2:.3e} — fallback to 1.0')

        if new_rho_sq['obj'] > 0:
            s = new_rho_sq['obj']
            for k in new_rho_sq:
                new_rho_sq[k] /= s

        self.rho_sq = new_rho_sq
        if self.rank == 0:
            est = {k: float(np.sqrt(v)) for k, v in new_rho_sq.items()}
            logger.info(f'rho estimated (sqrt, obj-normalized): '
                        f"obj={est['obj']:.3e} prb={est['prb']:.3e} "
                        f"pos={est['pos']:.3e} tp={est['tp']:.3e}")

    # ------------------------------------------------------------------
    # Functional / data generation
    # ------------------------------------------------------------------
    @timer
    def min(self, prb, obj, pos, proj, tp):
        out = cp.zeros(1, dtype="float32")

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _min(self, out,
                 proj, pos_base, t_chunk, data,     # proper
                 pos, tp, prb):                     # nonproper (global)
            self._t_chunk = t_chunk
            self.cl_shift.coeff_cache_reset()
            r = pos_base + pos[None]
            for k in range(self.ndist):
                self._dist_idx = k
                x = [prb[k], proj, r[:, k], tp[k]]
                y = self.apply_F_from(x, 1)     # applies F4, F3, F2, F1
                out[:] += self.F0(y, data[:, k])

        _min(self, out, proj, self.pos_base, self.t_local, self.data, pos, tp, prb)

        out = out[0]
        if self.rank == 0 and hasattr(self, 'cl_prb_term'):
            out += self.cl_prb_term.energy_local(prb)
        if hasattr(self, 'cl_lap_term'):
            out += self.cl_lap_term.energy_local()
        return self.allreduce(np.array(out.get(), dtype='float32'))

    def gen_sqrt_data(self, vars, out):
        """Generate synthetic data. vars['tp'] must already hold the linear
        shrink parameters (A, B) and self.pos_base the per-angle shifts."""
        vars["obj"] /= self.norm_const
        self.fwd_tomo(vars["obj"], out=self.proj_tmp)
        self.redist(self.proj_tmp, vars['proj'])

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _gen_data(self, out,
                      proj, pos_base, t_chunk,     # proper
                      pos, tp, prb):               # nonproper
            self._t_chunk = t_chunk
            self.cl_shift.coeff_cache_reset()
            r = pos_base + pos[None]
            for k in range(self.ndist):
                self._dist_idx = k
                x = [prb[k], proj, r[:, k], tp[k]]
                y = self.apply_F_from(x, 1)
                out[:, k] = cp.abs(y)

        _gen_data(self, out, vars['proj'], self.pos_base, self.t_local,
                  vars['pos'], vars['tp'], vars['prb'])
        vars["obj"] *= self.norm_const

    def compute_residual(self, vars):
        """Return float32 numpy array [local_ntheta, ndist, nz, n]: |F(vars)| - sqrt(data)."""
        res = np.empty([self.local_ntheta, self.ndist, self.nz, self.n], dtype='float32')
        pos_full = self.pos_full(vars['pos'])
        for theta_st in range(0, self.local_ntheta, self.nchunk):
            theta_end = min(theta_st + self.nchunk, self.local_ntheta)
            self._t_chunk = self.t_local[theta_st:theta_end]
            self.cl_shift.coeff_cache_reset()
            proj_ch = cp.array(vars['proj'][theta_st:theta_end])
            r_ch    = pos_full[theta_st:theta_end]
            for k in range(self.ndist):
                self._dist_idx = k
                x = [vars['prb'][k], proj_ch, r_ch[:, k], vars['tp'][k]]
                x = self.apply_F_from(x, 1)
                res[theta_st:theta_end, k] = cp.asnumpy(cp.abs(x)) - self.data[theta_st:theta_end, k]
        return res

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def error_debug(self, vars, i):
        """Parent error/shrink logging, plus the full position report."""
        super().error_debug(vars, i)
        if i != -1 and not (i % self.error_step == 0 and self.error_step != -1):
            return
        if self.rank != 0:
            return
        self._log_pos_stats(vars['pos'], i)

    def _log_pos_stats(self, pos, i):
        """Print every refined (tile, distance) offset at each error checkpoint.

        `pos` is the correction sitting on top of the measured per-angle shifts
        held in pos_base, so it IS the position error of the encoder / tile
        placement, in detector pixels at the current binning. There are only
        ndist*2 of them, so unlike the per-projection parent they can be dumped
        in full: one line per tile, plus a header with the extremes and how far
        the offsets moved since the previous checkpoint (that step size going to
        zero is the signal that the position block has converged).

        Also appended to {path_out}/pos.csv — one row per checkpoint, columns
        y0,x0,y1,x1,... in the same tile-major order as ndist — for plotting the
        trajectory afterwards. Rank 0 only; pos is global.
        """
        p = cp.asnumpy(pos - self.pos_init).astype('float64')      # (ndist, 2)
        prev = getattr(self, '_pos_last', None)
        step = p - prev if prev is not None else np.zeros_like(p)
        self._pos_last = p.copy()

        mag = np.sqrt((p ** 2).sum(1))
        logger.warning(
            f"iter={i}: pos error [px]  max|dy|={np.abs(p[:, 0]).max():.3f}  "
            f"max|dx|={np.abs(p[:, 1]).max():.3f}  max|d|={mag.max():.3f}  "
            f"mean|d|={mag.mean():.3f}  max step={np.abs(step).max():.4f}")

        # Mosaic runs flatten (tile, distance) tile-major into ndist, so the
        # rows below are tiles. Single-tile runs fall back to one row.
        nd_tile = getattr(self, 'ndist_tile', None) or self.ndist
        tiles = list(getattr(self, 'tiles', None) or [])
        for t in range(max(self.ndist // nd_tile, 1)):
            label = tiles[t] if t < len(tiles) else f"tile{t}"
            row = "   ".join(
                f"d{k}: {p[t * nd_tile + k, 0]:+.3f},{p[t * nd_tile + k, 1]:+.3f}"
                for k in range(nd_tile) if t * nd_tile + k < self.ndist)
            logger.warning(f"    {label:>9}  (y,x)  {row}")

        if not hasattr(self, 'path_out'):
            return
        name = f"{self.path_out}/pos.csv"
        os.makedirs(os.path.dirname(name), exist_ok=True)
        write_header = not os.path.exists(name)
        with open(name, 'a') as f:
            if write_header:
                cols = ",".join(f"{a}{j}" for j in range(self.ndist)
                                for a in ('y', 'x'))
                f.write(f"iter,{cols}\n")
            f.write(f"{i}," + ",".join(f"{v:.6f}" for v in p.reshape(-1)) + "\n")

    def vis_debug(self, vars, i, writer=None):
        """Per-iter checkpoint write. The writer is handed the TOTAL
        per-projection positions so the checkpoint /pos dataset keeps its usual
        (ntheta, ndist, 2) meaning and stays readable by every other tool."""
        if writer is None or not (i % self.checkpoint_step == 0
                                  and self.checkpoint_step != -1) or i <= self.start_iter:
            return

        residual = None  # self.compute_residual(vars)
        shrink_now  = self._tp_to_shrink_local(vars['tp'])
        shrink_init = self._tp_to_shrink_local(self.tp_init)
        vars_out = dict(vars)
        vars_out['pos'] = self.pos_full(vars['pos'])
        writer.write_checkpoint(vars_out, i, self.norm_const,
                                residual=residual,
                                pos_init=self.pos_full(self.pos_init),
                                shrink=shrink_now, shrink_init=shrink_init,
                                shrink_gt=getattr(self, 'shrink_gt', None))
