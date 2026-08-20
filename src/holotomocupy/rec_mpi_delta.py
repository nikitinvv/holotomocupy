"""
RecDelta: cascade-based parameterization u = delta * (1 + i * bd)

Adds a NEW cascade level F4: (prb, proj, bd, pos) -> (prb, proj*(1+i*bd), pos).
The Radon transform R: delta -> proj stays OUTSIDE the cascade (chunked over z),
analogous to how parent Rec keeps the Radon outside; what was parent's `adj_tomo`
(RT for grad_obj) is called `adj_tomo` here.

Variables:
  vars['obj']  = delta       (real 3D, float32)
  vars['proj'] = R(delta)    (REAL float32 — proj-side MPI uses a separate float32 MPIClass)
  vars['bd']   = scalar      (cupy shape (1,) float32)
  vars['prb']  = probe       (complex)
  vars['pos']  = positions   (real)

Cascade variables at level 4: (prb, proj, bd, pos)
Cascade variables at level 3: (prb, proj_complex, pos)        -- after F4
F4 promotes proj (real) to proj_complex (complex) by multiplying by (1+i*bd).

Parent's per-dist F1/F3 cascade methods are inherited as-is. Only gF3 is
overridden so gF4 receives the coeff'd Deltapsi value (coeff applied per-dist;
by linearity equivalent to parent's coeff-after-sum at the cost of ndist FFT
pairs instead of one).

args.rho is length 4: [obj, prb, pos, bd]
"""

import numpy as np
import cupy as cp
import nvtx
import pandas as pd
import os

from .rec_mpi import Rec
from .utils import mshow_approx, make_pinned, logger, time, timer
from .mpi_functions import MPIClass


class RecDelta(Rec):
    # RecDelta overrides every cascade kernel with a genuinely per-distance
    # version (extra 'bd' variable, F4 level), so the parent's hoisted
    # distance loop does not apply. This pins ndistchunk to 1 before
    # Rec.__init__ sizes the chunking pool.
    hoist_dist_loop = False

    def __init__(self, args):
        if args.obj_dtype != 'complex64':
            raise ValueError(
                f"RecDelta requires args.obj_dtype='complex64' "
                f"(proj/shift/MPI machinery is complex); got {args.obj_dtype!r}. "
                f"obj-side arrays are reallocated as float32 internally."
            )
        if not hasattr(args, 'rho') or len(args.rho) != 4:
            raise ValueError(
                f"RecDelta requires args.rho of length 4 "
                f"[obj, prb, pos, bd]; got {getattr(args, 'rho', None)}"
            )
        super().__init__(args)
        # Extend cascade lists with F4 (innermost). RecDelta's only F-family override
        # is gF3 (applies coeff per-dist for gF4); everything else inherits parent's
        # per-dist methods unchanged.
        self.F      = [self.F0, self.F1, self.F2, self.F3, self.F4]
        self.gF     = [self.gF0, self.gF1, self.gF2, self.gF3, self.gF4]
        self.dF     = [self.dF0, self.dF1, self.dF2, self.dF3, self.dF4]
        self.d2F_dF = [self.d2F_dF0, self.d2F_dF1, self.d2F_dF2, self.d2F_dF3, self.d2F_dF4]
        self.rho_sq['bd'] = float(args.rho[3]) ** 2
        # RecDelta overrides hessian_cascade and compute_alpha (extra 'bd'
        # variable, F4 level), so the parent's fused single-sweep step does not
        # apply. Forced here rather than as a class attribute because
        # Rec.__init__ copies every config key onto self.
        self.fused_hessian = False

        # Second MPIClass for proj-side traffic in float32 (proj is now real).
        # Parent's self.cl_mpi (complex64) is unused by RecDelta but kept so inherited
        # methods don't break.
        self.cl_mpi_real = MPIClass(args.comm, self.nzobj, self.ntheta, self.nobj, 'float32')
        self.redist  = self.cl_mpi_real.redist

    # ------------------------------------------------------------------ alloc
    def alloc_arrays(self):
        # delta and R(delta) are real-valued, so temporarily flip self.obj_dtype
        # to float32 just for the parent allocations. Restored immediately so
        # downstream code (cl_shift, cl_prop, cl_mpi) still sees complex64.
        saved, self.obj_dtype = self.obj_dtype, 'float32'
        try:
            super().alloc_arrays()
        finally:
            self.obj_dtype = saved

        # bd scalar
        self.vars['bd']  = cp.zeros((1,), dtype='float32')
        self.grads['bd'] = cp.zeros((1,), dtype='float32')
        self.etas['bd']  = cp.zeros((1,), dtype='float32')
        
    # ============================================================ NEW F4 layer
    @nvtx.annotate("F4", color="green")
    def F4(self, x):
        """In: (prb, proj, bd, pos)  Out: (prb, proj * (1 + i*bd), pos)"""
        prb, proj, bd, pos = x
        proj_complex = proj * (1.0 + 1j * bd)
        return [prb, proj_complex, pos]

    @nvtx.annotate("dF4", color="green")
    def dF4(self, x, y, return_x=True):
        """In: x=(prb,proj,bd,pos), y=(yprb,yproj,ybd,ypos)
        Out: ([prb,F4_proj,pos], [yprb,dF4_y,ypos])  if return_x else just second."""
        prb, proj, bd, pos = x
        yprb, yproj, ybd, ypos = y
        yproj_complex = (1.0 + 1j * bd) * yproj + 1j * proj * ybd
        if return_x:
            xproj_complex = proj * (1.0 + 1j * bd)
            return ([prb, xproj_complex, pos], [yprb, yproj_complex, ypos])
        return [yprb, yproj_complex, ypos]

    @nvtx.annotate("d2F_dF4", color="purple")
    def d2F_dF4(self, x, y, z, w):
        """Cascade composition contribution at level 4. Cross derivative i*(yproj*zbd + zproj*ybd)
        in proj slot; propagates accumulator w through dF4. Returns 3-element list (level 3 shape)."""
        prb, proj, bd, pos = x
        _, yproj, ybd, _ = y
        _, zproj, zbd, _ = z

        yproj_complex = 1j * (yproj * zbd + zproj * ybd)

        wprb_out, wpos_out = None, None
        if w is not None and w[1] is not None:
            w_prb, w_proj, w_bd, w_pos = w
            yproj_complex += (1.0 + 1j * bd) * w_proj + 1j * proj * w_bd
            wprb_out = w_prb
            wpos_out = w_pos

        # F4 doesn't touch pos; emit zero pos perturbation to keep d2F_dF3's (w[1], w[2]) invariant.
        if wpos_out is None:
            wpos_out = cp.zeros_like(pos)

        return [wprb_out, yproj_complex, wpos_out]

    @nvtx.annotate("gF4", color="green")
    def gF4(self, x, y):
        """Adjoint of dF4. proj is REAL.
          natural grad wrt proj = Re(grad_y) + bd*Im(grad_y)
          natural grad wrt bd   = sum_chunk proj * Im(grad_y)  (scalar)
        """
        prb, proj, bd_arr, pos = x
        yprb, yproj_complex, ypos = y
        bd = bd_arr.reshape(())

        yproj_out = yproj_complex.real + bd * yproj_complex.imag
        ybd_out = (cp.sum(proj * yproj_complex.imag)).reshape(1)

        return [yprb, yproj_out, ybd_out, ypos]

    # ======================================================== gF3 override
    @nvtx.annotate("gF3", color="green")
    def gF3(self, x, y):
        """Parent's per-dist gF3 returns un-coeff'd Deltapsi (caller is expected to sum
        across distances and apply cl_shift.coeff once). RecDelta's gF4 needs the
        coeff'd value, so apply coeff per-dist here. By linearity of coeff this is
        equivalent to coeff-after-sum, at the cost of ndist FFT pairs instead of one."""
        out = super().gF3(x, y)
        out[1] = self.cl_shift.coeff(out[1])
        return out


    def compute_gradient(self, vars, grads):
        """gradients (RecDelta 5-level + adj_tomo) then propagate grads['obj']
        through R via cl_mpi_real to populate grads['proj']."""
        with nvtx.annotate("gradients"):
            self.gradients(vars, grads)
        with nvtx.annotate(":::BH:fwd_tomo"):
            self.fwd_tomo(grads["obj"], out=self.proj_tmp)
        with nvtx.annotate(":::BH:redist", color='red'):
            self.redist(self.proj_tmp, grads['proj'])

    def compute_alpha(self, vars, grads, etas, beta):
        """Parent's compute_alpha + 'bd' in the rho_sq-weighted numerator."""
        with nvtx.annotate(":::BH:calc_alpha"):
            top = 0
            for v in ("obj", "pos", "bd"):
                top -= self.linear_redot_batch(etas[v], grads[v], beta, -1) / self.rho_sq[v]
            dot_prb = self.linear_redot_batch(etas['prb'], grads['prb'], beta, -1)
            if self.rank == 0:
                top -= dot_prb / self.rho_sq['prb']
            self.linear_batch(etas['proj'], grads['proj'], beta, -1)
            bottom = self.hessian(vars, etas, etas)
            top, bottom = self.allreduce2(top, bottom)
            alpha = top / bottom
        return alpha, top, bottom

    def apply_step(self, vars, etas, alpha):
        """var ← var + alpha·eta for every variable (incl. bd), then refresh proj = R(delta)."""
        for v in ("obj", "prb", "pos", "proj", "bd"):
            self.linear_batch(vars[v], etas[v], 1, alpha)
        
    # ============================================================ gradients (outer-dist cascade)
    @timer
    def gradients_cascade(self, vars, grads):
        """Cascade gradient over the 5-level cascade. Outer-dist loop:
        per k-iter, upload vars['prb'][k] once and run @gpu_batch over theta chunks.
        grads['proj'] is in/out for cross-k accumulation. RecDelta's gF[3] override
        already applies coeff per-dist, so no final coeff pass is needed (vs parent)."""
        grads['bd'][:] = 0
        grads['proj'][:] = 0   # zero accumulator before outer-k loop
        grads['prb'][:]  = 0   # each k overwrites its own slot

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=4)
        def _grad_dist(self, gradproj_out, gradpos, gradprb, gradbd,
                       gradproj_in, d, proj, pos, eff_demag, prb, bd):
            self._eff_demag_chunk = eff_demag
            self.cl_shift.coeff_cache_reset()
            self.apply_F_cache_reset()
            x = [prb, proj, bd, pos]
            y = d
            for id in range(len(self.gF)):
                y = self.gF[id](x, y)
            # y is 4-element at level 4: [yprb, yproj, ybd, ypos]
            gradprb += y[0] * self.rho_sq['prb']
            gradproj_out[:] = gradproj_in + y[1] * self.rho_sq['obj']
            gradbd += y[2] * self.rho_sq['bd']
            gradpos[:] = y[3] * self.rho_sq['pos']

        for k in range(self.ndist):
            self._dist_idx = k
            _grad_dist(self,
                       grads['proj'], grads['pos'][k], grads['prb'][k], grads['bd'],
                       grads['proj'], self.data[k],
                       vars['proj'], vars['pos'][k],
                       self.eff_demag[k], vars['prb'][k], vars['bd'])

    def gradients(self, vars, grads):
        """Full gradient: cascade -> proj, adj_tomo (RT) -> obj, regularization + allreduces."""
        self.gradients_cascade(vars, grads)

        with nvtx.annotate(":::BH:redist back", color='red'):
            self.redist(grads['proj'], self.proj_tmp, direction='backward')
        self.adj_tomo(grads['obj'], self.proj_tmp)

        if hasattr(self, 'cl_lap_term'):
            self.cl_lap_term.gradient(grads['obj'])

        if self.rank == 0 and hasattr(self, 'cl_prb_term'):
            self.cl_prb_term.gradient(grads["prb"], vars["prb"], self.rho_sq['prb'])
        grads['prb'][:] = self.allreduce(grads['prb'])

        grads['bd'][:] = cp.array(self.allreduce(grads['bd'].get()))

    # ============================================================ hessian (per-dist cascade)
    @timer
    def hessian_cascade(self, vars, grads, etas):
        """Cascade Hessian-weighted inner product <H · grads, etas> over 5 levels. Per-dist.
        coeff_cache_reset() is called *per k-iteration* so id() collisions between
        per-iter ephemeral arrays (x_proj_complex from F4) can't return stale entries."""
        out = cp.zeros(1, dtype="float32")

        y_is_z = grads['prb'] is etas['prb']

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _hess_dist(self, out, d, eff_demag,
                       x_pos, y_pos, z_pos,
                       x_proj, y_proj, z_proj,
                       x_prb, y_prb, z_prb,
                       x_bd, y_bd, z_bd):
            self._eff_demag_chunk = eff_demag
            self.cl_shift.coeff_cache_reset()
            self.apply_F_cache_reset()
            x = [x_prb, x_proj, x_bd, x_pos]
            y = [y_prb, y_proj, y_bd, y_pos]
            z = y if y_is_z else [z_prb, z_proj, z_bd, z_pos]
            w = [None, None, None, None]
            for id in range(1, len(self.F))[::-1]:  # 4, 3, 2, 1
                w = self.d2F_dF[id](x, y, z, w)
                fx, y = self.dF[id](x, y)
                z = y if y_is_z else self.dF[id](x, z, return_x=False)
                x = fx
            out[:] += self.d2F_dF[0](x, y, z, w, d)

        for k in range(self.ndist):
            self._dist_idx = k
            _hess_dist(self, out, self.data[k], self.eff_demag[k],
                       vars['pos'][k], grads['pos'][k], etas['pos'][k],
                       vars['proj'], grads['proj'], etas['proj'],
                       vars['prb'][k], grads['prb'][k], etas['prb'][k],
                       vars['bd'], grads['bd'], etas['bd'])

        return out[0].get()

    # ============================================================ min (outer-dist loop)
    @timer
    def min(self, prb, obj, pos, proj, bd):
        """Loss evaluation. 5-arg signature (with bd). Outer-dist loop:
        per k-iter, upload prb[k] once; inner gpu_batch handles theta chunking."""
        out = cp.zeros(1, dtype="float32")

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _min_dist(self, out, proj, pos, data, eff_demag, prb, bd):
            self._eff_demag_chunk = eff_demag
            self.cl_shift.coeff_cache_reset()
            self.apply_F_cache_reset()
            x = [prb, proj, bd, pos]
            y = self.apply_F_from(x, 1)
            out[:] += self.F0(y, data)

        for k in range(self.ndist):
            self._dist_idx = k
            _min_dist(self, out, proj, pos[k], self.data[k],
                      self.eff_demag[k], prb[k], bd)
        out = out[0]

        if self.rank == 0 and hasattr(self, 'cl_prb_term'):
            out += self.cl_prb_term.energy_local(prb)
        lap_e = self.cl_lap_term.energy_local() if hasattr(self, 'cl_lap_term') else 0
        return self.allreduce(np.array(out.get() + lap_e, dtype='float32'))

    # ============================================================ synthetic data
    def gen_sqrt_data(self, vars, out):
        """Generate synthetic |F(vars)| from real delta + scalar bd. Outer-dist loop:
        per k-iter, upload vars['prb'][k] once; inner @gpu_batch chunks over theta."""
        self.eff_demag[:] = (1 + self.shrink_nd) / cp.array(self.norm_magnifications[:, None])
        vars["obj"] /= self.norm_const
        self.fwd_tomo(vars['obj'], out=self.proj_tmp)
        self.redist(self.proj_tmp, vars['proj'])

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _gen_data_dist(self, out, proj, pos, eff_demag, prb, bd):
            self._eff_demag_chunk = eff_demag
            self.cl_shift.coeff_cache_reset()
            self.apply_F_cache_reset()
            x = [prb, proj, bd, pos]
            y = self.apply_F_from(x, 1)
            out[:] = cp.abs(y)

        for k in range(self.ndist):
            self._dist_idx = k
            _gen_data_dist(self, out[k], vars['proj'], vars['pos'][k],
                           self.eff_demag[k], vars['prb'][k], vars['bd'])
        vars["obj"] *= self.norm_const

    # ============================================================ logging / vis
    def check_approximation(self, vars, etas, top, bottom, alpha, i, writer=None):
        """Parent's check_approximation but with extra `bd` variable + 5-arg min."""
        if i != -1 and not (i % self.checkpoint_step == 0 and self.checkpoint_step != -1):
            return

        if not hasattr(self, '_chk_objt'):
            self._chk_objt  = make_pinned(vars['obj'].shape,  vars['obj'].dtype)
            self._chk_projt = make_pinned(vars['proj'].shape, vars['proj'].dtype)
            self._chk_prbt  = cp.empty_like(vars['prb'])
            self._chk_post  = cp.empty_like(vars['pos'])
            self._chk_bdt   = cp.empty_like(vars['bd'])

        objt, prbt, post, projt, bdt = (self._chk_objt, self._chk_prbt, self._chk_post,
                                        self._chk_projt, self._chk_bdt)

        npp = 5
        t = np.linspace(0, 2 * alpha, npp).astype('float32')
        err_real = np.zeros(npp, dtype='float32')

        for k in range(npp):
            self.linear_batch(vars['obj'],  etas['obj'],  1, t[k], out=objt)
            self.linear_batch(vars['prb'],  etas['prb'],  1, t[k], out=prbt)
            self.linear_batch(vars['pos'],  etas['pos'],  1, t[k], out=post)
            self.linear_batch(vars['proj'], etas['proj'], 1, t[k], out=projt)
            self.linear_batch(vars['bd'],   etas['bd'],   1, t[k], out=bdt)
            err_real[k] = self.min(prbt, objt, post, projt, bdt)

        f0 = self.min(vars['prb'], vars['obj'], vars['pos'], vars['proj'], vars['bd'])
        err_approx = f0 - top * t + 0.5 * bottom * t * t

        if self.rank != 0:
            return

        if writer is None:
            mshow_approx(t, err_real, err_approx, True)
            return

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(t, err_real,   "o-", label="real")
        ax.plot(t, err_approx, "x-", label="approx")
        ax.set_xlabel("t")
        ax.set_ylabel("functional")
        ax.set_title(f"iter {i}: alpha={alpha:.3e}")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        out_dir  = os.path.join(writer.path_out, "check_approximation")
        os.makedirs(out_dir, exist_ok=True)
        png_path = os.path.join(out_dir, f"check_approx_{i:04}.png")
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        logger.info(f"check_approximation plot → {png_path}")

    def error_debug(self, vars, i):
        """Same as parent + log bd and 1/bd, also save delta/beta to conv.csv.
        i=-1 is the initial-state call from BH and is always logged."""
        if i != -1 and not (i % self.error_step == 0 and self.error_step != -1):
            return
        err = self.min(vars["prb"], vars["obj"], vars["pos"], vars["proj"], vars["bd"])
        if self.rank == 0:
            bd = float(vars['bd'][0])
            inv_bd = 1.0 / bd if bd != 0 else float('inf')
            if 'delta_beta' not in self.table.columns:
                self.table['delta_beta'] = pd.NA
            if i == -1:
                logger.warning(f"Initial {err=:1.5e}  delta/beta={inv_bd:.1f}")
                self.table.loc[len(self.table)] = [i, err, 0, inv_bd]
            else:
                ittime = time.time() - self.time_start
                logger.warning(f"iter={i}: {ittime:.4f}sec {err=:1.5e}  delta/beta={inv_bd:.1f}")
                self.table.loc[len(self.table)] = [i, err, ittime, inv_bd]
            self.time_start = time.time()
            if hasattr(self, 'path_out'):
                name = f"{self.path_out}/conv.csv"
                os.makedirs(os.path.dirname(name), exist_ok=True)
                self.table.to_csv(name, index=False)

