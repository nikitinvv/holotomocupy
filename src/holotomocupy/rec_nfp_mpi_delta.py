"""
RecNFPDelta: cascade-based parameterization u = delta * (1 + i * bd) for NFP.

Adds a NEW cascade level F4: (prb, proj, bd, pos) -> (prb, proj*(1+i*bd), pos),
analogous to RecDelta wrt Rec. NFP has no Radon, so unlike RecDelta there is no
outside-cascade RT step (no gF5) — the cascade now runs F0..F4 end to end.

Variables:
  vars['proj'] = delta-like (REAL 2D float32; allocated by parent with obj_dtype
                 temporarily flipped to 'float32', then restored to 'complex64'
                 so cl_shift/cl_prop still see complex)
  vars['bd']   = scalar (cupy shape (1,) float32)
  vars['prb']  = probe (complex)
  vars['pos']  = positions (real)

Cascade at level 4: (prb, proj_real, bd, pos)
Cascade at level 3: (prb, proj_complex, pos)        -- after F4

args.rho is length 4: [proj, prb, pos, bd]
"""

import os
import numpy as np
import cupy as cp
import pandas as pd
import time

from .rec_nfp_mpi import RecNFP
from .utils import redot, reprod, mshow, mshow_polar, mshow_pos, logger
from .mpi_functions import *


class RecNFPDelta(RecNFP):
    # BH loop walks all four variables; parent helpers (compute_gradient, compute_beta,
    # apply_step, ...) iterate over this list, so no BH override needed.
    _var_names = ("prb", "proj", "pos", "bd")

    def __init__(self, args):
        if args.obj_dtype != 'complex64':
            raise ValueError(
                f"RecNFPDelta requires args.obj_dtype='complex64' "
                f"(cl_shift/cl_prop machinery is complex); got {args.obj_dtype!r}. "
                f"proj is reallocated as float32 internally."
            )
        if not hasattr(args, 'rho') or len(args.rho) != 4:
            raise ValueError(
                f"RecNFPDelta requires args.rho of length 4 "
                f"[proj, prb, pos, bd]; got {getattr(args, 'rho', None)}"
            )
        super().__init__(args)
        # Extend cascade with F4 (innermost). Parent apply_F_from / gF0..gF3 walk the
        # full self.F list, so the new level is picked up automatically.
        self.F      = [self.F0, self.F1, self.F2, self.F3, self.F4]
        self.gF     = [self.gF0, self.gF1, self.gF2, self.gF3, self.gF4]
        self.dF     = [self.dF0, self.dF1, self.dF2, self.dF3, self.dF4]
        self.d2F_dF = [self.d2F_dF0, self.d2F_dF1, self.d2F_dF2, self.d2F_dF3, self.d2F_dF4]
        self.rho_sq['bd'] = float(args.rho[3]) ** 2

    # ------------------------------------------------------------------ alloc
    def alloc_arrays(self):
        # Flip obj_dtype to float32 only during parent allocation so that vars['proj'],
        # grads['proj'], etas['proj'] come out real. cl_shift/cl_prop already exist
        # and saw 'complex64' in __init__, so they're unaffected.
        saved, self.obj_dtype = self.obj_dtype, 'float32'
        try:
            super().alloc_arrays()
        finally:
            self.obj_dtype = saved

        self.vars['bd']  = cp.zeros((1,), dtype='float32')
        self.grads['bd'] = cp.zeros((1,), dtype='float32')
        self.etas['bd']  = cp.zeros((1,), dtype='float32')

    # ============================================================ NEW F4 layer
    def F4(self, x):
        """In: (prb, proj_real, bd, pos)  Out: (prb, proj * (1+i*bd), pos)."""
        prb, proj, bd, pos = x
        proj_complex = proj * (1.0 + 1j * bd)
        return [prb, proj_complex, pos]

    def dF4(self, x, y, return_x=True):
        prb, proj, bd, pos = x
        yprb, yproj, ybd, ypos = y
        yproj_complex = (1.0 + 1j * bd) * yproj + 1j * proj * ybd
        if return_x:
            xproj_complex = proj * (1.0 + 1j * bd)
            return ([prb, xproj_complex, pos], [yprb, yproj_complex, ypos])
        return [yprb, yproj_complex, ypos]

    def d2F_dF4(self, x, y, z, w):
        """Innermost level — second-derivative cross term proj <-> bd, plus
        propagation of accumulator w (None at innermost call site)."""
        prb, proj, bd, pos = x
        _, yproj, ybd, _ = y
        _, zproj, zbd, _ = z

        yproj_complex = 1j * (yproj * zbd + zproj * ybd)

        wpos_out = None
        if w is not None and w[1] is not None:
            _, w_proj, w_bd, _ = w
            yproj_complex += (1.0 + 1j * bd) * w_proj + 1j * proj * w_bd

        # parent's d2F_dF3 unpacks w as 3-tuple; emit a zero pos contribution
        # to keep arities consistent.
        if wpos_out is None:
            wpos_out = cp.zeros_like(pos)
        return [None, yproj_complex, wpos_out]

    def gF4(self, x, y):
        """Adjoint of F4. In: y=(yprb, yproj_complex, ypos) at level 3.
        Out: 4-element [yprb, yproj_real, ybd_scalar, ypos]."""
        prb, proj, bd_arr, pos = x
        yprb, yproj_complex, ypos = y
        bd = bd_arr.reshape(())

        yproj_out = yproj_complex.real + bd * yproj_complex.imag
        ybd_out   = (cp.sum(proj * yproj_complex.imag)).reshape(1)

        return [yprb, yproj_out, ybd_out, ypos]

    # BH is inherited from RecNFP; it iterates over self._var_names = (prb, proj, pos, bd).

    # ============================================================ gradients (5-level cascade)
    def gradients_cascade(self, vars, grads):
        grads['prb'][:]  = 0
        grads['proj'][:] = 0
        grads['bd'][:]   = 0

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=4)
        def _gradients_cascade(self,
                               gradpos, gradprb, gradproj, gradbd,
                               d, pos, prb, proj, bd):
            self.cl_shift.coeff_cache_reset()
            x = [prb, proj, bd, pos]
            y = d
            for id in range(len(self.gF)):                    # 0..4
                y = self.gF[id](x, y)
            # y is 4-element at level 4: [yprb, yproj, ybd, ypos]
            gradprb[:]  += y[0]
            gradproj[:] += y[1]
            gradbd[:]   += y[2]
            gradpos[:]   = y[3]

        _gradients_cascade(self,
                           grads['pos'], grads['prb'], grads['proj'], grads['bd'],
                           self.data, vars['pos'], vars['prb'], vars['proj'], vars['bd'])

    def gradients(self, vars, grads):
        self.gradients_cascade(vars, grads)
        grads['prb'][:]  = cp.array(self.allreduce(grads['prb'].get()))
        grads['proj'][:] = cp.array(self.allreduce(grads['proj'].get()))
        grads['bd'][:]   = cp.array(self.allreduce(grads['bd'].get()))

    # ============================================================ hessian (5-level cascade)
    def hessian_cascade(self, vars, grads, etas):
        out = cp.zeros(1, dtype="float32")

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _hessian_cascade(
            self, out, d,
            x_pos, y_pos, z_pos,
            x_proj, y_proj, z_proj,
            x_prb, y_prb, z_prb,
            x_bd, y_bd, z_bd,
        ):
            self.cl_shift.coeff_cache_reset()
            x = [x_prb, x_proj, x_bd, x_pos]
            y = [y_prb, y_proj, y_bd, y_pos]
            z = [z_prb, z_proj, z_bd, z_pos]
            w = [None, None, None, None]
            y_is_z = y[0] is z[0]

            for id in range(1, len(self.F))[::-1]:           # 4, 3, 2, 1
                w = self.d2F_dF[id](x, y, z, w)
                fx, y = self.dF[id](x, y)
                if y_is_z:
                    z = y
                else:
                    z = self.dF[id](x, z, return_x=False)
                x = fx

            out[:] += self.d2F_dF[0](x, y, z, w, d)

        _hessian_cascade(
            self, out, self.data,
            vars['pos'],  grads['pos'],  etas['pos'],
            vars['proj'], grads['proj'], etas['proj'],
            vars['prb'],  grads['prb'],  etas['prb'],
            vars['bd'],   grads['bd'],   etas['bd'],
        )
        return out[0].get()

    # ============================================================ min (5-arg)
    def min(self, prb, proj, pos, bd):
        out = cp.zeros(1, dtype="float32")

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _min(self, out, pos, data, prb, proj, bd):
            self.cl_shift.coeff_cache_reset()
            x = [prb, proj, bd, pos]
            y = self.apply_F_from(x, 1)
            out[:] += self.F0(y, data)

        _min(self, out, pos, self.data, prb, proj, bd)
        return float(self.allreduce(np.array([out[0].get()], dtype='float32'))[0])

    # ============================================================ synthetic data
    def gen_sqrt_data(self, vars, out):
        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _gen_data(self, out, pos, prb, proj, bd):
            self.cl_shift.coeff_cache_reset()
            x = [prb, proj, bd, pos]
            y = self.apply_F_from(x, 1)
            out[:] = cp.abs(y)
        _gen_data(self, out, vars['pos'], vars['prb'], vars['proj'], vars['bd'])

    # ============================================================ error / vis
    def error_debug(self, vars, i):
        """Like parent + log bd and 1/bd, also writes delta_beta column to csv.
        i=-1 is the initial-state call from BH and is always logged."""
        if i != -1 and not (i % self.error_step == 0 and self.error_step != -1):
            return
        err = self.min(vars['prb'], vars['proj'], vars['pos'], vars['bd'])

        pos_err = (vars['pos'] - self.pos_init).get()
        all_pos_err = self.cl_mpi.comm.gather(pos_err, root=0)

        if self.rank != 0:
            return

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

        pos_err_all = np.concatenate(all_pos_err, axis=0)
        logger.warning(f"  pos err y: {np.array2string(pos_err_all[:, 0], precision=4, separator=', ')}")
        logger.warning(f"  pos err x: {np.array2string(pos_err_all[:, 1], precision=4, separator=', ')}")
        self.time_start = time.time()

        if hasattr(self, 'path_out'):
            name = f"{self.path_out}/conv_nfp_delta.csv"
            os.makedirs(os.path.dirname(name), exist_ok=True)
            self.table.to_csv(name, index=False)

    def vis_debug(self, vars, i, writer=None):
        if not (i % self.checkpoint_step == 0 and self.checkpoint_step != -1):
            return
        if writer is not None:
            if i > self.start_iter:
                writer.write_checkpoint(vars, i)
            return
        if self.rank == 0:
            if hasattr(self, 'path_out'):
                import tifffile
                from .utils import write_tiff
                tiff_dir = os.path.join(self.path_out, 'checkpoints_tiff')
                os.makedirs(tiff_dir, exist_ok=True)
                logger.info(f"Saving iter {i}: proj (delta), prb, bd to {tiff_dir}")
                write_tiff(vars['proj'],          f'{tiff_dir}/proj{i:04}')   # real delta
                write_tiff(cp.angle(vars['prb']), f'{tiff_dir}/prb{i:04}')
                np.save(f'{tiff_dir}/prb{i:04}.npy', vars['prb'].get())
                logger.warning(f"  iter={i}: bd={float(vars['bd'][0]):.6e}  delta/beta={1.0/float(vars['bd'][0]):.1f}")
            else:
                mshow(vars['proj'], True)
                mshow_polar(vars['prb'], True)
                mshow_pos(vars['pos'] - self.pos_init, True)
