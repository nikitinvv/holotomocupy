#!/usr/bin/env python
"""Compare two checkpoints by:
  (a) object mid-slice (delta + beta) + convergence curve
  (b) full forward chain — Radon, Shift+demag, Probe-multiply, Propagate, Data —
      saving per-stage diffs to one H5 and one PNG per stage.

Usage:
    python compare_iters.py <step6.conf> [--i1 128] [--i2 512]
                            [--js 0,500,1000] [--k 0]

Outputs (all in cfg.path_out):
    diff_iter{i2:04d}_minus_{i1:04d}.h5          obj diff (full volume)
    diff_iter{i2:04d}_minus_{i1:04d}.png         obj slices + convergence
    forward_diff_iter{i2:04d}_minus_{i1:04d}.h5  per-stage diffs (per selected angle)
    forward_stage{N}_<name>_iter{i2:04d}_minus_{i1:04d}.png   one PNG per stage
"""

import os
import sys
import argparse
import h5py
import numpy as np
import cupy as cp
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, '/home/beams/VNIKITIN/holotomocupy_mpi/src')
from holotomocupy.shift import Shift
from holotomocupy.shift_fft import ShiftFFT
from holotomocupy.tomo import Tomo
from holotomocupy.propagation import Propagation
from holotomocupy.config import parse_args

# Single-rank guard: this script reads checkpoints + runs the forward chain on
# one GPU. If launched under mpiexec (e.g. from polaris_run.sh) we let only
# rank 0 do the work — the other ranks immediately exit so they don't race on
# HDF5 writes or duplicate every figure.
try:
    from mpi4py import MPI
    _comm = MPI.COMM_WORLD
    _rank = _comm.Get_rank()
    if _rank != 0:
        sys.exit(0)
except ImportError:
    pass


# ============================================================ checkpoint I/O

def load_checkpoint(path):
    """Return (obj_complex, prb_complex, pos)."""
    with h5py.File(path, 'r') as f:
        obj_re = f['/obj_re'][:].astype('float32')
        obj_im = f['/obj_im'][:].astype('float32') if '/obj_im' in f else np.zeros_like(obj_re)
        obj = (obj_re + 1j * obj_im).astype('complex64')
        prb_abs   = f['/prb_abs'][:].astype('float32')
        prb_phase = f['/prb_phase'][:].astype('float32')
        prb = (prb_abs * np.exp(1j * prb_phase)).astype('complex64')
        pos = f['/pos'][:].astype('float32')
    return obj, prb, pos


def load_convergence(path_out):
    csv_path = os.path.join(path_out, 'conv.csv')
    if not os.path.exists(csv_path):
        return None, None
    df = pd.read_csv(csv_path)
    df = df[df['iter'] >= 0]
    return df['iter'].to_numpy(), df['err'].to_numpy()


# ============================================================ obj-only figure

def plot_obj_compare(re1, im1, re2, im2, diff_re, diff_im,
                     path_out, i1, i2, out_png):
    nz = re1.shape[0]
    mid = nz // 2
    iters, errs = load_convergence(path_out)
    has_conv = iters is not None and len(iters) > 0

    if has_conv:
        fig = plt.figure(figsize=(15, 13), constrained_layout=True)
        gs  = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 0.55])
        ax_slice = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]
        ax_conv  = fig.add_subplot(gs[2, :])
    else:
        fig, ax_slice = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
        ax_slice = ax_slice.tolist()
        ax_conv  = None

    def panel(ax, img, title, cmap='gray', sym=False):
        if sym:
            v = float(np.percentile(np.abs(img), 99))
            im = ax.imshow(img, cmap=cmap, vmin=-v, vmax=+v)
        else:
            vmin = float(np.percentile(img, 1))
            vmax = float(np.percentile(img, 99))
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        return im

    for row, (a, b, d, lbl) in enumerate([
            (re1, re2, diff_re, 'delta'),
            (im1, im2, diff_im, 'beta '),
    ]):
        im0 = panel(ax_slice[row][0], a[mid], f'{lbl} @ iter {i1}')
        im1_ = panel(ax_slice[row][1], b[mid], f'{lbl} @ iter {i2}')
        imd = panel(ax_slice[row][2], d[mid],
                    f'{lbl} diff ({i2} - {i1})', cmap='RdBu_r', sym=True)
        fig.colorbar(im0,  ax=ax_slice[row][0], shrink=0.85)
        fig.colorbar(im1_, ax=ax_slice[row][1], shrink=0.85)
        fig.colorbar(imd,  ax=ax_slice[row][2], shrink=0.85)

    if has_conv:
        ax_conv.plot(iters, errs, lw=1.2)
        ax_conv.set_yscale('log')
        ax_conv.set_xlabel('iteration'); ax_conv.set_ylabel('err (log)')
        ax_conv.grid(True, which='both', alpha=0.3)
        ax_conv.set_title('Convergence (conv.csv)', fontsize=10)
        for it, col, lbl in [(i1, 'tab:blue', f'iter {i1}'),
                             (i2, 'tab:red',  f'iter {i2}')]:
            idx = int(np.argmin(np.abs(iters - it)))
            ax_conv.axvline(iters[idx], color=col, ls='--', lw=1.0, alpha=0.7)
            ax_conv.plot(iters[idx], errs[idx], 'o', color=col, markersize=7, label=lbl)
        ax_conv.legend(loc='best', fontsize=9)

    fig.suptitle(f'{os.path.basename(path_out)}  —  middle z-slice (z={mid})',
                 fontsize=12)
    plt.savefig(out_png, dpi=120)
    plt.close(fig)


# ============================================================ forward stages

def run_forward_per_angle(obj_d, prb_d, pos_full, j_ids, theta, eff_demag,
                          cl_tomo, cl_shift, cl_prop, ndist, nchunk_z):
    """For each j in j_ids run the forward chain once, return dict of stage arrays.

    Radon is computed in z-slabs of size nchunk_z to bound GPU memory —
    cl_tomo MUST be built with nz==nchunk_z so its internal buffers match.
    """
    nzobj  = obj_d.shape[0]
    nobj   = obj_d.shape[2]
    nz_det = prb_d.shape[1]
    n_det  = prb_d.shape[2]
    n_pick = len(j_ids)
    j_arr  = np.asarray(j_ids, dtype='int32')

    # Stash only selected angles on CPU, full nzobj per angle.
    proj_per_angle = np.empty((n_pick, nzobj, nobj), dtype='complex64')

    for z0 in range(0, nzobj, nchunk_z):
        z1   = min(z0 + nchunk_z, nzobj)
        actual = z1 - z0
        if actual == nchunk_z:
            slab = obj_d[z0:z1]
        else:
            # pad the trailing z-slab so Tomo's pre-allocated buffer fits
            slab = cp.zeros((nchunk_z, nobj, nobj), dtype=obj_d.dtype)
            slab[:actual] = obj_d[z0:z1]
        proj_slab = cl_tomo.R(slab)                       # (ntheta, nchunk_z, nobj)
        sel = cp.asnumpy(proj_slab[j_arr, :actual, :])    # (n_pick, actual, nobj)
        proj_per_angle[:, z0:z1, :] = sel
        del slab, proj_slab, sel
    cp.get_default_memory_pool().free_all_blocks()

    out = {
        'radon': proj_per_angle,                                                       # (n_pick, nzobj, nobj)
        'shift': np.empty((n_pick, ndist, nz_det, n_det), dtype='complex64'),
        'probe': np.empty((n_pick, ndist, nz_det, n_det), dtype='complex64'),
        'prop':  np.empty((n_pick, ndist, nz_det, n_det), dtype='complex64'),
        'amp':   np.empty((n_pick, ndist, nz_det, n_det), dtype='float32'),
    }

    for i, j in enumerate(j_ids):
        proj_j_d = cp.asarray(proj_per_angle[i:i+1])                # (1, nzobj, nobj)
        for k in range(ndist):
            r_jk = pos_full[j:j+1, k]                               # (1, 2)
            m_jk = (1.0 / eff_demag[j, k])[None].astype('float32')  # (1, 2)
            shifted = cl_shift.curlyS(proj_j_d, cp.asarray(r_jk), cp.asarray(m_jk))[0]
            out['shift'][i, k] = shifted.get()

            wave = prb_d[k] * cp.exp(1j * shifted)
            out['probe'][i, k] = wave.get()

            propagated = cl_prop.D(wave, k)
            out['prop'][i, k] = propagated.get()
            out['amp'][i, k]  = cp.abs(propagated).get()
        del proj_j_d
    cp.get_default_memory_pool().free_all_blocks()
    return out


def plot_probe_compare(prb1, prb2, out_dir, i1, i2):
    """Compare two complex probes per-distance, four figures: real, imag, abs, phase.

    Each figure: ndist rows × 3 cols (iter1, iter2, diff). iter2 is phase-aligned
    to iter1 per-distance via phi = angle(sum(conj(prb1[k]) * prb2[k])),
    prb2_aligned[k] = prb2[k] * exp(-i*phi). Diff = iter2_aligned - iter1.
    """
    ndist = prb1.shape[0]
    prb2_aln = prb2.copy()
    phis = np.empty(ndist, dtype='float32')
    for k in range(ndist):
        phi = float(np.angle((np.conj(prb1[k]) * prb2[k]).sum()))
        phis[k] = phi
        prb2_aln[k] = prb2[k] * np.exp(-1j * phi)
    diff = prb2_aln - prb1

    components = [
        ('real',  lambda x: x.real,                       'gray',   False),
        ('imag',  lambda x: x.imag,                       'gray',   False),
        ('abs',   lambda x: np.abs(x),                    'gray',   False),
        ('phase', lambda x: np.angle(x),                  'twilight', True),  # wrap-aware cmap
    ]
    for comp_name, fn, cmap, is_phase in components:
        x1 = fn(prb1)         # (ndist, nz, n)
        x2 = fn(prb2_aln)
        if comp_name == 'phase':
            dd = np.angle(np.exp(1j * (x2 - x1)))  # wrap diff to (-pi, pi]
        else:
            dd = x2 - x1

        nrows, ncols = ndist, 3
        fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.5 * nrows),
                                 constrained_layout=True, squeeze=False)
        if comp_name == 'phase':
            vmin, vmax = -np.pi, np.pi
            vlim = np.pi
        else:
            vmin = float(np.percentile(np.stack([x1, x2]), 1))
            vmax = float(np.percentile(np.stack([x1, x2]), 99))
            vlim = float(np.percentile(np.abs(dd), 99))

        for k in range(ndist):
            im0 = axes[k, 0].imshow(x1[k], cmap=cmap, vmin=vmin, vmax=vmax)
            im1 = axes[k, 1].imshow(x2[k], cmap=cmap, vmin=vmin, vmax=vmax)
            im2 = axes[k, 2].imshow(dd[k], cmap='RdBu_r', vmin=-vlim, vmax=+vlim)
            for c, im in zip(range(3), [im0, im1, im2]):
                axes[k, c].set_xticks([]); axes[k, c].set_yticks([])
                fig.colorbar(im, ax=axes[k, c], shrink=0.85)
            axes[k, 0].set_ylabel(f'dist k={k}', fontsize=10)
        rel = float(np.linalg.norm(dd) / np.linalg.norm(x1)) if np.linalg.norm(x1) > 0 else float('nan')
        axes[0, 0].set_title('iter 1', fontsize=11)
        axes[0, 1].set_title('iter 2 (phase-aligned)', fontsize=11)
        diff_title = ('diff (wrapped to ±π)' if comp_name == 'phase'
                      else 'diff (iter2 − iter1)')
        axes[0, 2].set_title(f'{diff_title}   ||diff||/||iter1|| = {rel:.3e}', fontsize=11)
        phi_str = ', '.join(f'{p:+.3f}' for p in phis)
        fig.suptitle(f'probe — {comp_name}   [per-dist phi applied to iter2: {phi_str}]',
                     fontsize=11)
        out_png = os.path.join(out_dir,
                               f'probe_{comp_name}_iter{i2:04d}_minus_{i1:04d}.png')
        plt.savefig(out_png, dpi=110)
        plt.close(fig)
        print(f'  saved {out_png}')


def panel_grid(arrs_iter1, arrs_iter2, j_ids, k_show, stage_name,
               out_png, complex_part='abs', align_phase=False):
    """3-col grid: iter1 / iter2 / diff for each angle (one distance).

    align_phase=True (complex inputs only) removes a per-angle constant phase
    from iter2 before extracting real/imag: phi = angle(sum(conj(a)*b)),
    b_aligned = b * exp(-i*phi). Minimises ||a - b_aligned||^2 per panel.
    """
    n_pick = len(j_ids)

    # Slice distance axis if present, keep complex for now
    a_c = arrs_iter1[:, k_show] if arrs_iter1.ndim == 4 else arrs_iter1
    b_c = arrs_iter2[:, k_show] if arrs_iter2.ndim == 4 else arrs_iter2

    phis = None
    if align_phase and np.iscomplexobj(a_c):
        b_c = b_c.copy()
        phis = np.empty(n_pick, dtype='float32')
        for i in range(n_pick):
            phi = float(np.angle((np.conj(a_c[i]) * b_c[i]).sum()))
            phis[i] = phi
            b_c[i] = b_c[i] * np.exp(-1j * phi)

    def part(x):
        if not np.iscomplexobj(x): return x
        if   complex_part == 'abs':   return np.abs(x)
        elif complex_part == 'real':  return x.real
        elif complex_part == 'imag':  return x.imag
        elif complex_part == 'phase': return np.angle(x)
        else: raise ValueError(complex_part)

    x1 = part(a_c)
    x2 = part(b_c)
    if complex_part == 'phase':
        dd = np.angle(np.exp(1j * (x2 - x1)))   # wrap diff into (-pi, pi]
    else:
        dd = x2 - x1

    fig, axes = plt.subplots(n_pick, 3, figsize=(11, 3.5 * n_pick),
                             constrained_layout=True, squeeze=False)
    if complex_part == 'phase':
        vmin, vmax, vlim = -np.pi, np.pi, np.pi
        cmap_img = 'twilight'
    else:
        vmin = float(np.percentile(np.stack([x1, x2]), 1))
        vmax = float(np.percentile(np.stack([x1, x2]), 99))
        vlim = float(np.percentile(np.abs(dd), 99))
        cmap_img = 'gray'
    for i, j in enumerate(j_ids):
        im0 = axes[i, 0].imshow(x1[i], cmap=cmap_img, vmin=vmin, vmax=vmax)
        im1 = axes[i, 1].imshow(x2[i], cmap=cmap_img, vmin=vmin, vmax=vmax)
        im2 = axes[i, 2].imshow(dd[i], cmap='RdBu_r', vmin=-vlim, vmax=+vlim)
        for c, im in zip(range(3), [im0, im1, im2]):
            axes[i, c].set_xticks([]); axes[i, c].set_yticks([])
            fig.colorbar(im, ax=axes[i, c], shrink=0.85)
        axes[i, 0].set_ylabel(f'angle j={j}', fontsize=10)
    # Relative error: ||iter2 − iter1|| / ||iter1||  (on the plotted component)
    a64 = x1.astype(np.float64)
    d64 = dd.astype(np.float64)
    norm_d = float(np.linalg.norm(d64))
    norm_a = float(np.linalg.norm(a64))
    rel    = norm_d / norm_a if norm_a > 0 else float('nan')

    axes[0, 0].set_title('iter 1', fontsize=11)
    axes[0, 1].set_title('iter 2' + (' (phase-aligned)' if phis is not None else ''), fontsize=11)
    axes[0, 2].set_title(f'diff (iter2 − iter1)   ||diff||/||iter1|| = {rel:.3e}',
                         fontsize=11)
    suffix = '' if not np.iscomplexobj(arrs_iter1) else f'  ({complex_part})'
    if phis is not None:
        suffix += f'  [phi per angle = {[f"{p:.3f}" for p in phis]}]'
    fig.suptitle(f'{stage_name}  —  distance k={k_show}{suffix}', fontsize=11)
    plt.savefig(out_png, dpi=110)
    plt.close(fig)


# ============================================================ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config',   help='step6 config file')
    ap.add_argument('--i1',     type=int, default=128)
    ap.add_argument('--i2',     type=int, default=512)
    ap.add_argument('--js',     default='0,500,1000',
                    help='comma-separated angle indices (in the *subsampled* range)')
    ap.add_argument('--k',      type=int, default=0,
                    help='distance index to show in per-stage PNG figures (default 0)')
    args = ap.parse_args()

    j_list = [int(s) for s in args.js.split(',')]
    cfg = parse_args(args.config)

    # ---- read geometry from the steps15 HDF5 (= cfg.in_file) ----
    with h5py.File(cfg.in_file, 'r') as f:
        z1                      = f['/exchange/z1'][:].astype('float64')
        focustodetectordistance = float(f['/exchange/focusdetectordistance'][0])
        detector_pixelsize      = float(f['/exchange/detector_pixelsize'][0])
        energy                  = float(f['/exchange/energy'][0])
        theta_raw               = f['/exchange/theta'][:, 0].astype('float32')
        shrink_nd               = f['/exchange/shrink'][:].astype('float32')
        cshifts                 = f['/exchange/cshifts_final'][:].astype('float32')

    ndist               = z1.size
    magnifications      = focustodetectordistance / z1
    norm_mag            = (magnifications / magnifications[0]).astype('float32')
    z2                  = focustodetectordistance - z1
    distances           = ((z1 * z2) / focustodetectordistance * norm_mag**2).astype('float32')
    wavelength          = 1.24e-9 / energy
    voxelsize           = detector_pixelsize / magnifications[0] * (2**cfg.bin)
    n, nobj, bin_       = cfg.n, cfg.nobj, cfg.bin
    ntheta, start_theta = cfg.ntheta, cfg.start_theta

    # Subsample angles same way reader.py does it
    ntheta0 = len(theta_raw)
    ids     = np.arange(start_theta, ntheta0, ntheta0 / ntheta)[:ntheta].astype('int')
    theta   = (-theta_raw[ids] / 180.0 * np.pi).astype('float32')

    # Pos at the bin scale + rotation_center_shift, matching step6 / reader convention
    scale   = 1.0 / 2**bin_
    pos_all = cshifts[ids] * scale
    pos_all[..., 1] += cfg.rotation_center_shift * scale + 0.5 * (scale - 1)
    shrink_sub = shrink_nd[ids]
    eff_demag  = (1 + shrink_sub) / norm_mag[None, :, None]          # (ntheta, ndist, 2)

    # ---- load checkpoints ----
    ckpt_dir = os.path.join(cfg.path_out, 'checkpoints')
    p1 = os.path.join(ckpt_dir, f'checkpoint_{args.i1:04d}.h5')
    p2 = os.path.join(ckpt_dir, f'checkpoint_{args.i2:04d}.h5')
    print(f'loading {p1}')
    obj1, prb1, pos1 = load_checkpoint(p1)
    print(f'loading {p2}')
    obj2, prb2, pos2 = load_checkpoint(p2)
    # Checkpoint pos is already at the subsampled cfg.ntheta size AND already
    # contains rotation_center_shift baked in (reader.read_pos applies it before
    # BH starts; writer stores vars['pos'] as-is afterwards).
    assert pos1.shape[0] == ntheta, \
        f"unexpected pos shape {pos1.shape}: expected first axis = ntheta={ntheta}"

    # ---- single output folder for ALL h5 + png from this comparison ----------
    out_dir = os.path.join(cfg.path_out,
                           f'compare_iter{args.i1:04d}_vs_{args.i2:04d}')
    os.makedirs(out_dir, exist_ok=True)
    print(f'output folder → {out_dir}')

    # ---- obj diff h5 + figure ------------------------------------------------
    re1, im1 = obj1.real, obj1.imag
    re2, im2 = obj2.real, obj2.imag
    diff_re, diff_im = re2 - re1, im2 - im1
    obj_h5  = os.path.join(out_dir, f'diff_iter{args.i2:04d}_minus_{args.i1:04d}.h5')
    obj_png = obj_h5.replace('.h5', '.png')
    with h5py.File(obj_h5, 'w') as f:
        f.create_dataset('/obj_re', data=diff_re)
        f.create_dataset('/obj_im', data=diff_im)
        f.attrs['iter1'] = args.i1; f.attrs['iter2'] = args.i2
        f.attrs['source_path'] = cfg.path_out
    plot_obj_compare(re1, im1, re2, im2, diff_re, diff_im,
                     cfg.path_out, args.i1, args.i2, obj_png)
    print(f'obj diff h5  → {obj_h5}')
    print(f'obj diff png → {obj_png}')

    # ---- probe comparison: real, imag, abs, phase ----------------------------
    print('plotting probe comparisons (real, imag, abs, phase)')
    plot_probe_compare(prb1, prb2, out_dir, args.i1, args.i2)

    # ---- build operators -----------------------------------------------------
    print(f'building operators (n={n}, nobj={nobj}, ndist={ndist}, ntheta={ntheta})')
    if cfg.shift_type == 'fft':
        cl_shift = ShiftFFT(n, nobj, n, nobj, 'complex64')
    else:
        cl_shift = Shift(n, nobj, n, nobj, 'complex64')
    cl_prop = Propagation(n, n, 1, ndist, wavelength, voxelsize,
                          cp.asarray(distances, dtype='float32'))
    # Tomo's `nz` arg sets the z-slab size for R(obj); use cfg.nchunk to bound
    # GPU memory. run_forward_per_angle below loops over z in nchunk-sized slabs.
    cl_tomo = Tomo(nobj, cfg.nchunk, theta, mask_r=cfg.mask if cfg.mask > 0 else 1.0)

    # ---- forward chain per checkpoint ----------------------------------------
    print(f'running forward chain for iter {args.i1}')
    o1_d = cp.asarray(obj1)
    p1_d = cp.asarray(prb1)
    f1   = run_forward_per_angle(o1_d, p1_d, cp.asarray(pos1),
                                 j_list, theta, eff_demag,
                                 cl_tomo, cl_shift, cl_prop, ndist, cfg.nchunk)
    del o1_d, p1_d
    cp.get_default_memory_pool().free_all_blocks()

    print(f'running forward chain for iter {args.i2}')
    o2_d = cp.asarray(obj2)
    p2_d = cp.asarray(prb2)
    f2   = run_forward_per_angle(o2_d, p2_d, cp.asarray(pos2),
                                 j_list, theta, eff_demag,
                                 cl_tomo, cl_shift, cl_prop, ndist, cfg.nchunk)
    del o2_d, p2_d
    cp.get_default_memory_pool().free_all_blocks()

    # ---- read measured sqrt(data) for the final stage ------------------------
    print('reading measured sqrt(data) for final comparison')
    sqd = np.empty((len(j_list), ndist, n, n), dtype='float32')
    with h5py.File(cfg.in_file, 'r') as f:
        for i, j in enumerate(j_list):
            j_full = int(ids[j])
            for k in range(ndist):
                ds = f[f'/exchange/pdata{k}_{bin_}']
                sqd[i, k] = np.sqrt(np.clip(ds[j_full, :n, :n].astype('float32'), 0, None))

    # ---- save per-stage diffs to one h5 + one png each -----------------------
    stages = [
        ('radon', '1_radon'),
        ('shift', '2_shift'),
        ('probe', '3_probe'),
        ('prop',  '4_propagate'),
        ('amp',   '5_amp'),
    ]
    fwd_h5 = os.path.join(out_dir,
                          f'forward_diff_iter{args.i2:04d}_minus_{args.i1:04d}.h5')
    print(f'writing forward diffs → {fwd_h5}')
    with h5py.File(fwd_h5, 'w') as f:
        f.attrs['iter1'] = args.i1; f.attrs['iter2'] = args.i2
        f.attrs['j_list']      = np.array(j_list, dtype='int32')
        f.attrs['source_path'] = cfg.path_out
        for key, name in stages:
            a1, a2 = f1[key], f2[key]
            diff = (a2 - a1)
            f.create_dataset(f'/{name}/iter1', data=a1)
            f.create_dataset(f'/{name}/iter2', data=a2)
            f.create_dataset(f'/{name}/diff',  data=diff)
        # data residual (|prop| - sqrt(data))
        amp1, amp2 = f1['amp'], f2['amp']
        res1 = amp1 - sqd
        res2 = amp2 - sqd
        f.create_dataset('/6_data_residual/iter1', data=res1)
        f.create_dataset('/6_data_residual/iter2', data=res2)
        f.create_dataset('/6_data_residual/diff',  data=res2 - res1)
        f.create_dataset('/6_data_residual/sqrt_data', data=sqd)

    # ---- one PNG per stage ----------------------------------------------------
    print(f'plotting per-stage figures (distance k={args.k})')
    for key, name in stages:
        # Stages 3 (probe) and 4 (prop): four figures (real, imag, abs, phase),
        # per-angle global phase alignment so the abs/phase split is meaningful.
        # Other complex stages: real + imag, no alignment.
        # Real-valued stages: single magnitude figure.
        align = key in ('probe', 'prop')
        if not np.iscomplexobj(f1[key]):
            parts = ['abs']
        elif align:
            parts = ['real', 'imag', 'abs', 'phase']
        else:
            parts = ['real', 'imag']
        for part in parts:
            suffix = f'_{part}' if len(parts) > 1 else ''
            out_png = os.path.join(
                out_dir,
                f'forward_stage_{name}{suffix}_iter{args.i2:04d}_minus_{args.i1:04d}.png')
            title   = f'{name}  ({part})' if len(parts) > 1 else name
            panel_grid(f1[key], f2[key], j_list, args.k, title,
                       out_png, complex_part=part, align_phase=align)
            print(f'  saved {out_png}')

    # residual figure
    out_png = os.path.join(
        out_dir,
        f'forward_stage_6_data_residual_iter{args.i2:04d}_minus_{args.i1:04d}.png')
    panel_grid(amp1 - sqd, amp2 - sqd, j_list, args.k,
               '6_data_residual (|prop| − sqrt(data))', out_png, complex_part='abs')
    print(f'  saved {out_png}')

    print()
    print('Per-stage  ||iter2 - iter1|| / ||iter1||  (no phase alignment)')
    cols = [f'k={k}' for k in range(ndist)] + ['all']
    hdr  = f'  {"stage":18s}  ' + '  '.join(f'{c:>13s}' for c in cols)
    print(hdr)

    def rel_per_dist(a, d):
        """Return list of len ndist+1 with ||d||/||a|| for each k and overall."""
        out = []
        for k in range(ndist):
            ak = a[:, k] if a.ndim == 4 else a            # radon: no dist axis
            dk = d[:, k] if d.ndim == 4 else d
            nd_, na_ = float(np.linalg.norm(dk)), float(np.linalg.norm(ak))
            out.append(nd_ / na_ if na_ > 0 else float('nan'))
            if a.ndim != 4:
                # radon: same value for every "k" column — just copy once and
                # break out so it's printed only once aligned under 'all'
                return [float('nan')] * ndist + [out[0]]
        nd_, na_ = float(np.linalg.norm(d)), float(np.linalg.norm(a))
        out.append(nd_ / na_ if na_ > 0 else float('nan'))
        return out

    # 0_obj: full complex volume diff (no distance axis)
    nd_obj = float(np.linalg.norm(obj2 - obj1))
    na_obj = float(np.linalg.norm(obj1))
    rel_obj = nd_obj / na_obj if na_obj > 0 else float('nan')
    cells_obj = '  '.join('     —       ' for _ in range(ndist)) + f'  {rel_obj:13.4e}'
    print(f'  {"0_obj":18s}  {cells_obj}')

    for key, name in stages:
        a   = f1[key].astype(np.complex128 if np.iscomplexobj(f1[key]) else np.float64)
        d   = (f2[key] - f1[key]).astype(a.dtype)
        row = rel_per_dist(a, d)
        cells = '  '.join('     —       ' if (isinstance(v, float) and np.isnan(v))
                          else f'{v:13.4e}' for v in row)
        print(f'  {name:18s}  {cells}')

    res1 = (amp1 - sqd).astype(np.float64)
    res2 = (amp2 - sqd).astype(np.float64)
    d    = res2 - res1
    row  = rel_per_dist(res1, d)                          # (n_pick, ndist, n, n)
    cells = '  '.join(f'{v:13.4e}' for v in row)
    print(f'  {"6_data_residual":18s}  {cells}')


if __name__ == '__main__':
    main()
