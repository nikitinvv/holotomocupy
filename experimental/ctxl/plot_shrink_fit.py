#!/usr/bin/env python
"""Plot stored per-angle shrink vs. the tanh-approximated version used as
the initial guess for `vars['tp']` in rec_mpi_shrink.

Two input modes:
  1. Config-file mode (reads /exchange/shrink from args.in_file):
       python experimental/ctxl/plot_shrink_fit.py config_step6.conf
     Applies the same theta subsampling Reader does.

  2. .mat-file mode (reads shrink array directly from a Matlab file):
       python experimental/ctxl/plot_shrink_fit.py \
              /local/tomodata2/vnikitin/ctxl/ctxl_HT6d_4K_008p5nm_0001_/shapp.mat
     No subsampling — uses the array as-is. Handles both scipy-readable
     .mat versions and v7.3 (HDF5-based) via a fallback.

Runs the same fit as Rec.init_tp_from_shrink and writes shrink_fit.png
with 2 rows (y, x axes) × ndist columns.

Each subplot shows the stored per-angle shrink points, the fitted
tanh curve, and the fitted (A, k, B) plus RMS in the title.
"""

import os
import sys
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from holotomocupy.config import parse_args


K_CANDIDATES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def fit_tanh(t, y, k_candidates=K_CANDIDATES, constrain_B_zero=False):
    """Same fit as Rec.init_tp_from_shrink — grid over k, linear LS on (A, B).
    If `constrain_B_zero` (used for dist=0 by convention B[0]=0), fit only A.
    A is clipped to 0 (parametrization requires A >= 0). Returns (A, k, B).
    """
    best = None
    for k in k_candidates:
        u = np.tanh(k * t)
        if constrain_B_zero:
            # y ≈ A · u   →   A = (u·y) / (u·u), B fixed at 0.
            uu = float(np.dot(u, u))
            A_fit = float(np.dot(u, y) / uu) if uu > 0 else 0.0
            B_fit = 0.0
        else:
            M = np.column_stack([np.ones_like(t), u])
            (B_fit, A_fit), *_ = np.linalg.lstsq(M, y, rcond=None)
            B_fit, A_fit = float(B_fit), float(A_fit)
        y_pred = B_fit + A_fit * u
        res    = float(np.sum((y - y_pred) ** 2))
        if best is None or res < best[0]:
            best = (res, A_fit, float(k), B_fit)
    _, A, k, B = best
    if A < 0.0:
        A = 0.0
    return A, k, B


def load_shrink(in_file, ntheta, ndist, start_theta=0):
    """Read /exchange/shrink, subsample to `ntheta` angles the same way
    the Reader does. Returns (shrink[ntheta, ndist, 2], ids[ntheta])."""
    with h5py.File(in_file, 'r') as f:
        if '/exchange/shrink' not in f:
            raise KeyError(f'/exchange/shrink missing in {in_file}')
        raw = f['/exchange/shrink']
        print(f'  raw /exchange/shrink shape={raw.shape}, ndim={raw.ndim}')
        if raw.ndim == 3:
            data = raw[:, :ndist, :2].astype('float32')
        else:
            flat = raw[:, :ndist].astype('float32')
            data = np.broadcast_to(flat[..., None],
                                   (flat.shape[0], ndist, 2)).copy()
    ntheta_full = data.shape[0]
    step = ntheta_full / ntheta
    ids  = np.arange(start_theta, ntheta_full, step)[:ntheta].astype('int')
    return data[ids], ids


def load_shrink_mat(mat_file, ndist=None):
    """Read a shrink array from a Matlab .mat file (v4/v6/v7 or v7.3).

    Tries scipy.io.loadmat first; falls back to h5py for v7.3. Returns the
    first array-like dataset whose largest dimension looks like ntheta
    (>= 100). Result is shape (ntheta, ndist, 2) — if the source is 2D
    (ntheta, ndist), the y and x axes are broadcast to the same values.
    """
    arr = None
    src_shape = None
    src_key = None
    # Try scipy first.
    try:
        from scipy.io import loadmat
        mat = loadmat(mat_file, squeeze_me=True)
        for k, v in mat.items():
            if k.startswith('__'):
                continue
            if isinstance(v, np.ndarray) and v.ndim in (2, 3) and max(v.shape) >= 100:
                arr = np.asarray(v, dtype='float32')
                src_key = k
                src_shape = arr.shape
                break
    except (NotImplementedError, ValueError):
        # v7.3 files fail with NotImplementedError; loadmat's warning path
        # can also raise ValueError. Fall through to h5py.
        pass

    if arr is None:
        # v7.3 = HDF5 under the hood — walk for the first numeric dataset.
        with h5py.File(mat_file, 'r') as f:
            def visit(name, obj):
                nonlocal arr, src_key, src_shape
                if arr is not None:
                    return
                if isinstance(obj, h5py.Dataset) and obj.ndim in (2, 3) and max(obj.shape) >= 100:
                    v = obj[()]
                    # Matlab HDF5 arrays come back column-major → transpose so
                    # the largest dim is axis 0 (ntheta convention).
                    v = np.asarray(v, dtype='float32')
                    if v.ndim == 2 and v.shape[0] < v.shape[1]:
                        v = v.T
                    if v.ndim == 3 and v.shape[0] < v.shape[-1]:
                        v = np.transpose(v, (v.ndim - 1, *range(v.ndim - 1)))
                    arr = v
                    src_key   = name
                    src_shape = v.shape
            f.visititems(visit)

    if arr is None:
        raise RuntimeError(f'Could not find a shrink-like array in {mat_file}')

    print(f'  loaded {src_key!r} from {mat_file}  shape={src_shape}')

    # Normalize to (ntheta, ndist, 2).
    if arr.ndim == 2:
        ntheta_full, nd = arr.shape
        arr = np.broadcast_to(arr[:, :, None], (ntheta_full, nd, 2)).copy()

    if ndist is None:
        ndist = arr.shape[1]
    arr = arr[:, :ndist, :2]
    return arr


def main(source_path, out_png='shrink_fit.png', ndist_override=None):
    ext = os.path.splitext(source_path)[1].lower()
    if ext == '.mat':
        print(f'Reading shrink from {source_path} (Matlab .mat mode)')
        shrink = load_shrink_mat(source_path, ndist=ndist_override)
    elif ext in ('.h5', '.hdf5'):
        # Direct HDF5: read all angles from /exchange/shrink, no subsampling.
        print(f'Reading shrink from {source_path} (HDF5 mode)')
        with h5py.File(source_path, 'r') as f:
            if '/exchange/shrink' not in f:
                raise KeyError(f'/exchange/shrink missing in {source_path}')
            raw = f['/exchange/shrink']
            print(f'  raw /exchange/shrink shape={raw.shape}, ndim={raw.ndim}')
            if raw.ndim == 3:
                shrink = raw[:].astype('float32')
            else:
                flat = raw[:].astype('float32')
                shrink = np.broadcast_to(flat[..., None],
                                         (flat.shape[0], flat.shape[1], 2)).copy()
        if ndist_override is not None:
            shrink = shrink[:, :ndist_override, :]
    else:
        args = parse_args(source_path)
        print(f'Reading shrink from {args.in_file} (config mode: {source_path})')
        shrink, _ids = load_shrink(args.in_file, args.ntheta, args.ndist,
                                   args.start_theta)

    source_label = os.path.basename(source_path)
    ntheta, ndist, _ = shrink.shape
    print(f'  final shape: ntheta={ntheta}, ndist={ndist}')

    t         = np.arange(ntheta, dtype='float64') / max(ntheta - 1, 1)
    theta_idx = np.arange(ntheta)
    axis_names = ('y', 'x')

    fig, axes = plt.subplots(2, ndist, figsize=(4 * ndist, 6), sharex=True)
    if ndist == 1:
        axes = axes[:, np.newaxis]

    for d in range(ndist):
        for a in range(2):
            ax = axes[a, d]
            y_data = shrink[:, d, a].astype('float64')
            A, k, B = fit_tanh(t, y_data, constrain_B_zero=(d == 0))
            y_fit  = B + A * np.tanh(k * t)
            rms    = float(np.sqrt(np.mean((y_data - y_fit) ** 2)))
            ax.plot(theta_idx, y_data, 'o', markersize=2, alpha=0.5,
                    label='stored', color='C0')
            ax.plot(theta_idx, y_fit, '-', label='tanh fit',
                    color='C3', linewidth=1.5)
            ax.set_title(
                f'dist {d}, {axis_names[a]}\n'
                f'A={A:+.3e}  k={k:g}  B={B:+.3e}  RMS={rms:.2e}',
                fontsize=9)
            ax.grid(True, linestyle=':')
            ax.legend(fontsize=8, loc='best')
            if a == 1:
                ax.set_xlabel('theta idx')
            ax.set_ylabel('shrink')

    fig.suptitle(f'shrink stored vs tanh fit — {source_label}', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_png}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} <config_or_matfile> [out_png] [ndist]')
        sys.exit(1)
    src     = sys.argv[1]
    out_png = sys.argv[2] if len(sys.argv) > 2 else 'shrink_fit.png'
    ndist_o = int(sys.argv[3]) if len(sys.argv) > 3 else None
    main(src, out_png, ndist_o)
