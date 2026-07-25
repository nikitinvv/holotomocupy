#!/usr/bin/env python
"""Single-distance test of per-axis cumulative shrinkage through Tomo + Shift.

Builds a small 3-D phantom, forward Radon projects at NTHETA angles, then
applies cubic Shift.S per projection with a per-angle, per-axis magnification
where the cumulative shrink ramps linearly from 0 at θ_idx=0 to
(SHRINK_V_END, SHRINK_H_END) at θ_idx=NTHETA-1. Renders the first distance at
3 angles (0, NTHETA/2, NTHETA-1) so the per-axis cumulative scaling is visible.

Output: tests/shift/per_axis_mag_out.h5
  /phantom              (NZ_OBJ, NOBJ, NOBJ) float32
  /sino_object          (NTHETA, NZ_OBJ, NOBJ) float32 — Radon at object scale
  /sino_per_dist_cubic  (NTHETA, NDIST, NZ, N) float32 — per-(θ, k) per-axis m
  /m_per_angle_dist     (NTHETA, NDIST, 2)    float32 — (my, mx) per (θ, k)
  /shrink_h_per_angle   (NTHETA,)              float32
  /shrink_v_per_angle   (NTHETA,)              float32
  /norm_mag             (NDIST,)               float32

Run with:
  python tests/shift/test_per_axis_mag.py
"""

import os
import sys
import numpy as np
import cupy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from holotomocupy.shift     import Shift
from holotomocupy.shift_fft import ShiftFFT
from holotomocupy.tomo      import Tomo


# Geometry — small enough to run in seconds; large enough to see scaling effects.
# Object grid (Radon input) is 2× the data grid so that at norm_mag[5] ≈ 0.5
# (demag ≈ 2) the data exactly covers the full object FOV — same shape as
# ctxl's nobj=3264, n=2048 (binned).
N      = 256        # data in-plane size (x = column dim of sinogram output)
NOBJ   = 512        # object in-plane size (x = column dim of Radon input)
NZ     = 256        # data z size (row dim of sinogram output)
NZ_OBJ = 512        # object z size (row dim of Radon input)
NTHETA = 180

# Cumulative-per-angle shrink across ALL distances. Each distance's rotation
# adds its own INC_* on top of the cumulative left over from previous
# distances (Peter's `cumsum(shrink_list)` convention applied per rotation).
#
#   INC_H[k] = increment added during dist k's NTHETA-angle rotation
#   INC_V[k] = same for vertical
#
# Per-distance start/end cumulative shrink:
#   start_h[k] = sum(INC_H[:k])              (0 at dist 0)
#   end_h[k]   = sum(INC_H[:k+1])            (carry into dist k+1's start)
#
# With the values below:
#   dist 0: shrink_h ramps 0.00 → 0.50, shrink_v 0.00 → 1.00
#   dist 1: shrink_h ramps 0.50 → 0.75, shrink_v 1.00 → 1.50
INC_H    = np.array([0.50, 0.25], dtype='float32')
INC_V    = np.array([1.00, 0.50], dtype='float32')
NORM_MAG = np.array([1.0,  1.0 ], dtype='float32')
NDIST    = len(INC_H)

# Angles to render in the per-projection plot.
ANGLE_IDS_FRACTIONS = [0.0, 0.5, 1.0]   # → 0, NTHETA/2, NTHETA-1


def build_phantom(nz, n):
    """3-D phantom with y- and x-distinct features so axis scaling is visible.

    Layers:
      * cube outline (centered) — common reference
      * vertical bars at fixed x positions (sensitive to x-axis mag)
      * horizontal bars at fixed y positions (sensitive to y-axis mag)
      * a few dense spots off-center (so rotation produces obvious motion)
    """
    obj = np.zeros((nz, n, n), dtype='float32')
    cy, cx = n // 2, n // 2

    # Central cube
    half = n // 6
    obj[nz // 2 - half // 2 : nz // 2 + half // 2,
        cy - half : cy + half,
        cx - half : cx + half] = 0.5

    # Vertical bars (constant in y, narrow in x — only x-mag distorts spacing)
    for dx_frac in (-0.30, -0.15, 0.15, 0.30):
        x_idx = int(cx + dx_frac * n)
        obj[:, cy - n // 4 : cy + n // 4, x_idx - 1 : x_idx + 2] += 1.0

    # Horizontal bars (constant in x, narrow in y — only y-mag distorts spacing)
    for dy_frac in (-0.30, -0.15, 0.15, 0.30):
        y_idx = int(cy + dy_frac * n)
        obj[:, y_idx - 1 : y_idx + 2, cx - n // 4 : cx + n // 4] += 1.0

    # Off-axis blobs — rotate visibly across 180 angles.
    for dy_frac, dx_frac in [(0.35, 0.0), (-0.35, 0.0), (0.0, 0.35), (0.0, -0.35)]:
        yy = int(cy + dy_frac * n)
        xx = int(cx + dx_frac * n)
        rr = 4
        Y, X = np.ogrid[-rr:rr+1, -rr:rr+1]
        mask2d = (Y*Y + X*X) <= rr*rr
        obj[nz // 4 : 3 * nz // 4, yy - rr : yy + rr + 1, xx - rr : xx + rr + 1] += 2.0 * mask2d
    return obj


def per_axis_mag(norm_mag, shrink_v, shrink_h):
    """m of shape (ndist, 2): axis 1 is (my, mx). Matches what steps15 builds."""
    m = np.empty((len(norm_mag), 2), dtype='float32')
    m[:, 0] = (1 + shrink_v) / norm_mag      # my
    m[:, 1] = (1 + shrink_h) / norm_mag      # mx
    return m


def apply_shift_per_dist(shift_op, sino_obj, m_per_angle_dist):
    """Apply shift_op.S to each projection with its per-angle, per-distance m.

    Input  sino_obj         : (ntheta, NZ_OBJ, NOBJ)  — object pixel scale.
           m_per_angle_dist : (ntheta, ndist, 2)      — (my, mx) per (θ, k).
    Output                  : (ntheta, ndist, NZ, N)  — per-distance data scale.
    """
    ntheta, ndist, _ = m_per_angle_dist.shape
    out = np.empty((ntheta, ndist, shift_op.nz, shift_op.n), dtype='float32')

    # Promote to complex64 (Shift internal dtype).
    c_in   = cp.asarray(sino_obj, dtype='complex64')
    r_zero = cp.zeros((ntheta, 2), dtype='float32')

    for k in range(ndist):
        m_k = cp.ascontiguousarray(cp.asarray(m_per_angle_dist[:, k], dtype='float32'))
        # curlyS handles its own prefilter (cubic) / identity (fft) — input is
        # always treated as image samples, so cubic and fft outputs are
        # directly comparable in amplitude.
        out_k = shift_op.curlyS(c_in, r_zero, m_k)
        out[:, k] = cp.asnumpy(out_k).real.astype('float32')
    return out


def main():
    print(f'building phantom {NZ_OBJ}x{NOBJ}x{NOBJ}')
    phantom = build_phantom(NZ_OBJ, NOBJ)

    print(f'forward Radon at {NTHETA} angles on object grid (NZ_OBJ={NZ_OBJ}, NOBJ={NOBJ})')
    theta = np.linspace(0, np.pi, NTHETA, endpoint=False).astype('float32')
    cl_tomo = Tomo(NOBJ, NZ_OBJ, theta, mask_r=0.9)
    sino_obj_gpu = cl_tomo.R(cp.asarray(phantom))
    sino_obj = cp.asnumpy(sino_obj_gpu).real.astype('float32')   # (NTHETA, NZ_OBJ, NOBJ)
    del sino_obj_gpu

    # Build cumulative shrink per axis.
    # Per-distance start cumulative (= cumsum of previous distances' increments).
    start_h = np.concatenate([[0.0], np.cumsum(INC_H)[:-1]]).astype('float32')  # (NDIST,)
    start_v = np.concatenate([[0.0], np.cumsum(INC_V)[:-1]]).astype('float32')

    # Per-angle, per-distance cumulative shrink: starts at start_*[k] at
    # θ_idx=0 and ramps linearly to start_*[k] + INC_*[k] at θ_idx=NTHETA-1.
    # Result shape (NTHETA, NDIST).
    j_frac       = (np.arange(NTHETA, dtype='float32') / max(NTHETA - 1, 1))
    shrink_h_pa  = start_h[None, :] + INC_H[None, :] * j_frac[:, None]
    shrink_v_pa  = start_v[None, :] + INC_V[None, :] * j_frac[:, None]
    norm_mag     = NORM_MAG                                     # (NDIST,)

    # Build m of shape (NTHETA, NDIST, 2) — axis 2 is (my, mx).
    m_pa = np.empty((NTHETA, NDIST, 2), dtype='float32')
    m_pa[..., 0] = (1 + shrink_v_pa) / norm_mag[None, :]
    m_pa[..., 1] = (1 + shrink_h_pa) / norm_mag[None, :]

    print('per-angle m at the three angles we render:')
    angle_ids = [int(round(f * (NTHETA - 1))) for f in ANGLE_IDS_FRACTIONS]
    for a in angle_ids:
        for k in range(NDIST):
            print(f'  θ_idx={a}  dist={k}: cum_h={shrink_h_pa[a,k]:.4f}  cum_v={shrink_v_pa[a,k]:.4f}  '
                  f'my={m_pa[a,k,0]:.4f}  mx={m_pa[a,k,1]:.4f}')

    # Shift maps (NZ_OBJ, NOBJ) → (NZ, N) per projection with per-axis m.
    # Signature: Shift(n, npsi, nz, nzpsi, ...).
    print(f'applying per-axis Shift: (NZ_OBJ={NZ_OBJ}, NOBJ={NOBJ}) → (NZ={NZ}, N={N})')
    cl_shift_cubic = Shift   (N, NOBJ, NZ, NZ_OBJ, 'complex64')
    cl_shift_fft   = ShiftFFT(N, NOBJ, NZ, NZ_OBJ, 'complex64')
    sino_cubic     = apply_shift_per_dist(cl_shift_cubic, sino_obj, m_pa)
    sino_fft       = apply_shift_per_dist(cl_shift_fft,   sino_obj, m_pa)

    # How much do the two backends disagree?
    diff_backends = np.linalg.norm(sino_cubic - sino_fft) / max(np.linalg.norm(sino_cubic), 1e-30)
    print(f'rel diff cubic vs fft (per-axis): {diff_backends:.4e}')

    # --- Plot all distances at angles {0, NTHETA/2, NTHETA-1}, cubic vs fft.
    n_angles = len(angle_ids)
    n_cols   = 2 * NDIST   # for each distance: cubic, fft side by side
    png_path = os.path.join(os.path.dirname(__file__), 'per_axis_mag_first_proj.png')
    fig, ax = plt.subplots(n_angles, n_cols,
                           figsize=(4.5 * n_cols, 3.0 * n_angles),
                           constrained_layout=True)
    fig.suptitle(
        f'Real part — cubic vs fft Shift across {NDIST} distances\n'
        f'cumulative shrink per distance (start → end):  '
        + '  '.join(f'd{k}: v={start_v[k]:.2f}→{start_v[k]+INC_V[k]:.2f}  '
                    f'h={start_h[k]:.2f}→{start_h[k]+INC_H[k]:.2f}'
                    for k in range(NDIST))
    )

    for ai, a in enumerate(angle_ids):
        for k in range(NDIST):
            for bi, (label, stack) in enumerate([('cubic', sino_cubic),
                                                 ('fft',   sino_fft)]):
                col = 2 * k + bi   # dist 0: cols 0,1; dist 1: cols 2,3
                im = ax[ai, col].imshow(stack[a, k], cmap='gray')
                ax[ai, col].set_title(
                    f'{label}  dist={k}  θ_idx={a} / {NTHETA - 1}\n'
                    f'my={m_pa[a,k,0]:.3f}  mx={m_pa[a,k,1]:.3f}',
                    fontsize=9,
                )
                ax[ai, col].set_xlabel('x (col)', fontsize=8)
                ax[ai, col].set_ylabel('y (row)', fontsize=8)
                ax[ai, col].tick_params(labelsize=7)
                fig.colorbar(im, ax=ax[ai, col], fraction=0.046, pad=0.02)

    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    print(f'wrote {png_path}')



if __name__ == '__main__':
    main()
