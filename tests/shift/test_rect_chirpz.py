"""Visual comparison: shift + scale a centered rectangle from a 3N/2 input
grid to an N output grid, using cubic-spline interpolation vs the chirp-z
(FFT) path with magnification m ≠ 1.

Input grid:  3N/2 × 3N/2  (nzpsi = npsi = 3N/2)
Output grid: N × N        (nz = n = N)

Rows:
  0: input  (3N/2 × 3N/2 rectangle, sharp by default)
  1: cubic  — Shift  with B-spline prefilter (sees magnification m via the kernel)
  2: chirp-z — ShiftFFT, Fourier-shift + Bluestein magnification path
  3: cubic − chirp-z  (diverging colormap, per-column symmetric scale)

Columns: different (m, r) combinations.
"""

import argparse
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import scipy.ndimage as snd

from holotomocupy.shift     import Shift
from holotomocupy.shift_fft import ShiftFFT


def make_rectangle(npsi, half_size=18, smooth=0.0):
    """Centered rectangle in an npsi×npsi complex64 image, split into 4
    quadrants with values (0.25, 0.5, 0.75, 1.0) so internal step edges are
    visible too. `smooth` > 0 applies a Gaussian blur (band-limits the input
    and reduces Gibbs ringing in the FFT path)."""
    cy = cx = npsi // 2
    img = np.zeros((npsi, npsi), dtype='float32')
    quadrants = [
        (slice(cy - half_size, cy            ), slice(cx - half_size, cx            ), 0.25),
        (slice(cy - half_size, cy            ), slice(cx,             cx + half_size), 0.50),
        (slice(cy,             cy + half_size), slice(cx - half_size, cx            ), 0.75),
        (slice(cy,             cy + half_size), slice(cx,             cx + half_size), 1.00),
    ]
    for ys, xs, v in quadrants:
        img[ys, xs] = v
    if smooth > 0:
        img = snd.gaussian_filter(img, sigma=smooth, mode='constant', cval=0.0)
    return (img + 1j * 0.0).astype('complex64')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=64,
                        help='Output grid size N. Input grid is 3N/2.')
    parser.add_argument('--half-size', type=int, default=18,
                        help='Half side length of the rectangle on the INPUT grid.')
    parser.add_argument('--smooth', type=float, default=0.0,
                        help='Gaussian-blur sigma applied to rectangle edges '
                             '(0 = sharp).')
    args = parser.parse_args()

    n    = args.n
    npsi = 3 * n // 2
    assert n % 2 == 0,        "use even N so 3N/2 is integer"
    assert npsi == 3 * n // 2

    img = make_rectangle(npsi, half_size=args.half_size, smooth=args.smooth)
    arr_gpu = cp.asarray(img)[cp.newaxis]   # [1, npsi, npsi]

    # Cubic and chirp-z operators, same input/output shapes.
    cubic   = Shift   (n=n, npsi=npsi, nz=n, nzpsi=npsi, obj_dtype='complex64')
    chirpz  = ShiftFFT(n=n, npsi=npsi, nz=n, nzpsi=npsi, obj_dtype='complex64')
    methods = [('cubic', cubic), ('chirp-z', chirpz)]

    # (m, ry, rx) cases:
    #   col 0: m<1   — zoom in (rectangle appears bigger in the output)
    #   col 1: m=1   — pure shift (control — chirp-z degenerates to FFT shift)
    #   col 2: m>1   — zoom out (rectangle appears smaller in the output)
    #   col 3: large m  + offset (combined shift+scale, stress test)
    cases = [
        (0.85, ( 2.5, -1.5)),
        (1.00, ( 4.0,  3.0)),
        (1.15, (-3.0,  2.0)),
        (1.30, ( 6.0, -4.0)),
    ]

    nrows = 1 + len(methods) + 1   # input + methods + (cubic − chirp-z) diff row
    ncols = len(cases)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows))
    if ncols == 1:
        axs = axs[:, None]

    title_fs = 16
    cbar_fs  = 14
    sup_fs   = 20

    def style_cbar(cb):
        cb.ax.tick_params(labelsize=cbar_fs)

    vmin, vmax = -0.2, 1.2
    for col, (m_val, (ry, rx)) in enumerate(cases):
        r = cp.asarray([[ry, rx]], dtype='float32')
        m = cp.asarray([m_val],    dtype='float32')

        # Row 0: input — same in every column, captioned with this column's (m, r).
        ax = axs[0, col]
        im = ax.imshow(img.real, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f'input {npsi}×{npsi}\nm={m_val:.2f}  r=({ry:+.1f}, {rx:+.1f})',
                     fontsize=title_fs)
        ax.set_xticks([]); ax.set_yticks([])
        style_cbar(fig.colorbar(im, ax=ax, fraction=0.046))

        # Rows 1, 2: one per method, all output shape n×n.
        outs = []
        for row, (label, op) in enumerate(methods, start=1):
            out = op.curlyS(arr_gpu, r, m)[0].get().real
            outs.append(out)
            ax = axs[row, col]
            im = ax.imshow(out, cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_title(f'{label}  {n}×{n}', fontsize=title_fs)
            ax.set_xticks([]); ax.set_yticks([])
            style_cbar(fig.colorbar(im, ax=ax, fraction=0.046))

        # Row 3: per-column difference (cubic − chirp-z), diverging colormap,
        # symmetric scale around 0 chosen from this column's data.
        out_cubic, out_chirpz = outs
        diff = out_cubic - out_chirpz
        dlim = max(float(np.abs(diff).max()), 1e-8)
        ax = axs[1 + len(methods), col]
        im = ax.imshow(diff, cmap='RdBu_r', vmin=-dlim, vmax=dlim)
        ax.set_title(f'cubic − chirp-z   ±{dlim:.2e}', fontsize=title_fs)
        ax.set_xticks([]); ax.set_yticks([])
        style_cbar(fig.colorbar(im, ax=ax, fraction=0.046))

        # Numeric pixel-wise difference on the interior.
        margin = 6
        s = (slice(margin, -margin), slice(margin, -margin))
        err = np.max(np.abs(diff[s]))
        rms = np.sqrt(np.mean(diff[s] ** 2))
        print(f'  m={m_val:.2f}  r=({ry:+.1f}, {rx:+.1f})   '
              f'max|cubic-chirpz|={err:.3e}   rms={rms:.3e}   '
              f'full-frame max={float(np.abs(diff).max()):.3e}')

    plt.suptitle(
        f'shift + scaling: rectangle on input {npsi}×{npsi} → output {n}×{n}  '
        f'(half_size={args.half_size}, smooth={args.smooth})',
        fontsize=sup_fs, y=1.0)
    plt.tight_layout()
    suffix = f'_smooth{args.smooth:g}' if args.smooth > 0 else '_sharp'
    out_png = (f'/home/beams2/VNIKITIN/holotomocupy_mpi/tests/shift/'
               f'test_rect_chirpz{suffix}.png')
    plt.savefig(out_png, dpi=110)
    print(f'\nSaved figure: {out_png}')
    print()
    print('Expected differences:')
    print('  - Sharp rectangle: chirp-z shows Gibbs ringing at the step edges,')
    print('    cubic does not. Try --smooth 1.5 to band-limit the input.')
    print('  - m≠1 cubic uses cubic B-spline interpolation at the m-scaled positions;')
    print('    chirp-z uses the band-limited sinc interpolant at the same positions.')
    print('    Both should agree away from edges for a smoothly band-limited input.')
    print('  - m=1 column is a control: chirp-z degenerates to the pure FFT shift.')


if __name__ == '__main__':
    main()
