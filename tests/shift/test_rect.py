"""Visual comparison: shift a centered rectangle through all four shift modes.

Compares cubic-spline and FFT shifts, each in both symmetric=False (unpadded,
periodic BC) and symmetric=True (mirror-padded to 2× grid) flavors. The
rectangle is a sharp-edged step function with 4 quadrant values, so internal
discontinuities are visible too. Across columns: small / medium / large shifts.

Rows:
  0: input
  1: cubic, symmetric=False  — current default; FFT prefilter assumes periodic
  2: cubic, symmetric=True   — mirror-pad + prefilter on 2× grid
  3: fft,   symmetric=False  — raw FFT shift, periodic wrap-around on big shifts
  4: fft,   symmetric=True   — current ShiftFFT default; mirror-pad to 2× grid
"""

import argparse
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import scipy.ndimage as snd

from holotomocupy.shift     import Shift
from holotomocupy.shift_fft import ShiftFFT


def make_rectangle(n, half_size=12, smooth=0.0):
    """Rectangle centered in an n×n complex64 image, split into 4 quadrants
    with values (0.25, 0.5, 0.75, 1.0) so that internal step edges are also
    visible (lets you see Gibbs ringing on internal discontinuities, not
    just the outer rectangle border). `smooth` > 0 applies a Gaussian blur
    of that sigma to all edges."""
    yy, xx = np.mgrid[0:n, 0:n]
    cy = cx = n // 2
    img = np.zeros((n, n), dtype='float32')
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
    parser.add_argument('--smooth', type=float, default=0.0,
                        help='Gaussian-blur sigma applied to rectangle edges '
                             '(0 = sharp). Try 1.0, 2.0, 4.0 to see FFT Gibbs '
                             'ringing fade as the input becomes band-limited.')
    parser.add_argument('--half-size', type=int, default=14,
                        help='Half side length of the rectangle.')
    parser.add_argument('--n', type=int, default=96,
                        help='Image grid size.')
    args = parser.parse_args()

    n = args.n
    img = make_rectangle(n, half_size=args.half_size, smooth=args.smooth)
    arr_gpu = cp.asarray(img)[cp.newaxis]   # [1, n, n]

    # Four shift operators — same input, output and grid sizes, just different
    # backends and symmetric-padding choices.
    methods = [
        ('cubic, sym=False', Shift   (n=n, npsi=n, nz=n, nzpsi=n, obj_dtype='complex64', symmetric=False)),
        ('cubic, sym=True',  Shift   (n=n, npsi=n, nz=n, nzpsi=n, obj_dtype='complex64', symmetric=True )),
        ('fft,   sym=False', ShiftFFT(n=n, npsi=n, nz=n, nzpsi=n, obj_dtype='complex64', symmetric=False)),
        ('fft,   sym=True',  ShiftFFT(n=n, npsi=n, nz=n, nzpsi=n, obj_dtype='complex64', symmetric=True )),
    ]

    shifts = [(2.5, -1.5), (8.7, -6.3), (28.0, -25.0)]
    m = cp.ones(1, dtype='float32')

    nrows = 1 + len(methods)              # input + one row per method
    ncols = len(shifts)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if ncols == 1:
        axs = axs[:, None]

    vmin, vmax = -0.2, 1.2
    for col, (ry, rx) in enumerate(shifts):
        r = cp.asarray([[ry, rx]], dtype='float32')

        # Row 0: input (same in every column — repeated for context next to its shift).
        ax = axs[0, col]
        im = ax.imshow(img.real, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f'input  (ry={ry}, rx={rx})', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)

        # Rows 1..4: one per method.
        for row, (label, op) in enumerate(methods, start=1):
            out = op.curlyS(arr_gpu, r, m)[0].get().real
            ax = axs[row, col]
            im = ax.imshow(out, cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_title(label, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(f'rectangle shift  (n={n}, half_size={args.half_size}, smooth={args.smooth})',
                 fontsize=12, y=1.0)
    plt.tight_layout()
    suffix = f'_smooth{args.smooth:g}' if args.smooth > 0 else '_sharp'
    out_png = f'/home/beams2/VNIKITIN/holotomocupy_mpi/tests/shift/test_rect_compare{suffix}.png'
    plt.savefig(out_png, dpi=110)
    print(f"Saved figure: {out_png}")
    print()
    print("Expected differences:")
    print("  - cubic sym=False vs sym=True: tiny near-boundary differences (the")
    print("    mirror-padded prefilter fixes wrong coefficients at the object edge).")
    print("  - fft sym=False, large shift: visible periodic wrap-around (rectangle")
    print("    re-appears on the opposite side). sym=True eliminates the wrap.")
    print("  - fft (either sym): Gibbs ringing around every step edge of the")
    print("    rectangle; cubic does not ring.")


if __name__ == '__main__':
    main()
