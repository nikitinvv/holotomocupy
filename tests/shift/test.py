"""Compare holotomocupy.Shift.curlyS with scipy.ndimage.shift.

When npsi == n and magnification m == 1, curlyS becomes a pure subpixel shift,
which can be cross-checked against scipy.ndimage.shift (cubic B-spline, order=3).

The kernel maps: output[ty, tx] = input[ty - ry, tx - rx], which is the same
sign convention as scipy.ndimage.shift(arr, [ry, rx]).
"""

import numpy as np
import cupy as cp
import scipy.ndimage as snd
import matplotlib.pyplot as plt

from holotomocupy.shift import Shift


def make_input(n, seed=0):
    rng = np.random.default_rng(seed)
    # Smooth-ish complex test object (sum of a few sinusoids + noise)
    yy, xx = np.mgrid[0:n, 0:n].astype('float32')
    re = (np.sin(2 * np.pi * xx / n * 3) * np.cos(2 * np.pi * yy / n * 2)
          + 0.3 * rng.standard_normal((n, n), dtype='float32'))
    im = (np.cos(2 * np.pi * xx / n * 4) * np.sin(2 * np.pi * yy / n * 5)
          + 0.3 * rng.standard_normal((n, n), dtype='float32'))
    return (re + 1j * im).astype('complex64')


def shift_scipy(arr, ry, rx, order=3):
    # mode='mirror' matches the kernel's sym_idx (cuda_kernels.py:140):
    #   i<0 → -i  and  i>=N → 2N-2-i  (reflect without duplicating the edge).
    # scipy's 'reflect' duplicates the edge and would NOT match.
    re = snd.shift(arr.real, [ry, rx], order=order, mode='mirror')
    im = snd.shift(arr.imag, [ry, rx], order=order, mode='mirror')
    return (re + 1j * im).astype('complex64')


def shift_holotomo(cl, arr_gpu, ry, rx):
    pos = cp.asarray([[[ry, rx]]], dtype='float32')   # [ntheta=1, ndist=1, 2]
    m   = cp.asarray([1.0], dtype='float32')          # unit magnification
    return cl.curlyS(arr_gpu, pos[:, 0], m)


def interior_error(a, b, margin):
    """Max-abs error inside the array, ignoring the boundary `margin` pixels
    where the two methods handle out-of-bounds samples differently."""
    s = (slice(margin, -margin), slice(margin, -margin))
    return float(np.max(np.abs(a[s] - b[s])))


def main():
    n = 64
    margin = 4          # cubic B-spline support is 4 pixels — ignore boundary
    arr_np  = make_input(n)
    arr_gpu = cp.asarray(arr_np)[cp.newaxis]   # [1, n, n]

    cl = Shift(n=n, npsi=n, nz=n, nzpsi=n)

    shifts = [(0.0, 0.0),
              (0.5, 0.0),
              (0.0, 0.5),
              (1.3, -2.7),
              (-3.4, 4.1),
              (7.0, 7.0)]

    print(f"{'ry':>8} {'rx':>8}   {'max|holo-scipy|':>18}   {'rel':>10}")
    print("-" * 56)
    results = []
    for ry, rx in shifts:
        b_holo  = shift_holotomo(cl, arr_gpu, ry, rx)[0].get()
        b_scipy = shift_scipy(arr_np, ry, rx, order=3)

        err = interior_error(b_holo, b_scipy, margin)
        ref = float(np.max(np.abs(b_scipy[margin:-margin, margin:-margin]))) + 1e-30
        print(f"{ry:8.3f} {rx:8.3f}   {err:18.3e}   {err/ref:10.3e}")
        results.append((ry, rx, b_holo, b_scipy))

    # ------------------------------------------------------------------
    # Adjoint test: <S(c), b> == <c, Sadj(b)>
    # ------------------------------------------------------------------
    ntheta = 4
    pos_adj = (cp.random.random([ntheta, 2], dtype='float32') - 0.5) * 20
    m_adj   = cp.ones(ntheta, dtype='float32')

    c = (cp.random.random([ntheta, n, n], dtype='float32')
         + 1j * cp.random.random([ntheta, n, n], dtype='float32')).astype('complex64')
    b = (cp.random.random([ntheta, n, n], dtype='float32')
         + 1j * cp.random.random([ntheta, n, n], dtype='float32')).astype('complex64')

    Sc    = cl.S(c, pos_adj, m_adj)
    Sadjb = cl.Sadj(b, pos_adj, m_adj)
    lhs = float(cp.real(cp.sum(cp.conj(Sc) * b)))
    rhs = float(cp.real(cp.sum(cp.conj(c) * Sadjb)))
    print()
    print(f"Adjoint test (S/Sadj): lhs={lhs:.6e}  rhs={rhs:.6e}  "
          f"rel={abs(lhs - rhs) / (abs(lhs) + 1e-30):.3e}")

    # ------------------------------------------------------------------
    # Visualization — restrict to the central window so boundary effects
    # (mirror-FFT vs mirror-IIR prefilter differences) don't dominate the
    # color scale of the 2D error map.
    # ------------------------------------------------------------------
    vis_margin = n // 4                   # show only the central half
    s   = (slice(vis_margin, -vis_margin), slice(vis_margin, -vis_margin))
    row = (n // 2) - vis_margin           # central row inside the cropped view

    fig, axs = plt.subplots(len(results), 4, figsize=(14, 3 * len(results)))
    for i, (ry, rx, b_holo, b_scipy) in enumerate(results):
        h_in = b_holo[s]
        s_in = b_scipy[s]
        diff = np.abs(h_in - s_in)

        im0 = axs[i, 0].imshow(h_in.real, cmap='gray')
        axs[i, 0].set_title(f'holo Re  (ry={ry}, rx={rx})  central')
        im1 = axs[i, 1].imshow(s_in.real, cmap='gray')
        axs[i, 1].set_title('scipy Re  central')
        im2 = axs[i, 2].imshow(diff,      cmap='hot')
        axs[i, 2].set_title(f'|holo - scipy|  central  (max {diff.max():.2e})')

        axs[i, 3].plot(h_in.real[row],       label='holo')
        axs[i, 3].plot(s_in.real[row], '--', label='scipy')
        axs[i, 3].set_title(f'row {n//2} (real, central)')
        axs[i, 3].legend(); axs[i, 3].grid()

        for ax in axs[i, :3]:
            ax.axis('off')
        fig.colorbar(im2, ax=axs[i, 2], fraction=0.046)
    plt.tight_layout()
    out_png = '/home/beams2/VNIKITIN/holotomocupy_mpi/tests/shift/test_shift_compare.png'
    plt.savefig(out_png, dpi=110)
    print(f"\nSaved figure: {out_png}")


if __name__ == '__main__':
    main()
