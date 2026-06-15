"""Cubic B-spline prefilter for stacks of 2D complex64 arrays, using CuPy.

A separable two-pole IIR filter (Unser et al.) for the cubic B-spline. The output
coefficients are what you pass to a cubic B-spline interpolator so that the
interpolant exactly passes through the original samples.

Boundary conditions are whole-sample symmetric mirror (``c[-k]=c[k]``,
``c[N-1+k]=c[N-1-k]``), matching the shift kernel's ``sym_idx`` in
``src/holotomocupy/cuda_kernels.py``.

Only ``complex64`` is supported — the recursion is real-linear (single real pole)
so it runs componentwise on .x/.y inside one kernel.

Usage
-----
    import cupy as cp
    from prefilter_tests import prefilter2d

    x = cp.asarray(samples, dtype=cp.complex64)   # (n, n) or (batch, n, n)
    c = prefilter2d(x)                            # cubic B-spline coefficients
"""

import cupy as cp


_POLE = -0.267949192431123  # sqrt(3) - 2, the single pole of the cubic B-spline


# Whole-sample symmetric mirror BC (c[-k]=c[k], c[N-1+k]=c[N-1-k]).
# Matches the shift kernel's sym_idx in src/holotomocupy/cuda_kernels.py.
_KERNEL_SRC = r"""
extern "C" __global__
void prefilter_rows(float2* image, const int n, const int batch)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int b   = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n || b >= batch) return;

    const float Pole   = -0.267949192431123f;
    const float Lambda = (1.0f - Pole) * (1.0f - 1.0f / Pole);
    const int Horizon = 12 < n ? 12 : n;

    float2* c = image + (size_t)b * n * n + (size_t)row * n;

    // Causal init under WSS:  c+[0] = sum_{k=0}^{H-1} z1^k * c[k]
    float2 sum = c[0];
    float zn = Pole;
    for (int k = 1; k < Horizon; ++k) {
        sum.x += zn * c[k].x;
        sum.y += zn * c[k].y;
        zn   *= Pole;
    }

    float2 prev;
    prev.x = Lambda * sum.x;
    prev.y = Lambda * sum.y;
    c[0] = prev;
    for (int i = 1; i < n; ++i) {
        prev.x = Lambda * c[i].x + Pole * prev.x;
        prev.y = Lambda * c[i].y + Pole * prev.y;
        c[i] = prev;
    }

    // Anti-causal init under WSS:  c-[N-1] = (z1/(z1^2-1)) * (c+[N-1] + z1 * c+[N-2])
    const float scale = Pole / (Pole * Pole - 1.0f);
    prev.x = scale * (c[n - 1].x + Pole * c[n - 2].x);
    prev.y = scale * (c[n - 1].y + Pole * c[n - 2].y);
    c[n - 1] = prev;
    for (int i = n - 2; i >= 0; --i) {
        prev.x = Pole * (prev.x - c[i].x);
        prev.y = Pole * (prev.y - c[i].y);
        c[i] = prev;
    }
}
"""


_KERNEL = cp.RawKernel(_KERNEL_SRC, "prefilter_rows")


def prefilter2d(data, inplace=False):
    """Cubic B-spline prefilter applied along both axes of a complex64 n×n stack.

    Uses whole-sample symmetric mirror BC (matches the shift kernel's ``sym_idx``).

    Parameters
    ----------
    data : cupy.ndarray (complex64)
        Shape ``(n, n)`` or ``(batch, n, n)``.
    inplace : bool, default False
        If True, overwrite ``data`` and return it.
    """
    
    squeeze = data.ndim == 2
    if squeeze:
        data = data[None]
    elif data.ndim != 3:
        raise ValueError("data must be 2D (n,n) or 3D (batch,n,n)")

    batch, h, w = data.shape
    if h != w:
        raise ValueError(f"expected square arrays, got ({h}, {w})")
    n = h

    work = data if inplace else data.copy()
    work = cp.ascontiguousarray(work)

    block = (16, 16, 1)
    grid = ((n + block[0] - 1) // block[0],
            (batch + block[1] - 1) // block[1],
            1)

    # Filter along the last axis (rows), then the other axis via transpose.
    _KERNEL(grid, block, (work, cp.int32(n), cp.int32(batch)))
    work = cp.ascontiguousarray(work.transpose(0, 2, 1))
    _KERNEL(grid, block, (work, cp.int32(n), cp.int32(batch)))
    work = cp.ascontiguousarray(work.transpose(0, 2, 1))

    return work[0] if squeeze else work


def prefilter2d_fft(data):
    """Cubic B-spline prefilter via division by the DFT of B3 (periodic boundary).

    Equivalent to ``IFFT(FFT(x) / B3_hat)`` where ``B3_hat = phi(0) + 2*phi(1)*cos(2pi k/N)``
    with ``phi(0) = 2/3, phi(1) = 1/6``. Exact inverse of the periodic [1,4,1]/6
    convolution; differs from the IIR variant only near the edges.
    """
    if data.dtype != cp.complex64:
        raise TypeError(f"data must be complex64, got {data.dtype}")
    if data.ndim == 2:
        data = data[None]
        squeeze = True
    else:
        squeeze = False
    _, m, n = data.shape

    x = cp.linspace(-0.5, 0.5 - 1.0 / n, n, dtype=cp.float32)
    y = cp.linspace(-0.5, 0.5 - 1.0 / m, m, dtype=cp.float32)
    phi0, phi1 = cp.float32(2.0 / 3.0), cp.float32(1.0 / 6.0)
    divx = phi0 + 2 * phi1 * cp.cos(2 * cp.pi * x)
    divy = phi0 + 2 * phi1 * cp.cos(2 * cp.pi * y)
    ifB3 = 1 / cp.fft.fftshift(cp.outer(divy, divx), axes=(-1, -2))

    out = cp.fft.ifft2(cp.fft.fft2(data) * ifB3).astype(cp.complex64)
    return out[0] if squeeze else out


def adjoint_test(n=64, batch=4, seed=0):
    """Check self-adjointness: |<Px, y> - <x, Py>| / (||Px|| ||y||).

    The prefilter has real coefficients, so ``<P x, y> = <x, P y>`` under the
    standard Hermitian inner product. Returns the relative residual.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    shape = (batch, n, n)
    x = cp.asarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape),
                   dtype=cp.complex64)
    y = cp.asarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape),
                   dtype=cp.complex64)

    Px = prefilter2d(x)
    Py = prefilter2d(y)
    lhs = cp.vdot(Px, y)
    rhs = cp.vdot(x, Py)
    denom = cp.linalg.norm(Px) * cp.linalg.norm(y) + 1e-30
    return float(cp.abs(lhs - rhs) / denom), complex(lhs), complex(rhs)


if __name__ == "__main__":
    # Sanity check: prefilter inverts the [1,4,1]/6 convolution under WSS BC.
    # Reconstruction padding must match the BC — numpy mode='reflect' = WSS.
    import numpy as np

    rng = np.random.default_rng(0)
    x = cp.asarray(rng.standard_normal((4, 64, 64))
                   + 1j * rng.standard_normal((4, 64, 64)), dtype=cp.complex64)

    k = cp.asarray([1, 4, 1], dtype=cp.float32) / 6

    c = prefilter2d(x)
    pad = cp.pad(c, ((0, 0), (1, 1), (1, 1)), mode="reflect")
    rec = k[0] * pad[:, :-2, 1:-1] + k[1] * pad[:, 1:-1, 1:-1] + k[2] * pad[:, 2:, 1:-1]
    pad = cp.pad(rec, ((0, 0), (0, 0), (1, 1)), mode="reflect")
    rec = k[0] * pad[:, :, :-2] + k[1] * pad[:, :, 1:-1] + k[2] * pad[:, :, 2:]

    err = float(cp.max(cp.abs(rec - x)))
    rel, _, _ = adjoint_test(n=64, batch=4)
    d_fft = cp.abs(c - prefilter2d_fft(x))
    print(f"max |B3(P x) - x| = {err:.2e}   adjoint rel = {rel:.2e}")
    print(f"vs FFT (periodic): full = {float(cp.max(d_fft)):.2e}, "
          f"interior(8px) = {float(cp.max(d_fft[:, 8:-8, 8:-8])):.2e}")
