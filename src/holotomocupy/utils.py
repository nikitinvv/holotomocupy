import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import tifffile
import os
import time
import psutil
from functools import wraps
from .logger_config import logger

from matplotlib_scalebar.scalebar import ScaleBar

# Cached once — avoids a system call on every timed invocation
_process = psutil.Process()


def copy_to_pinned(data):
    buf = cp.cuda.alloc_pinned_memory(data.nbytes)
    buf = np.frombuffer(buf, dtype=data.dtype, count=data.size).reshape(data.shape)
    buf[:] = data
    return buf


def make_pinned(shape, dtype):
    n      = int(np.prod(shape))
    nbytes = n * np.dtype(dtype).itemsize
    logger.debug(f'Allocate {shape} {dtype}: {nbytes / 1024**3:.3f} GB')
    buf = cp.cuda.alloc_pinned_memory(nbytes)
    return np.frombuffer(buf, dtype=dtype, count=n).reshape(shape, copy=False)


def mshow(a, show=False, figsize=(6, 6), **args):
    if show:
        if isinstance(a, cp.ndarray):
            a = a.get()
        fig, axs = plt.subplots(1, 1, figsize=figsize)
        im = axs.imshow(a, cmap="gray", **args)
        fig.colorbar(im, fraction=0.046, pad=0.04)
        plt.show()


def mshow_complex(a, show=False, figsize=(14, 6), **args):
    if show:
        if isinstance(a, cp.ndarray):
            a = a.get()
        fig, axs = plt.subplots(1, 2, figsize=figsize)
        im = axs[0].imshow(a.real, cmap="gray", **args)
        scalebar = ScaleBar(0.015, "um", length_fraction=0.25,
                            font_properties={"family": "serif"},
                            location="lower right")
        axs[0].add_artist(scalebar)
        fig.colorbar(im, fraction=0.046, pad=0.04)
        im = axs[1].imshow(a.imag, cmap="gray", **args)
        fig.colorbar(im, fraction=0.046, pad=0.04)
        plt.show()


def mshow_polar(a, show=False, figsize=(14, 6), **args):
    if show:
        if isinstance(a, cp.ndarray):
            a = a.get()
        fig, axs = plt.subplots(1, 2, figsize=figsize)
        im = axs[0].imshow(np.abs(a), cmap="gray", **args)
        axs[0].set_title("abs")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        im = axs[1].imshow(np.angle(a), cmap="gray", **args)
        axs[1].set_title("phase")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        plt.show()


def mshow_pos(pos, show=False, figsize=(10, 4), **args):
    if show:
        if isinstance(pos, cp.ndarray):
            pos = pos.get()
        _, ax = plt.subplots(1, 2, figsize=figsize)
        ax[0].plot(pos[..., 1], ".")
        ax[0].set_title("x")
        ax[1].plot(pos[..., 0], ".")
        ax[1].set_title("y")
        ax[0].grid()
        ax[1].grid()
        plt.show()


def mshow_approx(t, err_real, err_approx, show=False):
    if show:
        plt.figure(figsize=(4, 4))
        plt.plot(t, err_real,   "o-", label="real")
        plt.plot(t, err_approx, "x-", label="approx")
        plt.legend()
        plt.grid()
        plt.show()


def reprod(a, b):
    return cp.real(a) * cp.real(b) + cp.imag(a) * cp.imag(b)


def redot(a, b, axis=None):
    if axis is None:
        return cp.vdot(a.view('float32'), b.view('float32'))
    return cp.sum(reprod(a, b), axis=axis)


class CoeffCacheMixin:
    """Memoization of ``coeff(psi)`` shared by the shift operators.

    The host class must define ``coeff`` and initialise ``coeff_cache`` (dict),
    ``coeff_hits`` and ``coeff_misses`` (ints).
    """

    def coeff_cached(self, psi):
        """coeff(psi) memoized by id(psi). MUST be paired with explicit
        coeff_cache_reset() at safe lifetime boundaries (e.g., per chunk):
        id() values can be reused across distinct Python objects once an
        earlier one is garbage-collected. Hit/miss counters are exposed for
        verification; reset along with the cache."""
        key = id(psi)
        cached = self.coeff_cache.get(key)
        if cached is None:
            self.coeff_misses += 1
            cached = self.coeff(psi)
            self.coeff_cache[key] = cached
        else:
            self.coeff_hits += 1
        return cached

    def coeff_cache_reset(self):
        self.coeff_cache = {}

    def coeff_cache_stats(self, reset=False):
        """Return (hits, misses) accumulated since the last stats reset.
        Set reset=True to zero the counters."""
        stats = (self.coeff_hits, self.coeff_misses)
        if reset:
            self.coeff_hits = 0
            self.coeff_misses = 0
        return stats


def lap(a, b, c):
    """5-stencil Laplacian on the centre slice b given its z-neighbours a, c.
    Output shape matches b: ∇²b ≈ a + c + roll(b,±1,axis=1) + roll(b,±1,axis=2) - 6·b."""
    return (a + c
            + cp.roll(b, -1, axis=1) + cp.roll(b,  1, axis=1)
            + cp.roll(b, -1, axis=2) + cp.roll(b,  1, axis=2)
            - 6 * b)


def write_tiff(a, name, **args):
    if isinstance(a, cp.ndarray):
        a = a.get()
    dirname = os.path.dirname(name)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    tifffile.imwrite(name + '.tiff', a)


def read_tiff(name):
    return tifffile.imread(name)[:]



def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start  = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        mem = _process.memory_info().rss / 1024**3
        free, total = cp.cuda.runtime.memGetInfo()
        gpu_mem = (total - free) / 1024**3
        logger.debug(f"{func.__name__}: {elapsed:.4f} sec, process memory {mem:.2f} GB, GPU memory {gpu_mem:.2f} GB")
        return result
    return wrapper


def empty_like(x):
    if isinstance(x, cp.ndarray):
        return cp.empty_like(x)
    return np.empty_like(x)

