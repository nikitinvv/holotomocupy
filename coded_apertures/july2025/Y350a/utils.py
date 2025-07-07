import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import tifffile
import os
from concurrent.futures import wait
from functools import wraps
import time
import psutil

def mshow(a, show=False, **args):
    if show:
        if isinstance(a, cp.ndarray):
            a = a.get()
        fig, axs = plt.subplots(1, 1, figsize=(6,6))
        im = axs.imshow(a, cmap="gray", **args)
        fig.colorbar(im, fraction=0.046, pad=0.04)
        plt.show()


def mshow_complex(a, show=False, **args):
    if show:
        if isinstance(a, cp.ndarray):
            a = a.get()
        fig, axs = plt.subplots(1, 2, figsize=(18,6))
        im = axs[0].imshow(a.real, cmap="gray", **args)
        axs[0].set_title("real")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        im = axs[1].imshow(a.imag, cmap="gray", **args)
        axs[1].set_title("imag")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        plt.show()


def mshow_polar(a, show=False, **args):
    if show:
        if isinstance(a, cp.ndarray):
            a = a.get()
        fig, axs = plt.subplots(1, 2, figsize=(18,6))
        im = axs[0].imshow(np.abs(a), cmap="gray", **args)
        axs[0].set_title("abs")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        im = axs[1].imshow(np.angle(a), cmap="gray", **args)
        axs[1].set_title("phase")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        plt.show()

def mplot_positions(a, show=False, **args):
    if show:
        if isinstance(a, cp.ndarray):
            a = a.get()
        plt.plot(a[:,1],a[:,0],'.')
        plt.grid()
        plt.axis('square')
        plt.show()
    
def reprod(a, b):
    return a.real * b.real + a.imag * b.imag


def redot(a, b, axis=None):
    res = cp.sum(reprod(a, b), axis=axis)
    return res

def chunking_flg(pars):
    return  any(isinstance(a, np.ndarray) for a in pars)
    


def write_tiff(a, name, **args):
    if isinstance(a, cp.ndarray):
        a = a.get()
    os.makedirs(os.path.dirname(name), exist_ok=True)
    tifffile.imwrite(name+'.tiff', a)

def read_tiff(name):
    a = tifffile.imread(name)[:]
    return a


def _linear(res, x, y, a, b,st, end):
    res[st:end] = a*x[st:end]+b*y[st:end]

def _mulc(res, x, a, st, end):
    res[st:end] = a*x[st:end]

def mulc(x, a,pool):
    res = np.empty_like(x)
    nthreads = pool._max_workers
    nthreads = min(nthreads, res.shape[0])
    nchunk = int(np.ceil(res.shape[0] / nthreads))
    futures = [
        pool.submit(_mulc, res, x, a, k * nchunk, min((k + 1) * nchunk, res.shape[0]))
        for k in range(nthreads)
    ]
    wait(futures)
    return res

def linear(res, x,y,a,b,pool):
    nthreads = pool._max_workers
    nthreads = min(nthreads, res.shape[0])
    nchunk = int(np.ceil(res.shape[0] / nthreads))
    futures = [
        pool.submit(_linear, res, x, y, a, b, k * nchunk, min((k + 1) * nchunk, res.shape[0]))
        for k in range(nthreads)
    ]
    wait(futures)

def initR(n):
    # usfft parameters
    eps = 1e-3  # accuracy of usfft
    mu = -cp.log(eps) / (2 * n * n)
    m = int(cp.ceil(2 * n * 1 / cp.pi * cp.sqrt(-mu *
            cp.log(eps) + (mu * n) * (mu * n) / 4)))
    # extra arrays
    # interpolation kernel
    t = cp.linspace(-1/2, 1/2, n, endpoint=False).astype('float32')
    [dx, dy] = cp.meshgrid(t, t)
    phi = cp.exp((mu * (n * n) * (dx * dx + dy * dy)).astype('float32')) * (1-n % 4)

    
    # (+1,-1) arrays for fftshift
    c1dfftshift = (1-2*((cp.arange(1, n+1) % 2))).astype('int8')
    c2dtmp = 1-2*((cp.arange(1, 2*n+1) % 2)).astype('int8')
    c2dfftshift = cp.outer(c2dtmp, c2dtmp)
    return [m, mu, phi, c1dfftshift, c2dfftshift]    
    

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        process = psutil.Process(os.getpid())
        #if (end_time - start_time>0.1):
            #print(f"{func.__name__}: {end_time - start_time:.4f} sec, {process.memory_info().rss / (1024**3):.2f} GB", flush=True)
        return result
    return wrapper
