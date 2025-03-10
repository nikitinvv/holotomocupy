import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import tifffile
import os

def mshow(a, show=False, **args):
    if show:
        if isinstance(a, cp.ndarray):
            a = a.get()
        fig, axs = plt.subplots(1, 1, figsize=(5, 5))
        im = axs.imshow(a, cmap="gray", **args)
        fig.colorbar(im, fraction=0.046, pad=0.04)
        plt.show()


def mshow_complex(a, show=False, **args):
    if show:
        if isinstance(a, cp.ndarray):
            a = a.get()
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
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
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
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