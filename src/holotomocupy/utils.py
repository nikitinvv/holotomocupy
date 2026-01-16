import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import tifffile
import os
import time
import psutil
import scipy as sp
from functools import wraps


from matplotlib_scalebar.scalebar import ScaleBar


def mshow(a, show=False, **args):
    if show:
        if isinstance(a, cp.ndarray):
            a = a.get()
        fig, axs = plt.subplots(1, 1, figsize=(6,6))
        im = axs.imshow(a, cmap="gray", **args)
        fig.colorbar(im, fraction=0.046, pad=0.04)
        plt.show()


def mshow_complex(a,show=False, **args):
    if show:
        if isinstance(a, cp.ndarray):
            a = a.get()
        fig, axs = plt.subplots(1, 2, figsize=(14,6))
        im = axs[0].imshow(a.real, cmap="gray", **args)
        scalebar = ScaleBar(0.015, "um", length_fraction=0.25, font_properties={
            "family": "serif",
        },  # For more information, see the cell below
        location="lower right")
        axs[0].add_artist(scalebar)
        fig.colorbar(im, fraction=0.046, pad=0.04)
        im = axs[1].imshow(a.imag, cmap="gray", **args)
        scalebar = ScaleBar(0.015, "um", length_fraction=0.25, font_properties={
            "family": "serif",
        },  # For more information, see the cell below        
        location="lower right")
        # axs[1].add_artist(scalebar)
        fig.colorbar(im, fraction=0.046, pad=0.04)
        plt.show()

def mshow_polar(a, show=False, **args):
    if show:
        if isinstance(a, cp.ndarray):
            a = a.get()
        fig, axs = plt.subplots(1, 2, figsize=(14,6))
        im = axs[0].imshow(np.abs(a), cmap="gray", **args)
        axs[0].set_title("abs")
        # fig.colorbar(im, fraction=0.046, pad=0.04)
        im = axs[1].imshow(np.angle(a), cmap="gray", **args)
        axs[1].set_title("phase")
        # fig.colorbar(im, fraction=0.046, pad=0.04)
        plt.show()
        
def mshow_pos(pos, show=False, **args):
    if isinstance(pos, cp.ndarray):
        pos = pos.get()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(pos[:,:,1], ".")
    ax[0].set_title("x")
    ax[1].plot(pos[:,:,0], ".")
    ax[1].set_title("y")
    ax[0].grid()
    ax[1].grid()
    if show:
        plt.show()
    

def mshow_approx(t,err_real,err_approx,show=False):
    if show:
        plt.figure(figsize=(4, 4))
        plt.plot(t, err_real, "o-", label="real")
        plt.plot(t, err_approx, "x-", label="approx")
        plt.legend()
        plt.grid()
        plt.show()

def reprod(a, b):
    return a.real * b.real + a.imag * b.imag


def redot(a, b, axis=None):
    res = cp.sum(reprod(a, b), axis=axis)
    return res

def write_tiff(a, name, **args):
    if isinstance(a, cp.ndarray):
        a = a.get()
    os.makedirs(os.path.dirname(name), exist_ok=True)
    tifffile.imwrite(name+'.tiff', a)

def read_tiff(name):
    a = tifffile.imread(name)[:]
    return a

def downsample(data, binning):
    res = data.copy()
    for k in range(binning):
        res = 0.5*(res[..., ::2,:]+res[..., 1::2,:])
        res = 0.5*(res[..., :, ::2]+res[..., :, 1::2])
    return res


def fftupsample(f, dims):
    dtype = f.dtype
    paddim = np.zeros([np.ndim(f), 2], dtype='int32')
    dims = np.asarray(dims).astype('int32')
    paddim[dims, 0] = np.asarray(f.shape)[dims]//2
    paddim[dims, 1] = np.asarray(f.shape)[dims]//2
    fsize = f.size
    f = sp.fft.ifftshift(sp.fft.fftn(sp.fft.fftshift(
        f, dims), axes=dims, workers=-1), dims)
    f = np.pad(f, paddim)
    f = sp.fft.fftshift(f, dims)
    f = sp.fft.ifftn(f, axes=dims, workers=-1)
    f = sp.fft.ifftshift(f, dims)
    if dtype=='float32':
        f=f.real
    return f*(f.size/fsize)

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        process = psutil.Process(os.getpid())
        # if (end_time - start_time>0.1):
        #     print(f"{func.__name__}: {end_time - start_time:.4f} sec, {process.memory_info().rss / (1024**3):.2f} GB", flush=True)
        return result
    return wrapper

def empty_like(x):
    if isinstance(x, cp.ndarray):
        return cp.empty_like(x)
    else:
        return np.empty_like(x)

def save_intermediate(obj,prb,pos,path,i):
    write_tiff(obj.real, f"{path}/rec_obj_real/{i:04}")
    if obj.dtype=='complex64':
        write_tiff(obj.imag, f"{path}/rec_obj_imag/{i:04}")
    write_tiff(obj[obj.shape[0]//2].real, f"{path}/rec_objz/{i:04}")
    write_tiff(obj[:, obj.shape[1]//2].real, f"{path}/rec_objy/{i:04}")
    for k in range(prb.shape[0]):
        write_tiff(np.angle(prb[k]), f"{path}/rec_prb_angle{k}/{i:04}")
        write_tiff(np.abs(prb[k]), f"{path}/rec_prb_abs{k}/{i:04}")
    np.save(f"{path}/pos{i:04}", pos)
