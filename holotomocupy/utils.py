import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import dxchange
from .chunking import gpu_batch

def mshow(a, show=False, **args):
    """Plot the 2D array, handling arrays on GPU      

    Parameters
    ----------
    a : (ny, nx) float32
        2D array for visualization
    args : 
        Other parameters for imshow    
    """
    if not show:
        return

    if isinstance(a, cp.ndarray):
        a = a.get()
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    im = axs.imshow(a, cmap='gray', **args)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.show()


def mshow_complex(a, show=False, **args):
    """Plot the 2D array in the rectangular representation with the real and imag parts, 
    handling arrays on GPU   

    Parameters
    ----------
    a : (ny, nx) complex64
        2D array for visualization
    args : 
        Other parameters for imshow    
    """
    if not show:
        return
    if isinstance(a, cp.ndarray):
        a = a.get()
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    im = axs[0].imshow(a.real, cmap='gray', **args)
    axs[0].set_title('real')
    fig.colorbar(im, fraction=0.046, pad=0.04)
    im = axs[1].imshow(a.imag, cmap='gray', **args)
    axs[1].set_title('imag')
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.show()


def mshow_polar(a, show=False, **args):
    """Plot the 2D array in the polar representation with the absolute value and phase,
    handling arrays on GPU       

    Parameters
    ----------
    a : (ny, nx) complex64
        2D array for visualization
    args : 
        Other parameters for imshow    
    """
    if not show:
        return
    if isinstance(a, cp.ndarray):
        a = a.get()
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    im = axs[0].imshow(np.abs(a), cmap='gray', **args)
    axs[0].set_title('abs')
    fig.colorbar(im, fraction=0.046, pad=0.04)
    im = axs[1].imshow(np.angle(a), cmap='gray', **args)
    axs[1].set_title('phase')
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.show()


def write_tiff(a, name, **args):
    """Write numpy/cupy array as a tiff file,

    Parameters
    ----------
    a : ndarray
        Input numpy 2D/3D array to write to a tiff file
    name : str
        Output file name
    args : 
        Other parameters for dxchange.write_tiff        

    """
    if isinstance(a, cp.ndarray):
        a = a.get()
    dxchange.write_tiff(a, name, overwrite=True, **args)


def read_tiff(name, **args):
    """Read tiff to a numpy array.

    Parameters
    ----------
    name : str
        Output file name
    args : 
        Other parameters for dxchange.read_tiff 

    Returns
    -------
    a : ndarray
        Output numpy array
    """
    
    a = dxchange.read_tiff(name, **args)[:]
    return a

def reprod(a,b):
    return a.real*b.real+a.imag*b.imag

def redot(a,b,axis=None):    
    res = np.sum(reprod(a,b),axis=axis)        
    return res

def improd(a,b):
    return -a.real*b.imag+a.imag*b.real

def imdot(a,b,axis=None):    
    res = np.sum(improd(a,b),axis=axis)        
    return res
