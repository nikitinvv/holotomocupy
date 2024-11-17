import cupy as cp
from cuda_kernels import pad_kernel
from chunking import gpu_batch
from utils import *

def _fwd_pad(f):
    """Fwd data padding"""
    [ntheta, n] = f.shape[:2]
    fpad = cp.zeros([ntheta, 2*n, 2*n], dtype='complex64')
    pad_kernel((int(cp.ceil(2*n/32)), int(cp.ceil(2*n/32)), ntheta),
               (32, 32, 1), (fpad, f, n, ntheta, 0))
    return fpad/2


def _adj_pad(fpad):
    """Adj data padding"""
    [ntheta, n] = fpad.shape[:2]
    n //= 2
    f = cp.zeros([ntheta, n, n], dtype='complex64')
    pad_kernel((int(cp.ceil(2*n/32)), int(cp.ceil(2*n/32)), ntheta),
               (32, 32, 1), (fpad, f, n, ntheta, 1))
    return f/2


@gpu_batch
def G(f, wavelength, voxelsize, z, ptype='constant'):
    """Fresnel transform

    Parameters
    ----------
    f : ndarray
        Input 3D array to perform the Fresnel transform wrt the last two dimensions
    wavelength : float
        Wave length in m
    voxelsize : float
        Voxel size in m
    z : float
        Propagation distance in m
    
    Returns
    -------
    ff : ndarray
        Propagated function
    """
    
    n = f.shape[-1]
    fx = cp.fft.fftfreq(2*n, d=voxelsize).astype('float32')
    [fx, fy] = cp.meshgrid(fx, fx)
    fP = cp.exp(-1j*cp.pi*wavelength*z*(fx**2+fy**2))
    ff = f.copy()
    if ptype=='symmetric':
        ff = _fwd_pad(ff)
        v = cp.ones(2*n,dtype='float32')
        v[:n//2] = cp.sin(cp.linspace(0,1,n//2)*cp.pi/2)
        v[-n//2:] = cp.cos(cp.linspace(0,1,n//2)*cp.pi/2)
        v = cp.outer(v,v)
        ff *= v
    else:
        ff = cp.pad(ff,((0,0),(n//2,n//2),(n//2,n//2)))
    ff = cp.fft.ifft2(cp.fft.fft2(ff)*fP)
    # if ptype=='symmetric':
    #     ff = _adj_pad(ff)
    # else:
    ff = ff[:,n//2:-n//2,n//2:-n//2]
    
    return ff*2


@gpu_batch
def GT(f, wavelength, voxelsize, z, ptype='constant'):
    """Adjoint Fresnel transform (propagation with -z distance)

    Parameters
    ----------
    f : ndarray
        Input 3D array to perform the Fresnel transform wrt the last two dimensions
    wavelength : float
        Wave length in m
    voxelsize : float
        Voxel size in m
    z : float
        Propagation distance in m
    
    Returns
    -------
    ff : ndarray
        Propagated function
    """
    
    n = f.shape[-1]
    fx = cp.fft.fftfreq(2*n, d=voxelsize).astype('float32')
    [fx, fy] = cp.meshgrid(fx, fx)
    fP = cp.exp(1j*cp.pi*wavelength*z*(fx**2+fy**2))
    ff= f.copy()
    # if ptype=='symmetric':
    #     ff = _fwd_pad(ff)
    # else:
    ff = cp.pad(ff,((0,0),(n//2,n//2),(n//2,n//2)))
    ff = cp.fft.ifft2(cp.fft.fft2(ff)*fP)
    if ptype=='symmetric':
        ff = _adj_pad(ff)
    else:
        ff = ff[:,n//2:-n//2,n//2:-n//2]
    return ff*2

energy = 33.35  # [keV] xray energy
wavelength = 1.2398419840550367e-09/energy  # [m] wave length

f = 1+cp.random.random([1,2048,2048]).astype('complex64')
f = f*0+1
# f[0,64:-64,64:-64]=2

g = G(f,wavelength,10e-9,4e-3,'symmetric')
print(g.shape)
ff = GT(g,wavelength,10e-9,4e-3,'symmetric')
print(ff.shape)

print(f'{np.sum(f*np.conj(ff))}==\n{np.sum(g*np.conj(g))}')



mshow_polar(g[0],True)

