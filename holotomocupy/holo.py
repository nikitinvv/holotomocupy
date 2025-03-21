import cupy as cp
from .cuda_kernels import pad_kernel, pad_sym_kernel
from .chunking import gpu_batch


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

# @gpu_batch
# def G(f, wavelength, voxelsize, z, ptype='constant'):
#     n = f.shape[-1]
#     fx = cp.fft.fftfreq(n, d=voxelsize).astype('float32')
#     [fx, fy] = cp.meshgrid(fx, fx)
#     fP = cp.exp(-1j*cp.pi*wavelength*z*(fx**2+fy**2))
#     ff = f.copy()
#     ff = cp.fft.ifft2(cp.fft.fft2(ff)*fP)
#     return ff


# @gpu_batch
# def GT(f, wavelength, voxelsize, z, ptype='constant'):
#     n = f.shape[-1]
#     fx = cp.fft.fftfreq(n, d=voxelsize).astype('float32')
#     [fx, fy] = cp.meshgrid(fx, fx)
#     fP = cp.exp(1j*cp.pi*wavelength*z*(fx**2+fy**2))
#     ff= f.copy()
#     ff = cp.fft.ifft2(cp.fft.fft2(ff)*fP)
#     return ff
def _G0(f, wavelength, voxelsize, z):
    n = f.shape[-1]
    fx = cp.fft.fftfreq(n, d=voxelsize).astype('float32')
    [fx, fy] = cp.meshgrid(fx, fx)
    fP = cp.exp(-1j*cp.pi*wavelength*z*(fx**2+fy**2))
    ff = f.copy()
    ff = cp.fft.ifft2(cp.fft.fft2(ff)*fP)
    return ff

def _GT0(f, wavelength, voxelsize, z):
    n = f.shape[-1]
    fx = cp.fft.fftfreq(n, d=voxelsize).astype('float32')
    [fx, fy] = cp.meshgrid(fx, fx)
    fP = cp.exp(1j*cp.pi*wavelength*z*(fx**2+fy**2))
    ff = f.copy()
    ff = cp.fft.ifft2(cp.fft.fft2(ff)*fP)
    return ff


@gpu_batch
def G(f, wavelength, voxelsize, z,ptype='constant',fP=None):
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
    if ptype=='no':
        return _G0(f,wavelength, voxelsize, z)
    n = f.shape[-1]
    fx = cp.fft.fftfreq(2*n, d=voxelsize).astype('float32')
    [fx, fy] = cp.meshgrid(fx, fx)
    if fP is None:
        fP = cp.exp(-1j*cp.pi*wavelength*z*(fx**2+fy**2))
    ff = f.copy()
    if ptype=='symmetric':
        ff = _fwd_pad(ff)
        v = cp.ones(2*n,dtype='float32')
        v[:n//2] = cp.sin(cp.linspace(0,1,n//2)*cp.pi/2)
        v[-n//2:] = cp.cos(cp.linspace(0,1,n//2)*cp.pi/2)
        v = cp.outer(v,v)
        ff *= v*2        
    else:
        ff = cp.pad(ff,((0,0),(n//2,n//2),(n//2,n//2)))
    ff = cp.fft.ifft2(cp.fft.fft2(ff)*fP)
    ff = ff[:,n//2:-n//2,n//2:-n//2]
    
    return ff


@gpu_batch
def GT(f, wavelength, voxelsize, z, ptype='constant',fP=None):
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
    if ptype=='no':
        return _GT0(f,wavelength, voxelsize, z)
    n = f.shape[-1]
    fx = cp.fft.fftfreq(2*n, d=voxelsize).astype('float32')
    [fx, fy] = cp.meshgrid(fx, fx)
    if fP is None:
        fP = cp.exp(1j*cp.pi*wavelength*z*(fx**2+fy**2))
    ff= f.copy()
    
    ff = cp.pad(ff,((0,0),(n//2,n//2),(n//2,n//2)))
    ff = cp.fft.ifft2(cp.fft.fft2(ff)*fP)
    if ptype=='symmetric':
        v = cp.ones(2*n,dtype='float32')
        v[:n//2] = cp.sin(cp.linspace(0,1,n//2)*cp.pi/2)
        v[-n//2:] = cp.cos(cp.linspace(0,1,n//2)*cp.pi/2)
        v = cp.outer(v,v)
        ff*=v*2
        ff = _adj_pad(ff)
    else:
        ff = ff[:,n//2:-n//2,n//2:-n//2]
    return ff


def _fwd_pad_sym(f,pad):
    """Fwd data padding"""
    [ntheta, n] = f.shape[:2]
    fpad = cp.zeros([ntheta, n+2*pad, n+2*pad], dtype='complex64')
    pad_sym_kernel((int(cp.ceil((n+2*pad)/32)), int(cp.ceil((n+2*pad)/32)), ntheta),
               (32, 32, 1), (fpad, f, pad, n, n, ntheta, 0))
    return fpad

def _adj_pad_sym(fpad,pad):
    """Fwd data padding"""
    [ntheta, n] = fpad.shape[:2]
    n-=2*pad
    f = cp.zeros([ntheta, n, n], dtype='complex64')
    pad_sym_kernel((int(cp.ceil((n+2*pad)/32)), int(cp.ceil((n+2*pad)/32)), ntheta),
               (32, 32, 1), (fpad, f, pad, n, n, ntheta, 1))
    return f

# import matplotlib.pyplot as plt
# import tifffile
# a =  cp.array(tifffile.imread('../../tests/data/delta-chip-192.tiff'))
# a = a-1j*a
# z = 4e-3
# wavelength = 1.2398419840550367e-09/17.05
# voxelsize = 10e-9*8
# aa = G(a, wavelength, voxelsize, z)
# aaa = GT(aa, wavelength, voxelsize, z)
# print(cp.sum(a*cp.conj(aaa)))
# print(cp.sum(aa*cp.conj(aa)))


# plt.figure()
# plt.imshow(aa[92].real.get())
# plt.colorbar()
# plt.savefig('t.png')


# import tifffile
# a =  cp.array(tifffile.imread('../../tests/data/delta-chip-192.tiff'))
# a = a-1j*a
# aa = fwd_pad(a)
# aaa = adj_pad(aa)

# plt.figure()
# plt.imshow(aa[92].real.get())
# plt.colorbar()
# plt.savefig('t.png')

# plt.figure()
# plt.imshow(aaa[92].real.get())
# plt.colorbar()
# plt.savefig('t1.png')
