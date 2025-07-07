import cupy as cp
import tifffile
import sys
sys.path.insert(0, '..')
from cuda_kernels import *
import h5py

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
    return m, mu, phi, c1dfftshift, c2dfftshift
        
def _R(data, obj, theta, rotation_axis):
                
    [nz, n, n] = obj.shape
    ntheta=len(theta)
    theta = cp.array(theta, dtype='float32')
    m, mu, phi, c1dfftshift, c2dfftshift = initR(n)
    sino = cp.zeros([ntheta,nz,  n], dtype='complex64')

    # STEP0: multiplication by phi, padding
    fde = obj*phi
    fde = cp.pad(fde, ((0, 0), (n//2, n//2), (n//2, n//2)))
    # STEP1: fft 2d
    fde = cp.fft.fft2(fde*c2dfftshift)*c2dfftshift
    fde = cp.pad(fde, ((0, 0), (m, m), (m, m)))
    # STEP2: fft 2d
    wrap_kernel((int(cp.ceil((2 * n + 2 * m)/32)),
                int(cp.ceil((2 * n + 2 * m)/32)), nz), (32, 32, 1), (fde, n, nz, m))

    mua = cp.array([mu], dtype='float32')

    gather_kernel((int(cp.ceil(n/32)), int(cp.ceil(ntheta/32)), nz),
                (32, 32, 1), (sino, fde, theta, m, mua, n, ntheta, nz, 0))
    # STEP3: ifft 1d
    sino = cp.fft.ifft(c1dfftshift*sino)*c1dfftshift

    # STEP4: Shift based on the rotation axis
    t = cp.fft.fftfreq(n).astype('float32')
    w = cp.exp(-2*cp.pi*1j*t*(rotation_axis + n/2))
    sino = cp.fft.ifft(w*cp.fft.fft(sino))
    # normalization for the unity test
    sino /= cp.float32(4*n)#*np.sqrt(n*self.ntheta))                

    data[:] = sino

def _RT(obj, data, theta,rotation_axis):
    [ntheta,nz,npsi] = data.shape
    [m, mu, phi, c1dfftshift, c2dfftshift] = initR(npsi)
    t = (cp.arange(-npsi//2,npsi//2)/npsi).astype('float32')
    w = cp.exp(2*cp.pi*1j*t*(rotation_axis - npsi/2)).astype('complex64')            
    sino = cp.fft.fft(c1dfftshift*data)*c1dfftshift*w
    mua = cp.array([mu], dtype='float32')
    fde = cp.zeros([nz, 2*m+2*npsi, 2*m+2*npsi], dtype='complex64')
    gather_kernel((int(cp.ceil(npsi/32)), int(cp.ceil(ntheta/32)), nz),
                (32, 32, 1), (sino, fde, theta, m, mua, npsi, ntheta, nz, 1))
    wrapadj_kernel((int(cp.ceil((2 * npsi + 2 * m)/32)),
                    int(cp.ceil((2 * npsi + 2 * m)/32)), nz), (32, 32, 1), (fde, npsi, nz, m))
    fde = fde[:, m:-m, m:-m]
    fde = cp.fft.ifft2(fde*c2dfftshift)*c2dfftshift
    fde = fde[:, npsi//2:3*npsi//2, npsi//2:3*npsi//2]*phi

    obj[:] = fde/npsi
 
def _wint(n, t):

    N = len(t)
    s = cp.linspace(1e-40, 1, n)
    # Inverse vandermonde matrix
    tmp1 = cp.arange(n)
    tmp2 = cp.arange(1, n+2)
    iv = cp.linalg.inv(cp.exp(cp.outer(tmp1, cp.log(s))))
    u = cp.diff(cp.exp(cp.outer(tmp2, cp.log(s)))*cp.tile(1.0 /
                tmp2[..., cp.newaxis], [1, n]))  # integration over short intervals
    W1 = cp.matmul(iv, u[1:n+1, :])  # x*pn(x) term
    W2 = cp.matmul(iv, u[0:n, :])  # const*pn(x) term

    # Compensate for overlapping short intervals
    tmp1 = cp.arange(1, n)
    tmp2 = (n-1)*cp.ones((N-2*(n-1)-1))
    tmp3 = cp.arange(n-1, 0, -1)
    p = 1/cp.concatenate((tmp1, tmp2, tmp3))
    w = cp.zeros(N)
    for j in range(N-n+1):
        # Change coordinates, and constant and linear parts
        W = ((t[j+n-1]-t[j])**2)*W1+(t[j+n-1]-t[j])*t[j]*W2

        for k in range(n-1):
            w[j:j+n] = w[j:j+n] + p[j+k]*W[:, k]

    wn = w
    wn[-40:] = (w[-40])/(N-40)*cp.arange(N-40, N)
    return wn

def calc_filter(n, filter):
    d = 0.5
    t = cp.arange(0, n/2+1)/n

    if filter == 'none':
        wfa = n*0.5+t*0
        wfa[0] *= 2  # fixed later
    elif filter == 'ramp':
        wfa = n*0.5*_wint(12, t)
    elif filter == 'shepp':
        wfa = n*0.5*_wint(12, t)*cp.sinc(t/(2*d))*(t/d <= 2)
    elif filter == 'cosine':
        wfa = n*0.5*_wint(12, t)*cp.cos(cp.pi*t/(2*d))*(t/d <= 1)
    elif filter == 'cosine2':
        wfa = n*0.5*_wint(12, t) * \
            (cp.cos(cp.pi*t/(2*d)))**2*(t/d <= 1)
    elif filter == 'hamming':
        wfa = n*0.5 * \
            _wint(12, t)*(.54 + .46 * cp.cos(cp.pi*t/d))*(t/d <= 1)
    elif filter == 'hann':
        wfa = n*0.5*_wint(12, t) * \
            (1+cp.cos(cp.pi*t/d)) / 2.0*(t/d <= 1)
    elif filter == 'parzen':
        wfa = n*0.5*_wint(12, t)*pow(1-t/d, 3)*(t/d <= 1)

    wfa = 2*wfa*(wfa >= 0)
    wfa[0] *= 2
    wfa = wfa.astype('float32')
    return wfa

               
pfile = f'Y350c_HT_015nm'
path_out = f'/data/vnikitin/ESRF/ID16A/brain_rec/20240515/Y350c'
with h5py.File(f'{path_out}/{pfile}.h5') as fid:
    detector_pixelsize = fid['/exchange/detector_pixelsize'][0]    
    focusToDetectorDistance = fid['/exchange/focusdetectordistance'][0]    
    theta = fid['/exchange/theta'][:,0]

obj = tifffile.imread('/data/vnikitin/ESRF/ID16A/brain_rec/20240515/Y350c/Y350c_HT_015nm/rec_final4_4_1763.0_0.0_50.0_20.0/rec_uz/0080.tiff')
obj=cp.array(obj[cp.newaxis].astype('complex64'))
theta = cp.array(theta)
rotation_axis=(879-(1616-obj.shape[-1]//2)//2+2.5)*2

data = cp.zeros([len(theta),1,obj.shape[-1]],dtype='complex64')    
rec=0*cp.array(obj.astype('complex64'))

n = data.shape[-1]
_R(data,obj,theta,rotation_axis)
filt = calc_filter(2*n,'parzen')
data = cp.pad(data,((0,0),(0,0),(n//2,n//2)))
data = cp.fft.irfft(cp.fft.rfft(data.real)*filt)[...,n//2:-n//2].astype('complex64')
_RT(rec,data,theta,rotation_axis)


print(cp.linalg.norm(data))
print(cp.sum(data*data.conj()))
print(cp.sum(obj*rec.conj()))
print(rec.shape,data.shape)
# tifffile.imwrite('/data/tmp/r/r/0',rec[0].real.get())
# tifffile.imwrite('/data/tmp/r/r/1',rec[0].real.get())
tifffile.imwrite('/data/tmp/r/r/2',rec[0].real.get())


