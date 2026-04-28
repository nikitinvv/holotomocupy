import numpy as np
import cupy as cp
from scipy.fft import fftn, ifftn, fftshift, fft2, ifft2
import scipy.ndimage as ndimage
from mpi4py import MPI
from types import SimpleNamespace

from holotomocupy.rec_mpi import Rec
from holotomocupy.writer import Writer
from holotomocupy.utils import *
from holotomocupy.logger_config import set_log_level
set_log_level('INFO')

#### Acquisition Parameters
n      = 256                                          # detector size (pixels)
ntheta = 360                                          # number of projection angles
ndist  = 4                                            # number of propagation distances

energy                  = 17.1                        # X-ray energy (keV)
detector_pixelsize      = 1.4760147601476e-6 * 2 * 8 # effective pixel size (m), binned
focustodetectordistance = 1.217                       # focus-to-detector distance (m)
z1 = np.array([5.110, 5.464, 6.879, 9.817]) * 1e-3  # sample-to-focus distances (m)

nobj = 3 * n // 2  # object volume side length (pixels)

#### Synthetic Phantom Object
def _draw_frame_edges_inplace(cube, p1, p2):
    cube[p1:p2, p1, p1] = 1; cube[p1:p2, p1, p2] = 1
    cube[p1:p2, p2, p1] = 1; cube[p1:p2, p2, p2] = 1
    cube[p1, p1:p2, p1] = 1; cube[p1, p1:p2, p2] = 1
    cube[p2, p1:p2, p1] = 1; cube[p2, p1:p2, p2] = 1
    cube[p1, p1, p1:p2] = 1; cube[p1, p2, p1:p2] = 1
    cube[p2, p1, p1:p2] = 1; cube[p2, p2, p1:p2] = 1

def rotate3d_once(vol, ang_xy_deg=28, ang_xz_deg=45, order=1):
    a = np.deg2rad(ang_xy_deg)
    b = np.deg2rad(ang_xz_deg)
    Rz = np.array([[ np.cos(a), -np.sin(a), 0],
                   [ np.sin(a),  np.cos(a), 0],
                   [ 0,          0,         1]], dtype=np.float64)
    Ry = np.array([[ np.cos(b), 0, np.sin(b)],
                   [ 0,         1, 0        ],
                   [-np.sin(b), 0, np.cos(b)]], dtype=np.float64)
    R = Ry @ Rz
    A = np.linalg.inv(R)
    center = (np.array(vol.shape) - 1) / 2.0
    offset = center - A @ center
    return ndimage.affine_transform(
        vol, A, offset=offset, order=order, mode="constant", cval=0.0, prefilter=(order > 1)
    )

def gen_object(n, delta, beta):
    obj = np.zeros((n, n, n), dtype=np.float32)
    rr = (np.ones(8) * n * 0.2).astype(np.int32)
    amps = np.array([3, -3, 1, 3, -4, 1, 4], dtype=np.float32)
    dil  = (np.array([33, 28, 25, 21, 16, 10, 3], dtype=np.float32) / 256.0) * n
    ax = np.arange(-n//2, n//2, dtype=np.float32)
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    r2 = x*x + y*y + z*z
    del x, y, z
    fcirc_list = []
    for d in dil:
        circ = (r2 < (d*d)).astype(np.float32, copy=False)
        fcirc_list.append(fftn(fftshift(circ), workers=-1).astype(np.complex64, copy=False))
    cube = np.zeros((n, n, n), dtype=np.float32)
    fcube_list = []
    for kk in range(len(amps)):
        cube.fill(0.0)
        r = int(rr[kk])
        p1 = n//2 - r//2
        p2 = n//2 + r//2
        _draw_frame_edges_inplace(cube, p1, p2)
        fcube_list.append(fftn(fftshift(cube), workers=-1).astype(np.complex64, copy=False))
    work = np.empty((n, n, n), dtype=np.complex64)
    for kk, a in enumerate(amps):
        np.multiply(fcube_list[kk], fcirc_list[kk], out=work)
        conv = fftshift(ifftn(work, workers=-1)).real
        obj += a * (conv > 1.0)
    obj = rotate3d_once(obj, 28, 45, order=1)
    obj = np.roll(obj, -15*n//256, axis=2)
    obj = np.roll(obj, -10*n//256, axis=1)
    np.maximum(obj, 0, out=obj)
    v = (np.arange(-n//2, n//2, dtype=np.float32) / n)
    vx, vy, vz = np.meshgrid(v, v, v, indexing="ij")
    filt = fftshift(np.exp(-3.0 * (vx*vx + vy*vy + vz*vz)).astype(np.float32))
    fu = fftn((obj))
    obj = ifftn((fu * filt)).real
    obj[obj < 0] = 0
    return (obj * (-delta + 1j*beta)).astype(np.complex64, copy=False)

obj = gen_object(nobj, 1, 1e-2)

#### Probe — load from pre-saved ID16A TIFF files
prb_abs   = read_tiff('data/prb_id16a/prb_abs_2048.tiff')[:ndist]
prb_phase = read_tiff('data/prb_id16a/prb_phase_2048.tiff')[:ndist]
prb = prb_abs * np.exp(1j * prb_phase).astype('complex64')
prb = prb[:, prb.shape[1]//2-n//2:prb.shape[1]//2+n//2,
             prb.shape[2]//2-n//2:prb.shape[2]//2+n//2]
v = (np.arange(-n//2, n//2, dtype=np.float32) / n)
vx, vy = np.meshgrid(v, v, indexing="ij")
filt = fftshift(np.exp(-4.0 * (vx*vx + vy*vy)).astype(np.float32))
fu = fft2((prb))
prb = ifft2((fu * filt))
prb /= np.mean(np.abs(prb), axis=(1, 2))[:, None, None]

#### Angles and Positions
np.random.seed(10)
pos     = 30 * (np.random.random([ntheta, ndist, 2]).astype('float32') - 0.5)
pos_err =      (np.random.random([ntheta, ndist, 2]).astype('float32') - 0.5)
theta   = np.linspace(0, np.pi, ntheta, dtype='float32')

#### Initialise Rec
args = SimpleNamespace()

# --- acquisition / physics ---
args.energy                  = energy                  # X-ray energy (keV)
args.detector_pixelsize      = detector_pixelsize      # effective pixel size (m)
args.focustodetectordistance = focustodetectordistance # focus-to-detector distance (m)
args.z1                      = z1                      # sample-to-focus distances per distance (m)
args.theta                   = theta                   # projection angles (radians)
args.ndist                   = ndist                   # number of propagation distances
args.ntheta                  = ntheta                  # number of projections
args.nz                      = n                       # detector height (pixels)
args.n                       = n                       # detector width (pixels)
args.nzobj                   = nobj                    # object volume height (pixels)
args.nobj                    = nobj                    # object volume width/depth (pixels)

# --- solver / regularisation ---
args.obj_dtype   = 'complex64'      # object dtype: 'complex64' (phase+absorption) or 'float32' (phase only)
args.mask        = 0.9              # support mask radius as fraction of field of view
args.lam_prbfit  = 2e-3            # probe-fit regularisation weight
args.rho         = [1, 0.05, 0.02] # gradient step-size scales for [obj, prb, pos]
args.niter       = 129             # total number of BH iterations
args.nchunk      = 16              # projections/slices processed per GPU pass (tune to GPU memory)
args.vis_step    = 16              # save checkpoint every N iterations (-1 = never)
args.err_step    = 4               # log error every N iterations (-1 = never)
args.start_iter  = 0               # resume from this iteration (0 = fresh start)

# --- MPI ---
args.comm = MPI.COMM_WORLD

cl = Rec(args)

#### Create Writer
writer = Writer(
    path_out    = 'test_results',
    comm        = args.comm,
    st_obj      = cl.st_obj,
    end_obj     = cl.end_obj,
    nzobj       = nobj,
    nobj        = nobj,
    st_theta    = cl.st_theta,
    end_theta   = cl.end_theta,
    ntheta      = ntheta,
    ndist       = ndist,
    nz          = n,
    n           = n,
    obj_dtype   = args.obj_dtype,
)

#### Set Ground-Truth Variables and Generate Synthetic Data
# Each rank owns a slice of obj (obj-axis) and pos (theta-axis)
cl.vars['obj'][:] = obj[cl.st_obj:cl.end_obj]
cl.vars['prb'][:] = cp.array(prb)
cl.vars['pos'][:] = cp.array(pos[cl.st_theta:cl.end_theta])

cl.gen_sqrt_data(cl.vars, cl.data)
cl.gen_sqrt_ref(cl.vars['prb'], cl.ref)

#### Reconstruction
cl.vars['obj'][:] = 0
cl.vars['prb'][:] = cp.array(1)
cl.vars['pos'][:] = cp.array((pos + pos_err)[cl.st_theta:cl.end_theta])

cl.BH(writer=writer)

if MPI.COMM_WORLD.Get_rank() == 0:
    import os
    print(f"\nCheckpoints saved to: {os.path.abspath('test_results')}/checkpoint_NNNN.h5")
