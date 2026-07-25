"""Coordinate search over rho on the test-8 configuration.

Runs many short BH trials (16 iterations each) with different rho values,
picking the best one variable at a time in the order prb → pos → tp.
`obj` rho stays at 1 throughout (that's the reference normalization).

Search per variable:
  1. Try rho = 1 (init), 2, 0.5. Take whichever gives the lowest final err.
  2. If 1 is best  →  keep 1, move to the next variable.
  3. If 2 is best  →  try 4, 8, 16, ... doubling until it stops improving.
  4. If 0.5 is best →  try 0.25, 0.125, ... halving until it stops improving.

Once a variable's best rho is fixed, move on. The next variable starts with
the previously fixed rho values as its base (only its own rho varies).

Emits `rho_coord_search.png` (err vs rho, one subplot per variable) and
prints the final [obj, prb, pos, tp] rho vector."""

import sys
import numpy as np
import cupy as cp
from scipy.fft import fftn, ifftn, fftshift, fft2, ifft2
import scipy.ndimage as ndimage
from mpi4py import MPI
from types import SimpleNamespace
import pandas as pd

_cfg = dict(rho2=2e-1, rho3=1e-3, pos_err_amp=2.0, use_shrink_recon=False)
if MPI.COMM_WORLD.Get_rank() == 0:
    print(f'==== TEST rho_coord: {_cfg} ====', flush=True)

from holotomocupy.rec_mpi_shrink import Rec
from holotomocupy.utils import *
from holotomocupy.logger_config import set_log_level
set_log_level('INFO')

#### Acquisition parameters (same as test_hessian_rho.py)
n      = 256
ntheta = 360
ndist  = 4
energy                  = 17.1
detector_pixelsize      = 1.4760147601476e-6 * 2 * 8
focustodetectordistance = 1.217
z1 = np.array([5.110, 5.464, 6.879, 9.817]) * 1e-3
nobj = 3 * n // 2

#### Synthetic phantom
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
    _ax_c = np.arange(-n//2, n//2, dtype=np.float32)
    _Zc, _Yc, _Xc = np.meshgrid(_ax_c, _ax_c, _ax_c, indexing='ij')
    _rng_tube = np.random.default_rng(9)
    _n_tubes  = 60
    _tube_amp = 6.2
    for _ in range(_n_tubes):
        _center = _rng_tube.uniform(-n * 0.25, n * 0.25, 3).astype(np.float32)
        _axis   = _rng_tube.standard_normal(3).astype(np.float32)
        _axis  /= np.linalg.norm(_axis)
        _r_out  = float(_rng_tube.uniform(3.0, 8.0))
        _thick  = float(_rng_tube.uniform(1.0, 2.0))
        _r_in   = max(_r_out - _thick, 0.5)
        _length = float(_rng_tube.uniform(n * 0.15, n * 0.35))
        _dz = _Zc - _center[0]
        _dy = _Yc - _center[1]
        _dx = _Xc - _center[2]
        _along = _dz * _axis[0] + _dy * _axis[1] + _dx * _axis[2]
        _perp2 = _dz * _dz + _dy * _dy + _dx * _dx - _along * _along
        _shell = (
            (_perp2 < _r_out * _r_out)
            & (_perp2 >= _r_in * _r_in)
            & (np.abs(_along) < 0.5 * _length)
        )
        obj[_shell] += _tube_amp
    del _ax_c, _Zc, _Yc, _Xc, _dz, _dy, _dx, _along, _perp2, _shell
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

obj = gen_object(nobj, 2, 2e-2)

#### Probe
_data_dir = '/home/beams/VNIKITIN/holotomocupy_mpi_deform/tests/holotomo3d/data'
prb_abs   = read_tiff(f'{_data_dir}/prb_abs_2048.tiff')[:ndist]
prb_phase = read_tiff(f'{_data_dir}/prb_phase_2048.tiff')[:ndist]
prb = prb_abs * np.exp(1j * prb_phase).astype('complex64')
prb = prb[:, prb.shape[1]//2-n//2:prb.shape[1]//2+n//2,
             prb.shape[2]//2-n//2:prb.shape[2]//2+n//2]
v = (np.arange(-n//2, n//2, dtype=np.float32) / n)
vx, vy = np.meshgrid(v, v, indexing="ij")
filt = fftshift(np.exp(-4.0 * (vx*vx + vy*vy)).astype(np.float32))
fu = fft2((prb))
prb = ifft2((fu * filt))
prb /= np.mean(np.abs(prb), axis=(1, 2))[:, None, None]

#### Angles and positions
np.random.seed(15)
pos     = 30 * (np.random.random([ntheta, ndist, 2]).astype('float32') - 0.5)
pos_err = (_cfg['pos_err_amp'] * (np.random.random([ntheta, ndist, 2]).astype('float32') - 0.5)).astype('float32')
theta   = np.linspace(0, np.pi, ntheta, dtype='float32')

#### Rec — short BH per trial (16 iters), error every iter, no re-estimate
args = SimpleNamespace()
args.energy                  = energy
args.detector_pixelsize      = detector_pixelsize
args.focustodetectordistance = focustodetectordistance
args.z1                      = z1
args.theta                   = theta
args.ndist                   = ndist
args.ntheta                  = ntheta
args.nz                      = n
args.n                       = n
args.nzobj                   = nobj
args.nobj                    = nobj

args.obj_dtype   = 'complex64'
args.mask        = 0.9
args.lam_prbfit  = 2e-3
args.rho         = [1.0, 5e-2, 2e-2, 2e-4]   # hand-tuned baseline; used as init for the coord search
args.niter       = 513                     # full BH run after rho is found
args.nchunk      = 32
args.checkpoint_step = -1                  # disables check_approximation's inline plt.show()
args.error_step      = 4
args.start_iter  = 0
args.lam_laplacian = 0
args.shift_type  = 'cubic'
# Rho tuning: run coordinate search on rho[prb, pos, tp] before BH iterates.
args.estimate_rho       = True
args.rho_estimate_niter = 16               # BH iters per trial in the search
args.comm = MPI.COMM_WORLD

cl = Rec(args)

#### GT tp
tanh_k_gt   = 2.0
end_shrink  = np.array([0.06, 0.12], dtype='float32')
dist_decay  = 0.7
_geom_sum   = (1.0 - dist_decay ** ndist) / (1.0 - dist_decay)
_A_base     = end_shrink / (np.tanh(tanh_k_gt) * _geom_sum)
A_gt        = (_A_base[None, :] * (dist_decay ** np.arange(ndist)[:, None])).astype('float32')
_per_dist_end = A_gt * np.tanh(tanh_k_gt)
B_gt        = np.zeros_like(A_gt)
B_gt[1:]    = np.cumsum(_per_dist_end[:-1], axis=0)
tp_gt = np.zeros((ndist, 3, 2), dtype='float32')
tp_gt[:, 0, :] = np.sqrt(A_gt)
tp_gt[:, 1, :] = np.sqrt(tanh_k_gt)
tp_gt[:, 2, :] = B_gt
_t_all      = (np.arange(ntheta, dtype='float32') / max(ntheta - 1, 1))
shrink_nd   = (B_gt[None, :, :]
               + A_gt[None, :, :] * np.tanh(tanh_k_gt * _t_all[:, None, None])).astype('float32')

#### Ground-truth vars → generate synthetic data
cl.vars['obj'][:] = obj[cl.st_obj:cl.end_obj]
cl.vars['prb'][:] = cp.array(prb)
cl.vars['pos'][:] = cp.array(pos[cl.st_theta:cl.end_theta])
cl.vars['tp'][:]  = cp.asarray(tp_gt)

cl.gen_sqrt_data(cl.vars, cl.data)
cl.cl_prb_term.gen_sqrt_ref(cl.vars['prb'], cl.ref)

#### Initial guesses (same as test_hessian_rho.py)
_sigma_blur = nobj / 16
_obj_blur_re = ndimage.gaussian_filter(obj.real, sigma=_sigma_blur)
_obj_blur_im = ndimage.gaussian_filter(obj.imag, sigma=_sigma_blur)
_obj_blur    = (_obj_blur_re + 1j * _obj_blur_im).astype('complex64')
cl.vars['obj'][:] = _obj_blur[cl.st_obj:cl.end_obj]
del _obj_blur_re, _obj_blur_im, _obj_blur

cl.vars['prb'][:] = cp.array(1)
cl.vars['pos'][:] = cp.array((pos+pos_err)[cl.st_theta:cl.end_theta])

_tp_init = np.zeros_like(tp_gt)
_mag  = 0.3 + np.random.random([ndist, 2]) * 0.2
_sign = np.where(np.random.random([ndist, 2]) < 0.5, -1.0, 1.0)
err   = 1.0 + _sign * _mag
_A_init = A_gt * err[:, 0:1]
_k_init = tanh_k_gt * err[:, 1:2]
_tp_init[:, 0, :] = np.sqrt(_A_init)
_tp_init[:, 1, :] = np.sqrt(_k_init)
_ends_init      = _A_init * np.tanh(_k_init)
_tp_init[0, 2, :] = 0.0
if ndist > 1:
    _tp_init[1:, 2, :] = np.cumsum(_ends_init[:-1], axis=0)
cl.vars['tp'][:] = cp.asarray(_tp_init)

#### Run BH — args.estimate_rho=True triggers cl.estimate_rho_coord() inside
# BH before iteration. The coord search writes its history to cl._rho_history
# (captured by a light monkey-patch below) so we can plot it after the run.
_search_history = {}
_orig_estimate = cl.estimate_rho_coord
def _estimate_wrap(vars, grads, etas, **kw):
    h = _orig_estimate(vars, grads, etas, **kw)
    _search_history.update(h)
    return h
cl.estimate_rho_coord = _estimate_wrap

cl.BH(writer=None, shrink_gt=shrink_nd)

if cl.rank == 0:
    best_prb = float(np.sqrt(cl.rho_sq['prb']))
    best_pos = float(np.sqrt(cl.rho_sq['pos']))
    best_tp  = float(np.sqrt(cl.rho_sq['tp']))
    print(f'\n==== FINAL best rho = [obj=1, prb={best_prb:g}, '
          f'pos={best_pos:g}, tp={best_tp:g}] ====', flush=True)

#### Plot
if cl.rank == 0 and _search_history:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    per_var = [
        ('prb', _search_history['prb'], best_prb),
        ('pos', _search_history['pos'], best_pos),
        ('tp',  _search_history['tp'],  best_tp),
    ]
    for ax, (name, hist, best) in zip(axes, per_var):
        xs = [v for v, _ in hist]
        ys = [e for _, e in hist]
        ax.plot(xs, ys, 'o-', markersize=6)
        best_err = dict(hist)[best]
        ax.plot([best], [best_err], marker='*', markersize=18,
                color='red', linestyle='none', label=f'best={best:g}')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel(f'rho[{name}]')
        ax.set_ylabel(f'err after {args.rho_estimate_niter} iters')
        ax.set_title(f'{name}  search')
        ax.grid(True, which='both', linestyle=':')
        ax.legend(loc='best')
    fig.suptitle(f'rho coordinate search (test-8 config, '
                 f'{args.rho_estimate_niter} iters/trial)\n'
                 f'best rho = [obj=1, prb={best_prb:g}, pos={best_pos:g}, tp={best_tp:g}]')
    fig.tight_layout()
    _out = 'rho_coord_search.png'
    fig.savefig(_out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {_out}')
