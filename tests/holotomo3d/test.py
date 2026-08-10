import sys
import numpy as np
import cupy as cp
from scipy.fft import fftn, ifftn, fftshift, fft2, ifft2
import scipy.ndimage as ndimage
from mpi4py import MPI
from types import SimpleNamespace

# --- Test-variant selector (1..8) ---
# rho[2]=pos step, rho[3]=demag step. Cases 1-7 keep rho[3]=1e-6 (demag effectively
# frozen at its init). Case 8 uses rho[3]=2e-2: demag actively optimized alongside
# obj/prb/pos (starts from no-shrink init since use_shrink_recon=False).
# 1: no pos_err, correct shrink for recon
# 2: no pos_err, no shrink for recon (data still has shrink)
# 3-4: pos_err amp=2, correct shrink
# 5: pos_err amp=2, no shrink for recon
# 6: pos_err amp=2, correct shrink, larger rho[2]
# 7: pos_err amp=2, no shrink for recon, larger rho[2]
# 8: pos_err amp=2, no shrink init, demag is OPTIMIZED (rho[3]=2e-2)
TEST_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 1
_configs = {
    1: dict(rho2=1e-6, rho3=1e-6, pos_err_amp=0.0, use_shrink_recon=True),
    2: dict(rho2=1e-6, rho3=1e-6, pos_err_amp=0.0, use_shrink_recon=False),
    3: dict(rho2=1e-6, rho3=1e-6, pos_err_amp=2.0, use_shrink_recon=True),
    4: dict(rho2=1e-6, rho3=1e-6, pos_err_amp=2.0, use_shrink_recon=True),
    5: dict(rho2=1e-6, rho3=1e-6, pos_err_amp=2.0, use_shrink_recon=False),
    6: dict(rho2=2e-2, rho3=1e-6, pos_err_amp=2.0, use_shrink_recon=True),
    7: dict(rho2=2e-2, rho3=1e-6, pos_err_amp=2.0, use_shrink_recon=False),
    8: dict(rho2=2e-2, rho3=1e-4, pos_err_amp=2.0, use_shrink_recon=False),
}
_cfg = _configs[TEST_ID]
if MPI.COMM_WORLD.Get_rank() == 0:
    print(f'==== TEST {TEST_ID}: {_cfg} ====', flush=True)

from holotomocupy.rec_mpi_shrink import Rec
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

    # --- add hollow cylinders (shell only) of different sizes/orientations ---
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

#### Probe — load from pre-saved ID16A TIFF files
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

#### Angles and Positions
np.random.seed(15)
pos     = 30 * (np.random.random([ntheta, ndist, 2]).astype('float32') - 0.5)
pos_err = (_cfg['pos_err_amp'] * (np.random.random([ntheta, ndist, 2]).astype('float32') - 0.5)).astype('float32')
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
args.rho         = [1, 0.05, _cfg['rho2'], _cfg['rho3']]  # step-size scales: [obj, prb, pos, demag]
args.niter       = 513             # total number of BH iterations
args.nchunk      = 32              # projections/slices processed per GPU pass (tune to GPU memory)
args.checkpoint_step = 16          # save checkpoint every N iterations (-1 = never)
args.error_step      = 4           # log error every N iterations (-1 = never)
args.start_iter  = 0               # resume from this iteration (0 = fresh start)
args.lam_laplacian = 0
args.shift_type  = 'cubic'           # 'fft' or 'cubic'
# --- MPI ---
args.comm = MPI.COMM_WORLD

cl = Rec(args)

#### Ground-truth linear-parameterized shrinkage with per-dist offset
# rec_mpi_shrink's F4 models  shrink(t; A, B) = A · t + B  with t = θ_idx/(ntheta−1).
# Per (dist, axis): A is the slope, B is the intercept — both free floats.
# B is built by continuity so the shrink profile is continuous across distances:
# at t = 1 shrink is A + B, so B[j] = Σ_{i<j} A[i]  (with B[0] = 0).
# Target total: 1.5% (y) / 3% (x) at last dist.
end_shrink  = np.array([0.02, 0.04], dtype='float32')                                 # (y, x) at last dist
# Per-dist slope A decays by dist_decay each dist — shrink slows down as
# distance grows. Normalize so the cumulative shrink at the last dist matches
# end_shrink exactly:  A[j] = base·decay^j,  Σ_j A[j] = end_shrink.
dist_decay  = 0.7
_geom_sum   = (1.0 - dist_decay ** ndist) / (1.0 - dist_decay)                        # Σ decay^j, j=0..ndist-1
_A_base     = end_shrink / _geom_sum                                                  # slope for dist 0
A_gt        = (_A_base[None, :] * (dist_decay ** np.arange(ndist)[:, None])).astype('float32')  # (ndist, 2)
B_gt        = np.zeros_like(A_gt)
B_gt[1:]    = np.cumsum(A_gt[:-1], axis=0)                                            # (ndist, 2)
# tp_gt storage layout is (A, B) — both stored directly (no reparameterization).
tp_gt = np.zeros((ndist, 2, 2), dtype='float32')
tp_gt[:, 0, :] = A_gt
tp_gt[:, 1, :] = B_gt
# Evaluate the shrink profile once for plotting/logging.
_t_all      = (np.arange(ntheta, dtype='float32') / max(ntheta - 1, 1))              # (ntheta,)
shrink_nd   = (A_gt[None, :, :] * _t_all[:, None, None]
               + B_gt[None, :, :]).astype('float32')                                  # (ntheta, ndist, 2)

#### Create Writer
writer = Writer(
    path_out    = '/data2/vnikitin/tmp/test_results',
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
# Each rank owns a slice of obj (obj-axis) and pos (theta-axis). tp is global.
cl.vars['obj'][:] = obj[cl.st_obj:cl.end_obj]
cl.vars['prb'][:] = cp.array(prb)
cl.vars['pos'][:] = cp.array(pos[cl.st_theta:cl.end_theta])
cl.vars['tp'][:]  = cp.asarray(tp_gt)

cl.gen_sqrt_data(cl.vars, cl.data)
cl.cl_prb_term.gen_sqrt_ref(cl.vars['prb'], cl.ref)

#### Save synthetic sqrt-data on disk, one folder per distance
import os
_data_out_dir = '/data2/vnikitin/tmp/test_results/data'
if cl.rank == 0:
    for _k in range(ndist):
        os.makedirs(f'{_data_out_dir}/dist{_k}', exist_ok=True)
cl.comm.Barrier()
_data_np = cp.asnumpy(cl.data)      # (local_ntheta, ndist, nz, n)
for _k in range(ndist):
    for _jj in range(_data_np.shape[0]):
        _j_global = cl.st_theta + _jj
        write_tiff(_data_np[_jj, _k],
                   f'{_data_out_dir}/dist{_k}/proj_{_j_global:05d}')
if cl.rank == 0:
    print(f'wrote synthetic sqrt-data to {_data_out_dir}/dist{{0..{ndist-1}}}/')

#### Save ground-truth shrinkage plot in the SAME layout as writer._save_shrink_plot
# (2×ndist grid: row 0 = y, row 1 = x; solid=GT). Saved to CWD as shrink_gt.png.
if cl.rank == 0:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _fig_gt, _axes_gt = plt.subplots(2, ndist, figsize=(5 * ndist, 6))
    if ndist == 1:
        _axes_gt = _axes_gt[:, np.newaxis]
    _theta_idx = np.arange(ntheta)
    for _j in range(ndist):
        for _d, _lbl in enumerate(['y', 'x']):
            _ax = _axes_gt[_d, _j]
            _ax.plot(_theta_idx, shrink_nd[:, _j, _d], label='ground truth', color='C0')
            _ax.set_title(f"dist {_j}, {_lbl}")
            _ax.set_xlabel("theta index")
            _ax.set_ylabel("shrink")
            _ax.grid(True)
            _ax.legend(fontsize=8)
    _fig_gt.tight_layout()
    _sh_path = 'shrink_gt.png'                                     # current working directory
    _fig_gt.savefig(_sh_path, dpi=150)
    plt.close(_fig_gt)
    print(f'wrote {_sh_path}')

#### Reconstruction
# initial guess for obj: a heavily-blurred ground truth (rather than zero)
_sigma_blur = nobj / 16                                              # ~24 voxels
_obj_blur_re = ndimage.gaussian_filter(obj.real, sigma=_sigma_blur)
_obj_blur_im = ndimage.gaussian_filter(obj.imag, sigma=_sigma_blur)
_obj_blur    = (_obj_blur_re + 1j * _obj_blur_im).astype('complex64')
cl.vars['obj'][:] = _obj_blur[cl.st_obj:cl.end_obj]
del _obj_blur_re, _obj_blur_im, _obj_blur

cl.vars['prb'][:] = cp.array(1)
cl.vars['pos'][:] = cp.array((pos+pos_err)[cl.st_theta:cl.end_theta])

# tp init for the RECONSTRUCTION: perturbed A around GT; B built by continuity
# from the perturbed A so the initial shrink profile has no jumps between dists.
# During BH iterations B is free (no ongoing continuity constraint), but the
# *starting* profile is continuous.
_tp_init = np.zeros_like(tp_gt)
_mag  = 0.3 + np.random.random([ndist, 1]) * 0.2                              # (ndist, 1): |offset| ∈ [0.3, 0.5]
_sign = np.where(np.random.random([ndist, 1]) < 0.5, -1.0, 1.0)                # random ± per dist
err   = 1.0 + _sign * _mag                                                     # err ∈ [0.5, 0.7] ∪ [1.3, 1.5]
_A_init = A_gt * err                                                     # perturbed A (ndist, 2)
_tp_init[:, 0, :] = _A_init
# Continuous B derived from perturbed A: B[0]=0, B[j] = Σ_{i<j} A_init[i]
_tp_init[0, 1, :] = 0.0
if ndist > 1:
    _tp_init[1:, 1, :] = np.cumsum(_A_init[:-1], axis=0)
cl.vars['tp'][:] = cp.asarray(_tp_init)



cl.vars['tp'][:] = 0
cl.vars['tp'][:, 0, :] = 0.05      # A (slope)



cl.BH(writer=writer, shrink_gt=shrink_nd)

#### Post-reconstruction: horizontal + vertical mid-slices, fixed clim,
# cropped 40 px on each side. Gathered on rank 0.
_local_obj = cp.asnumpy(cl.vars['obj']).real                                     # (local_nzobj, nobj, nobj)
_all_obj   = cl.comm.gather(_local_obj, root=0)
if cl.rank == 0:
    _obj_full = np.concatenate(_all_obj, axis=0)                                 # (nzobj, nobj, nobj)
    _c    = 40
    _vmin = -13.0
    _vmax =   1.0
    _hslice = _obj_full[nobj // 2, _c:-_c, _c:-_c]                               # axial (Z-mid)
    _vslice = _obj_full[_c:-_c, nobj // 2, _c:-_c]                               # coronal (Y-mid)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (ax_h, ax_v) = plt.subplots(1, 2, figsize=(11, 5))
    fig.subplots_adjust(wspace=0.03)
    im_h = ax_h.imshow(_hslice, cmap='gray', vmin=_vmin, vmax=_vmax)
    ax_h.set_title(f'horizontal (Z = {nobj//2}), crop {_c}px')
    ax_h.axis('off')
    ax_v.imshow(_vslice, cmap='gray', vmin=_vmin, vmax=_vmax)
    ax_v.set_title(f'vertical (Y = {nobj//2}), crop {_c}px')
    ax_v.axis('off')
    fig.colorbar(im_h, ax=[ax_h, ax_v], fraction=0.03, shrink=0.9)
    fig.suptitle(f'test {TEST_ID}: rho[2]={_cfg["rho2"]:g}, '
                 f'pos_err_amp={_cfg["pos_err_amp"]:g}, '
                 f'use_shrink_recon={_cfg["use_shrink_recon"]}',
                 fontsize=10)
    _out_slices = f'recon_slices_test{TEST_ID}.png'
    fig.savefig(_out_slices, dpi=110, bbox_inches='tight')
    print(f'wrote {_out_slices}  (clim=[{_vmin:g}, {_vmax:g}])')

if MPI.COMM_WORLD.Get_rank() == 0:
    import os
    print(f"\nCheckpoints saved to: /data2/vnikitin/tmp/test_results/checkpoint_NNNN.h5")
