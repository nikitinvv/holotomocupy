import os
import h5py
import numpy as np
import scipy.ndimage as ndimage
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

os.makedirs('figs', exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
nx, ny, nz = 3216, 3216, 2048
path_out   = '/data2/vnikitin/brain_rec/20251115/Y350a'
s1         = [1507-(3264-2048)//2, 1351-(3264-3216)//2, 1738-(3264-3216)//2]

# ── Matplotlib style ──────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size":        24,
    "xtick.labelsize":  16,
    "ytick.labelsize":  16,
})

def mshow(a, **kwargs):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(a, cmap="gray", **kwargs)
    ax.axis('off')

def mshow1(a, **kwargs):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(a, cmap="gray", **kwargs)
    ax.axis('off')

# ── Physics helpers (same as plots2.py) ──────────────────────────────────────
r_e = 2.8179403262e-15  # classical electron radius [m]

def wavelength_from_keV(E_keV):
    return 1.239841984e-6 / (E_keV * 1e3)

def rho_e_from_delta_A3(delta, lam_m):
    """delta (internal units) → electron density [e/Å³]"""
    return (2 * np.pi / (r_e * lam_m**2)) * (-delta * 1e-9 * 1.2) * 1e-30

def find_min_max(data):
    h, e = np.histogram(data[:], 1000)
    st, end = np.where(h > np.max(h) * 0.01)[0][[0, -1]]
    return e[st], e[end + 1]

lam = wavelength_from_keV(17.1)

# ── Load mrec from checkpoint ─────────────────────────────────────────────────
iter      = 1280
ckpt_path = f'/data2/vnikitin/alcf/brain/Y350a_dist1234/rec_levels_17p23/checkpoint_{iter}.h5'
print("Loading mrec …")
with h5py.File(ckpt_path, 'r') as fid:
    ds   = fid['obj_re']
    mrec = ds[ds.shape[0]//2 - nz//2 : ds.shape[0]//2 + nz//2,
              ds.shape[1]//2 - ny//2 : ds.shape[1]//2 + ny//2,
              ds.shape[2]//2 - nx//2 : ds.shape[2]//2 + nx//2]

# ── Compute scale from prec/mrec centre slices (same as plots2.py) ────────────
prec_path = '/data2/vnikitin/brain/20251115/Y350a_HT_20nm_8dist_rec1234_/newFeb2026/Y350a_HT_20nm_8dist_rec_cm_.vol'
dtype     = np.dtype("<f4")
print("Loading prec centre slice for calibration …")
with open(prec_path, "rb") as f:
    f.seek(int(nz // 2) * ny * nx * dtype.itemsize)
    prec_slice = np.frombuffer(f.read(ny * nx * dtype.itemsize), dtype=dtype).reshape(ny, nx)

aa_cal = rho_e_from_delta_A3(prec_slice[ny//2-512:ny//2+512, ny//2-512:ny//2+512], lam)
bb_cal = rho_e_from_delta_A3(mrec[nz//2,  ny//2-512:ny//2+512, ny//2-512:ny//2+512], lam)
mmin1, mmax1 = find_min_max(aa_cal)
mmin2, mmax2 = find_min_max(bb_cal)
mmin1 *= 1.3;  mmax1 *= 1.07
scale = mmax1 / mmax2
print(f"scale = {scale:.4f}")

# crop to synchrotron region, apply physics scaling + scale factor
ss   = 128
mrec = mrec[s1[0]-ss:s1[0]+ss, s1[1]-ss:s1[1]+ss, s1[2]-ss:s1[2]+ss]
aa   = rho_e_from_delta_A3(mrec, lam) * scale

# ── Centred sub-crop with median filter ───────────────────────────────────────
ss2 = 100
ac  = aa[aa.shape[0]//2-ss2 : aa.shape[0]//2+ss2,
         aa.shape[1]//2-ss2 : aa.shape[1]//2+ss2,
         aa.shape[2]//2-ss2 : aa.shape[2]//2+ss2].copy()
aaa = ndimage.median_filter(ac, 3)

# ── Raw slices ────────────────────────────────────────────────────────────────
mshow1(aa[aa.shape[0]//2-2],      vmax=0.5, vmin=0.15); plt.savefig("figs/synz.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(aa[:, aa.shape[1]//2],     vmax=0.5, vmin=0.15); plt.savefig("figs/syny.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(aa[:, :, aa.shape[2]//2],  vmax=0.5, vmin=0.15); plt.savefig("figs/synx.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()

# ── Filtered slices ───────────────────────────────────────────────────────────
mshow1(aaa[ac.shape[0]//2-2],     vmax=0.5, vmin=0.15); plt.savefig("figs/fsynz.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(aaa[:, ac.shape[1]//2],    vmax=0.5, vmin=0.15); plt.savefig("figs/fsyny.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(aaa[:, :, ac.shape[2]//2], vmax=0.5, vmin=0.15); plt.savefig("figs/fsynx.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()

# ── Probe images ──────────────────────────────────────────────────────────────
mpl.rcParams.update({"font.size": 24, "xtick.labelsize": 20, "ytick.labelsize": 20})
iter_prb = 64 + 16
prb_base = '/data2//vnikitin/brain_rec/20251115/Y350a1234/4500_2048_0_0.0_0.003_0.05_0.02_20_1.1_0'

for dist in [0, 3]:
    prb_abs   = tifffile.imread(f'{prb_base}/rec_prb_abs{dist}/{iter_prb:04}.tiff')
    prb_angle = tifffile.imread(f'{prb_base}/rec_prb_angle{dist}/{iter_prb:04}.tiff')
    mshow(prb_abs * 5000, vmax=7000, vmin=1000); plt.savefig(f"figs/prbabs{dist}.png",   dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
    mshow(prb_angle,      vmax=2.4,  vmin=-2.4); plt.savefig(f"figs/prbangle{dist}.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()

# ── Position error ────────────────────────────────────────────────────────────
mpl.rcParams.update({"font.size": 28, "xtick.labelsize": 28, "ytick.labelsize": 28})
z1_ids      = np.array([0, 1, 2, 3])
str_z1_ids  = ''.join(map(str, z1_ids + 1))

rpos = np.load(f'{prb_base}/pos{iter_prb:04}.npy')
with h5py.File(f'{path_out}/data{str_z1_ids}.h5') as fid:
    pos = fid['/exchange/cshifts_final'][:].astype('float32')
rotation_center_shift = -8.780113000000028
rpos[:, :, 1] -= rotation_center_shift
err    = pos - rpos
labels = ["0", "1", "2", "3"]

for comp, ylabel, fname in [(1, 'horizontal error', 'errx0'), (0, 'vertical error', 'erry0')]:
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    l = ax.plot(err[::2, :, comp], ".")
    ax.set_xlabel('angle'); ax.set_ylabel(ylabel)
    leg = ax.legend(l, labels, title="distance", ncol=len(labels), loc='upper right',
                    columnspacing=1, handletextpad=0.1, handlelength=0.6,
                    fontsize=26, markerscale=4.0)
    leg._legend_box.align = "left"
    ax.grid()
    plt.savefig(f"figs/{fname}.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()

print("Done — figures saved to figs/")
