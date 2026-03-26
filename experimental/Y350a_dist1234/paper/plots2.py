import os
import h5py
import numpy as np
import scipy.ndimage as ndimage
from concurrent.futures import ThreadPoolExecutor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

os.makedirs('figs', exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
iter    = 1280-32
nx, ny, nz = 3216, 3216, 2048
N_THREADS  = 16

# ── Matplotlib style ──────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size":        22,
    "xtick.labelsize":  16,
    "ytick.labelsize":  16,
})

wb = LinearSegmentedColormap.from_list("white_black", ["white", "black"])

def mshow2(a, **kwargs):
    """No colorbar, no scalebar — returns ax for overlay functions."""
    _, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(a, cmap=wb, **kwargs)
    ax.axis('off')
    return ax

def _add_metrics(ax, data, fontsize=27):
    h, w = data.shape[:2]
    roi = np.clip(data[h//2-512:h//2+512, w//2-512:w//2+512].astype('float32'), 0.305, 0.535)
    sharpness = float(np.std(ndimage.laplace(roi)))
    flat = ndimage.median_filter(roi, size=3).ravel()
    thresh = np.median(flat)
    low, high = flat[flat < thresh], flat[flat >= thresh]
    pooled_std = np.sqrt((np.std(low)**2 + np.std(high)**2) / 2)
    hs = float(abs(np.mean(high) - np.mean(low)) / pooled_std) if pooled_std > 0 else float('nan')
    ax.text(0.96, 0.04, f"S={sharpness:.3f}\nHS={hs:.3f}", transform=ax.transAxes,
            va='bottom', ha='right', fontsize=fontsize, color='white',
            bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.5))

def _add_histogram(ax, data):
    h, w = data.shape[:2]
    flat = np.clip(data[h//2-512:h//2+512, w//2-512:w//2+512].astype('float32'), 0.305, 0.535).ravel()
    counts, edges = np.histogram(flat, bins=80, range=(0.305, 0.535), density=True)
    counts, edges = counts[1:-1], edges[1:-1]
    centers = (edges[:-1] + edges[1:]) / 2
    axh = ax.inset_axes([0.62, 0.77, 0.36, 0.21])
    axh.bar(centers, counts, width=edges[1]-edges[0], color='steelblue', linewidth=0)
    axh.grid(axis='both', linewidth=0.5, color='gray', alpha=0.6)
    axh.set_axisbelow(True)
    axh.set_xlim(0.305, 0.535)
    axh.set_ylim(0, counts.max() * 1.05)
    axh.set_xticks([]); axh.set_yticks([])
    for spine in axh.spines.values():
        spine.set_visible(False)
    axh.patch.set_facecolor('white')
    axh.patch.set_alpha(0.85)
    axh.patch.set_visible(True)

def mshow1(a, **kwargs):
    _, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(a, cmap=wb, **kwargs)
    ax.axis('off')

# ── Physics helpers ───────────────────────────────────────────────────────────
r_e = 2.8179403262e-15  # classical electron radius [m]

def wavelength_from_keV(E_keV):
    return 1.239841984e-6 / (E_keV * 1e3)

def prec_to_rho(arr):
    """prec .vol → rho_e [e/Å³]. 3-pt fit: lipid(-442→0.334), protein(-722→0.430), myelin(-942→0.503)."""
    return arr * (-3.382e-4) + 0.18485

def mrec_to_rho(arr):
    """mrec checkpoint → rho_e [e/Å³]. 3-pt fit: lipid(-2.667→0.334), protein(-4.7→0.430), myelin(-5.6→0.503)."""
    return arr * (-5.620e-2) + 0.17942

# ── Brain electron densities [e/Å³] ──────────────────────────────────────────
_brain_rho = {
    'water':        0.334,
    'lipid':        0.334,
    'gray matter':  0.370,
    'white matter': 0.390,
    'protein':      0.430,
    'myelin':       0.503,
}
print("Brain electron densities [e/Å³]:")
for tissue, rho in _brain_rho.items():
    print(f"  {tissue:<14} {rho:.3f}")

def find_min_max(data):
    h, e = np.histogram(data[:], 1000)
    st, end = np.where(h > np.max(h) * 0.01)[0][[0, -1]]
    return e[st], e[end + 1]

lam = wavelength_from_keV(17.1)

# ── Load prec from .vol ───────────────────────────────────────────────────────
print("Loading prec …")
prec_path = '/data2/vnikitin/brain/20251115/Y350a_HT_20nm_8dist_rec1234_/newFeb2026/Y350a_HT_20nm_8dist_rec_cm_.vol'
dtype = np.dtype("<f4")
with open(prec_path, "rb") as f:
    prec = np.frombuffer(f.read(nz * ny * nx * dtype.itemsize), dtype=dtype).reshape(nz, ny, nx)

# ── Load mrec from checkpoint (threaded memmap) ───────────────────────────────
print("Loading mrec …")
ckpt_path = f'/data2/vnikitin/alcf/brain/Y350a_dist1234/rec_levels_17p23/checkpoint_{iter}.h5'
with h5py.File(ckpt_path, 'r') as fid:
    ds    = fid['obj_re']
    shape = ds.shape
    dt    = ds.dtype
    byte_offset = ds.id.get_offset()
    z0 = shape[0]//2 - nz//2;  z1_ = shape[0]//2 + nz//2
    y0 = shape[1]//2 - ny//2;  y1  = shape[1]//2 + ny//2
    x0 = shape[2]//2 - nx//2;  x1  = shape[2]//2 + nx//2

z_edges = np.linspace(z0, z1_, N_THREADS + 1, dtype=int)
mrec    = np.empty((nz, y1 - y0, x1 - x0), dtype='float32')


if byte_offset is not None:
    mm = np.memmap(ckpt_path, dtype=dt, mode='r', offset=byte_offset, shape=shape)
    def _load(i):
        za, zb = z_edges[i], z_edges[i + 1]
        mrec[za - z0 : zb - z0] = mm[za:zb, y0:y1, x0:x1]
    with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
        list(ex.map(_load, range(N_THREADS)))
    del mm
else:
    def _load(i):
        za, zb = z_edges[i], z_edges[i + 1]
        with h5py.File(ckpt_path, 'r') as f:
            mrec[za - z0 : zb - z0] = f['obj_re'][za:zb, y0:y1, x0:x1]
    with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
        list(ex.map(_load, range(N_THREADS)))

# tifffile.imwrite('/data2/tmp/prec',prec[nz//2])

# ss

# ── Histograms ────────────────────────────────────────────────────────────────
aa_cal = prec_to_rho(prec[nz//2, ny//2-512:ny//2+512, ny//2-512:ny//2+512])
bb_cal = mrec_to_rho(mrec[nz//2, ny//2-512:ny//2+512, ny//2-512:ny//2+512])
plt.figure(figsize=(3, 1.5))
plt.hist(aa_cal.ravel(), bins=np.linspace(0.305, 0.535, 200))
plt.grid(); plt.tick_params(axis="y", labelleft=False)
plt.savefig("figs/histp.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()

plt.figure(figsize=(3, 1.5))
plt.hist(bb_cal.ravel(), bins=np.linspace(0.305, 0.535, 200))
plt.grid(); plt.tick_params(axis="y", labelleft=False)
plt.savefig("figs/histm.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()

plt.figure(figsize=(8, 3))
plt.hist(bb_cal.ravel(), bins=np.linspace(0.305, 0.535, 200))
plt.xticks(np.linspace(0.305, 0.535, 21), rotation=45)
plt.grid(); plt.tight_layout()
plt.savefig("figs/histm_peaks.png", dpi=200, bbox_inches="tight", pad_inches=0.05); plt.close()

# ── Full z-slice ──────────────────────────────────────────────────────────────
aa = prec_to_rho(prec[nz//2])

bb = mrec_to_rho(mrec[nz//2]) 
# import tifffile
# tifffile.imwrite('/data2/tmp/mrec',bb)
# exit()

ax = mshow2(aa, vmax=0.535, vmin=0.305); ax.axis('on'); _add_metrics(ax, aa, fontsize=20); _add_histogram(ax, aa); plt.savefig("figs/precz.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
ax = mshow2(bb, vmax=0.535, vmin=0.305); ax.axis('on');_add_metrics(ax, bb, fontsize=20); _add_histogram(ax, bb); plt.savefig("figs/mrecz.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(aa, vmax=0.535, vmin=0.305); plt.savefig("figs/precz_raw.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bb, vmax=0.535, vmin=0.305); plt.savefig("figs/mrecz_raw.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()

# ── Full y-slice ──────────────────────────────────────────────────────────────
aa = prec_to_rho(prec[:, ny//2])
bb = mrec_to_rho(mrec[:, ny//2])
ax = mshow2(aa, vmax=0.535, vmin=0.305); ax.axis('on'); _add_metrics(ax, aa, fontsize=20); plt.savefig("figs/precy.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
ax = mshow2(bb, vmax=0.535, vmin=0.305); ax.axis('on');_add_metrics(ax, bb, fontsize=20); plt.savefig("figs/mrecy.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(aa, vmax=0.535, vmin=0.305); plt.savefig("figs/precy_raw.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bb, vmax=0.535, vmin=0.305); plt.savefig("figs/mrecy_raw.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()

# ── Cropped z-slices ──────────────────────────────────────────────────────────
aa = prec_to_rho(prec[nz//2])
bb = mrec_to_rho(mrec[nz//2]) 
s, st = 320, 1200
mshow1(aa[ny//2-s:ny//2+3*s, nx//2-s+st:nx//2+s+st], vmax=0.535, vmin=0.305); plt.savefig("figs/preczz.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bb[ny//2-s:ny//2+3*s, nx//2-s+st:nx//2+s+st], vmax=0.535, vmin=0.305); plt.savefig("figs/mreczz.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
s = 160
mshow1(aa[ny//2-s:ny//2+s, nx//2-s:nx//2+s],         vmax=0.535, vmin=0.305); plt.savefig("figs/precz0.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bb[ny//2-s:ny//2+s, nx//2-s:nx//2+s],         vmax=0.535, vmin=0.305); plt.savefig("figs/mrecz0.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
stx, sty = 1000, 300
mshow1(aa[ny//2-s+sty:ny//2+s+sty, nx//2-s+stx:nx//2+s+stx], vmax=0.535, vmin=0.305); plt.savefig("figs/precz1.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bb[ny//2-s+sty:ny//2+s+sty, nx//2-s+stx:nx//2+s+stx], vmax=0.535, vmin=0.305); plt.savefig("figs/mrecz1.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()

# ── Cropped y-slices ──────────────────────────────────────────────────────────
aa = prec_to_rho(prec[:, ny//2])
bb = mrec_to_rho(mrec[:, ny//2]) 
s, st = 320, 1200
mshow1(aa[nz//2-2*s:nz//2+2*s, nx//2-s+st:nx//2+s+st], vmax=0.535, vmin=0.305); plt.savefig("figs/precyy.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bb[nz//2-2*s:nz//2+2*s, nx//2-s+st:nx//2+s+st], vmax=0.535, vmin=0.305); plt.savefig("figs/mrecyy.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
s = 160
mshow1(aa[nz//2-s:nz//2+s, nx//2-s:nx//2+s], vmax=0.535, vmin=0.305); plt.savefig("figs/precy0.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bb[nz//2-s:nz//2+s, nx//2-s:nx//2+s], vmax=0.535, vmin=0.305); plt.savefig("figs/mrecy0.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
stx, sty = 1000, 300
mshow1(aa[nz//2-s+sty:nz//2+s+sty, nx//2-s+stx:nx//2+s+stx], vmax=0.535, vmin=0.305); plt.savefig("figs/precy1.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bb[nz//2-s+sty:nz//2+s+sty, nx//2-s+stx:nx//2+s+stx], vmax=0.535, vmin=0.305); plt.savefig("figs/mrecy1.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()

# ── Zoomed patches ────────────────────────────────────────────────────────────
for tag, middle, xoff in [("0", [nz//2, ny//2+64,   nx//2-60],  4),
                           ("1", [nz//2, 2500+60,    2100+90],   0)]:
    ss = 128
    aa = prec_to_rho(prec[middle[0]-ss:middle[0]+ss, middle[1]-ss:middle[1]+ss, middle[2]-ss:middle[2]+ss])
    bb = mrec_to_rho(mrec[middle[0]-ss:middle[0]+ss, middle[1]-ss:middle[1]+ss, middle[2]-ss:middle[2]+ss])
    mshow1(aa[ss],        vmax=0.535, vmin=0.305); plt.savefig(f"figs/preczp{tag}.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
    mshow1(bb[ss],        vmax=0.535, vmin=0.305); plt.savefig(f"figs/mreczp{tag}.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
    mshow1(aa[:, ss],     vmax=0.535, vmin=0.305); plt.savefig(f"figs/precyp{tag}.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
    mshow1(bb[:, ss],     vmax=0.535, vmin=0.305); plt.savefig(f"figs/mrecyp{tag}.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
    mshow1(aa[:, :, ss],       vmax=0.535, vmin=0.305); plt.savefig(f"figs/precxp{tag}.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
    mshow1(bb[:, :, ss+xoff],  vmax=0.535, vmin=0.305); plt.savefig(f"figs/mrecxp{tag}.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
# ── Synchrotron region (same crop as plots_syn2.py) ──────────────────────────
s1 = [1507-(3264-2048)//2, 1351-(3264-3216)//2, 1738-(3264-3216)//2]
ss = 128
aa = prec_to_rho(prec[s1[0]-ss:s1[0]+ss, s1[1]-ss:s1[1]+ss, s1[2]-ss:s1[2]+ss])
bb = mrec_to_rho(mrec[s1[0]-ss:s1[0]+ss, s1[1]-ss:s1[1]+ss, s1[2]-ss:s1[2]+ss])

aaa = ndimage.median_filter(aa, 2)
bbb = ndimage.median_filter(bb, 2)

mshow1(aa[ss],      vmax=0.535, vmin=0.305); plt.savefig("figs/precsynz.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bb[ss],      vmax=0.535, vmin=0.305); plt.savefig("figs/mrecsynz.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(aa[:, ss],   vmax=0.535, vmin=0.305); plt.savefig("figs/precsyny.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bb[:, ss],   vmax=0.535, vmin=0.305); plt.savefig("figs/mrecsyny.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(aa[:,:,ss],  vmax=0.535, vmin=0.305); plt.savefig("figs/precsynx.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bb[:,:,ss],  vmax=0.535, vmin=0.305); plt.savefig("figs/mrecsynx.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()

mshow1(aaa[ss],     vmax=0.535, vmin=0.305); plt.savefig("figs/fprecsynz.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bbb[ss],     vmax=0.535, vmin=0.305); plt.savefig("figs/fmrecsynz.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(aaa[:, ss],  vmax=0.535, vmin=0.305); plt.savefig("figs/fprecsyny.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bbb[:, ss],  vmax=0.535, vmin=0.305); plt.savefig("figs/fmrecsyny.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(aaa[:,:,ss], vmax=0.535, vmin=0.305); plt.savefig("figs/fprecsynx.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bbb[:,:,ss], vmax=0.535, vmin=0.305); plt.savefig("figs/fmrecsynx.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()

print("Done — figures saved to figs/")
