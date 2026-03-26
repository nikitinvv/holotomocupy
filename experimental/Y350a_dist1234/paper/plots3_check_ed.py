import os
import json
import h5py
import numpy as np
import scipy.ndimage as ndimage
from concurrent.futures import ThreadPoolExecutor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib_scalebar.scalebar import ScaleBar
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

def mshow(a, **kwargs):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im   = ax.imshow(a, cmap=wb, **kwargs)
    cbar = fig.colorbar(im, fraction=0.046, pad=0.1,
                        orientation="horizontal", location="bottom")
    ticks = np.array(cbar.get_ticks())
    if ticks.size >= 2:
        cbar.set_ticks([ticks[0], ticks[-1]])
        cbar.set_ticklabels([f"{ticks[0]:g}", f"{ticks[-1]:g}"])
    cbar.set_label(r"Electron density (e/Å$^3$)", labelpad=-22, fontsize=17)
    cbar.ax.tick_params(labelsize=18)
    ax.add_artist(ScaleBar(20e-3, "um", length_fraction=0.25,
                           font_properties={"family": "serif", "size": 20},
                           location="lower right"))

def mshow1(a, **kwargs):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(a, cmap=wb, **kwargs)
    ax.axis('off')

# ── Physics helpers ───────────────────────────────────────────────────────────
r_e = 2.8179403262e-15  # classical electron radius [m]

def wavelength_from_keV(E_keV):
    return 1.239841984e-6 / (E_keV * 1e3)

def to_rho_physical(arr):
    """prec .vol → rho_e [e/Å³]. 3-pt fit: lipid(-442→0.334), protein(-722→0.430), myelin(-942→0.503)."""
    return arr * (-3.382e-4) + 0.18485

def to_rho_physical(arr):
    """mrec checkpoint (after mrec_scale) → rho_e [e/Å³]. 3-pt fit: lipid(-2.667→0.334), protein(-4.7→0.430), myelin(-5.6→0.503)."""
    #return arr * (-5.620e-2 / mrec_scale) + 0.17942
    return arr * (-3.382e-4) + 0.18485

def to_rho_physical(arr, E_keV=17.2):
    """Physical formula: arr ~ -k·δ [cm⁻¹]  →  ρ_e [e/Å³].
    δ = -arr·1e2 / k,   ρ_e = 2π·δ / (r_e·λ²)  converted m⁻³ → Å⁻³."""
    lam = wavelength_from_keV(E_keV)
    k   = 2 * np.pi / lam
    delta = -arr * 1e2 / k          # cm⁻¹ → m⁻¹ → dimensionless
    return 2 * np.pi * delta / (r_e * lam**2) * 1e-30   # m⁻³ → Å⁻³

# ── Physical slope (static, no data needed) ───────────────────────────────────
_lam_phys = wavelength_from_keV(17.2)
_slope_phys = 1e2 / (r_e * _lam_phys) * 1e-30   # |slope| for prec = -k·δ [cm⁻¹]
print(f"\nPhysical slope (17.2 keV, prec=-k·δ in cm⁻¹): {_slope_phys:.4e} e/Å³ per cm⁻¹")
print(f"Empirical slope:                                 {3.382e-4:.4e} e/Å³ per cm⁻¹")
print(f"Ratio phys/emp:                                  {_slope_phys/3.382e-4:.4f}")

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

def compute_metrics(img2d, vmin=0.3, vmax=0.5):
    d = np.clip(img2d.astype('float32'), vmin, vmax)
    sharpness = float(np.std(ndimage.laplace(d)))
    flat = d.ravel()
    thresh = np.median(flat)
    low, high = flat[flat < thresh], flat[flat > thresh]
    pooled_std = np.sqrt((np.std(low)**2 + np.std(high)**2) / 2)
    hs = float(abs(np.mean(high) - np.mean(low)) / pooled_std) if pooled_std > 0 else float('nan')
    return {'sharpness': sharpness, 'hs': hs}

metrics_dict = {}

def find_min_max(data):
    h, e = np.histogram(data[:], 1000)
    st, end = np.where(h > np.max(h) * 0.01)[0][[0, -1]]
    return e[st], e[end + 1]

lam = wavelength_from_keV(17.1)

# Physical scale: mrec (obj_re code units) → prec-equivalent units [cm⁻¹]
# Encodes voxelsize/wavelength and nz/nobj normalization from the Radon transform
mrec_scale = 20e-9 / wavelength_from_keV(17.2) * 2048 / 3264   # ≈ 174.09
print(f"mrec_scale = {mrec_scale:.6f}")

# ── Load prec from .vol ───────────────────────────────────────────────────────
print("Loading prec …")
prec_path = '/data2/vnikitin/brain/20251115/Y350a_HT_20nm_8dist_rec1234_/newFeb2026/Y350a_HT_20nm_8dist_rec_cm_.vol'
dtype = np.dtype("<f4")
with open(prec_path, "rb") as f:
    prec = np.frombuffer(f.read(nz * ny * nx * dtype.itemsize), dtype=dtype).reshape(nz, ny, nx)
prec_bg = float(prec[1000:1090, 300:450, 600:750].mean())
prec = prec - prec_bg
print(f"prec background (region [1000:1090,300:450,600:750]): {prec_bg:.4f}")

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
mrec_bg = float(mrec[1000:1090, 300:450, 600:750].mean())
mrec = mrec - mrec_bg
print(f"mrec background (region [1000:1090,300:450,600:750]): {mrec_bg:.4f}")

# ── Physical vs empirical comparison (background-subtracted) ──────────────────
_cal_pts_raw = [('lipid',   -442., 0.334),
                ('protein', -722., 0.430),
                ('myelin',  -942., 0.503)]
print(f"\nCalibration comparison after background subtraction (prec_bg={prec_bg:.2f}):")
print(f"{'Tissue':<10} {'Δprec':>8}  {'emp ρ_e':>10}  {'phys ρ_e':>10}  {'emp/phys':>8}")
for name, pval_raw, rho_emp in _cal_pts_raw:
    pval = pval_raw - prec_bg          # background-corrected prec value
    rho_phys = float(to_rho_physical(np.array([pval]))[0])
    print(f"{name:<10} {pval:>8.1f}  {rho_emp:>10.4f}  {rho_phys:>10.4f}  {rho_emp/rho_phys:>8.4f}")

# ── Middle 512³ average comparison ───────────────────────────────────────────
_s = 256  # half-size → 512 wide
_cz, _cy, _cx = nz//2, prec.shape[1]//2, prec.shape[2]//2
prec_mid = prec[_cz-_s:_cz+_s, _cy-_s:_cy+_s, _cx-_s:_cx+_s]
mrec_mid = mrec[_cz-_s:_cz+_s, _cy-_s:_cy+_s, _cx-_s:_cx+_s]
prec_mid_mean = float(prec_mid.mean())
mrec_mid_mean = float(mrec_mid.mean())
print(f"\nMiddle 512³ region averages (background-subtracted):")
print(f"  prec mean = {prec_mid_mean:.4f}  →  ρ_e = {to_rho_physical(np.array([prec_mid_mean]))[0]:.4f} e/Å³")
print(f"  mrec mean = {mrec_mid_mean:.4f}  →  ρ_e = {to_rho_physical(np.array([mrec_mid_mean]))[0]:.4f} e/Å³")
print(f"  ratio prec/mrec = {prec_mid_mean/mrec_mid_mean:.4f}")

# import tifffile
# tifffile.imwrite('/data2/tmp/mrec',mrec)
# tifffile.imwrite('/data2/tmp/prec',prec)
# sss

# ss

# ── Histograms ────────────────────────────────────────────────────────────────
aa_cal = to_rho_physical(prec[nz//2, ny//2-512:ny//2+512, ny//2-512:ny//2+512])
bb_cal = to_rho_physical(mrec[nz//2, ny//2-512:ny//2+512, ny//2-512:ny//2+512])
plt.figure(figsize=(3, 1.5))
plt.hist(aa_cal.ravel(), bins=np.linspace(0.3, 0.5, 200))
plt.grid(); plt.tick_params(axis="y", labelleft=False)
plt.savefig("figs/histp.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()

plt.figure(figsize=(3, 1.5))
plt.hist(bb_cal.ravel(), bins=np.linspace(0.3, 0.5, 200))
plt.grid(); plt.tick_params(axis="y", labelleft=False)
plt.savefig("figs/histm.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()

plt.figure(figsize=(8, 3))
plt.hist(bb_cal.ravel(), bins=np.linspace(0.3, 0.5, 200))
plt.xticks(np.linspace(0.3, 0.5, 21), rotation=45)
plt.grid(); plt.tight_layout()
plt.savefig("figs/histm_peaks.png", dpi=200, bbox_inches="tight", pad_inches=0.05); plt.close()

# ── Full z-slice ──────────────────────────────────────────────────────────────
aa = to_rho_physical(prec[nz//2])
bb = to_rho_physical(mrec[nz//2]) 
mshow(aa, vmax=0.5, vmin=0.3); plt.savefig("figs/precz.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow(bb, vmax=0.5, vmin=0.3); plt.savefig("figs/mrecz.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(aa, vmax=0.5, vmin=0.3); plt.savefig("figs/precz_raw.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bb, vmax=0.5, vmin=0.3); plt.savefig("figs/mrecz_raw.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
def save_metrics_with_hist(key, data2d, vmin=0.3, vmax=0.5, bins=80):
    d = np.clip(data2d[1024:2048, 1024:2048], vmin, vmax).ravel()
    counts, edges = np.histogram(d, bins=bins, range=(vmin, vmax), density=True)
    m = compute_metrics(data2d[1024:2048, 1024:2048])
    m['hist_counts'] = counts.tolist()
    m['hist_edges']  = edges.tolist()
    metrics_dict[key] = m

save_metrics_with_hist('precz', aa)
save_metrics_with_hist('mrecz', bb)

# ── Full y-slice ──────────────────────────────────────────────────────────────
aa = to_rho_physical(prec[:, ny//2])
bb = to_rho_physical(mrec[:, ny//2]) 
mshow(aa, vmax=0.5, vmin=0.3); plt.savefig("figs/precy.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow(bb, vmax=0.5, vmin=0.3); plt.savefig("figs/mrecy.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(aa, vmax=0.5, vmin=0.3); plt.savefig("figs/precy_raw.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bb, vmax=0.5, vmin=0.3); plt.savefig("figs/mrecy_raw.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
save_metrics_with_hist('precy', aa)
save_metrics_with_hist('mrecy', bb)

# ── Cropped z-slices ──────────────────────────────────────────────────────────
aa = to_rho_physical(prec[nz//2])
bb = to_rho_physical(mrec[nz//2]) 
s, st = 320, 1200
mshow1(aa[ny//2-s:ny//2+3*s, nx//2-s+st:nx//2+s+st], vmax=0.5, vmin=0.3); plt.savefig("figs/preczz.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bb[ny//2-s:ny//2+3*s, nx//2-s+st:nx//2+s+st], vmax=0.5, vmin=0.3); plt.savefig("figs/mreczz.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
s = 160
mshow1(aa[ny//2-s:ny//2+s, nx//2-s:nx//2+s],         vmax=0.5, vmin=0.3); plt.savefig("figs/precz0.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bb[ny//2-s:ny//2+s, nx//2-s:nx//2+s],         vmax=0.5, vmin=0.3); plt.savefig("figs/mrecz0.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
stx, sty = 1000, 300
mshow1(aa[ny//2-s+sty:ny//2+s+sty, nx//2-s+stx:nx//2+s+stx], vmax=0.5, vmin=0.3); plt.savefig("figs/precz1.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bb[ny//2-s+sty:ny//2+s+sty, nx//2-s+stx:nx//2+s+stx], vmax=0.5, vmin=0.3); plt.savefig("figs/mrecz1.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()

# ── Cropped y-slices ──────────────────────────────────────────────────────────
aa = to_rho_physical(prec[:, ny//2])
bb = to_rho_physical(mrec[:, ny//2]) 
s, st = 320, 1200
mshow1(aa[nz//2-2*s:nz//2+2*s, nx//2-s+st:nx//2+s+st], vmax=0.5, vmin=0.3); plt.savefig("figs/precyy.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bb[nz//2-2*s:nz//2+2*s, nx//2-s+st:nx//2+s+st], vmax=0.5, vmin=0.3); plt.savefig("figs/mrecyy.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
s = 160
mshow1(aa[nz//2-s:nz//2+s, nx//2-s:nx//2+s], vmax=0.5, vmin=0.3); plt.savefig("figs/precy0.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bb[nz//2-s:nz//2+s, nx//2-s:nx//2+s], vmax=0.5, vmin=0.3); plt.savefig("figs/mrecy0.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
stx, sty = 1000, 300
mshow1(aa[nz//2-s+sty:nz//2+s+sty, nx//2-s+stx:nx//2+s+stx], vmax=0.5, vmin=0.3); plt.savefig("figs/precy1.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bb[nz//2-s+sty:nz//2+s+sty, nx//2-s+stx:nx//2+s+stx], vmax=0.5, vmin=0.3); plt.savefig("figs/mrecy1.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()

# ── Zoomed patches ────────────────────────────────────────────────────────────
for tag, middle, xoff in [("0", [nz//2, ny//2+64,   nx//2-60],  4),
                           ("1", [nz//2, 2500+60,    2100+90],   0)]:
    ss = 128
    aa = to_rho_physical(prec[middle[0]-ss:middle[0]+ss, middle[1]-ss:middle[1]+ss, middle[2]-ss:middle[2]+ss])
    bb = to_rho_physical(mrec[middle[0]-ss:middle[0]+ss, middle[1]-ss:middle[1]+ss, middle[2]-ss:middle[2]+ss])
    mshow1(aa[ss],        vmax=0.5, vmin=0.3); plt.savefig(f"figs/preczp{tag}.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
    mshow1(bb[ss],        vmax=0.5, vmin=0.3); plt.savefig(f"figs/mreczp{tag}.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
    mshow1(aa[:, ss],     vmax=0.5, vmin=0.3); plt.savefig(f"figs/precyp{tag}.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
    mshow1(bb[:, ss],     vmax=0.5, vmin=0.3); plt.savefig(f"figs/mrecyp{tag}.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
    mshow1(aa[:, :, ss],       vmax=0.5, vmin=0.3); plt.savefig(f"figs/precxp{tag}.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
    mshow1(bb[:, :, ss+xoff],  vmax=0.5, vmin=0.3); plt.savefig(f"figs/mrecxp{tag}.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
    metrics_dict[f'preczp{tag}'] = compute_metrics(aa[ss])
    metrics_dict[f'mreczp{tag}'] = compute_metrics(bb[ss])
    metrics_dict[f'precyp{tag}'] = compute_metrics(aa[:, ss])
    metrics_dict[f'mrecyp{tag}'] = compute_metrics(bb[:, ss])
    metrics_dict[f'precxp{tag}'] = compute_metrics(aa[:, :, ss])
    metrics_dict[f'mrecxp{tag}'] = compute_metrics(bb[:, :, ss+xoff])

# ── Synchrotron region (same crop as plots_syn2.py) ──────────────────────────
s1 = [1507-(3264-2048)//2, 1351-(3264-3216)//2, 1738-(3264-3216)//2]
ss = 128
aa = to_rho_physical(prec[s1[0]-ss:s1[0]+ss, s1[1]-ss:s1[1]+ss, s1[2]-ss:s1[2]+ss])
bb = to_rho_physical(mrec[s1[0]-ss:s1[0]+ss, s1[1]-ss:s1[1]+ss, s1[2]-ss:s1[2]+ss])

aaa = ndimage.median_filter(aa, 2)
bbb = ndimage.median_filter(bb, 2)

mshow1(aa[ss],      vmax=0.5, vmin=0.3); plt.savefig("figs/precsynz.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bb[ss],      vmax=0.5, vmin=0.3); plt.savefig("figs/mrecsynz.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(aa[:, ss],   vmax=0.5, vmin=0.3); plt.savefig("figs/precsyny.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bb[:, ss],   vmax=0.5, vmin=0.3); plt.savefig("figs/mrecsyny.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(aa[:,:,ss],  vmax=0.5, vmin=0.3); plt.savefig("figs/precsynx.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bb[:,:,ss],  vmax=0.5, vmin=0.3); plt.savefig("figs/mrecsynx.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()

mshow1(aaa[ss],     vmax=0.5, vmin=0.3); plt.savefig("figs/fprecsynz.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bbb[ss],     vmax=0.5, vmin=0.3); plt.savefig("figs/fmrecsynz.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(aaa[:, ss],  vmax=0.5, vmin=0.3); plt.savefig("figs/fprecsyny.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bbb[:, ss],  vmax=0.5, vmin=0.3); plt.savefig("figs/fmrecsyny.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(aaa[:,:,ss], vmax=0.5, vmin=0.3); plt.savefig("figs/fprecsynx.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bbb[:,:,ss], vmax=0.5, vmin=0.3); plt.savefig("figs/fmrecsynx.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()

with open('figs/metrics.json', 'w') as f:
    json.dump(metrics_dict, f, indent=2)

print("Done — figures saved to figs/")
