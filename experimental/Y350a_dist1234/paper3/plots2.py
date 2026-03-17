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

def rho_e_from_delta_A3(delta, lam_m=None):
    """Raw checkpoint delta → electron density [e/Å³].
    Two-point calibration: lipid/water (-2.9 → 0.334), protein (-4.8 → 0.430).
    Verified: myelin (-6.25 → 0.503 e/Å³).
    """
    return -0.0505 * delta + 0.1875

def compute_metrics(img2d, vmin=0.15, vmax=0.5):
    d = img2d.astype('float32')
    sharpness = float(np.std(ndimage.laplace(d)))
    noise = np.std(d - ndimage.gaussian_filter(d, sigma=2))
    snr = float(20 * np.log10(np.mean(d) / noise)) if noise > 0 else float('inf')
    flat = d.ravel()
    thresh = np.median(flat)
    low, high = flat[flat < thresh], flat[flat > thresh]
    pooled_std = np.sqrt((np.std(low)**2 + np.std(high)**2) / 2)
    hs = float(abs(np.mean(high) - np.mean(low)) / pooled_std) if pooled_std > 0 else float('nan')
    return {'sharpness': sharpness, 'snr': snr, 'hs': hs}

metrics_dict = {}

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

# ── Compute scale factor to align mrec → prec ─────────────────────────────────
aa_cal = rho_e_from_delta_A3(prec[nz//2, ny//2-512:ny//2+512, ny//2-512:ny//2+512], lam)
bb_cal = rho_e_from_delta_A3(mrec[nz//2, ny//2-512:ny//2+512, ny//2-512:ny//2+512], lam)
mmin1, mmax1 = find_min_max(aa_cal)
mmin2, mmax2 = find_min_max(bb_cal)
print(f"prec range: {mmin1:.4f} – {mmax1:.4f}")
# mmin2*=0.9
mmin1 *= 1.3;  mmax1 *= 1.07
scale = mmax1 / mmax2
print(f"scale = {scale:.4f}")

# ── Histograms ────────────────────────────────────────────────────────────────
plt.figure(figsize=(3, 1.5))
plt.hist(aa_cal.ravel(), bins=np.linspace(0.15, 0.5, 200))
plt.grid(); plt.tick_params(axis="y", labelleft=False)
plt.savefig("figs/histp.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()

plt.figure(figsize=(3, 1.5))
plt.hist(bb_cal.ravel() * scale, bins=np.linspace(0.15, 0.5, 200))
plt.grid(); plt.tick_params(axis="y", labelleft=False)
plt.savefig("figs/histm.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()

# ── Full z-slice ──────────────────────────────────────────────────────────────
aa = rho_e_from_delta_A3(prec[nz//2], lam)
bb = rho_e_from_delta_A3(mrec[nz//2], lam) * scale
mshow(aa, vmax=0.5, vmin=0.15); plt.savefig("figs/precz.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow(bb, vmax=0.5, vmin=0.15); plt.savefig("figs/mrecz.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(aa, vmax=0.5, vmin=0.15); plt.savefig("figs/precz_raw.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bb, vmax=0.5, vmin=0.15); plt.savefig("figs/mrecz_raw.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
def save_metrics_with_hist(key, data2d, vmin=0.15, vmax=0.5, bins=80):
    d = np.clip(data2d[1024:2048, 1024:2048], vmin, vmax).ravel()
    counts, edges = np.histogram(d, bins=bins, range=(vmin, vmax), density=True)
    m = compute_metrics(data2d[1024:2048, 1024:2048])
    m['hist_counts'] = counts.tolist()
    m['hist_edges']  = edges.tolist()
    metrics_dict[key] = m

save_metrics_with_hist('precz', aa)
save_metrics_with_hist('mrecz', bb)

# ── Full y-slice ──────────────────────────────────────────────────────────────
aa = rho_e_from_delta_A3(prec[:, ny//2], lam)
bb = rho_e_from_delta_A3(mrec[:, ny//2], lam) * scale
mshow(aa, vmax=0.5, vmin=0.15); plt.savefig("figs/precy.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow(bb, vmax=0.5, vmin=0.15); plt.savefig("figs/mrecy.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(aa, vmax=0.5, vmin=0.15); plt.savefig("figs/precy_raw.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bb, vmax=0.5, vmin=0.15); plt.savefig("figs/mrecy_raw.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
save_metrics_with_hist('precy', aa)
save_metrics_with_hist('mrecy', bb)

# ── Cropped z-slices ──────────────────────────────────────────────────────────
aa = rho_e_from_delta_A3(prec[nz//2], lam)
bb = rho_e_from_delta_A3(mrec[nz//2], lam) * scale
s, st = 320, 1200
mshow1(aa[ny//2-s:ny//2+3*s, nx//2-s+st:nx//2+s+st], vmax=0.5, vmin=0.15); plt.savefig("figs/preczz.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bb[ny//2-s:ny//2+3*s, nx//2-s+st:nx//2+s+st], vmax=0.5, vmin=0.15); plt.savefig("figs/mreczz.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
s = 160
mshow1(aa[ny//2-s:ny//2+s, nx//2-s:nx//2+s],         vmax=0.5, vmin=0.15); plt.savefig("figs/precz0.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bb[ny//2-s:ny//2+s, nx//2-s:nx//2+s],         vmax=0.5, vmin=0.15); plt.savefig("figs/mrecz0.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
stx, sty = 1000, 300
mshow1(aa[ny//2-s+sty:ny//2+s+sty, nx//2-s+stx:nx//2+s+stx], vmax=0.5, vmin=0.15); plt.savefig("figs/precz1.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bb[ny//2-s+sty:ny//2+s+sty, nx//2-s+stx:nx//2+s+stx], vmax=0.5, vmin=0.15); plt.savefig("figs/mrecz1.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()

# ── Cropped y-slices ──────────────────────────────────────────────────────────
aa = rho_e_from_delta_A3(prec[:, ny//2], lam)
bb = rho_e_from_delta_A3(mrec[:, ny//2], lam) * scale
s, st = 320, 1200
mshow1(aa[nz//2-2*s:nz//2+2*s, nx//2-s+st:nx//2+s+st], vmax=0.5, vmin=0.15); plt.savefig("figs/precyy.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bb[nz//2-2*s:nz//2+2*s, nx//2-s+st:nx//2+s+st], vmax=0.5, vmin=0.15); plt.savefig("figs/mrecyy.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
s = 160
mshow1(aa[nz//2-s:nz//2+s, nx//2-s:nx//2+s], vmax=0.5, vmin=0.15); plt.savefig("figs/precy0.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bb[nz//2-s:nz//2+s, nx//2-s:nx//2+s], vmax=0.5, vmin=0.15); plt.savefig("figs/mrecy0.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
stx, sty = 1000, 300
mshow1(aa[nz//2-s+sty:nz//2+s+sty, nx//2-s+stx:nx//2+s+stx], vmax=0.5, vmin=0.15); plt.savefig("figs/precy1.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bb[nz//2-s+sty:nz//2+s+sty, nx//2-s+stx:nx//2+s+stx], vmax=0.5, vmin=0.15); plt.savefig("figs/mrecy1.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()

# ── Zoomed patches ────────────────────────────────────────────────────────────
for tag, middle, xoff in [("0", [nz//2, ny//2+64,   nx//2-60],  4),
                           ("1", [nz//2, 2500+60,    2100+90],   0)]:
    ss = 128
    aa = rho_e_from_delta_A3(prec[middle[0]-ss:middle[0]+ss, middle[1]-ss:middle[1]+ss, middle[2]-ss:middle[2]+ss], lam)
    bb = rho_e_from_delta_A3(mrec[middle[0]-ss:middle[0]+ss, middle[1]-ss:middle[1]+ss, middle[2]-ss:middle[2]+ss], lam) * scale
    mshow1(aa[ss],        vmax=0.5, vmin=0.15); plt.savefig(f"figs/preczp{tag}.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
    mshow1(bb[ss],        vmax=0.5, vmin=0.15); plt.savefig(f"figs/mreczp{tag}.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
    mshow1(aa[:, ss],     vmax=0.5, vmin=0.15); plt.savefig(f"figs/precyp{tag}.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
    mshow1(bb[:, ss],     vmax=0.5, vmin=0.15); plt.savefig(f"figs/mrecyp{tag}.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
    mshow1(aa[:, :, ss],       vmax=0.5, vmin=0.15); plt.savefig(f"figs/precxp{tag}.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
    mshow1(bb[:, :, ss+xoff],  vmax=0.5, vmin=0.15); plt.savefig(f"figs/mrecxp{tag}.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
    metrics_dict[f'preczp{tag}'] = compute_metrics(aa[ss])
    metrics_dict[f'mreczp{tag}'] = compute_metrics(bb[ss])
    metrics_dict[f'precyp{tag}'] = compute_metrics(aa[:, ss])
    metrics_dict[f'mrecyp{tag}'] = compute_metrics(bb[:, ss])
    metrics_dict[f'precxp{tag}'] = compute_metrics(aa[:, :, ss])
    metrics_dict[f'mrecxp{tag}'] = compute_metrics(bb[:, :, ss+xoff])

# ── Synchrotron region (same crop as plots_syn2.py) ──────────────────────────
s1 = [1507-(3264-2048)//2, 1351-(3264-3216)//2, 1738-(3264-3216)//2]
ss = 128
aa = rho_e_from_delta_A3(prec[s1[0]-ss:s1[0]+ss, s1[1]-ss:s1[1]+ss, s1[2]-ss:s1[2]+ss], lam)
bb = rho_e_from_delta_A3(mrec[s1[0]-ss:s1[0]+ss, s1[1]-ss:s1[1]+ss, s1[2]-ss:s1[2]+ss], lam) * scale

aaa = ndimage.median_filter(aa, 2)
bbb = ndimage.median_filter(bb, 2)

mshow1(aa[ss],      vmax=0.5, vmin=0.15); plt.savefig("figs/precsynz.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bb[ss],      vmax=0.5, vmin=0.15); plt.savefig("figs/mrecsynz.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(aa[:, ss],   vmax=0.5, vmin=0.15); plt.savefig("figs/precsyny.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bb[:, ss],   vmax=0.5, vmin=0.15); plt.savefig("figs/mrecsyny.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(aa[:,:,ss],  vmax=0.5, vmin=0.15); plt.savefig("figs/precsynx.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bb[:,:,ss],  vmax=0.5, vmin=0.15); plt.savefig("figs/mrecsynx.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()

mshow1(aaa[ss],     vmax=0.5, vmin=0.15); plt.savefig("figs/fprecsynz.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bbb[ss],     vmax=0.5, vmin=0.15); plt.savefig("figs/fmrecsynz.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(aaa[:, ss],  vmax=0.5, vmin=0.15); plt.savefig("figs/fprecsyny.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bbb[:, ss],  vmax=0.5, vmin=0.15); plt.savefig("figs/fmrecsyny.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(aaa[:,:,ss], vmax=0.5, vmin=0.15); plt.savefig("figs/fprecsynx.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bbb[:,:,ss], vmax=0.5, vmin=0.15); plt.savefig("figs/fmrecsynx.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()

with open('figs/metrics.json', 'w') as f:
    json.dump(metrics_dict, f, indent=2)

print("Done — figures saved to figs/")
