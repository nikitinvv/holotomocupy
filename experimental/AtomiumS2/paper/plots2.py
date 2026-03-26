import os
import h5py
import glob
import tifffile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib as mpl
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import map_coordinates, median_filter, laplace

os.makedirs('figs', exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
crop = 128+16
nx, ny, nz = 2048-crop, 2048-crop, 2048-crop
N_THREADS   = 16

# slice selections (kept from original plots2.ipynb)
z_slice     = 140+80  # z-slice at nz//2 - z_slice
y_slice_off = 32   # y-slice at ny//2 + y_slice_off

prec_path = '/data2/vnikitin/atomium_rec/20240924/AtomiumS2/rec_peter65/'
ckpt_path = '/data2/vnikitin/alcf/atomium/checkpoint_1344.h5'

# ── Matplotlib style ──────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size":        28,
    "xtick.labelsize":  22,
    "ytick.labelsize":  22,
})

def mshow(a, **kwargs):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im   = ax.imshow(a, cmap="gray", **kwargs)
    cbar = fig.colorbar(im, fraction=0.07, pad=0.05)
    cbar.ax.tick_params(labelsize=20)
    ax.add_artist(ScaleBar(7e-3, "um", length_fraction=0.25,
                           font_properties={"family": "serif", "size": 20},
                           location="lower right"))

def mshow1(a, **kwargs):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(a, cmap="gray", **kwargs)
    ax.axis('off')

def mshow2(a, **kwargs):
    """No colorbar, no scalebar — keeps x/y tick labels."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(a, cmap="gray", **kwargs)
    ax.tick_params(labelsize=17)
    return ax

def find_min_max(data):
    h, e = np.histogram(data[:], 1000)
    st, end = np.where(h > np.max(h) * 0.03)[0][[0, -1]]
    return e[st], e[end + 1]

def norm(a, mn, mx):
    return np.clip((a - mn) / (mx - mn), 0, 1)

def extract_profile_3d(vol, p0, p1, num=300):
    """Sample intensities along a line in a 3D volume.
    vol: 3D array; p0, p1: (z, y, x) endpoints in array coordinates."""
    coords = [np.linspace(p0[k], p1[k], num) for k in range(3)]
    vals = map_coordinates(vol, coords, order=1)
    dist = np.linspace(0, np.sqrt(sum((p1[k]-p0[k])**2 for k in range(3))), num)
    return dist, vals

def mshow1_line(a, p0, p1, color='r', **kwargs):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(a, cmap="gray", **kwargs)
    ax.plot([p0[1], p1[1]], [p0[0], p1[0]], color=color, lw=4.5)
    ax.axis('off')
    return ax

def _add_metrics(ax, data, fontsize=27):
    gray = data.astype('float32')
    sharpness = float(np.std(laplace(gray)))
    flat = median_filter(gray, size=3).ravel()
    # Otsu threshold to separate background from sample
    counts, edges = np.histogram(flat, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(counts); w1 = w0[-1] - w0
    s0 = np.cumsum(counts * centers)
    mu0 = np.where(w0 > 0, s0 / w0, 0)
    mu1 = np.where(w1 > 0, (s0[-1] - s0) / np.where(w1 > 0, w1, 1), 0)
    bg_thresh = centers[np.argmax(w0 * w1 * (mu0 - mu1) ** 2)]
    sample = flat[flat < bg_thresh]
    # HS: median split within sample, pooled std
    thresh = np.median(sample)
    low, high = sample[sample < thresh], sample[sample >= thresh]
    pooled_std = np.sqrt((np.std(low)**2 + np.std(high)**2) / 2)
    hs = float(abs(np.mean(high) - np.mean(low)) / pooled_std) if pooled_std > 0 else float('nan')
    ax.text(0.96, 0.04, f"S={sharpness:.3f}\nHS={hs:.3f}", transform=ax.transAxes,
            va='bottom', ha='right', fontsize=fontsize, color='white',
            bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.5))

def _add_histogram(ax, data):
    gray = data.astype('float32')
    flat = gray.ravel()
    counts_o, edges_o = np.histogram(flat, bins=256)
    centers_o = (edges_o[:-1] + edges_o[1:]) / 2
    w0 = np.cumsum(counts_o); w1 = w0[-1] - w0
    s0 = np.cumsum(counts_o * centers_o)
    mu0 = np.where(w0 > 0, s0 / w0, 0)
    mu1 = np.where(w1 > 0, (s0[-1] - s0) / np.where(w1 > 0, w1, 1), 0)
    bg_thresh = centers_o[np.argmax(w0 * w1 * (mu0 - mu1) ** 2)]
    sample = flat[flat < bg_thresh]
    counts, edges = np.histogram(1 - sample, bins=80, density=True)
    counts, edges = counts[1:-1], edges[1:-1]
    centers = (edges[:-1] + edges[1:]) / 2
    axh = ax.inset_axes([0.62, 0.77, 0.36, 0.21])
    axh.bar(centers, counts, width=edges[1]-edges[0], color='steelblue', linewidth=0)
    axh.grid(axis='both', linewidth=0.5, color='gray', alpha=0.6)
    axh.set_axisbelow(True)
    axh.set_xlim(edges[0], edges[-1])
    axh.set_ylim(0, counts.max() * 1.05)
    axh.set_xticks([]); axh.set_yticks([])
    for spine in axh.spines.values():
        spine.set_visible(False)
    axh.patch.set_facecolor('white')
    axh.patch.set_alpha(0.85)
    axh.patch.set_visible(True)

def save_profile(va, vb, p0, p1, fname, median_size=3):
    """va, vb: 3D normalized subvolumes; p0, p1: (z,y,x) endpoints."""
    va = median_filter(va, size=median_size)
    vb = median_filter(vb, size=median_size)
    dist, pa = extract_profile_3d(va, p0, p1)
    _,    pb = extract_profile_3d(vb, p0, p1)
    dist_nm = dist * 7  # 7 nm per pixel
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(dist_nm, 1 - pa, label='conventional', color='limegreen', lw=3)
    ax.plot(dist_nm, 1 - pb, label='proposed',     color='red',       lw=3)
    ax.set_xlabel('nm', fontsize=32)
    ax.set_xlim(0, dist_nm[-1])
    ax.yaxis.set_ticklabels([])
    ax.yaxis.set_ticks_position('none')

    ax.xaxis.set_major_locator(plt.MultipleLocator(80))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(20))
    ax.tick_params(axis='x', which='major', labelsize=30)
    ax.grid(True, which='major', ls='--', alpha=0.4)
    ax.grid(True, which='minor', ls=':', alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()

# ── Load prec from tiff stack (parallel, crop-on-read) ────────────────────────
print("Loading prec ...")
prec_shift = -4  # shift prec region by 2 pixels in x,y,z to align with mrec
_files = sorted(glob.glob(f'{prec_path}/r_*.tiff'))
_n_all = len(_files)
_tmp   = tifffile.imread(_files[0])
_fy, _fx = _tmp.shape
_y0, _y1 = _fy//2 - ny//2 + prec_shift, _fy//2 + ny//2 + prec_shift
_x0, _x1 = _fx//2 - nx//2 + prec_shift, _fx//2 + nx//2 + prec_shift
_z0, _z1 = crop + prec_shift, nz + crop + prec_shift
print(_z0,_z1)
_sel = _files[_z0:_z1]
prec = np.empty((nz, ny, nx), dtype='float32')
def _read_slice(i):
    prec[i] = tifffile.imread(_sel[i])[_y0:_y1, _x0:_x1]
with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
    list(ex.map(_read_slice, range(nz)))
print(f"prec shape: {prec.shape}")

# ── Load mrec from checkpoint (threaded memmap) ───────────────────────────────
print("Loading mrec ...")
with h5py.File(ckpt_path, 'r') as fid:
    ds          = fid['obj_re']
    shape       = ds.shape
    dt          = ds.dtype
    byte_offset = ds.id.get_offset()
    z0 = shape[0]//2 - 1024 - 400+crop;  z1_ = z0+nz
    print(z0,z1_)
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
print(f"mrec shape: {mrec.shape}")

# ── Normalization constants ───────────────────────────────────────────────────
cal_z = nz//2 - z_slice
cal   = slice(ny//2 - 512, ny//2 + 512)
mmin1, mmax1 = find_min_max(prec[cal_z, cal, cal])
mmin2, mmax2 = find_min_max(mrec[cal_z, cal, cal])
print(f"prec  [{mmin1:.4f}, {mmax1:.4f}]")
print(f"mrec  [{mmin2:.4f}, {mmax2:.4f}]")

# ── Threshold debug images ────────────────────────────────────────────────────
def _otsu_thresh(img):
    flat = img.ravel()
    counts, edges = np.histogram(flat, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(counts); w1 = w0[-1] - w0
    s0 = np.cumsum(counts * centers)
    mu0 = np.where(w0 > 0, s0 / w0, 0)
    mu1 = np.where(w1 > 0, (s0[-1] - s0) / np.where(w1 > 0, w1, 1), 0)
    return centers[np.argmax(w0 * w1 * (mu0 - mu1) ** 2)]

_aa = norm(prec[nz//2 - z_slice], mmin1, mmax1)
_bb = norm(mrec[nz//2 - z_slice], mmin2, mmax2)
_t1 = _otsu_thresh(_aa); _t2 = _otsu_thresh(_bb)
print(f"Otsu thresh  prec={_t1:.4f}  mrec={_t2:.4f}")
mshow1(_aa < _t1, vmin=0, vmax=1); plt.savefig("figs/thresh_prec.png", dpi=150, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(_bb < _t2, vmin=0, vmax=1); plt.savefig("figs/thresh_mrec.png", dpi=150, bbox_inches="tight", pad_inches=0.02); plt.close()

# ── Full z-slice ──────────────────────────────────────────────────────────────
aa = norm(prec[nz//2 - z_slice], mmin1, mmax1)
bb = norm(mrec[nz//2 - z_slice], mmin2, mmax2)
ax = mshow2(aa, vmax=0.95, vmin=-0.05); _add_metrics(ax, aa, fontsize=24); _add_histogram(ax, aa); plt.savefig("figs/precz.png",     dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
ax = mshow2(bb, vmax=0.95, vmin=-0.05); _add_metrics(ax, bb, fontsize=24); _add_histogram(ax, bb); plt.savefig("figs/mrecz.png",     dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(aa, vmax=0.95, vmin=-0.05); plt.savefig("figs/precz_raw.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bb, vmax=0.95, vmin=-0.05); plt.savefig("figs/mrecz_raw.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()

# ── Full y-slice ──────────────────────────────────────────────────────────────
aa = norm(prec[:, ny//2 + y_slice_off], mmin1, mmax1)
bb = norm(mrec[:, ny//2 + y_slice_off], mmin2, mmax2)
ax = mshow2(aa, vmax=0.95, vmin=-0.05); _add_metrics(ax, aa); plt.savefig("figs/precy.png",     dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
ax = mshow2(bb, vmax=0.95, vmin=-0.05); _add_metrics(ax, bb); plt.savefig("figs/mrecy.png",     dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(aa, vmax=0.95, vmin=-0.05); plt.savefig("figs/precy_raw.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
mshow1(bb, vmax=0.95, vmin=-0.05); plt.savefig("figs/mrecy_raw.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()


# ── Cropped z-slices ──────────────────────────────────────────────────────────
iz = nz//2 - z_slice
aa_full = norm(prec[iz], mmin1, mmax1)
bb_full = norm(mrec[iz], mmin2, mmax2)
# thin slab (iz-1 : iz+2) for 3-D interpolation; middle slice index = 1
aa_slab = norm(prec[iz-1:iz+2], mmin1, mmax1)
bb_slab = norm(mrec[iz-1:iz+2], mmin2, mmax2)

s, stx, sty = 200, -70, 10
p0, p1 = [282, 282], [236, 236], 
y0c, y1c = ny//2-s+sty, ny//2+s+sty
x0c, x1c = nx//2-s+stx, nx//2+s+stx
aa_c = aa_full[y0c:y1c, x0c:x1c]
bb_c = bb_full[y0c:y1c, x0c:x1c]
ax = mshow1_line(aa_c, p0, p1, color='limegreen', vmax=0.95, vmin=-0.05); _add_metrics(ax, aa_c); plt.savefig("figs/preczz.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
ax = mshow1_line(bb_c, p0, p1, color='red', vmax=0.95, vmin=-0.05); _add_metrics(ax, bb_c); plt.savefig("figs/mreczz.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
save_profile(aa_slab[:, y0c:y1c, x0c:x1c], bb_slab[:, y0c:y1c, x0c:x1c],
             [1, p0[0], p0[1]], [1, p1[0], p1[1]], "figs/profile_zz.png")

s, stx, sty = 200, 0, -400
p0, p1 = [204, 284], [153, 233] 
y0c, y1c = ny//2-s+sty, ny//2+s+sty
x0c, x1c = nx//2-s+stx, nx//2+s+stx
aa_c = aa_full[y0c:y1c, x0c:x1c]
bb_c = bb_full[y0c:y1c, x0c:x1c]
ax = mshow1_line(aa_c, p0, p1, color='limegreen', vmax=0.95, vmin=-0.05); _add_metrics(ax, aa_c); plt.savefig("figs/preczz2.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
ax = mshow1_line(bb_c, p0, p1, color='red', vmax=0.95, vmin=-0.05); _add_metrics(ax, bb_c); plt.savefig("figs/mreczz2.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
save_profile(aa_slab[:, y0c:y1c, x0c:x1c], bb_slab[:, y0c:y1c, x0c:x1c],
             [1, p0[0], p0[1]], [1, p1[0], p1[1]], "figs/profile_zz2.png")

# ── Cropped y-slices ──────────────────────────────────────────────────────────
iy = ny//2 + y_slice_off
aa_full = norm(prec[:, iy], mmin1, mmax1)
bb_full = norm(mrec[:, iy], mmin2, mmax2)
# thin slab (iy-1 : iy+2) for 3-D interpolation; middle y index = 1
aa_slab = norm(prec[:, iy-1:iy+2], mmin1, mmax1)
bb_slab = norm(mrec[:, iy-1:iy+2], mmin2, mmax2)

s, stx, sty = 200, -90, -200
p0, p1 = [90, 113], [90, 180]
z0c, z1c = nz//2-s+sty, nz//2+s+sty
x0c, x1c = nx//2-s+stx, nx//2+s+stx
aa_c = aa_full[z0c:z1c, x0c+8:x1c+8]
bb_c = bb_full[z0c:z1c, x0c:x1c]
ax = mshow1_line(aa_c, p0, p1, color='limegreen', vmax=0.95, vmin=-0.05); _add_metrics(ax, aa_c); plt.savefig("figs/precyy.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
ax = mshow1_line(bb_c, p0, p1, color='red', vmax=0.95, vmin=-0.05); _add_metrics(ax, bb_c); plt.savefig("figs/mrecyy.png",  dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
save_profile(aa_slab[z0c:z1c, :, x0c+8:x1c+8], bb_slab[z0c:z1c, :, x0c:x1c],
             [p0[0], 1, p0[1]], [p1[0], 1, p1[1]], "figs/profile_yy.png")


s, stx, sty = 200, -220, 20
p0, p1 = [46, 171], [100, 200]
z0c, z1c = nz//2-s+sty, nz//2+s+sty
x0c, x1c = nx//2-s+stx, nx//2+s+stx
aa_c = aa_full[z0c+4:z1c+4, x0c+5:x1c+5]
bb_c = bb_full[z0c:z1c, x0c:x1c]
ax = mshow1_line(aa_c, p0, p1, color='limegreen', vmax=0.95, vmin=-0.05); _add_metrics(ax, aa_c); plt.savefig("figs/precyy3.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
ax = mshow1_line(bb_c, p0, p1, color='red', vmax=0.95, vmin=-0.05); _add_metrics(ax, bb_c); plt.savefig("figs/mrecyy3.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
save_profile(aa_slab[z0c+4:z1c+4, :, x0c+5:x1c+5], bb_slab[z0c:z1c, :, x0c:x1c],
             [p0[0], 1, p0[1]], [p1[0], 1, p1[1]], "figs/profile_yy3.png")

# # ── Zoomed patches ────────────────────────────────────────────────────────────
# for tag, middle in [("0", [nz//2, ny//2+64,  nx//2-60])]:
#                     # ("1", [nz//2, 2500+60,   2100+90])]:
#     ss = 128
#     ap = prec[middle[0]-ss:middle[0]+ss, middle[1]-ss:middle[1]+ss, middle[2]-ss:middle[2]+ss]
#     bp = mrec[middle[0]-ss:middle[0]+ss, middle[1]-ss:middle[1]+ss, middle[2]-ss:middle[2]+ss]
#     aa = norm(ap, mmin1, mmax1)
#     bb = norm(bp, mmin2, mmax2)

#     mshow1(aa[ss],        vmax=0.8, vmin=0); plt.savefig(f"figs/preczp{tag}.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
#     mshow1(bb[ss],        vmax=1,   vmin=0); plt.savefig(f"figs/mreczp{tag}.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
#     mshow1(aa[:, ss],     vmax=0.8, vmin=0); plt.savefig(f"figs/precyp{tag}.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
#     mshow1(bb[:, ss],     vmax=1,   vmin=0); plt.savefig(f"figs/mrecyp{tag}.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
#     mshow1(aa[:, :, ss],  vmax=0.8, vmin=0); plt.savefig(f"figs/precxp{tag}.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()
#     mshow1(bb[:, :, ss],  vmax=1,   vmin=0); plt.savefig(f"figs/mrecxp{tag}.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()

# ── Electron density ──────────────────────────────────────────────────────────
# Two-point calibration: ZnO (δ_rec=-2.4) → 1.576 e⁻/Å³, Al₂O₃ (δ_rec=-1.26) → 1.177 e⁻/Å³
_slope = (1.576 - 1.177) / (-2.4 - (-1.26))   # -0.350 e⁻/Å³ per raw unit
_inter = 1.576 - _slope * (-2.4)               #  0.736 e⁻/Å³

def to_rho_e(raw):
    return _slope * raw + _inter

iz = nz//2 - z_slice
rho_prec = to_rho_e(prec[iz])
rho_mrec = to_rho_e(mrec[iz])

cal = slice(ny//2 - 512, ny//2 + 512)
vmin_rho, vmax_rho = find_min_max(rho_mrec[cal, cal])
print(f"rho_e range  [{vmin_rho:.3f}, {vmax_rho:.3f}] e⁻/Å³")

for tag, img in [('prec', rho_prec), ('mrec', rho_mrec)]:
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img, cmap='gray_r', vmin=vmin_rho, vmax=vmax_rho)
    ax.axis('off')
    ax.add_artist(ScaleBar(7e-3, "um", length_fraction=0.25,
                           font_properties={"family": "serif", "size": 20},
                           location="lower right"))
    plt.savefig(f"figs/{tag}z_rho.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()

# Standalone colorbar (higher ρ_e = black = gray_r top)
fig, ax = plt.subplots(figsize=(8, 1.1))
fig.subplots_adjust(bottom=0.35, top=0.75, left=0.05, right=0.95)
sm  = plt.cm.ScalarMappable(cmap='gray_r',
                             norm=plt.Normalize(vmin=vmin_rho, vmax=vmax_rho))
sm.set_array([])
cbar = fig.colorbar(sm, cax=ax, orientation='horizontal')
cbar.set_ticks([0.5, 1.6])
cbar.ax.tick_params(labelsize=36)
plt.savefig("figs/colorbar_rho.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.close()

print("Done — figures saved to figs/")
