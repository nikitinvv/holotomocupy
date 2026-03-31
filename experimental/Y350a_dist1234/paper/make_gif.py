"""
Read prec.tiff, mrec.tiff, drec.tif from /data2/vnikitin/video_brain/,
extract the middle slice of each 3-D volume, and write a comparison GIF.
Run from the paper/ directory; output is saved alongside this script.
"""
import sys
import numpy as np
import scipy.ndimage
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = '/data2/vnikitin/video_brain'
FILES = {
    'prec': f'{DATA_DIR}/prec.tif',
    'mrec': f'{DATA_DIR}/mrec.tif',
    'drec': f'{DATA_DIR}/drec.tif',
}
LABELS = {
    'prec': 'Conventional',
    'mrec': 'Proposed (joint)',
    'drec': 'Proposed (joint) + ML denoised',
}
OUT_GIF           = 'brain_comparison.gif'
OUT_GIF_ZOOM      = 'brain_comparison_zoom.gif'
OUT_GIF2          = 'brain_comparison2.gif'
OUT_GIF2_ZOOM     = 'brain_comparison2_zoom.gif'
OUT_GIF2_ZOOM2    = 'brain_comparison2_zoom2.gif'
ZOOM2_Y = slice(1100, 1400)
ZOOM2_X = slice(1400, 2000)
ZOOM_Y = slice(1000, 1600)   # 600 px tall
ZOOM_X = slice(800, 2000)    # 1200 px wide → 1200:600 = 16:8
DURATION   = 2000   # ms per frame
DPI        = 150
PIXEL_NM   = 20     # nm per pixel

# ── Load middle slices ────────────────────────────────────────────────────────
slices = {}
for key, path in FILES.items():
    print(f'Reading {path} ...', flush=True)
    vol = tifffile.imread(path)
    print(f'  shape={vol.shape}  dtype={vol.dtype}')
    # Middle slice along first axis (z/depth)
    mid = vol.shape[0] // 2
    sl = vol[mid].astype(np.float32)
    if key in ('mrec', 'drec'):
        sl = scipy.ndimage.shift(sl, shift=(-2, -3), mode='nearest')
    slices[key] = sl

# ── Normalise each slice to [0, 1] with shared or per-slice clipping ─────────
# Use a robust percentile clip so extreme values don't crush contrast
def clip_norm(arr, lo=1, hi=99):
    vlo, vhi = np.percentile(arr, lo), np.percentile(arr, hi)
    arr = np.clip(arr, vlo, vhi)
    return (arr - vlo) / (vhi - vlo + 1e-12)

# prec and mrec share the same colorbar limits computed from their combined central ROI
def center_roi(arr):
    cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
    return arr[cy-256:cy+256, cx-256:cx+256]

# prec + mrec shared colorbar
combined_pm = np.concatenate([center_roi(slices['prec']).ravel(),
                              center_roi(slices['mrec']).ravel()])
lo_pm = np.percentile(combined_pm, 1)
hi_pm = np.percentile(combined_pm, 99)

frames_norm = {}
for k in ('prec', 'mrec'):
    arr = np.clip(slices[k], lo_pm, hi_pm)
    frames_norm[k] = (arr - lo_pm) / (hi_pm - lo_pm + 1e-12)

# mrec + drec shared colorbar
combined_md = np.concatenate([center_roi(slices['mrec']).ravel(),
                              center_roi(slices['drec']).ravel()])
lo_md = np.percentile(combined_md, 1)
hi_md = np.percentile(combined_md, 99)

frames_norm2 = {}
for k in ('mrec', 'drec'):
    arr = np.clip(slices[k], lo_md, hi_md)
    frames_norm2[k] = (arr - lo_md) / (hi_md - lo_md + 1e-12)

# ── Scalebar helper ───────────────────────────────────────────────────────────
def add_scalebar(ax, img_w_px, scalebar_nm, pixel_nm=PIXEL_NM, fontsize=42,
                 dark=False, shadow=True):
    """Draw a scalebar in the bottom-right corner."""
    from matplotlib.patches import FancyBboxPatch
    bar_px    = scalebar_nm / pixel_nm
    label     = f'{scalebar_nm} nm' if scalebar_nm < 1000 else f'{scalebar_nm//1000} µm'
    bar_frac  = bar_px / img_w_px
    margin    = 0.03
    x1_frac   = 1 - margin
    x0_frac   = x1_frac - bar_frac
    xmid_frac = (x0_frac + x1_frac) / 2
    ymid_frac = 0.05
    color     = 'black' if dark else 'white'
    bg        = 'white'  if dark else 'black'
    if shadow:
        pad_x, pad_y = 0.015, 0.01
        rect = FancyBboxPatch(
            (x0_frac - pad_x, ymid_frac - pad_y),
            bar_frac + 2 * pad_x, 0.06,
            boxstyle='round,pad=0.005',
            transform=ax.transAxes, clip_on=False,
            fc=bg, alpha=0.5, lw=0)
        ax.add_patch(rect)
    ax.plot([x0_frac, x1_frac], [ymid_frac, ymid_frac],
            transform=ax.transAxes, color=color, lw=2, solid_capstyle='butt')
    ax.text(xmid_frac, ymid_frac + 0.01, label,
            transform=ax.transAxes,
            va='bottom', ha='center', fontsize=fontsize, color=color)

# ── Render each frame as a PNG in memory ─────────────────────────────────────
pil_frames = []
for key in ('prec', 'mrec'):
    img  = frames_norm[key]
    h, w = img.shape

    fig, ax = plt.subplots(figsize=(w / DPI, h / DPI), dpi=DPI)
    ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    ax.axis('off')

    ax.text(0.98, 0.97, LABELS[key],
            transform=ax.transAxes,
            va='top', ha='right', fontsize=42, color='white',
            bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6))

    add_scalebar(ax, w, scalebar_nm=5000, dark=True, shadow=False)   # 5 µm for full slice

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    pil_img = Image.frombuffer('RGBA', fig.canvas.get_width_height(), buf, 'raw', 'RGBA', 0, 1)
    pil_frames.append(pil_img.convert('RGB'))
    plt.close(fig)
    print(f'  rendered {key}: {pil_img.size}')

# ── Write GIF ────────────────────────────────────────────────────────────────
pil_frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=pil_frames[1:],
    duration=DURATION,
    loop=0,
)
print(f'\nSaved: {OUT_GIF}')

# ── Zoomed GIF (16:8 = 2:1, crop is exactly 1200×600 px) ────────────────────
pil_frames_zoom = []
for key in ('prec', 'mrec'):
    img = frames_norm[key][ZOOM_Y, ZOOM_X]
    h, w = img.shape   # 600 × 1200

    fig, ax = plt.subplots(figsize=(w / DPI, h / DPI), dpi=DPI)
    fig.patch.set_facecolor('black')
    ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    ax.axis('off')

    ax.text(0.98, 0.97, LABELS[key],
            transform=ax.transAxes,
            va='top', ha='right', fontsize=42/4, color='white',
            bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.5, lw=0))
    add_scalebar(ax, img.shape[1], scalebar_nm=1000, dark=True, fontsize=42/4)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    pil_img = Image.frombuffer('RGBA', fig.canvas.get_width_height(), buf, 'raw', 'RGBA', 0, 1)
    pil_frames_zoom.append(pil_img.convert('RGB'))
    plt.close(fig)
    print(f'  rendered zoom {key}: {pil_img.size}')

pil_frames_zoom[0].save(
    OUT_GIF_ZOOM,
    save_all=True,
    append_images=pil_frames_zoom[1:],
    duration=DURATION,
    loop=0,
)
print(f'Saved: {OUT_GIF_ZOOM}')

# ── GIF 2: mrec vs drec (full) ───────────────────────────────────────────────
pil_frames2 = []
for key in ('mrec', 'drec'):
    img  = frames_norm2[key]
    h, w = img.shape

    fig, ax = plt.subplots(figsize=(w / DPI, h / DPI), dpi=DPI)
    ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    ax.axis('off')
    ax.text(0.98, 0.97, LABELS[key],
            transform=ax.transAxes,
            va='top', ha='right', fontsize=42, color='white',
            bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6))
    add_scalebar(ax, w, scalebar_nm=5000, dark=True, shadow=False)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    pil_img = Image.frombuffer('RGBA', fig.canvas.get_width_height(), buf, 'raw', 'RGBA', 0, 1)
    pil_frames2.append(pil_img.convert('RGB'))
    plt.close(fig)
    print(f'  rendered2 {key}: {pil_img.size}')

pil_frames2[0].save(OUT_GIF2, save_all=True, append_images=pil_frames2[1:],
                    duration=DURATION, loop=0)
print(f'\nSaved: {OUT_GIF2}')

# ── GIF 2: mrec vs drec (zoomed) ─────────────────────────────────────────────
pil_frames2_zoom = []
for key in ('mrec', 'drec'):
    img = frames_norm2[key][ZOOM_Y, ZOOM_X]
    h, w = img.shape

    fig, ax = plt.subplots(figsize=(w / DPI, h / DPI), dpi=DPI)
    fig.patch.set_facecolor('black')
    ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    ax.axis('off')
    ax.text(0.98, 0.97, LABELS[key],
            transform=ax.transAxes,
            va='top', ha='right', fontsize=42/4, color='white',
            bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.5, lw=0))
    add_scalebar(ax, img.shape[1], scalebar_nm=1000, dark=True, fontsize=42/4)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    pil_img = Image.frombuffer('RGBA', fig.canvas.get_width_height(), buf, 'raw', 'RGBA', 0, 1)
    pil_frames2_zoom.append(pil_img.convert('RGB'))
    plt.close(fig)
    print(f'  rendered2 zoom {key}: {pil_img.size}')

pil_frames2_zoom[0].save(OUT_GIF2_ZOOM, save_all=True, append_images=pil_frames2_zoom[1:],
                         duration=DURATION, loop=0)
print(f'Saved: {OUT_GIF2_ZOOM}')

# ── GIF 2: mrec vs drec (zoom2) ───────────────────────────────────────────────
pil_frames2_zoom2 = []
for key in ('mrec', 'drec'):
    img = frames_norm2[key][ZOOM2_Y, ZOOM2_X]
    h, w = img.shape

    fig, ax = plt.subplots(figsize=(w / DPI, h / DPI), dpi=DPI)
    fig.patch.set_facecolor('black')
    ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    ax.axis('off')
    ax.text(0.98, 0.97, LABELS[key],
            transform=ax.transAxes,
            va='top', ha='right', fontsize=42/8, color='white',
            bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.5, lw=0))
    add_scalebar(ax, img.shape[1], scalebar_nm=500, dark=True, fontsize=42/8)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    pil_img = Image.frombuffer('RGBA', fig.canvas.get_width_height(), buf, 'raw', 'RGBA', 0, 1)
    pil_frames2_zoom2.append(pil_img.convert('RGB'))
    plt.close(fig)
    print(f'  rendered2 zoom2 {key}: {pil_img.size}')

pil_frames2_zoom2[0].save(OUT_GIF2_ZOOM2, save_all=True, append_images=pil_frames2_zoom2[1:],
                           duration=DURATION, loop=0)
print(f'Saved: {OUT_GIF2_ZOOM2}')

# ── Scrolling GIF: vertical slices (vol[:, y, x]) of drec, 10 fps ────────────
# Each frame is a yz-plane: vol[:, y, ZOOM_X], scrolling through ZOOM_Y range.
# Shift (-2,-3) compensated by offsetting y and x indices directly.
OUT_GIF_SCROLL  = 'brain_drec_scroll.gif'
SCROLL_DURATION = 83    # ms per frame = 12 fps
SCROLL_DPI      = 100
DY, DX = 2, 3
cx = (ZOOM_X.start + ZOOM_X.stop) // 2 + 200
half = int((ZOOM_X.stop - ZOOM_X.start) * 1.5 / 2)
SX = slice(cx - half + DX, cx + half + DX)          # 1.5× wider, same centre, shift offset

print(f'\nLoading drec volume for vertical scroll ...', flush=True)
drec_vol = tifffile.imread(FILES['drec'])           # shape (n_z, n_y, n_x)
n_z, n_y, n_x = drec_vol.shape

scroll_frames = []
y_range = range(ZOOM_Y.start + DY, ZOOM_Y.stop + DY)
for i, y in enumerate(y_range):
    frame = drec_vol[:, y, SX].astype(np.float32)  # shape (n_z, n_x_crop)
    target_w = int(frame.shape[0] * 16 / 9)
    frame = frame[:, -target_w:]                    # crop from left → 16:9
    frame = np.clip(frame, lo_md, hi_md)
    frame = (frame - lo_md) / (hi_md - lo_md + 1e-12)

    h, w = frame.shape
    fig, ax = plt.subplots(figsize=(w / SCROLL_DPI, h / SCROLL_DPI), dpi=SCROLL_DPI)
    fig.patch.set_facecolor('black')
    ax.imshow(frame, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    ax.axis('off')
    add_scalebar(ax, w, scalebar_nm=1000, dark=True, fontsize=21)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    pil_img = Image.frombuffer('RGBA', fig.canvas.get_width_height(),
                               buf, 'raw', 'RGBA', 0, 1)
    scroll_frames.append(pil_img.convert('RGB'))
    plt.close(fig)
    if i % 50 == 0:
        print(f'  y {y} ({i}/{len(y_range)})', flush=True)

scroll_frames[0].save(OUT_GIF_SCROLL, save_all=True, append_images=scroll_frames[1:],
                      duration=SCROLL_DURATION, loop=0)
print(f'Saved: {OUT_GIF_SCROLL}  ({len(scroll_frames)} frames)')
