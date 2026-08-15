#!/usr/bin/env python
"""Write the tiles of one projection side by side, butt-joined with no gaps.

Each frame is first shifted back by -shifts_final (in detector px of its own
plane, i.e. the object-plane value times norm_magnifications[k], so the shift
gets smaller on the more magnified planes) — pass --no-shift to skip that.
No registration between tiles: they are simply concatenated in the config's
tile order (left → right) so the seams can be judged by eye.

Three files land in --out-dir, in increasing order of how much is done to the
grey values:
    tiles_raw_dist{k}.png    tile medians equalised, then one common grey scale
    tiles_norm_dist{k}.png   each tile flattened (the illumination bowl removed)
                             and stretched to its own percentiles
    tiles_band_dist{k}.png   band-passed (DoG) and destriped, structure only

Contrast is set by --clip (percentile cut, larger = harder) and --bsig.

Usage:
    python show_tiles.py [config_steps14.conf] [--dist 1] [--theta 0]
        [--navg 8] [--no-shift] [--clip 5] [--bsig 1.5] [--bin 1]
        [--ref-tile NAME] [--out-dir DIR]
"""

import argparse
import os
import sys

import numpy as np
import scipy.ndimage as ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', '..', 'src'))
sys.path.insert(0, _here)
from holotomocupy.config import parse_args_steps15          # noqa: E402
from plot_shifts import shift_terms, tile_geometry          # noqa: E402
from plot_return_frames import load_refs, corrected, pfile_tile   # noqa: E402


def bin_image(a, f):
    if f <= 1:
        return a
    ny, nx = a.shape
    return a[:ny // f * f, :nx // f * f].reshape(ny // f, f, nx // f, f).mean((1, 3))


def mean_projections(dname, pfile, k, theta, navg, dark, ref):
    """Average ``navg`` consecutive projections, skipping any file that is not
    there. Returns None if none of them could be read."""
    acc, got, missing = None, 0, 0
    for j in range(navg):
        try:
            a = corrected(dname, pfile, k, theta + j, dark, ref)
        except (OSError, IOError):
            missing += 1
            continue
        acc = a if acc is None else acc + a
        got += 1
    if missing:
        print(f'    {missing} of {navg} projections missing')
    return None if got == 0 else (acc / got).astype('float32')


def band(a, s1=2.0, s2=40.0):
    """Band-pass and destripe: kill the illumination bowl and the fixed
    vertical stripes, keep the sample texture."""
    b = ndimage.gaussian_filter(a, s1) - ndimage.gaussian_filter(a, s2)
    b = b - b.mean(0, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return b / (b.std() + 1e-9)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config', nargs='?', default=os.path.join(_here, 'config_steps14.conf'))
    ap.add_argument('--dist', type=int, default=1)
    ap.add_argument('--theta', type=int, default=0)
    ap.add_argument('--navg', type=int, default=8,
                    help='average this many consecutive projections (default 8, '
                         'the sample turns 0.03 deg per projection)')
    ap.add_argument('--ref-tile', default=None,
                    help='use this tile\'s flats and darks for every tile '
                         '(default: each tile uses its own)')
    ap.add_argument('--no-shift', dest='shift', action='store_false',
                    help='join the raw frames instead of shifting them back by '
                         '-shifts_final')
    ap.add_argument('--clip', type=float, default=5.0,
                    help='grey scale runs from this percentile to 100-this; '
                         'smaller = softer, larger = harder contrast (default 5)')
    ap.add_argument('--bsig', type=float, default=1.5,
                    help='band-passed image is shown over +-this many sigma '
                         '(default 1.5)')
    ap.add_argument('--bin', type=int, default=1)
    ap.add_argument('--out-dir', default=_here)
    args = ap.parse_args()

    cfg = parse_args_steps15(args.config)
    path = cfg.path.rstrip('/')
    tiles = cfg.tiles if isinstance(cfg.tiles, (list, tuple)) else cfg.tiles.split(',')
    tiles = [str(t).strip() for t in tiles if str(t).strip()]
    k = args.dist - 1
    geom = None
    for t in tiles:                       # the first tile with a readable .info
        try:
            geom = tile_geometry(path, pfile_tile(cfg.pfile, t, cfg.scan_suffix))
            break
        except (OSError, IOError):
            continue
    if geom is None:
        sys.exit('no tile has a readable .info')
    ntheta, ndist, norm_mag = geom
    print(f'tiles left→right: {tiles}   dist {args.dist}  theta {args.theta}'
          f'  averaging {args.navg} projections')

    # flats/darks: each tile's own by default, or one tile's for all of them
    ref_dname = None
    if args.ref_tile:
        ref_dname = (f'{path}/'
                     f'{pfile_tile(cfg.pfile, args.ref_tile, cfg.scan_suffix)}'
                     f'_{args.dist}_')
        print(f'using the flats and darks of tile "{args.ref_tile}" for every tile')

    imgs, kept = [], []
    for t in tiles:
        pfile = pfile_tile(cfg.pfile, t, cfg.scan_suffix)
        dname = f'{path}/{pfile}_{args.dist}_'
        try:
            dark, ref0, _ = load_refs(ref_dname or dname, ntheta)
            a = mean_projections(dname, pfile, args.dist, args.theta, args.navg,
                                 dark, ref0)
        except (OSError, IOError, IndexError) as e:
            print(f'  {t:9s} MISSING — {e.__class__.__name__}, left blank')
            a = None
        if a is None:
            imgs.append(None)          # keep the slot so the others stay in place
            kept.append(t)
            continue
        if args.shift:
            # shifts_final is in object-plane px on the finest grid; this frame
            # is in detector px of plane k, hence the x norm_mag[k] (which makes
            # the shift smaller on the more magnified planes)
            terms, _ = shift_terms(path, pfile, ntheta, ndist, norm_mag, cfg.ref_dist)
            obj = np.sum([v[args.theta, k] for _, v in terms], axis=0)
            s = obj * norm_mag[k]
            a = ndimage.shift(a, (-s[0], -s[1]), order=3, mode='nearest')
            print(f'  {t:9s} shifts_final v={obj[0]:+7.3f} h={obj[1]:+7.3f} obj px'
                  f'  ->  shifted back by v={-s[0]:+7.3f} h={-s[1]:+7.3f} det px')
        imgs.append(bin_image(a, args.bin))
        kept.append(t)
        print(f'  {t:9s} mean {imgs[-1].mean():.3f}  std {imgs[-1].std():.4f}')

    have = [a for a in imgs if a is not None]
    if not have:
        sys.exit(f'no tile has data at dist {args.dist}')
    ny, nx = have[0].shape
    blank = np.full((ny, nx), np.nan, dtype='float32')   # missing tiles stay empty
    tag = f'dist{args.dist}'
    tot = nx * len(imgs)
    px = args.bin          # one column of the figure is this many detector px

    def save(name, panels, vmin, vmax, title):
        """One image with a column axis, so the overlap can be read off by eye."""
        cat = np.concatenate([blank if a is None else a for a in panels], axis=1)
        figw = 26.0
        dpi = min(400, max(120, tot / figw))
        fig, ax = plt.subplots(figsize=(figw, figw * ny / tot + 1.6))
        ax.imshow(cat, cmap='gray', vmin=vmin, vmax=vmax,
                  extent=[0, tot * px, ny * px, 0], interpolation='nearest')
        ax.set_facecolor('0.15')
        step = 256 * max(1, int(round(tot * px / 256 / 45)))
        ax.set_xticks(np.arange(0, tot * px + 1, step))
        ax.set_yticks(np.arange(0, ny * px + 1, step))
        ax.tick_params(labelsize=8)
        ax.grid(True, color='tab:cyan', lw=0.4, alpha=0.35)
        for i, t in enumerate(tiles):
            if i:
                ax.axvline(i * nx * px, color='tab:red', lw=1.0)
            ax.text((i + 0.5) * nx * px, -0.02 * ny * px,
                    f'{t}   (cols {i * nx * px}…{(i + 1) * nx * px - 1})',
                    color='tab:red', ha='center', va='bottom', fontsize=10)
        ax.set_xlabel('column in the butt-joined image [detector px]')
        ax.set_ylabel('row [px]')
        ax.set_title(f'{cfg.pfile}  dist {args.dist}  theta {args.theta}'
                     f'  (mean of {args.navg})   {title}   '
                     + ('shifted back by -shifts_final x norm_mag'
                        if args.shift else 'raw, no shift'),
                     fontsize=11, pad=26)
        fig.tight_layout()
        out = os.path.join(args.out_dir, name)
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        print(f'saved  {out}   ({ny * px} x {tot * px} px, seams every {nx * px} columns)')

    c = args.clip
    # tile means run 0.60, 0.40, 0.31, 0.30, 0.40 — on a single grey scale the
    # middle of the mosaic goes black, so put every tile on the same median first
    eq = [None if a is None else a * (1.0 / np.median(a[::4, ::4])) for a in imgs]
    lo, hi = np.percentile(np.concatenate([a for a in eq if a is not None],
                                          axis=1)[::4, ::4], [c, 100 - c])
    save(f'tiles_raw_{tag}.png', eq, lo, hi,
         f'common grey scale after equalising the tile medians, '
         f'clipped at {c}/{100 - c}%')

    def stretch(a):
        """Per-tile contrast stretch. The percentiles are taken on a band-passed
        copy so the illumination bowl inside a tile does not eat the range."""
        b = ndimage.gaussian_filter(a, 2.0) - ndimage.gaussian_filter(a, 60.0)
        p0, p1 = np.percentile(b[::4, ::4], [c, 100 - c])
        return np.clip((b - p0) / (p1 - p0 + 1e-9), -0.2, 1.2)

    save(f'tiles_norm_{tag}.png', [None if a is None else stretch(a) for a in imgs],
         0.0, 1.0, f'each tile flattened and stretched to its own {c}/{100 - c}%')
    save(f'tiles_band_{tag}.png', [None if a is None else band(a) for a in imgs],
         -args.bsig, args.bsig,
         f'band-passed and destriped, +-{args.bsig} sigma')


if __name__ == '__main__':
    main()
