"""Schematic of the acquisition sequence and the shrink it accumulates.

Draws what steps15 does with the shrinkage of a mosaic scan:

  * top    — a Gantt strip, one row per tile in MOSAIC order (as `tiles=` in the
             config lists them, left to right), one bar per distance. Rows that
             light up out of top-to-bottom order are tiles acquired out of
             mosaic order, which is the whole point: `tile_order=` in the config
             gives the order they really ran in.
  * bottom — the shrink. The unit drawn is one (tile, distance) acquisition,
             since the shrink is measured per distance, but the tiles were taken
             one at a time — all of a tile's distances before the next tile
             started — so the sample's history is just the tiles laid end to end,
             each starting from everything the ones before it contributed. Every
             thick coloured segment is one distance, in the colour of its tile;
             together they are the session curve, one continuous line. The
             dotted curve under each is the same tile's profile as its own
             shapp.mat reports it — zero at its own first frame, as if the sample
             had reset for it — and the arrow is the offset steps15 adds.

Within one acquisition the shrink is drawn as a linear ramp, matching
`shrink_angle_ramp=true`: it goes from cum[k] at the first projection of plane k
to cum[k] + inc[k] at the last. With `shrink_angle_ramp=false` (Peter's
convention) each plane is instead flat at its mid value; the accumulation across
acquisitions is the same either way.

The x axis counts acquisitions, not wall clock. The sample only deforms while it
is being scanned, so the (long) hours between tiles are not part of its history
and are not on the axis.

Usage:
    python plot_shrink_timeline.py config_steps15.conf [-o out.png]
    python plot_shrink_timeline.py --demo -o out.png      # no data needed

Without the data the script falls back to `--demo`: the real YY037A tile order
with INVENTED shrink increments, so the shape of the schematic is right and the
numbers are not. The figure says so on its face.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from holotomocupy.reader import load_scan_infos, load_shrink_profile

# YY037A: the order the five tiles were acquired in, whole tile at a time. Not
# the mosaic order — see `tile_order=` in config_steps15.conf.
DEMO_ORDER = ['center', 'left', 'right', 'farright', 'farleft']
DEMO_TILES = ['farright', 'right', 'center', 'left', 'farleft']   # mosaic order
DEMO_NDIST = 4
# Invented shrink for --demo: a rate that decays with the scanning already done,
# integrated over each acquisition. Only the shape is meant to be believable.
DEMO_RATE0 = np.array([1.6e-3, 1.0e-3])   # (v, h) per acquisition, at the start
DEMO_TAU   = 6.0                          # in acquisitions


def demo_scans():
    """`--demo` stand-in for the ``(start, total)`` read from disk."""
    scans, done = {}, 0
    for tile in DEMO_ORDER:
        inc = np.stack([DEMO_RATE0 * DEMO_TAU
                        * (np.exp(-(done + k) / DEMO_TAU)
                           - np.exp(-(done + k + 1) / DEMO_TAU))
                        for k in range(DEMO_NDIST)])
        done += DEMO_NDIST
        # A tile measures only against its OWN first frame, so its profile is
        # the zero-based cumulative sum of its own increments — which is what
        # load_shrink_profile returns, and what knows nothing about any other
        # tile.
        start = np.concatenate([np.zeros((1, 2)), np.cumsum(inc, axis=0)[:-1]])
        scans[tile] = (start, inc.sum(axis=0))
    return scans, DEMO_TILES, DEMO_ORDER


def real_scans(cfg_file):
    """Per-tile ``(start, total)`` from the data the config points at."""
    from holotomocupy.config import parse_args_steps15
    args  = parse_args_steps15(cfg_file)
    path  = args.path + '/'
    tiles = args.tiles if args.tiles else ['']
    order = args.tile_order or tiles
    if sorted(order) != sorted(tiles):
        raise SystemExit(f'tile_order={args.tile_order} is not a permutation of '
                         f'tiles={tiles}')
    scans = {}
    for tile in tiles:
        pfile = (f'{args.pfile}_{tile}_{args.scan_suffix}' if tile and args.scan_suffix
                 else f'{args.pfile}_{tile}' if tile else args.pfile)
        infos = load_scan_infos(path, pfile)
        scans[tile] = load_shrink_profile(path, pfile, len(infos),
                                          int(infos[0]['TOMO_N']))
    out = args.path_out or '.'
    return (scans, tiles, order, args.shrink_angle_ramp,
            os.path.join(out, 'shrink_timeline.png'))


def acquisitions(scans, order):
    """One flat list of distances, in the order they ran — the unit of the model.

    Returns ``(acq, end)`` where each entry of ``acq`` is
    ``(tile, k, j, own, begin, inc)``: which distance ``k`` of which tile,
    its position ``j`` in the session, the shrink its own profile reports at its
    start (``own``, zero-based within its tile), the shrink the session had
    already accumulated when it started (``begin``), and what it adds (``inc``).
    ``end`` is where the session finishes.

    The whole model is here: tiles chained in ``order``, each tile's distances
    chained inside it.
    """
    acq, run, j = [], np.zeros(2), 0
    for tile in order:
        start, total = scans[tile]
        inc = np.diff(np.concatenate([start, np.asarray(total)[None]]), axis=0)
        for k in range(len(inc)):
            acq.append((tile, k, j, start[k], run + inc[:k].sum(axis=0), inc[k]))
            j += 1
        run = run + inc.sum(axis=0)
    return acq, run


def plot(scans, tiles, order, angle_ramp, out, demo=False, npix=3216):
    acq, end = acquisitions(scans, order)
    cols = plt.get_cmap('tab10').colors

    fig = plt.figure(figsize=(13, 7.4))
    gs  = fig.add_gridspec(2, 1, height_ratios=[len(tiles) * 0.34, 1],
                           hspace=0.10,
                           left=0.08, right=0.93, top=0.90, bottom=0.17)
    ax_t = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1], sharex=ax_t)

    # ---- Gantt: one bar per (tile, distance) ------------------------------
    for tile, k, j, *_ in acq:
        r = tiles.index(tile)
        ax_t.barh(r, 0.9, left=j + 0.05, height=0.55, color=cols[r % 10],
                  alpha=0.85, zorder=3)
        ax_t.text(j + 0.5, r, f'{k + 1}', ha='center', va='center',
                  fontsize=7, color='w', zorder=4)
    ax_t.set_ylim(len(tiles) - 0.5, -0.7)
    ax_t.set_yticks(range(len(tiles)))
    ax_t.set_yticklabels(tiles, fontsize=9)
    ax_t.grid(axis='x', ls=':', alpha=0.35)
    ax_t.tick_params(labelbottom=False, left=False)

    # ---- shrink -----------------------------------------------------------
    # One thick segment per acquisition, ramping from what the session had
    # already accumulated to that plus this acquisition's own increment. Laid
    # end to end they meet exactly, so the coloured segments ARE the session
    # curve — the colour only says which tile was being scanned.
    for tile, k, j, own, begin, inc in acq:
        r  = tiles.index(tile)
        xx = np.array([j, j + 1.0])
        for c, ls in enumerate(['-', '--']):
            ax_b.plot(xx, [begin[c], begin[c] + inc[c]], color=cols[r % 10],
                      ls=ls, lw=4.5, alpha=0.5, zorder=4)
        ax_b.plot(xx[0], begin[0], 'o', ms=3.5, color=cols[r % 10], zorder=6)

    # The same tiles as their own shapp.mat / shrink_list.mat reports them: zero
    # at the tile's own first frame. The gap to the coloured segment above it is
    # the offset steps15 adds.
    for r, tile in enumerate(tiles):
        mine = [a for a in acq if a[0] == tile]
        xx = np.concatenate([[a[2], a[2] + 1.0] for a in mine])
        yy = np.concatenate([[a[3][0], a[3][0] + a[5][0]] for a in mine])
        ax_b.plot(xx, yy, color=cols[r % 10], ls=':', lw=1.2, alpha=0.7,
                  zorder=3)
        if mine[0][4][0] > 0:      # the offset steps15 adds at the first frame
            ax_b.annotate('', xy=(xx[0], mine[0][4][0]), xytext=(xx[0], 0),
                          zorder=7,
                          arrowprops=dict(arrowstyle='->', lw=1.0,
                                          color=cols[r % 10],
                                          shrinkA=0, shrinkB=0))

    # A thin black line over the top, so the accumulation reads as one curve.
    xs = np.concatenate([[a[2], a[2] + 1.0] for a in acq])
    for c, ls, lw in [(0, '-', 1.9), (1, '--', 1.3)]:
        ys = np.concatenate([[a[4][c], a[4][c] + a[5][c]] for a in acq])
        ax_b.plot(xs, ys, color='k', ls=ls, lw=lw, zorder=5)

    ax_b.grid(ls=':', alpha=0.35)
    ax_b.set_ylabel('shrink' + (' — INVENTED values' if demo else ''))
    # Shrink is a scale factor, so what it costs in pixels depends where you
    # are in the frame: eff_mag = norm_mag/(1+shrink) moves a feature r px from
    # the centre by about r*shrink. The worst case is the frame edge, r = n/2.
    ax_px = ax_b.secondary_yaxis(
        'right', functions=(lambda v: v * npix / 2, lambda p: p * 2 / npix))
    ax_px.set_ylabel(f'px at the frame edge (n={npix})', fontsize=9)
    ax_b.set_xlabel('acquisitions, in the order they ran   —   the hours between '
                    'tiles are not on this axis: nothing was scanning, so '
                    'nothing deformed', fontsize=9)
    ax_b.set_xlim(-0.5, len(acq) + 0.1)

    # Tile boundaries: where the session moved on to the next tile.
    for tile in order[1:]:
        j0 = min(a[2] for a in acq if a[0] == tile)
        for ax in (ax_t, ax_b):
            ax.axvline(j0, color='0.55', lw=1.1, ls=(0, (4, 3)), zorder=2)

    handles = [plt.Line2D([], [], color='k', lw=1.9, label='session s(t)'),
               plt.Line2D([], [], color='0.5', lw=4.5, alpha=0.5,
                          label='one acquisition, in its tile\'s colour'),
               plt.Line2D([], [], color='0.5', lw=1.2, ls=':',
                          label='the tile\'s own profile (starts at 0)'),
               plt.Line2D([], [], color='k', lw=1.9, ls='-', label='vertical'),
               plt.Line2D([], [], color='k', lw=1.3, ls='--',
                          label='horizontal')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.10),
               ncol=len(handles), fontsize=8, frameon=False)
    ax_t.set_title(f'tiles, rows in mosaic order, acquired '
                   f'{" → ".join(order)}   (bar = one distance, numbered)',
                   loc='left', fontsize=9)
    fig.suptitle(
        'Acquisition sequence and accumulated shrink'
        + ('   —   SCHEMATIC: real order, invented shrink values'
           if demo else ''),
        fontsize=12)
    fig.text(0.5, 0.035,
             'shrink ramps linearly across each scan (shrink_angle_ramp=true); '
             'every (tile, distance) adds its own increment on top of the ones '
             'before it — a plain cumulative sum in the order they ran',
             ha='center', fontsize=8, color='0.35')
    fig.savefig(out, dpi=140)
    print(f'wrote {out}')

    # One constant per tile: everything the tiles before it contributed.
    print(f'\noffsets steps15 adds; px is what the shrink costs at the edge of '
          f'an n={npix} frame, i.e. shrink*n/2:')
    for tile in order:
        off = next(a[4] - a[3] for a in acq if a[0] == tile)
        print(f'  {tile:<10s} v={off[0]:+.6f} h={off[1]:+.6f}'
              f'   ({off[0] * npix / 2:+.2f}, {off[1] * npix / 2:+.2f}) px')
    print(f'  end of session       v={end[0]:+.6f} h={end[1]:+.6f}'
          f'   ({end[0] * npix / 2:+.2f}, {end[1] * npix / 2:+.2f}) px')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config', nargs='?', help='steps15 config file')
    p.add_argument('-o', '--out', default=None, help='output PNG')
    p.add_argument('--demo', action='store_true',
                   help='hardcoded YY037A tile order with invented shrink values')
    p.add_argument('--npix', type=int, default=3216,
                   help='frame width in px, for the secondary "px at the frame '
                        'edge" axis (default 3216)')
    a = p.parse_args()

    ramp_on, out = True, 'shrink_timeline.png'
    if a.config and not a.demo:
        try:
            scans, tiles, order, ramp_on, out = real_scans(a.config)
        except (FileNotFoundError, OSError) as e:
            print(f'{e}\nfalling back to --demo', file=sys.stderr)
            scans, tiles, order = demo_scans()
            a.demo = True
    else:
        scans, tiles, order = demo_scans()
        a.demo = True

    if not ramp_on:
        print('note: the config has shrink_angle_ramp=false, so steps15 holds '
              'each plane flat at its mid value; the plot still draws the ramp.',
              file=sys.stderr)
    plot(scans, tiles, order, ramp_on, a.out or out, demo=a.demo, npix=a.npix)


if __name__ == '__main__':
    main()
