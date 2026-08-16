"""Schematic of the acquisition sequence and the shrink it accumulates.

Draws what steps15 does with the shrinkage of a mosaic scan:

  * top    — a Gantt strip, one row per tile in MOSAIC order (as `tiles=` in the
             config lists them, left to right), one bar per distance. Bars that
             zig-zag between rows are tiles acquired out of mosaic order; bars
             that interleave are tiles acquired alternately, one distance each.
  * bottom — the shrink. The unit is one (tile, distance) acquisition: the shrink
             is measured per distance, and nothing deforms between one
             acquisition and the next, so the sample's whole history is just the
             distances laid end to end in the order they ran, each adding its own
             increment on top of everything before it. Every thick coloured
             segment is one such acquisition, drawn in the colour of the tile
             that was being scanned; together they are the session curve, one
             continuous line. The dotted curve under each is the same tile's
             profile as its own shapp.mat reports it — zero at its own first
             frame, as if the sample had reset for it — and the arrow is the
             offset steps15 adds to close the gap.

Within one acquisition the shrink is drawn as a linear ramp, matching
`shrink_angle_ramp=true`: it goes from cum[k] at the first projection of plane k
to cum[k] + inc[k] at the last. With `shrink_angle_ramp=false` (Peter's
convention) each plane is instead flat at its mid value; the accumulation across
acquisitions is the same either way.

Usage:
    python plot_shrink_timeline.py config_steps15.conf [-o out.png]
    python plot_shrink_timeline.py --demo -o out.png      # no data needed

Without the data the script falls back to `--demo`: the real YY037A `Date=`
timestamps with INVENTED shrink increments, so the shape of the schematic is
right and the numbers are not. The figure says so on its face.
"""

import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from holotomocupy.reader import (load_scan_infos, scan_times,
                                 load_shrink_profile, shrink_sequence)


def scan_spans(times_by_key):
    """When each acquisition started and stopped, in the order they ran.

    Drawing only: the accumulation itself needs the start times and nothing
    else, since the increments are measured per distance. This is here so the
    Gantt bars have a length and the ramps have a width.

    ``times_by_key[key]`` is one scan's ``(ndist,)`` distance start times, as
    ``scan_times`` returns them. An acquisition runs until the next one starts,
    whichever scan that one belongs to — which is what an interleaved pair
    needs: left's distance 1 stops when right's distance 1 starts four minutes
    later, not eight minutes later when left's own distance 2 does.

    The last acquisition before a break has no next start to stop it, so it is
    given the median length of that scan's other acquisitions in the same run
    (or of the whole run, if it has none). Using the scan's own distance
    spacing instead would be wrong exactly where it matters: during an
    interleave that spacing counts the partner's acquisition too, so the last
    distance of the pair would be drawn twice as long as the seven before it.

    Returns a list of ``(t0, t1, key, i)``, sorted by ``t0``.
    """
    seq  = sorted((float(t), key, i) for key, ts in times_by_key.items()
                  for i, t in enumerate(ts))
    gaps = [seq[j + 1][0] - seq[j][0] for j in range(len(seq) - 1)] + [np.inf]

    # How far apart a scan puts its own distances — the yardstick for "did the
    # session carry straight on here, or stop?". A scan with a single distance
    # says nothing about that, so it borrows the other scans' figure.
    spacings = [float(np.median(np.diff(ts)))
                for ts in times_by_key.values() if len(ts) > 1]
    fallback = (float(np.median(spacings)) if spacings else
                float(np.median([g for g in gaps if np.isfinite(g)] or [1.0])))
    typ = {k: (float(np.median(np.diff(ts))) if len(ts) > 1 else fallback)
           for k, ts in times_by_key.items()}

    out, run = [], []
    for j, (t0, key, i) in enumerate(seq):
        if gaps[j] <= typ[key]:          # the next scan starts before this one
            run.append((t0, t0 + gaps[j], key, i))   # would have finished
            continue
        lens = ([t1 - ta for ta, t1, k, _i in run if k == key]
                or [t1 - ta for ta, t1, _k, _i in run])
        out.extend(run)
        out.append((t0, t0 + (float(np.median(lens)) if lens else typ[key]),
                    key, i))
        run = []
    return out


# The real YY037A timestamps, for --demo. Same source as the .info Date= lines:
# the acquisition ran Apr 18-19 2026, five tiles, four distances each.
DEMO_DATES = {
    'center':   ['Sat Apr 18 07:47:10 2026', 'Sat Apr 18 07:55:12 2026',
                 'Sat Apr 18 08:03:15 2026', 'Sat Apr 18 08:11:18 2026'],
    'left':     ['Sat Apr 18 18:34:02 2026', 'Sat Apr 18 18:42:05 2026',
                 'Sat Apr 18 18:50:08 2026', 'Sat Apr 18 18:58:11 2026'],
    'right':    ['Sat Apr 18 18:38:31 2026', 'Sat Apr 18 18:46:34 2026',
                 'Sat Apr 18 18:54:37 2026', 'Sat Apr 18 19:02:40 2026'],
    'farright': ['Sun Apr 19 09:12:44 2026', 'Sun Apr 19 09:22:47 2026',
                 'Sun Apr 19 09:32:50 2026', 'Sun Apr 19 09:42:53 2026'],
    'farleft':  ['Sun Apr 19 10:07:21 2026', 'Sun Apr 19 10:18:24 2026',
                 'Sun Apr 19 10:29:27 2026', 'Sun Apr 19 10:37:30 2026'],
}
# Invented shrink for --demo: a rate that decays with the scanning already done,
# integrated over each acquisition. Only the shape is meant to be believable.
DEMO_RATE0 = np.array([1.6e-3, 1.0e-3])   # (v, h) per hour of scanning, at t=0
DEMO_TAU   = 1.6                          # hours OF SCANNING, not of wall clock


def _times(dates):
    return np.array([time.mktime(time.strptime(d, '%a %b %d %H:%M:%S %Y'))
                     for d in dates], dtype='float64')


def demo_scans():
    """`--demo` stand-in for the ``(times, start, total)`` read from disk."""
    ts = {t: _times(d) for t, d in DEMO_DATES.items()}
    # Walk the acquisitions in the order they ran, so the invented rate decays
    # with scanning done rather than with the calendar — the idle hours are not
    # part of the sample's history.
    inc  = {t: np.zeros((len(v), 2)) for t, v in ts.items()}
    done = 0.0
    for t0, t1, tile, k in scan_spans(ts):
        dur = (t1 - t0) / 3600.0
        inc[tile][k] = (DEMO_RATE0 * DEMO_TAU
                        * (np.exp(-done / DEMO_TAU)
                           - np.exp(-(done + dur) / DEMO_TAU)))
        done += dur
    # Each tile measures only against its OWN first frame, so its profile is the
    # zero-based cumulative sum of its own increments — what load_shrink_profile
    # returns, and what knows nothing about any other tile.
    return {t: (ts[t], np.concatenate([np.zeros((1, 2)),
                                       np.cumsum(inc[t], axis=0)[:-1]]),
                inc[t].sum(axis=0))
            for t in ts}


def real_scans(cfg_file):
    """Per-tile ``(times, start, total)`` from the data the config points at."""
    from holotomocupy.config import parse_args_steps15
    args  = parse_args_steps15(cfg_file)
    path  = args.path + '/'
    tiles = args.tiles if args.tiles else ['']
    scans = {}
    for tile in tiles:
        pfile = (f'{args.pfile}_{tile}_{args.scan_suffix}' if tile and args.scan_suffix
                 else f'{args.pfile}_{tile}' if tile else args.pfile)
        infos = load_scan_infos(path, pfile)
        ts    = scan_times(infos, where=pfile)
        if ts is None:
            raise SystemExit(f'{pfile}: no usable Date= in the .info files, so '
                             f'there is no sequence to plot.')
        start, total = load_shrink_profile(path, pfile, len(infos),
                                           int(infos[0]['TOMO_N']))
        scans[tile] = (np.asarray(sorted(ts), dtype='float64'), start, total)
    out = args.path_out or '.'
    return scans, args.shrink_angle_ramp, os.path.join(out, 'shrink_timeline.png')


def acquisitions(scans):
    """One flat list of scans, in the order they ran — the unit of the model.

    Returns ``(acq, end)`` where each entry of ``acq`` is
    ``(tile, k, t0, t1, own, begin, inc)``: which distance of which tile, when it
    started and stopped, the shrink its own profile reports at its start
    (``own``, zero-based within its tile), the shrink the session had already
    accumulated when it started (``begin``, from :func:`shrink_sequence`), and
    what it adds (``inc``). ``end`` is where the session finishes.

    A scan stops when the next one starts (``scan_spans``), which is what makes
    the interleaving visible: left's distance 1 stops at 18:38 because right's
    starts there, not at 18:42 when left's own distance 2 does.
    """
    inc = {t: np.diff(np.concatenate([st, tot[None]]), axis=0)
           for t, (_ts, st, tot) in scans.items()}
    begins, end = shrink_sequence(scans)
    acq = [(tile, k, t0, t1, scans[tile][1][k], begins[tile][k], inc[tile][k])
           for t0, t1, tile, k in
           scan_spans({t: ts for t, (ts, _s, _o) in scans.items()})]
    return acq, end


def active_axis(acq):
    """Map wall-clock time to SCANNING time, with the idle taken out.

    The sample only deforms while it is being scanned, so the hours between
    scans are not part of its history at all: on a wall-clock axis they would
    draw as long flat stretches of s(t) that mean nothing, and squeeze the scans
    themselves into slivers. Dropping them leaves one continuous curve with no
    gaps and no flat parts, which is exactly the model.

    Returns ``(x, span, idle)``: ``x(t)`` the seconds of scanning elapsed before
    wall-clock ``t``, the total scanning time, and ``idle`` a list of
    ``(x_position, seconds)`` for each stretch that was removed — worth marking
    even though it takes no width, since the calendar gap is real.
    """
    bp, xs, idle = [acq[0][2]], [0.0], []
    for _tile, _k, t0, t1, *_ in acq:
        if t0 > bp[-1]:                       # nothing was running in here
            idle.append((xs[-1], t0 - bp[-1]))
            bp.append(t0)
            xs.append(xs[-1])
        bp.append(t1)
        xs.append(xs[-1] + (t1 - t0))
    bp, xs = np.array(bp), np.array(xs)
    return (lambda t: np.interp(t, bp, xs)), xs[-1], idle


def plot(scans, angle_ramp, out, demo=False, order=None, npix=3216):
    tiles = order or list(scans)
    acq, end = acquisitions(scans)
    cols = plt.get_cmap('tab10').colors
    x, span, idle = active_axis(acq)
    hr = 3600.0                       # the axis is drawn in hours of scanning

    fig = plt.figure(figsize=(13, 7.4))
    gs  = fig.add_gridspec(2, 1, height_ratios=[len(tiles) * 0.34, 1],
                           hspace=0.10,
                           left=0.08, right=0.93, top=0.90, bottom=0.17)
    ax_t = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1], sharex=ax_t)

    # ---- Gantt: one bar per (tile, distance) ------------------------------
    for tile, k, t0, t1, *_ in acq:
        r  = tiles.index(tile)
        xa, xb = x(t0) / hr, x(t1) / hr
        ax_t.barh(r, xb - xa, left=xa, height=0.55, color=cols[r % 10],
                  alpha=0.85, zorder=3)
        ax_t.text((xa + xb) / 2, r, f'{k + 1}', ha='center', va='center',
                  fontsize=7, color='w', zorder=4)
    ax_t.set_ylim(len(tiles) - 0.5, -0.7)
    ax_t.set_yticks(range(len(tiles)))
    ax_t.set_yticklabels(tiles, fontsize=9)
    ax_t.grid(axis='x', ls=':', alpha=0.35)
    ax_t.tick_params(labelbottom=False, left=False)

    # ---- shrink -----------------------------------------------------------
    # One thick segment per acquisition, ramping from what the session had
    # already accumulated to that plus this acquisition's own increment. Laid
    # end to end on the scanning axis they meet exactly, so the coloured
    # segments ARE the session curve — the colour only says who was scanning.
    for tile, k, t0, t1, own, begin, inc in acq:
        r  = tiles.index(tile)
        xx = np.array([x(t0), x(t1)]) / hr
        for c, ls in enumerate(['-', '--']):
            ax_b.plot(xx, [begin[c], begin[c] + inc[c]], color=cols[r % 10],
                      ls=ls, lw=4.5, alpha=0.5, zorder=4)
        ax_b.plot(xx[0], begin[0], 'o', ms=3.5, color=cols[r % 10], zorder=6)

    # The same tiles as their own shapp.mat / shrink_list.mat reports them: zero
    # at the tile's own first frame. Flat across a stretch scanned by another
    # tile, because that tile's profile has no idea it happened.
    for r, tile in enumerate(tiles):
        mine = [a for a in acq if a[0] == tile]
        xx = np.concatenate([[x(a[2]) / hr, x(a[3]) / hr] for a in mine])
        yy = np.concatenate([[a[4][0], a[4][0] + a[6][0]] for a in mine])
        ax_b.plot(xx, yy, color=cols[r % 10], ls=':', lw=1.2, alpha=0.7,
                  zorder=3)
        if mine[0][5][0] > 0:      # the offset steps15 adds at the first frame
            ax_b.annotate('', xy=(xx[0], mine[0][5][0]), xytext=(xx[0], 0),
                          zorder=7,
                          arrowprops=dict(arrowstyle='->', lw=1.0,
                                          color=cols[r % 10],
                                          shrinkA=0, shrinkB=0))

    # A thin black line over the top, so the accumulation reads as one curve.
    xs = np.concatenate([[x(a[2]) / hr, x(a[3]) / hr] for a in acq])
    for c, ls, lw in [(0, '-', 1.9), (1, '--', 1.3)]:
        ys = np.concatenate([[a[5][c], a[5][c] + a[6][c]] for a in acq])
        ax_b.plot(xs, ys, color='k', ls=ls, lw=lw, zorder=5)

    ax_b.grid(ls=':', alpha=0.35)
    ax_b.set_ylabel('shrink' + (' — INVENTED values' if demo else ''))
    # Shrink is a scale factor, so what it costs in pixels depends where you
    # are in the frame: eff_mag = norm_mag/(1+shrink) moves a feature r px from
    # the centre by about r*shrink. The worst case is the frame edge, r = n/2.
    ax_px = ax_b.secondary_yaxis(
        'right', functions=(lambda v: v * npix / 2, lambda p: p * 2 / npix))
    ax_px.set_ylabel(f'px at the frame edge (n={npix})', fontsize=9)
    ax_b.set_xlabel('scanning time (h)   —   the hours between scans are not on '
                    'this axis: nothing was scanning, so nothing deformed',
                    fontsize=9)
    # Left margin holds the wall-clock stamps written beside each tile's first bar.
    ax_b.set_xlim(-0.38, span / hr + 0.03)

    # The removed stretches take no width, but the calendar gap is real, so
    # mark where the long ones sat and how long they were. Short ones (a tile
    # change, a refill) are collapsed silently — annotating them would be noise.
    for xi, gap in idle:
        if gap < 1800:
            continue
        for ax in (ax_t, ax_b):
            ax.axvline(xi / hr, color='0.55', lw=1.1, ls=(0, (4, 3)), zorder=2)
        ax_t.text(xi / hr, -0.62, f' {gap/3600:.0f} h between scans ',
                  ha='center', va='top', fontsize=8, color='0.35',
                  bbox=dict(fc='w', ec='0.75', lw=0.6, pad=1.5))

    # Wall-clock stamp on each tile's first bar, so the collapsed axis can still
    # be read back against the .info dates. On the bars rather than on the axis:
    # left and right start four minutes apart, which no shared tick row can show.
    for r, tile in enumerate(tiles):
        t0 = scans[tile][0][0]
        ax_t.text(x(t0) / hr - 0.015, r,
                  time.strftime('%a %H:%M', time.localtime(t0)) + ' ',
                  ha='right', va='center', fontsize=7.5, color='0.35')

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
    ax_t.set_title('tiles, in mosaic order   (bar = one distance, numbered; '
                   'a bar ends where the next scan starts)',
                   loc='left', fontsize=9)
    fig.suptitle(
        'Acquisition sequence and accumulated shrink'
        + ('   —   SCHEMATIC: real timestamps, invented shrink values'
           if demo else ''),
        fontsize=12)
    fig.text(0.5, 0.035,
             'shrink ramps linearly across each scan (shrink_angle_ramp=true); '
             'every (tile, distance) adds its own increment on top of the ones '
             'before it — a plain cumulative sum in the order they ran',
             ha='center', fontsize=8, color='0.35')
    fig.savefig(out, dpi=140)
    print(f'wrote {out}')

    # The offset steps15 adds is per distance, not per tile: for an interleaved
    # pair it grows across the tile's own distances, because the partner keeps
    # slipping increments in between.
    print(f'\noffsets steps15 adds (begin minus the tile\'s own profile); px is '
          f'what the shrink costs at the edge of an n={npix} frame, i.e. '
          f'shrink*n/2:')
    for tile in sorted(tiles, key=lambda t: scans[t][0][0]):
        off = np.stack([a[5] - a[4] for a in acq if a[0] == tile])
        # Constant only to rounding for a tile taken in one go: the running sum
        # reaches its offset from a non-zero base, the tile's own profile from
        # zero, so the two disagree in the last bits.
        rng = ('' if np.abs(off - off[0]).max() <= 1e-6 * np.abs(off).max() else
               f' .. v={off[-1, 0]:+.6f} h={off[-1, 1]:+.6f}  (grows: interleaved)')
        print(f'  {tile:<10s} '
              f'{time.strftime("%a %H:%M", time.localtime(scans[tile][0][0]))}'
              f'   v={off[0, 0]:+.6f} h={off[0, 1]:+.6f}{rng}'
              f'   ({off[0, 0] * npix / 2:+.2f}, {off[0, 1] * npix / 2:+.2f}) px')
    print(f'  end of session       v={end[0]:+.6f} h={end[1]:+.6f}'
          f'   ({end[0] * npix / 2:+.2f}, {end[1] * npix / 2:+.2f}) px')
    print(f'  {span / hr:.1f} h of scanning, over a '
          f'{(acq[-1][3] - acq[0][2]) / hr:.1f} h session')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config', nargs='?', help='steps15 config file')
    p.add_argument('-o', '--out', default=None, help='output PNG')
    p.add_argument('--demo', action='store_true',
                   help='hardcoded YY037A dates with invented shrink values')
    p.add_argument('--npix', type=int, default=3216,
                   help='frame width in px, for the secondary "px at the frame '
                        'edge" axis (default 3216)')
    a = p.parse_args()

    if a.config and not a.demo:
        try:
            scans, ramp_on, out = real_scans(a.config)
        except (FileNotFoundError, OSError) as e:
            print(f'{e}\nfalling back to --demo', file=sys.stderr)
            scans, ramp_on, out = demo_scans(), True, 'shrink_timeline.png'
            a.demo = True
        # Keep the config's mosaic order for the rows.
        from holotomocupy.config import parse_args_steps15
        order = [t for t in parse_args_steps15(a.config).tiles if t in scans]
    else:
        scans, ramp_on, out = demo_scans(), True, 'shrink_timeline.png'
        a.demo = True
        order = ['farright', 'right', 'center', 'left', 'farleft']

    if not ramp_on:
        print('note: the config has shrink_angle_ramp=false, so steps15 holds '
              'each plane flat at its mid value; the plot still draws the ramp.',
              file=sys.stderr)
    plot(scans, ramp_on, a.out or out, demo=a.demo, order=order or None,
         npix=a.npix)


if __name__ == '__main__':
    main()
