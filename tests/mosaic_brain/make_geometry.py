#!/usr/bin/env python
"""Write the text files that define the synthetic mosaic scan, all in shift_dir:

  shifts/tile_offsets.txt  where each tile sits on the mosaic
  shifts/<tile>.txt        the per-angle sample shift of each tile

Both are in object pixels on the FINEST grid (bin 0, 100 nm voxels), the same
units and sign convention as /exchange/cshifts_final in the real YY037A files.
gen_data.py reads them back and writes

    cshifts_final[itheta, tile*ndist + k] = tile_offset[tile] + shift[itheta, k]

Run:
    python make_geometry.py config_gen.conf            # synthesize the shifts
    python make_geometry.py config_gen.conf --from-h5 DIR
                                                       # take real cshifts_final
                                                       # from the YY037A tile
                                                       # files in DIR instead

Edit either file by hand afterwards; gen_data.py never regenerates them.
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'src'))
from holotomocupy.config import parse_args_gen   # noqa: E402

def build_tiles(args):
    """[(name, v, h)] for the whole ntile_v x ntile_h grid, row-major.

    Nominal placement: a regular grid of tile_step_v x tile_step_h object px
    centred on the object grid, generated from the tile index alone.

    The name is "{row}_{col}", both zero-based, ordered as the tile lands in the
    composed mosaic: row 0 on top, column 0 on the left.  The offsets are sample
    SHIFTS, so they run opposite to the object-grid axes -- a tile shifted by +h
    appears at smaller x -- hence both steps count down from the centre.
    """
    v_rows = ((args.ntile_v - 1) / 2 - np.arange(args.ntile_v)) * args.tile_step_v
    h_cols = ((args.ntile_h - 1) / 2 - np.arange(args.ntile_h)) * args.tile_step_h
    return [(f'{r}_{c}', float(v_rows[r]), float(h_cols[c]))
            for r in range(args.ntile_v) for c in range(args.ntile_h)]


def synth_shifts(args, ntiles, ndist, norm_mag):
    """[ntiles, ntheta, ndist, 2] synthetic sample shift, finest-grid object px.

    Encoder jitter only: uniform +-shift_rand_px DETECTOR px, independent per
    angle and per distance, divided by the distance's normalised magnification
    to reach object px.  This is the random_shifts part of the real
    cshifts_final.
    """
    rng = np.random.default_rng(0)
    nt = args.ntheta
    out = np.zeros([ntiles, nt, ndist, 2], dtype='float32')
    if args.shift_rand_px > 0:
        for t in range(ntiles):
            jit = rng.uniform(-args.shift_rand_px, args.shift_rand_px, size=(nt, ndist, 2))
            out[t] = (jit / norm_mag[None, :, None]).astype('float32')
    return out


def shifts_from_h5(args, tiles, ndist, h5dir):
    """Real /exchange/cshifts_final, one tile file per mosaic column.

    The tile .h5 files in h5dir are used in sorted order, one per mosaic column
    and cycled if there are fewer files than columns.  Tiles in the same column
    but different rows reuse that column's shifts with a fresh per-row circular
    rotation in angle, so the rows are not identical.
    """
    import h5py
    nt = args.ntheta
    out = np.zeros([len(tiles), nt, ndist, 2], dtype='float32')
    files = sorted(f for f in os.listdir(h5dir)
                   if f.endswith('.h5') and '_mosaic' not in f)
    if not files:
        raise SystemExit(f'no tile .h5 files in {h5dir}')
    cache = {}
    for i, (name, _, _) in enumerate(tiles):
        col = files[(i % args.ntile_h) % len(files)]
        if col not in cache:
            with h5py.File(os.path.join(h5dir, col), 'r') as f:
                cs = f['/exchange/cshifts_final'][:].astype('float32')
            print(f'  column {i % args.ntile_h}: cshifts_final {cs.shape} from {col}')
            cache[col] = cs
        cs = cache[col]
        if cs.shape[0] < nt:
            raise SystemExit(f'{col}: file has {cs.shape[0]} angles, config asks for {nt}')
        if cs.shape[1] < ndist:
            raise SystemExit(f'{col}: file has {cs.shape[1]} distances, config asks for {ndist}')
        out[i] = np.roll(cs[:nt, :ndist], i * (nt // max(len(tiles), 1)), axis=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--from-h5', metavar='DIR', default=None,
                    help='directory of real YY037A tile .h5 files to take '
                         'cshifts_final from instead of synthesizing it')
    opt = ap.parse_args()

    args  = parse_args_gen(opt.config)
    ndist = len(args.z1)
    tiles = build_tiles(args)

    mag      = args.focustodetectordistance / np.array(args.z1)
    norm_mag = mag / mag[0]

    # --- shifts/tile_offsets.txt --------------------------------------------
    os.makedirs(args.shift_dir, exist_ok=True)
    with open(args.tile_file, 'w') as f:
        f.write('# Tile placement for the synthetic YY037A-like mosaic.\n')
        f.write('# Object px on the FINEST grid (bin 0, 100.000 nm voxels).\n')
        f.write(f'# {args.ntile_v} rows x {args.ntile_h} columns, row-major; the flat\n')
        f.write(f'# distance index used in the HDF5 file is tile*{ndist} + k.\n')
        f.write(f'# Nominal grid, {args.tile_step_v:g} x {args.tile_step_h:g} px steps.\n')
        f.write('# name is "{row}_{col}", zero-based, as the tile lands in the\n')
        f.write('# composed mosaic: row 0 on top, column 0 on the left.\n')
        f.write('#\n# index  name       v            h\n')
        for i, (name, v, h) in enumerate(tiles):
            f.write(f'{i:5d}  {name:<8s} {v:12.4f} {h:12.4f}\n')
    print(f'wrote {args.tile_file}  ({len(tiles)} tiles)')

    # --- shifts/<tile>.txt --------------------------------------------------
    if opt.from_h5:
        shifts = shifts_from_h5(args, tiles, ndist, opt.from_h5)
        src = f'real cshifts_final from {opt.from_h5}'
    else:
        shifts = synth_shifts(args, len(tiles), ndist, norm_mag)
        src = f'synthetic: {args.shift_rand_px} det px encoder jitter'

    cols = ' '.join(f'v{k} h{k}' for k in range(ndist))
    for i, (name, _, _) in enumerate(tiles):
        path = os.path.join(args.shift_dir, f'{name}.txt')
        flat = shifts[i].reshape(args.ntheta, ndist * 2)
        np.savetxt(path, flat, fmt='%12.5f',
                   header=(f'Per-angle sample shift of tile {name}.\n'
                           f'Object px on the FINEST grid (bin 0, 100.000 nm voxels),\n'
                           f'same units and sign as /exchange/cshifts_final.\n'
                           f'{args.ntheta} rows x {ndist*2} cols: {cols}\n'
                           f'source: {src}'))
    print(f'wrote {len(tiles)} files in {args.shift_dir}/  ({src})')

    a = np.abs(shifts)
    print(f'shift magnitude (finest px):  v max {a[...,0].max():7.2f}  '
          f'h max {a[...,1].max():7.2f}')

    # --- placement sanity check --------------------------------------------
    # The shift kernel samples the object at  mag0*(i - (n-1)/2) - r + (npsi-1)/2
    # with mag0 = 1/norm_mag; every tile must stay inside the object grid.
    half = 0.5 * (args.ndet - 1) / norm_mag.min()
    off  = np.array([[v, h] for _, v, h in tiles], dtype='float32')
    reach_v = np.abs(off[:, None, None, 0] + shifts[..., 0]).max() + half
    reach_h = np.abs(off[:, None, None, 1] + shifts[..., 1]).max() + half
    print(f'tile reach: v {reach_v:8.1f} / {(args.nzobj-1)/2:8.1f}   '
          f'h {reach_h:8.1f} / {(args.nobj-1)/2:8.1f}   (finest px, half-extent)')
    if reach_v > (args.nzobj - 1) / 2 or reach_h > (args.nobj - 1) / 2:
        print('WARNING: tiles fall outside the object grid — increase nzobj / nobj.')


if __name__ == '__main__':
    main()
