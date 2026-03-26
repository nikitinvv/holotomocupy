from holotomocupy.utils import *
import numpy as np
import h5py

# ── Parameters ────────────────────────────────────────────────────────────────
crop    = 24
in_base = '/data2/vnikitin/alcf/brain/Y350a_dist1234'
out_dir = '/data2/vnikitin/alcf/brain/Y350a_dist1234/denoising/03132026/obj_complex'

pairs = [
    ('rec_levels_17p23_p0', 'even'),
    ('rec_levels_17p23_p1', 'odd'),
]

# ── Extract and write ─────────────────────────────────────────────────────────
for folder, tag in pairs:
    ckpt = f'{in_base}/{folder}/checkpoint_1536.h5'
    with h5py.File(ckpt, 'r') as fid:
        for key, name in [('obj_re', 'delta'), ('obj_im', 'beta')]:
            a = fid[key]
            a = a[a.shape[0]//2-1024:a.shape[0]//2+1024, crop:-crop, crop:-crop]
            write_tiff(a, f'{out_dir}/{name}_{tag}')
