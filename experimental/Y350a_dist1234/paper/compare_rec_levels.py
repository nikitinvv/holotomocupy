import h5py
import numpy as np
from scipy.ndimage import laplace, median_filter

base = '/data2/vnikitin/alcf/brain/Y350a_dist1234'
levels = [0, 1, 2, 3, 4, 5, 6]
files = [f'{base}/rec_levels_17p2{x}/checkpoint_1536.h5' for x in levels]

slices_re = []
slices_im = []
sharpness_list = []

for fpath in files:
    print(fpath)
    with h5py.File(fpath, 'r') as fid:
        for key, out in [('obj_re', slices_re), ('obj_im', slices_im)]:
            d = fid[key]
            mid = d.shape[0] // 2
            cy, cx = d.shape[1] // 2, d.shape[2] // 2
            vol = d[mid, cy-512:cy+512, cx-512:cx+512]
            out.append(median_filter(vol, size=3))

        d = fid['obj_re']
        mid = d.shape[0] // 2
        cy, cx = d.shape[1] // 2, d.shape[2] // 2
        subvol = d[mid-128:mid+128, cy-128:cy+128, cx-128:cx+128].astype('float32')
        sharpness_list.append(np.var(laplace(subvol)))

import tifffile

for i, (sl_re, sl_im) in enumerate(zip(slices_re, slices_im)):
    tifffile.imwrite(f'{base}/compare_rec_levels_re_{i}.tiff', sl_re.astype('float32'))
    tifffile.imwrite(f'{base}/compare_rec_levels_im_{i}.tiff', sl_im.astype('float32'))
    print(f'Level {i}: sharpness={sharpness_list[i]:.6f}')
