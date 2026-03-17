from holotomocupy.utils import *
import numpy as np
a = read_tiff(f'/data2/vnikitin/alcf/brain/Y350a_dist1234/run3/rec_objz/0864.tiff')
b = read_tiff(f'/data2/vnikitin/brain/20251115/Y350a_HT_20nm_8dist_rec1234_/naburec/results/20251122_190427//correct_3D_001024.tiff')
print(a.shape)
print(b.shape)

crop=a.shape[0]//2-b.shape[0]//2
a = a[crop:-crop,crop:-crop]


import h5py

with h5py.File('/data2/vnikitin/alcf/brain/Y350a_dist1234/rec_levels_17p23_p0_re/checkpoint_1536.h5','r') as fid:
    a = fid['obj_re']
    a = a[a.shape[0]//2-1024:a.shape[0]//2+1024,crop:-crop,crop:-crop]
    write_tiff(a,f'/data2/vnikitin/alcf/brain/Y350a_dist1234/denoising/03132026/obj_real/delta_even')


with h5py.File('/data2/vnikitin/alcf/brain/Y350a_dist1234/rec_levels_17p23_p1_re/checkpoint_1536.h5','r') as fid:
    a = fid['obj_re']
    a = a[a.shape[0]//2-1024:a.shape[0]//2+1024,crop:-crop,crop:-crop]
    write_tiff(a,f'/data2/vnikitin/alcf/brain/Y350a_dist1234/denoising/03132026/obj_real/delta_odd')

with h5py.File('/data2/vnikitin/alcf/brain/Y350a_dist1234/rec_levels_17p23_re/checkpoint_1536.h5','r') as fid:
    a = fid['obj_re']
    a = a[a.shape[0]//2-1024:a.shape[0]//2+1024,crop:-crop,crop:-crop]
    write_tiff(a,f'/data2/vnikitin/alcf/brain/Y350a_dist1234/denoising/03132026/obj_real/delta_all')



with h5py.File('/data2/vnikitin/alcf/brain/Y350a_dist1234/rec_levels_17p23_p0/checkpoint_1536.h5','r') as fid:
    a = fid['obj_re']
    a = a[a.shape[0]//2-1024:a.shape[0]//2+1024,crop:-crop,crop:-crop]
    write_tiff(a,f'/data2/vnikitin/alcf/brain/Y350a_dist1234/denoising/03132026/obj_complex/delta_even')


with h5py.File('/data2/vnikitin/alcf/brain/Y350a_dist1234/rec_levels_17p23_p1/checkpoint_1536.h5','r') as fid:
    a = fid['obj_re']
    a = a[a.shape[0]//2-1024:a.shape[0]//2+1024,crop:-crop,crop:-crop]
    write_tiff(a,f'/data2/vnikitin/alcf/brain/Y350a_dist1234/denoising/03132026/obj_complex/delta_odd')

with h5py.File('/data2/vnikitin/alcf/brain/Y350a_dist1234/rec_levels_17p23/checkpoint_1536.h5','r') as fid:
    a = fid['obj_re']
    a = a[a.shape[0]//2-1024:a.shape[0]//2+1024,crop:-crop,crop:-crop]
    write_tiff(a,f'/data2/vnikitin/alcf/brain/Y350a_dist1234/denoising/03132026/obj_complex/delta_all')


with h5py.File('/data2/vnikitin/alcf/brain/Y350a_dist1234/rec_levels_17p23_p0/checkpoint_1536.h5','r') as fid:
    a = fid['obj_im']
    a = a[a.shape[0]//2-1024:a.shape[0]//2+1024,crop:-crop,crop:-crop]
    write_tiff(a,f'/data2/vnikitin/alcf/brain/Y350a_dist1234/denoising/03132026/obj_complex/beta_even')


with h5py.File('/data2/vnikitin/alcf/brain/Y350a_dist1234/rec_levels_17p23_p1/checkpoint_1536.h5','r') as fid:
    a = fid['obj_im']
    a = a[a.shape[0]//2-1024:a.shape[0]//2+1024,crop:-crop,crop:-crop]
    write_tiff(a,f'/data2/vnikitin/alcf/brain/Y350a_dist1234/denoising/03132026/obj_complex/beta_odd')


with h5py.File('/data2/vnikitin/alcf/brain/Y350a_dist1234/rec_levels_17p23/checkpoint_1536.h5','r') as fid:
    a = fid['obj_im']
    a = a[a.shape[0]//2-1024:a.shape[0]//2+1024,crop:-crop,crop:-crop]
    write_tiff(a,f'/data2/vnikitin/alcf/brain/Y350a_dist1234/denoising/03132026/obj_complex/beta_all')

