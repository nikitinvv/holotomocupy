#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py 
import dxchange
import sys
sys.path.insert(0, '..')
from utils import *

n = 4096  # object size in each dimension
detector_pixelsize = 1.4760147601476e-6
energy = 33.35  # [keV] xray energy
wavelength = 1.2398419840550367e-09/energy  # [m] wave length
focusToDetectorDistance = 1.28  # [m]
sx0 = 1.28e-3
# z1 = np.array([4.58399e-3,4.76499e-3,5.48800e-3,6.98950e-3])-sx0
z1 = np.array([18.746e-3,19.495e-3,22.492e-3,28.716e-3])-sx0
z2 = focusToDetectorDistance-z1
distances = (z1*z2)/focusToDetectorDistance
magnifications = focusToDetectorDistance/z1
norm_magnifications = magnifications/magnifications[0]
voxelsize = np.abs(detector_pixelsize/magnifications[0]*4096/n)  # object voxel size
voxelsizes = np.abs(detector_pixelsize/magnifications*4096/n)  # object voxel size

show = True
path = f'/data/vnikitin/ESRF/ID16A/brain/20250604/Y350a/'
pfile = f'Y350a_HT_nobin_020nm_'
path_out = f'/data/vnikitin/ESRF/ID16A/brain_rec/20250604/Y350a/'
print(f'{voxelsize=}')
ntheta=4000
ndist=4
st=0


# In[ ]:


def find_angle(filename):    
    with open(filename, 'r',encoding="latin-1") as file:
        for line in file:
            if "motor_pos" in line:
                print(line.split())
                return float(line.split()[3])        

def find_pos_shifts(filename):    
    with open(filename, 'r',encoding="latin-1") as file:
        for line in file:
            if "motor_pos" in line:
                print(line.split())
                return np.array([float(line.split()[16]),float(line.split()[17])])                
data=[]
data_white0=[]
data_white1=[]
data_dark=[]
os.system(f'mkdir -p {path_out}')
with  h5py.File(f'{path_out}/{pfile}.h5','w') as fid:
    for k in range(4):
        data.append(fid.create_dataset(f'/exchange/data{k}',shape=(ntheta,4096,4096),dtype='uint16'))    
        data_white0.append(fid.create_dataset(f'/exchange/data_white_start{k}',shape=(20,4096,4096),dtype='uint16'))
        data_white1.append(fid.create_dataset(f'/exchange/data_white_end{k}',shape=(20,4096,4096),dtype='uint16'))    
        data_dark.append(fid.create_dataset(f'/exchange/data_dark{k}',shape=(20,4096,4096),dtype='uint16'))
    
    theta = fid.create_dataset('/exchange/theta',shape=(ntheta,4),dtype='float32')
    shifts = fid.create_dataset('/exchange/shifts',shape=(ntheta,4,2),dtype='float32')    
    attrs = fid.create_dataset('/exchange/attrs',shape=(ntheta,4,3),dtype='float32')    
    pos_shifts = fid.create_dataset('/exchange/pos_shifts',shape=(ntheta,4,2),dtype='float32')
    
    dvoxelsize = fid.create_dataset('/exchange/voxelsize',shape=(4,),dtype='float32')
    dz1 = fid.create_dataset('/exchange/z1',shape=(4,),dtype='float32')
    ddetector_pixelsize = fid.create_dataset('/exchange/detector_pixelsize',shape=(1,),dtype='float32')
    dfocusdetectordistance = fid.create_dataset('/exchange/focusdetectordistance',shape=(1,),dtype='float32')
    
    dvoxelsize[:]=voxelsizes
    dz1[:]=z1
    ddetector_pixelsize[0]=detector_pixelsize
    dfocusdetectordistance[0]=focusToDetectorDistance

    for k in range(4):
        shifts[:,k] = np.loadtxt(f'{path}/{pfile}_{k+1}_/correct.txt')[:ntheta]
        attrs[:,k] = np.loadtxt(f'{path}/{pfile}_{k+1}_/attributes.txt')[:ntheta]        

        for id in range(data_white0[k].shape[0]):
            fname = f'{path}/{pfile}_{k+1}_/ref{id:04}_0000.edf'
            data_white0[k][id] = dxchange.read_edf(fname)[0]

        for id in range(data_white1[k].shape[0]):
            fname = f'{path}/{pfile}_{k+1}_/ref{id:04}_{ntheta:04}.edf'
            data_white1[k][id] = dxchange.read_edf(fname)[0]

        for id in range(data_dark[k].shape[0]):
            fname = f'{path}/{pfile}_{k+1}_/darkend{id:04}.edf'
            data_dark[k][id] = dxchange.read_edf(fname)[0]

        for id in range(data[k].shape[0]):        
            fname = f'{path}/{pfile}_{k+1}_/{pfile}_{k+1}_{id:04}.edf'            
            ang = find_angle(fname)                    
            pshifts = find_pos_shifts(fname)
            print(fname)
            print(ang)  
            
            data[k][id] = dxchange.read_edf(fname)[0]
            theta[id,k] = ang
            pos_shifts[id,k] = pshifts
            ch = pos_shifts[id,k]-pos_shifts[0,0]
            print(shifts[id,k],ch[0]/voxelsize/1e6,-ch[-1]/voxelsize/1e6)  

