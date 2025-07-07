#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cupy as cp
import h5py
import matplotlib.pyplot as plt


# # Init data sizes and parametes of the PXM of ID16A

# In[2]:


step = 1


# In[3]:


path = f'/data/vnikitin/ESRF/ID16A/brain/20250604/Y350a/'
pfile = f'Y350a_HT_nobin_020nm'
path_out = f'/data/vnikitin/ESRF/ID16A/brain_rec/20250604/Y350a/'

with h5py.File(f'{path_out}/{pfile}.h5') as fid:
    detector_pixelsize = fid['/exchange/detector_pixelsize'][0]    
    focusToDetectorDistance = fid['/exchange/focusdetectordistance'][0]    
    z1 = fid['/exchange/z1'][:]        
    theta = fid['/exchange/theta'][::step]
    shifts = fid['/exchange/shifts'][::step]
    attrs = fid['/exchange/attrs'][::step]
    pos_shifts = fid['/exchange/pos_shifts'][::step]*1e-6
    shape = fid['/exchange/data0'][::step].shape
    shape_ref = fid['/exchange/data_white_start0'].shape
    shape_dark = fid['/exchange/data_dark0'].shape
    #pos_shifts-=pos_shifts[0]


# In[4]:


ndist=4
ntheta,n = shape[:2]
ndark = shape_dark[0]
nref = shape_ref[0]


# In[5]:


print(ndist,ntheta,n)
print(nref,ndark)


# In[6]:


energy = 33.35  # [keV] xray energy
wavelength = 1.2398419840550367e-09/energy  # [m] wave length
z2 = focusToDetectorDistance-z1
distances = (z1*z2)/focusToDetectorDistance
magnifications = focusToDetectorDistance/z1
norm_magnifications = magnifications/magnifications[0]
show = True


# In[7]:


data00 = np.empty([ntheta,ndist,n,n],dtype='float32')
ref00 = np.empty([nref,ndist,n,n],dtype='float32')
ref01 = np.empty([nref,ndist,n,n],dtype='float32')
dark00 = np.empty([ndark,ndist,n,n],dtype='float32')


# In[8]:


with h5py.File(f'{path_out}/{pfile}.h5') as fid:
    for k in range(4):
        print(k)
        data00[:,k] = fid[f'/exchange/data{k}'][::step]
        ref00[:,k] = fid[f'/exchange/data_white_start{k}'][:]
        ref01[:,k] = fid[f'/exchange/data_white_end{k}'][:]
        dark00[:,k] = fid[f'/exchange/data_dark{k}'][:]
        


# ## Read data

# In[9]:


import cupyx.scipy.ndimage as ndimage
def remove_outliers(data, dezinger, dezinger_threshold):    
    res = data.copy()
    w = [dezinger,dezinger]
    for k in range(data.shape[0]):
        data0 = cp.array(data[k])
        fdata = ndimage.median_filter(data0, w)
        print(np.sum(np.abs(data0-fdata)>fdata*dezinger_threshold))
        res[k] = np.where(np.abs(data0-fdata)>fdata*dezinger_threshold, fdata, data0).get()
    return res


# In[ ]:


data = data00
ref = ref00
dark = dark00

dark = np.mean(dark,axis=0)
ref = np.mean(ref,axis=0)
print('subtract')
data-=dark
ref-=dark

print('check')
data[data<0]=0
ref[ref<0]=0

# data[:,:,1320//3:1320//3+25//3,890//3:890//3+25//3] = data[:,:,1280//3:1280//3+25//3,890//3:890//3+25//3]
# ref[:,1320//3:1320//3+25//3,890//3:890//3+25//3] = ref[:,1280//3:1280//3+25//3,890//3:890//3+25//3]
# data[:,:,470-4:470+4,285-4:285+4] = data[:,:,470-4-10:470+4-10,285-4:285+4]
# ref[:,470-4:470+4,285-4:285+4] = ref[:,470-4-10:470+4-10,285-4:285+4]
radius = 3
threshold = 0.8
ref[:] = remove_outliers(ref[:], radius, threshold)     
for k in range(ndist):    
    data[:,k] = remove_outliers(data[:,k], radius, threshold)



# In[13]:


# mm = np.mean(data,axis=(2,3))

# plt.plot(mm.swapaxes(0,1).flatten()/mm[0,0],label='average')
# plt.plot(attrs[...,0].swapaxes(0,1).flatten()/attrs[0,0,0],label='1')
# plt.plot(attrs[...,1].swapaxes(0,1).flatten()/attrs[0,0,1],label='2')
# plt.plot(attrs[...,2].swapaxes(0,1).flatten()/attrs[0,0,2],label='3')
# plt.legend()
# plt.show()


# In[14]:
mm = np.mean(data,axis=(2,3))
mmr = np.mean(ref,axis=(1,2))

print('divide')
data/=np.mean(ref[0])
ref/=np.mean(ref[0])


# In[15]:


c = mm/mmr[0]#[...,1]/attrs[0,0,1]
cr = mmr/mmr[0]#[...,1]/attrs[0,0,1]
for k in range(ndist):
    data[:,k]/=c[:,k,np.newaxis,np.newaxis]
    ref[k]/=cr[k]    

mm = np.mean(data,axis=(2,3))
mmr = np.mean(ref,axis=(1,2))
print(mm,mmr) 


# In[16]:
with h5py.File(f'{path_out}/{pfile}_corr.h5','w') as fid:
    fid.create_dataset(f'/exchange/ref',data = ref)
    for k in range(4):
        fid.create_dataset(f'/exchange/data{k}',data = data[:,k])    
 


# In[ ]:




