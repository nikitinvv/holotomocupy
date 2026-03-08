import h5py
import numpy as np

energies = ['16p9','16p95','17p0','17p05','17p1','17p15','17p2','17p25','17p3','17p35','17p4']
dd = np.empty([len(energies),1632,1632],dtype='float32')
for i,k in enumerate(energies):
    with h5py.File(f'/eagle/APS_IRI/vnikitin/test{k}/checkpoint_0192.h5','r') as fid:
        t = fid['obj_re']
        dd[i]=t[t.shape[0]//2]
        
import numpy as np
from scipy.ndimage import median_filter


# 2D median filter with a 3x3 window
fd = median_filter(dd, size=(3, 3),axes=(1,2))
with h5py.File(f'/eagle/APS_IRI/vnikitin/test_merged_re.h5','w') as fid:
    fid.create_dataset('d',data=dd)
    fid.create_dataset('fd',data=fd)
        
    
    
    
for i,k in enumerate(energies):
    with h5py.File(f'/eagle/APS_IRI/vnikitin/test{k}/checkpoint_0192.h5','r') as fid:
        t = fid['obj_im']
        dd[i]=t[t.shape[0]//2]
    
import numpy as np
from scipy.ndimage import median_filter


# 2D median filter with a 3x3 window
fd = median_filter(dd, size=(3, 3),axes=(1,2))
with h5py.File(f'/eagle/APS_IRI/vnikitin/test_merged_im.h5','w') as fid:
    fid.create_dataset('d',data=dd)
    fid.create_dataset('fd',data=fd)
        
    