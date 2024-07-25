import h5py
import dxchange
import numpy as np
import re
import codecs

path = '/data/vnikitin/ESRF/ID16B/009/'
fname = '032_009_100nm'
n = 2048
ndist = 1
ntheta = 900
nref0 = 20
nref1 = 20
ndark = 1
energy = 29.63
detector_pixelsize = 0.65e-6
z12 =  0.704433
sx0h = 0.8525605999567023e-3; #1.077165773192669 for 75nm.
sx0v = 0.80170811624758109e-3; #1.110243284221266 for 75nm. 

data = np.zeros([ntheta,n,n],dtype='float32')
ref0 = np.zeros([nref0,n,n],dtype='float32')
ref1 = np.zeros([nref1,n,n],dtype='float32')
dark = np.zeros([1,n,n],dtype='float32')

fout = h5py.File(f'{path}{fname}.h5','w')
theta = np.linspace(360,ntheta).astype('float32')

z1 = np.zeros([ndist],dtype='float32')
for k in range(ndist):
    with codecs.open(f'{path}{fname}_{k+1}_/{fname}_{k+1}_0000.edf', 'r', encoding='utf-8',errors='ignore') as f:
        for num, line in enumerate(f, 1):
            if "motor_mne" in line:            
                keys = line.split(' ')[4:-1]   
            if "motor_pos" in line:            
                vals = line.split(' ')[4:-1]            
                break
    mdict = {}
    for i,v in enumerate(keys):
        mdict[v]=vals[i]
    z1[k] = float(mdict['sx'])*1e-3
print(z1)

fout.create_dataset(f'/instrument/energy',data=energy)
fout.create_dataset(f'/instrument/z1',data=z1)
fout.create_dataset(f'/instrument/sx0h',data=sx0h)
fout.create_dataset(f'/instrument/sx0v',data=sx0v)
fout.create_dataset(f'/instrument/detector_pixelsize',data=detector_pixelsize)
fout.create_dataset(f'/instrument/z12',data=z12)


for k in range(ndist):        
    for j in range(ntheta):
        file = f'{path}{fname}_{k+1}_/{fname}_{k+1}_{j:04}.edf'
        data[j] = dxchange.read_edf(file)
        
    for j in range(nref0):
        file = f'{path}{fname}_{k+1}_/ref{j:04}_0000.edf'
        ref0[j] = dxchange.read_edf(file)
    
    for j in range(nref1):
        file = f'{path}{fname}_{k+1}_/ref{j:04}_{ntheta:04}.edf'
        ref1[j] = dxchange.read_edf(file)
        
    file = f'{path}{fname}_{k+1}_/dark.edf'
    dark[0] = dxchange.read_edf(file)

    fout.create_dataset(f'/exchange/data{k}',data=data)
    fout.create_dataset(f'/exchange/data{k}_white0',data=ref0)
    fout.create_dataset(f'/exchange/data{k}_white1',data=ref1)
    fout.create_dataset(f'/exchange/data{k}_dark',data=dark)
    fout.create_dataset(f'/exchange/theta',data=theta)




