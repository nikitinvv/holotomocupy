import h5py
import numpy as np
import cupy as cp

def read_acquisition_pars(args):
    """Read data acquisition parameters for holotomography"""
    
    with h5py.File(args.in_file, 'r') as fid:
        args.detector_pixelsize = fid['/exchange/detector_pixelsize'][0]    
        args.focustodetectordistance = fid['/exchange/focusdetectordistance'][0]    
        args.z1 = fid['/exchange/z1'][:args.ndist] 
        args.energy = fid['/exchange/energy'][0] 
        args.focustodetectordistance = fid['/exchange/focusdetectordistance'][0] 
        
        ntheta0 = len(fid[f'/exchange/theta'])
        args.ids = np.arange(args.start_theta, ntheta0, ntheta0 / args.ntheta).astype('int')
        args.theta = -fid['/exchange/theta'][args.ids, 0] / 180 * np.pi

def read_obj(args,st,end):    
    """Read initial guess for the object"""
    
    bin = args.bin
    nzobj = args.nzobj// 2**bin
    nobj = args.nobj// 2**bin
    
    with h5py.File(args.in_file) as fid:        
        obj = fid[f'/exchange/obj_init_re{args.paganin}_{bin}']
        nzobj0,nobj0 = obj.shape[:2]
        stz,endz = (nzobj0//2-nzobj//2),(nzobj0//2+nzobj//2)
        stx,endx = (nobj0//2-nobj//2),(nobj0//2+nobj//2)
        obj = obj[stz+st:stz+end, stx:endx,stx:endx].astype(args.obj_dtype)
        
    return obj


def read_pos(args,st,end):
    """Read initial guess for positions"""
    
    bin = args.bin
    
    with h5py.File(args.in_file) as fid:        
        # positions initial guess        
        pos = (fid[f'/exchange/cshifts_final'][args.ids[st:end],:args.ndist]).astype('float32')        
        pos /= 2**bin
        s = args.rotation_center_shift
        for k in range(bin):
            s = (s - 0.5) / 2
        pos[..., 1] += s
        
    return pos

def read_prb(args):
    """"Read initial guess for the probe"""
    
    bin = args.bin
    nz = args.nz // 2**bin
    n = args.n // 2**bin    
    prb = cp.ones([args.ndist, nz, n], dtype='complex64')    
    return prb    


def read_data(args,st_theta,end_theta):
    """Read data"""

    nz = args.nz // 2**args.bin
    n = args.n // 2**args.bin
    data = np.empty([end_theta-st_theta, args.ndist, nz, n], dtype='float32')
    with h5py.File(args.in_file) as fid:
        for k in range(args.ndist):
            [nz0,n0] = fid[f'/exchange/pdata{k}_{args.bin}'].shape[1:]
            st,end = nz0//2-nz//2, nz0//2+nz//2
            ## note we take square root right after reading
            data[:, k] = np.sqrt(fid[f'/exchange/pdata{k}_{args.bin}'][args.ids[st_theta:end_theta],st:end])
        
    return data


def read_ref(args):
    """Read reference"""
    
    nz = args.nz // 2**args.bin
    n = args.n // 2**args.bin
    with h5py.File(args.in_file) as fid:
        [nz0,n0] = fid[f'/exchange/pref_{args.bin}'].shape[1:]
        st,end = nz0//2-nz//2, nz0//2+nz//2
        ## note we take square root right after reading
        ref = cp.sqrt(cp.array(fid[f'/exchange/pref_{args.bin}'][:args.ndist,st:end]))    

    return ref