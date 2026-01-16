import numpy as np
import cupy as cp
import h5py
import sys
from types import SimpleNamespace

from holotomocupy.rec import Rec
from holotomocupy.config import parse_args
from holotomocupy.utils import *

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
    
def read_initial_guess(args):
    """Read initial guess for the object, probe and positions with a given binning level"""

    bin = args.bin
    nz = args.nz // 2**bin
    n = args.n // 2**bin
    nzobj = args.nzobj// 2**bin
    nobj = args.nobj// 2**bin
    
    with h5py.File(args.in_file) as fid:        
        # positions initial guess        
        pos = (fid[f'/exchange/cshifts_final'][args.ids,:args.ndist]).astype('float32')        
        pos /= 2**bin
        s = args.rotation_center_shift
        for k in range(bin):
            s = (s - 0.5) / 2
        pos[..., 1] += s
        # object initial guess   
        obj = fid[f'/exchange/obj_init_re{args.paganin}_{bin}']
        nzobj0,nobj0 = obj.shape[:2]
        stz,endz = (nzobj0//2-nzobj//2),(nzobj0//2+nzobj//2)
        stx,endx = (nobj0//2-nobj//2),(nobj0//2+nobj//2)
        obj = obj[stz:endz,stx:endx,stx:endx].astype(args.obj_dtype)
        
    # probe initial guess
    prb = cp.ones([args.ndist, nz, n], dtype='complex64')    
    vars = {"obj":obj, "prb":prb, "pos":pos}    
    return vars    

def read_data(args):
    """Read data with a given binning level"""
    nz = args.nz // 2**args.bin
    n = args.n // 2**args.bin
    data = np.empty([args.ntheta, args.ndist, nz, n], dtype='float32')
    with h5py.File(args.in_file) as fid:
        for k in range(args.ndist):
            [nz0,n0] = fid[f'/exchange/pdata{k}_{args.bin}'].shape[1:]
            st,end = nz0//2-nz//2, nz0//2+nz//2
            ## note we take square root right after reading
            data[:, k] = np.sqrt(fid[f'/exchange/pdata{k}_{args.bin}'][args.ids,st:end])
        ref = cp.sqrt(cp.array(fid[f'/exchange/pref_{args.bin}'][:args.ndist,st:end]))    

    return data,ref

################################# 

## read acquisition parameters from file
args = parse_args(sys.argv[1])
read_acquisition_pars(args)
    
## set reconstrution parameters
rec_args = SimpleNamespace(
    ngpus=args.ngpus,
    ndist=args.ndist,
    ntheta=args.ntheta,
    energy=args.energy,        
    focustodetectordistance=args.focustodetectordistance,        
    z1=args.z1,   
    theta=args.theta,
    mask=args.mask,        
    lam_prbfit=args.lam_prbfit,
    lam_reg=args.lam_reg,
    obj_dtype=args.obj_dtype,
    rho=args.rho,      
    n = args.n // 2**args.bin,
    nz = args.nz // 2**args.bin,
    nobj = args.nobj // 2**args.bin,
    nzobj = args.nzobj // 2**args.bin,
    detector_pixelsize = args.detector_pixelsize * 2**args.bin,       
    nchunk = args.nchunk[0],
    niter = args.niter[0],
    vis_step = args.vis_step[0],
    err_step = args.err_step[0],
    path_out = f"{args.path_out}/{args.bin}",    
    start_iter = 0,
    show=False,            
)

### Read initial guess and data
vars = read_initial_guess(args)    
data, ref = read_data(args)
    
# create class and run reconstruction by the BH method
vars = Rec(rec_args).BH(data, ref, vars)          
