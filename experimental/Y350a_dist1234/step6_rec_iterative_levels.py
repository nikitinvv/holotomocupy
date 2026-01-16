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
        
        ntheta0 = np.array(fid[f'/exchange/data0'].shape[0])
        args.ids = np.arange(args.start_theta, ntheta0, ntheta0 / args.ntheta).astype('int')
        args.theta = -fid['/exchange/theta'][args.ids, 0] / 180 * np.pi
    
def read_initial_guess(args):
    """Read initial guess for the object, probe and positions with a given binning level"""

    bin = args.start_bin
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
        # if obj.dtype=='complex64':
        #     obj+=1j*fid[f'/exchange/obj_init_im{args.paganin}_{bin}'][st:end]

    # probe initial guess
    prb = cp.ones([args.ndist, nz, n], dtype='complex64')
    
    vars = {"obj":obj, "prb":prb, "pos":pos}    
    return vars    

def read_result_on_iter(args):
    """Read object, probe, positions from some iteration on a given binning level"""

    bin = args.start_bin
    iter = args.start_iter
    nz = args.nz // 2**bin
    n = args.n // 2**bin

    # read object
    obj = read_tiff(f"{args.path_out}/{bin}/rec_obj_real/{iter:04}.tiff").astype(args.obj_dtype)
    if args.obj_dtype=='complex64':
        obj+=1j*read_tiff(f"{args.path_out}/{bin}/rec_obj_imag/{iter:04}.tiff")
    
    # read probe
    prb = cp.ones([args.ndist,nz,n],dtype='complex64')
    for k in range(args.ndist):
        prb_abs = read_tiff(f"{args.path_out}/{bin}/rec_prb_abs{k}/{iter:04}.tiff")
        prb_angle = read_tiff(f"{args.path_out}/{bin}/rec_prb_angle{k}/{iter:04}.tiff")
        prb[k] = cp.array(prb_abs*np.exp(1j*prb_angle))
    
    # read positions
    pos = np.load(f"{args.path_out}/{bin}/pos{iter:04}.npy")        
    vars = {"obj": obj, "prb": prb, "pos": pos}
    
    return vars  
    
def read_data(args, bin):
    """Read data with a given binning level"""
    nz = args.nz // 2**bin
    n = args.n // 2**bin
    data = np.empty([args.ntheta, args.ndist, nz, n], dtype='float32')
    with h5py.File(args.in_file) as fid:
        for k in range(args.ndist):
            [nz0,n0] = fid[f'/exchange/pdata{k}_{bin}'].shape[1:]
            st,end = nz0//2-nz//2, nz0//2+nz//2
            ## note we take square root right after reading
            data[:, k] = np.sqrt(fid[f'/exchange/pdata{k}_{bin}'][args.ids,st:end])
        ref = cp.sqrt(cp.array(fid[f'/exchange/pref_{bin}'][:args.ndist,st:end]))    

    return data,ref

def update_rec_args(rec_args, args, bin):
    """Update reconstruction arguments for the given binning level"""

    rec_args.n = args.n // 2**bin
    rec_args.nz = args.nz // 2**bin
    rec_args.nobj = args.nobj // 2**bin
    rec_args.nzobj = args.nzobj // 2**bin
    rec_args.detector_pixelsize = args.detector_pixelsize * 2**bin       
    rec_args.nchunk = args.nchunk[bin]
    rec_args.niter = args.niter[bin]
    rec_args.vis_step = args.vis_step[bin]
    rec_args.err_step = args.err_step[bin]
    rec_args.clean_cache_step = args.clean_cache_step[bin]
    rec_args.path_out = f"{args.path_out}/{bin}"
    
    if bin==args.start_bin: ## start from start_iter on first bin level        
        rec_args.start_iter = max(args.start_iter,0)
    else: ## further bin levels start with 0
        rec_args.start_iter = 0


def upsample_vars(vars):
    """Upsample variables for the next level"""

    # upsample object
    obj = fftupsample(vars['obj'], [0])
    obj = fftupsample(obj, [1])
    obj = fftupsample(obj, [2])
    # n=vars['obj'].shape[-1]
    # obj = np.zeros([2*n,2*n,2*n],dtype=args.obj_dtype)
    # upsample probe
    prb = fftupsample(vars['prb'].get(), [1])
    prb = fftupsample(prb, [2])
    prb = cp.array(prb) # note probe is stored in gpu always
    # multiply positions by 2, shift by 0.5 to be at the detector middle
    pos = vars['pos'] * 2 + 0.5    
    # shift probe by 0.5 in x and y
    from cupyx.scipy.ndimage import shift
    prb = shift(prb, shift=(0, 0.5, 0.5), order=3, mode='nearest')
    vars['obj'] = obj
    vars['prb'] = prb
    vars['pos'] = pos
    return vars


args = parse_args(sys.argv[1])

## add acquisition parameters from file
read_acquisition_pars(args)
    
## reconstrution parameters
rec_args = SimpleNamespace(
    ngpus=cp.cuda.runtime.getDeviceCount(),
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
    show=False,        
)

### Read initial guess or result on iteration args.start_iter
if args.start_iter==-1: 
    vars = read_initial_guess(args)    
else: 
    vars = read_result_on_iter(args) 

### Reconstruction by bin levels, e.g. [2,1,0]
for bin in range(args.start_bin,args.start_bin-args.nbins,-1):  
    print(f'{bin=}')    
    
    update_rec_args(rec_args,args,bin)
    
    data, ref = read_data(args,bin)
    
    if bin < args.start_bin:
        vars = upsample_vars(vars)
    
    # create class and run reconstruction by the BH method
    vars = Rec(rec_args).BH(data, ref, vars)          


# bin=2
# 09:59:52 n=512 iter=0 err=6.88599e-02
# 10:00:00 n=512 iter=1 err=6.99182e-04