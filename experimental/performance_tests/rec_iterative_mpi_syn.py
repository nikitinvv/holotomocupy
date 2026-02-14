import numpy as np
import cupy as cp
import sys
from types import SimpleNamespace
from mpi4py import MPI
from holotomocupy.rec_mpi import Rec
from holotomocupy.config import parse_args
from holotomocupy.utils import *
from holotomocupy.reader import *
from holotomocupy.logger_config import logger

cp.cuda.set_pinned_memory_allocator(None)
pool = cp.get_default_memory_pool()

# # Limit pool to 5 GiB
# pool.set_limit(size=5 * 1024**3)

logger.info(f"Read acquisition parameters")
args = parse_args(sys.argv[1])
read_acquisition_pars(args)
# Form reconstruction parameters
rargs = SimpleNamespace(
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
    n = args.n * 2**args.unbin,
    nz = args.n  * 2**args.unbin,
    nobj = args.nobj  * 2**args.unbin,
    nzobj = args.nobj  * 2**args.unbin,
    detector_pixelsize = args.detector_pixelsize  / 2**args.unbin, 
    nchunk = args.nchunk[0],
    niter = args.niter[0],
    vis_step = args.vis_step[0],
    err_step = args.err_step[0],
    path_out = f"{args.path_out}",    
    start_iter = 0,
    show=False,         
    comm=MPI.COMM_WORLD
)




logger.info(f"Create class")
cl = Rec(rargs)
logger.info(f"obj-range [{cl.st_obj}:{cl.end_obj}), local size: {cl.end_obj-cl.st_obj} x {cl.nobj} x {cl.nobj}")
logger.info(f"proj-range [{cl.st_obj}:{cl.end_obj}), local size: {cl.end_obj-cl.st_obj} x {cl.ntheta} x {cl.nobj}")
logger.info(f"projt-range [{cl.st_theta}:{cl.end_theta}), local size: {cl.end_theta-cl.st_theta}x {cl.nzobj} x {cl.nobj}")




logger.info(f'Allocate memory and  read data with unbinning from {args.n} size to {cl.n}')
vars = {}
vars['prb'] = cp.empty([cl.ndist,cl.nz,cl.n],dtype="complex64")
read_prb_unbin(args,out = vars['prb'])    

vars['pos'] = make_pinned([cl.end_theta-cl.st_theta,cl.ndist,2],dtype="float32")
read_pos_unbin(args, cl.st_theta, cl.end_theta,out = vars['pos'])    

vars['obj'] = make_pinned([cl.end_obj-cl.st_obj,cl.nobj,cl.nobj],dtype=args.obj_dtype)
read_obj_unbin(args, cl.st_obj, cl.end_obj,out = vars['obj'])    

logger.debug(f"{vars['pos'].shape=}")
logger.debug(f"{vars['prb'].shape=}")
logger.debug(f"{vars['obj'].shape=}")



cl.comm.Barrier()
logger.info(f'Generate data')

vars['proj'] = make_pinned([cl.end_theta-cl.st_theta,cl.nzobj,cl.nobj],dtype=args.obj_dtype)
data = make_pinned([cl.end_theta-cl.st_theta,cl.ndist,cl.nz,cl.n],dtype="float32")
ref = cp.empty([cl.ndist,cl.nz,cl.n],dtype="float32")

cl.gen_sqrt_data(vars,data)
cl.gen_sqrt_ref(vars['prb'],ref)

logger.info(f'Save first from data for each node')
write_tiff(data[0,0],f'{args.path_out}/data0_{cl.rank:03}',overwrite=True)
write_tiff(data[0,-1],f'{args.path_out}/data1_{cl.rank:03}',overwrite=True)
logger.info(f'Saved as {args.path_out}/data0_{cl.rank:03}.tiff')




cl.comm.Barrier()
logger.info(f'Set initial guess for obj, prb, pos')
read_pos_error_unbin(args, cl.st_theta, cl.end_theta,out = vars['pos'])    
vars['prb'][:] = 1
vars['obj'][:] = 0



cl.comm.Barrier()
logger.info(f'Run reconstruction')
vars = cl.BH(data, ref, vars)   




logger.info(f'Save middle vertical slice from reconstruction for each node')
write_tiff(vars['obj'][:,cl.nobj//2].real,f'{args.path_out}/rec_{cl.rank:03}',overwrite=True)
logger.info(f'Saved as {args.path_out}/rec_{cl.rank:03}.tiff')




cl.comm.Barrier()
if cl.rank==0:
    logger.info(f'Save probe from node 0')
    
    write_tiff(cp.abs(vars['prb'][0]).get(),f'{args.path_out}/prb_abs0',overwrite=True)
    write_tiff(cp.angle(vars['prb'][0]).get(),f'{args.path_out}/prb_angle0',overwrite=True)
    write_tiff(cp.abs(vars['prb'][-1]).get(),f'{args.path_out}/prb_abs1',overwrite=True)
    write_tiff(cp.angle(vars['prb'][-1]).get(),f'{args.path_out}/prb_angle1',overwrite=True)
    
    logger.info(f'Saved as {args.path_out}/prb.tiff')

    logger.info(f'Read and stitch 1 slice by node 0')
    slice_stitched = np.empty([cl.nzobj,cl.nobj],dtype='float32')
    offset = 0
    for k in range(cl.comm.Get_size()):
        tmp = read_tiff(f'{args.path_out}/rec_{k:03}.tiff')
        slice_stitched[offset:offset+tmp.shape[0]] = tmp
        offset += tmp.shape[0]
    
    logger.info(f'Save stitched middle vertical slice')
    write_tiff(slice_stitched,f'{args.path_out}/rec_stitched',overwrite=True)
    logger.info(f'Saved as {args.path_out}/rec_stitched.tiff')
        
