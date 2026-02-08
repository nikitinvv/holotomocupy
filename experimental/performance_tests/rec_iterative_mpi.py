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

cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)


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
    comm=MPI.COMM_WORLD
)



logger.info(f"Create class")
cl = Rec(rargs)

logger.info(f"obj-range [{cl.st_obj}:{cl.end_obj}), local size: {cl.end_obj-cl.st_obj} x {cl.nobj} x {cl.nobj}")
logger.info(f"proj-range [{cl.st_obj}:{cl.end_obj}), local size: {cl.end_obj-cl.st_obj} x {cl.ntheta} x {cl.nobj}")
logger.info(f"projt-range [{cl.st_theta}:{cl.end_theta}), local size: {cl.end_theta-cl.st_theta}x {cl.nzobj} x {cl.nobj}")


logger.info(f'Read data')
vars = {}
vars['obj'] = read_obj(args, cl.st_obj, cl.end_obj)    
vars['pos'] = read_pos(args, cl.st_theta, cl.end_theta)    
data = read_data(args, cl.st_theta, cl.end_theta)
vars['prb'] = read_prb(args)    
ref = read_ref(args)



logger.info(f'Copy to pinned')
vars['obj'] = copy_to_pinned(vars['obj'] )
vars['pos'] = copy_to_pinned(vars['pos'])
data = copy_to_pinned(data)



logger.info(f'Run reconstruction')
vars = cl.BH(data, ref, vars)   

