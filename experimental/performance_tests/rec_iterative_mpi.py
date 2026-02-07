import numpy as np
import cupy as cp
import sys
from types import SimpleNamespace
from mpi4py import MPI
from holotomocupy.rec_mpi import Rec
from holotomocupy.config import parse_args
from holotomocupy.utils import *
from holotomocupy.reader import *

cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)

## read acquisition parameters from file
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

## Create class
cl = Rec(rargs)

## distrbution by chunks
cl.cl_mpi.comm.Barrier()
for r in range(cl.cl_mpi.size):
    if cl.rank == r:
        print(f"Rank {cl.rank}: obj-range [{cl.st_obj}:{cl.end_obj}), local size: {cl.end_obj-cl.st_obj} x {cl.nobj} x {cl.nobj}")
        print(f"Rank {cl.rank}: proj-range [{cl.st_obj}:{cl.end_obj}), local size: {cl.end_obj-cl.st_obj} x {cl.ntheta} x {cl.nobj}")
        print(f"Rank {cl.rank}: projt-range [{cl.st_theta}:{cl.end_theta}), local size: {cl.end_theta-cl.st_theta}x {cl.nzobj} x {cl.nobj}")
cl.cl_mpi.comm.Barrier()

## Read data
obj = read_obj(args, cl.st_obj, cl.end_obj)    
pos = read_pos(args, cl.st_theta, cl.end_theta)    
data = read_data(args, cl.st_theta, cl.end_theta)
prb = read_prb(args)    
ref = read_ref(args)

# copy to pinned
vars = {'obj': obj, 'prb': prb, 'pos': pos}
vars['prb'] = prb # gpu 
vars['obj'] = copy_to_pinned(obj)
vars['pos'] = copy_to_pinned(pos)
data = copy_to_pinned(data)

vars = cl.BH(data, ref, vars)   

