import sys
from mpi4py import MPI
from holotomocupy.rec_mpi import Rec
from holotomocupy.config import parse_args
from holotomocupy.mpi_functions import MPIClass
from holotomocupy.reader import Reader, find_latest_checkpoint
from holotomocupy.writer import Writer
from holotomocupy.logger_config import logger

import cupy as cp
cp.cuda.set_pinned_memory_allocator(None)


args = parse_args(sys.argv[1])
comm = MPI.COMM_WORLD
args.comm = comm

cl_mpi = MPIClass(comm, args.nzobj, args.ntheta, args.nobj, args.obj_dtype)

reader = Reader(
    args.in_file, comm,
    cl_mpi.st_src, cl_mpi.end_src, args.nzobj, args.nobj,
    cl_mpi.st_dst, cl_mpi.end_dst, args.ntheta,
    args.ndist, args.nz, args.n, args.obj_dtype,
    args.paganin, args.rotation_center_shift, args.start_theta, args.bin,
)
writer = Writer(
    args.path_out, comm,
    cl_mpi.st_src, cl_mpi.end_src, args.nzobj, args.nobj,
    cl_mpi.st_dst, cl_mpi.end_dst, args.ntheta,
    args.ndist, args.nz, args.n, args.obj_dtype,
)

# physics parameters read from the data file
args.energy                  = reader.energy
args.focustodetectordistance = reader.focustodetectordistance
args.z1                      = reader.z1
args.detector_pixelsize      = reader.detector_pixelsize
args.theta                   = reader.theta

logger.info("Create class")
cl = Rec(args)

logger.info(f"obj-range [{cl.st_obj}:{cl.end_obj}), local size: {cl.end_obj-cl.st_obj} x {cl.nobj} x {cl.nobj}")
logger.info(f"proj-range [{cl.st_obj}:{cl.end_obj}), local size: {cl.end_obj-cl.st_obj} x {cl.ntheta} x {cl.nobj}")
logger.info(f"projt-range [{cl.st_theta}:{cl.end_theta}), local size: {cl.end_theta-cl.st_theta} x {cl.nzobj} x {cl.nobj}")

logger.info("Read data")
reader.read_data(out=cl.data)
reader.read_ref(out=cl.ref)

logger.info("Read initial variables")
ckpt = find_latest_checkpoint(args.path_out, args.start_iter)
if ckpt:
    logger.info(f"Resuming from checkpoint: {ckpt}")
    reader.read_checkpoint(ckpt, out_obj=cl.vars['obj'], out_pos=cl.vars['pos'], out_prb=cl.vars['prb'])
else:
    reader.read_obj(out=cl.vars['obj'])
    reader.read_pos(out=cl.vars['pos'])
    reader.read_prb(out=cl.vars['prb'])

logger.info("Run reconstruction")
cl.BH(writer=writer)
