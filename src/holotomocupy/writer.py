import os
import h5py
import numpy as np
import cupy as cp
import tifffile
from .logger_config import logger


class Writer:
    """MPI-aware HDF5 writer for reconstruction checkpoints.

    Uses parallel HDF5 (mpio driver). All ranks open the file collectively;
    obj and pos are written with collective I/O; prb is written by rank 0 only.

    File layout — {path_out}/checkpoint_{iter:04}.h5:
      /obj_re  (nzobj, nobj, nobj)  float32 — real part of obj (assembled from all ranks)
      /obj_im  (nzobj, nobj, nobj)  float32 — imag part of obj (complex64 dtype only)
      /prb_abs   (ndist, nz, n)     float32 — probe amplitude (from rank 0)
      /prb_phase (ndist, nz, n)     float32 — probe phase     (from rank 0)
      /pos     (ntheta, ndist, 2)   float32 — assembled from all ranks (theta-distributed)

    Attrs on the root group:
      iter, obj_dtype
    """

    def __init__(self, path_out, comm,
                 st_obj, end_obj, nzobj, nobj,
                 st_theta, end_theta, ntheta,
                 ndist, nz, n, obj_dtype):
        self.path_out  = path_out
        self.comm      = comm
        self.rank      = comm.Get_rank()
        self.size      = comm.Get_size()
        self.st_obj    = st_obj
        self.end_obj   = end_obj
        self.nzobj     = nzobj
        self.nobj      = nobj
        self.st_theta  = st_theta
        self.end_theta = end_theta
        self.ntheta    = ntheta
        self.ndist     = ndist
        self.nz        = nz
        self.n         = n
        self.obj_dtype = obj_dtype

        if self.rank == 0:
            os.makedirs(path_out, exist_ok=True)
        comm.Barrier()  # ensure directory exists before other ranks proceed

    @staticmethod
    def _cpu(x):
        """Move a CuPy or NumPy array to a contiguous CPU NumPy array."""
        if isinstance(x, cp.ndarray):
            return x.get()
        return np.asarray(x)

    def write_checkpoint(self, vars, i, norm_const):
        """Save obj, prb, pos for iteration i to an HDF5 checkpoint file.

        Parameters
        ----------
        vars : dict
            Reconstruction variables with keys 'obj', 'prb', 'pos'.
            obj is expected to be scaled by 1/norm_const (as during iteration).
        i : int
            Iteration number, used in the filename.
        norm_const : float
            Normalisation constant — obj is multiplied by this before saving.
        """
        path = os.path.join(self.path_out, f"checkpoint_{i:04}.h5")

        obj = self._cpu(vars['obj']) * norm_const  # denormalise
        pos = self._cpu(vars['pos'])
        prb = self._cpu(vars['prb']) 

        # mpio block: all ranks create datasets and write obj/pos collectively
        with h5py.File(path, 'w', driver="mpio", comm=self.comm) as f:
            f.attrs['iter']      = i
            f.attrs['obj_dtype'] = self.obj_dtype

            obj_shape = (self.nzobj, self.nobj, self.nobj)
            ds_re = f.create_dataset('obj_re', shape=obj_shape, dtype='float32')
            if self.obj_dtype == 'complex64':
                ds_im = f.create_dataset('obj_im', shape=obj_shape, dtype='float32')
            ds_pos = f.create_dataset('pos', shape=(self.ntheta, self.ndist, 2), dtype='float32')
            prb_shape = (self.ndist, self.nz, self.n)
            ds_prb_abs = f.create_dataset('prb_abs',   shape=prb_shape, dtype='float32')
            ds_prb_phase = f.create_dataset('prb_phase', shape=prb_shape, dtype='float32')

            ds_re[self.st_obj:self.end_obj] = obj.real
            if self.obj_dtype == 'complex64':
                ds_im[self.st_obj:self.end_obj] = obj.imag
            ds_pos[self.st_theta:self.end_theta] = pos

        # prb written by rank 0 only via serial driver after mpio block closes
        self.comm.Barrier()
        if self.rank == 0:
            with h5py.File(path, 'a') as f:
                f['prb_abs'][:]   = np.abs(prb).astype('float32')
                f['prb_phase'][:] = np.angle(prb).astype('float32')
        self.comm.Barrier()
        if self.rank == 0:
            logger.info(f"Writer: checkpoint saved → {path}")
            mid = self.nzobj // 2
            with h5py.File(path, 'r') as f:
                slice_re = f['obj_re'][mid]
            tiff_path = os.path.join(self.path_out, f"checkpoint_{i:04}_obj_re.tiff")
            tifffile.imwrite(tiff_path, slice_re)
            logger.info(f"Writer: mid-slice TIFF saved → {tiff_path}")
        
