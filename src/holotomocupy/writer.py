import os
import h5py
import numpy as np
import cupy as cp
from .logger_config import logger


class Writer:
    """MPI-aware HDF5 writer for reconstruction checkpoints.

    Writing is sequential per-rank (no parallel HDF5 required).

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

        # Rank 0 creates the file and writes prb (replicated, same on all ranks)
        if self.rank == 0:
            prb = self._cpu(vars['prb'])
            with h5py.File(path, 'w') as f:
                f.attrs['iter']      = i
                f.attrs['obj_dtype'] = self.obj_dtype
                shape = (self.nzobj, self.nobj, self.nobj)
                f.create_dataset('obj_re', shape=shape, dtype='float32')
                if self.obj_dtype == 'complex64':
                    f.create_dataset('obj_im', shape=shape, dtype='float32')
                f.create_dataset('pos', shape=(self.ntheta, self.ndist, 2),
                                 dtype='float32')
                f.create_dataset('prb_abs',   data=np.abs(prb).astype('float32'))
                f.create_dataset('prb_phase', data=np.angle(prb).astype('float32'))
                logger.info(f"Writer: created {path}")
        self.comm.Barrier()

        # Each rank fills its own slice sequentially (no parallel HDF5 needed)
        for r in range(self.size):
            if self.rank == r:
                with h5py.File(path, 'a') as f:
                    f['obj_re'][self.st_obj:self.end_obj] = obj.real
                    if self.obj_dtype == 'complex64':
                        f['obj_im'][self.st_obj:self.end_obj] = obj.imag
                    f['pos'][self.st_theta:self.end_theta] = pos
            self.comm.Barrier()

        if self.rank == 0:
            logger.info(f"Writer: checkpoint saved → {path}")
