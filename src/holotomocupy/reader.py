import glob
import os
import h5py
import numpy as np
import cupy as cp


def find_latest_checkpoint(path_out, start_iter):
    """Return the path to the most recent checkpoint in path_out, or None."""
    if start_iter > 0:
        files = sorted(glob.glob(os.path.join(path_out, f'checkpoint_*{start_iter:04}.h5')))
        return files[-1] if files else None
    else:
        return None


class Reader:
    """MPI-aware HDF5 reader for holotomography data.

    Mirrors Writer: captures all fixed parameters at construction time so each
    read_* method needs no extra arguments beyond what is rank-specific.

    Acquisition parameters (detector_pixelsize, focustodetectordistance, z1,
    energy, ids, theta) are read once in __init__ and stored as attributes.

    File datasets:
      /exchange/obj_init_re{paganin}_{bin}   initial object
      /exchange/cshifts_final                positions
      /exchange/pdata{k}_{bin}               projection data per distance
      /exchange/pref_{bin}                   reference (flat-field)
    """

    def __init__(self, in_file, comm,
                 st_obj, end_obj, nzobj, nobj,
                 st_theta, end_theta, ntheta,
                 ndist, nz, n, obj_dtype,
                 paganin, rotation_center_shift, start_theta, bin):
        self.in_file   = in_file
        self.comm      = comm
        self.rank      = comm.Get_rank()
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
        self.paganin   = paganin
        self.rotation_center_shift = rotation_center_shift
        self.bin       = bin

        # Read acquisition parameters once and store as attributes
        with h5py.File(in_file, 'r', driver="mpio", comm=self.comm) as fid:
            self.detector_pixelsize      = fid['/exchange/detector_pixelsize'][0]
            self.focustodetectordistance = fid['/exchange/focusdetectordistance'][0]
            self.z1                      = fid['/exchange/z1'][:ndist]
            self.energy                  = fid['/exchange/energy'][0]
            ntheta0 = len(fid['/exchange/theta'])
            # FIX: clip to exactly ntheta to avoid float-step off-by-one
            ids = np.arange(start_theta, ntheta0, ntheta0 / ntheta)
            self.ids   = ids[:ntheta].astype('int')
            self.theta = -fid['/exchange/theta'][self.ids, 0] / 180 * np.pi
            self.detector_pixelsize *= 2**self.bin

    def read_obj(self, out=None):
        """Read initial object guess for this rank's z-slice into out."""
        with h5py.File(self.in_file, 'r', driver="mpio", comm=self.comm) as fid:
            obj_ds = fid[f'/exchange/obj_init_re{self.paganin}_{self.bin}']
            nzobj0, nobj0 = obj_ds.shape[:2]
            stz  = nzobj0 // 2 - self.nzobj // 2
            stx  = nobj0  // 2 - self.nobj  // 2
            endx = nobj0  // 2 + self.nobj  // 2
            if out is None:
                out = obj_ds[stz + self.st_obj : stz + self.end_obj,
                             stx:endx, stx:endx].astype(self.obj_dtype)
            elif self.obj_dtype == 'complex64':
                out.real[:] = obj_ds[stz + self.st_obj : stz + self.end_obj, stx:endx, stx:endx]
                out.imag[:] = 0
            else:
                out[:] = obj_ds[stz + self.st_obj : stz + self.end_obj, stx:endx, stx:endx]
        return out

    def read_pos(self, out=None):
        """Read initial positions for this rank's theta-slice into out."""
        with h5py.File(self.in_file, 'r', driver="mpio", comm=self.comm) as fid:
            if out is None:
                out = fid[f'/exchange/cshifts_final'][
                    self.ids[self.st_theta:self.end_theta], :self.ndist
                ].astype('float32')
            else:
                out[:] = cp.array(fid[f'/exchange/cshifts_final'][
                    self.ids[self.st_theta:self.end_theta], :self.ndist
                ], dtype='float32')
        out /= 2**self.bin
        s = self.rotation_center_shift
        for _ in range(self.bin):
            s = (s - 0.5) / 2
        out[..., 1] += s
        return out

    def read_prb(self, out=None):
        """Initialise probe to all-ones (allocates CuPy array if None)."""
        if out is None:
            out = cp.empty([self.ndist, self.nz, self.n], dtype='complex64')
        out[:] = 1
        return out

    def read_data(self, out=None):
        """Read projection data for this rank's theta-slice into out.

        Reads directly into out (pinned if pre-allocated) and applies sqrt in-place,
        avoiding any intermediate allocation.
        """
        nz, n = self.nz, self.n
        if out is None:
            out = np.empty([self.end_theta - self.st_theta, self.ndist, nz, n],
                           dtype='float32')
        with h5py.File(self.in_file, 'r', driver="mpio", comm=self.comm) as fid:
            for k in range(self.ndist):
                nz0 = fid[f'/exchange/pdata{k}_{self.bin}'].shape[1]
                st, end = nz0 // 2 - nz // 2, nz0 // 2 + nz // 2
                out[:, k] = fid[f'/exchange/pdata{k}_{self.bin}'][
                    self.ids[self.st_theta:self.end_theta], st:end
                ]
                np.sqrt(out[:, k], out=out[:, k])
        return out

    def read_ref(self, out=None):
        """Read reference (flat-field) on rank 0 and broadcast to all ranks."""
        nz = self.nz
        n = self.n
        # FIX: read once on rank 0, broadcast — avoids N redundant identical reads
        raw_np = np.empty((self.ndist, nz, n), dtype='float32')
        if self.rank == 0:
            with h5py.File(self.in_file, 'r') as fid:
                nz0 = fid[f'/exchange/pref_{self.bin}'].shape[1]
                st, end = nz0 // 2 - nz // 2, nz0 // 2 + nz // 2
                raw_np[:] = fid[f'/exchange/pref_{self.bin}'][:self.ndist, st:end]
        self.comm.Bcast(raw_np, root=0)
        raw = cp.array(raw_np)
        if out is None:
            out = cp.sqrt(raw)
        else:
            cp.sqrt(raw, out=out)
        return out

    def read_checkpoint(self, path, out_obj=None, out_pos=None, out_prb=None):
        """Read reconstruction variables from an HDF5 checkpoint (this rank's slices).

        prb is read on rank 0 and broadcast; obj and pos are read in parallel.

        Returns dict with keys 'obj' (CPU NumPy), 'prb' (GPU CuPy), 'pos' (GPU CuPy).
        """
        # FIX: prb is identical on all ranks — read on rank 0, broadcast
        prb_shape = (self.ndist, self.nz, self.n)
        prb_np = np.empty(prb_shape, dtype='complex64')
        if self.rank == 0:
            with h5py.File(path, 'r') as f:
                prb_np[:] = (f['prb_abs'][:] * np.exp(1j * f['prb_phase'][:])).astype('complex64')
        self.comm.Bcast(prb_np, root=0)
        if out_prb is None:
            out_prb = cp.array(prb_np)
        else:
            out_prb[:] = cp.array(prb_np)

        # obj and pos: parallel read, each rank reads its own slice
        with h5py.File(path, 'r', driver="mpio", comm=self.comm) as f:
            obj_dtype = f.attrs['obj_dtype']
            if out_obj is None:
                obj_re = f['obj_re'][self.st_obj:self.end_obj]
                if obj_dtype == 'complex64':
                    out_obj = (obj_re + 1j * f['obj_im'][self.st_obj:self.end_obj]).astype('complex64')
                else:
                    out_obj = obj_re.astype('float32')
            elif obj_dtype == 'complex64':
                out_obj.real[:] = f['obj_re'][self.st_obj:self.end_obj]
                out_obj.imag[:] = f['obj_im'][self.st_obj:self.end_obj]
            else:
                out_obj[:] = f['obj_re'][self.st_obj:self.end_obj]

            if out_pos is None:
                out_pos = cp.array(f['pos'][self.st_theta:self.end_theta].astype('float32'))
            else:
                out_pos[:] = cp.array(f['pos'][self.st_theta:self.end_theta], dtype='float32')

        return {'obj': out_obj, 'prb': out_prb, 'pos': out_pos}

    def read_obj_unbin(self, out):
        """Read initial object in one bulk I/O call and upsample by 2**bin."""
        st, end = self.st_obj, self.end_obj
        n0 = end - st
        # FIX: scale = 2**bin (not 2**(-bin)) for upsampling
        scale = 2 ** self.bin
        nz_src = max(1, n0 // scale)
        st_src = st // scale
        with h5py.File(self.in_file, 'r', driver="mpio", comm=self.comm) as fid:
            # FIX: single bulk read instead of per-slice loop
            block = fid['/exchange/obj'][st_src : st_src + nz_src]
            if self.obj_dtype == 'float32':
                block = block.real.copy()
        # upsample spatial dimensions in memory
        block = np.repeat(np.repeat(block, scale, axis=1), scale, axis=2)
        # map source z-slices to output z-slices
        idx0 = np.clip(
            (np.arange(n0) * nz_src / n0).astype(np.intp),
            0, nz_src - 1,
        )
        out[:] = block[idx0].astype(self.obj_dtype)
        return out

    def read_pos_unbin(self, out):
        """Read initial positions with downsampling."""
        with h5py.File(self.in_file, 'r', driver="mpio", comm=self.comm) as fid:
            pos = cp.array(fid['/exchange/cshifts_final'][
                self.ids[self.st_theta:self.end_theta], :self.ndist
            ].astype('float32'))
        pos[..., 1] += self.rotation_center_shift
        out[:] = pos * 2**(-self.bin)
        return out

    def read_pos_error_unbin(self, out):
        """Read position errors with downsampling."""
        with h5py.File(self.in_file, 'r', driver="mpio", comm=self.comm) as fid:
            pos = cp.array(fid['/exchange/cshifts_error'][
                self.ids[self.st_theta:self.end_theta], :self.ndist
            ].astype('float32'))
        pos[..., 1] += self.rotation_center_shift
        out[:] = pos * 2**(-self.bin)
        return out

    def read_prb_unbin(self, out):
        """Read initial probe and upsample by 2**bin in spatial dimensions."""
        with h5py.File(self.in_file, 'r', driver="mpio", comm=self.comm) as fid:
            prb = fid['/exchange/prb'][:]
        scale = 2 ** (-self.bin)
        for axis in [2, 1]:
            prb = np.repeat(prb, scale, axis=axis)
        out[:] = cp.array(prb).astype('complex64')
        return out
