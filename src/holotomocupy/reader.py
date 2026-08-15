import glob
import math
import os
import h5py
import numpy as np
import cupy as cp
from .logger_config import logger, rank as _mpi_rank


def load_octave_text_mat(fpath, varname):
    """Parse Octave/MATLAB text-format .mat file and return named variable as ndarray."""
    with open(fpath, 'r') as f:
        lines = f.read().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == f'# name: {varname}':
            i += 1
            meta = {}
            while i < len(lines) and lines[i].startswith('#'):
                parts = lines[i][1:].strip().split(':', 1)
                if len(parts) == 2:
                    meta[parts[0].strip()] = parts[1].strip()
                i += 1
            if 'ndims' in meta:
                shape = tuple(int(x) for x in lines[i].split())
                i += 1
                n = 1
                for s in shape:
                    n *= s
                vals = []
                while len(vals) < n:
                    vals.extend(lines[i].split()); i += 1
                return np.array(vals, dtype='float64').reshape(shape, order='F')
            else:
                rows = int(meta.get('rows', 1))
                cols = int(meta.get('columns', 1))
                vals = []
                for _ in range(rows):
                    vals.extend(lines[i].split()); i += 1
                return np.array(vals, dtype='float64').reshape(rows, cols)
        i += 1
    raise KeyError(f'{varname!r} not found in {fpath}')


def load_shrink_from_mats(path, pfile, ndist, ntheta, angle_ramp=False):
    """Build [ntheta, ndist, 2] shrink array; last axis is (v, h).

    Preferred source: `{pfile}_/shapp.mat` — per-(distance, projection) shrink
    values Peter saves when the shrinkage is fitted with a model.
    Stored as an Octave 3D matrix [2, ndist, ntheta] with axis-0 = (v, h),
    relative to first projection of first plane.

    Fallback: per-distance `shrink_list.mat`, a (3,2) matrix whose row 0 gives
    the incremental shrink from the previous plane as [h, v]. Cumulated and
    broadcast over projections. Layout of the returned axis matches the (v, h)
    convention used for r/m elsewhere:
        shrink_nd[..., 0] = vertical   (y, row) = shrink_list[0, 1]
        shrink_nd[..., 1] = horizontal (x, col) = shrink_list[0, 0]

    angle_ramp=False (default, Peter's convention): constant per distance, equal
    to the mid-scan cumulative value (cum[k] + inc[k]/2). Matches the value
    Peter feeds to his Fresnel-number correction. Applies to the fallback only;
    shapp.mat is per-projection so the flag is ignored when it is present.

    angle_ramp=True: linear ramp over angles, growing from cum[k] at angle 0
    to cum[k] + inc[k] at angle ntheta. Angle-mean equals Peter's value.

    Returns a zero array if neither source is available.
    """
    kind, val = _read_shrink_mats(path, pfile, ndist, ntheta)
    if kind is None:
        return np.zeros((ntheta, ndist, 2), dtype='float32')
    if kind == 'shapp':
        return val
    cum, inc = val
    if angle_ramp:
        j_frac = (np.arange(ntheta) / ntheta).astype('float32')                # (ntheta,)
        shrink_nd = cum[None, :, :] + inc[None, :, :] * j_frac[:, None, None]  # (ntheta, ndist, 2)
    else:
        peter_val = cum + 0.5 * inc                                            # (ndist, 2)
        shrink_nd = np.broadcast_to(peter_val[None, :, :], (ntheta, ndist, 2)).copy()
    return shrink_nd.astype('float32')


def load_shrink_total(path, pfile, ndist, ntheta):
    """Shrink accumulated by the END of one scan, (2,) = (v, h).

    The value ``load_shrink_from_mats`` would report for the last frame the
    scan took — last projection of the last plane — which is where the next
    scan of the same sample starts from. Not the same as ``shrink_nd[-1, -1]``:
    under Peter's convention (``angle_ramp=False``) that is the mid-scan value
    of the last plane, half an increment short of the end. Taken from the mats
    directly so it is exact under either convention.

    Zero if neither shapp.mat nor shrink_list.mat is there.
    """
    kind, val = _read_shrink_mats(path, pfile, ndist, ntheta)
    if kind is None:
        return np.zeros(2, dtype='float32')
    if kind == 'shapp':
        return val[-1, -1].astype('float32')
    cum, inc = val
    return (cum[-1] + inc[-1]).astype('float32')


def _read_shrink_mats(path, pfile, ndist, ntheta):
    """Raw shrink sources for one scan, shared by the two loaders above.

    Returns ``('shapp', raw[ntheta, ndist, 2])`` when shapp.mat is present,
    ``('list', (cum, inc))`` with both ``(ndist, 2)`` when falling back to the
    per-distance shrink_list.mat, or ``(None, None)`` when neither exists.
    """
    shapp_path = f'{path}/{pfile}_/shapp.mat'
    if os.path.exists(shapp_path):
        if _mpi_rank == 0:
            logger.info(f'shrink: reading shapp from {shapp_path}')
        raw_octave = load_octave_text_mat(shapp_path, 'shapp')
        # Expect Octave 3D [2, ndist, ntheta_file] (v, h; distance; projection).
        # Fail loudly on unexpected layout instead of returning silently wrong
        # values — misordered axes here corrupt every downstream step.
        if raw_octave.ndim != 3 or raw_octave.shape[0] != 2 or raw_octave.shape[1] != ndist:
            raise ValueError(
                f'{shapp_path}: expected shape (2, ndist={ndist}, ntheta), '
                f'got {raw_octave.shape}'
            )
        raw = raw_octave.swapaxes(0, 2)[:ntheta]                # (ntheta, ndist, 2)
        return 'shapp', raw.astype('float32')

    if _mpi_rank == 0:
        logger.info(f'shrink: shapp.mat not found ({shapp_path}); falling back to per-distance shrink_list.mat')
    inc_v = []  # vertical (y) — col 2 of MATLAB (col 1 zero-indexed)
    inc_h = []  # horizontal (x) — col 1 of MATLAB (col 0 zero-indexed)
    for k in range(ndist):
        mat_path = f'{path}/{pfile}_{k + 1}_/shrink_list.mat'
        if not os.path.exists(mat_path):
            if _mpi_rank == 0:
                logger.warning(f'shrink_list.mat not found, returning zeros: {mat_path}')
            return None, None
        if _mpi_rank == 0:
            logger.info(f'shrink: reading {mat_path}')
        sl = load_octave_text_mat(mat_path, 'shrink_list')
        inc_h.append(float(sl[0, 0]))
        inc_v.append(float(sl[0, 1]))
    inc = np.stack([inc_v, inc_h], axis=-1)                                   # (ndist, 2)
    cum = np.concatenate([np.zeros((1, 2)), np.cumsum(inc, axis=0)])[:ndist]  # (ndist, 2)
    return 'list', (cum, inc)


def read_nxtomo_meta(nx_path):
    """Read geometry and scan metadata from an ESRF NXtomo (.nx) file.

    Returns a dict with:
      entry           str   — HDF5 entry group name
      energy          float — keV
      pixel_size      float — m  (physical detector pixel size)
      z1              float — m  (focus-to-sample propagation distance)
      z_total         float — m  (focus-to-detector distance)
      magnification   float
      voxelsize       float — m
      ny, nx          int   — full detector frame size
      data_ids        ndarray[int] — frame indices where image_key == 0
      flat_ids        ndarray[int] — frame indices where image_key == 1
      dark_ids        ndarray[int] — frame indices where image_key == 2
      x_trans         ndarray[float64] — mm, sample x_translation for data frames (≈ spy)
      y_trans         ndarray[float64] — mm, sample y_translation for data frames (≈ spz)
    """
    with h5py.File(nx_path, 'r') as f:
        entry = next(k for k in f if k.startswith('entry'))
        g = f[entry]

        energy     = float(g['instrument/beam/incident_energy'][()])           # keV
        pixel_size = float(g['instrument/detector/x_pixel_size'][()]) * 1e-6  # µm → m
        _src_dist  = float(g['instrument/source/distance'][()])                 # mm (negative)
        z1         = -_src_dist * 1e-3                                        # mm → m
        z_total    = (float(g['instrument/detector/distance'][()]) - _src_dist) * 1e-3  # mm → m

        image_key = g['instrument/detector/image_key'][:]
        data_ids  = np.where(image_key == 0)[0]
        flat_ids  = np.where(image_key == 1)[0]
        dark_ids  = np.where(image_key == 2)[0]

        ny, nx = g['instrument/detector/data'].shape[1:3]

        x_trans = g['sample/x_translation'][data_ids].astype('float64')  # mm (≈ spy)
        y_trans = g['sample/y_translation'][data_ids].astype('float64')  # mm (≈ spz)

    magnification = z_total / z1
    voxelsize     = pixel_size / magnification

    return dict(
        entry=entry, energy=energy, pixel_size=pixel_size,
        z1=z1, z_total=z_total, magnification=magnification, voxelsize=voxelsize,
        ny=ny, nx=nx,
        data_ids=data_ids, flat_ids=flat_ids, dark_ids=dark_ids,
        x_trans=x_trans, y_trans=y_trans,
    )


def find_latest_checkpoint(path_out, start_iter):
    """Return the path to the most recent checkpoint in path_out, or None."""
    if start_iter > 0:
        files = sorted(glob.glob(os.path.join(path_out, 'checkpoints', f'checkpoint_*{start_iter:04}.h5')))
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
            self.theta = -fid['/exchange/theta'][:, 0][self.ids] / 180 * np.pi
            self.detector_pixelsize *= 2**self.bin
            
    def read_obj(self, out=None):
        """Read initial object guess for this rank's z-slice into out."""
        # obj_init may be in a separate _obj.h5 file (written there by step 5
        # to avoid the ~16 TiB Lustre per-file size limit).
        obj_file = self.in_file.replace('.h5', '_obj.h5')
        if not os.path.exists(obj_file):
            obj_file = self.in_file
        if self.rank == 0:
            logger.info(f'read object from {obj_file}')
        with h5py.File(obj_file, 'r', driver="mpio", comm=self.comm) as fid:
            obj_ds_re = fid[f'/exchange/obj_init_re{self.paganin}_{self.bin}']
            im_key = f'/exchange/obj_init_im{self.paganin}_{self.bin}'
            obj_ds_im = fid[im_key] if im_key in fid else None
            nzobj0, nobj0 = obj_ds_re.shape[:2]
            stz  = nzobj0 // 2 - self.nzobj // 2
            stx  = nobj0  // 2 - self.nobj  // 2
            endx = nobj0  // 2 + self.nobj  // 2
            local_nz = self.end_obj - self.st_obj
            if out is None:
                out = np.empty([local_nz, self.nobj, self.nobj], dtype=self.obj_dtype)
            batch = max(1, (1 << 28) // (self.nobj * self.nobj * obj_ds_re.dtype.itemsize))
            for i0 in range(0, local_nz, batch):
                i1 = min(i0 + batch, local_nz)
                sl = (slice(stz + self.st_obj + i0, stz + self.st_obj + i1),
                      slice(stx, endx), slice(stx, endx))
                if out.dtype == np.complex64:
                    out[i0:i1].real[:] = obj_ds_re[sl]
                    out[i0:i1].imag[:] = obj_ds_im[sl] if obj_ds_im is not None else 0
                else:
                    out[i0:i1] = obj_ds_re[sl]
        return out

    def read_pos(self, out=None):
        """Read initial positions for this rank's theta-slice into out."""
        if self.rank == 0:
            logger.info(f'read_pos: /exchange/cshifts_final from {self.in_file} '
                        f'(rotation_center_shift={self.rotation_center_shift:.4f} @ bin={self.bin})')
        with h5py.File(self.in_file, 'r', driver="mpio", comm=self.comm) as fid:
            if out is None:
                out = fid[f'/exchange/cshifts_final'][
                    self.ids[self.st_theta:self.end_theta], :self.ndist
                ].astype('float32')
            else:
                out[:] = cp.array(fid[f'/exchange/cshifts_final'][
                    self.ids[self.st_theta:self.end_theta], :self.ndist
                ], dtype='float32')

        scale = np.float32(1.0 / 2**self.bin)
        out *= scale
        out[..., 1] += np.float32(self.rotation_center_shift * scale + 0.5 * (scale - 1))
        return out

    def read_shrink(self, out=None):
        """Read [local_ntheta, ndist, 2] raw shrink for this rank's theta-slice.

        HDF5 stores per-angle shrink at /exchange/shrink. This method returns
        the raw values (no demag conversion); the caller can convert or fit
        (e.g. rec_mpi_shrink.Rec.init_tp_from_shrink). Falls back to zeros
        when /exchange/shrink is absent, and upgrades a legacy 2D shrink
        dataset by broadcasting the single scalar to both (y, x) axes.
        """
        local_ntheta = self.end_theta - self.st_theta
        with h5py.File(self.in_file, 'r', driver="mpio", comm=self.comm) as fid:
            if '/exchange/shrink' not in fid:
                if self.rank == 0:
                    logger.warning(
                        f'read_shrink: /exchange/shrink absent in {self.in_file}, '
                        f'using shrink=0'
                    )
                shrink_nd = cp.zeros((local_ntheta, self.ndist, 2), dtype='float32')
            else:
                raw = fid['/exchange/shrink']
                if self.rank == 0:
                    logger.info(
                        f'read_shrink: /exchange/shrink from {self.in_file} '
                        f'(shape={raw.shape}, ndim={raw.ndim})'
                    )
                sl = self.ids[self.st_theta:self.end_theta]
                if raw.ndim == 3:
                    shrink_nd = cp.array(raw[sl, :self.ndist, :2].astype('float32'))
                else:
                    # Legacy 2D dataset with a single scalar shrink per (j, k).
                    flat = cp.array(raw[sl, :self.ndist].astype('float32'))
                    shrink_nd = cp.broadcast_to(
                        flat[..., None], (local_ntheta, self.ndist, 2)
                    ).copy()
        if out is not None:
            out[:] = shrink_nd
        else:
            return shrink_nd

    def read_demagnifications(self, out=None):
        """Read [local_ntheta, ndist, 2] demagnifications for this rank's
        theta-slice.

        HDF5 stores the raw shrink at /exchange/shrink; this method reads it
        and returns the derived demagnifications:
            demag = (1 + shrink_nd) / norm_magnifications[None, :, None]

        Axis 2 is (y, x) per the convention shared with r/m. Falls back to
        zeros for shrink (i.e., demag = 1 / norm_magnifications) when
        /exchange/shrink is absent, and upgrades a legacy 2D shrink dataset
        by broadcasting it to both axes.
        """
        local_ntheta = self.end_theta - self.st_theta
        with h5py.File(self.in_file, 'r', driver="mpio", comm=self.comm) as fid:
            if '/exchange/shrink' not in fid:
                if self.rank == 0:
                    logger.warning(
                        f'read_demagnifications: /exchange/shrink absent in '
                        f'{self.in_file}, using shrink=0'
                    )
                shrink_nd = cp.zeros((local_ntheta, self.ndist, 2), dtype='float32')
            else:
                raw = fid['/exchange/shrink']
                if self.rank == 0:
                    logger.info(
                        f'read_demagnifications: /exchange/shrink from '
                        f'{self.in_file} (shape={raw.shape}, ndim={raw.ndim})'
                    )
                sl = self.ids[self.st_theta:self.end_theta]
                if raw.ndim == 3:
                    shrink_nd = cp.array(raw[sl, :self.ndist, :2].astype('float32'))
                else:
                    # Legacy 2D dataset with a single scalar shrink per (j, k).
                    flat = cp.array(raw[sl, :self.ndist].astype('float32'))
                    shrink_nd = cp.broadcast_to(
                        flat[..., None], (local_ntheta, self.ndist, 2)
                    ).copy()

        nm = cp.array(self.norm_magnifications, dtype='float32')
        data = (1.0 + shrink_nd) / nm[None, :, None]
        if out is not None:
            out[:] = data
        else:
            return data

    def read_prb(self, prb_file=None, out=None):
        """Initialise probe. Loads all ndist probes from prb_file if given, else ones."""
        if out is None:
            out = cp.empty([self.ndist, self.nz, self.n], dtype='complex64')
        if prb_file:
            with h5py.File(prb_file, 'r') as _f:
                for k in range(self.ndist):
                    _amp   = _f['prb_amp'][k]
                    _phase = _f['prb_phase'][k]
                    prb = (_amp * np.exp(1j * _phase)).astype('complex64')
                    nz0, n0 = prb.shape
                    if nz0 > self.nz or n0 > self.n:
                        bz = nz0 // self.nz
                        bn = n0 // self.n
                        prb = prb.reshape(self.nz, bz, self.n, bn).mean(axis=(1, 3))
                    out[k] = cp.array(prb)
                if self.rank == 0:
                    logger.info(f'Probe read from {prb_file}, shape '
                                f'{tuple(_f["prb_amp"].shape)}')
        else:
            out[:] = 1
        return out

    def read_data(self, out=None):
        """Read projection data for this rank's theta-slice into out.

        Reads directly into out (pinned if pre-allocated) and applies sqrt in-place,
        avoiding any intermediate allocation.
        """
        nz, n = self.nz, self.n
        local_ntheta = self.end_theta - self.st_theta
        if out is None:
            out = np.empty([local_ntheta, self.ndist, nz, n], dtype='float32')
        # Batch reads to stay under 2^31 bytes (MPI-IO uses int for transfer sizes)
        batch = max(1, (1 << 28) // (nz * n))
        with h5py.File(self.in_file, 'r', driver="mpio", comm=self.comm) as fid:
            for k in range(self.ndist):
                nz0 = fid[f'/exchange/pdata{k}_{self.bin}'].shape[1]
                st, end = nz0 // 2 - nz // 2, nz0 // 2 + nz // 2
                ds = fid[f'/exchange/pdata{k}_{self.bin}']
                for i0 in range(0, local_ntheta, batch):
                    i1 = min(i0 + batch, local_ntheta)
                    out[i0:i1, k] = ds[self.ids[self.st_theta + i0:self.st_theta + i1], st:end]
                np.sqrt(out[:, k], out=out[:, k])
        return out

    def read_ref(self, out=None):
        """Read reference (flat-field) on rank 0 and broadcast to all ranks."""
        nz = self.nz
        n = self.n
        # Read once on rank 0 and broadcast to avoid N redundant identical reads.
        raw_np = np.empty((self.ndist, nz, n), dtype='float32')
        if self.rank == 0:
            with h5py.File(self.in_file, 'r') as fid:
                key = f'/exchange/pref_{self.bin}'
                nz0 = fid[key].shape[1]
                st, end = nz0 // 2 - nz // 2, nz0 // 2 + nz // 2
                raw_np[:] = fid[key][:self.ndist, st:end]
        self.comm.Bcast(raw_np, root=0)
        raw = cp.array(raw_np)
        if out is None:
            out = cp.sqrt(raw)
        else:
            cp.sqrt(raw, out=out)
        return out

    def read_checkpoint(self, path, out_obj=None, out_prb=None, out_pos=None, out_tp=None):
        """Read a checkpoint saved at a coarser resolution and upsample.

        Scale is inferred automatically from checkpoint n vs self.n.

        prb  : upsampled in y and x by scale (repeat).
        obj  : upsampled in x and y by scale (repeat); z mapped by nearest-neighbour.
        pos  : multiplied by scale (pixel coords scale with resolution).
        tp   : (ndist, 2, 2) — shrinkage linear parameters (A, B).
               NOT scaled by binning (shrink is a unitless ratio). If the
               checkpoint predates tp saves, tp is left untouched and a
               warning is logged on rank 0.
        """
        # --- infer scale and probe on rank 0, broadcast ---
        prb_np = np.empty((self.ndist, self.nz, self.n), dtype='complex64')
        if self.rank == 0:
            with h5py.File(path, 'r') as f:
                scale = self.n // f['prb_abs'].shape[-1]
                prb_raw = (f['prb_abs'][:] * np.exp(1j * f['prb_phase'][:])).astype('complex64')
            for axis in [2, 1]:
                prb_raw = np.repeat(prb_raw, scale, axis=axis)
            prb_np[:] = prb_raw
            del prb_raw

        scale_arr = np.zeros(1, dtype='int32')
        if self.rank == 0:
            scale_arr[0] = scale
        self.comm.Bcast(scale_arr, root=0)
        scale = int(scale_arr[0])
        self.comm.Bcast(prb_np, root=0)

        if out_prb is None:
            out_prb = cp.array(prb_np)
        else:
            out_prb[:] = cp.array(prb_np)
        del prb_np
        if scale > 1:
            from cupyx.scipy.ndimage import shift
            shift_val = 0
            out_prb[:] = shift(out_prb, shift=(0, 0, shift_val), order=3, mode='nearest')

        # --- obj: z-batched read to cap peak CPU RAM ---
        # Old code read all nz_src slices into obj_re + obj_im + block at once,
        # which can exceed tens of GB per rank for large objects.
        # Now we process one z-batch at a time: peak extra RAM ≈ 2 × batch × nobj0² × 8 B.
        with h5py.File(path, 'r', driver="mpio", comm=self.comm) as f:
            obj_dtype = f.attrs['obj_dtype']
            st_src  = self.st_obj  // scale
            end_src = self.end_obj // scale
            n0      = self.end_obj - self.st_obj
            nz_src  = max(1, end_src - st_src)
            ds_re   = f['obj_re']
            ds_im   = f['obj_im'] if obj_dtype == 'complex64' else None
            nobj0   = ds_re.shape[1]

            if out_obj is None:
                out_obj = np.empty((n0, self.nobj, self.nobj), dtype=self.obj_dtype)

            # Target ~256 MB per batch (complex64 = 8 B worst case)
            z_batch = max(1, (1 << 28) // (nobj0 * nobj0 * 8))

            for i0 in range(0, n0, z_batch):
                i1     = min(i0 + z_batch, n0)
                src_i0 = int(i0 * nz_src / n0)
                src_i1 = min(int((i1 - 1) * nz_src / n0) + 1, nz_src)
                nz_b   = src_i1 - src_i0

                # Read directly into a complex64 buffer via .real/.imag views
                blk = np.zeros((nz_b, nobj0, nobj0), dtype='complex64')
                _re = ds_re[st_src + src_i0:st_src + src_i1].astype('float32')
                blk.real[:] = _re; del _re
                if ds_im is not None:
                    _im = ds_im[st_src + src_i0:st_src + src_i1].astype('float32')
                    blk.imag[:] = _im; del _im

                if scale > 1:
                    for axis in [2, 1]:
                        blk = np.repeat(blk, scale, axis=axis)

                idx_local = np.clip(
                    (np.arange(i0, i1) * nz_src / n0).astype(np.intp), 0, nz_src - 1
                ) - src_i0

                if out_obj.dtype == np.complex64:
                    out_obj[i0:i1] = blk[idx_local]
                else:
                    out_obj[i0:i1] = blk[idx_local].real
                del blk

            # --- pos: scale pixel coordinates up ---
            pos = f['pos'][self.st_theta:self.end_theta].astype('float32')

            # --- tp: read directly, NO binning scale (unitless ratios) ---
            has_tp = 'tp' in f
            if has_tp:
                tp_raw = f['tp'][:].astype('float32')

        pos_up = pos * scale
        if out_pos is None:
            out_pos = cp.array(pos_up)
        else:
            out_pos[:] = cp.array(pos_up, dtype='float32')

        if out_tp is not None:
            if has_tp:
                out_tp[:] = cp.asarray(tp_raw)
            elif self.rank == 0:
                logger.warning(
                    f'read_checkpoint: {path} has no /tp dataset '
                    f'(legacy checkpoint); leaving vars[tp] unchanged.')

        return {'obj': out_obj, 'prb': out_prb, 'pos': out_pos,
                'tp': out_tp if out_tp is not None else None}

    def read_pos_checkpoint(self, path, out=None, out_tp=None):
        """Read positions from a checkpoint file and upsample to current resolution.

        Scale is inferred from the checkpoint probe size vs self.n.
        If out_tp is provided, also load /tp (linear shrink params) — unscaled,
        since tp values are unitless ratios independent of binning.
        """
        if self.rank == 0:
            with h5py.File(path, 'r') as f:
                scale = self.n / f['prb_abs'].shape[-1]
        scale_arr = np.zeros(1, dtype='float32')
        if self.rank == 0:
            scale_arr[0] = scale
        self.comm.Bcast(scale_arr, root=0)
        scale = float(scale_arr[0])

        with h5py.File(path, 'r', driver="mpio", comm=self.comm) as f:
            pos = f['pos'][self.ids[self.st_theta:self.end_theta]].astype('float32')
            has_tp = 'tp' in f
            if has_tp and out_tp is not None:
                tp_raw = f['tp'][:].astype('float32')

        pos_up = pos * scale
        pos_up[..., 1] += np.float32(0.5 * (scale - 1))
        if out is None:
            out = cp.array(pos_up)
        else:
            out[:] = cp.array(pos_up, dtype='float32')

        if out_tp is not None:
            if has_tp:
                out_tp[:] = cp.asarray(tp_raw)
            elif self.rank == 0:
                logger.warning(
                    f'read_pos_checkpoint: {path} has no /tp dataset '
                    f'(legacy checkpoint); leaving vars[tp] unchanged.')
        return out

    def read_obj_unbin(self, out):
        """Read initial object in one bulk I/O call and upsample by 2**bin."""
        st, end = self.st_obj, self.end_obj
        n0 = end - st
        scale = 2 ** (-self.bin)
        nz_src = max(1, n0 // scale)
        st_src = st // scale
        with h5py.File(self.in_file, 'r', driver="mpio", comm=self.comm) as fid:
            ds = fid['/exchange/obj']
            batch = max(1, (1 << 28) // (ds.shape[1] * ds.shape[2] * ds.dtype.itemsize))
            block = np.empty((nz_src,) + ds.shape[1:], dtype=ds.dtype)
            for i0 in range(0, nz_src, batch):
                i1 = min(i0 + batch, nz_src)
                block[i0:i1] = ds[st_src + i0 : st_src + i1]
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

    def read_vol_obj(self, vol_path, out, scale=1.0, vol_dtype='float32'):
        """Read this rank's z-slice from a raw binary .vol file as object initial guess.

        Vol shape is nzobj*2^b x nobj*2^b x nobj*2^b where b is inferred from
        the file size. Block-averaging downsampling is applied when b > 0.
        Each rank reads independently (no MPI-IO needed for raw binary).
        """
        itemsize  = np.dtype(vol_dtype).itemsize
        file_size = os.path.getsize(vol_path)
        total_el  = file_size // itemsize

        # Infer power-of-2 bin level: total_el = nzobj * nobj^2 * 8^b
        base = self.nzobj * self.nobj * self.nobj
        if total_el % base != 0:
            raise ValueError(
                f"{vol_path}: file has {total_el} elements, "
                f"not a multiple of nzobj*nobj*nobj={base}"
            )
        ratio = total_el // base  # should be 8^b
        b = round(math.log2(ratio) / 3) if ratio > 1 else 0
        if 8 ** b != ratio:
            raise ValueError(
                f"{vol_path}: size ratio {ratio} is not a power of 8 "
                f"(expected nzobj*nobj^2 * 8^b)"
            )
        factor   = 2 ** b
        nobj_vol = self.nobj  * factor
        nz_vol   = self.nzobj * factor

        # Centre offsets — symmetric by construction when factor is a power of 2
        stz_vol = (nz_vol   - self.nzobj * factor) // 2  # always 0
        stx_vol = (nobj_vol - self.nobj  * factor) // 2  # always 0

        slice_pixels = nobj_vol * nobj_vol
        local_nz     = self.end_obj - self.st_obj

        logger.info(
            f"read_vol_obj: rank {self.rank} reading z=[{self.st_obj}:{self.end_obj}] "
            f"from vol [{nz_vol},{nobj_vol},{nobj_vol}]"
            + (f" (downsample 2^{b})" if b > 0 else "")
            + f" -> rec [{self.nzobj},{self.nobj},{self.nobj}]"
        )
        with open(vol_path, 'rb') as fh:
            for i in range(local_nz):
                acc = np.zeros([self.nobj * factor, self.nobj * factor], dtype='float32')
                for bz in range(factor):
                    z_vol = stz_vol + (self.st_obj + i) * factor + bz
                    if not (0 <= z_vol < nz_vol):
                        continue
                    fh.seek(z_vol * slice_pixels * itemsize)
                    row = np.frombuffer(fh.read(slice_pixels * itemsize), dtype=vol_dtype).astype('float32')
                    acc += row.reshape(nobj_vol, nobj_vol)[
                        stx_vol:stx_vol + self.nobj * factor,
                        stx_vol:stx_vol + self.nobj * factor,
                    ]
                acc /= factor
                if factor > 1:
                    acc = acc.reshape(self.nobj, factor, self.nobj, factor).mean(axis=(1, 3))
                if out.dtype == np.complex64:
                    out[i].real[:] = acc
                    out[i].imag[:] = 0
                else:
                    out[i][:] = acc

        if scale != 1.0:
            out /= np.float32(scale)
        logger.info(f"read_vol_obj: rank {self.rank} done (scale={scale})")
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


class MosaicReader(Reader):
    """Reader over N tile scans presented to the solver as one wide scan.

    A mosaic acquisition is N separate scans of the same rotation axis, the
    stage stepped sideways between them, each with its own ``ndist_tile``
    propagation distances. There is no reason for the solver to know that: from
    its point of view the mosaic is a single object seen through
    ``ndist = ntiles * ndist_tile`` "distances", of which every group of
    ``ndist_tile`` happens to share a z1 and to sit at a different lateral
    position. ``rec_mpi.Rec`` derives magnifications, propagation distances and
    voxel size from ``z1`` alone, so tiling ``z1`` is all it takes — no solver
    change.

    Index order is **tile-major**: entry ``t * ndist_tile + k`` is tile ``t``,
    distance ``k``.

    The tile's place on the mosaic rides in the position, not in a paste
    origin: ``pos = (cshifts_final + tile_offsets[t]) / 2**bin``, plus the usual
    rotation-centre term. ``tile_offsets`` is what step 5's ``estimate_overlap``
    measured, stored next to ``cshifts_final`` in every tile file.

    Shrinkage is applied exactly as in the single-tile reader: every tile file
    carries its own ``/exchange/shrink`` (step 3 of ``steps15.py``), and both
    ``read_shrink`` and ``read_demagnifications`` lay those out on the same
    flattened tile x distance axis as the data.
    """

    def __init__(self, tile_files, mosaic_file, comm,
                 st_obj, end_obj, nzobj, nobj,
                 st_theta, end_theta, ntheta,
                 ndist_tile, nz, n, obj_dtype,
                 paganin, rotation_center_shift, start_theta, bin,
                 tiles=None):
        # Scalar acquisition parameters are read from the first tile; they are
        # identical across tiles (same optics, same energy, same angles) and
        # _check_tiles below verifies that rather than assuming it.
        super().__init__(tile_files[0], comm,
                         st_obj, end_obj, nzobj, nobj,
                         st_theta, end_theta, ntheta,
                         ndist_tile, nz, n, obj_dtype,
                         paganin, rotation_center_shift, start_theta, bin)

        self.tile_files  = list(tile_files)
        self.ntiles      = len(self.tile_files)
        self.ndist_tile  = ndist_tile
        self.mosaic_file = mosaic_file
        self.tiles       = ([str(t) for t in tiles] if tiles
                            else [str(i) for i in range(self.ntiles)])
        if len(self.tiles) != self.ntiles:
            raise ValueError(f'{len(self.tiles)} tile names for '
                             f'{self.ntiles} tile files')

        z1_tile = np.asarray(self.z1, dtype='float64').copy()
        self._check_tiles(z1_tile)

        # Expand to the flattened tile x distance axis.
        self.z1    = np.tile(z1_tile, self.ntiles)
        self.ndist = self.ntiles * ndist_tile

        # Filled by read_pos; kept for logging / cross-checks by the driver.
        self.tile_offsets = None

        if self.rank == 0:
            logger.info(f'MosaicReader: {self.ntiles} tiles x {ndist_tile} '
                        f'distances -> ndist={self.ndist} (tile-major: '
                        f'idx = tile*{ndist_tile} + dist)')
            for t, path in enumerate(self.tile_files):
                logger.info(f'  tile {t} {self.tiles[t]:<10s} {path}')

    # ------------------------------------------------------------------ setup

    def _check_tiles(self, z1_tile):
        """Fail loudly if the tiles do not share the same acquisition geometry.

        A mismatch here means the tiles are not what the caller thinks they
        are, and every array below would be silently misassembled.
        """
        for t, path in enumerate(self.tile_files[1:], start=1):
            with h5py.File(path, 'r', driver="mpio", comm=self.comm) as fid:
                z1_t = fid['/exchange/z1'][:self.ndist_tile]
                dpx  = float(fid['/exchange/detector_pixelsize'][0]) * 2**self.bin
                fdd  = float(fid['/exchange/focusdetectordistance'][0])
                nth  = len(fid['/exchange/theta'])
            if len(z1_t) != self.ndist_tile:
                raise ValueError(f'{path}: /exchange/z1 has {len(z1_t)} entries, '
                                 f'need ndist_tile={self.ndist_tile}')
            if not np.allclose(z1_t, z1_tile, rtol=0, atol=1e-9):
                raise ValueError(
                    f'{path}: z1 {np.array2string(np.asarray(z1_t) * 1e3, precision=4)} mm '
                    f'differs from tile 0 '
                    f'{np.array2string(z1_tile * 1e3, precision=4)} mm')
            if not np.isclose(dpx, self.detector_pixelsize, rtol=1e-9, atol=0):
                raise ValueError(f'{path}: detector_pixelsize {dpx} differs from '
                                 f'tile 0 {self.detector_pixelsize}')
            if not np.isclose(fdd, self.focustodetectordistance, rtol=0, atol=1e-9):
                raise ValueError(f'{path}: focustodetectordistance {fdd} differs '
                                 f'from tile 0 {self.focustodetectordistance}')
            if nth < self.ids[-1] + 1:
                raise ValueError(f'{path}: only {nth} angles, but angle index '
                                 f'{self.ids[-1]} is requested')

    def _norm_magnifications(self):
        """M_k / M_0 over the flattened tile x distance axis.

        Computed here rather than taken from an attribute, so it cannot drift
        from what ``rec_mpi.Rec`` derives from the same (tiled) ``z1``.
        """
        mag = self.focustodetectordistance / np.asarray(self.z1, dtype='float64')
        return (mag / mag[0]).astype('float32')

    def _tile_offset(self, fid, t, path):
        """Row ``t`` of /exchange/tile_offsets, in object px on the finest grid."""
        key = '/exchange/tile_offsets'
        if key not in fid:
            raise RuntimeError(
                f'{path}: {key} is missing. It is written by step 5 of '
                f'steps15.py (the estimate_overlap block); without it the tile '
                f'placement is unknown and the reconstruction would be '
                f'meaningless. Re-run step 5 with estimate_overlap=true.')
        ds    = fid[key]
        table = np.asarray(ds[...], dtype='float32')
        if table.shape != (self.ntiles, 2):
            raise ValueError(f'{path}:{key} has shape {table.shape}, '
                             f'expected ({self.ntiles}, 2)')

        idx   = t
        names = [str(x) for x in ds.attrs.get('tiles', [])]
        if names:
            if names == self.tiles:
                stored = ds.attrs.get('index', None)
                if stored is not None and int(stored) != t and self.rank == 0:
                    logger.warning(f'{path}:{key} index attr is {int(stored)} but '
                                   f'this file is tile {t} in the configured '
                                   f'order; going by the configured order')
            else:
                # The table keeps its own order; find our tile inside it rather
                # than trusting the row number.
                if self.tiles[t] not in names:
                    raise ValueError(
                        f'{path}:{key} was written for tiles {names}, which does '
                        f'not contain {self.tiles[t]!r} (config order '
                        f'{self.tiles}). Re-run step 5 with the same tile list.')
                idx = names.index(self.tiles[t])
                if self.rank == 0:
                    logger.warning(f'{path}:{key} tile order {names} differs from '
                                   f'the configured {self.tiles}; taking row '
                                   f'{idx} for {self.tiles[t]!r}')
        return table[idx]

    # ------------------------------------------------------------------- reads

    def read_pos(self, out=None):
        """Positions for this rank's angles: cshifts_final + tile_offsets.

        Step 5 splits a tile offset into an integer paste origin plus a
        fractional shift; here there is no paste, so the whole offset goes into
        the shift. Writing out the mosaic-frame window centre both ways,

            step5: origin_x[t] + (nobj_tile_bin-1)/2 - frac[t,1] - cs*scale - rcs
                   with origin_x[t] = (nobj_bin - nobj_tile_bin)//2 - ioff[t,1]
                 = (nobj_bin-1)/2 - off_bin[t,1] - cs*scale - rcs
            step6: (nobj-1)/2 - pos[...,1],   nobj = step 5's nobj_bin

        gives pos = (cshifts_final + tile_offsets) * scale + rcs term, exactly
        (both halves of the //2 split are even, so no half-pixel drift).
        """
        local_ntheta = self.end_theta - self.st_theta
        ids  = self.ids[self.st_theta:self.end_theta]
        nd   = self.ndist_tile
        pos  = np.empty((local_ntheta, self.ndist, 2), dtype='float32')
        offs = np.zeros((self.ntiles, 2), dtype='float32')

        for t, path in enumerate(self.tile_files):
            with h5py.File(path, 'r', driver="mpio", comm=self.comm) as fid:
                cs       = np.asarray(fid['/exchange/cshifts_final'][ids, :nd],
                                      dtype='float32')
                offs[t]  = self._tile_offset(fid, t, path)
            pos[:, t * nd:(t + 1) * nd] = cs + offs[t][None, None, :]

        scale = np.float32(1.0 / 2**self.bin)
        pos *= scale
        pos[..., 1] += np.float32(self.rotation_center_shift * scale
                                  + 0.5 * (scale - 1))
        self.tile_offsets = offs

        if self.rank == 0:
            logger.info(f'read_pos: cshifts_final + tile_offsets, '
                        f'rotation_center_shift={self.rotation_center_shift:.4f} '
                        f'@ bin={self.bin}')
            for t in range(self.ntiles):
                logger.info(f'  {self.tiles[t]:<10s} tile_offset '
                            f'v={offs[t, 0]:+9.4f} h={offs[t, 1]:+11.4f} '
                            f'finest-grid px  ->  v={offs[t, 0] * scale:+8.3f} '
                            f'h={offs[t, 1] * scale:+10.3f} bin px')

        if out is None:
            return cp.array(pos)
        out[:] = cp.array(pos)
        return out

    def read_data(self, out=None):
        """Projection data for this rank's angles, all tiles, tile-major."""
        nz, n = self.nz, self.n
        local_ntheta = self.end_theta - self.st_theta
        if out is None:
            out = np.empty([local_ntheta, self.ndist, nz, n], dtype='float32')
        # Batch reads to stay under 2^31 bytes (MPI-IO uses int transfer sizes)
        batch = max(1, (1 << 28) // (nz * n))
        for t, path in enumerate(self.tile_files):
            with h5py.File(path, 'r', driver="mpio", comm=self.comm) as fid:
                for k in range(self.ndist_tile):
                    ds = fid[f'/exchange/pdata{k}_{self.bin}']
                    nz0 = ds.shape[1]
                    st, end = nz0 // 2 - nz // 2, nz0 // 2 + nz // 2
                    kk = t * self.ndist_tile + k
                    for i0 in range(0, local_ntheta, batch):
                        i1 = min(i0 + batch, local_ntheta)
                        out[i0:i1, kk] = ds[
                            self.ids[self.st_theta + i0:self.st_theta + i1], st:end]
                    np.sqrt(out[:, kk], out=out[:, kk])
        return out

    def read_ref(self, out=None):
        """Flat fields for all tiles, read on rank 0 and broadcast."""
        nz, n = self.nz, self.n
        raw_np = np.empty((self.ndist, nz, n), dtype='float32')
        if self.rank == 0:
            for t, path in enumerate(self.tile_files):
                with h5py.File(path, 'r') as fid:
                    key = f'/exchange/pref_{self.bin}'
                    nz0 = fid[key].shape[1]
                    st, end = nz0 // 2 - nz // 2, nz0 // 2 + nz // 2
                    raw_np[t * self.ndist_tile:(t + 1) * self.ndist_tile] = \
                        fid[key][:self.ndist_tile, st:end]
        self.comm.Bcast(raw_np, root=0)
        raw = cp.array(raw_np)
        if out is None:
            return cp.sqrt(raw)
        cp.sqrt(raw, out=out)
        return out

    def read_prb(self, prb_file=None, out=None):
        """Initialise the ndist = ntiles*ndist_tile probes.

        Each tile is its own scan with its own measured flat field, so the
        probes stay independent. A probe file holding only ``ndist_tile``
        entries (one per distance, as a single-tile run writes) is repeated
        across the tiles as a starting guess.
        """
        if out is None:
            out = cp.empty([self.ndist, self.nz, self.n], dtype='complex64')
        if not prb_file:
            out[:] = 1
            return out

        with h5py.File(prb_file, 'r') as _f:
            nprb = _f['prb_amp'].shape[0]
            if nprb == self.ndist:
                src = list(range(self.ndist))
            elif nprb == self.ndist_tile:
                src = [k % self.ndist_tile for k in range(self.ndist)]
                if self.rank == 0:
                    logger.info(f'read_prb: {prb_file} holds {nprb} probes '
                                f'(one per distance); repeating them across the '
                                f'{self.ntiles} tiles')
            else:
                raise ValueError(f'{prb_file}: {nprb} probes, expected '
                                 f'{self.ndist} (tile x distance) or '
                                 f'{self.ndist_tile} (per distance)')
            for kk, k in enumerate(src):
                prb = (_f['prb_amp'][k]
                       * np.exp(1j * _f['prb_phase'][k])).astype('complex64')
                nz0, n0 = prb.shape
                if nz0 > self.nz or n0 > self.n:
                    bz, bn = nz0 // self.nz, n0 // self.n
                    prb = prb.reshape(self.nz, bz, self.n, bn).mean(axis=(1, 3))
                out[kk] = cp.array(prb)
            if self.rank == 0:
                logger.info(f'read_prb: {self.ndist} probes from {prb_file} '
                            f'(file shape {tuple(_f["prb_amp"].shape)})')
        return out

    def read_demagnifications(self, out=None):
        """(1 + shrink) / norm_magnifications over the tile x distance axis.

        Same definition as ``Reader.read_demagnifications``, with the shrink of
        each tile read from that tile's own ``/exchange/shrink`` (see
        ``read_shrink``). ``rec_mpi_shrink.Rec`` does not use this — it derives
        demag from the fitted ``vars['tp']`` — but ``rec_mpi.Rec`` does, and
        both must mean the same thing.
        """
        shrink_nd = self.read_shrink()
        nm = cp.array(self._norm_magnifications(), dtype='float32')
        data = (1.0 + shrink_nd) / nm[None, :, None]
        if self.rank == 0:
            logger.info('read_demagnifications: 1/norm_magnifications = '
                        + np.array2string(1.0 / self._norm_magnifications()[
                            :self.ndist_tile], precision=6)
                        + f' (repeated for all {self.ntiles} tiles), scaled by '
                        f'(1 + shrink) per angle')
        if out is not None:
            out[:] = data
            return out
        return data

    def read_shrink(self, out=None):
        """[local_ntheta, ndist, 2] raw shrink for this rank's angles, tile-major.

        Each tile is its own scan and was fitted its own shrinkage, so the
        values come from that tile's ``/exchange/shrink`` — shape
        (ntheta_file, ndist_tile, 2), written by step 3 of ``steps15.py`` from
        the tile's ``shapp.mat``/``shrink_list.mat`` — and land at
        ``t * ndist_tile + k``, matching how ``read_data`` lays out the frames.

        As in ``Reader.read_shrink``: a tile with no ``/exchange/shrink``
        contributes zeros (that tile then runs unshrunk), and a legacy 2-D
        (theta, dist) dataset is broadcast to both (y, x) axes.
        """
        local_ntheta = self.end_theta - self.st_theta
        ids = self.ids[self.st_theta:self.end_theta]
        nd  = self.ndist_tile
        shrink = np.zeros((local_ntheta, self.ndist, 2), dtype='float32')

        for t, path in enumerate(self.tile_files):
            with h5py.File(path, 'r', driver="mpio", comm=self.comm) as fid:
                if '/exchange/shrink' not in fid:
                    if self.rank == 0:
                        logger.warning(
                            f'read_shrink: /exchange/shrink absent in {path} '
                            f'(tile {self.tiles[t]}), using shrink=0 for it')
                    continue
                raw = fid['/exchange/shrink']
                if raw.ndim == 3:
                    blk = np.asarray(raw[ids, :nd, :2], dtype='float32')
                else:
                    # Legacy 2D dataset with a single scalar shrink per (j, k).
                    blk = np.asarray(raw[ids, :nd], dtype='float32')[..., None]
                    blk = np.broadcast_to(blk, (local_ntheta, nd, 2))
                shrink[:, t * nd:(t + 1) * nd] = blk
                if self.rank == 0:
                    logger.info(f'read_shrink: {self.tiles[t]:<10s} '
                                f'/exchange/shrink {raw.shape} -> ndist '
                                f'[{t * nd}:{(t + 1) * nd}), '
                                f'|shrink| max={np.abs(blk).max():.4e}')

        data = cp.asarray(shrink)
        if out is not None:
            out[:] = data
            return out
        return data

    def read_obj(self, out=None):
        """Read the step-5 mosaic Paganin volume for this rank's z-slice.

        Same body as ``Reader.read_obj``, except that the file is the mosaic's
        ``*_obj.h5`` (not a tile's) and the imaginary part is accepted under
        either spelling: ``steps15.py`` writes ``obj_init_imag{p}_{b}`` while
        ``Reader`` only ever looked for ``obj_init_im{p}_{b}``.
        """
        obj_file = self.mosaic_file.replace('.h5', '_obj.h5')
        if not os.path.exists(obj_file):
            raise FileNotFoundError(
                f'{obj_file} not found — this is what step 5 writes the mosaic '
                f'Paganin/FBP volume to. Run step 5 first, or point the config '
                f'at the right path/pfile.')
        with h5py.File(obj_file, 'r', driver="mpio", comm=self.comm) as fid:
            re_key = f'/exchange/obj_init_re{self.paganin}_{self.bin}'
            if re_key not in fid:
                have = sorted(k for k in fid.get('/exchange', {})
                              if k.startswith('obj_init_re'))
                raise KeyError(f'{obj_file}: {re_key} not found; it has {have}')
            obj_ds_re = fid[re_key]
            obj_ds_im = None
            for im_key in (f'/exchange/obj_init_imag{self.paganin}_{self.bin}',
                           f'/exchange/obj_init_im{self.paganin}_{self.bin}'):
                if im_key in fid:
                    obj_ds_im = fid[im_key]
                    break
            if self.rank == 0:
                logger.info(f'read_obj: {obj_file} {re_key} '
                            f'{obj_ds_re.shape} + '
                            f'{im_key if obj_ds_im is not None else "no imag part"}')

            nzobj0, nobj0 = obj_ds_re.shape[:2]
            stz  = nzobj0 // 2 - self.nzobj // 2
            stx  = nobj0  // 2 - self.nobj  // 2
            endx = nobj0  // 2 + self.nobj  // 2
            local_nz = self.end_obj - self.st_obj
            if out is None:
                out = np.empty([local_nz, self.nobj, self.nobj],
                               dtype=self.obj_dtype)
            batch = max(1, (1 << 28) // (self.nobj * self.nobj
                                         * obj_ds_re.dtype.itemsize))
            for i0 in range(0, local_nz, batch):
                i1 = min(i0 + batch, local_nz)
                sl = (slice(stz + self.st_obj + i0, stz + self.st_obj + i1),
                      slice(stx, endx), slice(stx, endx))
                if out.dtype == np.complex64:
                    out[i0:i1].real[:] = obj_ds_re[sl]
                    out[i0:i1].imag[:] = obj_ds_im[sl] if obj_ds_im is not None else 0
                else:
                    out[i0:i1] = obj_ds_re[sl]
        return out

    @staticmethod
    def mosaic_obj_shape(mosaic_file, paganin, bin):
        """(nzobj, nobj) of the step-5 mosaic volume, for auto-sizing.

        Plain (non-MPI) open — call on rank 0 and broadcast.
        """
        obj_file = mosaic_file.replace('.h5', '_obj.h5')
        with h5py.File(obj_file, 'r') as fid:
            key = f'/exchange/obj_init_re{paganin}_{bin}'
            if key not in fid:
                have = sorted(k for k in fid.get('/exchange', {})
                              if k.startswith('obj_init_re'))
                raise KeyError(f'{obj_file}: {key} not found; it has {have}')
            shape = fid[key].shape
        if shape[1] != shape[2]:
            raise ValueError(f'{obj_file}: {key} is {shape}, expected the two '
                             f'transverse axes to match')
        return int(shape[0]), int(shape[1])
