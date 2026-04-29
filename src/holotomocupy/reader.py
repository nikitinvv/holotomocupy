import glob
import math
import os
import h5py
import numpy as np
import cupy as cp
from .logger_config import logger


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


def load_shrink_from_mats(path, pfile, ndist, ntheta):
    """Build [ntheta, ndist] shrink array from per-distance shrink_list.mat files.

    Each shrink_list.mat contains a (3,2) matrix; row 0 gives [h, v] incremental
    shrink from the previous distance plane. The per-angle shrink for distance k at
    angle j is linearly interpolated: cumulative[k] + increments[k] * j / ntheta.
    Returns a zero array if any mat file is missing.
    """
    increments = []
    for k in range(ndist):
        mat_path = f'{path}/{pfile}_{k + 1}_/shrink_list.mat'
        if not os.path.exists(mat_path):
            return np.zeros((ntheta, ndist), dtype='float32')
        sl = load_octave_text_mat(mat_path, 'shrink_list')
        increments.append(float(sl[0, 0] + sl[0, 1]) / 2)
    cumulative = np.concatenate([[0.0], np.cumsum(increments)])[:ndist]
    j_frac = np.arange(ntheta) / ntheta
    shrink_nd = cumulative[None, :] + np.array(increments)[None, :] * j_frac[:, None]
    return shrink_nd.astype('float32')


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
            self.theta = -fid['/exchange/theta'][:, 0][self.ids] / 180 * np.pi
            self.detector_pixelsize *= 2**self.bin
            
    def read_obj(self, out=None):
        """Read initial object guess for this rank's z-slice into out."""
        # obj_init may be in a separate _obj.h5 file (written there by step 5
        # to avoid the ~16 TiB Lustre per-file size limit).
        obj_file = self.in_file.replace('.h5', '_obj.h5')
        if not os.path.exists(obj_file):
            obj_file = self.in_file
        print(f"read object from {obj_file}")
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
                if self.obj_dtype == 'complex64':
                    out[i0:i1].real[:] = obj_ds_re[sl]
                    out[i0:i1].imag[:] = obj_ds_im[sl] if obj_ds_im is not None else 0
                else:
                    out[i0:i1] = obj_ds_re[sl]
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

    def read_shrink(self, out=None):
        """Read [local_ntheta, ndist] shrink for this rank's theta-slice from HDF5.

        Falls back to zeros if /exchange/shrink is not present (e.g. old files).
        Writes into `out` if provided, otherwise returns a new array.
        """
        local_ntheta = self.end_theta - self.st_theta
        with h5py.File(self.in_file, 'r', driver="mpio", comm=self.comm) as fid:
            if '/exchange/shrink' not in fid:
                data = cp.zeros((local_ntheta, self.ndist), dtype='float32')
            else:
                data = cp.array(fid['/exchange/shrink'][
                    self.ids[self.st_theta:self.end_theta], :self.ndist
                ].astype('float32'))
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
                    print(f'Probe read from {prb_file}, shape {tuple(_f["prb_amp"].shape)}', flush=True)
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

    def read_checkpoint(self, path, out_obj=None, out_prb=None, out_pos=None):
        """Read a checkpoint saved at a coarser resolution and upsample.

        Scale is inferred automatically from checkpoint n vs self.n.

        prb  : upsampled in y and x by scale (repeat).
        obj  : upsampled in x and y by scale (repeat); z mapped by nearest-neighbour.
        pos  : multiplied by scale (pixel coords scale with resolution).
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
        if scale > 1:
            from cupyx.scipy.ndimage import shift
            shift_val = 0# -0.5 * (scale - 1)
            out_prb[:] = shift(out_prb, shift=(0, 0, shift_val), order=3, mode='nearest')

        # --- obj: parallel read of each rank's slice, upsample ---
        with h5py.File(path, 'r', driver="mpio", comm=self.comm) as f:
            obj_dtype = f.attrs['obj_dtype']
            st_src  = self.st_obj  // scale
            end_src = self.end_obj // scale
            n0      = self.end_obj - self.st_obj
            nz_src  = max(1, end_src - st_src)
            ds_re   = f['obj_re']
            batch   = max(1, (1 << 28) // (ds_re.shape[1] * ds_re.shape[2] * ds_re.dtype.itemsize))
            obj_re  = np.empty((nz_src,) + ds_re.shape[1:], dtype=ds_re.dtype)
            for i0 in range(0, nz_src, batch):
                i1 = min(i0 + batch, nz_src)
                obj_re[i0:i1] = ds_re[st_src + i0 : st_src + i1]
            if obj_dtype == 'complex64':
                obj_im = np.empty_like(obj_re)
                ds_im  = f['obj_im']
                for i0 in range(0, nz_src, batch):
                    i1 = min(i0 + batch, nz_src)
                    obj_im[i0:i1] = ds_im[st_src + i0 : st_src + i1]
                block = (obj_re + 1j * obj_im).astype('complex64')
            else:
                block = obj_re.astype('float32')
            # upsample x and y
            for axis in [2, 1]:
                block = np.repeat(block, scale, axis=axis)
            # map source z-slices → output z-slices (nearest-neighbour)
            idx0   = np.clip(
                (np.arange(n0) * nz_src / n0).astype(np.intp), 0, nz_src - 1
            )
            if out_obj is None:
                out_obj = block[idx0].astype(self.obj_dtype)
            else:
                out_obj[:] = block[idx0].astype(self.obj_dtype)

            # --- pos: scale pixel coordinates up ---
            pos = f['pos'][self.st_theta:self.end_theta].astype('float32')

        pos_up = pos * scale 
        pos_up[...,1] += 0.5 * (scale - 1)
        if out_pos is None:
            out_pos = cp.array(pos_up)
        else:
            out_pos[:] = cp.array(pos_up, dtype='float32')

        return {'obj': out_obj, 'prb': out_prb, 'pos': out_pos}

    def read_pos_checkpoint(self, path, out=None):
        """Read positions from a checkpoint file and upsample to current resolution.

        Scale is inferred from the checkpoint probe size vs self.n.
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
            pos = f['pos'][self.ids][self.st_theta:self.end_theta].astype('float32')

        pos_up = pos * scale
        pos_up[..., 1] += 0.5 * (scale - 1)
        if out is None:
            out = cp.array(pos_up)
        else:
            out[:] = cp.array(pos_up, dtype='float32')
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

    def read_pos_unbin(self, out):
        """Read initial positions with downsampling."""
        with h5py.File(self.in_file, 'r', driver="mpio", comm=self.comm) as fid:
            pos = cp.array(fid['/exchange/cshifts_final'][
                self.ids[self.st_theta:self.end_theta], :self.ndist
            ].astype('float32'))
        pos[..., 1] += self.rotation_center_shift
        out[:] = pos * 2**(-self.bin)
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
                if self.obj_dtype == 'complex64':
                    out[i].real[:] = acc
                    out[i].imag[:] = 0
                else:
                    out[i][:] = acc

        if scale != 1.0:
            out /= np.float32(scale)
        logger.info(f"read_vol_obj: rank {self.rank} done (scale={scale})")
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
