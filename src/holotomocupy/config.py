import os
import configparser
from types import SimpleNamespace

def get_list(c, key, cast=str, sep=","):
    s = c.get(key, fallback="")
    return [cast(x.strip()) for x in s.split(sep) if x.strip()]


def pfile_tile(pfile, tile, scan_suffix=""):
    """Per-tile scan prefix, e.g. YY037A_HT_F2_100nm_center_0001.

    Same rule as ``_pfile_tile`` in the mosaic ``steps15.py`` (which is what
    named the files in the first place); keep the two in step.
    """
    if not tile:
        return pfile
    if scan_suffix:
        return f"{pfile}_{tile}_{scan_suffix}"
    return f"{pfile}_{tile}"

def parse_args(config_file):
    parser = configparser.ConfigParser(inline_comment_prefixes=("#",))
    with open(config_file, "r", encoding="utf-8") as f:
        # Pretend everything belongs to a DEFAULT section
        parser.read_string("[DEFAULT]\n" + f.read())
    cfg = parser["DEFAULT"]

    try:
        args = SimpleNamespace()
        args.pfile    = cfg.get("pfile",    fallback=None)
        args.path_out = cfg.get("path_out").rstrip('/')
        # `path` points to the input directory (where the step1–5 HDF5 lives
        # along with auxiliary inputs like probe files). Defaults to
        # `path_out` for backward compatibility with configs that kept
        # input and output in the same directory.
        args.path     = cfg.get("path", fallback=args.path_out).rstrip('/')
        _path         = args.path
        if args.pfile:
            args.in_file = f"{args.path}/{args.pfile}.h5"
        else:
            args.in_file = os.path.join(_path, cfg.get("in_file"))
        # --- mosaic mode -------------------------------------------------
        # Empty `tiles` → single-tile, everything below is a no-op and the
        # config behaves exactly as before. With tiles listed (left to right on
        # the mosaic, the same order steps15.py used), the data lives in one h5
        # per tile and the initial object in the shared `{pfile}_mosaic_obj.h5`.
        # `ndist` in the config stays PER TILE; the driver expands it.
        args.tiles       = get_list(cfg, "tiles", str)
        args.scan_suffix = cfg.get("scan_suffix", fallback="").strip()
        if args.tiles:
            if not args.pfile:
                raise ValueError("tiles= requires pfile=")
            args.tile_files = [
                f"{args.path}/{pfile_tile(args.pfile, t, args.scan_suffix)}.h5"
                for t in args.tiles
            ]
            args.mosaic_file = f"{args.path}/{args.pfile}_mosaic.h5"
            # Only ever used to locate the `*_obj.h5` next to it; the mosaic
            # file itself is never written by step 5.
            args.in_file = args.mosaic_file
        else:
            args.tile_files  = None
            args.mosaic_file = None
        args.ntheta = cfg.getint("ntheta")
        args.start_theta = cfg.getint("start_theta")
        args.nz = cfg.getint("nz")
        args.n = cfg.getint("n")
        # 0 = let the driver size the object from the initial-guess volume on
        # disk. Only the mosaic step6 does that so far (step 5 picks the mosaic
        # grid from the measured tile spread, which is not obvious from a
        # config); everywhere else these stay explicit.
        args.nzobj = cfg.getint("nzobj")
        args.nobj = cfg.getint("nobj")
        args.ndist = cfg.getint("ndist")
        args.obj_dtype = cfg.get("obj_dtype")
        args.paganin = cfg.getint("paganin")
        args.mask = cfg.getfloat("mask")
        args.lam_prbfit    = cfg.getfloat("lam_prbfit")
        args.lam_laplacian = cfg.getfloat("lam_laplacian", fallback=0.0)
        args.rho = get_list(cfg, "rho", float)
        # Optional rho tuning (rec_mpi_shrink): False → args.rho used as-is.
        args.estimate_rho       = cfg.getboolean("estimate_rho",       fallback=False)
        args.rho_estimate_niter = cfg.getint    ("rho_estimate_niter", fallback=16)
        args.niter = cfg.getint("niter")
        args.nchunk = cfg.getint("nchunk")
        args.checkpoint_step = cfg.getint("checkpoint_step")
        args.error_step      = cfg.getint("error_step")
        args.start_iter = cfg.getint("start_iter")
        args.rotation_center_shift = cfg.getfloat("rotation_center_shift")
        args.bin = cfg.getint("bin")
        args.log_level = cfg.get("log_level", fallback="WARNING")
        args.energy = cfg.getfloat("energy", fallback=None)
        args.method = cfg.getint("method", fallback=0)
        args.start_method = cfg.getint("start_method", fallback=1)
        args.shift_type = cfg.get("shift_type", fallback="cubic").strip().lower()
        _pos_chk            = cfg.get("pos_checkpoint", fallback=None)
        args.pos_checkpoint = os.path.join(_path, _pos_chk) if _pos_chk else None
        _prb                = cfg.get("prb_file", fallback=None)
        args.prb_file       = os.path.join(args.path, _prb) if _prb else None
        _init_vol           = cfg.get("init_vol",        fallback=None)
        args.init_vol       = _init_vol.strip() if _init_vol and _init_vol.strip() else None
        args.init_vol_scale = cfg.getfloat("init_vol_scale", fallback=1.0)
        # Optional NXtomo file to source geometry (energy, z1, focustodetectordistance,
        # detector_pixelsize) from. When set, the caller (e.g. step6.py) overrides
        # these fields after reading from the converted HDF5 archive.
        _nx_file        = cfg.get("nx_file", fallback=None)
        if _nx_file and _nx_file.strip():
            _nx_file = _nx_file.strip()
            args.nx_file = _nx_file if os.path.isabs(_nx_file) \
                else os.path.join(_path, _nx_file)
        else:
            args.nx_file = None
    except configparser.NoOptionError as e:
        raise ValueError(f"Missing required field in {config_file}: {e}") from e

    return args


def read_nx_geometry(nx_path):
    """Read holotomography geometry from one ESRF NXtomo (.nx) file.

    Tested against the `nxtomomill_edf2nx` output (ID16A flavour) where:
      /entry/instrument/beam/incident_energy   : energy in keV
      /entry/instrument/detector/data          : (nframes, ny, nx) float32
      /entry/instrument/detector/x_pixel_size  : sample-plane pixel size, metres
      /entry/instrument/detector/distance      : Fresnel propagation distance
                                                 D = z1*z2/(z1+z2), metres
                                                 (NOT the geometric z1)
      /entry/instrument/detector/image_key     : 0=data, 1=flat, 2=dark
      /entry/sample/rotation_angle             : per-frame angle in degrees
      /entry/sample/{x,y,z}_translation        : per-frame stage position
      /entry/instrument/positioners/0/z1h        (optional, when present): all
                                                 focus-to-sample distances z1
                                                 (matches .info SourceDistance)
      /entry/instrument/positioners/0/distancesh : Fresnel D per distance, NOT z1
      /entry/instrument/positioners/0/Mh         (optional): per-distance
                                                 magnifications M = Z / z1

    Returns a dict with:
      entry, energy (keV), sample_pixelsize (m, at the sample plane for distance 0),
      z1 (m, ndarray — focus-to-sample distance per plane; length 1 if NX only had a scalar),
      magnifications (ndarray, M = Z/z1 per distance; NaN if the NX file lacks Mh),
      detector_pixelsize (m, sample_pixelsize * M[0]; NaN if M unknown),
      focustodetectordistance (m, M[0] * z1[0]; NaN if M unknown),
      ny, nx, data_ids, flat_ids, dark_ids,
      rotation_angle_deg (per data frame).
    """
    import h5py
    import numpy as np
    with h5py.File(nx_path, 'r') as f:
        entry = next(k for k in f if k.startswith('entry'))
        g = f[entry]

        energy           = float(g['instrument/beam/incident_energy'][()])
        sample_pixelsize = float(g['instrument/detector/x_pixel_size'][()])

        image_key = g['instrument/detector/image_key'][:]
        data_ids  = np.where(image_key == 0)[0]
        flat_ids  = np.where(image_key == 1)[0]
        dark_ids  = np.where(image_key == 2)[0]

        ny, nx = g['instrument/detector/data'].shape[1:3]

        rot_all = g['sample/rotation_angle'][:].astype('float64')
        rot_data = rot_all[data_ids]

        # Optional: ID16A "positioners/0" blob holds per-distance lists.
        # Prefer z1h (geometric focus-to-sample distance, matches .info
        # SourceDistance) over detector/distance — the latter actually holds
        # the Fresnel propagation distance z1*z2/(z1+z2), not z1 itself.
        z1 = None
        magnifications = np.array([np.nan], dtype='float64')
        try:
            raw = g['instrument/positioners/0/z1h'][()]
            if isinstance(raw, bytes):
                z1 = np.array([float(x) for x in raw.split()], dtype='float64')
        except KeyError:
            pass
        if z1 is None:
            z1 = np.array([float(g['instrument/detector/distance'][()])], dtype='float64')
        try:
            raw = g['instrument/positioners/0/Mh'][()]
            if isinstance(raw, bytes):
                magnifications = np.array([float(x) for x in raw.split()], dtype='float64')
        except KeyError:
            pass
        if magnifications.size != z1.size:
            magnifications = np.full(z1.shape, np.nan)

        # Detector pixel size: read directly from NX when available
        # (positioners/0/pixelsize_detector is Peter's stored value, in metres).
        detector_pixelsize = None
        try:
            raw = g['instrument/positioners/0/pixelsize_detector'][()]
            if isinstance(raw, bytes):
                detector_pixelsize = float(raw.decode().strip())
        except KeyError:
            pass

    # Derived quantities (use ID16A convention M = Z / z1, NOT z2 / z1):
    #   focustodetectordistance Z = M[0] * z1[0]
    # Fall back to sample_pixelsize * M[0] only if pixelsize_detector wasn't
    # stored in NX.
    if not np.isnan(magnifications[0]):
        focustodetectordistance = float(magnifications[0] * z1[0])
        if detector_pixelsize is None:
            detector_pixelsize = float(sample_pixelsize * magnifications[0])
    else:
        focustodetectordistance = float('nan')
        if detector_pixelsize is None:
            detector_pixelsize = float('nan')

    return dict(
        entry=entry,
        energy=energy,
        sample_pixelsize=sample_pixelsize,
        z1=z1,
        magnifications=magnifications,
        detector_pixelsize=detector_pixelsize,
        focustodetectordistance=focustodetectordistance,
        ny=int(ny), nx=int(nx),
        data_ids=data_ids, flat_ids=flat_ids, dark_ids=dark_ids,
        rotation_angle_deg=rot_data,
    )


def parse_args_step0(config_file):
    parser = configparser.ConfigParser(inline_comment_prefixes=("#",), interpolation=None)
    with open(config_file, "r", encoding="utf-8") as f:
        parser.read_string("[DEFAULT]\n" + f.read())
    cfg = parser["DEFAULT"]

    try:
        args = SimpleNamespace()
        path             = cfg.get("path").rstrip('/')
        args.path_out    = cfg.get("path_out").rstrip('/')
        args.scan_file   = os.path.join(path, cfg.get("scan_file"))
        args.meta_file   = os.path.join(path, cfg.get("meta_file"))
        args.h5_out      = os.path.join(args.path_out, cfg.get("h5_out"))
        args.dataset_ids = [int(x.strip()) for x in cfg.get("dataset_ids").split(",") if x.strip()]
        args.n           = cfg.getint("n",        fallback=2048)
        args.niter       = cfg.getint("niter",    fallback=129)
        args.nchunk      = cfg.getint("nchunk",   fallback=4)
        args.checkpoint_step = cfg.getint("checkpoint_step", fallback=32)
        args.error_step      = cfg.getint("error_step",      fallback=32)
        args.rho         = [float(x.strip()) for x in cfg.get("rho").split(",") if x.strip()]
        args.log_level   = cfg.get("log_level",   fallback="INFO")
        args.shift_type  = cfg.get("shift_type", fallback="cubic").strip().lower()
    except configparser.NoOptionError as e:
        raise ValueError(f"Missing required field in {config_file}: {e}") from e

    return args


def parse_args_step0_nx(config_file):
    """Parse config for step0.py reading ESRF NXtomo (.nx) files."""
    parser = configparser.ConfigParser(inline_comment_prefixes=("#",), interpolation=None)
    with open(config_file, "r", encoding="utf-8") as f:
        parser.read_string("[DEFAULT]\n" + f.read())
    cfg = parser["DEFAULT"]

    try:
        args = SimpleNamespace()
        path          = cfg.get("path").rstrip('/')
        args.path_out = cfg.get("path_out").rstrip('/')
        args.nx_file  = os.path.join(path, cfg.get("nx_file"))
        args.h5_out   = os.path.join(args.path_out, cfg.get("h5_out"))
        args.n        = cfg.getint("n",        fallback=2048)
        args.niter    = cfg.getint("niter",    fallback=129)
        args.nchunk   = cfg.getint("nchunk",   fallback=4)
        args.checkpoint_step = cfg.getint("checkpoint_step", fallback=-1)
        args.error_step      = cfg.getint("error_step",      fallback=32)
        args.rho      = [float(x.strip()) for x in cfg.get("rho").split(",") if x.strip()]
        args.log_level = cfg.get("log_level", fallback="INFO")
        args.shift_type = cfg.get("shift_type", fallback="cubic").strip().lower()
    except configparser.NoOptionError as e:
        raise ValueError(f"Missing required field in {config_file}: {e}") from e

    return args


def parse_args_steps15(config_file):
    parser = configparser.ConfigParser(inline_comment_prefixes=("#",))
    with open(config_file, "r", encoding="utf-8") as f:
        parser.read_string("[DEFAULT]\n" + f.read())
    cfg = parser["DEFAULT"]

    try:
        args = SimpleNamespace()
        args.path     = cfg.get("path").rstrip('/')
        args.pfile    = cfg.get("pfile")
        _path_out     = cfg.get("path_out", fallback=None)
        args.path_out = _path_out.strip() if _path_out else None
        args.start_step            = cfg.getint("start_step",            fallback=1)
        args.start_level_rec       = cfg.getint("start_level_rec",       fallback=0)
        args.rotation_center_shift = cfg.getfloat("rotation_center_shift", fallback=0.0)
        args.nlevels  = cfg.getint("nlevels",  fallback=4)
        args.paganin  = cfg.getfloat("paganin", fallback=120.0)
        args.nchunk   = cfg.getint("nchunk",   fallback=16)
        args.ref_dist = cfg.getint("ref_dist", fallback=0)
        # Shrinkage angle behavior. False (default) = Peter's convention
        # (constant per distance = cum[k] + inc[k]/2). True = linear ramp
        # over angles from cum[k] at angle 0 to cum[k] + inc[k] at angle ntheta.
        args.shrink_angle_ramp = cfg.getboolean("shrink_angle_ramp", fallback=False)
        # Mosaic-tile fields. Empty tiles → single-tile mode (backward compatible).
        args.tiles           = get_list(cfg, "tiles", str)
        args.scan_suffix     = cfg.get("scan_suffix", fallback="").strip()
        # The order the tiles were ACQUIRED in, which is not the left-to-right
        # `tiles` order. The sample keeps shrinking for the whole session, so a
        # tile's shrink starts where the previously acquired tile's ended; this
        # is what says which tile that was. Must be a permutation of `tiles`.
        # Empty → no accumulation, each tile's shapp.mat taken as it stands.
        # steps15 only: step 6 reads the accumulated result from
        # /exchange/shrink and refines it, so it needs no order of its own.
        args.tile_order      = get_list(cfg, "tile_order", str)
        args.overlap_width   = cfg.getint("overlap_width",   fallback=200)
        args.overlap_nangles = cfg.getint("overlap_nangles", fallback=10)
        args.overlap_check   = cfg.getboolean("overlap_check", fallback=False)
        # Nominal spacing between adjacent tiles, in object px on the finest
        # grid, decreasing with tile index (tiles are listed left to right).
        # This is only the starting guess — estimate_overlap measures the rest.
        args.tile_step = cfg.getfloat("tile_step", fallback=0.0)
        # Measure the tile placement in step 5, by correlating neighbouring
        # tiles once their distances have been assembled into one projection.
        # Uses overlap_nangles angles, starting from the tile_step guess.
        args.estimate_overlap = cfg.getboolean("estimate_overlap", fallback=False)
        # Reconstruct step 5 from a subset of the angles, for fast tests. 0 (or
        # >= ntheta) uses all of them; otherwise this many, spread evenly over
        # the full angular range, so the FBP still covers 0..180/360 -- just
        # more sparsely. Steps 1-4 always run over every angle.
        args.ntheta_rec = cfg.getint("ntheta_rec", fallback=0)
        # Object grids, both in object px on the finest grid, 0 → auto:
        #   nobj_tile   the tile-local grid   (auto from n and the magnifications)
        #   nobj/nzobj  the mosaic grid the tiles are composited onto in step 5
        #               (auto from the spread of the tile positions)
        # The mosaic pair carries the same two names step 6 uses for the same
        # two numbers — nzobj the height, i.e. the z extent of the volume, nobj
        # the width — except that here they are on the finest grid and step 6
        # takes them divided by 2**bin.
        #
        # In single-tile mode there is no mosaic grid, so a bare `nobj` is the
        # tile grid and is accepted as a spelling of nobj_tile: that is what
        # every single-tile config_steps15.conf in experimental/ still says.
        _n         = cfg.getint("n",         fallback=0)
        _nobj_tile = cfg.getint("nobj_tile", fallback=0)
        _nobj      = cfg.getint("nobj",      fallback=0)
        _nzobj     = cfg.getint("nzobj",     fallback=0)
        if not args.tiles:
            _nobj_tile = _nobj_tile or _nobj
        args.n         = _n         if _n         > 0 else None
        args.nobj_tile = _nobj_tile if _nobj_tile > 0 else None
        args.nobj      = _nobj      if _nobj      > 0 else None
        args.nzobj     = _nzobj     if _nzobj     > 0 else None
        args.log_level = cfg.get("log_level", fallback="INFO")
        # Optional NXtomo file to source geometry (energy, z1, focustodetectordistance,
        # detector_pixelsize) from, instead of the per-distance HDF5 scan files.
        # Resolved relative to `path` if not absolute.
        _nx_file       = cfg.get("nx_file", fallback=None)
        if _nx_file and _nx_file.strip():
            _nx_file = _nx_file.strip()
            args.nx_file = _nx_file if os.path.isabs(_nx_file) \
                else os.path.join(args.path, _nx_file)
        else:
            args.nx_file = None
    except configparser.NoOptionError as e:
        raise ValueError(f"Missing required field in {config_file}: {e}") from e

    return args
