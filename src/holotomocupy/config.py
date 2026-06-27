import os
import configparser
from types import SimpleNamespace

def get_list(c, key, cast=str, sep=","):
    s = c.get(key, fallback="")
    return [cast(x.strip()) for x in s.split(sep) if x.strip()]

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
        _path         = cfg.get("path", fallback=args.path_out).rstrip('/')
        if args.pfile:
            args.in_file = f"{args.path_out}/{args.pfile}.h5"
        else:
            args.in_file = os.path.join(_path, cfg.get("in_file"))
        args.ntheta = cfg.getint("ntheta")
        args.start_theta = cfg.getint("start_theta")
        args.nz = cfg.getint("nz")
        args.n = cfg.getint("n")
        args.nzobj = cfg.getint("nzobj")
        args.nobj = cfg.getint("nobj")
        args.ndist = cfg.getint("ndist")
        args.obj_dtype = cfg.get("obj_dtype")
        args.paganin = cfg.getint("paganin")
        args.mask = cfg.getfloat("mask")
        args.lam_prbfit    = cfg.getfloat("lam_prbfit")
        args.lam_laplacian = cfg.getfloat("lam_laplacian", fallback=0.0)
        args.rho = get_list(cfg, "rho", float)
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
        args.prb_file       = os.path.join(args.path_out, _prb) if _prb else None
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
      /entry/instrument/detector/distance      : focus-to-sample distance z1, metres
      /entry/instrument/detector/image_key     : 0=data, 1=flat, 2=dark
      /entry/sample/rotation_angle             : per-frame angle in degrees
      /entry/sample/{x,y,z}_translation        : per-frame stage position
      /entry/instrument/positioners/0/distancesh (optional, when present): all
                                                 focus-to-sample distances for the scan
      /entry/instrument/positioners/0/Mh         (optional): per-distance
                                                 magnifications M = Z / z1

    Returns a dict with:
      entry, energy (keV), sample_pixelsize (m, at the sample plane for distance 0),
      z1 (m, scalar — first distance), z1_all (m, ndarray — all distances when known),
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
        z1               = float(g['instrument/detector/distance'][()])

        image_key = g['instrument/detector/image_key'][:]
        data_ids  = np.where(image_key == 0)[0]
        flat_ids  = np.where(image_key == 1)[0]
        dark_ids  = np.where(image_key == 2)[0]

        ny, nx = g['instrument/detector/data'].shape[1:3]

        rot_all = g['sample/rotation_angle'][:].astype('float64')
        rot_data = rot_all[data_ids]

        # Optional: ID16A "positioners/0" blob holds per-distance lists.
        z1_all = np.array([z1], dtype='float64')
        magnifications = np.array([np.nan], dtype='float64')
        try:
            raw = g['instrument/positioners/0/distancesh'][()]
            if isinstance(raw, bytes):
                z1_all = np.array([float(x) for x in raw.split()], dtype='float64')
        except KeyError:
            pass
        try:
            raw = g['instrument/positioners/0/Mh'][()]
            if isinstance(raw, bytes):
                magnifications = np.array([float(x) for x in raw.split()], dtype='float64')
        except KeyError:
            pass
        if magnifications.size != z1_all.size:
            magnifications = np.full(z1_all.shape, np.nan)

    # Derived quantities (use ID16A convention M = Z / z1, NOT z2 / z1):
    #   focustodetectordistance Z = M[0] * z1[0]
    #   detector_pixelsize        = sample_pixelsize * M[0]
    # The sample-plane pixel for distance k is detector_pixelsize / M[k].
    if not np.isnan(magnifications[0]):
        focustodetectordistance = float(magnifications[0] * z1_all[0])
        detector_pixelsize      = float(sample_pixelsize * magnifications[0])
    else:
        focustodetectordistance = float('nan')
        detector_pixelsize      = float('nan')

    return dict(
        entry=entry,
        energy=energy,
        sample_pixelsize=sample_pixelsize,
        z1=z1, z1_all=z1_all,
        magnifications=magnifications,
        detector_pixelsize=detector_pixelsize,
        focustodetectordistance=focustodetectordistance,
        ny=int(ny), nx=int(nx),
        data_ids=data_ids, flat_ids=flat_ids, dark_ids=dark_ids,
        rotation_angle_deg=rot_data,
        # Back-compat alias for callers that already used `pixel_size`:
        pixel_size=sample_pixelsize,
    )


# Back-compat alias for existing callers in this module.
_read_nx_geometry = read_nx_geometry


def parse_args_nx(config_file):
    """Variant of parse_args() that reads geometry from ESRF NXtomo (.nx) files.

    Instead of pulling detector size, energy, angles, and propagation distances from
    a pre-converted /exchange/* HDF5 archive, this function reads them directly
    from one NXtomo file per propagation distance (the `nxtomomill_edf2nx`
    output produced at ID16A).

    Required config keys (geometry side):
      path        : directory containing the NX files (prefix for relative paths)
      path_out    : output directory
      nx_files    : comma-separated list of NX paths, one per distance, in order
                    — OR —
      nx_template : path with one `{}` placeholder for the distance id
      nx_ids      : comma-separated list of integers substituted into the template

    All other keys (nzobj, nobj, obj_dtype, paganin, mask, lam_prbfit,
    lam_laplacian, rho, niter, nchunk, checkpoint_step, error_step, start_iter,
    rotation_center_shift, bin, log_level, method, start_method, shift_type,
    pos_checkpoint, prb_file, init_vol, init_vol_scale, start_theta) are read
    from the config the same way as parse_args(). `ntheta` defaults to the
    number of projection frames found in the NX file; `n` and `nz` default to
    the detector frame width and height from NX.

    Geometry fields filled from NX (override any same-named config keys):
      ntheta, nz, n, ndist, energy, sample_pixelsize, z1 (ndist,),
      theta (ntheta,), ids (ntheta,)
    """
    import numpy as np

    parser = configparser.ConfigParser(inline_comment_prefixes=("#",), interpolation=None)
    with open(config_file, "r", encoding="utf-8") as f:
        parser.read_string("[DEFAULT]\n" + f.read())
    cfg = parser["DEFAULT"]

    try:
        args = SimpleNamespace()
        _path = cfg.get("path").rstrip('/')
        args.path = _path
        args.path_out = cfg.get("path_out").rstrip('/')

        # Resolve the NX file list ------------------------------------------------
        nx_files_raw = cfg.get("nx_files", fallback="")
        if nx_files_raw.strip():
            nx_files = [x.strip() for x in nx_files_raw.split(",") if x.strip()]
        else:
            template = cfg.get("nx_template")
            ids      = [x.strip() for x in cfg.get("nx_ids").split(",") if x.strip()]
            nx_files = [template.format(i) for i in ids]
        nx_files = [p if os.path.isabs(p) else os.path.join(_path, p) for p in nx_files]
        for p in nx_files:
            if not os.path.exists(p):
                raise FileNotFoundError(f"NX file not found: {p}")
        args.nx_files = nx_files
        args.in_file  = nx_files[0]   # for code paths that still expect a single in_file

        # Pull geometry from each NX file ----------------------------------------
        metas = [_read_nx_geometry(p) for p in nx_files]
        m0 = metas[0]

        # ndist + per-distance z1/magnifications:
        #   - single NX file with multiple distances in positioners/0  → use those
        #   - one NX file per distance                                  → assemble
        if len(metas) == 1 and m0["z1_all"].size > 1:
            args.z1 = m0["z1_all"]
            args.magnifications = m0["magnifications"]
        else:
            args.z1 = np.array([m["z1"] for m in metas], dtype="float64")
            args.magnifications = np.array(
                [m["magnifications"][0] for m in metas], dtype="float64")
        args.ndist = int(args.z1.size)

        ntheta_full = int(len(m0["data_ids"]))
        args.ntheta = cfg.getint("ntheta", fallback=ntheta_full)
        args.start_theta = cfg.getint("start_theta", fallback=0)
        step = ntheta_full / args.ntheta
        ids  = np.arange(args.start_theta, ntheta_full, step)[:args.ntheta].astype("int")
        args.ids = ids

        args.nz = cfg.getint("nz", fallback=m0["ny"])
        args.n  = cfg.getint("n",  fallback=m0["nx"])

        args.energy             = float(m0["energy"])
        args.sample_pixelsize   = float(m0["sample_pixelsize"])
        args.detector_pixelsize = float(m0["detector_pixelsize"])
        args.focustodetectordistance = float(m0["focustodetectordistance"])

        # theta in radians, same sign convention as Reader.__init__
        rot_deg = m0["rotation_angle_deg"][ids]
        args.theta = (-rot_deg / 180.0 * np.pi).astype("float64")

        # Reconstruction hyperparameters (same contract as parse_args) -----------
        args.pfile     = cfg.get("pfile", fallback=None)
        args.nzobj     = cfg.getint("nzobj", fallback=args.nz)
        args.nobj      = cfg.getint("nobj",  fallback=args.n)
        args.obj_dtype = cfg.get("obj_dtype")
        args.paganin   = cfg.getint("paganin")
        args.mask      = cfg.getfloat("mask")
        args.lam_prbfit    = cfg.getfloat("lam_prbfit")
        args.lam_laplacian = cfg.getfloat("lam_laplacian", fallback=0.0)
        args.rho       = get_list(cfg, "rho", float)
        args.niter     = cfg.getint("niter")
        args.nchunk    = cfg.getint("nchunk")
        args.checkpoint_step = cfg.getint("checkpoint_step")
        args.error_step      = cfg.getint("error_step")
        args.start_iter      = cfg.getint("start_iter")
        args.rotation_center_shift = cfg.getfloat("rotation_center_shift")
        args.bin       = cfg.getint("bin")
        args.log_level = cfg.get("log_level", fallback="WARNING")
        args.method        = cfg.getint("method",        fallback=0)
        args.start_method  = cfg.getint("start_method",  fallback=1)
        args.shift_type    = cfg.get("shift_type",       fallback="cubic").strip().lower()
        _pos_chk            = cfg.get("pos_checkpoint", fallback=None)
        args.pos_checkpoint = os.path.join(_path, _pos_chk) if _pos_chk else None
        _prb                = cfg.get("prb_file", fallback=None)
        args.prb_file       = os.path.join(args.path_out, _prb) if _prb else None
        _init_vol           = cfg.get("init_vol", fallback=None)
        args.init_vol       = _init_vol.strip() if _init_vol and _init_vol.strip() else None
        args.init_vol_scale = cfg.getfloat("init_vol_scale", fallback=1.0)
    except configparser.NoOptionError as e:
        raise ValueError(f"Missing required field in {config_file}: {e}") from e

    return args


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
        _n            = cfg.getint("n",    fallback=0)
        _nobj         = cfg.getint("nobj", fallback=0)
        args.n        = _n    if _n    > 0 else None
        args.nobj     = _nobj if _nobj > 0 else None
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
