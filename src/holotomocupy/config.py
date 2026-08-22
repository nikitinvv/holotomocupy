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
        args.paganin = cfg.getint("paganin")
        args.mask = cfg.getfloat("mask")
        args.lam_prbfit    = cfg.getfloat("lam_prbfit")
        args.lam_laplacian = cfg.getfloat("lam_laplacian", fallback=0.0)
        args.rho = get_list(cfg, "rho", float)
        # Optional rho tuning: False (default) -> args.rho used as-is; True ->
        # BH first coordinate-searches rho[prb, pos] around args.rho with short
        # trials of rho_estimate_niter iterations. See Rec.estimate_rho_coord.
        args.estimate_rho       = cfg.getboolean("estimate_rho",       fallback=False)
        args.rho_estimate_niter = cfg.getint    ("rho_estimate_niter", fallback=16)
        # Derive the CG beta/alpha from one Hessian sweep instead of three
        # (see the fused_hessian note on Rec). check_fused_hessian re-measures
        # every form the classic path would have measured and logs the relative
        # disagreement — 4 extra sweeps per iteration, for verification only.
        args.fused_hessian       = cfg.getboolean("fused_hessian",       fallback=True)
        args.check_fused_hessian = cfg.getboolean("check_fused_hessian", fallback=False)
        args.niter = cfg.getint("niter")
        args.nchunk = cfg.getint("nchunk")
        # How many distances share one upload of a theta chunk of proj inside
        # the cascade kernels. 0 = all of them (the default), 1 = the old
        # outer-distance loop. See Rec._resolve_ndistchunk.
        args.ndistchunk = cfg.getint("ndistchunk", fallback=0)
        args.checkpoint_step = cfg.getint("checkpoint_step")
        args.error_step      = cfg.getint("error_step")
        # Internal instrumentation (cache hit/miss counters and the like).
        # -1 = never, the default: these numbers only matter when tuning the
        # solver, not when running it.
        args.debug_step      = cfg.getint("debug_step", fallback=-1)
        args.start_iter = cfg.getint("start_iter")
        args.rotation_center_shift = cfg.getfloat("rotation_center_shift")
        args.bin = cfg.getint("bin")
        args.log_level = cfg.get("log_level", fallback="WARNING")
        args.energy = cfg.getfloat("energy", fallback=None)
        args.method = cfg.getint("method", fallback=0)
        args.start_method = cfg.getint("start_method", fallback=1)
        _pos_chk            = cfg.get("pos_checkpoint", fallback=None)
        args.pos_checkpoint = os.path.join(_path, _pos_chk) if _pos_chk else None
        _prb                = cfg.get("prb_file", fallback=None)
        args.prb_file       = os.path.join(args.path_out, _prb) if _prb else None
        _init_vol           = cfg.get("init_vol",        fallback=None)
        args.init_vol       = _init_vol.strip() if _init_vol and _init_vol.strip() else None
        args.init_vol_scale = cfg.getfloat("init_vol_scale", fallback=1.0)
        # Mosaic: one .h5 per tile, {path_out}/{pfile}_{tile}.h5.  Empty tiles=
        # is single-tile mode and leaves everything below None.  mosaic_file is
        # a NAME, not a file that has to exist: MosaicReader only uses it to
        # find the shared initial object {pfile}_obj.h5, exactly as step6 of
        # the YY037A pipeline uses {pfile}_mosaic.h5.
        args.tiles = get_list(cfg, "tiles", str)
        if args.tiles:
            if not args.pfile:
                raise ValueError("tiles= requires pfile=")
            args.tile_files  = [f"{args.path_out}/{args.pfile}_{t}.h5"
                                for t in args.tiles]
            args.mosaic_file = args.in_file
        else:
            args.tile_files  = None
            args.mosaic_file = None
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
    except configparser.NoOptionError as e:
        raise ValueError(f"Missing required field in {config_file}: {e}") from e

    return args


def parse_args_steps15(config_file):
    parser = configparser.ConfigParser(inline_comment_prefixes=("#",))
    with open(config_file, "r", encoding="utf-8") as f:
        parser.read_string("[DEFAULT]\n" + f.read())
    cfg = parser["DEFAULT"]
    here = os.path.dirname(os.path.abspath(config_file))

    def _rel(p):
        return p if os.path.isabs(p) else os.path.join(here, p)

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
        _n            = cfg.getint("n",    fallback=0)
        _nobj         = cfg.getint("nobj", fallback=0)
        args.n        = _n    if _n    > 0 else None
        args.nobj     = _nobj if _nobj > 0 else None
        args.log_level = cfg.get("log_level", fallback="INFO")

        # --- synthetic mosaic (tests/mosaic_brain/steps15.py) ---------------
        # All optional: the experimental single-tile steps15 scripts do not set
        # them and keep working off the fallbacks above.
        args.tiles      = get_list(cfg, "tiles", str)
        args.nzobj      = cfg.getint("nzobj",     fallback=0)
        args.bin        = cfg.getint("bin",       fallback=0)
        args.ntheta_rec = cfg.getint("ntheta_rec", fallback=0)
        args.nobj_tile  = cfg.getint("nobj_tile", fallback=0)
        args.mask       = cfg.getfloat("mask",      fallback=0.9)
        args.ntile_h    = cfg.getint("ntile_h", fallback=1)
        args.ntile_v    = cfg.getint("ntile_v", fallback=1)
        args.tile_step_h = cfg.getfloat("tile_step_h", fallback=0.0)
        args.tile_step_v = cfg.getfloat("tile_step_v", fallback=0.0)
        _sd             = cfg.get("shift_dir", fallback="")
        args.shift_dir  = _rel(_sd.strip()) if _sd and _sd.strip() else None
        args.tile_file  = (os.path.join(args.shift_dir, "tile_offsets.txt")
                           if args.shift_dir else None)
    except configparser.NoOptionError as e:
        raise ValueError(f"Missing required field in {config_file}: {e}") from e

    return args


def parse_args_gen(config_file):
    """Parse config for the synthetic mosaic generator (gen_data.py / make_geometry.py).

    Paths given relative to the config file are resolved against its directory,
    so the scripts can be launched from anywhere.
    """
    parser = configparser.ConfigParser(inline_comment_prefixes=("#",), interpolation=None)
    with open(config_file, "r", encoding="utf-8") as f:
        parser.read_string("[DEFAULT]\n" + f.read())
    cfg = parser["DEFAULT"]
    here = os.path.dirname(os.path.abspath(config_file))

    def _rel(p):
        return p if os.path.isabs(p) else os.path.join(here, p)

    try:
        args = SimpleNamespace()
        args.path_out = cfg.get("path_out").rstrip('/')
        args.pfile    = cfg.get("pfile")
        args.out_file = os.path.join(args.path_out, f"{args.pfile}.h5")

        args.energy                  = cfg.getfloat("energy")
        args.focustodetectordistance = cfg.getfloat("focustodetectordistance")
        args.z1                      = get_list(cfg, "z1", float)
        args.detector_pixelsize      = cfg.getfloat("detector_pixelsize")
        args.ndet                    = cfg.getint("ndet")

        args.ntheta      = cfg.getint("ntheta")
        args.theta_range = cfg.getfloat("theta_range", fallback=180.0)
        args.bin         = cfg.getint("bin")

        args.ntile_h     = cfg.getint("ntile_h")
        args.ntile_v     = cfg.getint("ntile_v")
        args.tile_step_h = cfg.getfloat("tile_step_h")
        args.tile_step_v = cfg.getfloat("tile_step_v")
        args.shift_dir   = _rel(cfg.get("shift_dir"))
        args.tile_file   = os.path.join(args.shift_dir, "tile_offsets.txt")
        args.nobj        = cfg.getint("nobj")
        args.nzobj       = cfg.getint("nzobj")

        args.shift_rand_px = cfg.getfloat("shift_rand_px", fallback=0.0)

        args.prb_abs   = _rel(cfg.get("prb_abs"))
        args.prb_phase = _rel(cfg.get("prb_phase"))

        _vol                 = cfg.get("obj_vol", fallback="")
        args.obj_vol         = _rel(_vol.strip()) if _vol and _vol.strip() else None
        args.delta_beta      = cfg.getfloat("delta_beta",      fallback=100.0)
        args.obj_span_px     = cfg.getfloat("obj_span_px",     fallback=0.0)
        # Multiplies the loaded volume: the sample file has arbitrary grey
        # levels, so this is what sets the projected phase excursion.
        args.obj_scale       = cfg.getfloat("obj_scale",       fallback=1.0)

        args.nchunk         = cfg.getint("nchunk", fallback=4)
        args.log_level      = cfg.get("log_level", fallback="INFO")
    except configparser.NoOptionError as e:
        raise ValueError(f"Missing required field in {config_file}: {e}") from e

    return args
