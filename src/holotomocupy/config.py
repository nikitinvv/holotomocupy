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
        _n            = cfg.getint("n",    fallback=0)
        _nobj         = cfg.getint("nobj", fallback=0)
        args.n        = _n    if _n    > 0 else None
        args.nobj     = _nobj if _nobj > 0 else None
        args.log_level = cfg.get("log_level", fallback="INFO")
    except configparser.NoOptionError as e:
        raise ValueError(f"Missing required field in {config_file}: {e}") from e

    return args
