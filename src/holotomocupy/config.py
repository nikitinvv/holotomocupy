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
        args.in_file = cfg.get("in_file")
        args.path_out = cfg.get("path_out")
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
        args.lam_prbfit = cfg.getfloat("lam_prbfit")
        args.rho = get_list(cfg, "rho", float)
        args.niter = cfg.getint("niter")
        args.nchunk = cfg.getint("nchunk")
        args.vis_step = cfg.getint("vis_step")
        args.err_step = cfg.getint("err_step")
        args.start_iter = cfg.getint("start_iter")
        args.rotation_center_shift = cfg.getfloat("rotation_center_shift")
        args.bin = cfg.getint("bin")
    except configparser.NoOptionError as e:
        raise ValueError(f"Missing required field in {config_file}: {e}") from e
    
    return args
