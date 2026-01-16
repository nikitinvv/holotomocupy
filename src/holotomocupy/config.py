import configparser
from types import SimpleNamespace

def get_list(c, key, cast=str, sep=","):
    s = c.get(key, fallback="")
    # drop inline comments and full-line comments
    s = s.split("#", 1)[0]
    return [cast(x.strip()) for x in s.split(sep) if x.strip()]

def parse_args(config_file):
    
    cfg = configparser.ConfigParser(inline_comment_prefixes=("#"))    
    with open(config_file, "r", encoding="utf-8") as f:
        # Pretend everything belongs to a DEFAULT section
        cfg.read_string("[DEFAULT]\n" + f.read())
        cfg = cfg["DEFAULT"]
    
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
    args.lam_reg = cfg.getfloat("lam_reg") 
    args.lam_prbfit = cfg.getfloat("lam_prbfit")     
    args.rho = get_list(cfg, "rho", float)
    args.nbins = cfg.getint("nbins") 
    args.niter = get_list(cfg, "niter", int)
    args.nchunk = get_list(cfg, "nchunk", int)
    args.vis_step = get_list(cfg, "vis_step", int)
    args.err_step = get_list(cfg, "err_step", int)
    args.clean_cache_step = get_list(cfg, "clean_cache_step", int)
    args.start_bin = cfg.getint("start_bin") 
    args.start_iter = cfg.getint("start_iter") 
    args.rotation_center_shift = cfg.getfloat("rotation_center_shift") 
    args.ngpus = cfg.getint("ngpus") 
    args.bin = cfg.getint("bin") 
    return args