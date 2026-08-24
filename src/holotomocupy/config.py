"""Config-file parsing for the pipeline scripts.

Every step reads one flat key = value file. The five parsers below differ only
in *which* keys they read; the mechanics (implicit [DEFAULT] section, relative
path resolution, missing-field reporting) live in _Cfg.
"""

import os
import configparser
from types import SimpleNamespace

_MISSING = object()


class _Cfg:
    """Thin wrapper over a configparser section with clear error reporting.

    SectionProxy.get()/getint()/... return None for an absent key instead of
    raising NoOptionError, so a missing required field used to surface far from
    its cause (an AttributeError on None inside os.path.join or .rstrip). Here,
    an accessor called without an explicit ``fallback`` is required and raises
    ValueError naming both the config file and the key.
    """

    def __init__(self, cfg, source, here):
        self._c = cfg
        self._source = source
        self._here = here

    def _conv(self, fn, key, fallback):
        if key not in self._c:
            if fallback is _MISSING:
                raise ValueError(f"Missing required field in {self._source}: {key}")
            return fallback
        try:
            return fn(key)
        except ValueError as e:
            raise ValueError(f"Bad value for '{key}' in {self._source}: {e}") from e

    def str(self, key, fallback=_MISSING):
        return self._conv(self._c.get, key, fallback)

    def int(self, key, fallback=_MISSING):
        return self._conv(self._c.getint, key, fallback)

    def float(self, key, fallback=_MISSING):
        return self._conv(self._c.getfloat, key, fallback)

    def bool(self, key, fallback=_MISSING):
        return self._conv(self._c.getboolean, key, fallback)

    def path(self, key, fallback=_MISSING):
        """A string field with any trailing '/' stripped."""
        v = self.str(key, fallback)
        return v.rstrip('/') if isinstance(v, str) else v

    def opt_str(self, key):
        """Optional string: absent, empty, or whitespace-only all give None."""
        v = self._c.get(key, fallback=None)
        return v.strip() if v and v.strip() else None

    def list(self, key, cast=str, fallback=_MISSING, sep=","):
        s = self.str(key, "" if fallback is _MISSING else fallback)
        return [cast(x.strip()) for x in s.split(sep) if x.strip()]

    def rel(self, p):
        """Resolve a path given relative to the config file's directory."""
        return p if os.path.isabs(p) else os.path.join(self._here, p)


def _load(config_file, interpolation=configparser.BasicInterpolation()):
    """Read a flat key = value file as if every key were in [DEFAULT]."""
    parser = configparser.ConfigParser(inline_comment_prefixes=("#",),
                                       interpolation=interpolation)
    with open(config_file, "r", encoding="utf-8") as f:
        parser.read_string("[DEFAULT]\n" + f.read())
    here = os.path.dirname(os.path.abspath(config_file))
    return _Cfg(parser["DEFAULT"], config_file, here)


def get_list(c, key, cast=str, sep=","):
    """Back-compatible helper for callers holding a raw configparser section."""
    s = c.get(key, fallback="")
    return [cast(x.strip()) for x in s.split(sep) if x.strip()]


def parse_args(config_file):
    """Config for the main reconstruction (step6.py)."""
    cfg = _load(config_file)
    args = SimpleNamespace()

    args.pfile    = cfg.opt_str("pfile")
    args.path_out = cfg.path("path_out")
    _path         = cfg.path("path", fallback=args.path_out)
    if args.pfile:
        args.in_file = f"{args.path_out}/{args.pfile}.h5"
    else:
        args.in_file = os.path.join(_path, cfg.str("in_file"))

    args.ntheta      = cfg.int("ntheta")
    args.start_theta = cfg.int("start_theta")
    args.nz          = cfg.int("nz")
    args.n           = cfg.int("n")
    args.nzobj       = cfg.int("nzobj")
    args.nobj        = cfg.int("nobj")
    args.ndist       = cfg.int("ndist")
    args.paganin     = cfg.int("paganin")
    args.mask        = cfg.float("mask")

    args.lam_prbfit    = cfg.float("lam_prbfit")
    args.lam_laplacian = cfg.float("lam_laplacian", fallback=0.0)

    args.rho = cfg.list("rho", float)
    # Optional rho tuning: False (default) -> args.rho used as-is; True ->
    # BH first coordinate-searches rho[prb, pos] around args.rho with short
    # trials of rho_estimate_niter iterations. See Rec.estimate_rho_coord.
    args.estimate_rho       = cfg.bool("estimate_rho",       fallback=False)
    args.rho_estimate_niter = cfg.int ("rho_estimate_niter", fallback=16)
    # -1 (default) keeps estimate_rho_coord's trials silent; N > 0 logs the
    # error every N iterations inside each trial.
    args.rho_trial_error_step = cfg.int("rho_trial_error_step", fallback=-1)

    # Out-of-grid detector pixels: with eff_demag > 1 a detector pixel maps
    # back outside the object grid whenever nobj < n*max(eff_demag). Such
    # pixels carry only the shift kernel's boundary condition, so by default
    # Rec._build_data_mask drops them from the data fit. mask_oob_margin is
    # slack in detector pixels on top of the worst-case position shift.
    args.mask_oob        = cfg.bool ("mask_oob",        fallback=True)
    args.mask_oob_margin = cfg.float("mask_oob_margin", fallback=2.0)

    # Derive the CG beta/alpha from one Hessian sweep instead of three
    # (see the fused_hessian note on Rec). check_fused_hessian re-measures
    # every form the classic path would have measured and logs the relative
    # disagreement — 4 extra sweeps per iteration, for verification only.
    args.fused_hessian       = cfg.bool("fused_hessian",       fallback=True)
    args.check_fused_hessian = cfg.bool("check_fused_hessian", fallback=False)
    # Plot the real functional against the quadratic model that picked
    # alpha, on every checkpoint_step iteration (Rec.check_approximation).
    # npp extra evaluations of the full functional per triggered iteration,
    # so it is off by default. Not named check_approximation: args are
    # copied onto Rec verbatim and would shadow the method.
    args.check_approx = cfg.bool("check_approx", fallback=False)

    args.niter  = cfg.int("niter")
    args.nchunk = cfg.int("nchunk")
    # How many distances share one upload of a theta chunk of proj inside
    # the cascade kernels. 0 = all of them (the default), 1 = the old
    # outer-distance loop. See Rec._resolve_ndistchunk.
    args.ndistchunk      = cfg.int("ndistchunk", fallback=0)
    args.checkpoint_step = cfg.int("checkpoint_step")
    args.error_step      = cfg.int("error_step")
    # Internal instrumentation (cache hit/miss counters and the like).
    # -1 = never, the default: these numbers only matter when tuning the
    # solver, not when running it.
    args.debug_step = cfg.int("debug_step", fallback=-1)
    args.start_iter = cfg.int("start_iter")

    args.rotation_center_shift = cfg.float("rotation_center_shift")
    args.bin           = cfg.int("bin")
    args.log_level     = cfg.str("log_level", fallback="WARNING")
    args.energy        = cfg.float("energy", fallback=None)
    args.method        = cfg.int("method",        fallback=0)
    args.start_method  = cfg.int("start_method",  fallback=1)

    _pos_chk            = cfg.opt_str("pos_checkpoint")
    args.pos_checkpoint = os.path.join(_path, _pos_chk) if _pos_chk else None
    _prb                = cfg.opt_str("prb_file")
    args.prb_file       = os.path.join(args.path_out, _prb) if _prb else None
    args.init_vol       = cfg.opt_str("init_vol")
    args.init_vol_scale = cfg.float("init_vol_scale", fallback=1.0)

    # Mosaic: one .h5 per tile, {path_out}/{pfile}_{tile}.h5.  Empty tiles=
    # is single-tile mode and leaves everything below None.  mosaic_file is
    # a NAME, not a file that has to exist: MosaicReader only uses it to
    # find the shared initial object {pfile}_obj.h5, exactly as step6 of
    # the YY037A pipeline uses {pfile}_mosaic.h5.
    args.tiles = cfg.list("tiles", str)
    if args.tiles:
        if not args.pfile:
            raise ValueError("tiles= requires pfile=")
        args.tile_files  = [f"{args.path_out}/{args.pfile}_{t}.h5" for t in args.tiles]
        args.mosaic_file = args.in_file
    else:
        args.tile_files  = None
        args.mosaic_file = None

    return args


def parse_args_step0(config_file):
    """Config for step0.py reading ESRF scan + metadata files."""
    cfg = _load(config_file, interpolation=None)
    args = SimpleNamespace()

    path             = cfg.path("path")
    args.path_out    = cfg.path("path_out")
    args.scan_file   = os.path.join(path, cfg.str("scan_file"))
    args.meta_file   = os.path.join(path, cfg.str("meta_file"))
    args.h5_out      = os.path.join(args.path_out, cfg.str("h5_out"))
    args.dataset_ids = cfg.list("dataset_ids", int)
    args.n               = cfg.int("n",               fallback=2048)
    args.niter           = cfg.int("niter",           fallback=129)
    args.nchunk          = cfg.int("nchunk",          fallback=4)
    args.checkpoint_step = cfg.int("checkpoint_step", fallback=32)
    args.error_step      = cfg.int("error_step",      fallback=32)
    args.rho             = cfg.list("rho", float)
    args.log_level       = cfg.str("log_level", fallback="INFO")

    return args


def parse_args_step0_nx(config_file):
    """Config for step0.py reading ESRF NXtomo (.nx) files."""
    cfg = _load(config_file, interpolation=None)
    args = SimpleNamespace()

    path          = cfg.path("path")
    args.path_out = cfg.path("path_out")
    args.nx_file  = os.path.join(path, cfg.str("nx_file"))
    args.h5_out   = os.path.join(args.path_out, cfg.str("h5_out"))
    args.n               = cfg.int("n",               fallback=2048)
    args.niter           = cfg.int("niter",           fallback=129)
    args.nchunk          = cfg.int("nchunk",          fallback=4)
    args.checkpoint_step = cfg.int("checkpoint_step", fallback=-1)
    args.error_step      = cfg.int("error_step",      fallback=32)
    args.rho             = cfg.list("rho", float)
    args.log_level       = cfg.str("log_level", fallback="INFO")

    return args


def parse_args_steps15(config_file):
    """Config for steps15.py (EDF->HDF5, preprocessing, shift combination)."""
    cfg = _load(config_file)
    args = SimpleNamespace()

    args.path     = cfg.path("path")
    args.pfile    = cfg.str("pfile")
    args.path_out = cfg.opt_str("path_out")

    args.start_step            = cfg.int  ("start_step",            fallback=1)
    args.start_level_rec       = cfg.int  ("start_level_rec",       fallback=0)
    args.rotation_center_shift = cfg.float("rotation_center_shift", fallback=0.0)
    args.nlevels   = cfg.int  ("nlevels",   fallback=4)
    args.paganin   = cfg.float("paganin",   fallback=120.0)
    args.nchunk    = cfg.int  ("nchunk",    fallback=16)
    args.ref_dist  = cfg.int  ("ref_dist",  fallback=0)
    _n             = cfg.int  ("n",    fallback=0)
    _nobj          = cfg.int  ("nobj", fallback=0)
    args.n         = _n    if _n    > 0 else None
    args.nobj      = _nobj if _nobj > 0 else None
    args.log_level = cfg.str  ("log_level", fallback="INFO")

    # --- synthetic mosaic (tests/mosaic_brain/steps15.py) ------------------
    # All optional: the experimental single-tile steps15 scripts do not set
    # them and keep working off the fallbacks above.
    args.tiles       = cfg.list ("tiles", str)
    args.nzobj       = cfg.int  ("nzobj",       fallback=0)
    args.bin         = cfg.int  ("bin",         fallback=0)
    args.ntheta_rec  = cfg.int  ("ntheta_rec",  fallback=0)
    args.nobj_tile   = cfg.int  ("nobj_tile",   fallback=0)
    args.mask        = cfg.float("mask",        fallback=0.9)
    args.ntile_h     = cfg.int  ("ntile_h",     fallback=1)
    args.ntile_v     = cfg.int  ("ntile_v",     fallback=1)
    args.tile_step_h = cfg.float("tile_step_h", fallback=0.0)
    args.tile_step_v = cfg.float("tile_step_v", fallback=0.0)
    _sd              = cfg.opt_str("shift_dir")
    args.shift_dir   = cfg.rel(_sd) if _sd else None
    args.tile_file   = (os.path.join(args.shift_dir, "tile_offsets.txt")
                        if args.shift_dir else None)

    return args


def parse_args_gen(config_file):
    """Config for the synthetic mosaic generator (gen_data.py / make_geometry.py).

    Paths given relative to the config file are resolved against its directory,
    so the scripts can be launched from anywhere.
    """
    cfg = _load(config_file, interpolation=None)
    args = SimpleNamespace()

    args.path_out = cfg.path("path_out")
    args.pfile    = cfg.str("pfile")
    args.out_file = os.path.join(args.path_out, f"{args.pfile}.h5")

    args.energy                  = cfg.float("energy")
    args.focustodetectordistance = cfg.float("focustodetectordistance")
    args.z1                      = cfg.list ("z1", float)
    args.detector_pixelsize      = cfg.float("detector_pixelsize")
    args.ndet                    = cfg.int  ("ndet")

    args.ntheta      = cfg.int  ("ntheta")
    args.theta_range = cfg.float("theta_range", fallback=180.0)
    args.bin         = cfg.int  ("bin")

    args.ntile_h     = cfg.int  ("ntile_h")
    args.ntile_v     = cfg.int  ("ntile_v")
    args.tile_step_h = cfg.float("tile_step_h")
    args.tile_step_v = cfg.float("tile_step_v")
    args.shift_dir   = cfg.rel(cfg.str("shift_dir"))
    args.tile_file   = os.path.join(args.shift_dir, "tile_offsets.txt")
    args.nobj        = cfg.int  ("nobj")
    args.nzobj       = cfg.int  ("nzobj")

    args.shift_rand_px = cfg.float("shift_rand_px", fallback=0.0)

    args.prb_abs   = cfg.rel(cfg.str("prb_abs"))
    args.prb_phase = cfg.rel(cfg.str("prb_phase"))

    _vol             = cfg.opt_str("obj_vol")
    args.obj_vol     = cfg.rel(_vol) if _vol else None
    args.delta_beta  = cfg.float("delta_beta",  fallback=100.0)
    args.obj_span_px = cfg.float("obj_span_px", fallback=0.0)
    # Multiplies the loaded volume: the sample file has arbitrary grey
    # levels, so this is what sets the projected phase excursion.
    args.obj_scale   = cfg.float("obj_scale",   fallback=1.0)

    args.nchunk    = cfg.int("nchunk", fallback=4)
    args.log_level = cfg.str("log_level", fallback="INFO")

    return args
