import numpy as np
import cupy as cp
import os
import warnings
import pandas as pd
import nvtx
import cupy.fft

from .tomo import Tomo
from .propagation import Propagation
from .shift import Shift
from .chunking import Chunking
from .extra_terms import LaplacianTerm, PrbfitTerm
from .utils import *
from .mpi_functions import *
from .logger_config import logger
from .conv2d_cufftdx import precompile as cufftdx_precompile

np.set_printoptions(legacy="1.25")
warnings.filterwarnings("ignore", message=f".*peer.*")
cupy.fft.config.get_plan_cache().set_size(0) # dont waste GPU memory


class Rec:
    # --- fused CG step ---------------------------------------------------
    # The bilinear form B(y,z) = <y, H(vars)·z> is symmetric: _d2F_dF0_fused
    # contracts as (1-d0)*reprod(y,z) + d0*reprod(l0,y)*reprod(l0,z), which is
    # manifestly invariant under y<->z. So the three cascade sweeps the classic
    # path runs per iteration
    #
    #     B(g,e)                  -> beta numerator      (compute_beta)
    #     B(e,e)                  -> beta denominator    (compute_beta)
    #     B(b*e - g, b*e - g)     -> alpha denominator   (compute_alpha)
    #
    # all follow from the three forms {B(g,g), B(g,e), B(e,e)} of a SINGLE
    # sweep:
    #
    #     beta   = B(g,e) / B(e,e)
    #     bottom = b^2*B(e,e) - 2*b*B(g,e) + B(g,g)
    #
    # One sweep instead of three, i.e. one third of the host->device streaming
    # (the cascades are PCIe-bound), for two extra chunk-sized `w`
    # accumulators. Set False to fall back to the classic path; subclasses
    # that override hessian_cascade or compute_alpha must opt out.
    #
    # All three terms of the functional are expanded the same way, each
    # producing its three forms from one pass over its own data:
    # hessian_cascade3, PrbfitTerm.hessian3, LaplacianTerm.hessian3. The
    # regularization one needs grads['obj'] to carry ghost rows like
    # vars['obj'] and etas['obj'] already do, which costs 4 z-slabs of pinned
    # memory and only when lam_laplacian > 0.
    #
    # Per iteration: 1 cascade sweep (was 3), 1 regularization pass (was 3),
    # and 2 allreduces (was 3).
    fused_hessian = True

    # --- hoisted distance loop -------------------------------------------
    # The cascade kernels used to be wrapped in `for k in range(ndist)`, each
    # k a full @gpu_batch pass over theta. vars/grads/etas['proj'] do not
    # depend on k, so that layout re-streamed the whole (pinned, host) proj
    # slab over PCIe once per distance -- the dominant cost of a mosaic scan,
    # where ndist = ntiles * ndist_per_tile is tens.
    #
    # Now the distance loop lives INSIDE the theta-chunk kernel: a chunk of
    # proj is uploaded once and reused for `ndistchunk` distances. What has to
    # be resident per distance is only detector-plane -- data [nchunk,nz,n]
    # f32 and prb [nz,n] c64 -- and in a mosaic the detector plane is tens of
    # times smaller than the object plane (nz*n vs nzobj*nobj), so all ndist
    # of them together cost a fraction of the three proj chunks that were
    # already resident. See _resolve_ndistchunk.
    #
    # Set ndistchunk = 1 (config key or --ndistchunk) to reproduce the old
    # behaviour exactly.
    # Subclasses whose cascade kernels are genuinely per-distance (RecDelta)
    # set hoist_dist_loop = False, which pins ndistchunk to 1 so the chunking
    # pool is sized as before.
    hoist_dist_loop = True

    def __init__(self, args):

        # copy args to elements of the class
        for key, value in vars(args).items():
            setattr(self, key, value)

        # list of functionals, gradients, differentials, and second-order differentials
        self.F = [self.F0, self.F1, self.F2, self.F3]
        self.gF = [self.gF0, self.gF1, self.gF2, self.gF3]
        self.dF = [self.dF0, self.dF1, self.dF2, self.dF3]
        self.d2F_dF = [self.d2F_dF0, self.d2F_dF1, self.d2F_dF2,self.d2F_dF3]

        self.ndistchunk = self._resolve_ndistchunk()
        nbytes = self._chunking_pool_bytes()

        ### multinode processing
        self.cl_mpi = MPIClass(args.comm, self.nzobj, self.ntheta, self.nobj, args.obj_dtype)
        self.local_nzobj = self.cl_mpi.local_nzobj
        self.local_ntheta = self.cl_mpi.local_ntheta
        self.rank      = self.cl_mpi.rank
        self.st_obj    = self.cl_mpi.st_obj
        self.end_obj   = self.cl_mpi.end_obj
        self.st_theta  = self.cl_mpi.st_theta
        self.end_theta = self.cl_mpi.end_theta
        if self.rank == 0:
            # Report the ACTUAL cost of the hoist, not the sizing budget: the
            # chunking pool often does not grow at all (it is capped by the
            # obj-shape linear_batch candidate, not by the cascade one), so the
            # only real increase is the per-group prb bundles.
            nd = self.ndistchunk
            pool_1  = self._pool_bytes_for(1)
            pool_nd = self._pool_bytes_for(nd)
            prb_1   = 3 * self.nz * self.n * 8
            prb_nd  = 3 * nd * self.nz * self.n * 8
            logger.info(
                f"ndistchunk={nd}/{self.ndist}: "
                f"{len(self._dist_groups())} proj upload(s) per cascade sweep "
                f"(was {self.ndist}); GPU "
                f"{(pool_nd - pool_1 + prb_nd - prb_1)/2**20:+.0f} MiB "
                f"(pool {pool_1/2**30:.2f}->{pool_nd/2**30:.2f} GiB, "
                f"prb staging {prb_1/2**20:.0f}->{prb_nd/2**20:.0f} MiB); "
                f"host RAM unchanged")

        # X-ray propagation and magnification parameters for classes
        wavelength = 1.24e-09 / self.energy
        z2 = self.focustodetectordistance - self.z1
        magnifications = self.focustodetectordistance / self.z1
        norm_magnifications = magnifications / magnifications[0]
        distance = (self.z1 * z2) / self.focustodetectordistance * norm_magnifications**2
        voxelsize = self.detector_pixelsize / magnifications[0]

        # scaling variables
        self.rho_sq = {'obj': args.rho[0]**2, 'prb': args.rho[1]**2, 'pos': args.rho[2]**2}

        # cuFFTDx JIT compile: rank 0 builds the .so, then all ranks proceed
        if self.rank == 0:
            cufftdx_precompile(2 * self.nz, 2 * self.n)
        self.cl_mpi.comm.Barrier()

        # sizes for normalization
        self.data_size = self.ntheta * self.ndist * self.nz * self.n
        self.prb_size  = self.ndist * self.nz * self.n
        self.obj_size  = self.nzobj * self.nobj**2
        # normalization constant to address work with normal operators
        self.norm_const = np.float32(np.sqrt(self.nobj / self.ntheta))
        self.norm_magnifications = norm_magnifications

        # create classes (one GPU per MPI rank via CUDA_VISIBLE_DEVICES)
        self.cl_chunking = Chunking(nbytes, self.nchunk)
        self.cl_tomo  = Tomo(self.nobj, self.nchunk, self.theta, self.mask)
        self.cl_prop  = Propagation(self.n, self.nz, self.nchunk, self.ndist, wavelength, voxelsize, distance)
        self.cl_shift = Shift(self.n, self.nobj, self.nz, self.nzobj, self.obj_dtype, self.nchunk)
        if self.lam_laplacian > 0:
            self.cl_lap_term = LaplacianTerm(self.lam_laplacian, self.obj_size,
                                             self.local_nzobj, self.nobj, self.obj_dtype,
                                             self.cl_mpi, self.cl_chunking.gpu_batch,
                                             grad_pad=getattr(self, 'alloc_mode', 'full') != 'gen')
        if self.lam_prbfit > 0:
            self.cl_prb_term = PrbfitTerm(self.lam_prbfit, self.prb_size,
                                          self.ndist, self.nz, self.n, self.cl_prop)

        self.alloc_arrays()
       
        # fast refs
        self.gpu_batch = self.cl_chunking.gpu_batch
        self.redot_batch = self.cl_chunking.redot_batch
        self.linear_batch = self.cl_chunking.linear_batch
        self.linear_redot_batch = self.cl_chunking.linear_redot_batch
        self.mulc_batch = self.cl_chunking.mulc_batch
        self.redist = self.cl_mpi.redist
        self.allreduce  = self.cl_mpi.allreduce
        self.allreduce2 = self.cl_mpi.allreduce2
        self.allreduce_scalars = self.cl_mpi.allreduce_scalars
        
        # save convergence results
        self.table = pd.DataFrame(columns=["iter", "err", "time"])

        # apply_F_from memoization (per-chunk scope; reset in every cascade kernel)
        self._apply_F_cache  = {}
        self._apply_F_hits   = 0
        self._apply_F_misses = 0

    def _dist_bytes(self):
        """Per-distance chunking-pool cost inside a hoisted cascade kernel:
        one data chunk plus the three pos chunks and the eff_demag chunk."""
        return (self.nchunk * self.nz * self.n * 4      # data   [nchunk,nz,n] f32
                + 3 * self.nchunk * 2 * 4               # x/y/z pos [nchunk,2] f32
                + self.nchunk * 4)                      # eff_demag [nchunk]   f32

    def _resolve_ndistchunk(self):
        """How many distances share one upload of a theta chunk of proj.

        1     = the old outer-distance loop: a cascade sweep costs ndist*3 proj
                slabs of PCIe traffic.
        ndist = one upload per sweep, and one B-spline coeff prefilter per chunk
                instead of ndist of them. This is the default.

        Taking every distance is safe because the growth is entirely
        detector-plane: the extra cost is ndist data chunks plus the prb
        bundles, against the three object-plane proj chunks that dominate and
        do not scale with it. For the mosaic geometries this is built for
        (nzobj*nobj >> nz*n) the cascade stays well under the obj-shape
        linear_batch call site that actually sizes the chunking pool.

        `ndistchunk` (config key or --ndistchunk) overrides it; 0 or absent
        means all of them. Lower it only if a tight GPU needs the detector-plane
        residency back.
        """
        if not self.hoist_dist_loop:
            return 1
        want = int(getattr(self, 'ndistchunk', 0) or 0)
        if want > 0:
            return max(1, min(want, self.ndist))
        return self.ndist

    def _dist_groups(self):
        """range(ndist) partitioned into consecutive groups of <= ndistchunk.
        Each group is one @gpu_batch pass over theta."""
        nd = self.ndistchunk
        return [(k0, min(k0 + nd, self.ndist)) for k0 in range(0, self.ndist, nd)]

    def _assert_dist_group(self, nd):
        """Chunking classifies inputs as proper/non-proper by shape[axis]; the
        per-group prb bundle is [nd, nz, n] and would be silently mistaken for
        a proper (theta-chunked) input if nd happened to equal local_ntheta."""
        assert nd != self.local_ntheta, (
            f"ndistchunk group size {nd} aliases local_ntheta={self.local_ntheta}; "
            f"pick a different ndistchunk")

    def _chunking_pool_bytes(self):
        """Chunking-pool bytes at the configured ndistchunk."""
        return self._pool_bytes_for(self.ndistchunk)

    def _pool_bytes_for(self, ndistchunk):
        """Worst-case chunking-pool footprint across all gpu_batch callers.

        Candidates (proper-input/output buffers per single call, double-buffered):
          cascade kernels (distances hoisted inside the chunk, ndistchunk of
          them per call):
            _hess_dists:   3 proj + ndistchunk * (1 data + 3 pos + eff_demag)
            _grad_dists:   3 proj + ndistchunk * (1 data + 3 pos + eff_demag)
          linear_batch on vars['obj']:     3 obj-shape  (out + x + y)
          gradient_laplacian (inp_pad=4):  3 obj-shape + 4 obj-slabs from padding
          laplacian hessian3 (inp_pad=4):  2 obj-shape + 8 obj-slabs (two haloed
                                           inputs, g_pad and e_pad)
                                           (both only sized in when lam_laplacian > 0)
        linear_batch on vars['proj'] is 3 proj-shape — dominated by cascade.
        gen_sqrt_data (1 proj + ndistchunk data, as outputs) and min
        (1 proj + ndistchunk * small) are both under the cascade candidate.
        Any new @gpu_batch caller with a bigger footprint must be added.
        """
        obj_item   = np.dtype(self.obj_dtype).itemsize
        obj_slab   = self.nobj  * self.nobj  * obj_item     # one z-slab of obj
        proj_bytes = self.nchunk * self.nzobj * self.nobj  * obj_item
        obj_bytes  = self.nchunk * obj_slab
        dist_bytes = ndistchunk * self._dist_bytes()
        candidates = [3 * proj_bytes + dist_bytes, 3 * obj_bytes]  # cascade, lin_obj
        if self.lam_laplacian > 0:
            candidates.append(3 * obj_bytes + 4 * obj_slab)        # gradient_laplacian
            candidates.append(2 * obj_bytes + 8 * obj_slab)        # laplacian hessian3
        return int(2.1 * max(candidates))   # ×2 double-buffering + 10% slack

    def alloc_arrays(self):
        """Allocate all pinned CPU and CuPy GPU buffers used during reconstruction.

        args.alloc_mode = 'gen' (gen_data.py) skips the gradient and
        conjugate-direction buffers entirely: generation only ever touches
        vars / data / proj_tmp.  Allocating grads/etas and clearing the dicts
        afterwards still peaks at 3 obj-slabs + 3 proj-slabs of pinned memory,
        two thirds of which is never used -- at bin 1 that peak is 4.2 TiB
        against 1.8 TiB for what generation actually needs.  'full' (default)
        allocates everything, so reconstruction is unaffected.
        """
        gen = getattr(self, 'alloc_mode', 'full') == 'gen'
        prb_shape = [self.ndist, self.nz, self.n]
        obj_shape = [self.local_nzobj, self.nobj, self.nobj]
        # vars/etas/grads['obj'] alias the padded scratch owned by cl_lap_term when
        # the Laplacian term is active; otherwise plain obj-shape buffers. All
        # three carry ghost rows so a single chunked pass can differentiate any
        # of them (LaplacianTerm.hessian3).
        if hasattr(self, 'cl_lap_term'):
            obj_buf   = self.cl_lap_term.obj_view
            etas_obj  = self.cl_lap_term.etas_view
            grads_obj = self.cl_lap_term.grads_view
        else:
            obj_buf  = make_pinned(obj_shape, dtype=self.obj_dtype); obj_buf[:]  = 0
            # etas/grads['obj'] are gradient-side buffers: not allocated in gen mode.
            etas_obj  = None if gen else make_pinned(obj_shape, dtype=self.obj_dtype)
            grads_obj = None if gen else make_pinned(obj_shape, dtype=self.obj_dtype)
            for b in (etas_obj, grads_obj):
                if b is not None:
                    b[:] = 0

        # reconstruction variables. prb / pos are pinned CPU (uploaded once per
        # gpu_batch call by the chunking auto-cp.asarray on non-proper inputs).
        # grads['prb'] stays on GPU because it's a non-proper output of
        # gradients_cascade and the chunking machinery only accepts cp.ndarray
        # for non-proper outputs.
        self.vars = {
            'obj':  obj_buf,
            'pos':  make_pinned([self.ndist, self.local_ntheta, 2],         dtype='float32'),
            'prb':  make_pinned(prb_shape,                                 dtype='complex64'),
            'proj': make_pinned([self.local_ntheta, self.nzobj, self.nobj], dtype=self.obj_dtype),
        }
        # measurement data; ref is owned by cl_prb_term — aliased here for back-compat
        # so external code (readers, gen_sqrt_ref out-arg) can keep using cl.ref.
        self.data = make_pinned([self.ndist, self.local_ntheta, self.nz, self.n], dtype='float32')
        if hasattr(self, 'cl_prb_term'):
            self.ref = self.cl_prb_term.ref
        else:
            self.ref = cp.empty([self.ndist, self.nz, self.n], dtype='float32')
        # gradient and conjugate-direction buffers (reconstruction only)
        self.grads, self.etas = {}, {}
        if not gen:
            for ge in self.grads, self.etas:
                ge["pos"]  = make_pinned([self.ndist, self.local_ntheta, 2], dtype='float32')
                ge["proj"] = make_pinned([self.local_ntheta, self.nzobj, self.nobj], dtype=self.obj_dtype)
            # vars/grads/etas['prb'] all pinned. gradients_cascade uses a small per-k GPU
            # staging buffer to accumulate y[0]*rho_sq across theta chunks for one dist,
            # then D2H's the slot to grads['prb'][k] after each k's @gpu_batch.
            self.grads["prb"] = make_pinned(prb_shape, dtype='complex64')
            self.etas["prb"]  = make_pinned(prb_shape, dtype='complex64')
            self.etas["obj"]  = etas_obj
            self.grads["obj"] = grads_obj
            # etas must start at exactly zero, not at whatever the pinned pool
            # handed back: at start_iter the CG step is eta <- 0*eta - grad, and
            # 0*NaN is NaN. It also makes the first Hessian sweep return
            # B(g,e) = B(e,e) = 0 exactly, which is what lets that iteration take
            # the same code path as every other one.
            for k, v in self.etas.items():
                if k != "obj":      # obj is zeroed at allocation above
                    v[:] = 0
        self.proj_tmp    = make_pinned([self.ntheta, self.local_nzobj, self.nobj], dtype=self.obj_dtype)

        self.shrink_nd = cp.zeros((self.ndist, self.local_ntheta), dtype='float32')
        self.eff_demag = cp.zeros((self.ndist, self.local_ntheta), dtype='float32')
    
    def BH(self, writer=None):
        vars  = self.vars
        grads = self.grads
        etas  = self.etas

        self.precalc(vars)
        self.error_debug(vars, -1)

        self.time_start = time.time()
        for i in range(self.start_iter, self.niter):
            with nvtx.annotate(f"::BH:{i}"):
                self.compute_gradient(vars, grads)
                alpha, top, bottom = self.compute_step(vars, grads, etas, i)
                #self.check_approximation(vars, etas, top, bottom, alpha, i, writer)
                self.apply_step(vars, etas, alpha)
                self.log_iter(vars, i, writer)

        self.postcalc(vars)
        return vars
    
    def precalc(self, vars):
        """One-time setup at the start of BH: shrinkage, obj normalization,
        pos snapshot, initial proj from fwd_tomo + redist."""
        self.eff_demag[:] = (1 + self.shrink_nd) / cp.array(self.norm_magnifications[:, None])

        # normalize obj to work with normal operators (restored at BH exit)
        vars["obj"] /= self.norm_const
        if self.start_iter == 0:
            vars["obj"] *= self.cl_tomo.mask

        self.pos_init = vars['pos'].copy()

        self.fwd_tomo(vars["obj"], out=self.proj_tmp)
        self.redist(self.proj_tmp, vars['proj'])

    def postcalc(self, vars):
        """Restore obj normalization after the BH loop (inverse of precalc)."""
        vars["obj"] *= self.norm_const

    def compute_gradient(self, vars, grads):
        """gradients_cascade + propagate the obj-side gradient through
        fwd_tomo + redist so grads['proj'] is ready for the hessian calls."""
        with nvtx.annotate("gradients"):
            self.gradients(vars, grads)
        with nvtx.annotate(":::BH:fwd_tomo"):
            self.fwd_tomo(grads["obj"], out=self.proj_tmp)
        with nvtx.annotate(":::BH:redist", color='red'):
            self.redist(self.proj_tmp, grads['proj'])

    def compute_step(self, vars, grads, etas, i):
        """One CG step: update the search direction with the new beta and
        return (alpha, top, bottom). Fused single-sweep route when eligible
        (see the `fused_hessian` note on the class), classic three-sweep route
        otherwise."""
        if self.fused_hessian:
            return self._compute_step_fused(vars, grads, etas, i)
        beta = self.compute_beta(vars, grads, etas, i)
        return self.compute_alpha(vars, grads, etas, beta)

    def _compute_step_fused(self, vars, grads, etas, i):
        """beta and alpha from ONE cascade sweep instead of three."""
        with nvtx.annotate(":::BH:calc_step"):
            check = getattr(self, 'check_fused_hessian', False)

            # Every term — cascade, probe-fit, regularization — gives all three
            # forms in one pass over its own data. Reduce BEFORE expanding
            # `bottom` below: the expansion folds B(g,e) and B(e,g) into one
            # 2*beta*Bge term, and for the regularization those two are equal
            # only globally — per rank they differ by a biharmonic flux across
            # the slab boundary (see tests/extra_terms/test_hessian3.py).
            Qgg, Bge, Qee = self.allreduce_scalars(
                *self.hessian3(vars, grads, etas))

            # Steepest descent on the first iteration: eta_new = -g, so
            # bottom = B(-g,-g) = Qgg, which the expansion below already gives at
            # beta = 0. etas is zero there, so the sweep measured Bge = Qee = 0
            # and only the ratio needs the special case (0/0).
            beta = 0.0 if i == self.start_iter else Bge / Qee

            if check and i > self.start_iter:
                ref_beta = self._ref_beta(vars, grads, etas)
                self._log_fused_check(i, "beta", beta, ref_beta)

            # etas <- beta*etas - grads. Must run *after* the sweep: it
            # overwrites the direction the sweep just read (e_pad views it).
            top, = self.allreduce_scalars(self._update_etas(grads, etas, beta))

            bottom = beta * beta * Qee - 2.0 * beta * Bge + Qgg
            # This part is a Schur complement (Qgg - Bge^2/Qee): a difference of
            # positive quantities that nearly cancel when beta is large. The
            # arithmetic is float64, but warn when the result is non-positive
            # (the problem is nonconvex) or survives only a small fraction of
            # the terms that produced it.
            scale = abs(beta * beta * Qee) + abs(2.0 * beta * Bge) + abs(Qgg)
            if not bottom > 1e-6 * scale:
                logger.warning(
                    f"iter={i}: ill-conditioned alpha denominator bottom={bottom:.6e} "
                    f"from Qgg={Qgg:.6e} Bge={Bge:.6e} Qee={Qee:.6e} beta={beta:.6e}")

            if check:
                # etas holds the updated direction, so this is exactly the
                # sweep compute_alpha would have run.
                ref_bottom = self.allreduce_scalars(self.hessian(vars, etas, etas))[0]
                self._log_fused_check(i, "bottom", bottom, ref_bottom)

            alpha = top / bottom
        return alpha, top, bottom

    def _ref_beta(self, vars, grads, etas):
        """beta the classic way — two extra sweeps. Verification only."""
        t, b = self.allreduce_scalars(self.hessian(vars, grads, etas),
                                      self.hessian(vars, etas, etas))
        return t / b

    def _log_fused_check(self, i, name, got, ref):
        """Report |fused - measured| / |measured| for the fused-step check."""
        rel = abs(got - ref) / abs(ref) if ref != 0 else abs(got)
        logger.info(f"iter={i}: fused-check {name:>6}  fused={got:+.9e}  "
                    f"measured={ref:+.9e}  rel={rel:.3e}")

    def _update_etas(self, grads, etas, beta):
        """etas <- beta*etas - grads for every variable, returning the local
        alpha numerator -<eta_new, grad>/rho^2 summed over obj/pos (+prb on
        rank 0)."""
        top = 0
        for v in ("obj", "pos"):
            top -= self.linear_redot_batch(etas[v], grads[v], beta, -1) / self.rho_sq[v]
        # probe is shared across ranks; only rank 0 contributes to the rank-0 sum
        dot_prb = self.linear_redot_batch(etas['prb'], grads['prb'], beta, -1)
        if self.rank == 0:
            top -= dot_prb / self.rho_sq['prb']
        self.linear_batch(etas['proj'], grads['proj'], beta, -1)
        return top

    def compute_beta(self, vars, grads, etas, i):
        """CG coefficient beta = <grad, H·eta> / <eta, H·eta>.
        Returns 0 on the first iter (pure steepest descent)."""
        if i == self.start_iter:
            return 0
        with nvtx.annotate(":::BH:calc beta"):
            top    = self.hessian(vars, grads, etas)
            bottom = self.hessian(vars, etas, etas)
            top, bottom = self.allreduce2(top, bottom)
            return top / bottom

    def compute_alpha(self, vars, grads, etas, beta):
        """Update the search direction (etas) with the new beta, then compute
        the step size alpha. Returns (alpha, top, bottom); top/bottom are
        forwarded to check_approximation."""
        with nvtx.annotate(":::BH:calc_alpha"):
            top = self._update_etas(grads, etas, beta)
            bottom = self.hessian(vars, etas, etas)
            top, bottom = self.allreduce2(top, bottom)
            alpha = top / bottom
        return alpha, top, bottom

    def apply_step(self, vars, etas, alpha):
        """var ← var + alpha·eta for every variable."""
        for v in ("obj", "prb", "pos", "proj"):
            self.linear_batch(vars[v], etas[v], 1, alpha)

    def log_iter(self, vars, i, writer):
        """Error logging + visualization debug for this iter."""
        with nvtx.annotate(":::BH:calc error", color='gray'):
            self.error_debug(vars, i)
        with nvtx.annotate(":::BH:vis_debug", color='gray'):
            self.vis_debug(vars, i, writer)
        # Cache hit/miss for this iter (reset for next).
        if self.rank == 0:
            ch, cm = self.cl_shift.coeff_cache_stats(reset=True)
            ah, am = self.apply_F_cache_stats(reset=True)
            logger.info(f"iter={i}: coeff_cache    hits={ch} misses={cm}")
            logger.info(f"iter={i}: apply_F_cache  hits={ah} misses={am}")

    def hessian(self, vars, grads, etas):
        """Hessian for the full functional, is a sum of 3 terms:
        1. main data fit term calcuated with the cascade rule,
        2. probe fit term,
        3. regularization term

        All three addends are LOCAL; the caller allreduces the sum once. The
        regularization term takes its second direction from the stored e_pad
        (which views etas['obj']), not from the `etas` argument."""
        with nvtx.annotate("hessian"):
            w = self.hessian_cascade(vars, grads, etas)
            if self.rank == 0 and hasattr(self, 'cl_prb_term'):
                w += self.cl_prb_term.hessian(vars["prb"], grads["prb"], etas["prb"])
            if hasattr(self, 'cl_lap_term'):
                w += self.cl_lap_term.hessian(grads["obj"])
        return w

    @timer
    def hessian_cascade(self, vars, grads, etas):
        """"Cascade computation of the hessian for the main term, following the
            composition rule (Carlsson, 2025). Distances run INSIDE the
            theta-chunk loop, ndistchunk at a time, so one upload of the x/y/z
            proj chunks serves ndistchunk distances. `out` is a scalar cupy
            accumulator across every chunk and every k."""

        out = cp.zeros(1, dtype="float32")
        # Identity check on un-sliced pinned arrays (slices per-k would never be `is`-equal).
        y_is_z = grads['prb'] is etas['prb']

        for k0, k1 in self._dist_groups():
            nd = k1 - k0
            self._assert_dist_group(nd)

            @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
            def _hess_dists(self, out, *a):
                # proper inputs: nd*(d, eff_demag, x_pos, y_pos, z_pos) then the
                # three dist-independent proj chunks; non-proper: the three
                # [nd, nz, n] prb bundles.
                d     = a[0 * nd:1 * nd]
                ed    = a[1 * nd:2 * nd]
                x_pos = a[2 * nd:3 * nd]
                y_pos = a[3 * nd:4 * nd]
                z_pos = a[4 * nd:5 * nd]
                x_proj, y_proj, z_proj = a[5 * nd:5 * nd + 3]
                x_prb,  y_prb,  z_prb  = a[5 * nd + 3:5 * nd + 6]

                # coeff(proj) is distance-independent, so the B-spline prefilter
                # of each of the three proj chunks is computed once here and
                # reused by every k below -- nd x fewer full-grid FFT pairs.
                self.cl_shift.coeff_cache_reset()
                for j in range(nd):
                    self._dist_idx = k0 + j
                    self._eff_demag_chunk = ed[j]
                    # apply_F results ARE distance-dependent: reset per k, or nd
                    # detector-plane cascade states would pile up.
                    self.apply_F_cache_reset()
                    x = [x_prb[j], x_proj, x_pos[j]]
                    y = [y_prb[j], y_proj, y_pos[j]]
                    z = y if y_is_z else [z_prb[j], z_proj, z_pos[j]]
                    w = [None, None, None]
                    for id in range(1, len(self.F))[::-1]:
                        w = self.d2F_dF[id](x, y, z, w)
                        fx, y = self.dF[id](x, y)
                        z = y if y_is_z else self.dF[id](x, z, return_x=False)
                        x = fx
                    out[:] += self.d2F_dF[0](x, y, z, w, d[j])

            ks = range(k0, k1)
            _hess_dists(self, out,
                        *[self.data[k]      for k in ks],
                        *[self.eff_demag[k] for k in ks],
                        *[vars['pos'][k]    for k in ks],
                        *[grads['pos'][k]   for k in ks],
                        *[etas['pos'][k]    for k in ks],
                        vars['proj'], grads['proj'], etas['proj'],
                        vars['prb'][k0:k1], grads['prb'][k0:k1], etas['prb'][k0:k1])

        return out[0].get()

    def hessian3(self, vars, grads, etas):
        """{B(g,g), B(g,e), B(e,e)} — the same three terms hessian() sums, but
        as the three bilinear forms of (g,e) instead of one, each term
        contributing all three from a single pass over its own data. Local (not
        allreduced), like hessian().

        The regularization term ignores its arguments: g and e are the interiors
        of the padded slabs it owns (see LaplacianTerm.hessian3)."""
        with nvtx.annotate("hessian3"):
            hgg, hge, hee = self.hessian_cascade3(vars, grads, etas)
            if self.rank == 0 and hasattr(self, 'cl_prb_term'):
                pgg, pge, pee = self.cl_prb_term.hessian3(
                    vars["prb"], grads["prb"], etas["prb"])
                hgg = hgg + pgg
                hge = hge + pge
                hee = hee + pee
            if hasattr(self, 'cl_lap_term'):
                lgg, lge, lee = self.cl_lap_term.hessian3()
                hgg = hgg + lgg
                hge = hge + lge
                hee = hee + lee
        return hgg, hge, hee

    @timer
    def hessian_cascade3(self, vars, grads, etas):
        """The three bilinear forms {B(g,g), B(g,e), B(e,e)} of the main term
        from ONE cascade sweep.

        Same composition rule (Carlsson, 2025) as hessian_cascade, but the
        x-chain — the part that streams vars/grads/etas['proj'] and `data` over
        PCIe, i.e. essentially all of the cost — is advanced once and shared by
        all three pairs instead of being recomputed per call.

        `out` is a 3-element cupy accumulator. Chunking classifies it as a
        *non-proper* output (its axis-0 length is not the chunk's theta count)
        and hands it whole to every chunk, exactly as it does the scalar in
        hessian_cascade."""

        # ...which relies on 3 != the chunked axis length. Only violated by a
        # degenerate decomposition, but a silent mis-slice would be very hard
        # to spot in the numbers.
        assert self.local_ntheta != 3, "local_ntheta==3 aliases the accumulator shape"

        out = cp.zeros(3, dtype="float32")

        for k0, k1 in self._dist_groups():
            nd = k1 - k0
            self._assert_dist_group(nd)

            @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
            def _hess3_dists(self, out, *a):
                # proper inputs: nd*(d, eff_demag, x_pos, g_pos, e_pos) then the
                # three dist-independent proj chunks; non-proper: the three
                # [nd, nz, n] prb bundles.
                d     = a[0 * nd:1 * nd]
                ed    = a[1 * nd:2 * nd]
                x_pos = a[2 * nd:3 * nd]
                g_pos = a[3 * nd:4 * nd]
                e_pos = a[4 * nd:5 * nd]
                x_proj, g_proj, e_proj = a[5 * nd:5 * nd + 3]
                x_prb,  g_prb,  e_prb  = a[5 * nd + 3:5 * nd + 6]

                # coeff(x_proj/g_proj/e_proj) is distance-independent: three FFT
                # prefilters per chunk instead of three per (chunk, distance).
                self.cl_shift.coeff_cache_reset()
                for j in range(nd):
                    self._dist_idx = k0 + j
                    self._eff_demag_chunk = ed[j]
                    self.apply_F_cache_reset()   # per dist: cascade states are per-dist
                    x = [x_prb[j], x_proj, x_pos[j]]
                    g = [g_prb[j], g_proj, g_pos[j]]
                    e = [e_prb[j], e_proj, e_pos[j]]
                    # Passing the *same* list object for y and z on the diagonal forms
                    # keeps the `y12 is z12` / cached-coeff fast paths in d2F_dF1 and
                    # d2F_dF3 alive — the y_is_z flag becomes structural.
                    wgg = [None, None, None]
                    wge = [None, None, None]
                    wee = [None, None, None]
                    for id in range(1, len(self.F))[::-1]:
                        # d2F_dF[id] must see the pre-update x, g, e — hence all three
                        # contractions before dF[id] advances the chains.
                        wgg = self.d2F_dF[id](x, g, g, wgg)
                        wge = self.d2F_dF[id](x, g, e, wge)
                        wee = self.d2F_dF[id](x, e, e, wee)
                        fx, gn = self.dF[id](x, g)
                        en = self.dF[id](x, e, return_x=False)
                        x, g, e = fx, gn, en
                    out[0:1] += self.d2F_dF[0](x, g, g, wgg, d[j])
                    out[1:2] += self.d2F_dF[0](x, g, e, wge, d[j])
                    out[2:3] += self.d2F_dF[0](x, e, e, wee, d[j])

            ks = range(k0, k1)
            _hess3_dists(self, out,
                         *[self.data[k]      for k in ks],
                         *[self.eff_demag[k] for k in ks],
                         *[vars['pos'][k]    for k in ks],
                         *[grads['pos'][k]   for k in ks],
                         *[etas['pos'][k]    for k in ks],
                         vars['proj'], grads['proj'], etas['proj'],
                         vars['prb'][k0:k1], grads['prb'][k0:k1], etas['prb'][k0:k1])

        h = out.get()
        return float(h[0]), float(h[1]), float(h[2])

    def gradients(self, vars, grads):
        """Full gradient, consists of 2 terms:
        1. main data fit term calcuated with the cascade rule,
        2. probe fit term,
        """
        self.gradients_cascade(vars, grads)

        with nvtx.annotate(":::BH:redist back", color='red'):
            self.redist(grads['proj'], self.proj_tmp, direction='backward')

        # part2, parallelization over object slices: adjoint Radon
        self.adj_tomo(grads['obj'], self.proj_tmp)
        if hasattr(self, 'cl_lap_term'):
            self.cl_lap_term.gradient(grads['obj'])

        if self.rank == 0 and hasattr(self, 'cl_prb_term'):
            self.cl_prb_term.gradient(grads["prb"], vars["prb"], self.rho_sq['prb'])

        grads['prb'][:] = self.allreduce(grads['prb'])
        
    @timer
    def gradients_cascade(self, vars, grads):
        """Cascade gradient for the main term (Carlsson, 2025).

        Distances run INSIDE the theta-chunk loop, ndistchunk per @gpu_batch
        pass: one upload of the vars['proj'] chunk feeds them all, and their
        Deltapsi contributions are summed on the GPU instead of through a
        per-distance read-modify-write of grads['proj'] across PCIe.

        grads['proj'] is still passed as both proper input and proper output so
        that successive *groups* accumulate; with ndistchunk >= ndist there is
        one group and that costs a single extra read of a zeroed slab. The
        trailing cl_shift.coeff (parent's gF3 returns un-coeff'd Deltapsi) is
        applied by the last group, once the sum over all distances is complete.
        """

        grads['proj'][:] = 0   # zero accumulator before the group loop
        grads['prb'][:]  = 0   # each k writes its own slot

        groups = self._dist_groups()
        for gi, (k0, k1) in enumerate(groups):
            nd   = k1 - k0
            last = (gi == len(groups) - 1)
            self._assert_dist_group(nd)

            @self.gpu_batch(axis_out=0, axis_inp=0, nout=nd + 2)
            def _grad_dists(self, gradproj_out, *a):
                # outputs: nd*gradpos (proper), gradprb [nd,nz,n] (non-proper)
                gradpos     = a[0:nd]
                gradprb     = a[nd]
                # proper inputs: gradproj_in, nd*(d, eff_demag, pos), proj;
                # non-proper: the [nd, nz, n] prb bundle.
                b           = a[nd + 1:]
                gradproj_in = b[0]
                d    = b[1 + 0 * nd:1 + 1 * nd]
                ed   = b[1 + 1 * nd:1 + 2 * nd]
                pos  = b[1 + 2 * nd:1 + 3 * nd]
                proj = b[1 + 3 * nd]
                prb  = b[2 + 3 * nd]

                # coeff(proj) is distance-independent -- one prefilter per chunk.
                self.cl_shift.coeff_cache_reset()
                # Accumulate straight into the output chunk buffer. gF3's y[1] is
                # a freshly zeroed [chunk, nzpsi, npsi] from dcurlySadjc, so it can
                # be scaled in place -- no separate accumulator and no `acc * rho`
                # temporary, i.e. two object-plane slabs saved over the naive form.
                gradproj_out[:] = gradproj_in
                for j in range(nd):
                    self._dist_idx = k0 + j
                    self._eff_demag_chunk = ed[j]
                    self.apply_F_cache_reset()   # per dist: cascade states are per-dist
                    x = [prb[j], proj, pos[j]]
                    y = d[j]
                    for id in range(len(self.gF)):
                        y = self.gF[id](x, y)

                    gradprb[j] += y[0] * self.rho_sq['prb']
                    gradpos[j][:] = y[2] * self.rho_sq['pos']
                    y[1] *= self.rho_sq['obj']
                    gradproj_out += y[1]
                if last:
                    gradproj_out[:] = self.cl_shift.coeff(gradproj_out)

            ks = range(k0, k1)
            _grad_dists(self,
                        grads['proj'],                                  # out
                        *[grads['pos'][k] for k in ks],                 # out
                        grads['prb'][k0:k1],                            # out
                        grads['proj'],                                  # inp
                        *[self.data[k]      for k in ks],
                        *[self.eff_demag[k] for k in ks],
                        *[vars['pos'][k]    for k in ks],
                        vars['proj'], vars['prb'][k0:k1])

    @timer
    def fwd_tomo(self, obj, out):
        """Forward tomography operator"""
        
        @self.gpu_batch(axis_out=1, axis_inp=0,nout=1)
        def _fwd_tomo(self, out, obj):
            out[:] = self.cl_tomo.R(obj)
            
        _fwd_tomo(self, out, obj)
        return out    
    
    @timer
    def adj_tomo(self, gradu, gradproj):
        @self.gpu_batch(axis_out=0, axis_inp=1, nout=1)
        def _adj_tomo(self, gradu, gradproj):
            gradu[:] = self.cl_tomo.RT(gradproj)
        
        _adj_tomo(self, gradu, gradproj)

    ####################### Functions for the cascade (following math notes for variables)
    # F* - functional
    # dF* - differential
    # d2F* - second order term for hessian
    # gF* - gradient
    #######################################################################################

    def apply_F_from(self, x, from_level):
        """Apply F[from_level], F[from_level+1], ..., F[len(F)-1] inside-out, returning the
        partial cascade value at level (from_level-1). Used by gF* methods that need to
        "fast-forward" the input x through cascade levels above their own.

        Memoized recursively by (id(x), from_level): within one cascade pass gF0..gFN
        share the same outer x, so without caching F3 would be re-evaluated N times and
        F2 (N-1) times etc. MUST be paired with apply_F_cache_reset() per chunk —
        id() can be reused across distinct objects once an earlier one is GC'd.
        """
        if from_level >= len(self.F):
            return x
        key = (id(x), from_level)
        cached = self._apply_F_cache.get(key)
        if cached is not None:
            self._apply_F_hits += 1
            return cached
        self._apply_F_misses += 1
        upstream = self.apply_F_from(x, from_level + 1)
        result = self.F[from_level](upstream)
        self._apply_F_cache[key] = result
        return result

    def apply_F_cache_reset(self):
        self._apply_F_cache = {}

    def apply_F_cache_stats(self, reset=False):
        stats = (self._apply_F_hits, self._apply_F_misses)
        if reset:
            self._apply_F_hits = 0
            self._apply_F_misses = 0
        return stats


    ####### F0(x0) = 1/n\||x0|-d\|_2^2
    @staticmethod
    @cp.fuse()
    def _F0_fused(x, d):
        t = cp.abs(x) - d
        return t * t

    @nvtx.annotate("F0", color="green")
    def F0(self, x, d):
        """In: (x0), Out: const"""
        return 1 / self.data_size * cp.sum(self._F0_fused(x, d))

    @staticmethod
    @cp.fuse()
    def _dF0_fused(x, d):
        return x - d * (x / cp.abs(x))

    @nvtx.annotate("dF0", color="green")
    def dF0(self, x, y, d, return_x=False):
        """In: (x0,y0), Out: const"""
        return 2 / self.data_size * redot(self._dF0_fused(x, d), y)

    @staticmethod
    @cp.fuse()
    def _d2F_dF0_fused(x, y, z, w, d):
        absval = cp.abs(x)
        l0 = x / absval
        d0 = d / absval
        v = (1 - d0) * reprod(y, z) + d0 * reprod(l0, y) * reprod(l0, z)
        if w is not None:
            v += reprod(x - d * l0, w)
        return v

    @nvtx.annotate("d2F0_dF0", color="purple")
    def d2F_dF0(self, x, y, z, w, d):
        """In: (x0,y0,z0,w0), Out: const"""
        return 2 / self.data_size * cp.sum(self._d2F_dF0_fused(x, y, z, w, d))

    @staticmethod
    @cp.fuse()
    def _gF0_fused(x, y, scale):
        td = y * (x / cp.abs(x))
        return scale * (x - td)

    @nvtx.annotate("gF0", color="green")
    def gF0(self, x, y):
        """In: x, y = F0(F1(..(x)))), Out: y0"""
        x = self.apply_F_from(x, 1)
        return self._gF0_fused(x, y, np.float32(2 / self.data_size))
    
    ####### x0 = F1(x11,x12) = D(x11\cdot x12)
    @nvtx.annotate("F1", color="green")
    def F1(self, x):
        """In: (x11,x12), Out: x0. Per-dist: arrays carry singleton dist axis; uses self._dist_idx."""

        x11, x12 = x  # x11: [nz, n], x12: [chunk, nz, n]
        return self.cl_prop.D(x11 * x12, self._dist_idx)

    @nvtx.annotate("dF1", color="green")
    def dF1(self, x, y, return_x=True):
        """In: (x11,x12),(y11,y12) Out: y0. Per-dist."""
        x11, x12 = x
        y11, y12 = y
        y0 = self.cl_prop.D(y11 * x12 + x11 * y12, self._dist_idx)
        if return_x:
            x0 = self.cl_prop.D(x11 * x12, self._dist_idx)
            return x0, y0
        return y0

    @nvtx.annotate("d2F_dF1", color="purple")
    def d2F_dF1(self, x, y, z, w):
        """In: (x11,x12),(y11,y12),(z11,z12) Out: y0. Per-dist."""
        x11, x12 = x
        y11, y12 = y
        z11, z12 = z
        w11, w12 = w

        if y12 is z12:
            y0 = 2 * y11 * y12
        else:
            y0 = y11 * z12 + z11 * y12

        if w11 is not None:
            y0 += w11 * x12
        if w12 is not None:
            y0 += x11 * w12

        return self.cl_prop.D(y0, self._dist_idx)

    @nvtx.annotate("gF1", color="green")
    def gF1(self, x, y):
        """In: x=(x01,x02,x03),(y0) Out: y11,y12. Per-dist."""
        y0 = y  # [chunk, nz, n]
        x = self.apply_F_from(x, 2)
        x11, x12 = x  # x11: [nz, n], x12: [chunk, nz, n]
        y12 = self.cl_prop.DT(y0, self._dist_idx)
        y11 = cp.sum(y12 * np.conj(x12), axis=0)  # [nz, n]
        y12 *= np.conj(x11)
        return y11, y12

    ######## (x11,x12) = F2(x21,x22) = (x21,e^{1j x22})
    @staticmethod
    @cp.fuse()
    def _F2_fused(x22):
        return cp.exp(1j * x22)

    @nvtx.annotate("F2", color="green")
    def F2(self, x):
        """In: (x21,x22) Out: (x11,x12)"""

        x21, x22 = x
        x11 = x21
        x12 = self._F2_fused(x22)
        return x11, x12

    @staticmethod
    @cp.fuse()
    def _dF2_fused(x22, y22):
        x12 = cp.exp(1j * x22)
        y12 = x12 * 1j * y22
        return x12, y12

    @nvtx.annotate("dF2", color="green")
    def dF2(self, x, y, return_x=True):
        """In: (x21,x22),(y21,y22) Out: (x11,x12),(y11,y12)"""

        x21, x22 = x
        y21, y22 = y

        x12, y12 = self._dF2_fused(x22, y22)
        x11 = x21
        y11 = y21

        return ([x11, x12], [y11, y12]) if return_x else [y11, y12]
    
    @staticmethod
    @cp.fuse()
    def _d2F_dF2_fused(x22, y22, z22, w22):
        y12 = cp.exp(1j * x22) * (-y22 * z22)
        if w22 is not None:
            y12 = y12 + cp.exp(1j * x22) * 1j * w22
        return y12

    @nvtx.annotate("d2F_dF2", color="purple")
    def d2F_dF2(self, x, y, z, w):
        """In: (x21,x22),(y21,y22),(z21,z22),(w21,w22) Out: (y11,y12)"""

        x21, x22 = x
        y21, y22 = y
        z21, z22 = z
        w21, w22 = w

        y12 = self._d2F_dF2_fused(x22, y22, z22, w22)
        y11 = w21

        return [y11, y12]
    
    @staticmethod
    @cp.fuse()
    def _gF2_fused(x22, y12):
        return (-1j) * y12 * cp.conj(cp.exp(1j * x22))

    @nvtx.annotate("gF2", color="green")
    def gF2(self, x, y):
        """In: x(x01, x02, x03) ,(y11,y12) Out: (y21,y22)"""

        y11, y12 = y

        x = self.apply_F_from(x, 3)
        x21, x22 = x

        y22 = self._gF2_fused(x22, y12)
        y22 = y22.real if self.obj_dtype == 'float32' else y22

        y21 = y11
        return [y21, y22]
    
    ####### (x21,x22) = F3(x31,x32,x33) = (x31,S_{x_33}(x32))
    @nvtx.annotate("F3", color="green")
    def F3(self, x):
        """In: (x31, x32, x33)  Out: (x21,x22). Per-dist; uses self._dist_idx."""
        x31, x32, x33 = x  # x32: [chunk, nzobj, nobj] dist-agnostic, x33: [chunk, 2]
        c   = self.cl_shift.coeff_cached(x32)
        ed  = self._eff_demag_chunk
        x22 = self.cl_shift.curlySc(c, x33, ed)
        return [x31, x22]

    @nvtx.annotate("dF3", color="green")
    def dF3(self, x, y, return_x=True):
        """In: (x31, x32, x33),(y31, y32, y33)  Out: (y31, y22). Per-dist."""
        x31, x32, x33 = x
        y31, y32, y33 = y
        c   = self.cl_shift.coeff_cached(x32)
        c1  = self.cl_shift.coeff_cached(y32)
        ed  = self._eff_demag_chunk
        y22 = self.cl_shift.dcurlySc(c, x33, ed, c1, y33)
        if return_x:
            x22 = self.cl_shift.curlySc(c, x33, ed)
            return [x31, x22], [y31, y22]
        return [y31, y22]

    @nvtx.annotate("d2F_dF3", color="purple")
    def d2F_dF3(self, x, y, z, w):
        """In: (x31, x32, x33),(y31, y32, y33),(z31, z32, z33),(w31, w32, w33)  Out: (y21, y22). Per-dist."""
        x31, x32, x33 = x
        y31, y32, y33 = y
        z31, z32, z33 = z
        w31, w32, w33 = w

        c   = self.cl_shift.coeff_cached(x32)
        cy  = self.cl_shift.coeff_cached(y32)
        cz  = self.cl_shift.coeff_cached(z32)
        ed  = self._eff_demag_chunk
        y22 = self.cl_shift.d2curlySc(c, x33, ed, cy, y33, cz, z33)

        if w32 is not None:
            cw = self.cl_shift.coeff_cached(w32)
            y22 += self.cl_shift.dcurlySc(c, x33, ed, cw, w33)

        return [w31, y22]

    @nvtx.annotate("gF3", color="green")
    def gF3(self, x, y):
        """In: x(x01, x02, x03) ,(y21,y22) Out: (y31,y32). Per-dist.

        Returns un-coeff'd Deltapsi in y32; caller must sum over distances and apply
        cl_shift.coeff() once after the dist loop.
        """
        y21, y22 = y  # y22: [chunk, nz, n]
        x = self.apply_F_from(x, 4)
        x31, x32, x33 = x
        c = self.cl_shift.coeff_cached(x32)
        y32, y33 = self.cl_shift.dcurlySadjc(c, x33, self._eff_demag_chunk, y22)
        return [y21, y32, y33]

    @timer
    def min(self, prb, obj, pos, proj):
        out = cp.zeros(1, dtype="float32")

        for k0, k1 in self._dist_groups():
            nd = k1 - k0
            self._assert_dist_group(nd)

            @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
            def _min_dists(self, out, c_proj, *a):
                c_pos  = a[0 * nd:1 * nd]
                c_data = a[1 * nd:2 * nd]
                c_ed   = a[2 * nd:3 * nd]
                c_prb  = a[3 * nd]
                self.cl_shift.coeff_cache_reset()
                for j in range(nd):
                    self._dist_idx = k0 + j
                    self._eff_demag_chunk = c_ed[j]
                    self.apply_F_cache_reset()
                    x = [c_prb[j], c_proj, c_pos[j]]
                    y = self.apply_F_from(x, 1)
                    out[:] += self.F0(y, c_data[j])

            ks = range(k0, k1)
            _min_dists(self, out, proj,
                       *[pos[k]            for k in ks],
                       *[self.data[k]      for k in ks],
                       *[self.eff_demag[k] for k in ks],
                       prb[k0:k1])

        out = out[0]
        if self.rank == 0 and hasattr(self, 'cl_prb_term'):
            out += self.cl_prb_term.energy_local(prb)
        if hasattr(self, 'cl_lap_term'):
            out += self.cl_lap_term.energy_local()
        return self.allreduce(np.array(out.get(),dtype='float32'))

    def vis_debug(self, vars, i, writer=None):
        """Per-iter checkpoint write (pos-error plot bundled in)."""
        if writer is None or not (i % self.checkpoint_step == 0 and self.checkpoint_step != -1) or i <= self.start_iter:
            return
        writer.write_checkpoint(vars, i, self.norm_const, pos_init=self.pos_init)

    def check_approximation(self, vars, etas, top, bottom, alpha, i, writer=None):
        """Compare the real functional along the descent direction with the
        quadratic model used to pick alpha:
            f_real(t)   = self.min(vars + t*etas)
            f_approx(t) = self.min(vars) - top*t + 0.5*bottom*t**2
        Sampled at npp points in [0, 2*alpha]. With a writer, saves a PNG per
        triggered iteration under writer.path_out/check_approximation/; without
        a writer, shows the plot inline (rank 0 only).
        """
        if not (i % self.checkpoint_step == 0 and self.checkpoint_step != -1):
            return

        # lazy-allocate scratch buffers (only created on the first triggered iter)
        if not hasattr(self, '_chk_objt'):
            self._chk_objt  = make_pinned(vars['obj'].shape,  self.obj_dtype)
            self._chk_projt = make_pinned(vars['proj'].shape, self.obj_dtype)
            self._chk_prbt  = cp.empty_like(vars['prb'])
            self._chk_post  = cp.empty_like(vars['pos'])

        objt, prbt, post, projt = self._chk_objt, self._chk_prbt, self._chk_post, self._chk_projt

        npp = 5
        t = np.linspace(0, 2 * alpha, npp).astype('float32')
        err_real = np.zeros(npp, dtype='float32')

        for k in range(npp):
            self.linear_batch(vars['obj'],  etas['obj'],  1, t[k], out=objt)
            self.linear_batch(vars['prb'],  etas['prb'],  1, t[k], out=prbt)
            self.linear_batch(vars['pos'],  etas['pos'],  1, t[k], out=post)
            self.linear_batch(vars['proj'], etas['proj'], 1, t[k], out=projt)
            err_real[k] = self.min(prbt, objt, post, projt)

        f0 = self.min(vars['prb'], vars['obj'], vars['pos'], vars['proj'])
        err_approx = f0 - top * t + 0.5 * bottom * t * t

        if self.rank != 0:
            return

        if writer is None:
            mshow_approx(t, err_real, err_approx, True)
            return

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(t, err_real,   "o-", label="real")
        ax.plot(t, err_approx, "x-", label="approx")
        ax.set_xlabel("t")
        ax.set_ylabel("functional")
        ax.set_title(f"iter {i}: alpha={alpha:.3e}")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        out_dir  = os.path.join(writer.path_out, "check_approximation")
        os.makedirs(out_dir, exist_ok=True)
        png_path = os.path.join(out_dir, f"check_approx_{i:04}.png")
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        logger.info(f"check_approximation plot → {png_path}")

    def error_debug(self, vars, i):
        """Error logging and CSV checkpoint export. i=-1 is the initial-state call from
        BH (before the loop) and is always logged regardless of error_step."""
        if i != -1 and not (i % self.error_step == 0 and self.error_step != -1):
            return
            
        t_min = time.time()
        err = self.min(vars["prb"], vars["obj"], vars["pos"], vars["proj"])
        min_time = time.time() - t_min
        if self.rank==0:
            if i==-1:
                logger.warning(f"Initial {err=:1.5e} ")
                self.table.loc[len(self.table)] = [i, err, 0]
            else:
                ittime = time.time()-self.time_start
                logger.warning(f"iter={i}: {ittime:.4f}sec (-min={ittime - min_time:.4f}sec) {err=:1.5e} ")
                self.table.loc[len(self.table)] = [i, err, ittime]
            self.time_start = time.time()
            if hasattr(self, 'path_out'):
                name = f"{self.path_out}/conv.csv"
                os.makedirs(os.path.dirname(name), exist_ok=True)
                self.table.to_csv(name, index=False)

    def gen_sqrt_data(self, vars, out):
        """Generate synthetic data. Distances run inside the theta-chunk loop,
        ndistchunk per pass, so the vars['proj'] chunk is uploaded once for all
        of them instead of once per distance."""

        self.eff_demag[:]  = (1 + self.shrink_nd) / cp.array(self.norm_magnifications[:, None])
        vars["obj"] /= self.norm_const
        self.fwd_tomo(vars["obj"], out=self.proj_tmp)
        self.redist(self.proj_tmp, vars['proj'])

        for k0, k1 in self._dist_groups():
            nd = k1 - k0
            self._assert_dist_group(nd)

            @self.gpu_batch(axis_out=0, axis_inp=0, nout=nd)
            def _gen_data_dists(self, *a):
                o      = a[0:nd]                        # nd proper outputs
                c_proj = a[nd]
                c_pos  = a[nd + 1 + 0 * nd:nd + 1 + 1 * nd]
                c_ed   = a[nd + 1 + 1 * nd:nd + 1 + 2 * nd]
                c_prb  = a[nd + 1 + 2 * nd]
                self.cl_shift.coeff_cache_reset()
                for j in range(nd):
                    self._dist_idx = k0 + j
                    self._eff_demag_chunk = c_ed[j]
                    self.apply_F_cache_reset()
                    x = [c_prb[j], c_proj, c_pos[j]]
                    y = self.apply_F_from(x, 1)
                    o[j][:] = cp.abs(y)

            ks = range(k0, k1)
            _gen_data_dists(self,
                            *[out[k] for k in ks],
                            vars['proj'],
                            *[vars['pos'][k]    for k in ks],
                            *[self.eff_demag[k] for k in ks],
                            vars['prb'][k0:k1])

        vars["obj"] *= self.norm_const




