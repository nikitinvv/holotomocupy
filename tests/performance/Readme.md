# Performance benchmark — multi-distance holotomography

Fully-synthetic perf test for `holotomocupy.rec_mpi.Rec` and a log parser
that summarises one BH iteration.

## Files

| file | purpose |
|------|---------|
| `test.py`              | the benchmark — generates phantom, probe, positions, runs `Rec.BH()`, writes a perf log |
| `test_mosaic.py`       | the same benchmark for a MOSAIC scan (2 × 5 tiles × 4 distances = 40 distances over one wide object) |
| `run_mosaic.sh`        | drives `test_mosaic.py` over a list of binning levels, then parses each log |
| `run.sh`               | the same for `test.py`, over a list of detector sizes |
| `set_affinity_gpu.sh`  | per-rank GPU pinning (`CUDA_VISIBLE_DEVICES = local_rank % ngpus`) |
| `parse_perf_log.py`    | reads the perf log; reports per-iter timing breakdown + max process / GPU memory |
| `log512`               | example log from a `--n 512 --ntheta 450 --nchunk 4` run on 8 ranks × 4 GPUs |

> **Running this on a cluster?** Jump to [Running on a cluster](#running-on-a-cluster)
> — node counts, `nchunk` per bin, and the memory budget per rank.

## Quick start

Single rank:
```bash
python test.py --n 2048 --ntheta 1800 --nchunk 4
python parse_perf_log.py perf.log --iter 1
```

Multi-rank with one GPU per local rank (round-robin pinning):
```bash
mpirun -np 8 ./set_affinity_gpu.sh \
    python test.py --n 2048 --ntheta 1800 --nchunk 4 --log log2048_8
python parse_perf_log.py log2048_8 --iter 1
```

Or sweep several detector sizes and parse each log:
```bash
./run.sh                       # NP=4 ranks, sizes 512 / 1024 / 2048
NP=8 SIZES="1024 2048" ./run.sh
```

## `test.py` CLI

| flag       | default      | meaning |
|------------|--------------|---------|
| `--n`      | **required** | detector size (square: `nz = n`) |
| `--ntheta` | **required** | number of projection angles |
| `--nchunk` | `4`          | theta chunk size for the batched GPU ops — main perf knob |
| `--ndistchunk` | `0` (= all) | distances sharing one upload of a theta chunk of `proj`; `1` restores the old outer-distance loop |
| `--log`    | `perf.log`   | output log path on rank 0 (`''` disables file logging) |

Other knobs (object padding, `ndist`, `niter`, dtype, regularization, …)
are hardcoded at the top of `test.py` — edit there. Defaults are:
`ndist = 4`, `niter = 2`, `obj_dtype = complex64`,
`nobj = 3264·n/2048` (scales linearly with `n`, matches the brain-Y350 config),
`lam_prbfit = lam_laplacian = 0`, `checkpoint_step = -1`, `error_step = 1`.

The synthesised inputs (probe, object slab, positions) are deterministic
across machines and rank counts — seeded from a single `MASTER_SEED` via
`numpy.SeedSequence.spawn()` per global slice/distance/theta index.

## Mosaic benchmark — `test_mosaic.py`

Emulates a reconstruction of the mosaic scan implemented in
`tests/mosaic_brain` (`gen_data.py` → `steps15.py` → `step6.py`), without any
data on disk. The tiles are flattened onto the distance axis exactly as
`mosaic_reader.MosaicReader` presents them to the solver:

```
ndist = ntiles * ndist_tile,   flat index = tile*ndist_tile + k
z1    = tile(z1_tile, ntiles)
```

so the default 2 × 5 tiles × 4 distances give `ndist = 40` over one wide object
grid, and each tile's place on the mosaic lives in `vars['pos']`.

Nothing is forward-modelled — the point is the cost of one BH iteration at this
size, and BH runs a fixed number of iterations regardless of the values:

| input | value |
|-------|-------|
| `prb`  | `1` (flat probe, as a from-scratch `step6` starts) |
| `ref`  | `\|D·prb\|`, so the probe-fit regularizer has something to fit |
| `data` | random, positive (one random frame per distance × a per-angle scale) |
| `pos`  | tile offset + random per-angle encoder jitter (±30 detector px) |
| `obj`  | `0` — a from-scratch start, as `step6` with `write_obj_init=true` |

Sizes follow the detector: at `ndet = 2048` the scan has 6000 angles (the
`ntheta` of `mosaic_brain/config_gen.conf`), and both scale with the binning
level, so `--bin b` gives

```
n = nz = 2048 >> b      ntheta = 6000 >> b      nobj, nzobj = mosaic >> b
```

The object grid is derived from the tile layout itself — outermost tile offset
+ jitter + the half-FOV of the least-magnified distance, rounded up — so it
always holds the whole mosaic. For the default layout that is
`6144 × 12288 × 12288` at bin 0. Acquisition geometry, tile pitch, jitter,
`mask`, `rho` and `lam_prbfit` are copied from
`tests/mosaic_brain/config_gen.conf` + `config_step6.conf`, so the two are
directly comparable.

### Plan first — host memory is the limit

```bash
python test_mosaic.py --bin 2 --nchunk 8 --nranks 8 --plan
```

prints the pinned-host and (approximate) GPU footprint per rank without
allocating anything or touching a GPU. Rough pinned-host totals over all ranks
(`obj` + `proj` + `proj_tmp` + `data`):

| bin | detector | angles | object | host total | GPU/rank (`nchunk=4`) |
|-----|----------|--------|--------|------------|------------------------|
| 3   | 256      | 750    | 768 × 1536²   | 74 GB  | 0.8 GB |
| 2   | 512      | 1500   | 1536 × 3072²  | 594 GB | 3.3 GB |
| 1   | 1024     | 3000   | 3072 × 6144²  | 4.6 TB | 13.0 GB |
| 0   | 2048     | 6000   | 6144 × 12288² | 37 TB  | 51.9 GB |

The GPU side is comfortable by comparison up to bin 1, so the rank count is set
by RAM, not by the devices.

### Running

```bash
NP=8 BINS="3 2" ./run_mosaic.sh          # run + parse each bin
NP=8 BINS="3 2 1" ./run_mosaic.sh --plan # sizes only
```

or directly:

```bash
mpirun -np 8 ./set_affinity_gpu.sh \
    python test_mosaic.py --bin 2 --nchunk 8 --log logmosaic2_8
python parse_perf_log.py logmosaic2_8 --iter 1
```

The log format is identical to `test.py`'s, so `parse_perf_log.py` works
unchanged. Before iterating, rank 0 logs where every tile window lands on the
object grid — the same placement check `mosaic_brain/step6.py` prints.

### Memory

`@timer` records GPU memory as it happens to stand at the end of each timed
call, and `parse_perf_log.py` reports the maximum it saw. That is enough to
compare runs; transient peaks inside a call (cuFFT work areas, the scratch in
`Shift`/`Tomo`) and the allocation phase are not covered. For an up-front
estimate, run with `--plan`, which prints the pinned-host and GPU budgets per
rank without allocating anything.

### `test_mosaic.py` CLI

| flag | default | meaning |
|------|---------|---------|
| `--bin`        | **required** | `n = nz = 2048>>bin`, `ntheta = 6000>>bin` |
| `--nchunk`     | `4`   | theta chunk size — main perf knob |
| `--ndistchunk` | `0` (= all) | distances sharing one upload of a theta chunk of `proj`; `1` restores the old outer-distance loop |
| `--ntile-h`    | `5`   | tiles per row |
| `--ntile-v`    | `2`   | tile rows |
| `--ndist-tile` | `4`   | distances per tile (max 4) |
| `--niter`      | `2`   | BH iterations (iter 0 is warmup) |
| `--ntheta`     | auto  | override the `6000>>bin` angle count |
| `--nobj` / `--nzobj` | auto | override the object grid (binned px) |
| `--plan`       | off   | print sizes + memory and exit |
| `--nranks`     | MPI size | rank count to assume in `--plan` |
| `--log`        | `perf_mosaic.log` | log path on rank 0 |

## Running on a cluster

Written for a node with **8 GPUs (H100, 80 GB), 2 TB of host RAM**. Everything
below assumes **one MPI rank per GPU**, i.e. 8 ranks per node.

### What these benchmarks measure

Both run `Rec.BH()` — the Bilinear-Hessian conjugate-gradient solver — on
synthetic data for a fixed number of iterations, and time every internal stage.
Nothing is read from disk and no result is written; the phantom, probe and
positions are generated in memory from a fixed seed, so a run is reproducible
and depends only on `(bin, nranks, nchunk, ndistchunk)`. The point is the
timing breakdown, not the reconstruction.

* **`test_mosaic.py`** — the one that matters. A mosaic scan: 2 × 5 tiles at
  4 propagation distances each, flattened to **ndist = 40** distances over one
  wide object. This is the workload we are optimizing for.
* **`test.py`** — the single-tile baseline (ndist = 4), same solver. Useful as
  a sanity check and much cheaper.

`--bin` sets the problem size: `n = nz = 2048>>bin` detector pixels and
`ntheta = 6000>>bin` angles. **bin 0 is the full production problem**; bin 3 is
1/8 of it in every dimension and runs on a single node.

### Setup

```bash
pip install -e .                      # from the repo root
```
Needs `cupy` (matching the CUDA toolkit), `mpi4py` built against the site MPI,
plus `numpy`, `h5py`, `psutil`, `nvtx`, `tifffile`, `matplotlib`, `pandas`.
The CUDA kernels ship as source and are JIT-compiled by CuPy on first use —
that is why iteration 0 is a warmup and the summary reports iteration 1.

### Launching

`set_affinity_gpu.sh` pins each rank to one GPU via
`CUDA_VISIBLE_DEVICES = local_rank % ngpus`. It reads either the Open MPI or
the SLURM rank variables, so both launchers work:

```bash
# SLURM
srun -N 4 --ntasks-per-node 8 --gpus-per-task 1 \
     ./set_affinity_gpu.sh python test_mosaic.py --bin 1 --nchunk 6 --log logmosaic1_32

# Open MPI
mpirun -np 32 --map-by ppr:8:node ./set_affinity_gpu.sh \
     python test_mosaic.py --bin 1 --nchunk 6 --log logmosaic1_32
```

Each run writes its log on rank 0 and **appends its own parsed summary to the
end of that log**, so the log file alone is the deliverable — send those back.

### The two knobs

|  | drives | scales with |
|---|---|---|
| `--nchunk` | **GPU** memory | `nchunk × nzobj × nobj` — the *global* object grid |
| `--ndistchunk` | how many of the 40 distances share one upload of a `proj` chunk | detector plane only |

The important and slightly counter-intuitive part: **adding nodes buys host
memory, not GPU memory.** Host arrays are split over ranks, so pinned RAM per
rank falls as `1/nranks`. The GPU chunking pool is sized from `nzobj` and
`nobj`, which are global and do not shrink. So `nchunk` is a per-bin constant —
if you add nodes to fit a bigger problem in RAM, do *not* expect to raise
`nchunk` too.

**`--ndistchunk`: leave it at the default (0 = all 40).** Distances were
hoisted inside the theta-chunk loop so one upload of a `proj` chunk serves all
of them. What has to be resident per distance is only detector-plane
(`nz × n`), and the object plane (`nzobj × nobj`) is ~35× larger, so all 40
cost a fraction of the three `proj` chunks that were already there — the pool
is unchanged. The break-even is **107 distances** (independent of `nchunk` and
of `bin`), so at 40 there is nothing to tune. `--ndistchunk 1` restores the old
outer-distance loop; it is there to reproduce the pre-optimization numbers, and
it is 3–4× slower on the cascades. Use it only for an A/B.

### Recommended settings

`lth` is `local_ntheta` = angles per rank. `plan GPU` is what `--plan` reports;
`×3` is the realistic figure — see the calibration note below.

| bin | nodes | ranks | lth | `--nchunk` | pinned host/rank | host/node | plan GPU | ~actual GPU |
|-----|-------|-------|-----|-----------|------------------|-----------|----------|-------------|
| 3   | 1     | 8     | 94  | 64        | 9.3 GB           | 75 GB     | 12.9 GB  | ~39 GB      |
| 3   | 2     | 16    | 47  | 32        | 4.7 GB           | 38 GB     | 6.5 GB   | ~19 GB      |
| 2   | 1     | 8     | 188 | 24        | 74.5 GB          | 596 GB    | 19.5 GB  | ~59 GB      |
| 2   | 4     | 32    | 47  | 24        | 18.8 GB          | 150 GB    | 19.5 GB  | ~59 GB      |
| 1   | 4     | 32    | 94  | 6         | 149.5 GB         | 1196 GB   | 20.3 GB  | ~61 GB      |
| 1   | 8     | 64    | 47  | 6         | 75.2 GB          | 602 GB    | 20.3 GB  | ~61 GB      |
| 0   | 32    | 256   | 24  | 1         | 153.4 GB         | 1227 GB   | 17.2 GB  | ~52 GB      |
| 0   | 64    | 512   | 12  | 1         | 78.6 GB          | 629 GB    | 17.2 GB  | ~52 GB      |

Bins 1 and 0 are **host-memory bound**, which is what sets the minimum node
count: bin 1 needs 4 nodes, bin 0 needs 32. Note bin 0 only tolerates
`nchunk = 1`, so it will be inefficient — treat it as a feasibility run, not a
throughput measurement, and prefer bin 1 for scaling studies.

If you have fewer nodes than the table asks for, cut `--ntheta` (host memory is
close to linear in it) rather than trying to squeeze `nchunk`. E.g.
`--bin 1 --ntheta 750 --nchunk 6` on 1 node.

### Three rules the code enforces

1. **`nchunk ≤ lth`.** A chunk larger than the rank's own slab of angles just
   sizes the pool up for nothing.
2. **`lth ≥ 4`.** `rec_mpi` asserts `local_ntheta != 3`. Since `lth ≈
   ntheta/nranks`, that caps useful ranks at `ntheta/4`: 187 at bin 3, 375 at
   bin 2, 750 at bin 1, 1500 at bin 0.
3. **`ndistchunk ≠ lth`.** The chunking layer tells theta-chunked arrays from
   broadcast ones by comparing `shape[0]` to the chunk size, so a distance
   group of exactly `lth` would be misread. With `ndistchunk = 40` and rank
   counts that are multiples of 8 this never triggers for this geometry, but it
   will abort loudly (not silently) if you reach it with a custom `--ntheta`.

### Check before you burn allocation

`--plan` prints the per-rank host and GPU budget and exits without allocating
anything or touching a GPU:

```bash
python test_mosaic.py --bin 1 --nchunk 6 --nranks 32 --plan
```

**Calibration:** the host figure is accurate — measured RSS came in within 4%
of prediction. The GPU figure is a **lower bound**: it counts the chunking pool
and the persistent buffers but not cuFFT work areas or transient kernel
scratch. Across five recorded runs the CuPy pool measured **2.5–3.6× the
prediction**, so budget `plan × 3` and keep headroom. (Those runs predate a
change that removed ~600 MB of per-chunk temporaries, so the ratio should now
be a little better — but plan on 3× until you have a measurement of your own.)

### If it OOMs

* **GPU OOM** → halve `--nchunk` and rerun. It is the only GPU knob; node count
  will not help.
* **Host OOM / pinned allocation failure** → double the node count, or cut
  `--ntheta`.
* Confirm one rank per GPU: `set_affinity_gpu.sh` echoes
  `<rank> uses <dev> of <ngpus> <host>` for every rank at startup, and the log
  header records `gpus_used=` and the rank→device map. Two ranks sharing a
  device roughly halves effective PCIe bandwidth and quietly distorts every
  timing.

### What to report back

The log files (`logmosaic<bin>_<ranks>`). Each already ends with the
`parse_perf_log.py` breakdown for iteration 1: per-function timings and the
grouped categories (`gradient` / `hessian` / `redist` / `linear_batch` / `min`
/ `allreduce`). Also useful: node count, ranks, and the exact command line.

## `parse_perf_log.py`

```bash
python parse_perf_log.py [LOG] [--iter N]
```

Both benchmarks call this on their own log when they finish, so **every log
already ends with its own summary** — for the last iteration, i.e. the one after
the JIT warmup. Running it by hand is only needed for a different `--iter` or
for an older log.

Reads the perf log and prints:

1. **Max process memory and max GPU memory** seen across the whole run
   (parsed from every `@timer`-decorated function entry).
2. **Per-function summary for the requested iter** (default `--iter 1`,
   i.e. the second BH iteration, after JIT warmup). Columns: calls, total
   seconds, mean ms, percentage of iter total.
3. **Grouped categories** — totals for the BH building blocks of iter `N`:

   | category   | members                                                    |
   |------------|------------------------------------------------------------|
   | `gradient` | `gradients_cascade + adj_tomo + fwd_tomo + gradient` (the last being the extra_terms regularizer) |
   | `hessian`  | `hessian_cascade[3] + hessian[3]` — the cascade sweeps plus their extra_terms regularizers |
   | `redist`   | both MPI `redist` calls inside the iter                    |
   | `linear_batch` | `linear_batch + linear_redot_batch`                    |
   | `min`      | the line-search `Rec.min()`                                |
   | `allreduce`    | `allreduce + allreduce2 + allreduce_scalars`           |
   | `other`        | any remaining timed functions (empty on a healthy log) |

   With `fused_hessian=false` the solver runs three separate `Rec.hessian()`
   calls per iter (β-num, β-den, α-den) and the `hessian` row is followed by a
   `#1 / #2 / #3` breakdown, one per call. `fused_hessian=true` (the default)
   derives all three forms from one `hessian_cascade3` sweep, so there is a
   single instance and no breakdown.

   Iter boundaries are detected from `error_debug`'s `iter=N: <t>sec ...` lines,
   so the perf log needs `error_step ≥ 1` (default in `test.py`). The `<t>sec`
   anchor matters: `rec_mpi` also logs `iter=N: coeff_cache hits=…` right after
   that line, and matching those too used to reopen the bucket and leave the
   last iteration empty ("no timer calls recorded").

The same report is available as `report(path, iter_no, out=..., show_info=...,
show_header=...)` — that is what the benchmarks call to append
their own summary.

## What goes into the log

`test.py` writes (rank 0 only, plain text, no ANSI codes):

* **Machine info** — CPU model + cores, total RAM, RAM modules (best-effort
  via `dmidecode`/`/sys/firmware/dmi`, "unavailable" if neither works), GPU
  model + memory + visible count.
* **Job layout** — `ranks`, `hosts`, `gpus_used`; per-host summary; one line
  per rank with `(host, dev)` mapping. The physical device id is derived from
  `CUDA_VISIBLE_DEVICES` so it's correct even when one GPU is pinned per rank.
* **Run config** — the `perf-test: n= nz= nobj= ntheta= ndist= nchunk= …` line.
* **`@timer` lines** — one per call, e.g.
  `gradients_cascade: 0.1234 sec, process memory 1.23 GB, GPU memory 4.56 GB`.
* **`iter=N: …sec err=…`** — emitted by `error_debug` at the end of each
  BH iter, used as iter-boundary markers by the parser.
* **Final summary** — `BH done: niter iters in T s (ms/iter) [nranks, n, nobj, ntheta, ndist, nchunk]`.

## Tuning notes

* **`nchunk`** dominates memory vs throughput. Start at 4 and increase until
  the GPU OOMs.
* **GPU memory** for `Rec` scales with `nobj·max(nobj, nzobj, ntheta)·complex_item`
  per chunk (the formula in `Rec.__init__`); cross-check with the
  `Allocate […]` lines at startup.
* **`niter = 2`** so iter 0 covers the JIT warmup and iter 1 is the steady
  state — that's why `--iter 1` is the default for the parser.
* **Reproducibility**: the synthetic data depends only on global slice /
  theta / distance index, so the same `--n / --ntheta` is bit-identical
  whether you run with 1 rank or 32.
