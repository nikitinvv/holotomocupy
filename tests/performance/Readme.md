# Performance benchmark — multi-distance holotomography

Fully-synthetic perf test for `holotomocupy.rec_mpi.Rec` and a log parser
that summarises one BH iteration.

## Files

| file | purpose |
|------|---------|
| `test.py`              | the benchmark — generates phantom, probe, positions, runs `Rec.BH()`, writes a perf log |
| `test_mosaic.py`       | the same benchmark for a MOSAIC scan (2 × 5 tiles × 4 distances = 40 distances over one wide object) |
| `run_mosaic.sh`        | drives `test_mosaic.py` over a list of binning levels, then parses each log |
| `peak_mem.py`          | background sampler for peak GPU / host memory, used by both benchmarks |
| `set_affinity_gpu.sh`  | per-rank GPU pinning (`CUDA_VISIBLE_DEVICES = local_rank % ngpus`) |
| `parse_perf_log.py`    | reads the perf log; reports per-iter timing breakdown + max process / GPU memory |
| `log512`               | example log from a `--n 512 --ntheta 450 --nchunk 4` run on 8 ranks × 4 GPUs |

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

## `test.py` CLI

| flag       | default      | meaning |
|------------|--------------|---------|
| `--n`      | **required** | detector size (square: `nz = n`) |
| `--ntheta` | **required** | number of projection angles |
| `--nchunk` | `4`          | theta chunk size for the batched GPU ops — main perf knob |
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

### Peak memory

`@timer` only reports what it happens to see at the end of each timed call, so
transient peaks (cuFFT work areas, the scratch inside `Shift`/`Tomo`) and the
whole allocation phase are invisible to it. `peak_mem.py` polls instead, on a
background thread at 20 Hz, and the run ends with the worst rank's maxima:

```
perf-test: peak GPU during BH  6.41 GB in use of 47.27 GB  (cupy pool 5.98 GB, baseline before the run 1.33 GB)
perf-test: peak GPU during setup 2.10 GB in use  (cupy pool 1.74 GB)
perf-test: peak host RSS  setup 52.1 GB, BH 52.4 GB  (worst rank; predicted pinned 50.9 GB)
```

`in use` comes from `cudaMemGetInfo` and is device-wide — on a shared GPU it
counts other processes too, which is what `baseline` (usage before the sampler
started) is there to show. `cupy pool` is everything CuPy took from the driver,
free-list included. Compare either against the `--plan` prediction.

### `test_mosaic.py` CLI

| flag | default | meaning |
|------|---------|---------|
| `--bin`        | **required** | `n = nz = 2048>>bin`, `ntheta = 6000>>bin` |
| `--nchunk`     | `4`   | theta chunk size — main perf knob |
| `--ntile-h`    | `5`   | tiles per row |
| `--ntile-v`    | `2`   | tile rows |
| `--ndist-tile` | `4`   | distances per tile (max 4) |
| `--niter`      | `2`   | BH iterations (iter 0 is warmup) |
| `--ntheta`     | auto  | override the `6000>>bin` angle count |
| `--nobj` / `--nzobj` | auto | override the object grid (binned px) |
| `--plan`       | off   | print sizes + memory and exit |
| `--nranks`     | MPI size | rank count to assume in `--plan` |
| `--log`        | `perf_mosaic.log` | log path on rank 0 |

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
   | `gradient` | `gradients_cascade + adj_tomo + fwd_tomo`                  |
   | `hessian1`, `hessian2`, `hessian3` | the 3 `Rec.hessian()` calls per iter (β-num, β-den, α-den), each split into `hessian_cascade + hessian_laplacian [+ hessian_prbfit]` |
   | `redist`   | both MPI `redist` calls inside the iter                    |
   | `linear_batch` | `linear_batch + linear_redot_batch`                    |
   | `allreduce`    | `allreduce + allreduce2`                               |
   | `other`        | any remaining timed functions                          |

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
