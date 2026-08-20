# Performance benchmark — multi-distance holotomography

Fully-synthetic perf test for `holotomocupy.rec_mpi.Rec` and a log parser
that summarises one BH iteration.

## Files

| file | purpose |
|------|---------|
| `test.py`              | the benchmark — generates phantom, probe, positions, runs `Rec.BH()`, writes a perf log |
| `test_mosaic.py`       | the same benchmark for a MOSAIC scan (default 1 × 5 tiles × 4 distances = 20 distances over one wide object; `--ntile-v 2` gives 40) |
| `run_mosaic.sh`        | drives `test_mosaic.py` over a list of binning levels, then parses each log |
| `run.sh`               | the same for `test.py`, over a list of detector sizes (512 … 8192) |
| `set_affinity_gpu.sh`  | per-rank GPU pinning (`CUDA_VISIBLE_DEVICES = local_rank % ngpus`) |
| `parse_perf_log.py`    | reads the perf log; reports per-iter timing breakdown + max process / GPU memory |

> **Running this on a cluster?** Jump to [Running on a cluster](#running-on-a-cluster)
> — both benchmarks, node counts, `nchunk` / `ndistchunk` per size, and
> the memory budget per rank.

## Quick start

Single rank:
```bash
python test.py --n 2048 --ntheta 1800 --nchunk 4
python parse_perf_log.py perf.log --iter 0
```

Multi-rank with one GPU per local rank (round-robin pinning):
```bash
mpirun -np 8 ./set_affinity_gpu.sh \
    python test.py --n 2048 --ntheta 1800 --nchunk 4 --log log2048_8
python parse_perf_log.py log2048_8 --iter 0
```

Or sweep several detector sizes and parse each log:
```bash
./run.sh --plan                # sizes + memory for each n, no GPU, no run
./run.sh                       # NP=4 ranks, sizes 512 / 1024 / 2048
NP=8 SIZES="1024 2048 4096" ./run.sh
```

## `test.py` CLI

| flag       | default      | meaning |
|------------|--------------|---------|
| `--n`      | **required** | detector size (square: `nz = n`) |
| `--ntheta` | **required** | number of projection angles |
| `--nchunk` | `4`          | theta chunk size for the batched GPU ops — main perf knob |
| `--ndistchunk` | `0` (= all) | distances sharing one upload of a theta chunk of `proj`; `1` restores the old outer-distance loop |
| `--plan`   | off          | print the per-rank host/GPU budget and exit without allocating |
| `--nranks` | actual size  | rank count to assume in `--plan` |
| `--log`    | `perf.log`   | output log path on rank 0 (`''` disables file logging) |

Other knobs (object padding, `ndist`, `niter`, dtype, regularization, …)
are hardcoded at the top of `test.py` — edit there. Defaults are:
`ndist = 4`, `niter = 1`, `obj_dtype = complex64`,
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

so the default 1 × 5 tiles × 4 distances give `ndist = 20` over one wide object
grid, and each tile's place on the mosaic lives in `vars['pos']`.
`--ntile-v 2` gives the two-row mosaic, `ndist = 40`; both shapes are worth
benchmarking and `run_mosaic.sh` takes `TILES=1x5` / `TILES=2x5`.

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
always holds the whole mosaic. At bin 0 that is `4096 × 12288 × 12288` for the
default 1 × 5 layout and `6144 × 12288 × 12288` for 2 × 5 (a second tile row
makes the object taller, not wider). Acquisition geometry, tile pitch, jitter,
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

| bin | detector | angles | object (1×5) | host total | object (2×5) | host total |
|-----|----------|--------|--------------|------------|--------------|------------|
| 3   | 256      | 750    | 512 × 1536²   | 48 GB  | 768 × 1536²   | 74 GB  |
| 2   | 512      | 1500   | 1024 × 3072²  | 386 GB | 1536 × 3072²  | 594 GB |
| 1   | 1024     | 3000   | 2048 × 6144²  | 3.0 TB | 3072 × 6144²  | 4.6 TB |
| 0   | 2048     | 6000   | 4096 × 12288² | 24 TB  | 6144 × 12288² | 37 TB  |

`host total` is summed over all ranks, so it divides by the rank count — that
is what sets the minimum node count. GPU per rank does *not* divide; at
`nchunk = 4` it is 0.8 / 3.3 / 13.2 / 52.7 GB (1×5) and 0.9 / 3.5 / 13.9 /
55.7 GB (2×5), which is why bin 0 has to drop to `nchunk = 1`.

The GPU side is comfortable by comparison up to bin 1, so the rank count is set
by RAM, not by the devices. See
[Running on a cluster](#running-on-a-cluster) for concrete node counts.

### Running

```bash
NP=8 BINS="3 2" ./run_mosaic.sh          # run + parse each bin
NP=8 BINS="3 2 1" ./run_mosaic.sh --plan # sizes only
```

or directly:

```bash
mpirun -np 8 ./set_affinity_gpu.sh \
    python test_mosaic.py --bin 2 --nchunk 8 --log logmosaic2_8
python parse_perf_log.py logmosaic2_8 --iter 0
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
| `--ntile-v`    | `1`   | tile rows |
| `--ndist-tile` | `4`   | distances per tile (max 4) |
| `--niter`      | `1`   | BH iterations; with 2+ the last one excludes the JIT compile |
| `--ntheta`     | auto  | override the `6000>>bin` angle count |
| `--nobj` / `--nzobj` | auto | override the object grid (binned px) |
| `--plan`       | off   | print sizes + memory and exit |
| `--nranks`     | MPI size | rank count to assume in `--plan` |
| `--log`        | `perf_mosaic.log` | log path on rank 0 |

## Running on a cluster

Written for a node with **8 GPUs (H100, 80 GB), 2 TB of host RAM**. Everything
below assumes **one MPI rank per GPU**, i.e. 8 ranks per node.

There are **two benchmarks and both are wanted**. They are independent; run
them in either order.

| | script | driver | geometry | what it stresses |
|---|---|---|---|---|
| **A. Mosaic** | `test_mosaic.py` | `run_mosaic.sh` | a row (or two rows) of tiles at 4 distances each, flattened onto one distance axis over a single wide object | the workload the package is being optimized for; wide object, many distances |
| **B. Single tile** | `test.py` | `run.sh` | one tile, 4 distances, square object, swept over detector sizes 512 → 8192 | how the solver scales with raw detector size |

Both run `Rec.BH()` — the Bilinear-Hessian conjugate-gradient solver — on
synthetic data and time every internal stage. Nothing is read from disk and no
result is written; probe, positions and data are generated in memory from a
fixed seed, so a run is reproducible and depends only on the size knobs. The
point is the timing breakdown, not the reconstruction.

### Setup

```bash
pip install -e .                      # from the repo root
```
Needs `cupy` (matching the CUDA toolkit), `mpi4py` built against the site MPI,
plus `numpy`, `h5py`, `psutil`, `nvtx`, `tifffile`, `matplotlib`, `pandas`.
The CUDA kernels ship as source and are JIT-compiled by CuPy on first use —
iteration 0 therefore absorbs that compile. Both scripts default to
`--niter 1`, so the one reported iteration *includes* the JIT cost. Pass
`NITER=2` to either driver (or `--niter 2` directly) to get a clean
steady-state number; the drivers then parse the last iteration automatically.
At the largest sizes one iteration is already expensive, so `NITER=1` there and
note it in the report.

### Launching

`set_affinity_gpu.sh` pins each rank to one GPU via
`CUDA_VISIBLE_DEVICES = local_rank % ngpus`. It reads either the Open MPI or
the SLURM rank variables, so both launchers work:

```bash
# SLURM
srun -N 4 --ntasks-per-node 8 --gpus-per-task 1 \
     ./set_affinity_gpu.sh python test_mosaic.py --bin 1 --nchunk 7 --log logmosaic1_32

# Open MPI
mpirun -np 32 --map-by ppr:8:node ./set_affinity_gpu.sh \
     python test_mosaic.py --bin 1 --nchunk 7 --log logmosaic1_32
```

Each run writes its log on rank 0 and **appends its own parsed summary to the
end of that log**, so the log file alone is the deliverable — send those back.

### The two knobs

|  | drives | scales with |
|---|---|---|
| `--nchunk` | **GPU** memory | `nchunk × nzobj × nobj` — the *global* object grid |
| `--ndistchunk` | how many distances share one upload of a `proj` chunk | detector plane, `ndistchunk × nz × n` |

The important and slightly counter-intuitive part: **adding nodes buys host
memory, not GPU memory.** Host arrays are split over ranks, so pinned RAM per
rank falls as `1/nranks`. The GPU chunking pool is sized from `nzobj` and
`nobj`, which are global and do not shrink. So `nchunk` is essentially a
per-size constant — if you add nodes to fit a bigger problem in RAM, do *not*
expect to raise `nchunk` too.

**`--ndistchunk` matters for benchmark B and not for benchmark A**, and the
reason is worth stating because it is not obvious. The chunking pool holds
`max(3 × proj_chunk + ndistchunk × dist_chunk, 3 × obj_chunk)`. Distances only
add to the *first* term, so as long as that term stays under `3 × obj_chunk`
the distances are free. Setting the two equal gives the break-even:

```
ndist* = 6 · nobj · (nobj − nzobj) / (nz · n)          (independent of nchunk and of bin)
```

* **Mosaic**: the object is wider than it is tall (`nzobj < nobj`), so there is
  real headroom — `ndist*` is **144** at 1×5 and **108** at 2×5, against an
  actual `ndist` of 20 and 40. Nothing to tune: **leave `--ndistchunk` at 0.**
* **Single tile**: `nzobj == nobj`, so `ndist* = 0` — there is no headroom at
  all and every resident distance costs from the first one. At small `n` that
  is irrelevant, but at `n = 8192` dropping `--ndistchunk` from 4 to 2 is the
  difference between ~78 GB (does not fit) and ~66 GB (fits).

`--ndistchunk 1` restores the old outer-distance loop. It is 3–4× slower on the
cascades and exists to reproduce the pre-optimization numbers — use it for an
A/B, or as a last-resort memory lever at the largest single-tile size.

### Recommended settings — A. Mosaic

`--bin` sets the size: `n = nz = 2048>>bin` detector pixels and
`ntheta = 6000>>bin` angles. **bin 0 is the full production problem**; bin 3 is
1/8 of it in every dimension and runs on a single node. `lth` is
`local_ntheta` = angles per rank. `plan GPU` is what `--plan` reports; `~actual`
is the realistic figure — see the calibration note below.

Two mosaic shapes are wanted. `run_mosaic.sh` defaults to 1×5; pass
`TILES=2x5` for the other.

```bash
TILES=1x5 NP=32 BINS="3 2 1" ./run_mosaic.sh
TILES=2x5 NP=32 BINS="3 2 1" ./run_mosaic.sh
```

Two rows per bin: the **minimum** node count and one comfortable step up.

#### `TILES=1x5` — ndist = 20, object 4096 x 12288 x 12288 at bin 0  (ndistchunk crossover 144)
| bin | nodes | ranks | lth | `--nchunk` | `--ndistchunk` | host/rank | host/node | plan GPU | ~actual GPU |
|-----|-------|-------|-----|-----------|----------------|-----------|-----------|----------|-------------|
| 3 | 1 | 8 | 94 | 23 | 20 | 6.1 GB | 49 GB | 4.6 GB | ~14 GB |
|   | 2 | 16 | 47 | 11 | 20 | 3.0 GB | 24 GB | 2.2 GB | ~7 GB |
| 2 | 1 | 8 | 188 | 30 | 20 | 48.4 GB | 387 GB | 23.8 GB | ~71 GB |
|   | 2 | 16 | 94 | 23 | 20 | 24.3 GB | 194 GB | 18.3 GB | ~55 GB |
| 1 | 4 | 32 | 94 | 7 | 20 | 97.0 GB | 776 GB | 22.7 GB | ~68 GB |
|   | 8 | 64 | 47 | 7 | 20 | 48.8 GB | 390 GB | 22.7 GB | ~68 GB |
| 0 | 32 | 256 | 24 | 1 | 20 | 99.2 GB | 793 GB | 14.8 GB | ~44 GB |
|   | 64 | 512 | 12 | 1 | 20 | 50.5 GB | 404 GB | 14.8 GB | ~44 GB |

#### `TILES=2x5` — ndist = 40, object 6144 x 12288 x 12288 at bin 0  (ndistchunk crossover 108)
| bin | nodes | ranks | lth | `--nchunk` | `--ndistchunk` | host/rank | host/node | plan GPU | ~actual GPU |
|-----|-------|-------|-----|-----------|----------------|-----------|-----------|----------|-------------|
| 3 | 1 | 8 | 94 | 23 | 40 | 9.3 GB | 75 GB | 4.7 GB | ~14 GB |
|   | 2 | 16 | 47 | 11 | 40 | 4.7 GB | 38 GB | 2.3 GB | ~7 GB |
| 2 | 1 | 8 | 188 | 29 | 40 | 74.5 GB | 596 GB | 23.5 GB | ~71 GB |
|   | 2 | 16 | 94 | 23 | 40 | 37.4 GB | 299 GB | 18.7 GB | ~56 GB |
| 1 | 4 | 32 | 94 | 7 | 40 | 149.5 GB | 1196 GB | 23.5 GB | ~71 GB |
|   | 8 | 64 | 47 | 7 | 40 | 75.2 GB | 602 GB | 23.5 GB | ~71 GB |
| 0 | 32 | 256 | 24 | 1 | 40 | 153.4 GB | 1227 GB | 17.2 GB | ~52 GB |
|   | 64 | 512 | 12 | 1 | 40 | 78.6 GB | 629 GB | 17.2 GB | ~52 GB |

Bins 1 and 0 are **host-memory bound**, which is what sets the minimum node
count. Bin 0 only tolerates `nchunk = 1`, so it will be inefficient — treat it
as a feasibility run, not a throughput measurement, and prefer bin 1 for
scaling studies. If you have fewer nodes than the table asks for, cut
`--ntheta` (host memory is close to linear in it) rather than squeezing
`nchunk`: e.g. `--bin 1 --ntheta 750 --nchunk 7` on 1 node.

### Recommended settings — B. Single tile

`n` is the detector size; the object is `nobj = nzobj = 1.59 n` cubed and
`ntheta = 900 n / 1024`, both set by `run.sh`. `ndist` is always 4.

```bash
NP=8 SIZES="512 1024 2048" ./run.sh          # 1 node
NP=64 SIZES="4096" ./run.sh                  # 8 nodes
```

| n | nobj | ntheta | nodes | ranks | lth | `--nchunk` | `--ndistchunk` | host/rank | host/node | plan GPU | ~actual GPU |
|---|------|--------|-------|-------|-----|-----------|----------------|-----------|-----------|----------|-------------|
| 512 | 816 | 450 | 1 | 8 | 57 | 14 | 4 | 2.9 GB | 23 GB | 1.1 GB | ~3 GB |
|   |   |   | 2 | 16 | 29 | 7 | 4 | 1.5 GB | 12 GB | 0.6 GB | ~2 GB |
| 1024 | 1632 | 900 | 1 | 8 | 113 | 28 | 4 | 23.0 GB | 184 GB | 8.5 GB | ~25 GB |
|   |   |   | 2 | 16 | 57 | 14 | 4 | 11.6 GB | 93 GB | 4.3 GB | ~13 GB |
| 2048 | 3264 | 1800 | 1 | 8 | 225 | 19 | 4 | 183.0 GB | 1464 GB | 23.2 GB | ~70 GB |
|   |   |   | 2 | 16 | 113 | 19 | 4 | 91.9 GB | 735 GB | 23.2 GB | ~70 GB |
| 4096 | 6528 | 3600 | 8 | 64 | 57 | 4 | 4 | 185.1 GB | 1480 GB | 20.9 GB | ~63 GB |
|   |   |   | 16 | 128 | 29 | 4 | 4 | 93.9 GB | 751 GB | 20.9 GB | ~63 GB |
| 8192 | 13056 | 7200 | 128 | 1024 | 8 | 1 | 2 | 103.1 GB | 825 GB | 22.1 GB | ~66 GB |
|   |   |   | 256 | 2048 | 4 | 1 | 2 | 56.8 GB | 455 GB | 22.1 GB | ~66 GB |

Notes on this sweep:

* 512 / 1024 / 2048 all fit on **one node**. 2048 is right at the host limit at
  1 node (1464 GB of 2 TB) — use 2 nodes if the allocation allows.
* **4096 needs 8 nodes**, host-bound.
* **8192 is the hard one** and is GPU-bound, not host-bound: even at
  `nchunk = 1` the object-plane buffers (`nobj = 13056`) dominate. It needs
  `--ndistchunk 2` to fit an 80 GB card at all. 64 nodes also works with
  `--ndistchunk 1` at 1563 GB/node, which is uncomfortably close to 2 TB. If
  8192 is out of reach for your allocation, **run it with a reduced
  `--ntheta`** and say so in the report — the timing breakdown is still
  meaningful, the absolute numbers just are not comparable to the others.

### Three rules the code enforces

1. **`nchunk ≤ lth`.** A chunk larger than the rank's own slab of angles just
   sizes the pool up for nothing. (The tables above target roughly `lth/4` so
   there are ≥ 2 chunks in flight for the H2D/compute/D2H overlap; the memory
   maximum is higher.)
2. **`lth ≥ 4`.** `rec_mpi` asserts `local_ntheta != 3`. Since
   `lth ≈ ntheta/nranks`, that caps useful ranks at `ntheta/4`: 187 at bin 3,
   375 at bin 2, 750 at bin 1, 1500 at bin 0 for the mosaic; 112 / 225 / 450 /
   900 / 1800 for single-tile n = 512 … 8192.
3. **`ndistchunk ≠ lth`.** The chunking layer tells theta-chunked arrays from
   broadcast ones by comparing `shape[0]` to the chunk size, so a distance
   group of exactly `lth` would be misread. It aborts loudly, not silently, if
   you hit it with a custom `--ntheta` or rank count.

### Check before you burn allocation

`--plan` prints the per-rank host and GPU budget and exits without allocating
anything. Both scripts have it:

```bash
python test_mosaic.py --bin 1 --nchunk 7 --nranks 32 --plan
python test.py --n 8192 --ntheta 7200 --nchunk 1 --ndistchunk 2 --nranks 1024 --plan
./run_mosaic.sh --plan          # every bin
./run.sh --plan                 # every size
```

**Calibration:** the host figure is accurate — measured RSS came in within 4%
of prediction. The GPU figure is a **lower bound**: it counts the chunking pool
and the persistent buffers but not cuFFT work areas or transient kernel
scratch. Across five recorded runs the CuPy pool measured **2.5–3.6× the
prediction**, so budget `plan × 3` and keep headroom — that is what the
`~actual GPU` column is. (Those runs predate a change that removed ~600 MB of
per-chunk temporaries, so the ratio should now be a little better, but plan on
3× until you have a measurement of your own.)

### If it OOMs

* **GPU OOM** → halve `--nchunk` and rerun; if already at 1, lower
  `--ndistchunk` (single tile only — on the mosaic it will not help, see
  above). Node count does not help GPU memory.
* **Host OOM / pinned allocation failure** → double the node count, or cut
  `--ntheta`.
* Confirm one rank per GPU: `set_affinity_gpu.sh` echoes
  `<rank> uses <dev> of <ngpus> <host>` for every rank at startup, and the log
  header records `gpus_used=` and the rank→device map. Two ranks sharing a
  device roughly halves effective PCIe bandwidth and quietly distorts every
  timing.

### What to report back

The log files — `logmosaic<VxH>_<bin>_<ranks>` from benchmark A and
`log<n>_<nchunk>` from benchmark B. Each already ends with the
`parse_perf_log.py` breakdown: per-function timings and the grouped categories
(`gradient` / `hessian` / `redist` / `linear_batch` / `min` / `allreduce`).
Also useful: node count, ranks, the exact command line, and anything you had to
change to make it fit.

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
* **`niter`** defaults to 1 in both scripts, so the single reported iteration
  still contains the CuPy JIT compile. Use `--niter 2` (or `NITER=2` in the
  drivers) when you want the steady state; the parser's own default is
  `--iter 1`, which matches that two-iteration run.
* **Reproducibility**: the synthetic data depends only on global slice /
  theta / distance index, so the same `--n / --ntheta` is bit-identical
  whether you run with 1 rank or 32.
