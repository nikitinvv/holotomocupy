# Performance benchmark — multi-distance holotomography

Fully-synthetic perf test for `holotomocupy.rec_mpi.Rec` and a log parser
that summarises one BH iteration.

## Files

| file | purpose |
|------|---------|
| `test.py`              | the benchmark — generates phantom, probe, positions, runs `Rec.BH()`, writes a perf log |
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

## `parse_perf_log.py`

```bash
python parse_perf_log.py [LOG] [--iter N]
```

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

   Iter boundaries are detected from `error_debug`'s `iter=N: ...` lines,
   so the perf log needs `error_step ≥ 1` (default in `test.py`).

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
