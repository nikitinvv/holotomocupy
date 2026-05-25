#!/usr/bin/env python
"""
Parse the log produced by rec_iterative_mpi_syn.py and print:
  * max process memory + max GPU memory across the whole run
  * per-function timing breakdown for the requested BH iter (default: iter 1)
  * grouped category totals (redist / gradient / hessian / linear_batch / allreduce)

`@timer`-decorated methods log a line of the form
    [rank=0] FUNCNAME: 0.1234 sec, process memory 1.23 GB, GPU memory 4.56 GB
and `error_debug` emits `[rank=0] iter=N: ...sec err=...` (or `Initial err=...`
for the pre-loop warmup). Timer lines emitted between consecutive iter= markers
belong to the iter that closes the bucket.
"""

import argparse
import re
import sys
from collections import defaultdict, Counter


# ── regex patterns ──────────────────────────────────────────────────────────
ANSI = re.compile(r'\033\[[0-9;]*m')
TIMER_RE = re.compile(
    r'\[rank=\d+\]\s+(\w+):\s+([\d.]+)\s+sec,\s+process\s+memory\s+([\d.]+)\s+GB,'
    r'\s+GPU\s+memory\s+([\d.]+)\s+GB'
)
ITER_RE = re.compile(r'\[rank=\d+\]\s+iter=(-?\d+):')
INIT_RE = re.compile(r'\[rank=\d+\]\s+Initial\s+err=')
INFO_RE = re.compile(r'\[rank=\d+\]\s+(machine|job|perf-test|BH done):\s*(.+)')


# ── category mapping ────────────────────────────────────────────────────────
# Each iter's @timer calls are bucketed by function name into these groups.
# The 'hessian' bucket is split per-instance (hessian1, hessian2, ...) at print
# time using split_hessians(); see the categories-rendering loop below.
CATEGORIES = {
    'gradient':     {'gradients_cascade', 'gF4', 'fwd_tomo'},
    'hessian':      {'hessian_cascade', 'hessian_laplacian', 'hessian_prbfit'},
    'redist':       {'redist'},
    'linear_batch': {'linear_batch', 'linear_redot_batch'},
    'allreduce':    {'allreduce', 'allreduce2'},
}


def parse(path):
    """Walk the log; return (max_proc_gb, max_gpu_gb, iters_dict, info_lines).

    iters_dict[N] = list of (funcname, elapsed_sec, proc_mem_gb, gpu_mem_gb)
    where N is the BH iter index (-1 for the pre-loop warmup).
    info_lines is a list of (label, text) tuples for machine/job/perf-test
    metadata, preserved in log order.
    """
    pending = []                       # accumulates calls until next iter= marker
    iters = {}
    info_lines = []
    max_proc = max_gpu = 0.0
    with open(path) as f:
        for raw in f:
            line = ANSI.sub('', raw)
            m = INFO_RE.search(line)
            if m:
                info_lines.append((m.group(1), m.group(2).rstrip()))
                # don't 'continue' — INFO lines never match the patterns below
            m = ITER_RE.search(line)
            if m:
                iters[int(m.group(1))] = pending
                pending = []
                continue
            if INIT_RE.search(line):
                iters[-1] = pending
                pending = []
                continue
            m = TIMER_RE.search(line)
            if m:
                fname = m.group(1)
                t = float(m.group(2)); p = float(m.group(3)); g = float(m.group(4))
                pending.append((fname, t, p, g))
                if p > max_proc: max_proc = p
                if g > max_gpu:  max_gpu  = g
    # Any pending lines after the last iter= belong to no closed bucket — drop
    return max_proc, max_gpu, iters, info_lines


def summarise_iter(calls):
    """Aggregate a list of (fname, t, p, g) tuples by function name."""
    by_fn = defaultdict(lambda: [0, 0.0])       # fn -> [count, total_sec]
    for fn, t, _, _ in calls:
        by_fn[fn][0] += 1
        by_fn[fn][1] += t
    return by_fn


# Inside one BH iter `Rec.hessian()` always emits a hessian_cascade first, then
# (optionally) hessian_prbfit, then hessian_laplacian — in that order, with no
# other timed calls between them. Splitting by hessian_cascade boundaries gives
# the individual hessian-call instances (typically 3 per iter: β-num, β-den, α).
_HESSIAN_INNER = {'hessian_cascade', 'hessian_laplacian', 'hessian_prbfit'}

def split_hessians(calls):
    instances = []
    current = None
    for fn, t, *_ in calls:
        if fn == 'hessian_cascade':
            current = []
            instances.append(current)
        if current is not None and fn in _HESSIAN_INNER:
            current.append((fn, t))
        elif fn not in _HESSIAN_INNER:
            current = None
    return instances


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('log', nargs='?', default='perf.log',
                    help='path to the perf log file (default: perf.log)')
    ap.add_argument('--iter', type=int, default=1,
                    help='which BH iter to break down (default: 1 — second iter, post-JIT warmup)')
    args = ap.parse_args()

    try:
        max_proc, max_gpu, iters, info_lines = parse(args.log)
    except FileNotFoundError:
        print(f"log not found: {args.log}", file=sys.stderr); sys.exit(1)

    print(f"=== {args.log} ===")
    if info_lines:
        print("── machine / job ───────────────────────────────────────")
        for label, text in info_lines:
            print(f"{label}: {text}")
        print()
    print(f"max process memory: {max_proc:.3f} GB")
    print(f"max GPU memory    : {max_gpu:.3f} GB")
    print(f"iters seen        : {sorted(iters)}")

    if args.iter not in iters:
        print(f"\niter={args.iter} not in log (available: {sorted(iters)})", file=sys.stderr)
        sys.exit(2)

    calls = iters[args.iter]
    if not calls:
        print(f"\niter={args.iter}: no timer calls recorded "
              f"(was log_level=DEBUG when the run started?)")
        sys.exit(3)

    by_fn = summarise_iter(calls)
    total = sum(t for _, t in by_fn.values())

    print(f"\n── iter {args.iter}: per-function summary ────────────────────────")
    print(f"{'function':<24} {'calls':>6} {'total[s]':>12} {'mean[ms]':>12} {'pct':>7}")
    for fn in sorted(by_fn, key=lambda k: -by_fn[k][1]):
        cnt, tot = by_fn[fn]
        mean_ms  = (tot / cnt) * 1e3
        pct      = 100 * tot / total if total else 0
        print(f"{fn:<24} {cnt:>6} {tot:>12.4f} {mean_ms:>12.3f} {pct:>6.1f}%")
    print(f"{'TOTAL (timed)':<24} {sum(c for c, _ in by_fn.values()):>6} {total:>12.4f}")

    print(f"\n── iter {args.iter}: grouped categories ──────────────────────────")
    seen = set()
    hessian_instances = split_hessians(calls)
    print(f"{'category':<16} {'calls':>6} {'total[s]':>12} {'pct':>7}  members")
    for cat, members in CATEGORIES.items():
        if cat == 'hessian':
            # split into hessian1, hessian2, … (one per BH hessian() invocation)
            for k, inst in enumerate(hessian_instances, start=1):
                cnt = len(inst)
                tot = sum(t for _, t in inst)
                pct = 100 * tot / total if total else 0
                present = sorted({fn for fn, _ in inst})
                print(f"{'hessian' + str(k):<16} {cnt:>6} {tot:>12.4f} {pct:>6.1f}%  {present}")
            seen.update(members & set(by_fn))
            continue
        cnt = sum(c for fn, (c, _) in by_fn.items() if fn in members)
        tot = sum(t for fn, (_, t) in by_fn.items() if fn in members)
        pct = 100 * tot / total if total else 0
        present = sorted(members & set(by_fn))
        seen.update(present)
        print(f"{cat:<16} {cnt:>6} {tot:>12.4f} {pct:>6.1f}%  {present}")
    other_fns = sorted(set(by_fn) - seen)
    if other_fns:
        cnt = sum(by_fn[fn][0] for fn in other_fns)
        tot = sum(by_fn[fn][1] for fn in other_fns)
        pct = 100 * tot / total if total else 0
        print(f"{'other':<16} {cnt:>6} {tot:>12.4f} {pct:>6.1f}%  {other_fns}")


if __name__ == '__main__':
    main()
