#!/usr/bin/env bash
# Mosaic perf benchmark: one BH iteration of a 2 x 5 tile x 4 distance scan
# (ndist = 40) at each binning level.
#
#   ./run_mosaic.sh --plan        # sizes + memory for every bin, no GPU, no run
#   ./run_mosaic.sh               # NP=4 ranks, bins 3 and 2
#   NP=8 BINS="3 2 1" ./run_mosaic.sh
#
# Host pinned memory is what limits this test, not the GPU -- see --plan before
# launching.  Rough totals over ALL ranks (obj+proj+proj_tmp+data):
#     bin 3    74 GB     bin 2   594 GB     bin 1   4.6 TB     bin 0   37 TB
# so bin 1 and bin 0 need a multi-node job sized from the --plan output.
set -eu
cd "$(dirname "$(readlink -f "$0")")"

PY=${PY:-python}
NP=${NP:-4}
BINS=${BINS:-"3"}

# nchunk per bin: the chunking pool scales as nchunk * nzobj * nobj, so it has
# to come down as the object grows.  Raise until the GPU OOMs.
nchunk_for () { case "$1" in 3) echo 16;; 2) echo 8;; 1) echo 4;; *) echo 2;; esac; }

if [ "${1:-}" = "--plan" ]; then
    for b in $BINS; do
        "$PY" test_mosaic.py --bin "$b" --nchunk "$(nchunk_for "$b")" --plan
    done
    exit 0
fi

for b in $BINS; do
    nc=$(nchunk_for "$b")
    echo "=== bin $b  nchunk $nc  np $NP"
    mpirun -np "$NP" ./set_affinity_gpu.sh \
        "$PY" test_mosaic.py --bin "$b" --nchunk "$nc" --log "logmosaic${b}_${NP}"
    "$PY" parse_perf_log.py "logmosaic${b}_${NP}" --iter 1
done
