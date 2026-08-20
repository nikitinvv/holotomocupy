#!/usr/bin/env bash
# Single-tile perf benchmark: BH iterations on a square detector at each of a
# list of sizes.  ndist = 4 propagation distances, one tile.  The mosaic
# counterpart is run_mosaic.sh.
#
#   ./run.sh --plan               # sizes + memory for every n, no GPU, no run
#   ./run.sh                      # NP=4 ranks, sizes 512 / 1024 / 2048
#   NP=8 SIZES="4096" ./run.sh
#   NITER=2 ./run.sh              # add a warmup iteration, report the second
#   NDISTCHUNK=1 ./run.sh         # force the old outer-distance loop everywhere
#
# The big sizes need many ranks -- see "Running on a cluster" in Readme.md for
# the node counts and the per-size nchunk/ndistchunk that fit an 80 GB card.

set -eu
cd "$(dirname "$(readlink -f "$0")")"

PY=${PY:-python}
NP=${NP:-4}
SIZES=${SIZES:-"512 1024 2048"}
# BH iterations.  Iteration 0 absorbs the CuPy JIT compile, so NITER=2 gives a
# clean steady-state number -- worth it except at the sizes where one iteration
# is already expensive.
NITER=${NITER:-1}

# nchunk per size: the chunking pool scales as nchunk * nobj^2 (nobj = 1.59n),
# so it has to come down fast as n grows.  Raise until the GPU OOMs.
nchunk_for () { case "$1" in 512) echo 16;; 1024) echo 8;; 2048) echo 4;; 4096) echo 2;; *) echo 1;; esac; }
# ndistchunk per size: unlike the mosaic geometry, here nzobj == nobj, so the
# pool has no headroom and every extra resident distance costs memory from the
# first one.  At n >= 8192 that is the difference between fitting and not.
ndistchunk_for () { case "$1" in 8192) echo 1;; 4096) echo 2;; *) echo 0;; esac; }
ntheta_for () { echo $(( 900 * $1 / 1024 )); }

if [ "${1:-}" = "--plan" ]; then
    for n in $SIZES; do
        "$PY" test.py --n "$n" --ntheta "$(ntheta_for "$n")" \
                      --nchunk "$(nchunk_for "$n")" \
                      --ndistchunk "${NDISTCHUNK:-$(ndistchunk_for "$n")}" \
                      --nranks "$NP" --plan
    done
    exit 0
fi

for n in $SIZES; do
    nc=$(nchunk_for "$n")
    nt=$(ntheta_for "$n")
    ndc=${NDISTCHUNK:-$(ndistchunk_for "$n")}
    echo "=== n $n  ntheta $nt  nchunk $nc  ndistchunk $ndc  np $NP"
    mpirun -np "$NP" ./set_affinity_gpu.sh \
        "$PY" test.py --n "$n" --ntheta "$nt" --nchunk "$nc" \
                      --ndistchunk "$ndc" --niter "$NITER" --log "log${n}_${nc}"
    "$PY" parse_perf_log.py "log${n}_${nc}" --iter "$(( NITER - 1 ))"
done
