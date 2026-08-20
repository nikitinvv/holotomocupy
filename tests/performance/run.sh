#!/usr/bin/env bash
# Single-tile perf benchmark: BH iterations on a square detector at each of a
# list of sizes.  The mosaic counterpart is run_mosaic.sh.
#
#   ./run.sh                      # NP=4 ranks, sizes 512 / 1024 / 2048
#   NP=8 SIZES="1024 2048" ./run.sh
#   NDISTCHUNK=1 ./run.sh         # reproduce the old outer-distance loop
#
# ntheta is tied to the detector size (ntheta = 900*n/1024) so the three runs
# stay comparable per pixel.
set -eu
cd "$(dirname "$(readlink -f "$0")")"

PY=${PY:-python}
NP=${NP:-4}
SIZES=${SIZES:-"512 1024 2048"}
# 0 = all distances share one upload of a theta chunk of proj (the default);
# 1 = the old outer-distance loop.
NDISTCHUNK=${NDISTCHUNK:-0}

# nchunk per size: the chunking pool scales as nchunk * nobj^2, so it has to
# come down as the detector grows.  Raise until the GPU OOMs.
nchunk_for () { case "$1" in 512) echo 16;; 1024) echo 8;; 2048) echo 4;; *) echo 4;; esac; }
ntheta_for () { echo $(( 900 * $1 / 1024 )); }

for n in $SIZES; do
    nc=$(nchunk_for "$n")
    nt=$(ntheta_for "$n")
    echo "=== n $n  ntheta $nt  nchunk $nc  ndistchunk $NDISTCHUNK  np $NP"
    mpirun -np "$NP" ./set_affinity_gpu.sh \
        "$PY" test.py --n "$n" --ntheta "$nt" --nchunk "$nc" \
                      --ndistchunk "$NDISTCHUNK" --log "log${n}_${nc}"
    "$PY" parse_perf_log.py "log${n}_${nc}" --iter 1
done
