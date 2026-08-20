#!/usr/bin/env bash
# Mosaic perf benchmark: one BH iteration of a TILES x 4-distance scan at each
# binning level.  Two mosaic shapes are interesting:
#
#   1x5 tiles -> ndist = 20   (the default: a single row, wide flat object)
#   2x5 tiles -> ndist = 40   (two rows, 1.5x taller object)
#
#   ./run_mosaic.sh --plan            # sizes + memory for every bin, no GPU
#   ./run_mosaic.sh                   # NP=4 ranks, bins 3 / 2 / 1 / 0, 1x5
#   TILES=2x5 ./run_mosaic.sh         # the two-row mosaic
#   NP=8 BINS="3 2" ./run_mosaic.sh
#   NITER=2 ./run_mosaic.sh           # add a warmup iteration, report the second
#   NDISTCHUNK=1 ./run_mosaic.sh      # reproduce the old outer-distance loop

set -eu
cd "$(dirname "$(readlink -f "$0")")"

PY=${PY:-python}
NP=${NP:-4}
BINS=${BINS:-"3 2 1 0"}
# BH iterations.  Iteration 0 absorbs the CuPy JIT compile, so NITER=2 gives a
# clean steady-state number.
NITER=${NITER:-1}
TILES=${TILES:-1x5}
NTILE_V=${TILES%x*}
NTILE_H=${TILES#*x}
# 0 = all ndist distances share one upload of a theta chunk of proj (the
# default); 1 = the old outer-distance loop.  For these mosaic shapes the
# chunking pool has enough object-plane headroom that all of them are free
# (the crossover is 144 distances at 1x5, 108 at 2x5), so leave it at 0.
NDISTCHUNK=${NDISTCHUNK:-0}

# nchunk per bin: the chunking pool scales as nchunk * nzobj * nobj, so it has
# to come down as the object grows.  Raise until the GPU OOMs.
nchunk_for () { case "$1" in 3) echo 16;; 2) echo 8;; 1) echo 2;; *) echo 1;; esac; }

common_args () {
    echo --ntile-v "$NTILE_V" --ntile-h "$NTILE_H" --ndist-tile 4 \
         --nchunk "$(nchunk_for "$1")" --ndistchunk "$NDISTCHUNK" --niter "$NITER"
}

if [ "${1:-}" = "--plan" ]; then
    for b in $BINS; do
        "$PY" test_mosaic.py --bin "$b" $(common_args "$b") --nranks "$NP" --plan
    done
    exit 0
fi

for b in $BINS; do
    nc=$(nchunk_for "$b")
    echo "=== bin $b  tiles ${NTILE_V}x${NTILE_H}  nchunk $nc  ndistchunk $NDISTCHUNK  np $NP"
    mpirun -np "$NP" ./set_affinity_gpu.sh \
        "$PY" test_mosaic.py --bin "$b" $(common_args "$b") \
                             --log "logmosaic${NTILE_V}x${NTILE_H}_${b}_${NP}"
    "$PY" parse_perf_log.py "logmosaic${NTILE_V}x${NTILE_H}_${b}_${NP}" --iter "$(( NITER - 1 ))"
done
