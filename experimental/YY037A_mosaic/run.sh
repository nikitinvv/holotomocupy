#!/usr/bin/env bash
#
# TEMPORARY: tile placement-error sweep.
#
# For every tile, nudge it horizontally by -10..+10 object px (finest grid) on
# top of whatever step 5 measured, and keep the middle FBP slice. The tile whose
# error is best centred at 0 is correctly placed; a lopsided series means the
# measured offset for that tile is off by the amount of the minimum.
#
#   ./run.sh                         # all tiles, -10..10 step 1  (105 runs)
#   ./run.sh center                  # one tile
#   ./run.sh center left             # a couple of tiles
#   ERR_LO=-4 ERR_HI=4 ./run.sh      # narrower range
#   NP=4 ./run.sh                    # fewer ranks
#
# Output: $OUT/fbp_{tile}_{error+100}.tiff, plus a log per run in $OUT/logs.
# Runs whose tiff already exists are skipped, so an interrupted sweep can just
# be restarted.
#
# Meant for a single-bin config (start_level_rec == nlevels-1): `offsets`
# carries across bins, so on a multi-bin run the error would both accumulate
# and be measured away again by the next bin's estimate_overlap.

set -u

cd "$(dirname "$(readlink -f "$0")")" || exit 1

CONF=${CONF:-config_steps15.conf}
NP=${NP:-8}
PY=${PY:-/home/beams2/VNIKITIN/miniforge3/envs/holotomocupy/bin/python}
MPIRUN=/local/vnikitin/hpc_sdk_multi/Linux_x86_64/26.3/comm_libs/13.1/hpcx/hpcx-2.25.1/ompi/bin/mpirun
OUT=${HOLO_TILE_ERR_DIR:-/data2/vnikitin/tmp}

ERR_LO=${ERR_LO:--10}
ERR_HI=${ERR_HI:-10}
ERR_STEP=${ERR_STEP:-1}

# Tiles: whatever was asked for on the command line, else the config's own list,
# so the two cannot drift apart.
if [ $# -gt 0 ]; then
    TILES=("$@")
else
    mapfile -t TILES < <(sed -n 's/^[[:space:]]*tiles[[:space:]]*=[[:space:]]*//p' "$CONF" \
                         | head -1 | cut -d'#' -f1 | tr ',' '\n' | sed 's/[[:space:]]//g' | grep .)
fi
[ ${#TILES[@]} -gt 0 ] || { echo "no tiles= in $CONF and none given"; exit 1; }

ERRS=$(seq "$ERR_LO" "$ERR_STEP" "$ERR_HI")
NRUN=$(( ${#TILES[@]} * $(wc -w <<< "$ERRS") ))

mkdir -p "$OUT/logs"
echo "sweep: ${#TILES[@]} tiles [${TILES[*]}] x errors [$ERR_LO..$ERR_HI step $ERR_STEP] = $NRUN runs"
echo "config $CONF, $NP ranks, output $OUT"
echo

i=0
t0=$SECONDS
for tile in "${TILES[@]}"; do
    for e in $ERRS; do
        i=$((i + 1))
        # same %g formatting steps15.py uses, so the skip check below matches
        tag="${tile}_$(awk -v e="$e" 'BEGIN{printf "%g", e + 100}')"
        if [ -f "$OUT/fbp_$tag.tiff" ]; then
            echo "[$i/$NRUN] $tag  already done, skipping"
            continue
        fi
        echo "[$i/$NRUN] $tag  ($((SECONDS - t0))s elapsed)"
        HOLO_TILE_ERR=$e HOLO_TILE_PROC=$tile HOLO_TILE_ERR_DIR=$OUT \
            "$MPIRUN" -n "$NP" "$PY" steps15.py "$CONF" \
            > "$OUT/logs/$tag.log" 2>&1
        rc=$?
        if [ $rc -ne 0 ]; then
            echo "    FAILED (exit $rc) — see $OUT/logs/$tag.log"
            tail -5 "$OUT/logs/$tag.log" | sed 's/^/    /'
        elif [ ! -f "$OUT/fbp_$tag.tiff" ]; then
            echo "    ran but wrote no tiff — see $OUT/logs/$tag.log"
        fi
    done
done

echo
echo "done: $i runs in $((SECONDS - t0))s"
ls -1 "$OUT"/fbp_*.tiff 2>/dev/null | wc -l | xargs echo "tiffs in $OUT:"
