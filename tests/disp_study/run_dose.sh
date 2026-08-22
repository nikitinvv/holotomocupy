#!/bin/bash
# Dose-matched comparison of a four-distance and a single-distance acquisition
# of the synthetic phantom -- run_dose_brain.sh with the phantom in place of the
# brain volume, and otherwise the same optics, the same sizes and the same
# figure.
#
# The beam diverges from a focus, so at sample-to-focus distance z1 the fluence
# in the sample goes as 1/z1^2 and a far projection costs a fraction of a near
# one.  On the ID16A configuration used here (z1 = 8.25/8.60/10.01/12.95 mm) the
# four distances cost 1 + 0.920 + 0.679 + 0.406 = 3.005 near-distance
# projections, so the single-distance scan gets 3.005 x NTHETA angles and both
# scans put the same dose into the sample.  (Rounded to a multiple of NP -- the
# theta split handles a remainder, but an even one keeps the ranks balanced; at
# NP=4, 900 -> 2704.)
#
# Two deliberate differences from the earlier /data3/vnikitin/dose_study runs:
#   * the distances are the brain ones (GEOMETRY=brain), not the old study's,
#     so the dose factor is 3.005 rather than 2.697 and the angle counts differ;
#   * the positions are not refined (FREEZE_POS=1) -- they are generated exactly
#     and stay exactly, so the comparison is about the acquisition alone.
# Because of the first, this writes to its own root; the old results are left
# where they are.
#
#     ./run_dose.sh
#     NTHETA=450 NITER=129 ./run_dose.sh      # a quick shakedown
#     AMP=32 PRB_SMOOTH=1 ./run_dose.sh       # exactly the brain run's knobs
#     FREEZE_POS=0 POS_ERR=2 ./run_dose.sh    # refine positions after all
#
set -e
cd "$(dirname "$0")"

NP=${NP:-4}                                   # MPI ranks (one GPU each)
N=${N:-1024}                                  # detector size [px]
AMP=${AMP:-16}                                # displacement half-width [px]
PRB_SMOOTH=${PRB_SMOOTH:-2}                   # probe blur sigma [px], 0 = as measured
NTHETA=${NTHETA:-900}                         # angles per distance, multi-distance scan
NDIST=${NDIST:-4}
NITER=${NITER:-513}
POS_ERR=${POS_ERR:-0}                         # initial position error [px]
FREEZE_POS=${FREEZE_POS:-1}                   # 1 = do not refine positions at all
OBJ_BLUR=${OBJ_BLUR:-0}                       # sigma of the initial object blur [px]
MARGIN=${MARGIN:-192}                         # blank border on each side of the object grid
# The phantom is generated on the object grid, so NOBJ sets how much blank space
# surrounds it -- same rule as the brain test, NOBJ = N + 2*MARGIN, and MARGIN
# is also the cap on AMP.
NOBJ=${NOBJ:-$(( N + 2 * MARGIN ))}
PHOTONS=${PHOTONS:-0}                         # 0 = noiseless
CHECKPOINT_STEP=${CHECKPOINT_STEP:-32}
ERROR_STEP=${ERROR_STEP:-32}                  # conv.csv row spacing = convergence-plot density
# the two legs differ in ndist and ntheta, so the step sizes that balance object
# against probe are not the same for both; off by default, as in the brain test
ESTIMATE_RHO=${ESTIMATE_RHO:-0}
RHO_NITER=${RHO_NITER:-16}
OUT_ROOT=${OUT_ROOT:-/data3/vnikitin/dose_study_phantom}
DPI=${DPI:-300}
# the phantom's Re(obj) = -delta sits in roughly [-4, 1]; the brain's [-70, -20]
# is its own scale, the figure style is what the two share
VMIN=${VMIN:--4}
VMAX=${VMAX:-1}
# empty VMIN/VMAX = let the figure pick the range from the 99.8th percentile of
# the ground truth.  Built with if, not [ ] && -- under set -e a failing test as
# the last command of a && list exits the script.
VRANGE=""
if [ -n "$VMIN" ]; then VRANGE="$VRANGE --vmin $VMIN"; fi
if [ -n "$VMAX" ]; then VRANGE="$VRANGE --vmax $VMAX"; fi
REC_FLAGS=${REC_FLAGS:-}
if [ "$ESTIMATE_RHO" = "1" ]; then
    REC_FLAGS="$REC_FLAGS --estimate-rho --rho-estimate-niter $RHO_NITER"
fi
if [ "$FREEZE_POS" = "1" ]; then
    REC_FLAGS="$REC_FLAGS --freeze-pos"
fi
# ndist distances at once cost ndist times the memory per chunk, so the
# multi-distance run takes smaller passes than the single-distance one
NCHUNK_GEN=${NCHUNK_GEN:-8}
NCHUNK_REC=${NCHUNK_REC:-8}
export OUT_ROOT
# the same optics as the brain test: 2.963 um unbinned pixel and
# z1 = 8.25/8.60/10.01/12.95 mm, scaled to N.  Must be exported before any
# "import common" below.  GEOMETRY=default goes back to the old study's optics.
export DISP_STUDY_GEOMETRY=${GEOMETRY:-brain}
export DISP_STUDY_OUT="$OUT_ROOT"             # common.OUT_ROOT, e.g. for the object cache

PYPATH="import sys;sys.path.insert(0,'.');import common as C;"
NTHETA1=$(python -c "${PYPATH}print(int(round(C.dose_equivalent_ntheta($NTHETA, C.Z1_ALL[:$NDIST])/$NP))*$NP)")
FACTOR=$(python -c "${PYPATH}print('%.4f' % (C.dose_weights(C.Z1_ALL[:$NDIST]).sum()))")
name_of() { python -c "${PYPATH}print(C.dose_case_name($1,$2,$AMP,$PRB_SMOOTH))"; }

DIR_M=$OUT_ROOT/$(name_of "$NDIST" "$NTHETA")     # the multi-distance scan
DIR_1=$OUT_ROOT/$(name_of 1 "$NTHETA1")           # the dose-matched single-distance one

mkdir -p "$OUT_ROOT"
RUN="mpirun -np $NP ./set_affinity_gpu.sh python"

echo "=============== dose-matched phantom comparison ==============="
echo "  optics : $N x $N,  amp = +-$AMP px,  probe sigma = $PRB_SMOOTH px"
echo "  object : phantom on a $NOBJ px grid (${MARGIN} px margin)"
echo "  recon  : $NITER BH iterations, obj blur $OBJ_BLUR px, positions $([ "$FREEZE_POS" = 1 ] && echo "frozen" || echo "refined from +-$POS_ERR px")"
echo "  rho    : $([ "$ESTIMATE_RHO" = 1 ] && echo "searched, $RHO_NITER iters per trial" || echo "rec.py default")"
echo "  dose of $NDIST distances = $FACTOR near-distance projections"
echo "  $NDIST x $NTHETA angles   ->  $DIR_M"
echo "  1 x $NTHETA1 angles  ->  $DIR_1"
echo

# --- the multi-distance scan -------------------------------------------------
tag=$(basename "$DIR_M")
$RUN gen_data.py --n "$N" --ntheta "$NTHETA" --ndist "$NDIST" --amp "$AMP" \
    --prb-smooth "$PRB_SMOOTH" --nobj "$NOBJ" --photons "$PHOTONS" --out "$DIR_M" \
    --nchunk $((NCHUNK_GEN / NDIST > 0 ? NCHUNK_GEN / NDIST : 1)) \
    2>&1 | tee -a "$OUT_ROOT/log_gen_${tag}.txt"
$RUN rec.py --in "$DIR_M" --niter "$NITER" --pos-err "$POS_ERR" \
    --obj-blur "$OBJ_BLUR" --checkpoint-step "$CHECKPOINT_STEP" --error-step "$ERROR_STEP" \
    --nchunk $((NCHUNK_REC / NDIST > 0 ? NCHUNK_REC / NDIST : 1)) $REC_FLAGS \
    2>&1 | tee -a "$OUT_ROOT/log_rec_${tag}.txt"

# --- the dose-matched single-distance scan -----------------------------------
tag=$(basename "$DIR_1")
$RUN gen_data.py --n "$N" --ntheta "$NTHETA1" --ndist 1 --amp "$AMP" \
    --prb-smooth "$PRB_SMOOTH" --nobj "$NOBJ" --photons "$PHOTONS" --out "$DIR_1" \
    --nchunk "$NCHUNK_GEN" \
    2>&1 | tee -a "$OUT_ROOT/log_gen_${tag}.txt"
$RUN rec.py --in "$DIR_1" --niter "$NITER" --pos-err "$POS_ERR" \
    --obj-blur "$OBJ_BLUR" --checkpoint-step "$CHECKPOINT_STEP" --error-step "$ERROR_STEP" \
    --nchunk "$NCHUNK_REC" $REC_FLAGS \
    2>&1 | tee -a "$OUT_ROOT/log_rec_${tag}.txt"

# the same figure as the brain comparison -- same layout, same crop, same
# convergence panel under the ground truth, same resolution; only the grey range
# is the phantom's own
python compare_dose.py --root "$OUT_ROOT" --ndist "$NDIST" --ntheta "$NTHETA" \
    --ntheta1 "$NTHETA1" --np "$NP" --amp "$AMP" --prb-smooth "$PRB_SMOOTH" \
    --tag phantom $VRANGE --crop 0.10 --dpi "$DPI"

echo
echo "=============== summary ==============="
for d in "$DIR_M" "$DIR_1"; do
    f=$(ls -t "$d"/rec*/summary.txt 2>/dev/null | head -1)
    if [ -n "$f" ]; then echo "$(basename "$d")  $(grep nrmse_obj "$f")"; else echo "$(basename "$d")  no summary.txt"; fi
done
