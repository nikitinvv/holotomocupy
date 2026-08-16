#!/bin/bash
#PBS -A 14238
#PBS -l select=16:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:grand:eagle
#PBS -l walltime=3:00:00
#PBS -q prod
#PBS -N holotomo_YY037A
#PBS -j oe

# YY037A, 5-tile mosaic. Steps 1-5 (steps15.py) then step 6 (step6.py) on the
# assembled mosaic; uncomment the lines you want at the bottom.
#
# 16 nodes x 4 ranks = 64 ranks. Both step-6 levels split comfortably over
# that: bin 3 is 300 angles / 448 object slices, bin 2 is 900 / ~848. Steps 1-4
# of steps15 are per tile and per angle, so they scale with the rank count too;
# step 5 holds the whole mosaic grid (nzobj x nobj = 3584 x 12288 on the finest
# grid), which is why start_level_rec in config_steps15.conf decides how much
# of it is actually reconstructed.

NNODES=$(wc -l < $PBS_NODEFILE)
NRANKS=4
NTHREADS=4
NDEPTH=8
export NTOTRANKS=$(( NNODES * NRANKS ))

# Directory the job was submitted from (PBS_O_WORKDIR when submitted via qsub;
# falls back to the script's own directory for local ./polaris_run.sh testing).
SCRIPT_DIR="${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
echo $SCRIPT_DIR
rec_dir="$(dirname "${SCRIPT_DIR}")"

# snapshot only py and conf files into a dated folder inside rec_dir
scripts_dir="${rec_dir}/scripts$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "${scripts_dir}"
cp "${SCRIPT_DIR}"/*.py   "${scripts_dir}/" 2>/dev/null || true
cp "${SCRIPT_DIR}"/*.conf "${scripts_dir}/" 2>/dev/null || true

cd "${rec_dir}"
exec > >(tee "${scripts_dir}/slurm-${PBS_JOBID}.out" "${SCRIPT_DIR}/slurm-${PBS_JOBID}.out") 2>&1

# The configs in this directory point at the beamline machine (/data2/vnikitin).
# Rewrite that to the eagle copy in the snapshot, and run the snapshot configs,
# so the job records exactly what it ran and the working copies stay usable
# locally. Already-eagle paths are left alone.
DATA_ROOT_LOCAL=${DATA_ROOT_LOCAL:-/data2/vnikitin}
DATA_ROOT=${DATA_ROOT:-/eagle/APS_IRI/vnikitin}
sed -i "s|${DATA_ROOT_LOCAL}|${DATA_ROOT}|g" "${scripts_dir}"/*.conf
CFG="${scripts_dir}"
echo "Configs:     ${CFG}  (data root ${DATA_ROOT})"
grep -H '^path' "${CFG}"/config_steps15.conf

echo "Sample dir:  ${SCRIPT_DIR}"
echo "Rec dir:     ${rec_dir}"
echo "Snapshot:    ${scripts_dir}"
echo "Jobid: $PBS_JOBID"
echo "Running on host: $(hostname)"
echo "Running on nodes: $(cat $PBS_NODEFILE)"
echo "NUM_OF_NODES=${NNODES}  TOTAL_NUM_RANKS=${NTOTRANKS}  RANKS_PER_NODE=${NRANKS}"

module use /soft/modulefiles;  module load conda; conda activate base
CONDA_NAME=$(echo ${CONDA_PREFIX} | tr '\/' '\t' | sed -E 's/mconda3|\/base//g' | awk '{print $NF}')
VENV_DIR="/home/vvnikitin/venvs/${CONDA_NAME}"
source "${VENV_DIR}/bin/activate"

export MATHDX_ROOT=/eagle/APS_IRI/vnikitin/nvidia/nvidia-mathdx-25.12.1-cuda12/nvidia/mathdx/25.12/

MPI="mpiexec -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth \
     --env OMP_NUM_THREADS=${NTHREADS} ${SCRIPT_DIR}/set_affinity_gpu_polaris.sh python"

# --- Steps 1-5: convert, preprocess, shifts, binned data, mosaic Paganin+FBP
# One run does whatever start_step..5 the config asks for. Step 5 writes
# /exchange/tile_offsets and {pfile}_mosaic_obj.h5, which step 6 needs.
# ${MPI} "${SCRIPT_DIR}/steps15.py" "${CFG}/config_steps15.conf"

# --- TEMPORARY: tile placement-error sweep -----------------------------------
# The Polaris counterpart of run.sh. For each tile, nudge it horizontally by
# ERR_LO..ERR_HI object px on the finest grid, on top of whatever step 5
# measured, and keep the middle FBP slice as a tiff. The tile whose series is
# best centred on 0 is correctly placed; a lopsided one is off by the amount of
# its minimum. One mpiexec per (tile, error) pair, run one after another inside
# this job — each needs all the ranks, so they cannot overlap.
#
# steps15_sweep.py rather than steps15.py: it pins the placement AFTER writing
# /exchange/tile_offsets, so a sweep run varies only the deliberate error and
# what lands on disk stays the real measurement.
#
# Wants a single-bin config (start_level_rec == nlevels-1, as config_steps15.conf
# has it): `offsets` carries across bins, so on a multi-bin run the error would
# both accumulate and be measured away again by the next bin's estimate_overlap.
#
# Restartable: a pair whose tiff already exists is skipped, so re-queueing
# carries on from where the walltime cut it off. SWEEP_BUDGET stops it launching
# a run it cannot finish — one killed mid-write would leave a truncated tiff
# that the skip check would then trust forever. The default -10..10 is 21 runs
# per tile, so budget accordingly or raise the walltime at the top of this file.
#
#   sweep                      # every tile in the config
#   sweep center               # just one
#   sweep center left          # a couple
#   SWEEP_OUT=...  ERR_LO=-10 ERR_HI=10 ERR_STEP=1  SWEEP_BUDGET=<seconds>
sweep() {
    local out=${SWEEP_OUT:-${DATA_ROOT}/tmp/YY037A_sweep}
    local lo=${ERR_LO:--10} hi=${ERR_HI:-10} step=${ERR_STEP:-1}
    local budget=${SWEEP_BUDGET:-9000}
    local conf="${CFG}/config_steps15.conf"

    # Tiles named on the call, else every tile in the config — so a whole-mosaic
    # sweep and the run itself cannot drift apart, and a single tile is one word.
    local tiles
    if [ $# -gt 0 ]; then
        tiles=("$@")
    else
        mapfile -t tiles < <(sed -n 's/^[[:space:]]*tiles[[:space:]]*=[[:space:]]*//p' "${conf}" \
                             | head -1 | cut -d'#' -f1 | tr ',' '\n' \
                             | sed 's/[[:space:]]//g' | grep .)
    fi
    [ ${#tiles[@]} -gt 0 ] || { echo "sweep: no tiles given and no tiles= in ${conf}"; return 1; }

    local errs; errs=$(seq "${lo}" "${step}" "${hi}")
    local nrun=$(( ${#tiles[@]} * $(wc -w <<< "${errs}") ))
    mkdir -p "${out}/logs"
    echo "sweep: ${#tiles[@]} tiles [${tiles[*]}] x errors [${lo}..${hi} step ${step}]"
    echo "       = ${nrun} runs -> ${out}, budget ${budget}s"

    local i=0 t0=${SECONDS} per=0 tag s rc
    for tile in "${tiles[@]}"; do
        for e in ${errs}; do
            i=$(( i + 1 ))
            # same %g formatting steps15_sweep.py uses, so this skip check matches
            tag="${tile}_$(awk -v e="${e}" 'BEGIN{printf "%g", e + 100}')"
            if [ -f "${out}/fbp_${tag}.tiff" ]; then
                echo "[${i}/${nrun}] ${tag}  already done, skipping"
                continue
            fi
            if [ $(( SECONDS - t0 + per )) -gt "${budget}" ]; then
                echo "sweep: budget reached with ${tag} next — re-queue to carry on"
                return 0
            fi
            echo "[${i}/${nrun}] ${tag}  ($(( SECONDS - t0 ))s elapsed)"
            s=${SECONDS}
            # Written out rather than reusing ${MPI}: the three HOLO_ variables
            # have to reach the ranks, and --env is the only way to be sure of
            # that regardless of how the launcher forwards its environment.
            mpiexec -n "${NTOTRANKS}" --ppn "${NRANKS}" --depth="${NDEPTH}" \
                    --cpu-bind depth --env OMP_NUM_THREADS="${NTHREADS}" \
                    --env HOLO_TILE_ERR="${e}" \
                    --env HOLO_TILE_PROC="${tile}" \
                    --env HOLO_TILE_ERR_DIR="${out}" \
                    "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python \
                    "${SCRIPT_DIR}/steps15_sweep.py" "${conf}" \
                    > "${out}/logs/${tag}.log" 2>&1
            rc=$?
            per=$(( SECONDS - s ))
            if [ ${rc} -ne 0 ]; then
                echo "    FAILED (exit ${rc}) — see ${out}/logs/${tag}.log"
                tail -5 "${out}/logs/${tag}.log" | sed 's/^/    /'
            elif [ ! -f "${out}/fbp_${tag}.tiff" ]; then
                echo "    ran but wrote no tiff — see ${out}/logs/${tag}.log"
            fi
        done
    done
    echo "sweep: ${i} runs in $(( SECONDS - t0 ))s"
    ls -1 "${out}"/fbp_*.tiff 2>/dev/null | wc -l | xargs echo "tiffs in ${out}:"
}
sweep center
# sweep farright

# --- Step 6: full BH reconstruction of the mosaic, coarse to fine ------------
# ${MPI} "${SCRIPT_DIR}/step6.py" "${CFG}/config_step6_bin3.conf"
# ${MPI} "${SCRIPT_DIR}/step6.py" "${CFG}/config_step6_bin2.conf"
