#!/bin/bash
#PBS -A 14238
#PBS -l select=16:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:grand:eagle
#PBS -l walltime=6:00:00
#PBS -q prod
#PBS -N holoYY037A_rhosweep
#PBS -j oe

# YY037A 5-tile mosaic, step 6 posdist at bin 2 — sweep ONE parameter.
#
# Same solver as polaris_run_posdist.sh, run once per value of one parameter,
# everything else held fixed across the set so the swept quantity is the only
# thing that moves. Each run writes to its own path_out, so the runs never
# collide and can be compared afterwards:
#
#   python plot_rhosweep.py rhosweep-*.out --skip0
#
# SWEPT picks the parameter:
#
#   SWEPT=offx  horizontal placement of ONE tile, dx in FINEST-GRID px, applied
#               through the config's `tile_shift=<tile>:0:<dx>` on top of the
#               measured /exchange/tile_offsets (same units, added before the
#               1/2**bin scaling). This is the one thing steps 1-5 hand to
#               step 6 that step 6 cannot check for itself, and a wrong value
#               gives a plausible but meaningless reconstruction.
#               Configs config_step6_bin2_offx<tag>.conf, where the TAG IS
#               100 - dx (112 = dx-12, 100 = measured, 88 = dx+12) so the file
#               names carry no minus signs; the tile itself is named in the
#               config, not in the file name. VALS is in tags; the banner and
#               the plots are in dx.
#               Values dx = -12..12 step 2, i.e. -3..3 bin px at bin 2;
#               dx=0 is the measured placement.
#               These are SHORT runs, niter=65, error_step=8 — ranking the
#               offsets does not need convergence. rho is the tuned set,
#               rho_pos=0.001 rho_tp=3e-6, in every one of them.
#
#   SWEPT=tp    4th entry of rho=, the (ndist, 2, 2) linear shrink model
#               shrink[theta,dist,axis] = A*t + B, t = theta/(ntheta-1).
#               Values 0 1e-6 3e-6 1e-5 3e-5 1e-4, rho_pos=0.001.
#               DONE — optimum 3e-6 (err 2.1248e-04 @704, -2.71% vs frozen tp),
#               bracketed by 1e-6 (-2.45%) and 1e-5 (+3.28%). Logs 0-5.
#               The range stopped at 1e-4 on purpose: rho_tp=1e-3 multiplied the
#               shrink by ~25x in one BH step (2.6e-2 -> 6.5e-1 on the last
#               distance of every tile), pushed the demagnified sampling
#               coordinates off the object grid, and killed the job with
#               cudaErrorIllegalAddress (7486034). Larger fails the same way.
#               Configs deleted after the sweep; regenerate from a posdist
#               config by editing rho= and path_out if it needs redoing.
#
#   SWEPT=pos   3rd entry of rho=, one (y, x) shift per (tile x distance),
#               20 x 2 unknowns on top of the measured per-angle shifts.
#               An earlier sweep gave 1e-4 < 5e-3 < 1e-2 monotonically with no
#               interior minimum; 0.001 is what everything since has used.
#               NOT RE-RUN — configs deleted, see SWEPT=tp above.
#
# estimate_rho is FALSE in every sweep config — the whole point is to hold rho
# at the tuned value, and the coordinate search would immediately overwrite it.
#
# Steps 1-5 must already have run at bin 2.
#
# WALLTIME: the rho sweep took ~1 min per iteration at this rank count, so a
# 65-iteration offx run is ~1 h and the full 13 do NOT fit one 6 h job. Queue
# them in batches of four or five:
#
#   SWEPT=offx VALS="112 110 108 106 104"  qsub polaris_run_posdist_rhosweep.sh
#   SWEPT=offx VALS="102 100 98 96"        qsub polaris_run_posdist_rhosweep.sh
#   SWEPT=offx VALS="94 92 90 88"          qsub polaris_run_posdist_rhosweep.sh
#
# Every batch writes into the same sweep folder, so the collected slices end up
# together whatever order the jobs run in.
#
# step6.py resumes from the latest checkpoint in path_out, so re-queueing the
# same VALS carries on where the walltime cut it off. Careful with offx: a
# resumed run takes its positions from the checkpoint and does NOT re-apply
# tile_shift (step6.py warns), which is right — the offset is already baked in.
#
# OUTPUT: each run keeps its own path_out, and the middle slice of its final
# reconstruction is copied to a shared sweep folder as <tag>.tiff, with the
# convergence table as <tag>_conv.csv. Override the folder with SWEEP_DIR.

SWEPT=${SWEPT:-offx}
case "${SWEPT}" in
    offx) STEM=config_step6_bin2_offx; LABEL=offset_x; TAGGED=yes
          FIXED="farright tile, bin px, rho_pos=0.001 rho_tp=3e-6, niter=65"
          DEFAULT_VALS="112 110 108 106 104 102 100 98 96 94 92 90 88" ;;
    pos)  STEM=config_step6_bin2_posdist_posrho; LABEL=rho_pos
          FIXED="rho_tp pinned at 3e-6"
          DEFAULT_VALS="0 1e-5 3e-5 1e-4 3e-4 1e-3" ;;
    tp)   STEM=config_step6_bin2_posdist_rhotp;  LABEL=rho_tp
          FIXED="rho_pos pinned at 0.001"
          DEFAULT_VALS="0 1e-6 3e-6 1e-5 3e-5 1e-4" ;;
    *)    echo "SWEPT must be 'offx', 'pos' or 'tp', got '${SWEPT}'" >&2; exit 2 ;;
esac
# RHOS is the old name, still honoured so queued scripts keep working.
VALS=${VALS:-${RHOS:-${DEFAULT_VALS}}}

# offx names its configs by the tag 100-dx; everything reported to the log is
# the real dx, so the plots and the summary table are in physical units.
# Every other sweep uses the value itself as both.
untag() { if [ "${TAGGED:-}" = yes ]; then echo $((100 - $1)); else echo "$1"; fi; }
VALS_SHOWN=
for r in ${VALS}; do VALS_SHOWN="${VALS_SHOWN}$(untag ${r}) "; done

NNODES=$(wc -l < $PBS_NODEFILE)
NRANKS=4
NTHREADS=4
NDEPTH=8
export NTOTRANKS=$(( NNODES * NRANKS ))

SCRIPT_DIR="${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
rec_dir="$(dirname "${SCRIPT_DIR}")"          # .../experimental

# Snapshot the py and conf files actually used, into a dated folder, so the run
# records itself.
scripts_dir="${rec_dir}/scripts_rhosweep_${SWEPT}$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "${scripts_dir}"
cp "${SCRIPT_DIR}"/*.py   "${scripts_dir}/" 2>/dev/null || true
cp "${SCRIPT_DIR}"/*.conf "${scripts_dir}/" 2>/dev/null || true

cd "${rec_dir}"
exec > >(tee "${scripts_dir}/rhosweep-${PBS_JOBID}.out" \
              "${SCRIPT_DIR}/rhosweep-${PBS_JOBID}.out") 2>&1

# The working configs point at the beamline machine; rewrite to the eagle copy
# inside the snapshot and run the snapshot, leaving the working copies usable
# locally. Already-eagle paths are left alone.
DATA_ROOT_LOCAL=${DATA_ROOT_LOCAL:-/data2/vnikitin}
DATA_ROOT=${DATA_ROOT:-/eagle/APS_IRI/vnikitin}
sed -i "s|${DATA_ROOT_LOCAL}|${DATA_ROOT}|g" "${scripts_dir}"/*.conf
CFG="${scripts_dir}"

echo "Configs:     ${CFG}  (data root ${DATA_ROOT})"
echo "Sweep:       bin 2, ${LABEL} in ${VALS_SHOWN}  (${FIXED})"
grep -H '^path_out\|^rho=\|^tile_shift=' "${CFG}"/${STEM}*.conf
echo "Sample dir:  ${SCRIPT_DIR}"
echo "Rec dir:     ${rec_dir}"
echo "Snapshot:    ${scripts_dir}"
echo "Jobid: $PBS_JOBID"
echo "Running on host: $(hostname)"
echo "NUM_OF_NODES=${NNODES}  TOTAL_NUM_RANKS=${NTOTRANKS}  RANKS_PER_NODE=${NRANKS}"

module use /soft/modulefiles;  module load conda; conda activate base

export MATHDX_ROOT=/eagle/APS_IRI/vnikitin/nvidia/nvidia-mathdx-25.12.1-cuda12/nvidia/mathdx/25.12/

MPI="mpiexec -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth \
     --env OMP_NUM_THREADS=${NTHREADS} ${SCRIPT_DIR}/set_affinity_gpu_polaris.sh python"


# Every run keeps its own path_out with the full checkpoint tree, which is what
# the resume logic needs but is useless for comparing the runs by eye. So after
# each one, collect the middle slice of the FINAL reconstruction into one
# shared folder, one file per sweep point, named by the value that names the
# config: <tag>.tiff next to <tag>_conv.csv. Flip through them in the sweep
# folder and the placement error is immediately visible.
cfgval() {  # cfgval <conf> <key> — value with the inline comment stripped
    grep -m1 "^$2=" "$1" | cut -d= -f2- | cut -d'#' -f1 | tr -d ' \t'
}

# One folder for the whole sweep, beside the per-run path_outs. Taken from the
# first config that exists, so every batch of VALS lands in the same place.
if [ -z "${SWEEP_DIR:-}" ]; then
    for r in ${VALS}; do
        [ -f "${CFG}/${STEM}${r}.conf" ] || continue
        SWEEP_DIR="$(dirname "$(cfgval "${CFG}/${STEM}${r}.conf" path_out)")/YY037A_rec6_bin2_${SWEPT}_sweep"
        break
    done
fi
mkdir -p "${SWEEP_DIR}"
echo "Sweep slices: ${SWEEP_DIR}"

for r in ${VALS}; do
    v=$(untag ${r})
    conf="${CFG}/${STEM}${r}.conf"
    if [ ! -f "${conf}" ]; then
        echo "=== SKIP ${LABEL}=${v}: ${conf} not found"
        continue
    fi
    echo "=============================================================="
    echo "=== ${LABEL}=${v}   (config ${STEM}${r})   $(date)"
    grep '^path_out\|^rho=\|^tile_shift=' "${conf}"
    echo "=============================================================="
    ${MPI} "${SCRIPT_DIR}/step6.py" "${conf}"
    echo "=== ${LABEL}=${v} finished with status $?   $(date)"

    # Middle z of the last checkpoint: the writer saves z = nzobj/2 alongside
    # nzobj/2 +/- nzobj/8, and the last iteration of the loop is niter-1.
    out=$(cfgval "${conf}" path_out)
    last=$(printf '%04d' $(( $(cfgval "${conf}" niter) - 1 )))
    midz=$(printf '%04d' $(( $(cfgval "${conf}" nzobj) / 2 )))
    src="${out}/checkpoints_tiff/checkpoint_${last}_obj_re_z${midz}.tiff"
    if [ -f "${src}" ]; then
        cp "${src}" "${SWEEP_DIR}/${r}.tiff"
        echo "=== ${LABEL}=${v} mid slice -> ${SWEEP_DIR}/${r}.tiff"
    else
        echo "=== ${LABEL}=${v} NO mid slice at ${src} (run did not reach the end?)"
    fi
    [ -f "${out}/conv.csv" ] && cp "${out}/conv.csv" "${SWEEP_DIR}/${r}_conv.csv"
done

echo "=============================================================="
echo "Sweep slices collected in ${SWEEP_DIR}"
echo "  <tag>.tiff, tag = ${LABEL} $([ "${TAGGED:-}" = yes ] && echo 'encoded as 100 - dx' || echo 'itself')"
ls -la "${SWEEP_DIR}" 2>/dev/null
