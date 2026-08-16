#!/bin/bash
#PBS -A 14238
#PBS -l select=16:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:grand:eagle
#PBS -l walltime=6:00:00
#PBS -q prod
#PBS -N holoYY037A_rhosweep
#PBS -j oe

# YY037A 5-tile mosaic, step 6 posdist — sweep over rho_pos.
#
# Same solver as polaris_run_posdist.sh, run once per value of the POSITION
# entry of rho= (the third one; obj/prb/tp are held at 1 / 0.05 / 1e-4). Each
# run writes to its own path_out ending in _rho<value>, so the six runs never
# collide and can be compared side by side afterwards:
#
#   grep 'pos error' rhosweep-*.out
#   cat /data2/.../YY037A_rec6_bin2_posdist_rho*/pos.csv
#
# estimate_rho is FALSE in every sweep config — the whole point is to hold rho
# at the swept value, and the coordinate search would immediately overwrite it.
#
# Steps 1-5 must already have run. BIN=2 needs step 5 output at bin 2, BIN=3
# needs step 5 run with nlevels=4, start_level_rec=3.
#
# WALLTIME: the bin-2 configs are niter=1025 at ntheta=900 — six of those will
# not fit in one job. Either cut RHOS down and re-queue, lower niter in the
# sweep configs, or sweep at bin 3 first (BIN=3, niter=2048 but 1/4 the frame
# area and ntheta=300). The runs are checkpointed every 32 iters and step6.py
# resumes from the latest one, so re-queueing this script carries on where the
# walltime cut it off.

BIN=${BIN:-2}
RHOS=${RHOS:-"0.0001 0.005 0.01 0.02 0.05 0.1"}

NNODES=$(wc -l < $PBS_NODEFILE)
NRANKS=4
NTHREADS=4
NDEPTH=8
export NTOTRANKS=$(( NNODES * NRANKS ))

SCRIPT_DIR="${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
rec_dir="$(dirname "${SCRIPT_DIR}")"          # .../experimental

# Snapshot the py and conf files actually used, into a dated folder, so the run
# records itself.
scripts_dir="${rec_dir}/scripts_rhosweep$(date +%Y-%m-%d_%H-%M-%S)"
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
echo "Sweep:       bin=${BIN}  rho_pos in ${RHOS}"
grep -H '^path_out\|^rho=' "${CFG}"/config_step6_bin${BIN}_posdist_rho*.conf
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

for r in ${RHOS}; do
    conf="${CFG}/config_step6_bin${BIN}_posdist_rho${r}.conf"
    if [ ! -f "${conf}" ]; then
        echo "=== SKIP rho_pos=${r}: ${conf} not found"
        continue
    fi
    echo "=============================================================="
    echo "=== rho_pos=${r}   $(date)"
    grep '^path_out\|^rho=' "${conf}"
    echo "=============================================================="
    ${MPI} "${SCRIPT_DIR}/step6.py" "${conf}"
    echo "=== rho_pos=${r} finished with status $?   $(date)"
done
