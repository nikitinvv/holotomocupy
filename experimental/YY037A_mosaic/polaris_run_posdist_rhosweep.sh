#!/bin/bash
#PBS -A 14238
#PBS -l select=16:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:grand:eagle
#PBS -l walltime=6:00:00
#PBS -q prod
#PBS -N holoYY037A_rhosweep
#PBS -j oe

# YY037A 5-tile mosaic, step 6 posdist at bin 2 — sweep over rho for SHRINKAGE.
#
# Same solver as polaris_run_posdist.sh, run once per value of the tp entry of
# rho= (the FOURTH one). obj/prb are held at 1 / 0.05 and, importantly, pos is
# held at 0.001 in every config, so the shrinkage weight is the only thing that
# moves between runs. Each run writes to its own path_out ending in
# _rhotp<value>, so the runs never collide and can be compared afterwards:
#
#   grep 'err=' rhosweep-*.out
#   python plot_rhosweep.py rhosweep-*.out --skip0
#
# tp is the (ndist, 2, 2) linear shrink model: shrink[theta,dist,axis] = A*t + B
# with t = theta/(ntheta-1). rho_tp=0 freezes it at whatever steps 1-5 measured,
# which is the baseline every other run is compared against.
#
# estimate_rho is FALSE in every sweep config — the whole point is to hold rho
# at the swept value, and the coordinate search would immediately overwrite it.
#
# Steps 1-5 must already have run at bin 2.
#
# WALLTIME: niter=1025 at ntheta=1800, six times over, will not fit in one job.
# Cut RHOS down and queue several, or lower niter in the sweep configs:
#
#   RHOS="0 1e-5 1e-4" qsub polaris_run_posdist_rhosweep.sh
#   RHOS="1e-3 1e-2 1e-1" qsub polaris_run_posdist_rhosweep.sh
#
# The runs checkpoint every 32 iters and step6.py resumes from the latest one,
# so re-queueing the same RHOS carries on where the walltime cut it off.
#
# Configs exist for  0  1e-5  1e-4  1e-3  1e-2  1e-1.

RHOS=${RHOS:-"0 1e-5 1e-4 1e-3 1e-2 1e-1"}

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
echo "Sweep:       bin 2, rho_tp in ${RHOS}   (rho_pos pinned at 0.001)"
grep -H '^path_out\|^rho=' "${CFG}"/config_step6_bin2_posdist_rhotp*.conf
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
    conf="${CFG}/config_step6_bin2_posdist_rhotp${r}.conf"
    if [ ! -f "${conf}" ]; then
        echo "=== SKIP rho_tp=${r}: ${conf} not found"
        continue
    fi
    echo "=============================================================="
    echo "=== rho_tp=${r}   $(date)"
    grep '^path_out\|^rho=' "${conf}"
    echo "=============================================================="
    ${MPI} "${SCRIPT_DIR}/step6.py" "${conf}"
    echo "=== rho_tp=${r} finished with status $?   $(date)"
done
