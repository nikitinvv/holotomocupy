#!/bin/bash
#PBS -A 14238
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:eagle
#PBS -l walltime=0:59:00
#PBS -q debug
#PBS -N holo_c0
#PBS -j oe

# --- user configuration ---
# ROTATION-CENTRE SWEEP, candidate rotation_center_shift = 0 (bin-0 detector px).
# Generated 2026-08-25 from polaris_run_noshrink.sh; the differences are the job
# name and that the stages are hardcoded as consecutive mpiexec lines instead of
# selected by $CONFIG, so one qsub walks the whole ladder for this centre.
#
# The main HT run used -10.386198 and its reconstruction says that is wrong, so
# six candidates (0, -1, -2, -3, -4, -7) are each taken through the full ladder.
# noshrink only -- rho[tp]=0 at every level.
#
# One qsub per centre -- this script runs the whole ladder in sequence:
#     qsub polaris_run_noshrink_c0.sh
# steps15 (step 5 only) -> step 6 bin 2 (iters 0-512) -> bin 1 (512-768).
# bin 0 (768-1024) is the last mpiexec line, commented out: uncomment it when
# you want full resolution for this centre.
#
# Queue: the PBS header below is inherited from polaris_run_noshrink.sh --
# debug, 2 nodes, 59 min, which is sized for ONE step-6 level.  This script
# runs three stages back to back, so 59 min will not be enough; pick a queue
# and walltime that fit before submitting.  If it does wall, nothing partial
# is reusable: step 5 rewrites _obj.h5 from scratch, and each step-6 level
# restarts from its config's fixed start_iter.
#
# The steps15 job is step 5 only (start_step=5, start_level_rec=2): steps 1-4 do
# not depend on the centre, and re-running them would rewrite the shared 2.8 TB
# Y350a_HT_nobin_020nm.h5 -- do NOT qsub a config here with start_step < 5.
#
# ONE CENTRE AT A TIME.  Every centre reads and writes the SAME steps15 dir
# ..._rec; only the step-6 path_out is per-centre.  The steps15 stage
# overwrites the shared Y350a_HT_nobin_020nm_obj.h5, which is the only place
# the centre reaches step 6 (read at bin 2 with start_iter=0).  Because this
# script runs steps15 and bin 2 in the same job, that is safe -- as long as
# two centres are never in flight at once.  Do not qsub the next centre until
# this one has at least cleared its bin-2 stage.
#
# The first sweep steps15 destroys the main run's _obj.h5 (the -10.386198
# volume).  That is recoverable -- re-run config_steps15.conf with
# start_step=5 -- and the existing ..._rec6* ladders do not need it any more.
#
# The three step-6 levels share this centre's path_out
# (..._rec6_noshrink_c0), so each picks up the previous level's checkpoint.
# Software environment (modules + conda env). See the Polaris setup notes.
HTC_ENV=${HTC_ENV:-/eagle/APS_IRI/vvnikitin/sw/env.sh}
# --------------------------

NNODES=$(wc -l < $PBS_NODEFILE)
NRANKS=4
NTHREADS=4
NDEPTH=8
export NTOTRANKS=$(( NNODES * NRANKS ))

# Directory the job was submitted from (PBS_O_WORKDIR when submitted via qsub;
# falls back to the script's own directory for local ./polaris_run.sh testing).
# Plain $(pwd) does NOT work: PBS starts the job in $HOME, not where you qsub'd.
SCRIPT_DIR="${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
rec_dir="$(dirname "${SCRIPT_DIR}")"

cd "${rec_dir}"
exec > >(tee "${SCRIPT_DIR}/slurm-${PBS_JOBID}.out") 2>&1

echo "Sample dir:  ${SCRIPT_DIR}"
echo "Rec dir:     ${rec_dir}"
echo "Jobid: $PBS_JOBID"
echo "Running on host: $(hostname)"
echo "Running on nodes: $(cat $PBS_NODEFILE)"
echo "NUM_OF_NODES=${NNODES}  TOTAL_NUM_RANKS=${NTOTRANKS}  RANKS_PER_NODE=${NRANKS}"

# Modules + conda env. env.sh loads PrgEnv-gnu, cray-mpich, cudatoolkit,
# cray-hdf5-parallel and activates the holotomocupy env; it must be sourced
# inside the job, not just at install time, or the cray-mpich-linked mpi4py
# and h5py will not find their libraries.
[ -r "${HTC_ENV}" ] || { echo "ERROR: HTC_ENV not readable: ${HTC_ENV}"; exit 1; }
source "${HTC_ENV}"
echo "python: $(which python)"

# --- drop nodes whose GPUs cannot take a CUDA context ----------------------
# PBS has no Slurm-style --exclude: `-l select=` can pin a host (host=/vnode=)
# but cannot negate one, so a node that comes up with
#   cudaErrorDevicesUnavailable: CUDA-capable device(s) is/are busy or unavailable
# can only be filtered from inside the job.  This must run AFTER env.sh is
# sourced -- the probe needs cupy.
#
#   RUN_NODES=N   run on exactly N healthy nodes (over-request N+1 in select=
#                 so one bad node costs a node-hour, not the whole job).
#                 Default: use every healthy node.
#   HEALTHCHECK=0 skip the probe entirely (~30 s).
HOSTOPT=""
if [ "${HEALTHCHECK:-1}" = "1" ]; then
    GOOD="${SCRIPT_DIR}/nodes.good.${PBS_JOBID}"
    if bash "${rec_dir}/gpu_healthcheck.sh" "${GOOD}" "${NRANKS}" "${RUN_NODES:-1}"; then
        NEED=${RUN_NODES:-$(wc -l < "${GOOD}")}
        head -n "${NEED}" "${GOOD}" > "${GOOD}.run"
        NNODES=$(wc -l < "${GOOD}.run")
        export NTOTRANKS=$(( NNODES * NRANKS ))
        HOSTOPT="--hostfile ${GOOD}.run"
        echo "Running on ${NNODES} healthy nodes  TOTAL_NUM_RANKS=${NTOTRANKS}"
    else
        echo "ERROR: too few healthy nodes in this allocation; aborting."
        exit 1
    fi
fi

# Fallback: ALCF-provided base conda + venv layered on top
# module use /soft/modulefiles;  module load conda; conda activate base
# CONDA_NAME=$(echo ${CONDA_PREFIX} | tr '\/' '\t' | sed -E 's/mconda3|\/base//g' | awk '{print $NF}')
# source "/home/vvnikitin/venvs/${CONDA_NAME}/bin/activate"

mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/steps15.py" "${SCRIPT_DIR}/config_steps15_c0.conf"
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_bin2_noshrink_c0.conf"
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_bin1_noshrink_c0.conf"
# mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_bin0_noshrink_c0.conf"
