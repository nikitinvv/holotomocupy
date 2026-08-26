#!/bin/bash
#PBS -A 14238
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:eagle
#PBS -l walltime=12:00:00
#PBS -q preemptable
#PBS -N y350a006
#PBS -j oe
# ===========================================================================
# Y350a large random displacement, 6 nm voxels -- ESRF ID16A 2025-06-04.
# THE WHOLE PIPELINE IN ONE JOB.
#
#     qsub polaris_run.sh
#
# Four mpiexec calls, back to back, in a single allocation.  To run only part
# of the pipeline -- steps 1-5 already done, or resuming after a preemption --
# comment out the leading mpiexec lines at the bottom of this file.  Each line
# ends in `|| exit $?` so a failed stage stops the job instead of letting the
# next level seed itself from a checkpoint that was never written.
#
#   stage    script      config                     what it does
#   -------  ----------  -------------------------  ----------------------------
#   steps15  steps15.py  config_steps15.conf        EDF->HDF5, preprocess,
#                                                   shifts, binned data,
#                                                   Paganin+FBP for bins 2,1,0
#   bin2     step6.py    config_step6_bin2.conf     4x4  n=1024 iters    0->512
#   bin1     step6.py    config_step6_bin1.conf     2x2  n=2048 iters  512->768
#   bin0     step6.py    config_step6_bin0.conf     1x1  n=4096 iters  768->1024
#
# The three step-6 levels share one path_out, so each seeds itself from the
# previous level's checkpoint (start_iter) and Reader.read_checkpoint upsamples
# obj/prb/pos onto the finer grid.  Running them in one job is what that chain
# wants: no handoff, no requeue between levels.
#
# --- WALLTIME --------------------------------------------------------------
# Measured on the 20 nm sibling (identical n=4096, nobj=4608, ntheta=4000;
# 2 nodes, 8 ranks), per 32 iterations:  bin2 ~48 s, bin1 ~170 s, bin0 ~800 s.
# That is 13 min + 22 min + 1 h 47 min = ~2 h 25 min of step 6.  steps 1-5 were
# never timed; step 1 reads 128 GB of EDF and writes a few hundred GB back, so
# budget several hours of eagle I/O.  12 h total should be comfortable -- check
# the timestamps in the .out file and trim it for the next scan.
#
# 12 h does not fit the debug queue, hence -q preemptable (1-10 nodes, long
# walltime, evictable).  Preemption is survivable: checkpoint_step=32, so a
# resubmit loses at most 32 iterations -- comment out the stages that already
# finished and lower that level's start_iter to the last checkpoint on disk.
# Confirm the current queue limits with `qstat -Qf` before changing this.
#
# --- DISK ------------------------------------------------------------------
# Budget ~800 GB in the steps15 path_out (.../Y350a_FT_large_rand_disp_006nm_rec):
# the 20 nm sibling produced a 757 GB <pfile>.h5 plus a 12 GB _obj.h5.
#
# No shrinkage correction anywhere (rho[tp]=0 at every level) and no
# correct_motion.txt -- correct.txt is the only shift source.  See README.md.
# ===========================================================================

# --- user configuration ---
# Software environment (modules + conda env). See the Polaris setup notes.
HTC_ENV=${HTC_ENV:-/eagle/APS_IRI/vvnikitin/sw/env.sh}
# HEALTHCHECK=0  skips the ~30 s GPU probe;  RUN_NODES=N  uses N healthy nodes.
# --------------------------

NNODES=$(wc -l < $PBS_NODEFILE)
NRANKS=4          # one rank per Polaris A100
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

# --- the pipeline -----------------------------------------------------------
# Comment out the lines you do not want to run.  Times are the 20 nm sibling
# measurements on 2 nodes / 8 ranks (steps15 was never timed).

# steps 1-5   EDF->HDF5, preprocess, shifts, binned, Paganin+FBP
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/steps15.py" "${SCRIPT_DIR}/config_steps15.conf" || exit $?

# bin 2  4x4  n=1024  iters    0 -> 512   ~13 min
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_bin2.conf" || exit $?

# bin 1  2x2  n=2048  iters  512 -> 768   ~22 min
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_bin1.conf" || exit $?

# bin 0  1x1  n=4096  iters  768 ->1024   ~1 h 47 min
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_bin0.conf" || exit $?

echo "=== ALL STAGES DONE $(date) ==="
