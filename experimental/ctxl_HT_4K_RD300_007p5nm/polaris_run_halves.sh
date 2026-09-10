#!/bin/bash
#PBS -A 14238
#PBS -l select=24:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:eagle
#PBS -l walltime=2:30:00
#PBS -q prod
#PBS -N ctxl0075h
#PBS -j oe
# ===========================================================================
# ctxl cortex tissue, 4-distance HT, +-300 px random displacement, 7.5 nm
# voxels -- ESRF ID16A 2026-08-29..31, proposal ihls3888.
# THE TWO HALF-SET LADDERS IN ONE JOB:
#
#     qsub polaris_run_halves.sh
#
# p0 = even projections (0, 2, ..., 3998), p1 = odd (1, 3, ..., 3999), each
# reconstructed with the SAME three-level ladder and the SAME iteration counts
# as the full run, with the positions frozen at what the full 4000-projection
# run found.  Six mpiexec lines below: the p0 ladder, then the p1 ladder.
# Comment out the ones you do not want; each ends in `|| exit $?` so a failed
# level stops the job instead of letting the next one seed itself from a
# checkpoint that was never written.
#
# PREREQUISITES
#
#   * steps15 has already run -- this script does NOT run it.  Both halves read
#     the same <pfile>_rec/<pfile>.h5 and the same Paganin+FBP initial object.
#   * the full ladder has already run, and
#     <pfile>_rec6/checkpoints/checkpoint_1504.h5 exists.  That is where the
#     frozen positions come from; every one of the six configs names it.
#
# Outputs go to <pfile>_rec6_p0 and <pfile>_rec6_p1, both separate from the
# full run's <pfile>_rec6.
#
# Cost: half the projections is a little under half the full ladder, which took
# 0.71 + 0.66 + 2.78 h on 2 nodes.  Both halves together fit in ~5 h; walltime
# is 12 h for preemption headroom.
#
# The split, what is and is not independent between the halves, and why an FSC
# between them is not an unbiased resolution measure: see the header of
# config_step6_p0_bin2.conf.
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

# Drop nodes whose GPUs cannot take a CUDA context.  PBS has no Slurm-style
# --exclude -- `-l select=` can pin a host but cannot negate one -- so a node
# that comes up with cudaErrorDevicesUnavailable can only be filtered from
# inside the job.  Must run AFTER env.sh: the probe needs cupy.
HOSTOPT=""
if [ "${HEALTHCHECK:-1}" = "1" ]; then
    GOOD="${SCRIPT_DIR}/nodes.good.${PBS_JOBID}"
    bash "${rec_dir}/gpu_healthcheck.sh" "${GOOD}" "${NRANKS}" "${RUN_NODES:-1}" || { echo "ERROR: too few healthy nodes in this allocation; aborting."; exit 1; }
    head -n "${RUN_NODES:-$(wc -l < "${GOOD}")}" "${GOOD}" > "${GOOD}.run"
    NNODES=$(wc -l < "${GOOD}.run")
    export NTOTRANKS=$(( NNODES * NRANKS ))
    HOSTOPT="--hostfile ${GOOD}.run"
    echo "Running on ${NNODES} healthy nodes  TOTAL_NUM_RANKS=${NTOTRANKS}"
fi

# --- the two ladders; comment out a line to skip that level ------------------

# p0 -- even projections
echo "=== p0 bin2 START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_p0_bin2.conf" || exit $?
echo "=== p0 bin1 START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_p0_bin1.conf" || exit $?
echo "=== p0 bin0 START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_p0_bin0.conf" || exit $?

# p1 -- odd projections
# echo "=== p1 bin2 START $(date) ==="
# mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_p1_bin2.conf" || exit $?
# echo "=== p1 bin1 START $(date) ==="
# mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_p1_bin1.conf" || exit $?
# echo "=== p1 bin0 START $(date) ==="
# mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_p1_bin0.conf" || exit $?

echo "=== BOTH HALVES DONE $(date) ==="
