#!/bin/bash
#PBS -A 14238
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:eagle
#PBS -l walltime=18:00:00
#PBS -q preemptable
#PBS -N ctxl0075
#PBS -j oe
# ===========================================================================
# ctxl cortex tissue, 4-distance HT, +-300 px random displacement, 7.5 nm
# voxels -- ESRF ID16A 2026-08-29..31, proposal ihls3888.
# THE WHOLE PIPELINE IN ONE JOB:
#
#     qsub polaris_run.sh
#
# THIS IS THE tomo_upsample=2 ARM: the object x/y grid is half the projection
# plane (632/1264/2528 against 1264/2528/5056), and it writes ..._rec6_u2.  The
# historical tomo_upsample=1 arm is polaris_run_u1.sh -> ..._rec6_u1.  The two
# are
# run side by side and compared; they must not share an output directory,
# because their checkpoints have incompatible object shapes and share the
# iteration numbering.
#
# Steps 1-5 are shared by the two arms and are NOT affected by tomo_upsample:
# step 5 writes its Paganin+FBP init on the projection grid as it always has,
# and Reader.read_obj averages it 2x2 in x/y when a step-6 config asks for
# tomo_upsample=2.  So steps15 does not have to be re-run for either arm, and
# polaris_run_u1.sh has its steps15 line commented out.
#
# To run only part of it -- steps 1-5 already done, or resuming after a
# preemption -- COMMENT OUT the mpiexec lines at the bottom that you do not
# want.  Each line ends in `|| exit $?` so a failed stage stops the job
# instead of letting the next level seed itself from a checkpoint that was
# never written.
#
# What each stage does, the rotation-centre sweep to run before the ladder,
# how to resume, walltime and disk: see README.md, "Running it on Polaris".
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

# What step 3 is about to read -- shift files, row counts, bin factors, the
# resulting cshifts_final.  Pure numpy on rank 0's inputs, a few seconds, no
# GPU; it only prints.  Run it first so a missing or wrongly-scaled shift file
# shows up here rather than 20 minutes into steps15.
python "${rec_dir}/check_data_read.py" "${SCRIPT_DIR}/config_steps15.conf" || exit $?

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

# --- the pipeline; comment out a line to skip that stage ---------------------

# EDF->HDF5, preprocess, shifts, binned data, Paganin+FBP for bins 2,1,0
echo "=== steps15 START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/steps15.py" "${SCRIPT_DIR}/config_steps15.conf" || exit $?

# bin 2: 4x4  n=1024   (iteration range: start_iter/niter in the config)
echo "=== bin2 START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_bin2.conf" || exit $?

# bin 1: 2x2  n=2048   resumes from the checkpoint the bin-2 run left
echo "=== bin1 START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_bin1.conf" || exit $?

# bin 0: 1x1  n=4096   resumes from the checkpoint the bin-1 run left
echo "=== bin0 START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_bin0.conf" || exit $?

echo "=== ALL STAGES DONE $(date) ==="
