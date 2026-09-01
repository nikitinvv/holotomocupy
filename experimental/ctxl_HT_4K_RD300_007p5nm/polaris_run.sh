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
# THE WHOLE PIPELINE IN ONE JOB.
#
#     qsub polaris_run.sh                                # everything
#     qsub -v CONFIG=config_step6_bin1.conf polaris_run.sh   # one step-6 level
#
# With CONFIG set, only that one step-6 level runs (and steps 1-5 are skipped);
# with CONFIG unset, all four stages run back to back in a single allocation.
# Each stage ends in `|| exit $?` so a failure stops the job instead of letting
# the next level seed itself from a checkpoint that was never written.
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
# --- BEFORE THE FIRST FULL RUN ---------------------------------------------
# rotation_center_shift is measured to only +-1.7 px (this scan has exactly one
# exactly-opposed frame pair).  That is 14 nm at a 7.5 nm voxel.  Run steps 1-5
# first, sweep the centre, then start the ladder:
#
#     qsub -v STAGES=steps15 polaris_run.sh
#     python step5_center_sweep.py config_steps15.conf      # on a login node
#     # put the winner in config_steps15.conf AND all three config_step6_bin*.conf
#     qsub -v STAGES=bin2,bin1,bin0 polaris_run.sh
#
# --- WALLTIME --------------------------------------------------------------
# No timings exist for this scan; the closest measured ladder is ../Y350a_HT
# (also 4 distances, ntheta=4000, n=4096) at nobj=4608 on 2 nodes / 8 ranks,
# per 32 iterations:  bin2 ~48 s, bin1 ~170 s, bin0 ~800 s -- 13 min + 22 min +
# 1 h 47 min = ~2 h 25 min of step 6.  This scan's nobj is 5056, 1.10x linearly
# and 1.20x in area, so scale those by roughly 1.2: ~2 h 55 min.  Steps 1-5
# read 537 GB of EDF and write ~2.1 TB back, which is the real unknown; budget
# most of the allocation for it.  18 h should be comfortable for the first
# combined run -- check the timestamps in the .out file and trim it afterwards.
#
# 18 h does not fit the debug queue, hence -q preemptable (1-10 nodes, long
# walltime, evictable).  Preemption is survivable: checkpoint_step=32, so a
# resubmit loses at most 32 iterations -- set STAGES to the levels that have not
# finished and lower that level's start_iter to the last checkpoint on disk.
# Confirm the current queue limits with `qstat -Qf` before changing this.
#
# --- DISK ------------------------------------------------------------------
# In the steps15 path_out (.../ctxl_HT_4K_RD300_007p5nm_0001_rec):
#   <pfile>.h5      4000 x 4096^2 x 2 B x 4 dist  =  537 GB
#   bin-0 pdata     4000 x 4096^2 x 4 B x 4 dist  = 1074 GB
#   <pfile>_obj.h5  5056^3 x 4 B x 2 (re+im)      = 1034 GB
# about 2.7 TB, against 240 TB free on eagle as of 2026-08-31.  The step-6
# path_out (..._rec6) holds the checkpoints: 5056^3 complex64 is ~1 TB each at
# bin 0, so checkpoint_step=32 over 257 iterations is why step6.py keeps only
# the latest few -- watch it anyway on the first bin-0 run.
#
# --- WHAT IS AND IS NOT CORRECTED ------------------------------------------
# Shifts: random displacement + rhapp.mat (inter-plane) + ESRF's
# correct_motion.txt (drift).  No correct_correct3D.txt exists; step 3 uses
# zeros.  Shrinkage: NOT corrected, rho[tp]=0 at every level -- measured from
# this scan's own post-scan retakes as consistent with zero.  See README.md.
# ===========================================================================

# --- user configuration ---
# Software environment (modules + conda env). See the Polaris setup notes.
HTC_ENV=${HTC_ENV:-/eagle/APS_IRI/vvnikitin/sw/env.sh}
# HEALTHCHECK=0  skips the ~30 s GPU probe;  RUN_NODES=N  uses N healthy nodes.
# STAGES=steps15,bin2,bin1,bin0   comma-separated subset to run (default: all).
# CONFIG=config_step6_binK.conf   shorthand for STAGES=binK.
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

# CONFIG=config_step6_bin1.conf is shorthand for STAGES=bin1.
if [ -n "${CONFIG}" ] && [ -z "${STAGES}" ]; then
    STAGES=$(basename "${CONFIG}" .conf | sed 's/^config_step6_//;s/^config_//')
fi
STAGES=${STAGES:-steps15,bin2,bin1,bin0}
echo "Stages: ${STAGES}"
want() { case ",${STAGES}," in *",$1,"*) return 0;; *) return 1;; esac; }

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

run() {   # run <script> <config>
    echo "=== $2 START $(date) ==="
    mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} \
        --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} \
        "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" \
        python "${SCRIPT_DIR}/$1" "${SCRIPT_DIR}/$2" || exit $?
    echo "=== $2 DONE  $(date) ==="
}

# --- the pipeline -----------------------------------------------------------
want steps15 && run steps15.py config_steps15.conf   # EDF->HDF5, preprocess, shifts, binned, Paganin+FBP
want bin2    && run step6.py   config_step6_bin2.conf  # 4x4  n=1024  iters    0 -> 512
want bin1    && run step6.py   config_step6_bin1.conf  # 2x2  n=2048  iters  512 -> 768
want bin0    && run step6.py   config_step6_bin0.conf  # 1x1  n=4096  iters  768 ->1024

echo "=== ALL STAGES DONE $(date) ==="
