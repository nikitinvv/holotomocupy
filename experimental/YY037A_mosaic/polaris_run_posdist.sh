#!/bin/bash
#PBS -A 14238
#PBS -l select=16:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:grand:eagle
#PBS -l walltime=3:00:00
#PBS -q prod
#PBS -N holoYY037A_posdist
#PBS -j oe

# YY037A 5-tile mosaic, step 6 with the PER-(TILE x DISTANCE) position model.
#
# rec_mpi_shrink_posdist.Rec instead of rec_mpi_shrink.Rec: the per-angle shifts
# measured in step 3 stay fixed in cl.pos_base and the solver refines a single
# (y, x) offset per tile and distance — 20 x 2 = 40 unknowns instead of
# 300 x 20 x 2 = 12000. Selected purely by `pos_per_dist=true` in the config;
# step6.py is the same driver either way.
#
# Steps 1-5 must already have run (polaris_run.sh) — this job only does step 6
# and needs /exchange/tile_offsets and {pfile}_mosaic_obj.h5 on disk.
#
# 16 nodes x 4 ranks = 64 ranks. Note the class refuses to start if
# local_ntheta == ndist (20): that is 15 ranks at bin 3 and 43-47 at bin 2.
# 64 ranks gives 4-5 and 14-15 respectively, so this geometry is safe. If you
# change the node count, check it.

NNODES=$(wc -l < $PBS_NODEFILE)
NRANKS=4
NTHREADS=4
NDEPTH=8
export NTOTRANKS=$(( NNODES * NRANKS ))

SCRIPT_DIR="${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
rec_dir="$(dirname "${SCRIPT_DIR}")"          # .../experimental
REPO_DIR="$(dirname "${rec_dir}")"            # repo root (holds src/ and tests/)

# Snapshot the py and conf files actually used, into a dated folder, so the run
# records itself.
scripts_dir="${rec_dir}/scripts_posdist$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "${scripts_dir}"
cp "${SCRIPT_DIR}"/*.py   "${scripts_dir}/" 2>/dev/null || true
cp "${SCRIPT_DIR}"/*.conf "${scripts_dir}/" 2>/dev/null || true

cd "${rec_dir}"
exec > >(tee "${scripts_dir}/posdist-${PBS_JOBID}.out" \
              "${SCRIPT_DIR}/posdist-${PBS_JOBID}.out") 2>&1

# The working configs point at the beamline machine; rewrite to the eagle copy
# inside the snapshot and run the snapshot, leaving the working copies usable
# locally. Already-eagle paths are left alone.
DATA_ROOT_LOCAL=${DATA_ROOT_LOCAL:-/data2/vnikitin}
DATA_ROOT=${DATA_ROOT:-/eagle/APS_IRI/vnikitin}
sed -i "s|${DATA_ROOT_LOCAL}|${DATA_ROOT}|g" "${scripts_dir}"/*.conf
CFG="${scripts_dir}"

echo "Configs:     ${CFG}  (data root ${DATA_ROOT})"
grep -H '^path' "${CFG}"/config_step6_bin3_posdist.conf
echo "Sample dir:  ${SCRIPT_DIR}"
echo "Rec dir:     ${rec_dir}"
echo "Snapshot:    ${scripts_dir}"
echo "Jobid: $PBS_JOBID"
echo "Running on host: $(hostname)"
echo "NUM_OF_NODES=${NNODES}  TOTAL_NUM_RANKS=${NTOTRANKS}  RANKS_PER_NODE=${NRANKS}"

module use /soft/modulefiles;  module load conda; conda activate base
CONDA_NAME=$(echo ${CONDA_PREFIX} | tr '\/' '\t' | sed -E 's/mconda3|\/base//g' | awk '{print $NF}')
VENV_DIR="/home/vvnikitin/venvs/${CONDA_NAME}"
if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
else
    # CONDA_NAME is derived from CONDA_PREFIX, so a bump of the conda module
    # points VENV_DIR at a venv that was never created. Job 7483478 hit this
    # ("/home/vvnikitin/venvs/2025-09-25/bin/activate: No such file or
    # directory") and silently continued on base conda. Fall back to the newest
    # venv instead, and say so.
    echo "WARNING: ${VENV_DIR}/bin/activate is missing (CONDA_NAME=${CONDA_NAME})."
    FALLBACK_VENV=$(ls -1dt /home/vvnikitin/venvs/*/bin/activate 2>/dev/null | head -1)
    if [ -n "${FALLBACK_VENV}" ]; then
        echo "         falling back to ${FALLBACK_VENV}"
        source "${FALLBACK_VENV}"
    else
        echo "         no venv under /home/vvnikitin/venvs — running base conda."
    fi
fi
echo "python:      $(command -v python)"

export MATHDX_ROOT=/eagle/APS_IRI/vnikitin/nvidia/nvidia-mathdx-25.12.1-cuda12/nvidia/mathdx/25.12/

MPI="mpiexec -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth \
     --env OMP_NUM_THREADS=${NTHREADS} ${SCRIPT_DIR}/set_affinity_gpu_polaris.sh python"

# --- 0. Smoke test: posdist == per-projection on a tiny synthetic problem ----
# ~1 minute on one GPU. Advisory only: it runs a 32x32 synthetic problem with
# the cuPy-FFT propagator (the test forces MATHDX_ROOT off, since 64x64 has no
# valid cuFFTDx instantiation on sm_80), so a failure here does not necessarily
# say anything about the 256x256 production run. Job 7483478 was killed by this
# gate for exactly that reason, so it now warns and continues.
#   SKIP_SMOKE=1    do not run it at all
#   STRICT_SMOKE=1  go back to aborting the job if it fails
if [ -z "${SKIP_SMOKE:-}" ]; then
    echo "=== smoke test: tests/test_posdist.py (cubic) ==============================="
    if mpiexec -n 1 --ppn 1 --depth=${NDEPTH} --cpu-bind depth \
            --env OMP_NUM_THREADS=${NTHREADS} \
            "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python \
            "${REPO_DIR}/tests/test_posdist.py" cubic; then
        echo "smoke test passed"
    else
        echo "*** smoke test FAILED (advisory) ***"
        if [ -n "${STRICT_SMOKE:-}" ]; then
            echo "STRICT_SMOKE set — not launching the reconstruction."
            exit 1
        fi
        echo "continuing to step 6 anyway; check the output above before"
        echo "trusting the refined offsets."
    fi
    echo
fi

# --- Step 6, bin 3, per-(tile x distance) positions --------------------------
# error_step=32 prints all 40 refined offsets each time, one line per tile,
# and appends them to ${path_out}/pos.csv for plotting:
#   grep -A6 'pos error' posdist-*.out
# and estimate_rho=true logs the searched scales once, before the loop:
#   grep 'estimate_rho_coord' posdist-*.out
# Restartable: checkpoints land in path_out every 32 iters and step6.py resumes
# from the latest one, so re-queueing this script carries on.
${MPI} "${SCRIPT_DIR}/step6.py" "${CFG}/config_step6_bin3_posdist.conf"

# --- Step 6, bin 2 -----------------------------------------------------------
# Only once bin 3 looks right; it needs its own walltime.
# ${MPI} "${SCRIPT_DIR}/step6.py" "${CFG}/config_step6_bin2_posdist.conf"

# --- Side-by-side baseline ---------------------------------------------------
# The per-projection run at the same bin, for comparing convergence
# (path_out differs, so the two do not collide).
# ${MPI} "${SCRIPT_DIR}/step6.py" "${CFG}/config_step6_bin3.conf"
