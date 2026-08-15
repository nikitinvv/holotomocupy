#!/bin/bash
#PBS -A 14347
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

# --- Step 6: full BH reconstruction of the mosaic, coarse to fine ------------
${MPI} "${SCRIPT_DIR}/step6.py" "${CFG}/config_step6_bin3.conf"
# ${MPI} "${SCRIPT_DIR}/step6.py" "${CFG}/config_step6_bin2.conf"
