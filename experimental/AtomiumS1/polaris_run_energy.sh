#!/bin/bash
#PBS -A 14238
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:eagle
#PBS -l walltime=12:00:00
#PBS -q preemptable
#PBS -N AtomS1en
#PBS -j oe
# ===========================================================================
# Atomium S1 FT -- ENERGY SWEEP at bin 1.  17 energies, 16.900 .. 17.300 keV
# in 0.025 steps, 128 iterations each from the same finished bin-1 checkpoint.
#
#     qsub polaris_run_energy.sh
#
# ~29 min per energy on 2 nodes (26 min of iterations plus ~3 min to read the
# 68 GB seed), so ~8.2 h for all 17; the 12 h walltime is slack, not an
# estimate.  To run a subset -- three shorter jobs instead of one long one, or
# a resubmission after a preemption -- comment out the mpiexec lines you do not
# want.  The energies are independent: same read-only seed and data, one
# path_out each.  Nothing is checkpointed, so a preempted energy restarts from
# iteration 1280; the ones that finished are done.
#
# The configs are GENERATED -- edit make_energy_configs.py, not the .conf
# files.  What the sweep is for and how to read the result: README.md,
# "The energy sweep".  Polaris setup: README.md, "Running it on Polaris".
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
# falls back to the script's own directory for local ./polaris_run_energy.sh
# testing).  Plain $(pwd) does NOT work: PBS starts the job in $HOME.
SCRIPT_DIR="${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
rec_dir="$(dirname "${SCRIPT_DIR}")"

cd "${rec_dir}"
exec > >(tee "${SCRIPT_DIR}/energy-${PBS_JOBID}.out") 2>&1

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

# The seed every energy starts from.  Checked once, here, rather than 17 times
# half an hour apart: every config points at it, and if the bin-0 stage ever
# overwrites or prunes the bin-1 checkpoints this is the line that says so.
SEED=/eagle/APS_IRI/vnikitin/20260829/AtomiumS1/Atomium_S1_FT_4K_RD300_004p5nm_0001_rec6/checkpoints/checkpoint_1280.h5
[ -r "${SEED}" ] || { echo "ERROR: seed checkpoint not readable: ${SEED}"; exit 1; }
echo "Seed checkpoint: ${SEED}  ($(stat -c %s "${SEED}") bytes)"

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

# --- the sweep; comment out a line to skip that energy ----------------------

echo "=== E=16.900 keV START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6_energy.py" "${SCRIPT_DIR}/config_energy_bin1_e16p900.conf" || exit $?

echo "=== E=16.925 keV START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6_energy.py" "${SCRIPT_DIR}/config_energy_bin1_e16p925.conf" || exit $?

echo "=== E=16.950 keV START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6_energy.py" "${SCRIPT_DIR}/config_energy_bin1_e16p950.conf" || exit $?

echo "=== E=16.975 keV START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6_energy.py" "${SCRIPT_DIR}/config_energy_bin1_e16p975.conf" || exit $?

echo "=== E=17.000 keV START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6_energy.py" "${SCRIPT_DIR}/config_energy_bin1_e17p000.conf" || exit $?

echo "=== E=17.025 keV START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6_energy.py" "${SCRIPT_DIR}/config_energy_bin1_e17p025.conf" || exit $?

echo "=== E=17.050 keV START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6_energy.py" "${SCRIPT_DIR}/config_energy_bin1_e17p050.conf" || exit $?

echo "=== E=17.075 keV START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6_energy.py" "${SCRIPT_DIR}/config_energy_bin1_e17p075.conf" || exit $?

echo "=== E=17.100 keV START $(date) ==="   # <- nominal, the control
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6_energy.py" "${SCRIPT_DIR}/config_energy_bin1_e17p100.conf" || exit $?

echo "=== E=17.125 keV START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6_energy.py" "${SCRIPT_DIR}/config_energy_bin1_e17p125.conf" || exit $?

echo "=== E=17.150 keV START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6_energy.py" "${SCRIPT_DIR}/config_energy_bin1_e17p150.conf" || exit $?

echo "=== E=17.175 keV START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6_energy.py" "${SCRIPT_DIR}/config_energy_bin1_e17p175.conf" || exit $?

echo "=== E=17.200 keV START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6_energy.py" "${SCRIPT_DIR}/config_energy_bin1_e17p200.conf" || exit $?

echo "=== E=17.225 keV START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6_energy.py" "${SCRIPT_DIR}/config_energy_bin1_e17p225.conf" || exit $?

echo "=== E=17.250 keV START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6_energy.py" "${SCRIPT_DIR}/config_energy_bin1_e17p250.conf" || exit $?

echo "=== E=17.275 keV START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6_energy.py" "${SCRIPT_DIR}/config_energy_bin1_e17p275.conf" || exit $?

echo "=== E=17.300 keV START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step6_energy.py" "${SCRIPT_DIR}/config_energy_bin1_e17p300.conf" || exit $?

echo "=== SWEEP DONE $(date) ==="

# Every residual, side by side, in energy order.  Pure text, no GPU -- also runs
# fine on a login node afterwards if the job was preempted before reaching this
# line.  The minimum over the final column is the answer; the seed column is
# the same test with nothing yet refitted, and is the more sensitive of the two.
base=/eagle/APS_IRI/vnikitin/20260829/AtomiumS1/Atomium_S1_FT_4K_RD300_004p5nm_0001_rec6
printf "%-9s %-14s %-14s %s\n" energy seed_err final_err iter
for E in 16p900 16p925 16p950 16p975 17p000 17p025 17p050 17p075 17p100 17p125 17p150 17p175 17p200 17p225 17p250 17p275 17p300 ; do
    f="${base}_E${E}/conv.csv"
    [ -r "$f" ] || { printf "%-9s %s\n" "${E/p/.}" "no conv.csv"; continue; }
    printf "%-9s %-14s %-14s %s\n" "${E/p/.}" \
        "$(awk -F, '$1==-1{print $2}' "$f")" \
        "$(tail -1 "$f" | cut -d, -f2)" "$(tail -1 "$f" | cut -d, -f1)"
done
