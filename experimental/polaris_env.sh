#!/bin/bash
# Software environment for holotomocupy on ALCF Polaris.
#
#   source /eagle/APS_IRI/vvnikitin/sw/env.sh
#
# Must be sourced INSIDE the PBS job, not just at install time: batch jobs do
# not read ~/.bashrc, and the cray-mpich-linked mpi4py / h5py need the module
# library paths at run time, not only at build time.
#
# Copy this file to /eagle/APS_IRI/vvnikitin/sw/env.sh -- that is the path
# HTC_ENV in polaris_run.sh defaults to.

CONDA_ROOT=/lus/eagle/projects/APS_IRI/vvnikitin/sw/miniforge3
HTC_PREFIX=/lus/eagle/projects/APS_IRI/vvnikitin/sw/envs/htc

# --- modules FIRST -------------------------------------------------------
# conda activate prepends the env to PATH, so it must come last or the env's
# own binaries lose to the module ones.
module use /soft/modulefiles
module load PrgEnv-gnu
module load cray-mpich
module load cudatoolkit-standalone
module load cray-hdf5-parallel

# --- the line everything hinges on ---------------------------------------
# The Cray PE modules populate CRAY_LD_LIBRARY_PATH, NOT LD_LIBRARY_PATH.
# The cc/ftn compiler wrappers consult it at link time, but a bare `python`
# process does not, so mpi4py.MPI.so fails to resolve libmpi.so.12 and h5py
# fails to resolve libhdf5_parallel_gnu_*.so at import.  Export it explicitly.
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
# Belt and braces: some cray-mpich versions leave libmpi.so.12 only here.
[ -n "${CRAY_MPICH_DIR}" ] && export LD_LIBRARY_PATH="${CRAY_MPICH_DIR}/lib:${LD_LIBRARY_PATH}"

# GPU-aware MPI (needs libmpi_gtl_cuda, pulled in by craype-accel-nvidia80).
# Only enable it if that module loads -- it refuses unless a cuda module is
# already loaded, which is why cudatoolkit-standalone comes before it.
if module load craype-accel-nvidia80 2>/dev/null; then
    export MPICH_GPU_SUPPORT_ENABLED=1
    export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
else
    export MPICH_GPU_SUPPORT_ENABLED=0
fi

# --- conda LAST ----------------------------------------------------------
# `conda activate` is a shell function defined by this hook; the condabin/conda
# binary on its own cannot activate anything.
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${HTC_PREFIX}"   # by full path: sw/envs is not conda's default envs_dir

# --- sanity --------------------------------------------------------------
# Set HTC_ENV_CHECK=1 to print what actually resolved.
if [ "${HTC_ENV_CHECK:-0}" = "1" ]; then
    echo "python  : $(which python)"
    echo "libmpi  : $(ldd "$(python -c 'import mpi4py.MPI as m; print(m.__file__)')" 2>/dev/null | grep -i 'libmpi\.' || echo 'mpi4py did not import')"
    python -c "import mpi4py, h5py, cupy; print('mpi4py', mpi4py.__version__, '| h5py', h5py.__version__, 'mpi=', h5py.get_config().mpi, '| cupy', cupy.__version__)"
fi
