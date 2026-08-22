#!/bin/bash
# Software environment for holotomocupy on ALCF Polaris.
#
#   source /eagle/APS_IRI/vvnikitin/sw/env.sh
#
# Must be sourced INSIDE the PBS job, not just at install time: batch jobs do
# not read ~/.bashrc, and the cray-mpich-linked mpi4py / h5py need the module
# LD_LIBRARY_PATH at run time, not only at build time.
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

# --- conda LAST ----------------------------------------------------------
# `conda activate` is a shell function defined by this hook; the condabin/conda
# binary on its own cannot activate anything.
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${HTC_PREFIX}"   # by full path: sw/envs is not conda's default envs_dir

# --- sanity --------------------------------------------------------------
# Uncomment while debugging a fresh install.
# echo "python : $(which python)"
# python -c "import mpi4py, h5py, cupy; print('mpi4py', mpi4py.__version__, '| h5py', h5py.__version__, h5py.get_config().mpi, '| cupy', cupy.__version__)"
