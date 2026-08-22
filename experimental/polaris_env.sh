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

# HOW THIS ENV MUST BE BUILT
# --------------------------
# mpi4py and h5py must be pip-compiled against the Cray wrappers, NOT installed
# from conda-forge or a PyPI wheel:
#
#   module unload darshan          # see below -- required, not optional
#   conda remove --force -y mpi4py mpich openmpi mpi
#   MPICC=cc pip install --no-cache-dir --no-binary=mpi4py --no-build-isolation mpi4py
#   HDF5_MPI=ON CC=cc pip install --no-cache-dir --no-binary=h5py --no-build-isolation h5py
#
# Why: cray-mpich 9.1.0 ships libmpi_gnu.so.12 and an unversioned libmpi.so,
# but NO libmpi.so.12.  conda-forge mpi4py and the PyPI wheel are both linked
# against stock MPICH's libmpi.so.12, so they can never resolve here no matter
# what LD_LIBRARY_PATH says.  A `cc`-built mpi4py links -lmpi -> libmpi.so ->
# libmpi_gnu.so.12, which exists.  Symptom of getting this wrong:
#   ImportError: libmpi.so.12: cannot open shared object file
# and two variant files MPI.mpich.*.so / MPI.openmpi.*.so instead of one MPI.*.so.

CONDA_ROOT=/lus/eagle/projects/APS_IRI/vvnikitin/sw/miniforge3
HTC_PREFIX=/lus/eagle/projects/APS_IRI/vvnikitin/sw/envs/htc

# --- modules FIRST -------------------------------------------------------
# conda activate prepends the env to PATH, so it must come last or the env's
# own binaries lose to the module ones.
module use /soft/modulefiles
module load PrgEnv-gnu
module load cray-mpich
# Pin the version: the site default is 12.2.2, but cupy in this env is built
# for CUDA 13, so an unversioned load silently gives the wrong toolkit.
module load cudatoolkit-standalone/13.0.1
module load cray-hdf5-parallel
# Darshan MUST be unloaded.  The cc wrapper silently injects libdarshan.so.0
# into everything it links, and the installed Darshan build carries a NEEDED
# entry for libmpi_gnu_123.so.12 -- a cray-mpich version that no longer exists
# on the system.  The result is that a correctly-built mpi4py still dies with
#   ImportError: libmpi_gnu_123.so.12: cannot open shared object file
# even though its own libmpi_gnu.so.12 resolves fine.  Unload before BOTH the
# pip build and every job: the bad NEEDED entry is baked in at link time, so a
# module unloaded only at run time does not help an already-linked extension.
module unload darshan 2>/dev/null || true

# --- the line everything hinges on ---------------------------------------
# The Cray PE modules populate CRAY_LD_LIBRARY_PATH, NOT LD_LIBRARY_PATH.
# The cc/ftn compiler wrappers consult it at link time, but a bare `python`
# process does not, so mpi4py.MPI.so fails to resolve libmpi.so.12 and h5py
# fails to resolve libhdf5_parallel_gnu_*.so at import.  Export it explicitly.
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
# Belt and braces: some cray-mpich versions leave libmpi.so.12 only here.
[ -n "${CRAY_MPICH_DIR}" ] && export LD_LIBRARY_PATH="${CRAY_MPICH_DIR}/lib:${LD_LIBRARY_PATH}"

# GPU-aware MPI is deliberately OFF, and craype-accel-nvidia80 is not loaded.
# holotomocupy never hands a device pointer to MPI: the only Alltoallw buffer is
# rec_mpi.py:proj_tmp, allocated by utils.make_pinned -> cp.cuda.alloc_pinned_memory,
# i.e. page-locked HOST memory wrapped in a numpy array; the allreduces either
# take pinned host arrays or an explicit .get() copy.  So the CUDA GTL layer
# would add a dependency without ever being used.  If a future change starts
# passing cupy arrays to comm.*, load craype-accel-nvidia80 (after the cuda
# module -- it refuses otherwise), re-export CRAY_LD_LIBRARY_PATH for
# libmpi_gtl_cuda, and set this to 1.
export MPICH_GPU_SUPPORT_ENABLED=0

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
    # Import mpi4py.MPI, not just mpi4py: the bare package is pure Python and
    # imports even when the compiled extension cannot find its libmpi.
    python -c "from mpi4py import MPI; import mpi4py, h5py, cupy; print('mpi4py', mpi4py.__version__, MPI.Get_library_version().split(chr(10))[0], '| h5py', h5py.__version__, 'mpi=', h5py.get_config().mpi, '| cupy', cupy.__version__)"
fi
