#!/bin/bash
# One-time build of a miniforge-based holotomocupy environment on Polaris.
#
#   bash setup_polaris_miniforge.sh          # from a LOGIN node
#
# Run it once; afterwards every job sources env_polaris.sh instead.
#
# Design rule: conda supplies python and nothing else.  mpi4py and h5py are
# compiled from source against Cray MPICH / cray-hdf5-parallel with the `cc`
# wrapper, because the package uses Alltoallw with derived datatypes
# (mpi_functions.py) and opens HDF5 with driver="mpio" (reader.py, writer.py).
# A conda-forge mpich or an mpi_mpich h5py build would silently substitute a
# different MPI and take the run off Slingshot.

set -euo pipefail

ENV_NAME=${ENV_NAME:-myenv}
MINIFORGE=${MINIFORGE:-$HOME/miniforge3}
REPO=${REPO:-$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)}
MATHDX_DEFAULT=/eagle/APS_IRI/vnikitin/nvidia/nvidia-mathdx-25.12.1-cuda12/nvidia/mathdx/25.12/

echo "=== repo: $REPO"

# --- modules ---------------------------------------------------------------
module use /soft/modulefiles
module load PrgEnv-gnu                  # cc -> Cray MPICH, GNU ABI
module load cray-hdf5-parallel          # parallel HDF5 for the mpio driver
module load cudatoolkit-standalone      # nvcc, for the cuFFTDx propagator
module list 2>&1 | sed 's/^/    /'

# cray-hdf5-parallel exports its prefix under one of several names.
HDF5_PREFIX=${HDF5_ROOT:-${CRAY_HDF5_PARALLEL_PREFIX:-${HDF5_DIR:-}}}
if [ -z "$HDF5_PREFIX" ] || [ ! -d "$HDF5_PREFIX/include" ]; then
    echo "ERROR: cannot locate the cray-hdf5-parallel prefix." >&2
    echo "       Check 'module show cray-hdf5-parallel' and set HDF5_ROOT." >&2
    exit 1
fi
echo "=== HDF5 prefix: $HDF5_PREFIX"

# --- keep one GCC/MPI ABI across mpi4py, h5py and cray-mpich ---------------
# cray-hdf5-parallel ships a build per GCC ABI and its prefix ends in the one
# it was compiled with (.../gnu/12.3).  Lmod's default gcc-native may be newer;
# then libhdf5.so wants libmpi_gnu_123.so.12 while `cc` links libmpi_gnu_<new>,
# and the rank loads two MPI libraries.  Pin GCC to the HDF5 build if we can.
HDF5_GCC=$(echo "$HDF5_PREFIX" | sed -nE 's|.*/gnu/([0-9]+\.[0-9]+)/?$|\1|p')
CUR_GCC=$(module -t --redirect list 2>/dev/null | sed -nE 's|^gcc-native/(.*)$|\1|p')
if [ -n "$HDF5_GCC" ] && [ "${CUR_GCC%%.*}" != "${HDF5_GCC%%.*}" ]; then
    echo "=== gcc-native/${CUR_GCC:-?} loaded but HDF5 is a gnu/${HDF5_GCC} build; pinning GCC"
    if   module load "gcc-native/${HDF5_GCC}"      2>/dev/null; then :
    elif module load "gcc-native/${HDF5_GCC%%.*}"  2>/dev/null; then :
    else
        echo "    WARNING: no gcc-native/${HDF5_GCC} module.  Continuing on gcc-native/${CUR_GCC}." >&2
        echo "             If ranks later fail on 'libmpi_gnu_*.so: cannot open shared" >&2
        echo "             object file', env_polaris.sh repairs it via LD_LIBRARY_PATH." >&2
    fi
fi
echo "=== cray-mpich ABI dirs available:"
ls -1d /opt/cray/pe/mpich/*/ofi/gnu/*/lib 2>/dev/null | sed 's/^/    /' || echo "    (none found)"

# --- conda env -------------------------------------------------------------
# Locate a conda: $MINIFORGE, then the usual install spots, then one already
# on PATH (e.g. an interactive shell that has run `conda activate` by hand).
CONDA_SH=""
for cand in "$MINIFORGE" "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/mambaforge" "$HOME/anaconda3"; do
    if [ -f "$cand/etc/profile.d/conda.sh" ]; then CONDA_SH="$cand/etc/profile.d/conda.sh"; break; fi
done
if [ -z "$CONDA_SH" ] && command -v conda >/dev/null 2>&1; then
    CONDA_SH="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
    [ -f "$CONDA_SH" ] || CONDA_SH=""
fi
if [ -z "$CONDA_SH" ]; then
    cat >&2 <<'MSG'
ERROR: no conda installation found.

Install miniforge first (from a login node, takes ~2 min):

    cd ~
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3
    $HOME/miniforge3/bin/conda create -y -n myenv python=3.11

then re-run this script.  If conda is somewhere else, point at it:

    MINIFORGE=/path/to/miniforge3 bash setup_polaris_miniforge.sh
MSG
    exit 1
fi
echo "=== conda: $CONDA_SH"
. "$CONDA_SH"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "=== creating conda env '$ENV_NAME' (python 3.11)"
    conda create -y -n "$ENV_NAME" python=3.11
fi
conda activate "$ENV_NAME"
echo "=== python: $(command -v python)  $(python -V 2>&1)"

# Nothing below uses `conda install`.  Adding conda-forge mpich, hdf5 or
# openmpi here is the one thing that will break this environment.

# --- build prerequisites ---------------------------------------------------
pip install --no-cache-dir -U pip setuptools wheel
pip install --no-cache-dir "numpy<3" Cython pkgconfig

# --- mpi4py against Cray MPICH --------------------------------------------
# `cc -shared`: the Cray wrapper can default to static linking, which produces
# an unimportable extension.  If this fails, retry with plain MPICC=cc.
echo "=== building mpi4py against Cray MPICH"
MPICC="cc -shared" pip install --no-cache-dir --force-reinstall \
    --no-binary=mpi4py --no-build-isolation mpi4py

# --- h5py with parallel HDF5 ----------------------------------------------
echo "=== building h5py with HDF5_MPI=ON"
CC="cc -shared" HDF5_MPI=ON HDF5_DIR="$HDF5_PREFIX" \
    pip install --no-cache-dir --force-reinstall \
    --no-binary=h5py --no-build-isolation h5py

# --- cupy ------------------------------------------------------------------
# The cuda12x wheel bundles its own CUDA runtime libs, so it does not depend on
# whichever cudatoolkit module happens to be loaded at run time.  A100 = sm_80.
echo "=== installing cupy"
pip install --no-cache-dir cupy-cuda12x

# --- the rest --------------------------------------------------------------
pip install --no-cache-dir \
    scipy tifffile matplotlib matplotlib-scalebar pandas psutil nvtx dxchange

pip install --no-cache-dir -e "$REPO"

# --- verify ----------------------------------------------------------------
echo
echo "=== verification (login node) ==="
python - <<'PY'
import sys
ok = True

from mpi4py import MPI
v = MPI.Get_library_version().strip().splitlines()[0]
print(f"mpi4py  : {v}")
if "CRAY" not in v.upper() and "cray" not in v:
    print("  !! not Cray MPICH -- rebuild mpi4py with MPICC=cc"); ok = False

import h5py
mpi = h5py.get_config().mpi
print(f"h5py    : {h5py.version.version}, hdf5 {h5py.version.hdf5_version}, mpi={mpi}")
if not mpi:
    print("  !! built without MPI -- driver='mpio' will fail"); ok = False

import cupy
print(f"cupy    : {cupy.__version__}, CUDA runtime {cupy.cuda.runtime.runtimeGetVersion()}")

import holotomocupy
print(f"package : {holotomocupy.__file__}")

sys.exit(0 if ok else 1)
PY

echo
echo "GPU check must run on a compute node -- login nodes have no GPU:"
echo "    qsub -I -l select=1:system=polaris -l walltime=00:30:00 \\"
echo "         -l filesystems=home:eagle -q debug -A 14238"
echo "  then, in the interactive shell:"
echo "    cd $REPO/tests/performance && . ./env_polaris.sh"
echo "    python -c 'import cupy; print(cupy.cuda.runtime.getDeviceCount(), cupy.cuda.Device(0).compute_capability)'"
echo "    ./run_mosaic_polaris.sh --plan"
echo
echo "Set MATHDX_ROOT in env_polaris.sh if mathDX is not at:"
echo "    $MATHDX_DEFAULT"
