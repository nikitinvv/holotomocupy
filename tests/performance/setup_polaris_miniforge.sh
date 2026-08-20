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
MPICH_ABIS=$(ls -1d /opt/cray/pe/mpich/*/ofi/gnu/*/ 2>/dev/null \
             | sed -nE 's|.*/gnu/([0-9]+\.[0-9]+)/$|\1|p' | sort -u)
echo "=== cray-mpich GNU ABIs installed: $(echo $MPICH_ABIS | tr '\n' ' ')"
if [ -n "$HDF5_GCC" ]; then
    if [ "$(echo "$MPICH_ABIS" | wc -l)" = "1" ] && [ "$MPICH_ABIS" = "$HDF5_GCC" ]; then
        # Only one MPI ABI exists on the system and HDF5 uses it, so the craype
        # wrapper has nothing else to pick -- mpi4py and h5py land on the same
        # libmpi regardless of which gcc-native is loaded.  Nothing to pin.
        echo "=== single MPI ABI (gnu/${HDF5_GCC}) and HDF5 matches it -- consistent"
    elif ! echo "$MPICH_ABIS" | grep -qx "$HDF5_GCC"; then
        echo "    WARNING: HDF5 is a gnu/${HDF5_GCC} build but cray-mpich has no such ABI." >&2
        echo "             h5py and mpi4py may end up on different libmpi versions." >&2
    fi
fi

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

# Polaris login nodes cap processes/threads per user, and a VS Code remote
# server eats a large share of it.  conda 26.x's sharded-repodata solver opens
# a thread pool per channel and dies with "RuntimeError: can't start new
# thread" when the budget is gone.  Raise the soft limit to the hard one and
# make conda single-threaded.
ulimit -u "$(ulimit -Hu)" 2>/dev/null || true
echo "=== nproc limit: soft=$(ulimit -Su) hard=$(ulimit -Hu); threads in use by $USER: $(ps -u "$USER" -L --no-headers 2>/dev/null | wc -l)"
export CONDA_FETCH_THREADS=1
export CONDA_NUMBER_CHANNEL_NOTICES=0

# Same budget applies to the compile steps.  numpy's OpenBLAS opens one thread
# per core (64 on a Milan login node) the moment it is imported -- h5py's
# build_ext imports numpy -- and the failed pthread_create takes the build
# process down with SIGSEGV.  RLIMIT_NPROC is huge here, so the ceiling is the
# cgroup pid limit, not something ulimit can raise.  Build single-threaded.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MAKEFLAGS=-j1

VENV_DIR=${VENV_DIR:-$HOME/venvs/$ENV_NAME}

if [ -f "$VENV_DIR/bin/activate" ]; then
    echo "=== activating existing venv $VENV_DIR"
    . "$VENV_DIR/bin/activate"
elif conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda activate "$ENV_NAME"
elif [ "${USE_VENV:-0}" = "1" ]; then
    echo "=== creating venv $VENV_DIR from base python ($(python -V 2>&1))"
    mkdir -p "$(dirname "$VENV_DIR")"
    python -m venv "$VENV_DIR"
    . "$VENV_DIR/bin/activate"
else
    echo "=== creating conda env '$ENV_NAME' (python 3.11)"
    # -c conda-forge explicitly: a ~/.condarc with no channels makes a bare
    # `conda create` fail with NoChannelsConfiguredError.
    if ! conda create -y -n "$ENV_NAME" -c conda-forge --override-channels python=3.11; then
        echo "=== libmamba solve failed; retrying with the classic solver"
        if ! CONDA_SOLVER=classic conda create -y -n "$ENV_NAME" \
                 -c conda-forge --override-channels python=3.11; then
            # Both solvers need the thread pool.  A venv over the miniforge base
            # python needs no solver at all, and everything this package uses is
            # pip-installed anyway -- conda was only ever supplying the
            # interpreter.  Base python is 3.13; all deps have cp313 wheels or
            # are built from source here regardless.
            echo "=== conda cannot create an env on this node; falling back to a venv"
            mkdir -p "$(dirname "$VENV_DIR")"
            python -m venv "$VENV_DIR"
            . "$VENV_DIR/bin/activate"
        else
            conda activate "$ENV_NAME"
        fi
    else
        conda activate "$ENV_NAME"
    fi
fi
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
# --no-deps, NOT --force-reinstall: h5py built with MPI declares mpi4py as a
# runtime dependency, and --force-reinstall re-resolves dependencies too --
# which downloads the PyPI mpi4py manylinux wheel and overwrites the Cray build
# from the previous step with one that speaks TCP instead of libfabric.  numpy
# and mpi4py are already installed above, so there is nothing for pip to add.
CC="cc -shared" HDF5_MPI=ON HDF5_DIR="$HDF5_PREFIX" \
    pip install --no-cache-dir --no-deps \
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

# Anything above could in principle have pulled a PyPI mpi4py wheel over the
# Cray build.  Cheap to check, expensive to discover from a job that silently
# runs over TCP.
if ! python -c "from mpi4py import MPI; import sys; sys.exit(0 if 'CRAY' in MPI.Get_library_version().upper() else 1)" 2>/dev/null; then
    echo "=== a PyPI mpi4py replaced the Cray build; rebuilding"
    MPICC="cc -shared" pip install --no-cache-dir --force-reinstall \
        --no-binary=mpi4py --no-build-isolation mpi4py
fi

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
