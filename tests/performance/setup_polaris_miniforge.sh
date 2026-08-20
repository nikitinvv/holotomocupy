#!/bin/bash
# Build a holotomocupy environment on Polaris.
#
#   bash probe_polaris.sh                    # what is already available
#   bash setup_polaris_miniforge.sh          # build it
#   MODE=forge bash setup_polaris_miniforge.sh
#
# Two routes:
#
#   MODE=site  (default)  A venv with --system-site-packages over the ALCF site
#                         conda.  mpi4py, h5py and cupy are inherited already
#                         built for Cray MPICH / parallel HDF5 / A100, so
#                         nothing heavy is compiled.  Preferred: the login
#                         nodes frequently cannot fork a linker.
#   MODE=forge            A venv (or conda env) over $HOME/miniforge3, with
#                         mpi4py and h5py compiled from source against the
#                         Cray wrappers.  Only for when the site conda is
#                         unusable -- it needs a node that can actually spawn
#                         processes, which usually means an interactive job.
#
# Either way, whatever the base already provides correctly is left alone; only
# what is missing or wrong gets built.  env_polaris.sh finds the result.

set -euo pipefail

MODE=${MODE:-site}
REPO=${REPO:-$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)}
MINIFORGE=${MINIFORGE:-$HOME/miniforge3}
MATHDX_DEFAULT=/eagle/APS_IRI/vnikitin/nvidia/nvidia-mathdx-25.12.1-cuda12/nvidia/mathdx/25.12/

echo "=== repo: $REPO"
echo "=== mode: $MODE"

# --- modules ---------------------------------------------------------------
module use /soft/modulefiles
module load PrgEnv-gnu                  # cc -> Cray MPICH, GNU ABI
module load cray-hdf5-parallel          # parallel HDF5 for the mpio driver
module load cudatoolkit-standalone      # nvcc, for the cuFFTDx propagator

HDF5_PREFIX=${HDF5_ROOT:-${CRAY_HDF5_PARALLEL_PREFIX:-${HDF5_DIR:-}}}
if [ -z "$HDF5_PREFIX" ] || [ ! -d "$HDF5_PREFIX/include" ]; then
    echo "ERROR: cannot locate the cray-hdf5-parallel prefix." >&2
    echo "       Check 'module show cray-hdf5-parallel' and set HDF5_ROOT." >&2
    exit 1
fi
echo "=== HDF5 prefix: $HDF5_PREFIX"

# The Cray PE modules put their libraries on CRAY_LD_LIBRARY_PATH and leave
# LD_LIBRARY_PATH alone, so `cc` links mpi4py against libmpi_gnu_123.so.12 and
# the loader then cannot find it ("cannot open shared object file").  Merge the
# two.  This covers mpich, hdf5, libsci and pmi in one go.
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

HDF5_GCC=$(echo "$HDF5_PREFIX" | sed -nE 's|.*/gnu/([0-9]+\.[0-9]+)/?$|\1|p')
MPICH_ABIS=$(ls -1d /opt/cray/pe/mpich/*/ofi/gnu/*/ 2>/dev/null \
             | sed -nE 's|.*/gnu/([0-9]+\.[0-9]+)/$|\1|p' | sort -u)
echo "=== cray-mpich GNU ABIs installed: $(echo $MPICH_ABIS | tr '\n' ' ')"
if [ -n "$HDF5_GCC" ] && ! echo "$MPICH_ABIS" | grep -qx "$HDF5_GCC"; then
    echo "    WARNING: HDF5 is a gnu/${HDF5_GCC} build but cray-mpich has no such ABI." >&2
fi

# --- resource budget -------------------------------------------------------
# Polaris login nodes cap processes per user (cgroup pids, not RLIMIT_NPROC --
# ulimit cannot raise it).  numpy's OpenBLAS opens one thread per core on
# import, which is enough on its own to exhaust what is left; then collect2
# cannot fork ld and the build dies at the link step.  Build single-threaded.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAKEFLAGS=-j1
export CONDA_FETCH_THREADS=1 CONDA_NUMBER_CHANNEL_NOTICES=0
ulimit -u "$(ulimit -Hu)" 2>/dev/null || true
echo "=== threads in use by $USER: $(ps -u "$USER" -L --no-headers 2>/dev/null | wc -l)"

fork_advice() {
    cat >&2 <<'MSG'

This is not a problem with the build -- the node ran out of process slots.
Two ways forward, in order of preference:

  1. Use MODE=site (the default), which inherits mpi4py/h5py/cupy from the
     ALCF conda instead of compiling them.  Run `bash probe_polaris.sh` to
     confirm the site base has what is needed.

  2. Compile in an interactive job, where nothing competes for pids:
       qsub -I -l select=1:system=polaris -l walltime=01:00:00 \
            -l filesystems=home:eagle -q debug -A 14238
       cd REPO/tests/performance && MODE=forge bash setup_polaris_miniforge.sh

Closing the VS Code remote session and working over plain SSH also frees a
large number of slots -- its server and Python extension run hundreds.
MSG
}

run_build() {   # run_build <description> <command...>
    local what="$1"; shift
    echo "=== $what"
    if ! "$@"; then
        echo "ERROR: $what failed." >&2
        fork_advice
        exit 1
    fi
}

# --- pick the base interpreter and make the venv ---------------------------
SITE_CONDA=$(ls -1dt /soft/applications/conda/*/mconda3 2>/dev/null | head -1)

case "$MODE" in
site)
    if [ ! -x "$SITE_CONDA/bin/python" ]; then
        echo "ERROR: no site conda under /soft/applications/conda/*/mconda3." >&2
        echo "       Retry with MODE=forge." >&2
        exit 1
    fi
    BASE_PY="$SITE_CONDA/bin/python"
    VENV_DIR=${VENV_DIR:-$HOME/venvs/htc-site}
    SYSTEM_SITE=--system-site-packages
    ;;
forge)
    if [ ! -x "$MINIFORGE/bin/python" ]; then
        cat >&2 <<MSG
ERROR: no miniforge at $MINIFORGE.  Install it:
    cd ~
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash Miniforge3-Linux-x86_64.sh -b -p \$HOME/miniforge3
(behind the ALCF proxy: export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128)
MSG
        exit 1
    fi
    BASE_PY="$MINIFORGE/bin/python"
    VENV_DIR=${VENV_DIR:-$HOME/venvs/htc-forge}
    SYSTEM_SITE=
    ;;
*)  echo "ERROR: MODE must be 'site' or 'forge', got '$MODE'." >&2; exit 1 ;;
esac

echo "=== base python: $BASE_PY ($("$BASE_PY" -V 2>&1))"
if [ -f "$VENV_DIR/bin/activate" ]; then
    echo "=== reusing venv $VENV_DIR"
else
    # A venv is a directory copy and a config file -- no solver, no thread
    # pool, no network.  Unlike `conda create` it works on a saturated node.
    run_build "creating venv $VENV_DIR" \
        "$BASE_PY" -m venv $SYSTEM_SITE "$VENV_DIR"
fi
. "$VENV_DIR/bin/activate"
echo "=== python: $(command -v python)  $(python -V 2>&1)"

# --- what is already good? -------------------------------------------------
have_cray_mpi4py() {
    python -c "from mpi4py import MPI; import sys; sys.exit(0 if 'CRAY' in MPI.Get_library_version().upper() else 1)" 2>/dev/null
}
have_parallel_h5py() {
    python -c "import h5py, sys; sys.exit(0 if h5py.get_config().mpi else 1)" 2>/dev/null
}
have_cupy() { python -c "import cupy" 2>/dev/null; }

# In site mode the venv inherits the ALCF conda's site-packages, and those were
# built against the numpy that ships there.  `pip install -U numpy` would
# shadow it with a newer one for every inherited extension -- so only ever
# install a prerequisite that is genuinely absent, never upgrade one.
ensure() {   # ensure <import name> [pip name]
    python -c "import $1" 2>/dev/null && return 0
    run_build "installing ${2:-$1}" pip install --no-cache-dir "${2:-$1}"
}

why_mpi4py() {
    python - 2>&1 <<'DIAG'
try:
    from mpi4py import MPI
except Exception as e:
    print(f"not importable ({type(e).__name__}: {e})")
else:
    v = MPI.Get_library_version().strip().splitlines()[0]
    print(("Cray -- " if "CRAY" in v.upper() else "NOT Cray -- ") + v[:70])
DIAG
}
why_h5py() {
    python - 2>&1 <<'DIAG'
try:
    import h5py
except Exception as e:
    print(f"not importable ({type(e).__name__}: {e})")
else:
    print(f"{h5py.version.version}, mpi={h5py.get_config().mpi}")
DIAG
}

# --- mpi4py ----------------------------------------------------------------
if have_cray_mpi4py; then
    echo "=== mpi4py: already Cray MPICH, leaving it alone"
else
    echo "=== mpi4py must be built -- inherited state: $(why_mpi4py)"
    ensure setuptools; ensure wheel; ensure numpy; ensure Cython cython; ensure pkgconfig
    # `cc -shared`: the Cray wrapper can default to static linking, which
    # produces an unimportable extension.  If this fails, retry with MPICC=cc.
    run_build "building mpi4py against Cray MPICH" \
        env MPICC="cc -shared" pip install --no-cache-dir --force-reinstall \
            --no-binary=mpi4py --no-build-isolation mpi4py
fi

# --- h5py ------------------------------------------------------------------
if have_parallel_h5py; then
    echo "=== h5py: already built with MPI, leaving it alone"
else
    echo "=== h5py must be built -- inherited state: $(why_h5py)"
    ensure numpy; ensure Cython cython; ensure pkgconfig
    # --no-deps, NOT --force-reinstall: h5py built with MPI declares mpi4py as
    # a runtime dependency, and --force-reinstall re-resolves dependencies --
    # which downloads the PyPI mpi4py wheel and overwrites the Cray build with
    # one that speaks TCP instead of libfabric.
    run_build "building h5py with HDF5_MPI=ON" \
        env CC="cc -shared" HDF5_MPI=ON HDF5_DIR="$HDF5_PREFIX" \
            pip install --no-cache-dir --no-deps \
            --no-binary=h5py --no-build-isolation h5py
fi

# --- cupy ------------------------------------------------------------------
if have_cupy; then
    echo "=== cupy: already present, leaving it alone"
else
    # The cuda12x wheel bundles its own CUDA runtime, so it does not break when
    # a cudatoolkit module is bumped.  A100 = sm_80.
    run_build "installing cupy" pip install --no-cache-dir cupy-cuda12x
fi

# --- the rest --------------------------------------------------------------
ensure scipy; ensure tifffile; ensure matplotlib
ensure matplotlib_scalebar matplotlib-scalebar
ensure pandas; ensure psutil; ensure nvtx; ensure dxchange

run_build "installing holotomocupy (editable)" pip install --no-cache-dir --no-deps -e "$REPO"

# Anything above could in principle have pulled a PyPI mpi4py over the Cray
# build.  Cheap to check; expensive to discover from a job running over TCP.
if ! have_cray_mpi4py; then
    if python -c "import mpi4py.MPI" 2>&1 | grep -q "cannot open shared object file"; then
        # Rebuilding cannot fix a loader-path problem; say so instead of
        # burning another compile on a node that struggles to fork one.
        echo "ERROR: mpi4py is built but its MPI library is not on LD_LIBRARY_PATH:" >&2
        python -c "import mpi4py.MPI" 2>&1 | sed 's/^/       /' >&2
        echo "       CRAY_LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH:-<unset>}" >&2
        exit 1
    fi
    run_build "repairing a clobbered mpi4py" \
        env MPICC="cc -shared" pip install --no-cache-dir --force-reinstall \
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
if "CRAY" not in v.upper():
    print("  !! not Cray MPICH -- jobs will run over TCP, not Slingshot"); ok = False

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

cat <<MSG

Environment ready at $VENV_DIR  (env_polaris.sh picks it up automatically).

GPU check needs a compute node -- login nodes have none:
    qsub -I -l select=1:system=polaris -l walltime=00:30:00 \\
         -l filesystems=home:eagle -q debug -A 14238
  then:
    cd $REPO/tests/performance && . ./env_polaris.sh
    python -c 'import cupy; print(cupy.cuda.runtime.getDeviceCount(), cupy.cuda.Device(0).compute_capability)'
    ./run_mosaic_polaris.sh --plan

Set MATHDX_ROOT in env_polaris.sh if mathDX is not at:
    $MATHDX_DEFAULT
MSG
