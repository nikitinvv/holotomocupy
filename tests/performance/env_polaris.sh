# Environment for the Polaris jobs -- sourced by run_polaris.sh and
# run_mosaic_polaris.sh, so the ranks never depend on ~/.bashrc.
#
#   . ./env_polaris.sh
#
# Miniforge variant: the environment is built once by setup_polaris_miniforge.sh
# and this file only reactivates it.  The site `module load conda` route is not
# used here (its modulefile was broken by the August 2026 maintenance -- see the
# patched-modulefile note in the git history of this file).

set +eu                                  # conda/module scripts trip -u

ENV_NAME=${ENV_NAME:-myenv}
MINIFORGE=${MINIFORGE:-$HOME/miniforge3}

module use /soft/modulefiles
# PrgEnv-gnu must match what mpi4py was compiled against: it puts the
# libmpi_gnu_*.so that the extension is linked to back on the library path.
module load PrgEnv-gnu
module load cray-hdf5-parallel
module load cudatoolkit-standalone       # nvcc for the cuFFTDx JIT

# Same GCC pin as setup_polaris_miniforge.sh: the cray-hdf5-parallel prefix
# ends in the GCC ABI it was built for, and mpi4py/h5py were compiled under
# that ABI.  Lmod's default gcc-native may be newer, which moves the cray-mpich
# lib dir out from under them.
HDF5_PREFIX=${HDF5_ROOT:-${CRAY_HDF5_PARALLEL_PREFIX:-${HDF5_DIR:-}}}
# Only worth pinning when the system actually offers more than one MPI ABI to
# choose between; today Polaris ships gnu/12.3 alone, so this is a no-op.
HDF5_GCC=$(echo "${HDF5_PREFIX}" | sed -nE 's|.*/gnu/([0-9]+\.[0-9]+)/?$|\1|p')
MPICH_ABIS=$(ls -1d /opt/cray/pe/mpich/*/ofi/gnu/*/ 2>/dev/null \
             | sed -nE 's|.*/gnu/([0-9]+\.[0-9]+)/$|\1|p' | sort -u)
CUR_GCC=$(module -t --redirect list 2>/dev/null | sed -nE 's|^gcc-native/(.*)$|\1|p')
if [ -n "${HDF5_GCC}" ] && [ "$(echo "${MPICH_ABIS}" | wc -l)" -gt 1 ] \
   && [ "${CUR_GCC%%.*}" != "${HDF5_GCC%%.*}" ]; then
    module load "gcc-native/${HDF5_GCC}" 2>/dev/null \
        || module load "gcc-native/${HDF5_GCC%%.*}" 2>/dev/null \
        || echo "NOTE: no gcc-native/${HDF5_GCC}; relying on the ABI repair below."
fi

# Prefer a venv built by setup_polaris_miniforge.sh.  htc-site (over the ALCF
# conda, inheriting its Cray mpi4py / parallel h5py / cupy) before htc-forge
# (miniforge with those compiled locally); an explicit VENV_DIR beats both.
for cand in ${VENV_DIR:-} "$HOME/venvs/htc-site" "$HOME/venvs/htc-forge" "$HOME/venvs/${ENV_NAME}"; do
    if [ -n "${cand}" ] && [ -f "${cand}/bin/activate" ]; then
        . "${cand}/bin/activate"
        HTC_ENV_KIND="venv ${cand}"
        break
    fi
done

# Locate conda: $MINIFORGE, then the usual install spots, then one on PATH.
CONDA_SH=""
for cand in "${MINIFORGE}" "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/mambaforge" "$HOME/anaconda3"; do
    if [ -f "${cand}/etc/profile.d/conda.sh" ]; then CONDA_SH="${cand}/etc/profile.d/conda.sh"; break; fi
done
if [ -z "${CONDA_SH}" ] && command -v conda >/dev/null 2>&1; then
    CONDA_SH="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
    [ -f "${CONDA_SH}" ] || CONDA_SH=""
fi
if [ -z "${HTC_ENV_KIND:-}" ]; then
    if [ -z "${CONDA_SH}" ]; then
        echo "ERROR: no venv at ${VENV_DIR} and no conda found;" >&2
        echo "       run setup_polaris_miniforge.sh first." >&2
        exit 1
    fi
    . "${CONDA_SH}"
    conda activate "${ENV_NAME}"
    HTC_ENV_KIND="conda ${ENV_NAME}"
fi

# mpi4py ABI check: PrgEnv/gcc-native bumps move the cray-mpich lib dir and the
# extension then fails to load at rank start, deep inside the job.  Catch it here.
MPI_SO=$(python -c 'import glob,os,mpi4py; print((glob.glob(os.path.dirname(mpi4py.__file__)+"/MPI*.so")+[""])[0])' 2>/dev/null)
if [ -n "${MPI_SO}" ]; then
    NEED=$(ldd "${MPI_SO}" 2>/dev/null | awk '/libmpi_gnu_.*not found/ {print $1; exit}')
    if [ -n "${NEED}" ]; then
        ABI=${NEED#libmpi_gnu_}; ABI=${ABI%%.*}          # e.g. 123
        GCCVER="${ABI%${ABI#??}}.${ABI#??}"              # e.g. 12.3
        LIBDIR=$(ls -1d /opt/cray/pe/mpich/*/ofi/gnu/${GCCVER}/lib 2>/dev/null | tail -1)
        if [ -n "${LIBDIR}" ]; then
            echo "NOTE: mpi4py wants ${NEED} (GCC ${GCCVER}); prepending ${LIBDIR}"
            export LD_LIBRARY_PATH="${LIBDIR}:${LD_LIBRARY_PATH}"
        else
            echo "ERROR: mpi4py needs ${NEED} (GCC ${GCCVER}) but no" >&2
            echo "       /opt/cray/pe/mpich/*/ofi/gnu/${GCCVER}/lib exists.  Rebuild it:" >&2
            echo "         module load PrgEnv-gnu" >&2
            echo "         MPICC=\"cc -shared\" pip install --force-reinstall \\" >&2
            echo "             --no-cache-dir --no-binary=mpi4py --no-build-isolation mpi4py" >&2
            exit 1
        fi
    fi
fi

# cuFFTDx: without it propagation falls back to cuPy FFT and the timings are
# not comparable with the other machines.
export MATHDX_ROOT=${MATHDX_ROOT:-/eagle/APS_IRI/vnikitin/nvidia/nvidia-mathdx-25.12.1-cuda12/nvidia/mathdx/25.12/}
export NVCC=${NVCC:-$(command -v nvcc)}
export CUFFTDX_SM=${CUFFTDX_SM:-80}      # A100
# JIT output must not land in a read-only or shared-stale dir; one .so per size.
export CUFFTDX_SO_DIR=${CUFFTDX_SO_DIR:-$HOME/.cache/holotomocupy_cufftdx}
mkdir -p "${CUFFTDX_SO_DIR}"

set -eu
echo "env:         ${HTC_ENV_KIND}"
echo "python:      $(command -v python)"
echo "nvcc:        ${NVCC:-<none>}"
echo "MATHDX_ROOT: ${MATHDX_ROOT}"
