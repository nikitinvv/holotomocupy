# Environment for the Polaris perf jobs -- sourced by run_polaris.sh and
# run_mosaic_polaris.sh, so the ranks never depend on ~/.bashrc.
#
#   . ./env_polaris.sh
#
# Two parts: the normal conda + venv activation, and a TEMPORARY workaround for
# the August 2026 maintenance (see "TEMPORARY" below -- delete it once the site
# module is fixed and mpi4py is rebuilt).

set +eu

module use /soft/modulefiles
# TEMPORARY: a private copy of the conda modulefile with the retired version
# pins fixed (cray-hdf5-parallel 1.14.3.5 -> 1.14.3.9, gcc-native 14.2 -> what
# exists) shadows the site one, because `module use` prepends.  Create it with:
#   HDF5=$(module -t --redirect spider cray-hdf5-parallel | grep -oE 'cray-hdf5-parallel/[0-9.]+' | sort -V | tail -1)
#   GCCN=$(module -t --redirect spider gcc-native        | grep -oE 'gcc-native/[0-9.]+'         | sort -V | tail -1)
#   mkdir -p ~/modulefiles/conda
#   sed -e "s|cray-hdf5-parallel/1\.14\.3\.5|${HDF5}|g" -e "s|gcc-native/14\.2|${GCCN}|g" \
#       /soft/modulefiles/conda/2025-09-25.lua > ~/modulefiles/conda/2025-09-25.lua
# Remove ~/modulefiles/conda once the site module works again.
[ -d "$HOME/modulefiles" ] && module use "$HOME/modulefiles"
module load conda 2>/dev/null || true

if ! command -v conda >/dev/null 2>&1; then
    # TEMPORARY: no usable conda module (the site one depends on
    # gcc-native/14.2 and cray-hdf5-parallel/1.14.3.5, both retired in the
    # maintenance, and no patched copy is in ~/modulefiles).  The conda install
    # itself is untouched, so source it directly and skip Lmod.  This loses
    # whatever else the modulefile sets, so the patched-copy route above is
    # preferable.
    CONDA_SH=$(ls -1dt /soft/applications/conda/*/mconda3/etc/profile.d/conda.sh 2>/dev/null | head -1)
    if [ -n "${CONDA_SH}" ]; then
        echo "NOTE: 'module load conda' failed; sourcing ${CONDA_SH} instead."
        . "${CONDA_SH}"
    else
        echo "ERROR: no conda module and no conda install under /soft/applications/conda." >&2
        exit 1
    fi
fi
conda activate base

# The venv is keyed to the conda-module version, so a module bump points at one
# that was never created.  Fall back to the newest instead of silently running
# on base conda.
CONDA_NAME=$(echo ${CONDA_PREFIX} | tr '\/' '\t' | sed -E 's/mconda3|\/base//g' | awk '{print $NF}')
VENV_ROOT=${VENV_ROOT:-$HOME/venvs}
VENV_DIR="${VENV_ROOT}/${CONDA_NAME}"
if [ -f "${VENV_DIR}/bin/activate" ]; then
    . "${VENV_DIR}/bin/activate"
else
    echo "WARNING: ${VENV_DIR}/bin/activate is missing (CONDA_NAME=${CONDA_NAME})."
    FALLBACK=$(ls -1dt "${VENV_ROOT}"/*/bin/activate 2>/dev/null | head -1)
    if [ -n "${FALLBACK}" ]; then
        echo "         falling back to ${FALLBACK}"
        . "${FALLBACK}"
    else
        echo "         no venv under ${VENV_ROOT} -- running base conda."
    fi
fi

# TEMPORARY: mpich ABI repair. --------------------------------------------
# mpi4py in the venv is linked against the cray-mpich built with the GCC the
# venv was created under (libmpi_gnu_123.so.12 = GCC 12.3).  PrgEnv-gnu now
# defaults to gcc-native/14, so that lib dir is no longer on the path and every
# rank dies with "libmpi_gnu_123.so.12: cannot open shared object file".
# Find which ABI mpi4py wants and put the matching dir back.  Delete this block
# after rebuilding mpi4py:
#     module load PrgEnv-gnu
#     MPICC=cc pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
MPI_SO=$(python -c 'import glob,os,mpi4py; print((glob.glob(os.path.dirname(mpi4py.__file__)+"/MPI*.so")+[""])[0])' 2>/dev/null)
if [ -n "${MPI_SO}" ]; then
    NEED=$(ldd "${MPI_SO}" 2>/dev/null | awk '/libmpi_gnu_.*not found/ {print $1; exit}')
    if [ -n "${NEED}" ]; then
        ABI=${NEED#libmpi_gnu_}; ABI=${ABI%%.*}          # 123
        GCCVER="${ABI%${ABI#??}}.${ABI#??}"              # 12.3
        LIBDIR=$(ls -1d /opt/cray/pe/mpich/*/ofi/gnu/${GCCVER}/lib 2>/dev/null | tail -1)
        if [ -n "${LIBDIR}" ]; then
            echo "NOTE: mpi4py wants ${NEED} (GCC ${GCCVER}); prepending ${LIBDIR}"
            export LD_LIBRARY_PATH="${LIBDIR}:${LD_LIBRARY_PATH}"
        else
            echo "ERROR: mpi4py needs ${NEED} (GCC ${GCCVER}) but no" >&2
            echo "       /opt/cray/pe/mpich/*/ofi/gnu/${GCCVER}/lib exists." >&2
            echo "       Rebuild it in the venv:" >&2
            echo "         module load PrgEnv-gnu" >&2
            echo "         MPICC=cc pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py" >&2
            exit 1
        fi
    fi
fi
# -------------------------------------------------------------------------

# cuFFTDx: without it propagation falls back to cuPy FFT and the timings are
# not comparable with the other machines.
export MATHDX_ROOT=${MATHDX_ROOT:-/eagle/APS_IRI/vnikitin/nvidia/nvidia-mathdx-25.12.1-cuda12/nvidia/mathdx/25.12/}

set -eu
echo "python:      $(command -v python)"
echo "MATHDX_ROOT: ${MATHDX_ROOT}"
