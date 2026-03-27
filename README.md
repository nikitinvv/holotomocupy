# HolotomocuPy

GPU-accelerated X-ray holotomography reconstruction with MPI multi-GPU support.

Holotomography is a coherent imaging technique that reconstructs the 3-D complex refractive-index distribution of a sample by combining holography with tomography. This package provides iterative algorithms optimized for large datasets at modern synchrotron sources.

## Key features

- **GPU acceleration** via CuPy (drop-in GPU NumPy)
- **MPI multi-GPU** — one rank per GPU, tested with 1–1024 GPUs - close to linear performance gain
- **`@gpu_batch` decorator** — automatically chunks data that exceeds GPU memory
- **Modular operators** — tomographic projection, Fresnel propagator, B-spline shifts, scaling; all reusable independently
- **Bilinear-Hessian (BH) solver** — joint optimization of object, probe, and sample positions
- **Checkpointing** — automatically resumes from the latest saved iteration
- **Jupyter notebook pipeline** — steps 1–5 for data preparation, step 6 for iterative reconstruction

---

## Installation

### Requirements

| Dependency | Notes |
|---|---|
| CUDA ≥ 11 | One GPU per MPI rank |
| CuPy | GPU NumPy |
| mpi4py | MPI bindings |
| h5py | HDF5 I/O — must be built with parallel (OpenMPI) support |
| dxchange | Tomography I/O utilities |
| matplotlib | Visualization in notebooks |
| NVIDIA mathDX *(optional)* | Enables the cuFFTDx-based fast Fresnel propagator |

### Optional: cuFFTDx fast propagator (NVIDIA mathDX)

The Fresnel propagator (`propagation.py`) has two backends:

| Backend | Speed | Requirement |
|---|---|---|
| cuPy (default) | baseline | none — works out of the box |
| cuFFTDx | faster | NVIDIA mathDX + nvcc |

The cuFFTDx backend is selected automatically at startup when mathDX is found. If it is unavailable, the package silently falls back to cuPy with no loss of correctness.

**1. Download and install mathDX**

Follow the installation guide at https://docs.nvidia.com/cuda/mathdx/installation.html to download and unpack the mathDX package:

```bash
tar -xzf nvidia-mathdx-*.tar.gz -C /opt/nvidia
```

**2. Set environment variables**

```bash
export MATHDX_ROOT=/opt/nvidia/nvidia-mathdx-25.12.1-cuda13/nvidia/mathdx/25.12
export NVCC=/usr/local/cuda/bin/nvcc   # or wherever nvcc lives
# Optional overrides:
# export CUFFTDX_SM=80          # target SM version (default: 80)
# export CUFFTDX_SO_DIR=/tmp    # where JIT-compiled .so files are cached
```

Add these lines to your `~/.bashrc` or the job script so they persist.

**3. Verify detection**

```python
from holotomocupy.propagation import Propagation
# Should print: "cuFFTDx (mathDX) available — using fast cuFFTDx propagator."
```

If mathDX is not found you will see a `UserWarning` explaining which path is missing.

**JIT compilation and MPI**

The first time a new grid size is used, the package JIT-compiles a small CUDA shared library with `nvcc` and caches it in `CUFFTDX_SO_DIR`. In an MPI run, **only rank 0 compiles**; all other ranks wait at a barrier and then load the pre-built library. Subsequent runs reuse the cached `.so` and skip compilation entirely.

### Conda environment

```bash
conda create -n holotomocupy -c conda-forge \
    cupy mpi4py "h5py=*=mpi_openmpi*" dxchange \
    setuptools matplotlib psutil jupyter matplotlib-scalebar
conda activate holotomocupy
```

### Install the package

```bash
git clone https://github.com/nikitinvv/holotomocupy
cd holotomocupy
pip install -e .
```

---

## Reconstruction pipeline

The full pipeline is in `experimental/Y350a_dist1234/`. Steps 1–5 are Jupyter notebooks; step 6 is a Python script launched with `mpirun`.

### Step 1 — Convert raw data

```bash
jupyter nbconvert --to notebook --execute step1_convert.ipynb
```

Reads raw detector frames, applies flat-field correction, and writes a single HDF5 file consumed by all subsequent steps.

### Step 2 — Preprocessing

```bash
jupyter nbconvert --to notebook --execute step2_preprocessing.ipynb
```

Rings removal, background subtraction, binning.

### Step 3 — Find shifts

```bash
jupyter nbconvert --to notebook --execute step3_find_shifts.ipynb
```

Estimates sample position shifts between projection angles using cross-correlation.

### Step 4 — Make binned data

```bash
jupyter nbconvert --to notebook --execute step4_make_binned.ipynb
```

Produces downsampled datasets at multiple bin levels for fast prototyping.

### Step 5 — Paganin reconstruction

```bash
jupyter nbconvert --to notebook --execute step5_rec_paganin.ipynb
```

Fast single-distance phase retrieval (Paganin filter) used as the initial guess for step 6.

### Step 6 — Iterative MPI reconstruction

GPU-binding wrapper maps each MPI rank to a unique GPU:

```bash
mpirun -np <ngpus> ./bind.sh python step6_rec_iterative_mpi.py configs/config1.conf
```

Example with 4 GPUs:

```bash
cd experimental/Y350a_dist1234
mpirun -np 4 ./bind.sh python step6_rec_iterative_mpi.py configs/config1.conf
```

---

## Running on Polaris (ALCF)

Polaris is an A100 cluster at Argonne Leadership Computing Facility (ALCF). Each node has **4 A100 GPUs**. The `polaris/` directory contains ready-to-use scripts.

### Python environment

Polaris uses a shared conda base environment loaded via modules. Create a per-user virtual environment on top of it so you can install packages:

```bash
module use /soft/modulefiles; module load conda; conda activate base
CONDA_NAME=$(echo ${CONDA_PREFIX} | tr '\/' '\t' | sed -E 's/mconda3|\/base//g' | awk '{print $NF}')
VENV_DIR="$HOME/venvs/${CONDA_NAME}"
mkdir -p "${VENV_DIR}"
python -m venv "${VENV_DIR}" --system-site-packages
source "${VENV_DIR}/bin/activate"
pip install -e /path/to/holotomocupy
```

The same activation snippet (`module load conda` + `source …/activate`) is used in the PBS job script to reproduce the environment on compute nodes.

See [ALCF Python docs](https://docs.alcf.anl.gov/polaris/data-science/python/) for more details.

### GPU affinity

`polaris/set_affinity_gpu_polaris.sh` assigns GPUs in reverse order to match the Polaris PCIe topology (see [ALCF machine overview](https://www.alcf.anl.gov/support/user-guides/polaris/hardware-overview/machine-overview/index.html)):

```bash
#!/bin/bash -l
num_gpus=4
gpu=$((${num_gpus} - 1 - ${PMI_LOCAL_RANK} % ${num_gpus}))
export CUDA_VISIBLE_DEVICES=$gpu
exec "$@"
```

### PBS job script

`polaris/prun.sh` — submit with `qsub polaris/prun.sh` from the experiment directory:

```bash
#!/bin/bash
#PBS -A <project_id>
#PBS -l select=256:system=polaris   # number of nodes (4 GPUs each)
#PBS -l place=scatter
#PBS -l filesystems=home:grand:eagle
#PBS -l walltime=00:40:00
#PBS -q prod

cd $PBS_O_WORKDIR

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS=4          # 4 MPI ranks per node = 4 GPUs per node
NTHREADS=4
NDEPTH=8
export NTOTRANKS=$(( NNODES * NRANKS ))

echo "NUM_OF_NODES=${NNODES}  TOTAL_NUM_RANKS=${NTOTRANKS}  RANKS_PER_NODE=${NRANKS}"

# NCCL networking settings for Polaris HPE Slingshot interconnect
export NCCL_COLLNET_ENABLE=1
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_SOCKET_IFNAME=hsn1

# Activate conda + venv
module use /soft/modulefiles; module load conda; conda activate base
CONDA_NAME=$(echo ${CONDA_PREFIX} | tr '\/' '\t' | sed -E 's/mconda3|\/base//g' | awk '{print $NF}')
source "$HOME/venvs/${CONDA_NAME}/bin/activate"

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} \
    --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} \
    ./set_affinity_gpu_polaris.sh \
    python step6_rec_iterative_mpi.py configs/config1.conf
```

### Typical workflow on Polaris

```bash
# 1. Log in
ssh <user>@polaris.alcf.anl.gov

# 2. Create the virtual environment (one-time setup)
module use /soft/modulefiles; module load conda; conda activate base
CONDA_NAME=$(echo ${CONDA_PREFIX} | tr '\/' '\t' | sed -E 's/mconda3|\/base//g' | awk '{print $NF}')
VENV_DIR="$HOME/venvs/${CONDA_NAME}"
mkdir -p "${VENV_DIR}"
python -m venv "${VENV_DIR}" --system-site-packages
source "${VENV_DIR}/bin/activate"

# 3. Clone and install
git clone https://github.com/nikitinvv/holotomocupy
cd holotomocupy
pip install -e .

# 4. Copy input data to eagle filesystem
cp /path/to/data.h5 /eagle/<project>/data.h5

# 5. Edit the config to point to eagle paths
cd experimental/Y350a_dist1234
vi configs/config1.conf   # set in_file and path_out

# 6. Copy the Polaris scripts next to your step6 script
cp ../../polaris/set_affinity_gpu_polaris.sh .

# 7. Edit prun.sh: set project ID, node count, config path
vi ../../polaris/prun.sh

# 8. Submit
qsub ../../polaris/prun.sh

# 9. Monitor
qstat -u $USER
tail -f <jobid>.o
```

### Scaling

4 GPUs per node × N nodes = 4N total ranks. Typical settings:

| Nodes | Total GPUs | `nchunk` | Dataset size |
|-------|-----------|----------|--------------|
| 1 | 4 | 8 | small / debug |
| 16 | 64 | 8 | medium |
| 256 | 1024 | 8 | full 2500-angle 3D dataset |

---

## Configuration file

`configs/config1.conf` — all parameters as `key=value`, comments with `#`:

```ini
in_file=/data/dataset.h5       # input HDF5 file
path_out=/data/results         # output directory for checkpoints and results

# Data dimensions
ntheta=2500                    # number of projection angles
nz=1024                        # vertical detector size (pixels)
n=1024                         # horizontal detector size (pixels)
nzobj=1632                     # vertical object size (pixels)
nobj=1632                      # horizontal/lateral object size (pixels)
ndist=4                        # number of propagation distances
bin=1                          # binning level (0 = full, 1 = 2×, -1 = 2× unbin)

# Solver
niter=3                        # number of BH iterations
nchunk=8                       # projection chunk size (tune to GPU memory)
start_iter=0                   # resume from this iteration (0 = fresh start)
err_step=1                     # compute/print error every N iterations (-1 = never)
vis_step=-1                    # save visualization every N iterations (-1 = never)

# Physics
paganin=20                     # Paganin regularization constant
rotation_center_shift=-8.78    # rotation center offset from detector center (pixels)
mask=1.1                       # tomographic mask radius (fraction of detector half-width)
obj_dtype=complex64            # object dtype: complex64 or float32

# Regularization
lam_prbfit=3.1e-3              # probe fit weight
rho=1,0.05,0.02                # step-size scaling for object, probe, positions

# Misc
start_theta=0                  # first angle index
log_level=DEBUG                # DEBUG / INFO / WARNING / ERROR
```

---

## Running tests

Tests live in `tests/` as Jupyter notebooks:

| Notebook | What it tests |
|---|---|
| `tests/shift/test_shift.ipynb` | B-spline shift operators |
| `tests/chunking/test_chunking.ipynb` | `@gpu_batch` decorator |
| `tests/holotomo3d/test.ipynb` | End-to-end holotomography reconstruction |

Open a notebook and run all cells, or execute from the command line:

```bash
jupyter nbconvert --to notebook --execute tests/shift/test_shift.ipynb
jupyter nbconvert --to notebook --execute tests/chunking/test_chunking.ipynb
jupyter nbconvert --to notebook --execute tests/holotomo3d/test.ipynb
```

---

## Package layout

```
src/holotomocupy/
    rec_mpi.py          # BH iterative solver (MPI-aware)
    tomo.py             # tomographic projection (RT / FBP)
    shift.py            # B-spline sub-pixel shift operators
    propagation.py      # Fresnel propagator (cuFFTDx or cuPy backend)
    conv2d_cufftdx.py   # cuFFTDx JIT wrapper + availability flag
    cuda/conv2d.cu      # cuFFTDx 2-D convolution kernel source
    chunking.py         # @gpu_batch decorator
    cuda_kernels.py     # raw CUDA kernels (spline interpolation, NUFFT gather)
    reader.py           # MPI-aware HDF5 reader
    writer.py           # MPI-aware HDF5 writer / checkpointing
    mpi_functions.py    # MPI collective helpers
    config.py           # configuration file parser
    logger_config.py    # colored MPI-aware logger

experimental/
    Y350a_dist1234/     # full brain dataset pipeline (steps 1–6)
    performance_tests/  # timing benchmarks

polaris/
    prun.sh                      # PBS job script for Polaris (ALCF)
    set_affinity_gpu_polaris.sh  # GPU-to-rank binding for A100 topology

tests/
    shift/              # shift kernel tests
    chunking/           # gpu_batch decorator tests
    holotomo3d/         # end-to-end holotomography tests
```

---

## Citation

If you use this software, please cite:

> Viktor Nikitin, *HolotomocuPy — GPU-accelerated X-ray holotomography*, Argonne National Laboratory, https://github.com/nikitinvv/holotomocupy
