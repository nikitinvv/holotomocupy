# HolotomocuPy

GPU-accelerated X-ray holotomography reconstruction with MPI multi-GPU support.

Holotomography is a coherent imaging technique that reconstructs the 3-D complex refractive-index distribution of a sample by combining holography with tomography. This package provides iterative algorithms optimized for large datasets at modern synchrotron sources.

## Key features

- **GPU acceleration** via CuPy (drop-in GPU NumPy)
- **MPI multi-GPU** — one rank per GPU, tested with 1–1024 GPUs — close to linear performance gain
- **`@gpu_batch` decorator** — automatically chunks data that exceeds GPU memory
- **Modular operators** — tomographic projection (R / RT / FBP), Fresnel propagator, B-spline shifts; all reusable independently
- **Bilinear-Hessian (BH) solver** — joint optimization of object, probe, and sample positions
- **Checkpointing** — automatically resumes from the latest saved iteration; each checkpoint also writes a mid-slice TIFF of `obj_re` for quick visual inspection
- **External initial guess** — load a pre-reconstructed `.vol` binary file (e.g. from a standard FBP pipeline) via `init_vol` in the config
- **Position error monitoring** — RMS position errors per distance are logged at every checkpoint
- **Accurate GPU memory reporting** — reports bytes visible to `nvidia-smi` (not just the CuPy pool)
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
| tifffile | TIFF I/O for checkpoint slices |
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
    cupy mpi4py "h5py=*=mpi_openmpi*" dxchange tifffile \
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

A complete example is in `experimental/Y350a_dist1234/`. The pipeline has two stages:

```bash
cd experimental/Y350a_dist1234

# Stage 1 — data preparation (single node, steps 0–5)
python step0.py config_step0.conf          # NFP probe calibration (optional)
python steps15.py config_steps15.conf      # steps 1–5: convert, preprocess, shifts, Paganin

# Stage 2 — iterative reconstruction (multi-node/multi-GPU)
mpirun -np 4 ./bind.sh python step6.py config_step6.conf
```

### Step 0 — NFP probe calibration (optional)

Near-field ptychography (NFP) reconstruction of the illumination probe. Writes a probe HDF5 file that step 6 uses as its starting probe instead of a flat-field estimate.

### Steps 1–5 — Data preparation (`steps15.py`)

`steps15.py` runs all data preparation with MPI across multiple nodes and GPUs:

- **Step 1** — reads raw EDF detector frames in parallel, writes a single HDF5 file with all distances, flat/dark fields, encoder shifts, and beam-monitor attributes
- **Step 2** — outlier removal (median-filter spike detection) and intensity normalisation per projection (GPU)
- **Step 3** — combines all shift sources into `cshifts_final`: encoder shifts from `correct.txt`, inter-plane alignment from Peter's RHAPP pipeline (`rhapp.mat`), slow-drift motion correction (`correct_motion.txt`), and optional 3-D tomographic correction
- **Step 4** — multi-distance back-projection onto the object plane at multiple bin levels; includes amplitude normalisation across distances
- **Step 5** — multi-distance Paganin phase retrieval followed by FBP reconstruction at all bin levels to produce the initial object guess for step 6

### Step 6 — Iterative MPI reconstruction

Joint iterative refinement of object, probe, and sample positions using the Bilinear-Hessian (BH) algorithm. Scales across multiple nodes and GPUs — one MPI rank per GPU:

```bash
mpirun -np <ngpus> ./bind.sh python step6.py config_step6.conf
```

**Startup order of priority for initial object:**

1. Latest checkpoint in `path_out` (automatic resume)
2. External `.vol` file specified by `init_vol` in the config
3. Paganin reconstruction written by step 5

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
    python step6.py config_step6.conf
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
vi config_step6.conf   # set in_file and path_out

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

## Configuration file (`config_step6.conf`)

All parameters as `key=value`; inline comments with `#`:

```ini
in_file=/data/dataset.h5       # input HDF5 file
path_out=/data/results         # output directory for checkpoints and results
prb_file=/data/nfp_results.h5  # probe initial guess from step 0 (optional)
init_vol=/data/rec.vol         # external binary .vol initial guess (optional)

# Data dimensions
ntheta=2500                    # number of projection angles
nz=1024                        # vertical detector size (pixels)
n=1024                         # horizontal detector size (pixels)
nzobj=1632                     # vertical object size (pixels)
nobj=1632                      # horizontal/lateral object size (pixels)
ndist=4                        # number of propagation distances
bin=1                          # binning level (0 = full resolution, 1 = 2×, ...)

# Solver
niter=257                      # number of BH iterations
nchunk=8                       # projection chunk size (tune to GPU memory)
start_iter=0                   # resume from this iteration (0 = fresh start)
err_step=8                     # compute/print error every N iterations (-1 = never)
vis_step=8                     # save checkpoint every N iterations (-1 = never)

# Physics
energy=17.23                   # X-ray energy (keV)
paganin=20                     # Paganin regularization constant
rotation_center_shift=-8.78    # rotation center offset from detector center (pixels)
mask=1.1                       # tomographic mask radius (fraction of detector half-width)
obj_dtype=complex64            # object dtype: complex64 or float32

# Regularization
lam_prbfit=3.1e-3              # probe fit weight
lam_laplacian=0                # 3-D Laplacian regularization weight
rho=1,0.05,0.02                # step-size scaling for object, probe, positions

# Misc
start_theta=0                  # first angle index
log_level=WARNING              # DEBUG / INFO / WARNING / ERROR
pos_checkpoint=                # override positions from a checkpoint file (optional)
```

### `init_vol` — external initial object

When `init_vol` is set, the solver reads a raw binary file as the starting object instead of the Paganin reconstruction. The expected format is a C-order `float32` flat binary array of shape `nzobj·2^b × nobj·2^b × nobj·2^b` where `b ≥ 0` is inferred automatically from the file size. When `b > 0`, block-averaging downsampling is applied. Values are normalized by `nobj/4` to match the internal reconstruction scale.

---

## Checkpoint outputs

At each `vis_step` interval (when `vis_step != -1` and `i > start_iter`), the following files are written to `path_out`:

| File | Content |
|---|---|
| `checkpoint_{iter:04}.h5` | Full object (real + imag), probe, positions |
| `checkpoint_{iter:04}_obj_re.tiff` | Middle z-slice of `obj_re` for quick visual check |

The log also records the mean absolute position error per distance at each checkpoint:

```
iter=256: pos mean abs error [px]  d0:(0.023,0.019)  d1:(0.026,0.020)  d2:(0.030,0.024)  d3:(0.031,0.026)
```

---

## Running tests

Tests live in `tests/` as Jupyter notebooks:

| Notebook | What it tests |
|---|---|
| `tests/shift/` | B-spline shift operators (forward, adjoint, derivatives) |
| `tests/tomo/` | Radon transform R / RT / FBP |
| `tests/holotomo3d/` | End-to-end holotomography reconstruction |
| `tests/nfp/` | Near-field ptychography probe calibration |
| `tests/mosaic/` | Mosaic / tiled reconstruction |

Run from the command line:

```bash
jupyter nbconvert --to notebook --execute tests/shift/test_shift.ipynb
jupyter nbconvert --to notebook --execute tests/holotomo3d/test.ipynb
```

---

## Package layout

```
src/holotomocupy/
    rec_mpi.py          # BH iterative solver (MPI-aware)
    rec_nfp_mpi.py      # near-field ptychography probe calibration solver (MPI-aware)
    tomo.py             # tomographic projection: R, RT, FBP (ramp/shepp/parzen), rec_tomo CG
    shift.py            # B-spline sub-pixel shift operators (S, S*, curlyS, derivatives)
    propagation.py      # Fresnel propagator (cuFFTDx or cuPy backend)
    conv2d_cufftdx.py   # cuFFTDx JIT wrapper + availability flag
    cuda/conv2d.cu      # cuFFTDx 2-D convolution kernel source
    chunking.py         # @gpu_batch decorator — auto-chunks over GPU memory limit
    cuda_kernels.py     # raw CUDA kernels (spline interpolation, NUFFT gather/scatter)
    reader.py           # MPI-aware HDF5 reader, raw .vol binary reader, Octave mat loader
    writer.py           # MPI-aware HDF5 writer / checkpointing + TIFF slice output
    mpi_functions.py    # MPI collective helpers (allreduce, redistribute)
    config.py           # configuration file parser (step 6 and steps 1–5)
    utils.py            # GPU/CPU memory utilities, visualization helpers, timer decorator
    logger_config.py    # colored MPI-aware logger

experimental/
    Y350a_dist1234/     # brain dataset pipeline (steps 0–6, 4 distances)
    AtomiumS2/          # Atomium S2 dataset pipeline (steps 0–6, 4 distances)
    y350a_80um/         # y350a 80 µm dataset pipeline (steps 0–6, 4 distances)
    performance_tests/  # timing benchmarks and MPI scaling tests

tests/
    shift/              # B-spline shift kernel adjoint / derivative tests
    tomo/               # Radon transform consistency tests
    holotomo3d/         # end-to-end holotomography reconstruction test
    nfp/                # near-field ptychography test
    mosaic/             # mosaic reconstruction test
```

---

## Citation

If you use this software, please cite:

> Viktor Nikitin, *HolotomocuPy — GPU-accelerated X-ray holotomography*, Argonne National Laboratory, https://github.com/nikitinvv/holotomocupy
