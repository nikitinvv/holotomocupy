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
| h5py **with MPI support** | HDF5 I/O — must be built with parallel HDF5 (the default `pip install h5py` is serial-only and will fail at checkpoint writing) |
| dxchange | Tomography I/O utilities |
| matplotlib | Visualization in notebooks |

### Conda environment

```bash
conda create -n holotomocupy -c conda-forge \
    cupy mpi4py "h5py=*=mpi_openmpi*" dxchange \
    setuptools matplotlib psutil jupyter matplotlib-scalebar
conda activate holotomocupy
```

> **Important:** h5py must be built with parallel HDF5 (MPI) support — the `mpi_openmpi` build variant from conda-forge provides this. The default `pip install h5py` or the `nompi` conda variant will raise `ValueError: h5py was built without MPI support, can't use mpio driver` at runtime when saving checkpoints.

### Install the package

```bash
git clone https://github.com/nikitinvv/holotomocupy
cd holotomocupy
pip install -e .
```

---

## Running tests

Tests live in `tests/`:

| File | What it tests |
|---|---|
| `tests/test.ipynb` | End-to-end holotomography reconstruction (interactive, single process) |
| `tests/test.py` | Same test as a plain Python script, runnable with MPI |

### What the test does

1. Builds a synthetic 3-D phantom object and loads a realistic probe from pre-saved TIFF files.
2. Forward-simulates diffraction patterns and a flat-field reference from the ground-truth variables.
3. Resets variables to an imperfect initial guess and runs the BH iterative solver.
4. Saves reconstruction checkpoints to `tests/test_results/checkpoint_NNNN.h5` every `vis_step` iterations.

### Running `test.py` with MPI

`bind.sh` maps each MPI rank to a unique GPU via `CUDA_VISIBLE_DEVICES`:

```bash
cd tests
mpirun -np 2 ./bind.sh python test.py
```

Replace `2` with the number of available GPUs. Each rank processes a contiguous slice of the object volume (z-axis) and a contiguous block of projections (theta-axis).

To suppress UCX warnings add `--mca opal_common_ucx_opal_mem_hooks 1`:

```bash
mpirun --mca opal_common_ucx_opal_mem_hooks 1 -np 2 ./bind.sh python test.py
```

### Output

Checkpoints are written to `tests/test_results/` as HDF5 files:

```
tests/test_results/
    checkpoint_0016.h5   # iteration 16
    checkpoint_0032.h5   # iteration 32
    ...
```

Each file contains:

| Dataset | Shape | Description |
|---|---|---|
| `obj_re` | `(nzobj, nobj, nobj)` | Real part of reconstructed object (refractive index decrement δ) |
| `obj_im` | `(nzobj, nobj, nobj)` | Imaginary part (absorption β) — present when `obj_dtype=complex64` |
| `prb_abs` | `(ndist, nz, n)` | Probe amplitude |
| `prb_phase` | `(ndist, nz, n)` | Probe phase |
| `pos` | `(ntheta, ndist, 2)` | Refined sample positions (pixels) |

> **Note:** `h5py` with MPI support is required — see the Installation section.

---

## Reconstruction pipeline for experimental data

The full pipeline is in `experimental/Y350a_dist1234/`. Steps 1–5 are Jupyter notebooks; step 6 is a Python script launched with `mpirun`.

### Step 1 — Convert raw data

Open `step1_convert.ipynb` and run all cells.

Reads raw detector frames, applies flat-field correction, and writes a single HDF5 file consumed by all subsequent steps.

### Step 2 — Preprocessing

Open `step2_preprocessing.ipynb` and run all cells.

Rings removal, background subtraction, binning.

### Step 3 — Find shifts

Open `step3_find_shifts.ipynb` and run all cells.

Estimates sample position shifts between projection angles using cross-correlation.

### Step 4 — Make binned data

Open `step4_make_binned.ipynb` and run all cells.

Produces downsampled datasets at multiple bin levels for fast prototyping.

### Step 5 — Paganin reconstruction

Open `step5_rec_paganin.ipynb` and run all cells.

Fast single-distance phase retrieval (Paganin filter) used as the initial guess for step 6.

### Step 6 — Iterative MPI reconstruction

GPU-binding wrapper maps each MPI rank to a unique GPU:

```bash
mpirun -np <ngpus> ./bind.sh python step6_rec_iterative_mpi.py configs/config.conf
```

Example with 4 GPUs:

```bash
cd experimental/Y350a_dist1234
mpirun -np 4 ./bind.sh python step6_rec_iterative_mpi.py configs/config.conf
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

`experimental/Y350a_dist1234/set_affinity_gpu_polaris.sh` assigns GPUs in reverse order to match the Polaris PCIe topology (see [ALCF machine overview](https://www.alcf.anl.gov/support/user-guides/polaris/hardware-overview/machine-overview/index.html)):

```bash
#!/bin/bash -l
num_gpus=4
gpu=$((${num_gpus} - 1 - ${PMI_LOCAL_RANK} % ${num_gpus}))
export CUDA_VISIBLE_DEVICES=$gpu
exec "$@"
```

### PBS job script

`experimental/Y350a_dist1234/prun.sh` — submit with `qsub prun.sh` from the experiment directory:

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

# Activate conda + venv
module use /soft/modulefiles; module load conda; conda activate base
CONDA_NAME=$(echo ${CONDA_PREFIX} | tr '\/' '\t' | sed -E 's/mconda3|\/base//g' | awk '{print $NF}')
source "$HOME/venvs/${CONDA_NAME}/bin/activate"

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} \
    --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} \
    ./set_affinity_gpu_polaris.sh \
    python step6_rec_iterative_mpi.py configs/config.conf
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
vi configs/config.conf   # set in_file and path_out

# 6. Edit prun.sh: set project ID, node count, config path
vi prun.sh

# 7. Submit
qsub prun.sh

# 8. Monitor
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

`configs/config.conf` — all parameters as `key=value`, comments with `#`:

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

## Package layout

```
src/holotomocupy/
    rec_mpi.py          # BH iterative solver (MPI-aware)
    tomo.py             # tomographic projection (RT / FBP)
    shift.py            # B-spline sub-pixel shift operators
    propagation.py      # Fresnel propagator
    chunking.py         # @gpu_batch decorator
    cuda_kernels.py     # raw CUDA kernels (spline interpolation, NUFFT gather)
    reader.py           # MPI-aware HDF5 reader
    writer.py           # MPI-aware HDF5 writer / checkpointing
    mpi_functions.py    # MPI collective helpers
    config.py           # configuration file parser
    logger_config.py    # colored MPI-aware logger
    utils.py            # visualization helpers (mshow, mshow_complex, read_tiff, …)

experimental/
    Y350a_dist1234/     # full brain dataset pipeline (steps 1–6)
        bind.sh                      # GPU-to-rank binding (mpirun / SLURM)
        set_affinity_gpu_polaris.sh  # GPU-to-rank binding for Polaris A100 topology
        prun.sh                      # PBS job script for Polaris (ALCF)
    performance_tests/  # timing benchmarks

tests/
    test.ipynb          # end-to-end holotomography reconstruction test (interactive)
    test.py             # same test as MPI-runnable script (mpirun -np N ./bind.sh python test.py)
    bind.sh             # GPU-to-rank binding wrapper
```

---

## Citation

If you use this software, please cite:

> Viktor Nikitin, *HolotomocuPy — GPU-accelerated X-ray holotomography*, Argonne National Laboratory, https://github.com/nikitinvv/holotomocupy
