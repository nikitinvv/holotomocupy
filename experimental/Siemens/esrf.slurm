#!/bin/bash
#SBATCH --job-name=holotomo
#SBATCH --partition=low # Or another relevant partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1       # = number of GPUs you want to use
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=64          # important
#SBATCH --gpu-bind=closest
##SBATCH --exclusive
#
#SBATCH --time=0:30:00 # 1 hour time limit

# Load necessary modules (example)
hostname

module add mamba 
module add cuda
mamba activate /home/esrf/viktor17051990/.conda/holotomocupy
export MATHDX_ROOT=/home/esrf/viktor17051990/nvidia/nvidia-mathdx-25.12.1-cuda12/nvidia/mathdx/25.12
echo $CUDA_HOME
srun --mpi=pmix_v3 ./bind.sh python step0.py config_step0.conf
