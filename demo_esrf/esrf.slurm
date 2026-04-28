#!/bin/bash
#SBATCH --job-name=holotomo
#SBATCH --partition=low-gpu # Or another relevant partition
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2       # = number of GPUs you want to use
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32          # example
#SBATCH --gpu-bind=closest
#SBATCH --exclusive
#
##SBATCH --cpus-per-task=32 # Request 4 cores for this task
#SBATCH --time=1:00:00 # 1 hour time limit
#SBATCH --mem=700G # 
##SBATCH --nodelist=gpbm18-[01-21]
#SBATCH --nodelist=frcrg-gpu5h200-01

# Load necessary modules (example)
hostname

module add mamba 
module add cuda
mamba activate /home/esrf/viktor17051990/.conda/holotomocupy
export MATHDX_ROOT=/home/esrf/viktor17051990/nvidia/nvidia-mathdx-25.12.1-cuda12/nvidia/mathdx/25.12
srun --mpi=pmix_v3 ./bind.sh python rec_iterative_mpi.py config1.conf
