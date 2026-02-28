#!/bin/bash

rank="${OMPI_COMM_WORLD_RANK:-$SLURM_PROCID}"
local_rank="${OMPI_COMM_WORLD_LOCAL_RANK:-$SLURM_LOCALID}"
export OMP_NUM_THREADS=4
ngpus=$(nvidia-smi -L | wc -l)
export CUDA_VISIBLE_DEVICES=$(( $local_rank % $ngpus )) 
echo $rank" uses "${CUDA_VISIBLE_DEVICES}" of "$ngpus "  " `hostname` 
# bash -c 'source ~/.bashrc;conda activate holotomocupy; 
# cd /home/beams/TOMO/vnikitin/holotomocupy_work_perf/experimental/performance_tests; which python;
# python rec_iterative_mpi.py config1.conf & bash' 
$*