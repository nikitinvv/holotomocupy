#!/bin/bash
#PBS -A 14347
#PBS -l select=64:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:grand:eagle
#PBS -l walltime=2:59:00
#PBS -q prod
#PBS -N holotomo_complex64
##PBS -q debug-scaling
#PBS -o 17p23keV.o
#PBS -j oe
##PBS -m be
##PBS -M vnikitin@anl.gov

# The rest is an example of how an MPI job might be set up
echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

echo Jobid: $PBS_JOBID
echo Running on host `hostname`
echo Running on nodes `cat $PBS_NODEFILE`

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS=4           # Number of MPI ranks per node
NTHREADS=4
NDEPTH=8
export NTOTRANKS=$(( NNODES * NRANKS ))

echo "NUM_OF_NODES=${NNODES}  TOTAL_NUM_RANKS=${NTOTRANKS}  RANKS_PER_NODE=${NRANKS}"
# #export NCCL_IB_DISABLE=1
# export NCCL_COLLNET_ENABLE=1
# export NCCL_NET_GDR_LEVEL=PHB
# #export NCCL_P2P_DISABLE=1
# export NCCL_SOCKET_IFNAME=hsn1
# #OMPI_MCA_opal_cuda_support=true
# #export NCCL_DEBUG=INFO
# #export CUDA_DEVICE_ORDER="PCI_BUS_ID"

module use /soft/modulefiles;  module load conda; conda activate base
CONDA_NAME=$(echo ${CONDA_PREFIX} | tr '\/' '\t' | sed -E 's/mconda3|\/base//g' | awk '{print $NF}')
VENV_DIR="/home/vvnikitin/venvs/${CONDA_NAME}"
source "${VENV_DIR}/bin/activate"

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} ./set_affinity_gpu_polaris.sh python step6_rec_iterative_mpi.py configs/config.conf