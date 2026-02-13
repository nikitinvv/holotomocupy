mpirun -np 1 -H tomo5-ib:4 bash -lc '
source ~/.bashrc
which nvcc
source ~/conda/miniforge3/etc/profile.d/conda.sh
conda activate holotomocupy


PY="$CONDA_PREFIX/bin/python"
./bind.sh "$PY" rec_iterative_mpi_syn.py config2.conf'


# if [[ "${OMPI_COMM_WORLD_RANK:-}" == "0" ]]; then
#   nsys profile -o nsys_rank0 --force-overwrite=true --trace=nvtx,cuda \
#     ./bind.sh "$PY" rec_iterative_mpi_syn.py config2.conf
# else
#   ./bind.sh "$PY" rec_iterative_mpi_syn.py config2.conf
# fi'