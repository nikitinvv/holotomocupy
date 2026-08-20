mpirun -np 4 ./set_affinity_gpu.sh python test.py --n 512 --ntheta 450 --nchunk 16 --log log512_16
mpirun -np 4 ./set_affinity_gpu.sh python test.py --n 1024 --ntheta 900 --nchunk 8 --log log1024_8
mpirun -np 4 ./set_affinity_gpu.sh python test.py --n 2048 --ntheta 1800 --nchunk 4 --log log2048_4
