For Alexey: Big-node class — 8× 80 GiB GPU + 2 TiB RAM

1 node
mpirun -np 8 ./set_affinity_gpu.sh python test.py --n 512  --ntheta 450 --nchunk 64 --log log512_64
mpirun -np 8 ./set_affinity_gpu.sh python test.py --n 1024 --ntheta 900 --nchunk 64 --log log1024_64
mpirun -np 8 ./set_affinity_gpu.sh python test.py --n 2048 --ntheta 1800 --nchunk 16 --log log2048_8

8 nodes
mpirun -np 64 ./set_affinity_gpu.sh python test.py --n 4096 --ntheta 3600 --nchunk 4 --log log4096_64

32 nodes
mpirun -np 256 ./set_affinity_gpu.sh python test.py --n 6144 --ntheta 4800 --nchunk 2 --log log6144_256

64 nodes
mpirun -np 512 ./set_affinity_gpu.sh python test.py --n 8192 --ntheta 7200 --nchunk 1 --log log8192_512

