ssh -t tomo@tomo1 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/march2025/Chip_nfp_2048; \\
CUDA_VISIBLE_DEVICES=0 nohup python rec_nfp.py 1 0 0 0 nan 0 >r_0_0_0_nan_0 & \\
bash'"

ssh -t tomo@tomo2 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/march2025/Chip_nfp_2048; \\
CUDA_VISIBLE_DEVICES=0 nohup python rec_nfp.py 1 0 0 64 symmetric 0 >r_0_0_64_symmetric_0 & \\
CUDA_VISIBLE_DEVICES=1 nohup python rec_nfp.py 1 0 0 64 none 0 >r_0_0_64_none_0 & \\
bash'"

ssh -t tomo@tomo4 "bash -c 'source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/march2025/Chip_nfp_2048; \\
CUDA_VISIBLE_DEVICES=0 nohup python rec_nfp.py 1 0 0 0 symmetric 0 >r_0_0_0_symmetric_0 & \\
CUDA_VISIBLE_DEVICES=1 nohup python rec_nfp.py 1 0.002 0 384 symmetric 512 >r_0.002_0_384_symmetric_512 & \\
CUDA_VISIBLE_DEVICES=2 nohup python rec_nfp.py 1 0 0.001 384 symmetric 512 >r_0_0.001_384_symmetric_512 & \\
CUDA_VISIBLE_DEVICES=3 nohup python rec_nfp.py 1 0.002 0.001 384 symmetric 512 >r_0.002_0.001_384_symmetric_512 & \\
bash'"


ssh -t tomo@tomo3 "bash -c ' source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/march2025/Chip_nfp_2048; \\
CUDA_VISIBLE_DEVICES=0 nohup python rec_nfp.py 1 0 0 384 symmetric 512 >r_0_0_384_symmetric_512 & \\
bash'"




ssh -t tomo@tomo1 "bash -c ' source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/march2025/Chip_nfp_2048; \\
CUDA_VISIBLE_DEVICES=0 nohup python rec_nfp.py 1 0.005 0.005 384 symmetric 512 >r_0.005_0.005_384_symmetric_512 & \\
CUDA_VISIBLE_DEVICES=1 nohup python rec_nfp.py 1 0.005 0.005 384 none 512 >r_0.005_0.005_384_none_512 & \\
bash'"

ssh -t tomo@tomo2 "bash -c ' source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/march2025/Chip_nfp_2048; \\
CUDA_VISIBLE_DEVICES=0 nohup python rec_nfp.py 1 0.002 0.002 384 symmetric 576 >r_0.002_0.002_384_symmetric_576 & \\
CUDA_VISIBLE_DEVICES=1 nohup python rec_nfp.py 1 0.002 0.002 384 none 576 >r_0.002_0.002_384_none_576 & \\
bash'"


ssh -t tomo@tomo2 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/march2025/Chip_nfp_2048; \\
CUDA_VISIBLE_DEVICES=0 nohup python rec_nfp.py 1 0 0.0005 384 symmetric 512 >r_0_0.0005_384_symmetric_512 & \\
CUDA_VISIBLE_DEVICES=1 nohup python rec_nfp.py 1 0 0.001 384 symmetric 512 >r_0_0.001_384_symmetric_512 & \\
bash'"

ssh -t tomo@tomo3 "bash -c ' source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/march2025/Chip_nfp_2048; \\
CUDA_VISIBLE_DEVICES=0 nohup python rec_nfp.py 1 0.002 0.002 384 symmetric 1536 >r_0.002_0.002_384_symmetric_1536 & \\
CUDA_VISIBLE_DEVICES=0 nohup python rec_nfp.py 1 0.002 0.002 384 none 1536 >r_0.002_0.002_384_none_1536 & \\
bash'"

CUDA_VISIBLE_DEVICES=2 nohup python rec_nfp.py 1 0 0.0005 384 symmetric 512 >r_0_0.0005_384_symmetric_512 & \\
CUDA_VISIBLE_DEVICES=3 nohup python rec_nfp.py 1 0 0.001 384 symmetric 512 >r_0_0.001_384_symmetric_512 & \\