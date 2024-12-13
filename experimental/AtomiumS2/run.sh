
ssh -t tomo@tomo4 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/experimental/AtomiumS2/;ls; \\
nohup python rec_levels.py 150 0 0 >0 & nohup python rec_levels.py 150 150 1 >150 & nohup python rec_levels.py 150 300 2 >300 & nohup python rec_levels.py 150 450 3 >450 & bash'"



ssh -t tomo@tomo5 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/experimental/AtomiumS2/;ls; \\
nohup python rec_levels.py 150 600 0 >600 & nohup python rec_levels.py 150 750 1 >750 & nohup python rec_levels.py 150 900 2 >900 & nohup python rec_levels.py 150 1050 3 >1050 & bash'"

ssh -t tomo@tomo2 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/experimental/AtomiumS2/;ls; \\
nohup python rec_levels.py 150 1200 0 >1200 & nohup python rec_levels.py 150 1350 1 >1350 & bash'"

ssh -t tomo@tomo1 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/experimental/AtomiumS2/;ls; \\
nohup python rec_levels.py 150 1500 0 >1500 & nohup python rec_levels.py 150 1650 1 >1650 & bash'"






ssh -t tomo@tomo4 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/experimental/AtomiumS2/;ls; \\
nohup python rec_levels_frame.py 150 0 0 >0 & nohup python rec_levels_frame.py 150 150 1 >150 & nohup python rec_levels_frame.py 150 300 2 >300 & nohup python rec_levels_frame.py 150 450 3 >450 & bash'"



ssh -t tomo@tomo5 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/experimental/AtomiumS2/;ls; \\
nohup python rec_levels_frame.py 150 600 0 >600 & nohup python rec_levels_frame.py 150 750 1 >750 & nohup python rec_levels_frame.py 150 900 2 >900 & nohup python rec_levels_frame.py 150 1050 3 >1050 & bash'"

ssh -t tomo@tomo2 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/experimental/AtomiumS2/;ls; \\
nohup python rec_levels_frame.py 150 1200 0 >1200 & nohup python rec_levels_frame.py 150 1350 1 >1350 & bash'"

ssh -t tomo@tomo1 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/experimental/AtomiumS2/;ls; \\
nohup python rec_levels_frame.py 150 1500 0 >1500 & nohup python rec_levels_frame.py 150 1650 1 >1650 & bash'"




# lap1=float(sys.argv[1])
# lap2=float(sys.argv[2])
# gpu = int(sys.argv[6])
# frame = sys.argv[4]=='True'
# prbframe = sys.argv[5]=='True'
# pad = int(sys.argv[6])
# gltype = sys.argv[7]
# gpu = int(sys.argv[8])

ssh -t tomo@tomo5 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/work_esrf/;ls; \\
nohup python rec_nfp_siemens_reg_grad_frame_gpu.py 1e-1 0 True True 256 lap 0 >c1 & nohup python rec_nfp_siemens_reg_grad_frame_gpu.py 1e-1 0 False True 256 lap 1 >c2 & nohup python rec_nfp_siemens_reg_grad_frame_gpu.py 1e-1 0 True False 256 lap 2 >c3 & nohup python rec_nfp_siemens_reg_grad_frame_gpu.py 1e-1 0 False False 256 lap 3 >c4 & bash'"

ssh -t tomo@tomo1 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/work_esrf/;ls; \\
nohup python rec_nfp_siemens_reg_grad_frame_gpu.py 1e-1 0 False False 256 grad 0 >c5 & nohup python rec_nfp_siemens_reg_grad_frame_gpu.py 1e-1 0 False True 256 grad 1 >c6 & bash'"

ssh -t tomo@tomo2 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/work_esrf/;ls; \\
nohup python rec_nfp_siemens_reg_grad_frame_gpu.py 1e-1 0 False False 128 lap 0 >c7 & nohup python rec_nfp_siemens_reg_grad_frame_gpu.py 1e-1 0 False True 128 lap 1 >c8 & bash'"

ssh -t tomo@tomo4 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/work_esrf/;ls; \\
nohup python rec_nfp_siemens_reg_grad_frame_gpu.py 1e-1 0 True True 512 lap 0 >c9 & nohup python rec_nfp_siemens_reg_grad_frame_gpu.py 1e-1 0 False True 512 lap 1 >c10 & nohup python rec_nfp_siemens_reg_grad_frame_gpu.py 1e-1 0 True False 512 lap 2 >c11 & nohup python rec_nfp_siemens_reg_grad_frame_gpu.py 1e-1 0 False False 512 lap 3 >c12 & bash'"
