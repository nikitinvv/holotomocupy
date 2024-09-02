
python modeling.py 1 0; 
python modeling.py 2 0;
python modeling.py 3 0;
python modeling.py 1 1999;
python modeling.py 2 1999;
python modeling.py 3 1999;


python modeling_codes.py 180 0 -12e-3 ; 
python modeling_codes.py 360 0 -12e-3 ; 
python modeling_codes.py 540 0 -12e-3 ; 
python modeling_codes.py 180 2001 -12e-3 ; 
python modeling_codes.py 360 2001 -12e-3 ; 
python modeling_codes.py 540 2001 -12e-3 ; 

python modeling_codes_correct.py 180 1999 -12e-3 ; 
python modeling_codes_correct.py 360 1999 -12e-3 ; 
python modeling_codes_correct.py 540 1999 -12e-3 ; 



ssh -t tomo@tomo5 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/3d_ald/figs_paper;ls; \\
CUDA_VISIBLE_DEVICES=0 nohup python rec_reprojection_codes.py 360 1000 -12e-3   & CUDA_VISIBLE_DEVICES=1 nohup python rec_reprojection_codes.py 360 1500 -12e-3   & CUDA_VISIBLE_DEVICES=2 nohup python rec_reprojection.py 3 1000 -12e-3   & CUDA_VISIBLE_DEVICES=3 nohup python rec_reprojection.py 3 1500 -12e-3   & bash'"


ssh -t tomo@tomo4 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/3d_ald/figs_paper;ls; \\
CUDA_VISIBLE_DEVICES=0 nohup python rec_reprojection_codes.py 180 1000 -12e-3   & CUDA_VISIBLE_DEVICES=1 nohup python rec_reprojection_codes.py 180 1500 -12e-3   & CUDA_VISIBLE_DEVICES=2 nohup python rec_reprojection.py 2 1000 -12e-3   & CUDA_VISIBLE_DEVICES=3 nohup python rec_reprojection.py 2 1500 -12e-3   & bash'"

ssh -t tomo@tomo3 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/3d_ald/figs_paper;ls; \\
CUDA_VISIBLE_DEVICES=0 python rec_reprojection_codes.py 540 1000 -12e-3 ; CUDA_VISIBLE_DEVICES=0 python rec_reprojection_codes.py 540 1500 -12e-3   & bash'"



ssh -t tomo@tomo1 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/3d_ald/figs_paper;ls; \\
CUDA_VISIBLE_DEVICES=0 nohup python rec_reprojection.py 3 0 -12e-3  >3_0 & CUDA_VISIBLE_DEVICES=1 nohup python rec_reprojection.py 3 1500 -12e-3  >3_1500 & bash '"

ssh -t tomo@tomo4 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/3d_ald/figs_paper;ls; \\
CUDA_VISIBLE_DEVICES=0 nohup python rec_reprojection.py 1 0 -12e-3  >1_0 & CUDA_VISIBLE_DEVICES=1 nohup python rec_reprojection.py 1 1500 -12e-3  >1_1500 & bash '"

ssh -t tomo@tomo4 "bash -c 'source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/3d_ald/figs_paper;ls; \\
CUDA_VISIBLE_DEVICES=2 nohup python rec_reprojection.py 2 0 -12e-3  >2_0 & CUDA_VISIBLE_DEVICES=3 nohup python rec_reprojection.py 2 1500 -12e-3  >2_1500 & bash '"

ssh -t tomo@tomo2 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/3d_ald/figs_paper;ls; \\
CUDA_VISIBLE_DEVICES=0 nohup python rec_reprojection_codes.py 360 0 -12e-3  >360_0 & CUDA_VISIBLE_DEVICES=1 nohup python rec_reprojection_codes.py 360 1500 -12e-3  >360_1500 & bash '"


ssh -t tomo@tomo5 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/3d_ald/figs_paper;ls; \\
CUDA_VISIBLE_DEVICES=0 nohup python rec_reprojection_codes.py 180 0 -12e-3   & CUDA_VISIBLE_DEVICES=1 nohup python rec_reprojection_codes.py 180 1500 -12e-3   & bash'"


ssh -t tomo@tomo3 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/3d_ald/figs_paper;ls; \\
CUDA_VISIBLE_DEVICES=0 python rec_reprojection_codes.py 540 0 -12e-3 ; CUDA_VISIBLE_DEVICES=0 python rec_reprojection_codes.py 540 1500 -12e-3   & bash'"






ssh -t tomo@tomo3 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/3d_ald/figs_paper;ls; \\
CUDA_VISIBLE_DEVICES=0 python rec_reprojection_codes.py 540 2001 -12e-3   & bash'"




ssh -t tomo@tomo5 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/3d_ald/figs_paper;ls; \\
CUDA_VISIBLE_DEVICES=0 nohup python rec_reprojection_codes.py 180 2001 -12e-3   & CUDA_VISIBLE_DEVICES=1 nohup python rec_reprojection_codes.py 360 2001 -12e-3   & CUDA_VISIBLE_DEVICES=2 nohup python rec_reprojection.py 1 2001 -12e-3   & CUDA_VISIBLE_DEVICES=3 nohup python rec_reprojection.py 2 2001 -12e-3   & bash'"

ssh -t tomo@tomo2 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/3d_ald/figs_paper;ls; \\
CUDA_VISIBLE_DEVICES=0 nohup python rec_reprojection.py 3 2001 -12e-3   & bash'"

# ssh -t tomo@tomo2 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
# cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/3d_ald/figs_paper;ls; \\
# CUDA_VISIBLE_DEVICES=0 nohup python rec_reprojection_codes.py 360 0 17e-3 >17e-3 & CUDA_VISIBLE_DEVICES=1 nohup python rec_reprojection_codes.py 360 0 13e-3  >13e-3 & bash '"

# ssh -t tomo@tomo4 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
# cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/3d_ald/figs_paper;ls; \\
# CUDA_VISIBLE_DEVICES=0 nohup python rec_reprojection_codes.py 360 0 -12e-3  >-12e-3  & CUDA_VISIBLE_DEVICES=1 nohup python rec_reprojection_codes.py 360 0 14e-3  >14e-3 CUDA_VISIBLE_DEVICES=2 nohup python rec_reprojection_codes.py 360 0 16e-3 >16e-3 & CUDA_VISIBLE_DEVICES=3 nohup python rec_reprojection_codes.py 360 0 112e-3  >112e-3 & bash '"

# ssh -t tomo@tomo5 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
# cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/3d_ald/figs_paper;ls; \\
# CUDA_VISIBLE_DEVICES=0 nohup python rec_reprojection_codes.py 360 0 20e-3 >20e-3 & CUDA_VISIBLE_DEVICES=1 nohup python rec_reprojection_codes.py 360 0 22e-3  >22e-3 CUDA_VISIBLE_DEVICES=2 nohup python rec_reprojection_codes.py 360 0 24e-3 >24e-3 & CUDA_VISIBLE_DEVICES=3 nohup python rec_reprojection_codes.py 360 0 26e-3  >26e-3 & bash '"


# ssh -t tomo@tomo3 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
# cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/3d_ald/figs_paper;ls; \\
# CUDA_VISIBLE_DEVICES=0 nohup python rec_reprojection_codes.py 360 0 16e-3 >16e-3 & bash '"

ssh -t tomo@tomo2 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/3d_ald/figs_paper;ls; \\
CUDA_VISIBLE_DEVICES=0 nohup python rec_reprojection_codes_correct.py 180 1999 -12e-3   & CUDA_VISIBLE_DEVICES=1 nohup python rec_reprojection_codes_correct.py 360 1999 -12e-3   & bash'"
