# ssh -t tomo@tomo1 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
# cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/march2025/Chip;ls; \\
# nohup python rec_3d.py 1 1600 8 2 100 >r_1_1600_100 & bash'"

ssh -t tomo@tomo5 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/march2025/Chip1800;ls; \\
nohup python rec_3d_pad.py >r_1 & bash'"

ssh -t tomo@tomo3 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/march2025/Chip1800;ls; \\
nohup python rec_3d_pad.py >r_2 & bash'"

# ssh -t tomo@tomo2 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
# cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/march2025/Chip;ls; \\
# nohup python rec_3d.py 2 1600 8 2 100 >r_2_1600_100 & bash'"

