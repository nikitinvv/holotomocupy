ssh -t tomo@tomo1 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/march2025/AtomiumS2;ls; \\
nohup python rec_3d.py >r_1 & bash'"

ssh -t tomo@tomo5 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/may2025/AtomiumS2;ls; \\
nohup python rec_3d_approx.py >r_approx & bash'"

ssh -t tomo@tomo4 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/may2025/AtomiumS2;ls; \\
nohup python rec_3d_frommulti.py >r_multi & bash'"

ssh -t tomo@tomo1 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/march2025/AtomiumS2;ls; \\
nohup python rec_3d.py 1 3600 8 1 100 >r_1_3600_100 & bash'"

# ssh -t tomo@tomo5 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
# cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/march2025/AtomiumS2;ls; \\
# nohup python rec_3d.py 1 1800 8 4 100 >r_1_1800_100 & bash'"


# ssh -t tomo@tomo2 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
# cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/march2025/AtomiumS2;ls; \\
# nohup python rec_3d.py 0 3600 4 2 100 >r_0_3600_100 & bash'"