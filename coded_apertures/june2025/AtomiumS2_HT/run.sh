ssh -t tomo@tomo2 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/june2025/AtomiumS2_HT;ls; \\
nohup python step5_rec_real_fix_rot.py 50 2 >r_50_2 & bash'"

ssh -t tomo@tomo3 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/june2025/AtomiumS2_HT;ls; \\
nohup python step5_rec_real_fix_rot.py 75 1 25 >r_75_25_mem & bash'"

ssh -t tomo@tomo5 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/june2025/AtomiumS2_HT;ls; \\
nohup python step5_rec_real_fix_rot.py 25 4 >r_25 & bash'"


ssh -t tomo@tomo1 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/june2025/AtomiumS2_HT;ls; \\
nohup python step5_rec_real_fix_rot.py 0 2 50 20 >r_0_50_20 & bash'"


ssh -t tomo@tomo5 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/june2025/AtomiumS2_HT;ls; \\
nohup python step5_rec_real_fix_rot_continue.py 1000 4 50 20 >r_1000_50_20 & bash'"
