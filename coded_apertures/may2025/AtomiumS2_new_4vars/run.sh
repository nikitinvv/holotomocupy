ssh -t tomo@tomo1 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/may2025/AtomiumS2_new_4vars;ls; \\
nohup python step6_rec.py 1 5400 2 30 0 >r_1_5400_30_0 & bash'"

ssh -t tomo@tomo2 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/may2025/AtomiumS2_new_4vars;ls; \\
nohup python step6_rec.py 0 6400 2 30 0 >r_0_6400_30_0 & bash'"

ssh -t tomo@tomo5 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/may2025/AtomiumS2_new_4vars;ls; \\
nohup python step6_rec.py 0 5400 4 30 50 >r_0_5400_30_50 & bash'"

ssh -t tomo@tomo3 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/may2025/AtomiumS2_new_4vars;ls; \\
nohup python step6_rec.py 1 2700 1 30 50 >r_1_2700_30_50 & bash'"


ssh -t tomo@tomo3 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/may2025/AtomiumS2_new_4vars;ls; \\
nohup python step6_rec.py 1 2700 1 30 50 >r_1_2700_30_50 & bash'"



ssh -t tomo@tomo5 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/may2025/AtomiumS2_new_4vars;ls; \\
nohup python step4_find_shifts.py 4096 4 & bash'"


ssh -t tomo@tomo1 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/may2025/AtomiumS2_new_4vars;ls; \\
nohup python step4_find_shifts.py 64 2 & bash'"


ssh -t tomo@tomo2 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/may2025/AtomiumS2_new_4vars;ls; \\
nohup python step4_find_shifts.py 128 2 & bash'"



ssh -t tomo@tomo3 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/may2025/AtomiumS2_new_4vars;ls; \\
nohup python step4_find_shifts.py 8160 1 & bash'"

ssh -t tomo@tomo4 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/may2025/AtomiumS2_new_4vars;ls; \\
nohup python step4_find_shifts.py 256 4 & bash'"
