ssh -t tomo@tomo4 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/april2025/AtomiumS2_HT;ls; \\
nohup python step5_rec_real.py 1 4 1 0 30 30 795.5 0 4 >r_1_4_1_0_30_30_795.5_0 & bash'"

ssh -t tomo@tomo5 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/april2025/AtomiumS2_HT;ls; \\
nohup python step5_rec_real.py 1 4 1 1 0 0 795.5 0 4 >r_1_4_1_1_0_0_795.5_0 & bash'"


ssh -t tomo@tomo1 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/april2025/AtomiumS2_HT;ls; \\
nohup python step5_rec_real.py 1 4 1 0 30 30 795.25 0 2 >r_1_4_1_0_30_30_795.25_0 & bash'"

ssh -t tomo@tomo2 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/april2025/AtomiumS2_HT;ls; \\
nohup python step5_rec_real.py 1 4 1 0 30 30 795.25 30 2 >r_1_4_1_0_30_30_795.25_30 & bash'"

ssh -t tomo@tomo3 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/april2025/AtomiumS2_HT;ls; \\
nohup python step5_rec_real.py 1 1 1 0 30 30 795.5 30 1 >r_1_1_1_0_30_30_795.5_30 & bash'"


