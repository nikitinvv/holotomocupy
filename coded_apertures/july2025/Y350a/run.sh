
ssh -t tomo@tomo5 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/july2025/Y350a;ls; \\
nohup python step5_rec_real.py 15 0 120 4 >full_15_0_120_4 & bash'" 