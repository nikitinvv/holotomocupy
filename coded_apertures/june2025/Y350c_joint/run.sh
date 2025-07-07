
ssh -t tomo@tomo1 "bash -c 'source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/june2025/Y350c_joint;ls; \\
nohup python step4_rec_real_init.py 120 >s4_120 & bash'" 






ssh -t tomo@tomo5 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/june2025/Y350c_joint;ls; \\
nohup python step5_rec_real.py 15 0 100 1 >full_15_0_100_1 & bash'" 


ssh -t tomo@tomo2 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/june2025/Y350c_joint;ls; \\
nohup python step5_rec_real.py 0 0 100 1 >full_0_0_100_1 & bash'" 


ssh -t tomo@tomo1 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/june2025/Y350c_joint;ls; \\
nohup python step5_rec_real.py 15 0 110 1 >full_15_0_110_1 & bash'" 














ssh -t tomo@tomo1 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/june2025/Y350c_joint;ls; \\
nohup python step5_rec_real.py 15 0 100 10 >ss4_15_0_100_10 & bash'" 

ssh -t tomo@tomo2 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/june2025/Y350c_joint;ls; \\
nohup python step5_rec_real.py 15 0 70 10 >ss4_15_0_70_10 & bash'" 

ssh -t tomo@tomo4 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/june2025/Y350c_joint;ls; \\
nohup python step5_rec_real.py 15 0 110 10 >ss4_15_0_110_10 & bash'" 


ssh -t tomo@tomo1 "bash -c 'source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/june2025/Y350c_joint;ls; \\
nohup python step5_rec_real.py 15 0 120 10 >ss4_15_0_120_10 & bash'" 

ssh -t tomo@tomo2 "bash -c 'source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/june2025/Y350c_joint;ls; \\
nohup python step5_rec_real.py 15 0 95 10 >ss4_15_0_95_10 & bash'" 


ssh -t tomo@tomo5 "bash -c 'source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/june2025/Y350c_joint;ls; \\
nohup python step5_rec_real.py 15 0 90 10 >ss4_15_0_90_10 & bash'" 


ssh -t tomo@tomo1 "bash -c 'source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/june2025/Y350c_joint;ls; \\
nohup python step5_rec_real.py 15 0 95 10 >ss4_15_0_95_10 & bash'" 


ssh -t tomo@tomo1 "bash -c 'source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/june2025/Y350c_joint;ls; \\
nohup python step5_rec_real.py 15 0 75 10 >ss4_15_0_75_10 & bash'" 

ssh -t tomo@tomo5 "bash -c 'source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/june2025/Y350c_joint;ls; \\
nohup python step5_rec_real.py 15 0 105 10 >ss4_15_0_105_10 & bash'" 