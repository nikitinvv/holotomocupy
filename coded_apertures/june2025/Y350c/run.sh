ngpus = int(sys.argv[1])
ndist= int(sys.argv[2])
bin = int(sys.argv[3])
nchunk = int(sys.argv[4])
step = int(sys.argv[5])
ntheta = int(sys.argv[6])
st = int(sys.argv[7])


ssh -t tomo@tomo2 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/june2025/Y350c;ls; \\
nohup python step5_rec_real.py 2 4 2 32 2 100 0 >r_0 & bash'"

ssh -t tomo@tomo5 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/june2025/Y350c;ls; \\
nohup python step5_rec_real.py 4 4 2 32 2 100 1 >r_1 & bash'"

ssh -t tomo@tomo4 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/june2025/Y350c;ls; \\
nohup python step5_rec_real.py 4 4 2 32 2 100 2 >r_2 & bash'"

ssh -t tomo@tomo1 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/june2025/Y350c;ls; \\
nohup python step5_rec_real.py 2 4 2 32 2 100 3 >r_3 & bash'"





ssh -t tomo@tomo2 "bash -c 'source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/june2025/Y350c;ls; \\
nohup python step5_rec_real.py 2 4 2 32 2 100 4 >r_4 & bash'"

ssh -t tomo@tomo5 "bash -c 'source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/june2025/Y350c;ls; \\
nohup python step5_rec_real.py 4 4 2 32 2 100 5 >r_5 & bash'"

ssh -t tomo@tomo4 "bash -c 'source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/june2025/Y350c;ls; \\
nohup python step5_rec_real.py 4 4 2 32 2 100 6 >r_6 & bash'"

ssh -t tomo@tomo1 "bash -c 'source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/june2025/Y350c;ls; \\
nohup python step5_rec_real.py 2 4 2 32 2 100 7 >r_7 & bash'"




ssh -t tomo@tomo2 "bash -c 'source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/june2025/Y350c;ls; \\
nohup python step5_rec_real.py 2 4 2 32 2 100 8 >r_8 & bash'"

ssh -t tomo@tomo5 "bash -c 'source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/june2025/Y350c;ls; \\
nohup python step5_rec_real.py 4 4 2 32 2 100 9 >r_9 & bash'"

ssh -t tomo@tomo4 "bash -c 'source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/june2025/Y350c;ls; \\
nohup python step5_rec_real.py 4 4 2 32 2 100 10 >r_10 & bash'"

ssh -t tomo@tomo1 "bash -c 'source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/june2025/Y350c;ls; \\
nohup python step5_rec_real.py 2 4 2 32 2 100 11 >r_11 & bash'"




####

ssh -t tomo@tomo5 "bash -c 'source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/june2025/Y350c;ls; \\
nohup python step5_rec_real.py 4 4 2 32 2 100 12 >r_12 & bash'"

ssh -t tomo@tomo5 "bash -c 'source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/june2025/Y350c;ls; \\
nohup python step5_rec_real.py 4 4 2 32 2 100 13 >r_13 & bash'"

ssh -t tomo@tomo4 "bash -c 'source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/june2025/Y350c;ls; \\
nohup python step5_rec_real.py 4 4 2 32 2 100 14 >r_14 & bash'"
