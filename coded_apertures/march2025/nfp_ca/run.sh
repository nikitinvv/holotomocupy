ssh -t tomo@tomo4 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/BH-ptychography/rec_nfp_ca/;ls; \\
nohup python demo_object_probe_positions_large_chunk_ca_reg.py 4 0.05 256 >256 & bash'"

ssh -t tomo@tomo5 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/BH-ptychography/rec_nfp_ca/;ls; \\
nohup python demo_object_probe_positions_large_chunk_ca_reg.py 4 0.1 512 >512 & bash'"

ssh -t tomo@tomo1 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/BH-ptychography/rec_nfp_ca/;ls; \\
nohup python demo_object_probe_positions_large_chunk_ca_reg.py 2 0.1 128 >128 & bash'"

ssh -t tomo@tomo2 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/BH-ptychography/rec_nfp_ca/;ls; \\
nohup python demo_object_probe_positions_large_chunk_ca_reg.py 2 0.1 64 >64 & bash'"

ssh -t tomo@tomo3 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/BH-ptychography/rec_nfp_ca/;ls; \\
nohup python demo_object_probe_positions_large_chunk_ca_reg.py 1 0.1 512 >512 & bash'"



