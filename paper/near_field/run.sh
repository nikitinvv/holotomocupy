
ssh -t tomo@tomo4 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/paper/near_field/;ls; \\
nohup python rec_perf_all.py True True True BH-GD 514 0 >/dev/null & \\
nohup python rec_perf_all.py True True True BH-CG 514 1 >/dev/null & \\
nohup python rec_perf_all.py True True False BH-GD 4097 2 >/dev/null & \\
nohup python rec_perf_all.py True True False BH-CG 4097 3 >/dev/null & \\
nohup python rec_perf_all.py True False True BH-GD 514 0 >/dev/null & \\
nohup python rec_perf_all.py True False True BH-CG 514 1 >/dev/null & \\
nohup python rec_perf_all.py True False False BH-GD 4097 2 >/dev/null & \\
nohup python rec_perf_all.py True False False BH-CG 4097 3 >/dev/null & bash'"


ssh -t tomo@tomo2 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/paper/near_field/;ls; \\
nohup python rec_perf_all.py False False True BH-GD 514 0 >/dev/null & \\
nohup python rec_perf_all.py False False True BH-CG 514 1 >/dev/null & \\
nohup python rec_perf_all.py False False False BH-GD 514 0 >/dev/null & \\
nohup python rec_perf_all.py False False False BH-CG 514 1 >/dev/null & bash'"


ssh -t tomo@tomo5 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate ptychi; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/paper/near_field/;ls; \\
nohup python rec_ptychi.py True True True epie 514 0 >/dev/null & \\
nohup python rec_ptychi.py True True True lsqml 514 1 >/dev/null & \\
nohup python rec_ptychi.py True True False epie 4097 2 >/dev/null & \\
nohup python rec_ptychi.py True True False lsqml 4097 3 >/dev/null & \\
nohup python rec_ptychi.py True False True epie 514 0 >/dev/null & \\
nohup python rec_ptychi.py True False True lsqml 514 1 >/dev/null & \\
nohup python rec_ptychi.py True False False epie 4097 2 >/dev/null & \\
nohup python rec_ptychi.py True False False lsqml 4097 3 >/dev/null & bash'"


ssh -t tomo@tomo1 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate ptychi; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/paper/near_field/;ls; \\
nohup python rec_ptychi.py False False True epie 514 0 >/dev/null & \\
nohup python rec_ptychi.py False False True lsqml 514 1 >/dev/null & \\
nohup python rec_ptychi.py False False False epie 514 0 >/dev/null & \\
nohup python rec_ptychi.py False False False lsqml 514 1 >/dev/null & bash'"

ssh -t tomo@tomo4 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/paper/near_field/;ls; \\
nohup python rec_perf_all.py True True True DY-LS 514 0 >/dev/null & \\
nohup python rec_perf_all.py True True False DY-LS 4097 1 >/dev/null & \\
nohup python rec_perf_all.py True False True DY-LS 514 2 /dev/null & \\
nohup python rec_perf_all.py True False False DY-LS 4097 3 >/dev/null & bash'"

ssh -t tomo@tomo2 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/paper/near_field/;ls; \\
nohup python rec_perf_all.py False False True DY-LS  514 0 >/dev/null & \\
nohup python rec_perf_all.py False False False DY-LS 514 0 >/dev/null & bash'"




# ssh -t tomo@tomo5 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate ptychi; \\
# cd /home/beams/TOMO/vnikitin/holotomocupy/paper/near_field/;ls; \\
# nohup python rec_ptychi.py True True True epie 10241 0 >/dev/null & \\
# nohup python rec_ptychi.py True True True lsqml 10241 1 >/dev/null & \\
# nohup python rec_ptychi.py True True False epie 10241 2 >/dev/null & \\
# nohup python rec_ptychi.py True True False lsqml 10241 3 >/dev/null & bash'"



# ssh -t tomo@tomo4 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
# cd /home/beams/TOMO/vnikitin/holotomocupy/paper/near_field/;ls; \\
# nohup python rec_perf_all.py True True True BH-GD 10241 0 >/dev/null & \\
# nohup python rec_perf_all.py True True True BH-CG 10241 1 >/dev/null & \\
# nohup python rec_perf_all.py True True False BH-GD 10241 2 >/dev/null & \\
# nohup python rec_perf_all.py True True False BH-CG 10241 3 >/dev/null & bash'"

# ssh -t tomo@tomo4 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
# cd /home/beams/TOMO/vnikitin/holotomocupy/paper/near_field/;ls; \\
# nohup python rec_perf_all.py True True True BH-GD 10241 0 >/dev/null & \\
# nohup python rec_perf_all.py True True True BH-CG 10241 1 >/dev/null & \\
# nohup python rec_perf_all.py True True False BH-GD 10241 2 >/dev/null & \\
# nohup python rec_perf_all.py True True False BH-CG 10241 3 >/dev/null & \\
# nohup python rec_perf_all.py True False True BH-GD 10241 0 >/dev/null & \\
# nohup python rec_perf_all.py True False True BH-CG 10241 1 >/dev/null & \\
# nohup python rec_perf_all.py True False False BH-GD 10241 2 >/dev/null & \\
# nohup python rec_perf_all.py True False False BH-CG 10241 3 >/dev/null & bash'"


# ssh -t tomo@tomo5 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate ptychi; \\
# cd /home/beams/TOMO/vnikitin/holotomocupy/paper/near_field/;ls; \\
# nohup python rec_ptychi.py True True True epie 10241 0 >/dev/null & \\
# nohup python rec_ptychi.py True True True lsqml 10241 1 >/dev/null & \\
# nohup python rec_ptychi.py True True False epie 10241 2 >/dev/null & \\
# nohup python rec_ptychi.py True True False lsqml 10241 3 >/dev/null & bash'"


# ssh -t tomo@tomo4 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
# cd /home/beams/TOMO/vnikitin/holotomocupy/paper/near_field/;ls; \\
# nohup python rec_perf_all.py True True True BH-GD 10241 0 >/dev/null & \\
# nohup python rec_perf_all.py True True True BH-CG 10241 1 >/dev/null & \\
# nohup python rec_perf_all.py True True False BH-GD 10241 2 >/dev/null & \\
# nohup python rec_perf_all.py True True False BH-CG 10241 3 >/dev/null & \\
# nohup python rec_perf_all.py True False True BH-GD 10241 0 >/dev/null & \\
# nohup python rec_perf_all.py True False True BH-CG 10241 1 >/dev/null & \\
# nohup python rec_perf_all.py True False False BH-GD 10241 2 >/dev/null & \\
# nohup python rec_perf_all.py True False False BH-CG 10241 3 >/dev/null & bash'"

# pkill -9 python; source ~/.bashrc; conda activate holotomocupy;
# python rec_perf_all.py True True False BH-GD 10241 3;
# pkill -9 python; source ~/.bashrc; conda activate holotomocupy;
# python rec_perf_all.py True True False BH-CG 10241 3;
# pkill -9 python; source ~/.bashrc; conda activate ptychi;
# python rec_ptychi.py True True False epie 10241 3;
# pkill -9 python; source ~/.bashrc; conda activate ptychi;
# python rec_ptychi.py True True False lsqml 10241 3;