
ssh -t tomo@tomo4 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/paper/near_field/;ls; \\
nohup python rec_perf.py True True True gd 10001 0 >/dev/null & \\
nohup python rec_perf.py True True True cg 10001 1 >/dev/null & \\
nohup python rec_perf.py True True False gd 10001 2 >/dev/null & \\
nohup python rec_perf.py True True False cg 10001 3 >/dev/null & \\
nohup python rec_perf.py True False True gd 10001 0 >/dev/null & \\
nohup python rec_perf.py True False True cg 10001 1 >/dev/null & \\
nohup python rec_perf.py True False False gd 10001 2 >/dev/null & \\
nohup python rec_perf.py True False False cg 10001 3 >/dev/null & bash'"


ssh -t tomo@tomo2 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/paper/near_field/;ls; \\
nohup python rec_perf.py False False True gd 10001 0 >/dev/null & \\
nohup python rec_perf.py False False True cg 10001 1 >/dev/null & \\
nohup python rec_perf.py False False False gd 10001 0 >/dev/null & \\
nohup python rec_perf.py False False False cg 10001 1 >/dev/null & bash'"


ssh -t tomo@tomo5 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate ptychi; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/paper/near_field/;ls; \\
nohup python rec_ptychi.py True True True epie 10001 0 >/dev/null & \\
nohup python rec_ptychi.py True True True lsqml 10001 1 >/dev/null & \\
nohup python rec_ptychi.py True True False epie 10001 2 >/dev/null & \\
nohup python rec_ptychi.py True True False lsqml 10001 3 >/dev/null & \\
nohup python rec_ptychi.py True False True epie 10001 0 >/dev/null & \\
nohup python rec_ptychi.py True False True lsqml 10001 1 >/dev/null & \\
nohup python rec_ptychi.py True False False epie 10001 2 >/dev/null & \\
nohup python rec_ptychi.py True False False lsqml 10001 3 >/dev/null & bash'"


ssh -t tomo@tomo1 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate ptychi; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/paper/near_field/;ls; \\
nohup python rec_ptychi.py False False True epie 10001 0 >/dev/null & \\
nohup python rec_ptychi.py False False True lsqml 10001 1 >/dev/null & \\
nohup python rec_ptychi.py False False False epie 10001 0 >/dev/null & \\
nohup python rec_ptychi.py False False False lsqml 10001 1 >/dev/null & bash'"

# ssh -t tomo@tomo5 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate ptychi; \\
# cd /home/beams/TOMO/vnikitin/holotomocupy/paper/near_field/;ls; \\
# nohup python rec_ptychi.py True True True epie 10001 0 >/dev/null & \\
# nohup python rec_ptychi.py True True True lsqml 10001 1 >/dev/null & \\
# nohup python rec_ptychi.py True True False epie 10001 2 >/dev/null & \\
# nohup python rec_ptychi.py True True False lsqml 10001 3 >/dev/null & bash'"



# ssh -t tomo@tomo4 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
# cd /home/beams/TOMO/vnikitin/holotomocupy/paper/near_field/;ls; \\
# nohup python rec_perf.py True True True gd 10001 0 >/dev/null & \\
# nohup python rec_perf.py True True True cg 10001 1 >/dev/null & \\
# nohup python rec_perf.py True True False gd 10001 2 >/dev/null & \\
# nohup python rec_perf.py True True False cg 10001 3 >/dev/null & bash'"

# ssh -t tomo@tomo4 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
# cd /home/beams/TOMO/vnikitin/holotomocupy/paper/near_field/;ls; \\
# nohup python rec_perf.py True True True gd 10001 0 >/dev/null & \\
# nohup python rec_perf.py True True True cg 10001 1 >/dev/null & \\
# nohup python rec_perf.py True True False gd 10001 2 >/dev/null & \\
# nohup python rec_perf.py True True False cg 10001 3 >/dev/null & \\
# nohup python rec_perf.py True False True gd 10001 0 >/dev/null & \\
# nohup python rec_perf.py True False True cg 10001 1 >/dev/null & \\
# nohup python rec_perf.py True False False gd 10001 2 >/dev/null & \\
# nohup python rec_perf.py True False False cg 10001 3 >/dev/null & bash'"


# ssh -t tomo@tomo5 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate ptychi; \\
# cd /home/beams/TOMO/vnikitin/holotomocupy/paper/near_field/;ls; \\
# nohup python rec_ptychi.py True True True epie 10001 0 >/dev/null & \\
# nohup python rec_ptychi.py True True True lsqml 10001 1 >/dev/null & \\
# nohup python rec_ptychi.py True True False epie 10001 2 >/dev/null & \\
# nohup python rec_ptychi.py True True False lsqml 10001 3 >/dev/null & bash'"


# ssh -t tomo@tomo4 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
# cd /home/beams/TOMO/vnikitin/holotomocupy/paper/near_field/;ls; \\
# nohup python rec_perf.py True True True gd 10001 0 >/dev/null & \\
# nohup python rec_perf.py True True True cg 10001 1 >/dev/null & \\
# nohup python rec_perf.py True True False gd 10001 2 >/dev/null & \\
# nohup python rec_perf.py True True False cg 10001 3 >/dev/null & \\
# nohup python rec_perf.py True False True gd 10001 0 >/dev/null & \\
# nohup python rec_perf.py True False True cg 10001 1 >/dev/null & \\
# nohup python rec_perf.py True False False gd 10001 2 >/dev/null & \\
# nohup python rec_perf.py True False False cg 10001 3 >/dev/null & bash'"

# pkill -9 python; source ~/.bashrc; conda activate holotomocupy;
# python rec_perf.py True True False gd 10001 3;
# pkill -9 python; source ~/.bashrc; conda activate holotomocupy;
# python rec_perf.py True True False cg 10001 3;
# pkill -9 python; source ~/.bashrc; conda activate ptychi;
# python rec_ptychi.py True True False epie 10001 3;
# pkill -9 python; source ~/.bashrc; conda activate ptychi;
# python rec_ptychi.py True True False lsqml 10001 3;