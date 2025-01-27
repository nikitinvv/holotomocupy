
ssh -t tomo@tomo4 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/work_esrf/rec_nfp_ca/;ls; \\
nohup python rec_ca_test_Fresnel.py freq 48 164 256 8193 0 >/dev/null & \\
nohup python rec_ca_test_Fresnel.py freq 48 164 256 8193 1 >/dev/null & \\
nohup python rec_ca_test_Fresnel.py freq 48 164 256 8193 2 >/dev/null & \\
nohup python rec_ca_test_Fresnel.py freq 48 164 256 8193 3 >/dev/null & bash'"

ssh -t tomo@tomo5 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/work_esrf/rec_nfp_ca/;ls; \\
nohup python rec_ca_test_Fresnel.py space 48 164 256 8193 0 >/dev/null & \\
nohup python rec_ca_test_Fresnel.py space 48 164 128 8193 1 >/dev/null & \\
nohup python rec_ca_test_Fresnel.py space 48 164 64 8193 2 >/dev/null & \\
nohup python rec_ca_test_Fresnel.py space 48 164 32 8193 3 >/dev/null & bash'"

ssh -t tomo@tomo1 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/work_esrf/rec_nfp_ca/;ls; \\
nohup python rec_ca_test_Fresnel.py space 48 144 128 8193 0 >/dev/null & \\
nohup python rec_ca_test_Fresnel.py space 48 124 128 8193 1 >/dev/null & bash'"


ssh -t tomo@tomo2 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/work_esrf/rec_nfp_ca/;ls; \\
nohup python rec_ca_test_Fresnel.py space 48 104 128 8193 0 >/dev/null & \\
nohup python rec_ca_test_Fresnel.py space 48 84 128 8193 1 >/dev/null & bash'"

ssh -t tomo@tomo3 "bash -c 'pkill -9 python; source ~/.bashrc; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/coded_apertures/work_esrf/rec_nfp_ca/;ls; \\
nohup python rec_ca_test_Fresnel.py space 48 44 128 8193 0 >/dev/null & \\
nohup python rec_ca_test_Fresnel.py space 48 204 128 8193 0 >/dev/null & bash'"
