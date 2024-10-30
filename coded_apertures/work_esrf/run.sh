CUDA_VISIBLE_DEVICES=0 nohup python rec_nfp_siemens_3p3_gpu.py 0.1 5.5 >log/0.1_5.5 &
CUDA_VISIBLE_DEVICES=1 nohup python rec_nfp_siemens_3p3_gpu.py 0.2 5.5 >log/0.2_5.5 &


CUDA_VISIBLE_DEVICES=0 nohup python rec_nfp_siemens_3p3_gpu.py 0.1 5.6 >log/0.1_5.6 &
CUDA_VISIBLE_DEVICES=1 nohup python rec_nfp_siemens_3p3_gpu.py 0.2 5.6 >log/0.2_5.6 &

CUDA_VISIBLE_DEVICES=0 nohup python rec_nfp_siemens_3p3_gpu.py 0.1 5.4 >log/0.1_5.4 &
CUDA_VISIBLE_DEVICES=1 nohup python rec_nfp_siemens_3p3_gpu.py 0.2 5.4 >log/0.2_5.4 &

CUDA_VISIBLE_DEVICES=0 nohup python rec_nfp_siemens_3p3_gpu.py 0.0 5.5 >log/0_5.5 &
CUDA_VISIBLE_DEVICES=1 nohup python rec_nfp_siemens_3p3_gpu.py 1 5.5 >log/1_5.5 &


