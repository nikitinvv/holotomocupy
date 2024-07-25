tomo4
CUDA_VISIBLE_DEVICES=0 nohup python rec_iterative.py 0 &
CUDA_VISIBLE_DEVICES=1 nohup python rec_iterative.py 1 & 
CUDA_VISIBLE_DEVICES=2 nohup python rec_iterative.py 2 &
CUDA_VISIBLE_DEVICES=3 nohup python rec_iterative.py 3 &

tomo2
CUDA_VISIBLE_DEVICES=0 nohup python rec_iterative.py 4 &
CUDA_VISIBLE_DEVICES=1 nohup python rec_iterative.py 5 &


