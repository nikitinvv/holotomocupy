CUDA_VISIBLE_DEVICES=0 nohup python rec_fifth_review.py 0 0.2 128 360 &
CUDA_VISIBLE_DEVICES=1 nohup python rec_fifth_review.py 0 0.05 128 360 &

CUDA_VISIBLE_DEVICES=0 nohup python rec_fifth_review.py 0 0.1 128 360 &
CUDA_VISIBLE_DEVICES=1 nohup python rec_fifth_review.py 0 0.02 128 360 &


CUDA_VISIBLE_DEVICES=0 nohup python rec_fifth_review.py 1 0 256 360 &
CUDA_VISIBLE_DEVICES=1 nohup python rec_fifth_review.py 0.5 0 256 360 &

CUDA_VISIBLE_DEVICES=0 nohup python rec_fifth_review.py 2 0 256 360 &
CUDA_VISIBLE_DEVICES=1 nohup python rec_fifth_review.py 0.25 0 256 360 &



CUDA_VISIBLE_DEVICES=0 nohup python rec_fifth_review.py 0 0 128 540 &
CUDA_VISIBLE_DEVICES=1 nohup python rec_fifth_review.py 0 0.05 128 540 &

CUDA_VISIBLE_DEVICES=0 nohup python rec_fifth_review.py 0 0.1 128 540 &
CUDA_VISIBLE_DEVICES=1 nohup python rec_fifth_review.py 0 0.02 128 540 &


CUDA_VISIBLE_DEVICES=0 nohup python rec_fifth_review.py 1 0 128 540 &
CUDA_VISIBLE_DEVICES=1 nohup python rec_fifth_review.py 0.5 0 128 540 &

CUDA_VISIBLE_DEVICES=0 nohup python rec_fifth_review.py 2 0 128 540 &
CUDA_VISIBLE_DEVICES=1 nohup python rec_fifth_review.py 0.25 0 128 540 &

