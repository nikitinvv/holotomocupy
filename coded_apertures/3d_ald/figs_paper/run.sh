CUDA_VISIBLE_DEVICES=0 nohup python rec_reprojection_codes.py 180 &
CUDA_VISIBLE_DEVICES=1 nohup python rec_reprojection_codes.py 360 &
CUDA_VISIBLE_DEVICES=2 nohup python rec_reprojection_codes.py 540 &

CUDA_VISIBLE_DEVICES=0 nohup python rec_reprojection.py 1 &
CUDA_VISIBLE_DEVICES=1 nohup python rec_reprojection.py 2 &
CUDA_VISIBLE_DEVICES=2 nohup python rec_reprojection.py 3 &


#tomo4
CUDA_VISIBLE_DEVICES=2 nohup python rec_reprojection.py 2 &
CUDA_VISIBLE_DEVICES=3 nohup python rec_reprojection.py 3 &
#tomo5
CUDA_VISIBLE_DEVICES=3 nohup python rec_reprojection.py 1 &
#
CUDA_VISIBLE_DEVICES=0 nohup python rec_reprojection_codes.py 180 &
CUDA_VISIBLE_DEVICES=1 nohup python rec_reprojection_codes.py 360 &
CUDA_VISIBLE_DEVICES=0 nohup python rec_reprojection_codes.py 540 &
