 
python modeling.py 360 True 11e-3 
python modeling.py 360 False 12e-3 
python modeling.py 360 True 24e-3 
python modeling.py 360 False 24e-3 
python modeling.py 360 True 8e-3 
python modeling.py 360 False 8e-3 
python modeling.py 360 True 48e-3 
python modeling.py 360 False 48e-3 
python modeling.py 180 True 12e-3 
python modeling.py 180 False 12e-3
 #tomo3
  CUDA_VISIBLE_DEVICES=0 nohup python rec_reprojection.py 360 True 12e-3 &

 # tomo1
 CUDA_VISIBLE_DEVICES=0 nohup python rec_reprojection.py 180 True 12e-3 &
 CUDA_VISIBLE_DEVICES=1 nohup python rec_reprojection.py 360 True 24e-3 &
 
 

 # tomo2
 CUDA_VISIBLE_DEVICES=0 nohup python rec_reprojection.py 180 False 12e-3 &
 CUDA_VISIBLE_DEVICES=1 nohup python rec_reprojection.py 360 False 24e-3 &

 # tomo4
CUDA_VISIBLE_DEVICES=3 nohup python rec_reprojection.py 360 False 12e-3 &
CUDA_VISIBLE_DEVICES=1 nohup python rec_reprojection.py 360 True 8e-3 &
CUDA_VISIBLE_DEVICES=2 nohup python rec_reprojection.py 360 False 8e-3 &

 # tomo5
CUDA_VISIBLE_DEVICES=3 nohup python rec_reprojection.py 360 True 48e-3 &
CUDA_VISIBLE_DEVICES=2 nohup python rec_reprojection.py 360 False 48e-3 &

  
