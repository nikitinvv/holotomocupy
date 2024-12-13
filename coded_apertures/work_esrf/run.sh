nohup python rec_nfp_siemens_reg_lap_gpu.py 0 0 0 &
nohup python rec_nfp_siemens_reg_lap_gpu.py 1e-1 0 1 &
nohup python rec_nfp_siemens_reg_lap_gpu.py 5e-1 0 2 &
nohup python rec_nfp_siemens_reg_lap_gpu.py 1e-2 0 3 & 

nohup python rec_nfp_siemens_reg_grad_gpu.py 0 0 0 & 
nohup python rec_nfp_siemens_reg_grad_gpu.py 1e-1 0 1 & 
nohup python rec_nfp_siemens_reg_grad_gpu.py 5e-1 0 2 & 
nohup python rec_nfp_siemens_reg_grad_gpu.py 1e-2 0 3 & 

nohup python rec_nfp_siemens_reg_lap_gpu.py 5e-2 0 0 & 
nohup python rec_nfp_siemens_reg_grad_gpu.py 5e-2 0 1 & 

nohup python rec_nfp_siemens_reg_lap_gpu.py 1 0 0 & 
nohup python rec_nfp_siemens_reg_grad_gpu.py 1 0 1 & 

nohup python rec_nfp_siemens_reg_lap_gpu.py 1 0 0 & 
nohup python rec_nfp_siemens_reg_grad_gpu.py 1 0 1 & 

