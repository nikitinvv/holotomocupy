# Installation
conda create -n holotomocupy -c conda-forge cupy dxchange setuptools matplotlib psutil jupyter matplotlib-scalebar
conda activate holotomocupy
cd holotomocupy; pip instal .

# Run:
cd experimental/nvidia
python rec_iterative.py config1.conf

#Output example:

python rec_iterative.py config1.conf 
12:46:41 ngpus=4 n=512 ntheta=900 iter=0 err=6.88578e-02
12:47:04 ngpus=4 n=512 ntheta=900 iter=1 err=1.06501e-02
12:47:19 ngpus=4 n=512 ntheta=900 iter=2 err=4.94378e-03

In config1.conf file please change in_file name and output folder.

For performance tests adjust:

ngpus=4 # number of gpus
nchunk=16 # chunk size to fit to GPU
bin=2 # data binning (0,1,2)
ntheta=900 # number of projection angles (max 4500)
niter=3 # Number of iterations


Notes for NVIDIA:
Data sizes are controlled with bin parameter and number of angles ntheta. 
If bin==0 then the size is maximum, bin=1 makes data binning 2x2.
Examples of processing data:

bin=1, ntheta=900 - dataset is (900,4,1024,1024)
bin=0, ntheta=4500 - dataset is (4500,4,2048,2048) - whole data

niter=3 - number of iterations, there is no need to do many iterations for performance tests. 
All iteratrions except the first should have similar execution time.

On Tesla A100 I typically set nchunk=32 for bin=2, nchunk=16 for bin=1, nchunk=2 for bin=0





