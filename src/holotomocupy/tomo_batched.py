import numpy as np
import cupy as cp
import warnings
import cupyx.scipy.ndimage as ndimage

from .chunking import Chunking
from .cuda_kernels import *
from .utils import *
from .tomo import Tomo

np.set_printoptions(legacy="1.25")
warnings.filterwarnings("ignore", message=f".*peer.*")


class TomoBatched:
    def __init__(self, args):
        # copy args to elements of the class
        for key, value in vars(args).items():
            setattr(self, key, value)

        # estimate memory footprint for pinned + device buffer per GPU (complex64)
        multiplier = 16  # related to the number of arrays, experimentally chosen. the scheme will diverge if too low
        complex_item = np.dtype("complex64").itemsize
        max_dim = max(self.nzobj, self.ntheta)
        nbytes = int(multiplier * self.nchunk * self.nobj * max_dim * complex_item)

        self.cl_chunking = Chunking(nbytes, self.nchunk, self.ngpus)
        self.cl_tomo = [None for _ in range(self.ngpus)]
        
        for igpu in range(self.ngpus):
            with cp.cuda.Device(igpu):
                # initialize processing classes per gpu
                self.cl_tomo[igpu] = Tomo(self.nobj, self.theta, self.mask_r)
        
        # normalization constant to address work with normal operators
        self.norm_const = np.float32(np.sqrt(self.nobj / self.ntheta))

        self.gpu_batch = self.cl_chunking.gpu_batch
        self.redot_batch = self.cl_chunking.redot_batch
        self.linear_batch = self.cl_chunking.linear_batch
        self.mulc_batch = self.cl_chunking.mulc_batch
        
    def fwd_tomo(self, obj, exp=False, out=None):
        if out is None:
            out = np.empty([self.ntheta, obj.shape[0], self.nobj], dtype="complex64")

        @self.gpu_batch(axis_out=1, axis_inp=0,nout=1)
        def _fwd_tomo(self, out, obj):
            igpu = cp.cuda.Device().id
            out[:] = self.cl_tomo[igpu].R(obj)
            if exp:
                out[:] = cp.exp(1j * out)

        _fwd_tomo(self, out, obj)
        return out

    def adj_tomo(self, proj,out=None):
        if out is None:
            out = np.empty([proj.shape[1], self.nobj, self.nobj], dtype="complex64")

        @self.gpu_batch(axis_out=0, axis_inp=1,nout=1)
        def _adj_tomo(self, out, proj):
            igpu = cp.cuda.Device().id
            out[:] = self.cl_tomo[igpu].RT(proj)

        _adj_tomo(self, out, proj)
        return out

    def min(self, Robj, proj):
        """Minimization functional"""

        ## batched computation
        res = []
        for igpu in range(self.ngpus):
            with cp.cuda.Device(igpu):
                res.append(cp.zeros(1, dtype="float32"))

        ### main term
        @self.gpu_batch(axis_out=0, axis_inp=0,nout=1)
        def _min(self, res, Robj, proj):
            res[:] += cp.linalg.norm(Robj-proj)**2
        _min(self, res, Robj, proj)

        # collect results
        for k in range(1, len(res)):
            res[0] += res[k]
        res = res[0][0].get()
        return res
       
    def rec_tomo(self, proj, niter=1):
        """Regular tomography reconstrution for initial guess"""
        
        obj = np.zeros([proj.shape[1], self.nobj, self.nobj], dtype="complex64")
        
        grad = np.empty_like(obj)  
        eta = np.empty_like(obj)  
        
        Rgrad = np.empty_like(proj)
        Reta = np.empty_like(proj)        
        tmp = np.empty_like(proj)        
        
        Robj = self.fwd_tomo(obj)
        for k in range(niter):
            self.linear_batch(Robj,proj,2,-2,out=tmp)
            self.adj_tomo(tmp,out=grad)
            self.fwd_tomo(grad,out=Rgrad)
            if k == 0:
                self.mulc_batch(eta, grad, -1)
                self.mulc_batch(Reta, Rgrad, -1)
            else:
                beta = self.redot_batch(Rgrad, Reta) / self.redot_batch(Reta, Reta)
                self.linear_batch(eta, grad, beta, -1)
                self.linear_batch(Reta, Rgrad, beta, -1)
            alpha = -self.redot_batch(grad, eta) / (2 * self.redot_batch(Reta, Reta))
            self.linear_batch(obj, eta, 1, alpha)
            self.linear_batch(Robj, Reta, 1, alpha)
            if k % 4 == 0:
                print(f"{k} err={self.min(Robj,proj)}")
        # normalize back
        obj *= self.norm_const
        return obj

    