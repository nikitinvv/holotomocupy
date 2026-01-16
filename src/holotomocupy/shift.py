import numpy as np
import cupy as cp

from .cuda_kernels import *
from .utils import *

class Shift():    
    """Functionality for Shifts"""

    def __init__(self,n,npsi,nz,nzpsi,mag):
        self.n = n    
        self.npsi = npsi
        self.nz = nz    
        self.nzpsi = nzpsi
        self.mag = mag        
        self.fB3 = cp.zeros([len(mag),nzpsi,npsi],dtype='float32')

        x=cp.linspace(-1/2,1/2-1/npsi,npsi).astype('float32')
        y=cp.linspace(-1/2,1/2-1/nzpsi,nzpsi).astype('float32')
        for imagn in range(len(mag)):
            divx = self.phi(0).astype('float32')
            divy = self.phi(0).astype('float32')
            for k  in range(1,5):#max
                divx = divx+(2*self.phi(k/self.mag[imagn])*cp.cos(2*cp.pi*k*x)).astype('float32')        
                divy = divy+(2*self.phi(k/self.mag[imagn])*cp.cos(2*cp.pi*k*y)).astype('float32')        
                self.fB3[imagn] = cp.fft.fftshift(cp.outer(divy,divx),axes=(-1,-2))

    def phi(self,t):
        out=(-2<t)*(t<=-1)*(t+2)**3+(-1<t)*(t<=1)*(4-6*t**2+3*t**3*cp.sign(t))+(1<t)*(t<=2)*(2-t)**3
        return out  

    def dphi(self,t):
        out = (-2<t)*(t<=-1)*3*(t+2)**2+(-1<t)*(t<=1)*(-12*t+9*t**2*cp.sign(t))-(1<t)*(t<=2)*3*(2-t)**2
        return out  

    def d2phi(self,t):
        out=(-2<t)*(t<=-1)*6*(t+2)**1+(-1<t)*(t<=1)*(-12+18*t*cp.sign(t))+(1<t)*(t<=2)*6*(2-t)
        return out  

######## Mathematically curlyS depends on r and psi, S depends on psi alone with r fixed and T depends on r alone with psi fixed
    def coeff(self,psi,imagn):
        out=cp.fft.ifft2(cp.fft.fft2(psi)/self.fB3[imagn])
        return out    

    def S(self, c, r, imagn):
        ntheta = c.shape[0]
        spsi = cp.zeros([ntheta,self.nz,self.n], dtype="complex64")
        c = cp.ascontiguousarray(c)
        r = cp.ascontiguousarray(r)
        maga = cp.array(self.mag[imagn:imagn+1].astype('float32'))
        s_kernel(
                    (
                        int(cp.ceil(self.n / 32)),
                        int(cp.ceil(self.nz / 32)),
                        ntheta,
                    ),
                    (32, 32, 1),
                    (spsi, c, r, maga , self.n, self.npsi, self.nz, self.nzpsi, ntheta,0),
                )
        return spsi
    
    def Sadj(self, spsi, r, imagn):
        ntheta = spsi.shape[0]
        c = cp.zeros([ntheta,self.nzpsi,self.npsi], dtype="complex64")
        spsi = cp.ascontiguousarray(spsi)
        r = cp.ascontiguousarray(r)
        maga = cp.array(self.mag[imagn:imagn+1].astype('float32'))
        
        s_kernel(
                    (
                        int(cp.ceil(self.n / 32)),
                        int(cp.ceil(self.nz / 32)),
                        ntheta,
                    ),
                    (32, 32, 1),
                    (spsi, c, r, maga , self.n, self.npsi,self.nz, self.nzpsi, ntheta,1),
                )
        return c
    

    def dS(self, c, r, imagn, Deltac):
        return self.S(Deltac, r, imagn)    

    def dSadj(self, c, r, imagn, Deltac):
        """Shift adjoint"""
        return self.Sadj(Deltac, r, imagn)
   
    ######## no need to compute d2S since it is =0
    def dT(self, c, r, imagn, Deltar):
        """dT following formula (32) in the ptychography paper"""  
        ntheta = c.shape[0]        
        res = cp.zeros([ntheta,self.nz,self.n], dtype="complex64")
        c = cp.ascontiguousarray(c)
        r = cp.ascontiguousarray(r)
        Deltar = cp.ascontiguousarray(Deltar)
        maga = cp.array(self.mag[imagn:imagn+1].astype('float32'))
        dt_kernel(
                    (
                        int(cp.ceil(self.n / 32)),
                        int(cp.ceil(self.nz / 32)),
                        ntheta,
                    ),
                    (32, 32, 1),
                    (res, c, r, maga, Deltar, self.n,self.npsi,self.nz,self.nzpsi, ntheta),
                )
        return res   

    def dTadj(self, c, r, imagn, Deltaphi):
        ntheta = c.shape[0]     
        out = cp.zeros(r.shape, dtype="float32")
        dt1 = cp.zeros(Deltaphi.shape, dtype="complex64")
        dt2 = cp.zeros(Deltaphi.shape, dtype="complex64")      
        c = cp.ascontiguousarray(c)
        r = cp.ascontiguousarray(r)
        Deltaphi = cp.ascontiguousarray(Deltaphi)
        maga = cp.array(self.mag[imagn:imagn+1].astype('float32'))
        dtadj_kernel(
                    (
                        int(cp.ceil(self.n / 32)),
                        int(cp.ceil(self.nz / 32)),
                        ntheta,
                    ),
                    (32, 32, 1),
                    (dt1, dt2, c, r, maga, self.n, self.npsi,self.nz, self.nzpsi, ntheta),
                )
        
        out[:, 0] = redot(Deltaphi, dt1, axis=(1, 2))
        out[:, 1] = redot(Deltaphi, dt2, axis=(1, 2))
        return out   

    def d2T(self, c, r, imagn, Deltar1,Deltar2):
        """d2T following formula (33) in the ptychography paper"""
        ntheta = c.shape[0]        
        res = cp.zeros([ntheta,self.nz,self.n], dtype="complex64")
        c = cp.ascontiguousarray(c)
        r = cp.ascontiguousarray(r)
        Deltar1 = cp.ascontiguousarray(Deltar1)
        Deltar2 = cp.ascontiguousarray(Deltar2)
        maga = cp.array(self.mag[imagn:imagn+1].astype('float32'))
        d2t_kernel(
                    (
                        int(cp.ceil(self.n / 32)),
                        int(cp.ceil(self.nz / 32)),
                        ntheta,
                    ),
                    (32, 32, 1),
                    (res, c, r, maga, Deltar1, Deltar2, self.n, self.npsi,self.nz, self.nzpsi, ntheta),
                )
    
        return res   


    def curlyS(self, psi, r, imagn):
        out=self.S(self.coeff(psi,imagn),r,imagn)
        return out
    
    def curlySadj(self, psi, r, imagn):
        out = self.coeff(self.Sadj(psi, r, imagn),imagn)
        return out
        
    def dcurlyS(self, psi, r, imagn, Deltapsi, Deltar):
        c=self.coeff(psi,imagn) 
        c1=self.coeff(Deltapsi,imagn)
        out = self.dS(c,r,imagn,c1)+self.dT(c,r,imagn,Deltar)
        return out
   
    def dcurlySadj(self, psi, r, imagn, Deltaphi):
        c = self.coeff(psi,imagn)
        out1 = self.coeff(self.dSadj(c, r, imagn, Deltaphi),imagn)
        
        out2 = self.dTadj(c, r, imagn, Deltaphi)
        out = [out1, out2]
        return out
   
    def d2curlyS(self, psi, r, imagn, Deltapsi1, Deltar1, Deltapsi2, Deltar2):
        """dcurlyS following formula below (33) in the ptychography paper"""
        c=self.coeff(psi,imagn)
        c1=self.coeff(Deltapsi1,imagn)
        c2=self.coeff(Deltapsi2,imagn)
        out = self.dT(c1, r, imagn, Deltar2)+self.dT(c2, r, imagn, Deltar1)+self.d2T(c, r, imagn, Deltar1, Deltar2)
        return out
        