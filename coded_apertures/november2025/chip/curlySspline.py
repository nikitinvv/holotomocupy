import numpy as np
import cupy as cp
from cuda_kernels import *
from utils import *

class Shift():    
    #########################  Functionality for Shifts #########################    
    def __init__(self,n,npsi):
        self.n = n    
        self.npsi = npsi

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
    def coeff(self,psi,m):
        n=psi.shape[-1]
        x=cp.linspace(-1/2,1/2-1/n,n).astype('float32')
        
        div = self.phi(0).astype('float32')
        for k  in range(1,5):#max
            div = div+(2*self.phi(k/m)*cp.cos(2*cp.pi*k*x)).astype('float32')
        
        divx=div[:,None]
        divy=div[None,:]
        fB3 = divx*divy
        fpsi = cp.fft.fft2(cp.fft.fftshift(psi,axes=(-1,-2)))
        out=cp.fft.fftshift(cp.fft.ifft2(fpsi/cp.fft.fftshift(fB3,axes=(-1,-2))),axes=(-1,-2))
        return out    

    def S(self, c, r, mag):
        ntheta = c.shape[0]
        spsi = cp.zeros([ntheta,self.n,self.n], dtype="complex64")
        c = cp.ascontiguousarray(c)
        r = cp.ascontiguousarray(r)
        maga = cp.array([1],dtype='float32')
        maga[0] = mag
        s_kernel(
                    (
                        int(cp.ceil(self.n / 32)),
                        int(cp.ceil(self.n / 32)),
                        ntheta,
                    ),
                    (32, 32, 1),
                    (spsi, c, r, maga , self.n, self.npsi, ntheta,0),
                )
        return spsi
    
    def Sadj(self, spsi, r, mag):
        ntheta = spsi.shape[0]
        c = cp.zeros([ntheta,self.npsi,self.npsi], dtype="complex64")
        spsi = cp.ascontiguousarray(spsi)
        r = cp.ascontiguousarray(r)
        maga = cp.array([1],dtype='float32')
        maga[0] = mag
        s_kernel(
                    (
                        int(cp.ceil(self.n / 32)),
                        int(cp.ceil(self.n / 32)),
                        ntheta,
                    ),
                    (32, 32, 1),
                    (spsi, c, r, maga , self.n, self.npsi, ntheta,1),
                )
        return c
    

    def dS(self, c, r, mag, Deltac):
        return self.S(Deltac, r, mag)    

    def dSadj(self, c, r, mag, Deltac):
        """Shift adjoint"""
        return self.Sadj(Deltac, r, mag)
   
    ######## no need to compute d2S since it is =0
    def dT(self, c, r, mag, Deltar):
        """dT following formula (32) in the ptychography paper"""  
        ntheta = c.shape[0]        
        res = cp.zeros([ntheta,self.n,self.n], dtype="complex64")
        c = cp.ascontiguousarray(c)
        r = cp.ascontiguousarray(r)
        Deltar = cp.ascontiguousarray(Deltar)
        maga = cp.array([1],dtype='float32')
        maga[0] = mag
        dt_kernel(
                    (
                        int(cp.ceil(self.n / 32)),
                        int(cp.ceil(self.n / 32)),
                        ntheta,
                    ),
                    (32, 32, 1),
                    (res, c, r, maga, Deltar, self.n,self.npsi, ntheta),
                )
        return res   

    def dTadj(self, c, r, mag, Deltaphi):
        ntheta = c.shape[0]     
        out = cp.zeros(r.shape, dtype="float32")
        dt1 = cp.zeros(Deltaphi.shape, dtype="complex64")
        dt2 = cp.zeros(Deltaphi.shape, dtype="complex64")      
        c = cp.ascontiguousarray(c)
        r = cp.ascontiguousarray(r)
        Deltaphi = cp.ascontiguousarray(Deltaphi)
        maga = cp.array([1],dtype='float32')
        maga[0] = mag
        dtadj_kernel(
                    (
                        int(cp.ceil(self.n / 32)),
                        int(cp.ceil(self.n / 32)),
                        ntheta,
                    ),
                    (32, 32, 1),
                    (dt1, dt2, c, r, maga, self.n, self.npsi, ntheta),
                )
        
        out[:, 0] = redot(Deltaphi, dt1, axis=(1, 2))
        out[:, 1] = redot(Deltaphi, dt2, axis=(1, 2))
        return out   

    def d2T(self, c, r, mag, Deltar1,Deltar2):
        """d2T following formula (33) in the ptychography paper"""
        ntheta = c.shape[0]        
        res = cp.zeros([ntheta,self.n,self.n], dtype="complex64")
        c = cp.ascontiguousarray(c)
        r = cp.ascontiguousarray(r)
        Deltar1 = cp.ascontiguousarray(Deltar1)
        Deltar2 = cp.ascontiguousarray(Deltar2)
        maga = cp.array([1],dtype='float32')
        maga[0] = mag
        d2t_kernel(
                    (
                        int(cp.ceil(self.n / 32)),
                        int(cp.ceil(self.n / 32)),
                        ntheta,
                    ),
                    (32, 32, 1),
                    (res, c, r, maga, Deltar1, Deltar2, self.n, self.npsi, ntheta),
                )
    
        return res   


    def curlyS(self, psi, r, mag):
        out=self.S(self.coeff(psi,mag),r,mag)
        return out
        
    def dcurlyS(self, psi, r, mag, Deltapsi, Deltar):
        c=self.coeff(psi,mag) 
        c1=self.coeff(Deltapsi,mag)
        out = self.dS(c,r,mag,c1)+self.dT(c,r,mag,Deltar)
        return out
   
    def dcurlySadj(self, psi, r, mag, Deltaphi):
        c = self.coeff(psi,mag)
        out1 = self.coeff(self.dSadj(c, r, mag, Deltaphi),mag)
        
        out2 = self.dTadj(c, r, mag, Deltaphi)
        out = [out1, out2]
        return out
   
    def d2curlyS(self, psi, r, mag, Deltapsi1, Deltar1, Deltapsi2, Deltar2):
        """dcurlyS following formula below (33) in the ptychography paper"""
        c=self.coeff(psi,mag)
        c1=self.coeff(Deltapsi1,mag)
        c2=self.coeff(Deltapsi2,mag)
        out = self.dT(c1, r, mag, Deltar2)+self.dT(c2, r, mag, Deltar1)+self.d2T(c, r, mag, Deltar1, Deltar2)
        return out