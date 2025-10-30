import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import os

from cuda_kernels import *
from utils import *
from tomo import Tomo
from propagation import Propagation
from curlySspline import Shift as Shift2

np.set_printoptions(legacy="1.25")


class Rec:
    def __init__(self, args):

        # copy args to elements of the class
        for key, value in vars(args).items():
            setattr(self, key, value)

        # list of functionals, gradients, differentials, and second-order differentials
        self.F = [self.F0, self.F1, self.F2, self.F3]
        self.gF = [self.gF0, self.gF1, self.gF2, self.gF3]
        self.dF = [self.dF0, self.dF1, self.dF2, self.dF3]
        self.d2F = [self.d2F0, self.d2F1, self.d2F2, self.d2F3]
        self.noper = len(self.F)

        # tomo class
        cl_tomo = Tomo(args.npsi, args.theta, args.rotation_axis)
        self.R = cl_tomo.R
        self.RT = cl_tomo.RT
        self.expR = cl_tomo.expR

        # propagator class
        cl_prop = Propagation(args.n, args.ndist, args.wavelength, args.voxelsize, args.distance)
        self.D = cl_prop.D
        self.DT = cl_prop.DT
        # self.Dp = cl_prop.Dp
        # self.DpT = cl_prop.DpT

        cl_shift2 = Shift2(self.n,self.npsi)
        self.curlyS = cl_shift2.curlyS
        self.dcurlyS = cl_shift2.dcurlyS
        self.dcurlySadj = cl_shift2.dcurlySadj
        self.d2curlyS = cl_shift2.d2curlyS
        self.S = cl_shift2.S
        self.Sadj = cl_shift2.Sadj
       

        # # magnification class
        # cl_mag = Magnification(args.n, args.npsi, args.norm_magnifications)
        # self.M = cl_mag.M
        # self.MT = cl_mag.MT

    def BH(self, d, ref, vars):
        # keep data in class
        self.data = d
        self.ref = ref

        for i in range(self.niter):
            # debug plots
            self.error_debug(vars, i)
            self.vis_debug(vars, i)

            # gradients
            grads = self.gradients_new(vars)
            # Ru is stored to avoid recomputing
            
            grads["u"] *= self.rho[0] ** 2
            grads["q"] *= self.rho[1] ** 2
            grads["r"] *= self.rho[2] ** 2

            grads["Ru"] = self.R(grads["u"])
            # etas = {}
            # etas["u"] = -grads["u"]
            # etas["q"] = -grads["q"]
            # etas["r"] = -grads["r"]
            if i == 0:
                etas = {}
                etas["u"] = -grads["u"]
                etas["q"] = -grads["q"]
                etas["r"] = -grads["r"]
            else:
                # calc beta
                top = self.hessian_new(vars, grads, etas)
                bottom = self.hessian_new(vars, etas, etas)
                beta = top / bottom
                
                etas["u"] = etas["u"] * beta - grads["u"]
                etas["q"] = etas["q"] * beta - grads["q"]
                etas["r"] = etas["r"] * beta - grads["r"]
            etas["Ru"] = self.R(etas["u"])

            # calc alpha
            top = -(redot(grads["u"], etas["u"]) / self.rho[0] ** 2 + 
                    redot(grads["q"], etas["q"]) / self.rho[1] ** 2 + 
                    redot(grads["r"], etas["r"]) / self.rho[2] ** 2)
            bottom = self.hessian_new(vars, etas, etas)
            alpha = top / bottom
            #print(f"{alpha:.4e}")
            # debug approxmation
            self.plot_debug(vars, etas, top, bottom, alpha, i)
            vars["u"] += alpha * etas["u"]
            vars["q"] += alpha * etas["q"]
            vars["r"] += alpha * etas["r"]

            vars["psi"] = self.expR(self.R(vars["u"]))
        return vars

    # ###################### Cascade computation of functionals  ######################
    def hessian_new(self, vars, grads, etas):
        """Cascade hessian"""
        x = [vars["q"], vars["u"], vars["r"], vars["psi"]]  # psi is precalculated previously
        y = [grads["q"], grads["u"], grads["r"], grads["Ru"]]  # Ru is precalculated previously
        z = [etas["q"], etas["u"], etas["r"], etas["Ru"]]  # Ru is precalculated previously
        w = [cp.zeros_like(etas["q"]), cp.zeros_like(etas["u"]), cp.zeros_like(etas["r"]), cp.zeros_like(etas["Ru"])]

        # compute hessian by iterating from level 3 to level 1
        for id in range(self.noper)[::-1]:
            # term1
            d2f1 = self.d2F[id](x, y, z)
            # term2
            d2f2 = self.dF[id](x, w, return_x=False)
            # return sum for the last iter
            if id == 0:
                w = d2f1 + d2f2
                break
            # sum variables
            d2f = [v1 + v2 for v1, v2 in zip(d2f1, d2f2)]

            # recalc differentials for the next kle
            fx, dfy = self.dF[id](x, y)  # returns a pair [fx,dfx(y)]
            if 0:#y[0] is z[0]:  # if y and z are the same compute only one
                dfz = dfy
            else:
                dfz = self.dF[id](x, z, return_x=False)  # returns dfx(z)
            # assign for the next level
            x, y, z, w = fx, dfy, dfz, d2f

        w+=self.hessianq(vars['q'],grads['q'],etas['q'])   
        return w

    def gradients_new(self, vars):
        """Cascade gradients"""
        x = [vars["q"], vars["u"], vars["r"], vars["psi"]]  # assume psi is precalculated
        y = x  # forming output
        
        # compute functional by applying operators in reverse order
        for id in range(1, self.noper)[::-1]:
            y = self.F[id](y)
        
        for id in range(self.noper):
            y = self.gF[id](x, y)            
        # place in dictionary
        grads = {}
        grads["q"], grads["u"], grads["r"] = y

        grads["q"]+=self.gradientq(vars['q'])           
        return grads

    ####### F0(x0) = \||x1|-d\|_2^2
    def F0(self, x1):
        return cp.linalg.norm(cp.abs(x1) - self.data) ** 2
    
    def dF0(self, x, y, return_x=False):

        out = cp.empty_like(x)
        for j in range(self.ndist):
            out[:, j] = 2 * (x[:, j] - self.data[:, j] * (x[:, j] / (cp.abs(x[:, j]))))
        out = cp.array(out)
        y = cp.array(y)
        return redot(out, y)

    def gF0(self, x, y):
        out = cp.empty_like(y)
        for j in range(self.ndist):
            td = self.data[:, j] * (y[:, j] / (cp.abs(y[:, j])))
            out[:, j] = 2 * (y[:, j] - td)

        return out

    def d2F0(self, x, y, z):
        out = 0
        for j in range(self.ndist):
            l0 = x[:, j] / (cp.abs(x[:, j]))
            d0 = self.data[:, j] / (cp.abs(x[:, j]))
            v1 = cp.sum((1 - d0) * reprod(y[:, j], z[:, j]))
            v2 = cp.sum(d0 * reprod(l0, y[:, j]) * reprod(l0, z[:, j]))
            out += 2 * (v1 + v2)
        out = np.float32(out.get())
        
        return out

    ####### F1(x21,x22) = D(Dp(x21)\cdot x22)
    def F1(self, x):
        x21, x22 = x
        out = cp.empty([len(x22), self.ndist, self.n, self.n], dtype="complex64")

        for j in range(self.ndist):
            out[:, j] = self.D(x21[j] * x22[:, j], j)

        return out

    def dF1(self, x, y, return_x=True):
        x21, x22 = x
        y21, y22 = y

        if return_x:
            x1 = cp.empty([self.ntheta, self.ndist, self.n, self.n], dtype="complex64")
        y1 = cp.empty([self.ntheta, self.ndist, self.n, self.n], dtype="complex64")
        for j in range(self.ndist):
            if return_x:
                x1[:, j] = self.D(x21[j] * x22[:, j], j)
            y1[:, j] = self.D(y21[j] * x22[:, j], j)
            y1[:, j] += self.D(x21[j] * y22[:, j], j)
        if return_x:
            return x1, y1
        else:
            return y1

    def gF1(self, x, y):
        y11 = y
        y21 = cp.zeros([self.ndist, self.n, self.n], dtype="complex64")
        y22 = cp.empty([self.ntheta, self.ndist, self.n, self.n], dtype="complex64")

        q, u, r, psi = x

        for j in range(self.ndist):
            mspsi = self.curlyS(psi,r[:,j], 1/self.norm_magnifications[j])            
            y21[j] += cp.sum(self.DT(y11[:, j],j) * np.conj(mspsi), axis=0)
            y22[:, j] = self.DT(y11[:, j], j) * np.conj(q[j])

        return [y21, y22]

    def d2F1(self, x, y, z):
        x21, x22 = x
        y21, y22 = y
        z21, z22 = z

        y1 = cp.empty([self.ntheta, self.ndist, self.n, self.n], dtype="complex64")
        for j in range(self.ndist):
            if 0:#y22 is z22:
                y1[:, j] = 2 * self.D(y21[j] * y22[:, j], j)
            else:
                y1[:, j] = self.D(y21[j] * z22[:, j], j) + self.D(z21[j] * y22[:, j], j)
        return y1

    def F2(self, x):
        x31, x32, x33 = x

        x22 = cp.empty([len(x33), self.ndist, self.n, self.n], dtype="complex64")
        for k in range(self.ndist):
            x22[:,k] = self.curlyS(x32,x33[:,k],1/self.norm_magnifications[k])

        return [x31, x22]
    
    def dF2(self, x, y, return_x=True):
        x31, x32, x33 = x
        y31, y32, y33 = y

        if return_x:
            x22 = cp.zeros([self.ntheta, self.ndist, self.n, self.n], dtype="complex64")
        y22 = cp.zeros([self.ntheta, self.ndist, self.n, self.n], dtype="complex64")
        
        for k in range(self.ndist):
            r = x33[:, k]
            if return_x:
                x22[:, k] = self.curlyS(x32, r,1/self.norm_magnifications[k])            
            Deltar = y33[:,k]
            y22[:,k] = self.dcurlyS(x32, r, 1/self.norm_magnifications[k], y32, Deltar)

        if return_x:
            return [x31, x22], [y31, y22]
        else:
            return [y31, y22]
    
    def d2F2(self, x, y, z):
        x31, x32, x33 = x
        y31, y32, y33 = y
        z31, z32, z33 = z

        y22 = cp.zeros([self.ntheta, self.ndist, self.n, self.n], dtype="complex64")
        
        for k in range(self.ndist):
            r = x33[:, k]
            Deltar_y = y33[:,k]
            Deltar_z = z33[:,k]
            y22[:,k] = self.d2curlyS(x32, r,1/self.norm_magnifications[k], y32,  Deltar_y, z32, Deltar_z)

        return [cp.zeros_like(y31), y22]
    
    def gF2(self, x, y):
        "Warning the above x is the root x (x3) and not the x for this level (i.e. x2)"
        y21, y22 = y

        y33 = cp.empty([self.ntheta, self.ndist, 2], dtype="float32")
        y32 = cp.zeros([self.ntheta, self.npsi, self.npsi], dtype="complex64")

        x2 = self.F3(x)
        x21, x22,x23= x2
        psi = x22
        for k in range(self.ndist):                        
            Deltaphi = y22[:,k] 
            r = x23[:, k]
            Deltapsi, Deltar = self.dcurlySadj(psi, r, 1/self.norm_magnifications[k], Deltaphi)
            y32 += Deltapsi
            y33[:, k] = Deltar
        return [y21, y32, y33]
        
        
        
    ######## F3(x41,x42,x43)=(x41,e^{1j R(x42)},x43)
    ####### new F3(x41,x42,x43)=(D_-zeta(dr*e^{1j x41}),e^{1j R(x42)},x43), 
    # x41 being the (real-valued) phase of the probe on the detector
    def F3(self, x):

        x41, x42, x43, x44 = x
        # out = self.expR(self.R(x42))
        out = x44
        
        
        ### old version
        return x41, out, x43

        #### new version
        out1=cp.zeros([self.ndist,self.n,self.n],dtype="complex64")
        for k in range(self.ndist):            
            out1[k] = self.DT(self.ref[k]*cp.exp(1j*x41[k]),k)

        return out1, out, x43

    def dF3(self, x, y, return_x=True):
        x41, x42, x43, x44 = x
        y41, y42, y43, y44 = y

        if return_x:
            x32 = x44  # already computed

        y32 = x44 * 1j * y44



        ## oldf version
        if return_x:
            return [x41, x32, x43], [y41, y32, y43]
        else:
            return [y41, y32, y43]        

        #### new version
        y31=cp.zeros([self.ndist,self.n,self.n],dtype="complex64")
        for k in range(self.ndist):
            y31[k] = 1j*self.DT(self.ref[k]*cp.exp(1j*x41[k])*y41[k],k)
        ###
        if return_x:
            x31=cp.zeros([self.ndist,self.n,self.n],dtype="complex64")
            for k in range(self.ndist):            
                x31[k] = self.DT(self.ref[k]*cp.exp(1j*x41[k]),k)

        if return_x:
            return [x31, x32, x43], [y31, y32, y43]
        else:
            return [y31, y32, y43]

    def gF3(self, x, y):
        y31, y32, y33 = y
        x31, x32, x33, x34 = x
        y42 = (-1j) * self.RT(y32 * cp.conj(x34))
        
        y42=y42.real+1j*0
        ### old version
        return [y31, y42, y33]

        ### new version
        y41 = cp.zeros([self.ndist,self.n,self.n],dtype="float32")
        for k in range(self.ndist):
            y41[k] = cp.real(-1j*self.ref[k]*cp.exp(-1j*x31[k])*self.D(y31[k],k)[0])
        return [y41, y42, y33]

    def d2F3(self, x, y, z):
        x41, x42, x43, x44 = x
        y41, y42, y43, y44 = y
        z41, z42, z43, z44 = z
        if 0:#y44 is z44:
            y32 = x44 * (-(y44**2))
        else:
            y32 = x44 * (-y44 * z44)

        #### new version
        return [cp.zeros_like(y41), y32, cp.zeros_like(y43)]

        #### new version
        y31=cp.zeros([self.ndist,self.n,self.n],dtype="complex64")
        if 0:#y44 is z44:
            y32 = x44 * (-(y44**2))
            for k in range(self.ndist):
                y31[k] = -self.DT(self.ref[k]*cp.exp(1j*x41[k:k+1])*y41[k:k+1]**2,k)[0]
        else:
            y32 = x44 * (-y44 * z44)
            for k in range(self.ndist):
                y31[k] = -self.DT(self.ref[k]*cp.exp(1j*x41[k:k+1])*y41[k:k+1]*z41[k:k+1],k)

        #return [cp.zeros_like(y41), y32, cp.zeros_like(y43)]
        return [y31, y32, cp.zeros_like(y43)]

    ############################ Debug functions #########################
    def minF(self, big_psi):
        res = cp.linalg.norm(cp.abs(big_psi) - self.data) ** 2
        return res

    def fwd(self, r, u, q):
        psi = self.expR(self.R(u))
        x = [q, u, r, psi]
        y = x  # forming output
        # compute functional by applying operators in reverse order
        for id in range(1, len(self.F))[::-1]:
            y = self.F[id](y)

        return y

    def plot_debug(self, vars, etas, top, bottom, alpha, i):
        """Check the minimization functional behaviour"""
        if i % self.vis_step == 0 and self.vis_step != -1 and self.show:
            fig = plt.figure(figsize=(4,4))
            (q, u, r) = (vars["q"], vars["u"], vars["r"])
            (dq2, du2, dr2) = (etas["q"], etas["u"], etas["r"])
            npp = 9
            errt = cp.zeros(npp)
            errt2 = cp.zeros(npp)
            for k in range(0, npp):
                ut = u + (alpha * k / (npp // 2)) * du2
                qt = q + (alpha * k / (npp // 2)) * dq2
                rt = r + (alpha * k / (npp // 2)) * dr2
                tmp = self.fwd(rt, ut, qt)
                tmpq = self.fwdq(qt) 
                errt[k] = self.minF(tmp)+self.minFq(tmpq)

            t = alpha * (cp.arange(npp)) / (npp // 2)
            tmp = self.fwd(r, u, q)
            tmpq = self.fwdq(q)
            
            errt2 = self.minF(tmp)+self.minFq(tmpq)
            errt2 = errt2 - top * t + 0.5 * bottom * t**2
            plt.plot(
                t.get(),
                errt.get(),
                ".",
                label="real",
            )
            plt.plot(
                t.get(),
                errt2.get(),
                ".",
                label="appr",
            )
            plt.legend()
            plt.grid()
            plt.show()

    def vis_debug(self, vars, i):
        """Visualization and data saving"""
        if i % self.vis_step == 0 and self.vis_step != -1:
            (q, u, r) = (vars["q"], vars["u"], vars["r"])

            mshow_complex(u[u.shape[0] // 2].real + 1j * u[:, u.shape[1] // 2].real, self.show)
            # mshow_polar(q[0], self.show)
            # write_tiff(u.real, f"{self.path_out}/rec_u_real/{i:04}")
            # write_tiff(u.imag, f"{self.path_out}/rec_u_imag/{i:04}")
            # write_tiff(u[self.npsi // 2].real, f"{self.path_out}/rec_uz/{i:04}")
            # write_tiff(u[:, self.npsi // 2].real, f"{self.path_out}/rec_uy/{i:04}")
            # write_tiff(np.angle(q), f"{self.path_out}/rec_prb_angle/{i:04}")
            # write_tiff(np.abs(q), f"{self.path_out}/rec_prb_abs/{i:04}")
            # np.save(f"{self.path_out}/r{i:04}", r)

            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            if isinstance(vars["r"], cp.ndarray):
                ax[0].plot(vars["r"][:, :, 1].get() - vars["r_init"][:, :, 1].get(), ".")
                ax[1].plot(vars["r"][:, :, 0].get() - vars["r_init"][:, :, 0].get(), ".")
            else:
                ax[0].plot(vars["r"][:, :, 1].get() - vars["r_init"][:, :, 1].get(), ".")
                ax[1].plot(vars["r"][:, :, 0].get() - vars["r_init"][:, :, 0].get(), ".")

            #print(np.where(np.abs(vars["r"][:, :, 0].get() - vars["r_init"][:, :, 0].get())>0.08))
            # plt.savefig(f"{self.path_out}/rerr{i:04}.png")


            if self.show:
                plt.show()

    def error_debug(self, vars, i):
        """Visualization and data saving"""
        if i % self.err_step == 0 and self.err_step != -1:
            (q, u, r) = (vars["q"], vars["u"], vars["r"])
            big_psi = self.fwd(r, u, q)
            Dq = self.fwdq(q)            
            err = self.minF(big_psi)+self.minFq(Dq)
            print(f"{i}) {err=:1.5e}", flush=True)
            vars["table"].loc[len(vars["table"])] = [i, err.get(), time.time()]
            name = f"{self.path_out}/conv.csv"
            os.makedirs(os.path.dirname(name), exist_ok=True)
            vars["table"].to_csv(name, index=False)

    def gradientq(self,q):
        gradq = cp.zeros_like(q)
        for j in range(self.ndist):
            tmp = self.D(q[j:j+1],j)        
            td = self.ref[j:j+1] * (tmp / (cp.abs(tmp)))
            gradq[j:j+1] += self.DT(2 * (tmp - td),j)
        return self.lamq*gradq

    def hessianq(self, q, dq1,dq2):  
        res = 0
        for j in range(self.ndist): 
            Dq = self.D(q[j:j+1],j)  
            Ddq1 = self.D(dq1[j:j+1],j)  
            Ddq2 = self.D(dq2[j:j+1],j)  
            l0 = Dq / (cp.abs(Dq))
            d0 = self.ref[j:j+1] / (cp.abs(Dq))
            v1 = cp.sum((1 - d0) * reprod(Ddq1, Ddq2))
            v2 = cp.sum(d0 * reprod(l0, Ddq1) * reprod(l0, Ddq2))
            res += 2 * (v1 + v2)
        return self.lamq*res
    
    def minFq(self, Dq):
        res = cp.linalg.norm(cp.abs(Dq) - self.ref) ** 2
        return self.lamq*res
    
    def fwdq(self, q):
        Dq = cp.empty_like(q)
        for j in range(self.ndist): 
            Dq[j:j+1] = self.D(q[j:j+1],j)  

        return Dq