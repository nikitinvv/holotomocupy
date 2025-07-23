import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import os

from cuda_kernels import *
from utils import *
from tomo import Tomo
from propagation import Propagation
from shift import Shift
from magnification import Magnification
np.set_printoptions(legacy='1.25')

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
        cl_prop = Propagation(args.n, args.ndist, args.wavelength, args.voxelsize, args.distance, args.distancep)
        self.D = cl_prop.D
        self.DT = cl_prop.DT
        self.Dp = cl_prop.Dp
        self.DpT = cl_prop.DpT

        # shift class
        cl_shift = Shift()
        self.S = cl_shift.S
        self.ST = cl_shift.ST    

        # magnification class
        cl_mag = Magnification(args.n, args.npsi,args.norm_magnifications)
        self.M = cl_mag.M
        self.MT = cl_mag.MT        

    def BH(self, d, vars):
        # keep data in class
        self.data = d

        for i in range(self.niter):
            # debug plots
            self.error_debug(vars, i)
            self.vis_debug(vars, i)

            # gradients
            grads = self.gradients_new(vars)
            # Ru is stored to avoid recomputing
            grads["Ru"] = self.R(grads["u"])

            grads["u"] *= self.rho[0] ** 2
            grads["q"] *= self.rho[1] ** 2
            grads["r"] *= self.rho[2] ** 2
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
                # print(f"{beta=:.2e},{top=:.2e},{bottom=:.2e}")
                
                etas["u"] = etas["u"] * beta - grads["u"]
                etas["q"] = etas["q"] * beta - grads["q"]
                etas["r"] = etas["r"] * beta - grads["r"]            
            etas["Ru"] = self.R(etas["u"])
            
            # calc alpha                
            top = -(  redot(grads["u"], etas["u"])/self.rho[0] ** 2 
                    + redot(grads["q"], etas["q"])/self.rho[1] ** 2 
                    + redot(grads["r"], etas["r"])/self.rho[2] ** 2)
            bottom = self.hessian_new(vars, etas, etas)
            alpha = top / bottom
            # print(f"{alpha=:.2e},{top=:.2e},{bottom=:.2e}")
            # debug approxmation
            self.plot_debug(vars, etas, top, bottom, alpha, i)
            vars["u"] += alpha * etas["u"]
            vars["q"] += alpha * etas["q"]
            vars["r"] += alpha * etas["r"]

            self.R(vars["u"], out=vars["psi"])
            self.expR(vars["psi"], out=vars["psi"])
        return vars

    ###################### Cascade computation of functionals  ######################
    def hessian_new(self, vars, grads, etas):
        """Cascade hessian"""
        x = [vars["q"], vars["u"], vars["r"], vars["psi"]] # psi is precalculated previously
        y = [grads["q"], grads["u"], grads["r"], grads["Ru"]] # Ru is precalculated previously
        z = [etas["q"], etas["u"], etas["r"], etas["Ru"]] # Ru is precalculated previously

        out = 0
        # compute hessian by iterating from level 3 to level 1
        for id in range(self.noper)[::-1]:
            d2f = self.d2F[id](x, y, z) # returns d2fx(y,z), function understands if y and z are the same                       
            if id==0:# last iteration
                out += d2f
                break
            
            fx, dfy = self.dF[id](x, y) # returns a pair [fx,dfx(y)]
            
            if y is z: # if y and z are the same compute only one
                dfz = dfy 
            else:
                dfz = self.dF[id](x, z, return_x=False)# returns dfx(z)

            # assign for the next level
            x, y, z  = fx, dfy, dfz
            
            d2f = [x, d2f] # form a pair to send to the next level, x has already been computed
            
            # compute differentials by iterating from level id to level 1
            for idg in range(id)[::-1]:
                d2f = self.dF[idg](*d2f)

            out += d2f

        return out

    def gradients_new(self, vars):
        """Cascade gradients"""
        x = [vars["q"], vars["u"], vars["r"], vars['psi']] # assume psi is precalculated
        y = x # forming output
        
        # compute functional by applying operators in reverse order
        for id in range(1, self.noper)[::-1]: 
            y = self.F[id](y)
        
        # compute gradient by applying operators in order
        for id in range(self.noper):
            y = self.gF[id](x, y)
        
        # place in dictionary
        grads = {}
        grads["q"], grads["u"], grads["r"] = y
        return grads

    ####### F0(x0) = \||x1|-d\|_2^2
    def F0(self, x1):
        return cp.linalg.norm(cp.abs(x1) - self.data) ** 2

    def dF0(self, x, y):
        out = cp.empty_like(x)
        for j in range(self.ndist):
            td = self.data[:, j] * (x[:, j] / (cp.abs(x[:, j])))
            out[:, j] = 2 * (x[:, j] - td)

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

    ####### F1(x21,x22) = D(Dp(x21)\cdot M(x22))
    def F1(self, x):
        x21, x22 = x
        out = cp.empty([len(x22), self.ndist, self.n, self.n], dtype="complex64")
        for j in range(self.ndist):
            out[:, j] = self.D(self.Dp(x21,j) * self.M(x22[:, j], j), j)
        
        return out

    def dF1(self, x, y, return_x=True):
        x21, x22 = x
        y21, y22 = y

        if return_x:
            x1 = cp.empty([self.ntheta, self.ndist, self.n, self.n], dtype="complex64")
        y1 = cp.empty([self.ntheta, self.ndist, self.n, self.n], dtype="complex64")
        for j in range(self.ndist):
            if return_x:
                x1[:, j] = self.D(self.Dp(x21,j) * self.M(x22[:, j], j), j)
            y1[:, j] = self.D(self.Dp(y21,j) * self.M(x22[:, j], j), j)
            y1[:, j] += self.D(self.Dp(x21,j) * self.M(y22[:, j], j), j)
        if return_x:
            return x1, y1
        else:
            return y1

    def gF1(self, x, y):
        y11 = y
        y21 = cp.zeros([1,self.n, self.n], dtype="complex64")
        y22 = cp.empty([self.ntheta, self.ndist, self.npsi, self.npsi], dtype="complex64")

        q, u, r, psi= x

        for j in range(self.ndist):
            mspsi = self.M(self.S(r[:, j], psi), j)
            y21 += cp.sum(self.DpT(self.DT(y11[:, j], j) * np.conj(mspsi),j), axis=0)
            y22[:, j] = self.MT(self.DT(y11[:, j], j) * np.conj(self.Dp(q,j)), j)

        return [y21, y22]

    def d2F1(self, x, y, z):
        x21, x22 = x
        y21, y22 = y
        z21, z22 = z

        y1 = cp.empty([self.ntheta, self.ndist, self.n, self.n], dtype="complex64")
        for j in range(self.ndist):
            if y is z:
                y1[:, j] = 2 * self.D(self.Dp(y21,j) * self.M(y22[:, j], j), j)
            else:
                y1[:, j] = self.D(self.Dp(y21,j) * self.M(z22[:, j], j), j) + self.D(self.Dp(z21,j)* self.M(y22[:, j], j), j)
        return y1

    ####### F2(x31,x32,x33)=(x31,S_{x33}(x32))
    def F2(self, x):
        x31, x32, x33 = x

        x22 = cp.empty([len(x33), self.ndist, self.npsi, self.npsi], dtype="complex64")
        xi1 = cp.fft.fftfreq(self.npsi).astype("float32")
        [xi2, xi1] = cp.meshgrid(xi1, xi1)
        for j in range(self.ndist):
            x33j = x33[:, j, :, None, None]
            w = cp.exp(-2 * cp.pi * 1j * (xi2 * x33j[:, 1] + xi1 * x33j[:, 0]))
            x22[:, j] = cp.fft.ifft2(w * cp.fft.fft2(x32))

        return [x31, x22]

    def dF2(self, x, y, return_x=True):
        x31, x32, x33 = x
        y31, y32, y33 = y

        if return_x:
            x22 = cp.empty([self.ntheta, self.ndist, self.npsi, self.npsi], dtype="complex64")
        y22 = cp.empty([self.ntheta, self.ndist, self.npsi, self.npsi], dtype="complex64")
        xi1 = cp.fft.fftfreq(self.npsi).astype("float32")
        [xi2, xi1] = cp.meshgrid(xi1, xi1)
        for j in range(self.ndist):
            x33j = x33[:, j, :, None, None]
            dx33j = y33[:, j, :, None, None]
            w = cp.exp(-2 * cp.pi * 1j * (xi2 * x33j[:, 1] + xi1 * x33j[:, 0]))
            w1 = xi1 * dx33j[:, 0] + xi2 * dx33j[:, 1]

            if return_x:
                x22[:, j] = cp.fft.ifft2(w * cp.fft.fft2(x32))

            t = cp.fft.ifft2(w * cp.fft.fft2(y32))
            dt = -2 * cp.pi * 1j * cp.fft.ifft2(w * w1 * cp.fft.fft2(x32))
            y22[:, j] = t + dt

        if return_x:
            return [x31, x22], [y31, y22]
        else:
            return [y31, y22]

    def gF2(self, x, y):
        y21, y22 = y

        y33 = cp.empty([self.ntheta, self.ndist, 2], dtype="float32")
        y32 = cp.zeros([self.ntheta, self.npsi, self.npsi], dtype="complex64")

        xi1 = cp.fft.fftfreq(self.npsi).astype("float32")
        xi2, xi1 = cp.meshgrid(xi1, xi1)

        q, u, r,psi= x

        for j in range(self.ndist):
            rj = r[:, j, :, None, None]
            w = cp.exp(-2 * cp.pi * 1j * (xi2 * rj[:, 1] + xi1 * rj[:, 0]))

            tmp = cp.fft.fft2(y22[:, j])
            y32[:] += cp.fft.ifft2(1 / w * tmp)

            tmp = cp.fft.fft2(psi)

            dt1 = cp.fft.ifft2(w * xi1 * tmp)
            dt2 = cp.fft.ifft2(w * xi2 * tmp)
            dt1 = -2 * cp.pi * 1j * dt1
            dt2 = -2 * cp.pi * 1j * dt2

            y33[:, j, 0] = redot(y22[:, j], dt1, axis=(1, 2))
            y33[:, j, 1] = redot(y22[:, j], dt2, axis=(1, 2))

        return [y21, y32, y33]

    def d2F2(self, x, y, z):
        x31, x32, x33 = x
        y31, y32, y33 = y
        z31, z32, z33 = z

        y22 = cp.empty([self.ntheta, self.ndist, self.npsi, self.npsi], dtype="complex64")
        xi1 = cp.fft.fftfreq(self.npsi).astype("float32")
        [xi2, xi1] = cp.meshgrid(xi1, xi1)
        for j in range(self.ndist):
            x33j = x33[:, j, :, None, None]
            y33j = y33[:, j, :, None, None]
            z33j = z33[:, j, :, None, None]
            w = cp.exp(-2 * cp.pi * 1j * (xi2 * x33j[:, 1] + xi1 * x33j[:, 0]))
            w1 = xi1 * y33j[:, 0] + xi2 * y33j[:, 1]
            w2 = xi1 * z33j[:, 0] + xi2 * z33j[:, 1]
            w12 = (
                xi1**2 * y33j[:, 0] * z33j[:, 0]
                + xi1 * xi2 * (y33j[:, 0] * z33j[:, 1] + y33j[:, 1] * z33j[:, 0])
                + xi2**2 * y33j[:, 1] * z33j[:, 1]
            )

            tmp = cp.fft.fft2(y32)
            dt1 = -2 * cp.pi * 1j * cp.fft.ifft2(w * w1 * tmp)
            tmp = cp.fft.fft2(x32)
            d2t = -4 * cp.pi**2 * cp.fft.ifft2(w * w12 * tmp)
            if y is z:
                y22[:, j] = 2 * dt1 + d2t
            else:
                tmp = cp.fft.fft2(z32)
                dt2 = -2 * cp.pi * 1j * cp.fft.ifft2(w * w2 * tmp)
                y22[:, j] = dt1 + dt2 + d2t

        return [cp.zeros_like(y31), y22]

    ####### F3(x41,x42,x43)=(x41,e^{1j R(x42)},x43)
    def F3(self, x):
        
        x41,x42,x43,x44=x
        #out = self.expR(self.R(x42))
        out = x44
        return x41,out,x43

    def dF3(self, x, y, return_x=True):
        x41, x42, x43, x44 = x
        y41, y42, y43, y44 = y

        if return_x:
            x32 = x44  # already computed
        y32 = x44 * 1j * y44

        if return_x:
            return [x41, x32, x43], [y41, y32, y43]
        else:
            return [y41, y32, y43]

    def gF3(self, x, y):
        y31, y32, y33 = y
        q,  u,r,psi = x

        y42 = (-1j) * self.RT(y32 * cp.conj(psi))

        return [y31, y42, y33]

    def d2F3(self, x, y, z):
        x41, x42, x43, x44 = x
        y41, y42, y43, y44 = y
        z41, z42, z43, z44 = z

        if y is z:
            y32 = x44 * (-(y44**2))
        else:
            y32 = x44 * (-y44 * z44)

        return [cp.zeros_like(y41), y32, cp.zeros_like(y43)]






    ############################ Debug functions #########################
    def minF(self, big_psi):
        res = cp.linalg.norm(cp.abs(big_psi) - self.data) ** 2
        return res
    
    def fwd(self, r, u, q):
        psi = self.expR(self.R(u))
        x = [q,u,r,psi] 
        y = x # forming output        
        # compute functional by applying operators in reverse order
        for id in range(1, len(self.F))[::-1]: 
            y = self.F[id](y)
        
        return y

    def plot_debug(self, vars, etas, top, bottom, alpha, i):
        """Check the minimization functional behaviour"""
        if i % self.vis_step == 0 and self.vis_step != -1 and self.show:
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
                errt[k] = self.minF(tmp)

            t = alpha * (cp.arange(npp)) / (npp // 2)
            tmp = self.fwd(r, u, q)
            errt2 = self.minF(tmp)
            errt2 = errt2 - top * t + 0.5 * bottom * t**2
            plt.plot(
                alpha * np.arange(npp) / (npp // 2),
                errt.get(),
                ".",
                label="approximation",
            )
            plt.plot(
                alpha * np.arange(npp) / (npp // 2),
                errt2.get(),
                ".",
                label="real",
            )
            plt.legend()
            plt.grid()
            plt.show()

    def vis_debug(self, vars, i):
        """Visualization and data saving"""
        if i % self.vis_step == 0 and self.vis_step != -1:
            (q, u, r) = (vars["q"], vars["u"], vars["r"])

            mshow_complex(u[u.shape[0]//2].real + 1j * u[:, u.shape[1]//2].real, self.show)
            mshow_polar(q[0], self.show)
            write_tiff(u.real, f"{self.path_out}/rec_u_real/{i:04}")
            write_tiff(u.imag, f"{self.path_out}/rec_u_imag/{i:04}")
            write_tiff(u[self.npsi // 2].real, f"{self.path_out}/rec_uz/{i:04}")
            write_tiff(u[:, self.npsi // 2].real, f"{self.path_out}/rec_uy/{i:04}")
            write_tiff(np.angle(q), f"{self.path_out}/rec_prb_angle/{i:04}")
            write_tiff(np.abs(q), f"{self.path_out}/rec_prb_abs/{i:04}")
            np.save(f"{self.path_out}/r{i:04}", r)

            fig, ax = plt.subplots(1, 2, figsize=(15, 6))
            if isinstance(vars["r"], cp.ndarray):
                ax[0].plot(vars["r"][:, :, 1].get() - vars["r_init"][:, :, 1].get(), ".")
                ax[1].plot(vars["r"][:, :, 0].get() - vars["r_init"][:, :, 0].get(), ".")
            else:
                ax[0].plot(vars["r"][:, :, 1].get() - vars["r_init"][:, :, 1].get(), ".")
                ax[1].plot(vars["r"][:, :, 0].get() - vars["r_init"][:, :, 0].get(), ".")
            plt.savefig(f"{self.path_out}/rerr{i:04}.png")

            if self.show:
                plt.show()
            plt.close()

    def error_debug(self, vars, i):
        """Visualization and data saving"""
        if i % self.err_step == 0 and self.err_step != -1:
            (q, u, r) = (vars["q"], vars["u"], vars["r"])
            big_psi = self.fwd(r, u, q)
            err = self.minF(big_psi)
            print(f"{i}) {err=:1.5e}", flush=True)
            vars["table"].loc[len(vars["table"])] = [i, err.get(), time.time()]
            name = f"{self.path_out}/conv.csv"
            os.makedirs(os.path.dirname(name), exist_ok=True)
            vars["table"].to_csv(name, index=False)
