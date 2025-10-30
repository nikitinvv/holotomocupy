import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import os
import warnings
from cuda_kernels import *
from utils import *
from concurrent.futures import ThreadPoolExecutor
from chunking import gpu_batch

from tomo import Tomo
from propagation import Propagation
from curlySspline import Shift as Shift2

np.set_printoptions(legacy="1.25")

warnings.filterwarnings("ignore", message=f".*peer.*")


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

        # propagator class
        cl_prop = Propagation(args.n, args.ndist, args.wavelength, args.voxelsize, args.distance)
        self.D = cl_prop.D
        self.DT = cl_prop.DT

        cl_shift2 = Shift2(self.n, self.npsi)
        self.curlyS = cl_shift2.curlyS
        self.dcurlyS = cl_shift2.dcurlyS
        self.dcurlySadj = cl_shift2.dcurlySadj
        self.d2curlyS = cl_shift2.d2curlyS
        self.S = cl_shift2.S
        self.Sadj = cl_shift2.Sadj

        nbytes = 32 * args.nchunk * args.npsi**2 * np.dtype("complex64").itemsize

        # create CUDA streams and allocate pinned memory
        self.stream = [[[] for _ in range(3)] for _ in range(args.ngpus)]
        self.pinned_mem = [[] for _ in range(args.ngpus)]
        self.gpu_mem = [[] for _ in range(args.ngpus)]
        self.fker = [[[] for _ in range(args.ndist)] for _ in range(args.ngpus)]
        self.pool_inp = [[] for _ in range(args.ngpus)]
        self.pool_out = [[] for _ in range(args.ngpus)]
        self.pool = ThreadPoolExecutor(args.ngpus)

        for igpu in range(args.ngpus):
            with cp.cuda.Device(igpu):
                self.pinned_mem[igpu] = cp.cuda.alloc_pinned_memory(nbytes)
                self.gpu_mem[igpu] = cp.cuda.alloc(nbytes)
                for k in range(3):
                    self.stream[igpu][k] = cp.cuda.Stream(non_blocking=False)
                fx = cp.fft.fftfreq(2 * args.n, d=args.voxelsize).astype("float32")
                [fx, fy] = cp.meshgrid(fx, fx)
                for j in range(args.ndist):
                    self.fker[igpu][j] = cp.exp(
                        -1j * cp.pi * args.wavelength * args.distance[j] * (fx**2 + fy**2)
                    ).astype("complex64")

                self.pool_inp[igpu] = ThreadPoolExecutor(16 // args.ngpus)
                self.pool_out[igpu] = ThreadPoolExecutor(16 // args.ngpus)

        self.pool_cpu = ThreadPoolExecutor(8)

    def BH(self, d, ref, vars):
        # keep data in class
        self.data = d
        self.ref = ref

        for i in range(self.niter):
            # debug plots
            self.error_debug(vars, i)
            self.vis_debug(vars, i)

            # gradients
            grads = self.gradients(vars)
            # print(np.linalg.norm(grads['u']))
            # print(np.linalg.norm(grads['q']))
            # print(np.linalg.norm(grads['r']))

            grads["u"] *= self.rho[0] ** 2
            grads["q"] *= self.rho[1] ** 2
            grads["r"] *= self.rho[2] ** 2
            grads["Ru"] = self.Rr(grads["u"])

            if i == 0:
                etas = {}
                etas["u"] = -grads["u"]
                etas["q"] = -grads["q"]
                etas["r"] = -grads["r"]
            else:
                # calc beta
                top = self.hessian(vars, grads, etas)
                bottom = self.hessian(vars, etas, etas)
                beta = top / bottom

                etas["u"] = etas["u"] * beta - grads["u"]
                etas["q"] = etas["q"] * beta - grads["q"]
                etas["r"] = etas["r"] * beta - grads["r"]
            etas["Ru"] = self.Rr(etas["u"])

            # calc alpha
            top = -(
                self.redot_batch(grads["u"], etas["u"]) / self.rho[0] ** 2
                + self.redot_batch(grads["q"], etas["q"]) / self.rho[1] ** 2
                + self.redot_batch(grads["r"], etas["r"]) / self.rho[2] ** 2
            )
            bottom = self.hessian(vars, etas, etas)
            alpha = top / bottom
            # debug approxmation
            self.plot_debug(vars, etas, top, bottom, alpha, i)
            vars["u"] += alpha * etas["u"]
            vars["q"] += alpha * etas["q"]
            vars["r"] += alpha * etas["r"]
            vars["psi"] = self.expRr(self.Rr(vars["u"]))
        return vars

    # ###################### Cascade computation of functionals  ######################
    def hessian(self, vars, grads, etas):
        """Cascade hessian"""
        x = [vars["q"], vars["u"], vars["r"], vars["psi"]]  # psi is precalculated previously
        y = [grads["q"], grads["u"], grads["r"], grads["Ru"]]  # Ru is precalculated previously
        z = [etas["q"], etas["u"], etas["r"], etas["Ru"]]  # Ru is precalculated previously
        w = [
            np.zeros_like(etas["q"]),
            np.zeros_like(etas["u"]),
            np.zeros_like(etas["r"]),
            np.zeros_like(etas["Ru"]),
        ]

        # compute hessian by iterating from level 3 to level 1
        for id in range(self.noper)[::-1]:
            # term1
            d2f1 = self.d2F[id](x, y, z)
            d2f2 = self.dF[id](x, w, return_x=False)

            # return sum for the last iter
            if id == 0:
                w = d2f1 + d2f2
                break

            # sum variables
            d2f = [v1 + v2 for v1, v2 in zip(d2f1, d2f2)]

            # recalc differentials for the next kle
            fx, dfy = self.dF[id](x, y)  # returns a pair [fx,dfx(y)]
            if y[0] is z[0]:  # if y and z are the same compute only one
                dfz = dfy
            else:
                dfz = self.dF[id](x, z, return_x=False)  # returns dfx(z)
            x, y, z, w = fx, dfy, dfz, d2f

        w += self.hessianq(vars["q"], grads["q"], etas["q"])
        return w.get()

    def gradients(self, vars):
        """Cascade gradients"""
        x = [vars["q"], vars["u"], vars["r"], vars["psi"]]  # assume psi is precalculated
        y = x  # forming output

        # compute functional by applying operators in reverse order
        for id in range(1, self.noper)[::-1]:
            y = self.F[id](y)

        # compute gradient by applying operators in order
        for id in range(self.noper):
            y = self.gF[id](x, y)

        # place in dictionary
        grads = {}
        grads["q"], grads["u"], grads["r"] = y

        grads["q"] += self.gradientq(vars["q"])
        return grads

    ####### F0(x0) = \||x|-d\|_2^2
    def F0(self, x):
        flg = chunking_flg(locals().values())
        if flg:
            res = []
            for igpu in range(self.ngpus):
                with cp.cuda.Device(igpu):
                    res.append(cp.zeros(1, dtype="float32"))
        else:
            res = cp.zeros(1, dtype="float32")

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _F0(self, res, x, d):
            res[:] += cp.linalg.norm(cp.abs(x) - d) ** 2

        _F0(self, res, x, self.data)
        if flg:
            for k in range(1, len(res)):
                res[0] += res[k]
            res = res[0]
        res = res[0]
        return res

    def dF0(self, x, y, return_x=False):
        res = []
        for igpu in range(self.ngpus):
            with cp.cuda.Device(igpu):
                res.append(cp.zeros(1, dtype="float32"))

        y = np.array(y)

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _dF0(self, res, x, y, d):
            for j in range(self.ndist):
                tmp = 2 * (x[:, j] - d[:, j] * (x[:, j] / (cp.abs(x[:, j]))))
                res += redot(tmp, y[:, j])

        _dF0(self, res, x, y, self.data)
        for k in range(1, len(res)):
            res[0] += res[k]
        res = res[0][0]

        return res

    def gF0(self, x, y):
        flg = chunking_flg(locals().values())
        if flg:
            out = np.empty([y.shape[0], self.ndist, self.n, self.n], dtype="complex64")
        else:
            out = cp.empty([y.shape[0], self.ndist, self.n, self.n], dtype="complex64")

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _gF0(self, out, y, d):
            for j in range(self.ndist):
                td = d[:, j] * (y[:, j] / (cp.abs(y[:, j])))
                out[:, j] = 2 * (y[:, j] - td)

        _gF0(self, out, y, self.data)
        return out

    def d2F0(self, x, y, z):
        flg = chunking_flg(locals().values())
        if flg:
            res = []
            for igpu in range(self.ngpus):
                with cp.cuda.Device(igpu):
                    res.append(cp.zeros(1, dtype="float32"))
        else:
            res = cp.zeros(1, dtype="float32")

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _F0(self, res, x, y, z, d):
            for j in range(self.ndist):
                l0 = x[:, j] / (cp.abs(x[:, j]))
                d0 = d[:, j] / (cp.abs(x[:, j]))
                v1 = cp.sum((1 - d0) * reprod(y[:, j], z[:, j]))
                v2 = cp.sum(d0 * reprod(l0, y[:, j]) * reprod(l0, z[:, j]))
                res += 2 * (v1 + v2)

        _F0(self, res, x, y, z, self.data)
        if flg:
            for k in range(1, len(res)):
                res[0] += res[k]
            res = res[0]
        res = res[0]
        return res

    ####### F1(x21,x22) = D(Dp(x21)\cdot x22)

    def F1(self, x):
        x21, x22 = x
        flg = chunking_flg(locals().values())
        if flg:
            out = np.empty([x22.shape[0], self.ndist, self.n, self.n], dtype="complex64")
        else:
            out = cp.empty([x22.shape[0], self.ndist, self.n, self.n], dtype="complex64")

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _F1(self, out, x22, x21):
            for j in range(self.ndist):
                out[:, j] = self.D(x21[j] * x22[:, j], j)

        _F1(self, out, x22, x21)
        return out

    def dF1(self, x, y, return_x=True):
        x21, x22 = x
        y21, y22 = y

        flg = chunking_flg(locals().values())
        if flg:
            if return_x:
                x1 = np.empty([x22.shape[0], self.ndist, self.n, self.n], dtype="complex64")
            y1 = np.empty([x22.shape[0], self.ndist, self.n, self.n], dtype="complex64")
        else:
            if return_x:
                x1 = np.empty([x22.shape[0], self.ndist, self.n, self.n], dtype="complex64")
            y1 = np.empty([x22.shape[0], self.ndist, self.n, self.n], dtype="complex64")

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _dF10(self, x1, x22, x21):
            for j in range(self.ndist):
                x1[:, j] = self.D(x21[j] * x22[:, j], j)

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _dF11(self, y1, x22, y22, x21, y21):
            for j in range(self.ndist):
                y1[:, j] = self.D(y21[j] * x22[:, j], j)
                y1[:, j] += self.D(x21[j] * y22[:, j], j)

        _dF11(self, y1, x22, y22, x21, y21)
        if return_x:
            _dF10(self, x1, x22, x21)
            return x1, y1
        else:
            return y1

    def gF1(self, x, y):
        y11 = y
        q, u, r, psi = x

        y21 = []
        for igpu in range(self.ngpus):
            with cp.cuda.Device(igpu):
                y21.append(cp.zeros([self.ndist, self.n, self.n], dtype="complex64"))

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _gF10(self, y21, y11, r, psi):
            for j in range(self.ndist):
                mspsi = self.curlyS(psi, r[:, j], 1 / self.norm_magnifications[j])
                y21[j] += cp.sum(self.DT(y11[:, j], j) * np.conj(mspsi), axis=0)

        _gF10(self, y21, y11, r, psi)
        for k in range(1, len(y21)):
            y21[0] += y21[k]
        y21 = y21[0]

        y22 = np.empty([self.ntheta, self.ndist, self.n, self.n], dtype="complex64")

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _gF11(self, y22, y11, q):
            for j in range(self.ndist):
                y22[:, j] = self.DT(y11[:, j], j) * np.conj(q[j])

        _gF11(self, y22, y11, q)

        return [y21, y22]

    def d2F1(self, x, y, z):
        x21, x22 = x
        y21, y22 = y
        z21, z22 = z

        y1 = np.empty([self.ntheta, self.ndist, self.n, self.n], dtype="complex64")
        if y22 is z22:

            @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
            def _d2F1(self, y1, y22, y21):
                for j in range(self.ndist):
                    y1[:, j] = 2 * self.D(y21[j] * y22[:, j], j)

            _d2F1(self, y1, y22, y21)
        else:

            @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
            def _d2F1(self, y1, y22, z22, y21, z21):
                for j in range(self.ndist):
                    y1[:, j] = self.D(y21[j] * z22[:, j], j) + self.D(z21[j] * y22[:, j], j)

            _d2F1(self, y1, y22, z22, y21, z21)
        return y1

    def F2(self, x):
        x31, x32, x33 = x
        x22 = np.empty([len(x33), self.ndist, self.n, self.n], dtype="complex64")

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _F2(self, x22, x32, x33):
            for k in range(self.ndist):
                x22[:, k] = self.curlyS(x32, x33[:, k], 1 / self.norm_magnifications[k])

        _F2(self, x22, x32, x33)
        return [x31, x22]

    def dF2(self, x, y, return_x=True):
        x31, x32, x33 = x
        y31, y32, y33 = y

        y22 = np.zeros([self.ntheta, self.ndist, self.n, self.n], dtype="complex64")

        if return_x:
            x22 = np.zeros([self.ntheta, self.ndist, self.n, self.n], dtype="complex64")

            @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
            def _dF20(self, x22, x32, x33):
                for k in range(self.ndist):
                    r = x33[:, k]
                    x22[:, k] = self.curlyS(x32, r, 1 / self.norm_magnifications[k])

            _dF20(self, x22, x32, x33)

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _dF21(self, y22, x32, y32, x33, y33):
            for k in range(self.ndist):
                r = x33[:, k]
                Deltar = y33[:, k]
                y22[:, k] = self.dcurlyS(x32, r, 1 / self.norm_magnifications[k], y32, Deltar)

        _dF21(self, y22, x32, y32, x33, y33)

        if return_x:
            return [x31, x22], [y31, y22]
        else:
            return [y31, y22]

    def d2F2(self, x, y, z):
        x31, x32, x33 = x
        y31, y32, y33 = y
        z31, z32, z33 = z

        y22 = np.zeros([self.ntheta, self.ndist, self.n, self.n], dtype="complex64")

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _d2F2(self, y22, x32, y32, z32, x33, y33, z33):
            for k in range(self.ndist):
                r = x33[:, k]
                Deltar_y = y33[:, k]
                Deltar_z = z33[:, k]
                y22[:, k] = self.d2curlyS(
                    x32, r, 1 / self.norm_magnifications[k], y32, Deltar_y, z32, Deltar_z
                )

        _d2F2(self, y22, x32, y32, z32, x33, y33, z33)
        return [cp.zeros_like(y31), y22]

    def gF2(self, x, y):
        "Warning the above x is the root x (x3) and not the x for this level (i.e. x2)"
        y21, y22 = y

        y33 = np.empty([self.ntheta, self.ndist, 2], dtype="float32")
        y32 = np.empty([self.ntheta, self.npsi, self.npsi], dtype="complex64")

        x2 = self.F3(x)
        x21, x22, x23 = x2

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _gF20(self, y32, x22, y22, x23):
            tmp = cp.zeros([len(x22), self.npsi, self.npsi], dtype="complex64")
            for k in range(self.ndist):
                psi = x22
                Deltaphi = y22[:, k]
                r = x23[:, k]
                Deltapsi, Deltar = self.dcurlySadj(
                    psi, r, 1 / self.norm_magnifications[k], Deltaphi
                )
                tmp += Deltapsi
            y32[:] = tmp

        _gF20(self, y32, x22, y22, x23)

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _gF21(self, y33, x22, y22, x23):
            for k in range(self.ndist):
                psi = x22
                Deltaphi = y22[:, k]
                r = x23[:, k]
                Deltapsi, Deltar = self.dcurlySadj(
                    psi, r, 1 / self.norm_magnifications[k], Deltaphi
                )
                y33[:, k] = Deltar

        _gF21(self, y33, x22, y22, x23)

        return [y21, y32, y33]

    ######## F3(x41,x42,x43)=(x41,e^{1j R(x42)},x43)
    ####### new F3(x41,x42,x43)=(D_-zeta(dr*e^{1j x41}),e^{1j R(x42)},x43),
    # x41 being the (real-valued) phase of the probe on the detector
    def F3(self, x):

        x41, x42, x43, x44 = x
        # out = self.expR(self.R(x42))
        out = x44
        return x41, out, x43

    def dF3(self, x, y, return_x=True):
        x41, x42, x43, x44 = x
        y41, y42, y43, y44 = y

        if return_x:
            x32 = x44  # already computed

        y32 = np.zeros([self.ntheta, self.npsi, self.npsi], dtype="complex64")

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _dF3(self, y32, x44, y44):
            y32[:] = x44 * 1j * y44

        _dF3(self, y32, x44, y44)
        if return_x:
            return [x41, x32, x43], [y41, y32, y43]
        else:
            return [y41, y32, y43]

    def d2F3(self, x, y, z):
        x41, x42, x43, x44 = x
        y41, y42, y43, y44 = y
        z41, z42, z43, z44 = z

        y32 = np.zeros([self.ntheta, self.npsi, self.npsi], dtype="complex64")
        if y44 is z44:

            @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
            def _d2F3(self, y32, x44, y44):
                y32[:] = x44 * (-(y44**2))

            _d2F3(self, y32, x44, y44)
        else:

            @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
            def _d2F3(self, y32, x44, y44, z44):
                y32[:] = x44 * (-y44 * z44)

            _d2F3(self, y32, x44, y44, z44)

        #### new version
        return [np.zeros_like(y41), y32, np.zeros_like(y43)]

    def gF3(self, x, y):
        y31, y32, y33 = y
        x31, x32, x33, x34 = x
        tmp = np.empty([self.ntheta, self.npsi, self.npsi], dtype="complex64")

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _gF31(self, tmp, y32, x34):
            tmp[:] = (-1j) * y32 * cp.conj(x34)

        _gF31(self, tmp, y32, x34)

        y42 = np.empty([self.npsi, self.npsi, self.npsi], dtype="complex64")

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=1)
        def _gF32(self, y42, tmp):
            y42[:] = self.RT(tmp)
            y42[:].imag = 0

        _gF32(self, y42, tmp)
        # print(f'{np.linalg.norm(tmp)=}')
        # print(f'{np.sum(tmp)=}')
        # for k in range(180):
        #     print(f'{k},{np.linalg.norm(tmp[:,k])=},{np.linalg.norm(y42[k])=}')

        return [y31, y42, y33]

    ############################ Debug functions #########################
    def minF(self, big_psi):
        flg = chunking_flg(locals().values())
        if flg:
            res = []
            for igpu in range(self.ngpus):
                with cp.cuda.Device(igpu):
                    res.append(cp.zeros(1, dtype="float32"))
        else:
            res = cp.zeros(1, dtype="float32")

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _minF(self, res, big_psi, d):
            res[:] += cp.linalg.norm(cp.abs(big_psi) - d) ** 2

        _minF(self, res, big_psi, self.data)
        if flg:
            for k in range(1, len(res)):
                res[0] += res[k]
            res = res[0]
        res = res[0].get()
        return res

    def fwd(self, r, u, q):
        psi = self.expRr(self.Rr(u))

        x = [q, u, r, psi]
        y = x  # forming output
        # compute functional by applying operators in reverse order
        for id in range(1, len(self.F))[::-1]:
            y = self.F[id](y)

        return y

    def plot_debug(self, vars, etas, top, bottom, alpha, i):
        """Check the minimization functional behaviour"""
        if i % self.vis_step == 0 and self.vis_step != -1 and self.show:
            fig = plt.figure(figsize=(4, 4))
            (q, u, r) = (vars["q"], vars["u"], vars["r"])
            (dq2, du2, dr2) = (etas["q"], etas["u"], etas["r"])
            npp = 9
            errt = np.zeros(npp)
            errt2 = np.zeros(npp)
            for k in range(0, npp):
                ut = u + (alpha * k / (npp // 2)) * du2
                qt = q + (alpha * k / (npp // 2)) * dq2
                rt = r + (alpha * k / (npp // 2)) * dr2
                tmp = self.fwd(rt, ut, qt)
                tmpq = self.fwdq(qt)
                errt[k] = self.minF(tmp) + self.minFq(tmpq)

            t = alpha * (np.arange(npp)) / (npp // 2)
            tmp = self.fwd(r, u, q)
            tmpq = self.fwdq(q)

            errt2 = self.minF(tmp) + self.minFq(tmpq)
            errt2 = errt2 - top * t + 0.5 * bottom * t**2
            plt.plot(t, errt, ".", label="real")
            plt.plot(t, errt2, ".", label="appr")
            plt.legend()
            plt.grid()
            plt.show()

    def vis_debug(self, vars, i):
        """Visualization and data saving"""
        if i % self.vis_step == 0 and self.vis_step != -1:
            (q, u, r) = (vars["q"], vars["u"], vars["r"])

            mshow_complex(u[u.shape[0] // 2].real + 1j * u[:, u.shape[1] // 2].real, self.show)
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            if isinstance(vars["r"], cp.ndarray):
                ax[0].plot(vars["r"][:, :, 1] - vars["r_init"][:, :, 1], ".")
                ax[1].plot(vars["r"][:, :, 0] - vars["r_init"][:, :, 0], ".")
            else:
                ax[0].plot(vars["r"][:, :, 1] - vars["r_init"][:, :, 1], ".")
                ax[1].plot(vars["r"][:, :, 0] - vars["r_init"][:, :, 0], ".")

            if self.show:
                plt.show()

    def error_debug(self, vars, i):
        """Visualization and data saving"""
        if i % self.err_step == 0 and self.err_step != -1:
            (q, u, r) = (vars["q"], vars["u"], vars["r"])
            big_psi = self.fwd(r, u, q)
            Dq = self.fwdq(q)
            err = self.minF(big_psi) + self.minFq(Dq)
            print(f"{i}) {err=:1.5e}", flush=True)
            vars["table"].loc[len(vars["table"])] = [i, err, time.time()]
            name = f"{self.path_out}/conv.csv"
            os.makedirs(os.path.dirname(name), exist_ok=True)
            vars["table"].to_csv(name, index=False)

    def redot_batch(self, x, y):
        flg = chunking_flg(locals().values())
        if flg:
            res = []
            for igpu in range(self.ngpus):
                with cp.cuda.Device(igpu):
                    res.append(cp.zeros(1, dtype="float32"))
        else:
            res = cp.zeros(1, dtype="float32")

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _redot(self, res, x, y):
            res[:] += redot(x, y)
            return res

        _redot(self, res, x, y)
        if flg:
            for k in range(1, len(res)):
                res[0] += res[k]
            res = res[0]
        res = res[0].get()
        return res

    def Rr(self, u):

        psi = np.empty([self.ntheta, self.npsi, self.npsi], dtype="complex64")

        @gpu_batch(self.nchunk, self.ngpus, axis_out=1, axis_inp=0)
        def _R(self, psi, u):
            psi[:] = self.R(u)

        _R(self, psi, u)

        return psi

    def expRr(self, ru):

        psi = np.empty([self.ntheta, self.npsi, self.npsi], dtype="complex64")

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _expRr(self, psi, ru):
            psi[:] = np.exp(1j * ru)

        _expRr(self, psi, ru)
        return psi

    def gradientq(self, q):
        gradq = cp.zeros_like(q)
        for j in range(self.ndist):
            tmp = self.D(q[j : j + 1], j)
            td = self.ref[j : j + 1] * (tmp / (cp.abs(tmp)))
            gradq[j : j + 1] += self.DT(2 * (tmp - td), j)
        return self.lamq * gradq

    def hessianq(self, q, dq1, dq2):
        res = 0
        for j in range(self.ndist):
            Dq = self.D(q[j : j + 1], j)
            Ddq1 = self.D(dq1[j : j + 1], j)
            Ddq2 = self.D(dq2[j : j + 1], j)
            l0 = Dq / (cp.abs(Dq))
            d0 = self.ref[j : j + 1] / (cp.abs(Dq))
            v1 = cp.sum((1 - d0) * reprod(Ddq1, Ddq2))
            v2 = cp.sum(d0 * reprod(l0, Ddq1) * reprod(l0, Ddq2))
            res += 2 * (v1 + v2)
        return self.lamq * res

    def minFq(self, Dq):
        res = cp.linalg.norm(cp.abs(Dq) - self.ref) ** 2
        return (self.lamq * res).get()

    def fwdq(self, q):
        Dq = cp.empty_like(q)
        for j in range(self.ndist):
            Dq[j : j + 1] = self.D(q[j : j + 1], j)

        return Dq
