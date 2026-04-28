import math
import numpy as np
import cupy as cp
import cupyx.scipy.fft as cufft
from .cuda_kernels import gather_kernel
from .utils import redot, logger


class Tomo:
    """Functionality for Radon transforms and exp"""

    def __init__(self, n, nz, theta, mask_r):
        """Usfft parameters"""
        eps = 1e-3  # accuracy of usfft
        mu = -math.log(eps) / (2 * n * n)
        m  = math.ceil(2 * n / math.pi * math.sqrt(-mu * math.log(eps) + (mu * n) ** 2 / 4))

        # interpolation kernel
        t = cp.linspace(-1 / 2, 1 / 2, n, endpoint=False).astype("float32")
        dx, dy = cp.meshgrid(t, t)
        phi = cp.exp((mu * (n * n) * (dx * dx + dy * dy)).astype("float32")) * (1 - n % 4)

        # (+1,-1) sign arrays for fftshift-via-multiply
        c1dfftshift = (1 - 2 * ((cp.arange(1, n + 1) % 2))).astype("int8")
        c2dtmp      = (1 - 2 * ((cp.arange(1, 2 * n + 1) % 2))).astype("int8")
        c2dfftshift = cp.outer(c2dtmp, c2dtmp)

        mua = cp.array([mu], dtype="float32")

        self.n      = n
        self.ntheta = len(theta)
        self.theta  = cp.array(theta.astype("float32"))

        if mask_r > 0:
            t1d = np.linspace(-1, 1, self.n)
            x, y = np.meshgrid(t1d, t1d)
            circ  = (x**2 + y**2 < mask_r).astype("float32")
            g     = np.exp(-(20**2) * (x**2 + y**2))
            fcirc = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(circ)))
            fg    = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(g)))
            mask  = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fcirc * fg))).real.astype("float32")
            mask /= np.amax(mask)
        else:
            mask = 1.0

        self.mask = mask
        phi *= cp.array(mask / (n * np.sqrt(n * self.ntheta)))
        self.pars = m, mua, phi, c1dfftshift, c2dfftshift
        self._buf_fde = cp.empty([nz, 2 * n, 2 * n], dtype="complex64")

        self._nz       = nz
        self._buf_sino = cp.zeros([self.ntheta, nz, n], dtype="complex64")
        self._plan_2d  = cufft.get_fft_plan(self._buf_fde,  axes=(-2, -1), value_type='C2C')
        self._plan_1d  = cufft.get_fft_plan(self._buf_sino, axes=(-1,),    value_type='C2C')

    def R(self, obj):
        """Radon transform"""
        nz = obj.shape[0]
        n  = self.n
        m, mua, phi, c1dfftshift, c2dfftshift = self.pars

        # STEP0: fill full fde buffer, zero-padding extra z-slices
        self._buf_fde.fill(0)
        cp.multiply(obj, phi, out=self._buf_fde[:nz, n // 2 : 3 * n // 2, n // 2 : 3 * n // 2])
        # STEP1: 2D FFT on full buffer (always matches plan)
        self._buf_fde *= c2dfftshift
        with self._plan_2d:
            cufft.fft2(self._buf_fde, overwrite_x=True)
        self._buf_fde *= c2dfftshift
        # STEP2: NUFFT gather into full buf_sino (extra slices are zero, no effect)
        self._buf_sino.fill(0)
        gather_kernel(
            (math.ceil(n / 32), math.ceil(self.ntheta / 32), self._nz),
            (32, 32, 1),
            (self._buf_sino, self._buf_fde, self.theta, m, mua, n, self.ntheta, self._nz, 0),
        )
        # STEP3: 1D IFFT on full buf_sino
        self._buf_sino *= c1dfftshift
        with self._plan_1d:
            cufft.ifft(self._buf_sino, overwrite_x=True)
        self._buf_sino *= c1dfftshift
        # STEP4: normalization, crop
        result = self._buf_sino[:, :nz] / 4
        if obj.dtype == 'float32':
            result = result.real
        return cp.ascontiguousarray(result)

    def RT(self, data):
        """Adjoint Radon transform"""
        nz = data.shape[1]
        n  = self.n
        m, mua, phi, c1dfftshift, c2dfftshift = self.pars

        # STEP1: copy into full buf_sino, zero-pad, 1D FFT
        self._buf_sino[:, :nz] = (data * c1dfftshift).astype('complex64')
        self._buf_sino[:, nz:] = 0
        with self._plan_1d:
            cufft.fft(self._buf_sino, overwrite_x=True)
        self._buf_sino *= c1dfftshift
        # STEP2: NUFFT scatter from full buf_sino (extra slices are zero, contribute nothing)
        self._buf_fde.fill(0)
        gather_kernel(
            (math.ceil(n / 32), math.ceil(self.ntheta / 32), self._nz),
            (32, 32, 1),
            (self._buf_sino, self._buf_fde, self.theta, m, mua, n, self.ntheta, self._nz, 1),
        )
        # STEP3: 2D IFFT on full buffer (always matches plan)
        self._buf_fde *= c2dfftshift
        with self._plan_2d:
            cufft.ifft2(self._buf_fde, overwrite_x=True)
        self._buf_fde *= c2dfftshift
        # STEP4: unpadding, crop to nz
        result = self._buf_fde[:nz, n // 2 : 3 * n // 2, n // 2 : 3 * n // 2] * phi
        if data.dtype == 'float32':
            result = result.real
        return cp.ascontiguousarray(result)

    def _filter_sino(self, data, filter_name):
        """Apply a 1-D frequency-domain filter along the detector axis (last axis).

        Parameters
        ----------
        data : cupy ndarray, shape [ntheta, nz, n], float32 or complex64
        filter_name : str  — 'ramp', 'shepp', or 'parzen'

        Returns
        -------
        Filtered array with the same shape and dtype as `data`.
        """
        n  = self.n
        f  = cp.fft.fftfreq(n).astype('float32')   # f in [-0.5, 0.5)
        af = cp.abs(f)*4*n

        if filter_name == 'ramp':
            # Ram-Lak: |ω|
            h = af
        elif filter_name == 'shepp':
            # Shepp-Logan: |ω| × sinc(ω)
            # cp.sinc uses normalized sinc: sinc(x) = sin(πx)/(πx), sinc(0) = 1
            h = af * cp.sinc(f)
        elif filter_name == 'parzen':
            # Parzen (B-spline order-4) window applied to the ramp.
            # u = 2|f| maps [0, 0.5] → [0, 1]
            u = 2 * af
            w = cp.where(u <= 0.5,
                         1 - 6*u**2 + 6*u**3,   # inner region
                         2*(1 - u)**3)            # outer region (tapers to 0 at Nyquist)
            h = af * w
        else:
            raise ValueError(
                f"Unknown filter '{filter_name}'. Choose: ramp, shepp, parzen."
            )

        # Apply along last axis; preserve dtype throughout
        d_fft  = cp.fft.fft(data, axis=-1)
        d_fft *= h.astype(d_fft.dtype)         # broadcast [n] over [ntheta, nz, n]
        result = cp.fft.ifft(d_fft, axis=-1)

        if cp.iscomplexobj(data):
            return result.astype(data.dtype)
        return result.real.astype(data.dtype)

    def fbp(self, data, filter_name='ramp'):
        """Filtered back-projection: apply a 1-D filter then RT.

        Parameters
        ----------
        data : array_like [ntheta, nz, n], float32 or complex64
            Sinogram projections (numpy or cupy).
        filter_name : str
            'ramp'   — Ram-Lak ramp filter |ω|
            'shepp'  — Shepp-Logan:        |ω| × sinc(ω)
            'parzen' — Parzen B-spline-4:  |ω| × w_parzen(2ω)

        Returns
        -------
        Reconstruction array [nz, n, n], same dtype as `data`.
        """
        norm_const = np.float32(np.sqrt(self.n / self.ntheta))
        data = cp.asarray(data)
        res = self.RT(self._filter_sino(data, filter_name))
        res *= norm_const
        return  res

    def rec_tomo(self, d, niter=1):
        """Iterative CG tomography reconstruction for initial guess"""

        def minf(Ru, d):
            return np.linalg.norm(Ru - d) ** 2

        u  = cp.zeros([d.shape[1], self.n, self.n], dtype=d.dtype)
        Ru = self.R(u)
        for k in range(niter):
            if k % 32 == 0:
                logger.info(f"rec_tomo iter {k}: err={minf(Ru, d):.6e}")
            tmp   = 2 * (Ru - d)
            grad  = self.RT(tmp)
            Rgrad = self.R(grad)
            if k == 0:
                eta  = -grad
                Reta = -Rgrad
            else:
                beta = redot(Rgrad, Reta) / redot(Reta, Reta)
                eta  = beta * eta  - grad
                Reta = beta * Reta - Rgrad
            alpha = -redot(grad, eta) / (2 * redot(Reta, Reta))
            u  += alpha * eta
            Ru += alpha * Reta

        return u
