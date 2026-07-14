import cupy as cp
import cupyx.scipy.fft as cufft


class PropagationFar:
    """Far-field (Fraunhofer) propagation for 2-D ptychography.

    D(psi) is a centered orthogonal 2-D FFT: measurement = |D(prb * obj)|.
    The propagator is unitary, so DT is both the exact adjoint and the
    inverse of D. No sample-to-detector distance appears in the model.

    Interface mirrors Propagation.D / Propagation.DT (same call signature and
    input/output shape) so this class can be swapped into reconstruction
    modules that were written against Propagation. The distance index ``j``
    is accepted for signature compatibility and ignored.
    """

    def __init__(self, n, nz, ntheta):
        self.n       = n
        self.nz      = nz
        self._ntheta = ntheta

        self._buf = cp.empty([ntheta, nz, n], dtype='complex64')
        self._plan_2d = cufft.get_fft_plan(self._buf, axes=(-2, -1), value_type='C2C')

    def _apply(self, psi, inverse):
        added_dim = psi.ndim == 2
        if added_dim:
            psi = psi[cp.newaxis]

        ntheta = psi.shape[0]
        x = cp.fft.ifftshift(psi, axes=(-2, -1))

        if ntheta == self._ntheta:
            self._buf[:] = x
            with self._plan_2d:
                if inverse:
                    cufft.ifft2(self._buf, overwrite_x=True, norm='ortho')
                else:
                    cufft.fft2(self._buf, overwrite_x=True, norm='ortho')
            out = cp.fft.fftshift(self._buf, axes=(-2, -1)).copy()
        else:
            fn = cufft.ifft2 if inverse else cufft.fft2
            out = cp.fft.fftshift(fn(x, norm='ortho'), axes=(-2, -1))

        return out[0] if added_dim else out

    def D(self, psi, j=0):
        """Forward far-field propagator (centered orthogonal FFT)."""
        return self._apply(psi, inverse=False)

    def DT(self, psi, j=0):
        """Adjoint (= inverse) far-field propagator (centered orthogonal IFFT)."""
        return self._apply(psi, inverse=True)
