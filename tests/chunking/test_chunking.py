import logging
import time
import numpy as np
import cupy as cp
from holotomocupy.chunking import Chunking

# logger_config (imported transitively) resets root to WARNING and replaces
# handlers — override here after the import.
logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger("test_chunking")

ATOL = 1e-10  # absolute tolerance for pass/fail

# ── reproducible inputs ───────────────────────────────────────────────────────
np.random.seed(10)
a = np.random.random([15, 64, 64])
b = np.random.random([15, 64, 64])
nchunk = 4
nbytes = 64 * 64 * nchunk * 64

log.info("inputs: a%s  b%s  nchunk=%d  nbytes=%d", a.shape, b.shape, nchunk, nbytes)


def make_pinned(shape, dtype):
    """Allocate a page-locked (pinned) numpy array (required by memcpy2DAsync)."""
    n = int(np.prod(shape))
    buf = cp.cuda.alloc_pinned_memory(n * np.dtype(dtype).itemsize)
    arr = np.frombuffer(buf, dtype=dtype, count=n).reshape(shape)
    arr[:] = 0
    return arr


class mClass():
    def __init__(self, nbytes):
        self.cl_chunking = Chunking(nbytes, nchunk)

    def msum(self, a, b, d, c):
        """
        axis_inp=0: chunks along dim-0 of inputs  (a, b are [15,64,64])
        axis_out=1: chunks along dim-1 of outputs (out0, out00 are [64,15,64])
        out0,out00 – proper pinned outputs (D2H via memcpy2DAsync)
        out1,out2  – nonproper GPU accumulators (no chunking)
        """
        out0  = make_pinned(a.swapaxes(0, 1).shape, a.dtype)
        out00 = make_pinned(a.swapaxes(0, 1).shape, a.dtype)
        out1  = cp.zeros([1],        dtype='float64')
        out2  = cp.zeros([64, 1, 64], dtype='float64')

        @self.cl_chunking.gpu_batch(axis_inp=0, axis_out=1, nout=4)
        def _msum(self, out0, out00, out1, out2, a, b, d, c):
            out0[:]  = (a + b + d * c).swapaxes(0, 1)
            out00[:] = (a - b).swapaxes(0, 1)
            out1[:]  += cp.linalg.norm(a - b) ** 2
            out2[:]  += (cp.linalg.norm(a + b, axis=0) ** 2)[None].swapaxes(0, 1)

        _msum(self, out0, out00, out1, out2, a, b, d, c)
        return out0, out00, out1[0].get(), out2.get()


def check(name, got, ref, atol=ATOL):
    err  = np.linalg.norm(got - ref)
    norm = np.linalg.norm(ref)
    ok   = err < atol
    status = "PASS" if ok else "FAIL"
    log.info("%-6s %-30s  |ref|=%.6e  |err|=%.6e  [%s]", status, name, norm, err, status)
    return ok


# ── run ───────────────────────────────────────────────────────────────────────
cl = mClass(nbytes)
d  = cp.random.random(a.shape)[0:1]

log.info("--- msum (axis_inp=0, axis_out=1, nout=4) ---")
t0 = time.perf_counter()
o1, o11, o2, o22 = cl.msum(a, b, d, 2)
elapsed = time.perf_counter() - t0
log.info("elapsed: %.4f s", elapsed)

# ── reference (CPU) ───────────────────────────────────────────────────────────
d_np  = d.get()
oo1   = (a + b + 2 * d_np).swapaxes(0, 1)
oo11  = (a - b).swapaxes(0, 1)
oo2   = np.linalg.norm(a - b) ** 2
oo22  = (np.linalg.norm(a + b, axis=0) ** 2)[None].swapaxes(0, 1)

# ── checks ────────────────────────────────────────────────────────────────────
log.info("--- checking outputs ---")
results = [
    check("out0",  o1,  oo1),   # proper pinned, axis_out=1, swapaxes
    check("out00", o11, oo11),  # proper pinned, axis_out=1, swapaxes
    check("out1",  o2,  oo2),   # nonproper scalar accumulator
    check("out2",  o22, oo22),  # nonproper 3-D accumulator
]

log.info("--- summary ---")
n_pass = sum(results)
n_fail = len(results) - n_pass
log.info("%d/%d tests passed", n_pass, len(results))
if n_fail:
    log.error("%d test(s) FAILED", n_fail)
    raise SystemExit(1)
