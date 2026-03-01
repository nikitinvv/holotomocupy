import cupy as cp
import numpy as np
import os
from .utils import *
import nvtx

class Chunking:
    def __init__(self, nbytes, chunk):
        self.gpu_mem = cp.cuda.alloc(nbytes)
        self.stream  = [cp.cuda.Stream(non_blocking=True) for _ in range(3)]
        self.chunk   = chunk

    def gpu_batch(self, axis_out=0, axis_inp=0, nout=1):
        """
        Single-GPU chunked processing of functions with syntax
        f(out1_proper, ..., out1_nonproper, ...,
          inp1_proper, ..., inp1_nonproper, ..., inp1, inp2, ...)

        where
        out*_proper  are numpy or cupy arrays whose shape[axis_out] equals the
                     chunking dimension size. Numpy arrays are transferred D2H
                     per chunk; CuPy arrays are written in-place on the GPU.
        inp*_proper  are numpy or cupy arrays whose shape[axis_inp] equals the
                     chunking dimension size. Numpy arrays are transferred H2D
                     per chunk; CuPy arrays are sliced directly on the GPU.
        out*_nonproper are CuPy arrays of non-chunking shape (filled in-place,
                     no CPU transfer).
        inp*_nonproper are numpy/CuPy arrays of non-chunking shape (replicated
                     to the GPU once).
        """

        def decorator(func):
            def inner(*args):
                # if no numpy arrays present, run the function directly on GPU
                if not any(isinstance(a, np.ndarray) for a in args):
                    func(*args)
                    return

                cl  = args[0]
                out = args[1 : 1 + nout]
                inp = args[1 + nout :]

                size = inp[0].shape[axis_inp]

                proper_inp,   nonproper_inp   = 0, 0
                proper_out,   nonproper_out   = 0, 0

                for k in range(len(out)):
                    if ((isinstance(out[k], np.ndarray) or isinstance(out[k], cp.ndarray))
                            and len(out[k].shape) > axis_out + 1
                            and out[k].shape[axis_out] == size):
                        proper_out += 1
                    elif isinstance(out[k], cp.ndarray):
                        nonproper_out += 1

                for k in range(len(inp)):
                    if ((isinstance(inp[k], np.ndarray) or isinstance(inp[k], cp.ndarray))
                            and len(inp[k].shape) > axis_inp + 1
                            and inp[k].shape[axis_inp] == size):
                        proper_inp += 1
                    elif isinstance(inp[k], np.ndarray) or isinstance(inp[k], cp.ndarray):
                        nonproper_inp += 1

                # build argument lists for the single GPU
                ginp = [x for x in inp[:proper_inp]]
                ginp.extend(inp[proper_inp:])

                gout = [x for x in out[:proper_out]]
                gout.extend(out[proper_out:])

                if np.prod(gout[0].shape) == 0:
                    return

                self.run(cl, gout, ginp,
                         proper_inp, nonproper_inp,
                         proper_out, nonproper_out,
                         axis_out, axis_inp, func)

            return inner

        return decorator

    def run(self, cl, out, inp, proper_inp, nonproper_inp, proper_out, nonproper_out, axis_out, axis_inp, func):
        """Run by chunks with overlapped H2D / compute / D2H on three streams."""

        gpu_mem = self.gpu_mem
        stream  = self.stream

        size   = inp[0].shape[axis_inp]
        nchunk = int(np.ceil(size / self.chunk))

        # pre-allocate double-buffered GPU arrays
        out_gpu, offset = self.alloc_double_buffers(out[:proper_out], axis_out, gpu_mem, 0, self.chunk)
        inp_gpu, offset = self.alloc_double_buffers(inp[:proper_inp], axis_inp, gpu_mem, offset, self.chunk)

        # move non-proper numpy inputs to GPU once
        for k in range(proper_inp, proper_inp + nonproper_inp):
            inp[k] = cp.asarray(inp[k])

        def p2g(buf_id, k):
            st  = k * self.chunk
            end = min(size, (k + 1) * self.chunk)
            src = self.mk_slices(axis_inp, slice(st, end))
            dst = self.mk_slices(axis_inp, slice(0, end - st))
            cur_stream = cp.cuda.get_current_stream()
            for j in range(proper_inp):
                if axis_inp == 1:
                    c_src = inp[j][src]
                    c_dst = inp_gpu[buf_id][j][dst]
                    rows      = c_src.shape[0]
                    row_bytes = c_src[0].nbytes
                    cp.cuda.runtime.memcpy2DAsync(
                        c_dst.data.ptr,    c_dst.strides[0],
                        c_src.ctypes.data, c_src.strides[0],
                        row_bytes, rows,
                        cp.cuda.runtime.memcpyHostToDevice,
                        cur_stream.ptr,
                    )
                else:
                    if isinstance(inp[j], cp.ndarray):
                        cp.copyto(inp_gpu[buf_id][j][dst], inp[j][src])
                    else:
                        inp_gpu[buf_id][j][dst].set(inp[j][src])

        def g2p(buf_id, k):
            st  = k * self.chunk
            end = min(size, (k + 1) * self.chunk)
            src = self.mk_slices(axis_out, slice(0, end - st))
            dst = self.mk_slices(axis_out, slice(st, end))
            cur_stream = cp.cuda.get_current_stream()
            for j in range(proper_out):
                if axis_out == 1:
                    c_src = out_gpu[buf_id][j][src]
                    c_dst = out[j][dst]
                    rows      = c_src.shape[0]
                    row_bytes = c_src[0].nbytes
                    cp.cuda.runtime.memcpy2DAsync(
                        c_dst.ctypes.data, c_dst.strides[0],
                        c_src.data.ptr,    c_src.strides[0],
                        row_bytes, rows,
                        cp.cuda.runtime.memcpyDeviceToHost,
                        cur_stream.ptr,
                    )
                else:
                    if isinstance(out[j], cp.ndarray):
                        cp.copyto(out[j][dst], out_gpu[buf_id][j][src])
                    else:
                        out_gpu[buf_id][j][src].get(out=out[j][dst], blocking=False)

        def p(buf_id, k):
            st  = k * self.chunk
            end = min(size, (k + 1) * self.chunk)
            inp_gpu_c = self.slice_bufs(inp_gpu[buf_id], axis_inp, end - st)
            out_gpu_c = self.slice_bufs(out_gpu[buf_id], axis_out, end - st)
            func(
                cl,
                *out_gpu_c,
                *out[proper_out:],
                *inp_gpu_c,
                *inp[proper_inp : proper_inp + nonproper_inp],
                *inp[proper_inp + nonproper_inp :],
            )

        nvtx.push_range("pipeline processing with streams", color="yellow")
        for k in range(nchunk + 2):
            if k < nchunk:
                with stream[k % 3]:
                    p2g(k % 2, k)
            if 0 < k < nchunk + 1:
                with stream[(k - 1) % 3]:
                    p((k - 1) % 2, k - 1)
            if 1 < k:
                with stream[(k - 2) % 3]:
                    g2p((k - 2) % 2, k - 2)
            for s in stream:
                s.synchronize()
        nvtx.pop_range()

    def alloc_double_buffers(self, arrs, axis, gpu_mem, offset, chunk):
        """Allocate double-buffered GPU arrays from the pre-allocated pool."""
        gpu = [[], []]
        for j in (0, 1):
            for a in arrs:
                shape0 = list(a.shape)
                shape0[axis] = chunk
                shape0 = tuple(shape0)
                n       = int(np.prod(shape0))
                nbytes  = n * np.dtype(a.dtype).itemsize
                try:
                    gpu[j].append(cp.ndarray(shape0, dtype=a.dtype, memptr=gpu_mem + offset))
                except Exception as e:
                    raise RuntimeError("Failed to allocate GPU buffers") from e
                offset += nbytes
        return gpu, offset

    ####################### Slicing #########################
    def slice_bufs(self, bufs, axis, n):
        slc = [slice(None)] * 3
        slc[axis] = slice(0, n)
        return [b[tuple(slc)] for b in bufs]

    def mk_slices(self, axis, sl):
        res = [slice(None)] * 3
        res[axis] = sl
        return tuple(res)

    ####################### Simple batched functions #########################
    @timer
    def redot_batch(self, x, y, nout=1):
        """res = Re<x, y>"""
        if isinstance(x, cp.ndarray):
            return redot(x, y).get()
        res = cp.zeros(1, dtype="float32")

        @self.gpu_batch(axis_out=0, axis_inp=0)
        def _redot(self, res, x, y):
            res[:] += redot(x, y)

        _redot(self, res, x, y)
        return res[0].get()

    @timer
    def linear_batch(self, x, y, a, b, out=None):
        """w = ax + by"""
        if out is None:
            out = x
        if isinstance(x, cp.ndarray):
            out[:] = a * x + b * y
            return

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _linear(self, out, x, y, a, b):
            out[:] = a * x + b * y

        _linear(self, out, x, y, a, b)
    
    @timer
    def mulc_batch(self, out, x, a):
        """out = ax"""
        if isinstance(x, cp.ndarray):
            out[:] = a * x
            return

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _mulc(self, out, x, a):
            out[:] = a * x

        _mulc(self, out, x, a)
