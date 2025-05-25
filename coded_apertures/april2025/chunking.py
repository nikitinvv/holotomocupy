import cupy as cp
import numpy as np
from concurrent.futures import wait


def _copy(res, u, st, end):
    res[st:end] = u[st:end]
    return res


def copy(u, res, pool):
    nthreads = pool._max_workers
    nthreads = min(nthreads, u.shape[0])
    nchunk = int(np.ceil(u.shape[0] / nthreads))
    futures = [
        pool.submit(_copy, res, u, k * nchunk, min((k + 1) * nchunk, u.shape[0]))
        for k in range(nthreads)
    ]
    wait(futures)
    return res


def gpu_batch(chunk=8, ngpus=1, axis_out=0, axis_inp=0):
    def decorator(func):
        def inner(*args):
            # if no chunking the just run on the current gpu
            if not any(isinstance(a,np.ndarray) for a in args):
                func(*args)
                return      
            # extract inputs and outputs
            cl = args[0]
            out = args[1]
            inp = args[2:]
            
            # size of the chnunking dimension for out
            if isinstance(out,list):
                size_out = ngpus
            else:
                size_out = out.shape[axis_out]
            # size of the chnunking dimension for inp
            size = inp[0].shape[axis_inp]
            # adjust the number of gpus
            ngpus_adj = min(int(np.ceil(size / chunk)), ngpus)
            # sizes per gpu
            gsize_out = int(np.ceil(size_out / ngpus_adj))
            gsize = int(np.ceil(size / ngpus_adj))

            # proper is the number of variables of type numpy that have proper shape size in the chunking dimension
            # cproper is the number of variables of type cupy that have proper shape size in the chunking dimension
            # non proper is the number of variables of type numpy/cupy that have with nonproper shape size
            proper = 0
            cproper = 0
            nonproper = 0
            for k in range(0, len(inp)):
                if (
                    (isinstance(inp[k], np.ndarray))
                    and len(inp[k].shape) > axis_inp + 1
                    and inp[k].shape[axis_inp] == size
                ):
                    # cpu arrays of the proper shape for processing by chunks
                    proper += 1
                elif (
                    (isinstance(inp[k], cp.ndarray))
                    and len(inp[k].shape) > axis_inp + 1
                    and inp[k].shape[axis_inp] == size
                ):
                    # gpu arrays of the proper shape for processing by chunks
                    cproper += 1
                elif isinstance(inp[k], np.ndarray) or isinstance(inp[k], cp.ndarray):
                    # arrays of nonproper shape for processing by chunks
                    nonproper += 1

            # thread pool for each gpu
            futures = []
            # print(proper,cproper,nonproper)
            for igpu in range(ngpus_adj):
                if axis_out == 0:
                    if isinstance(out,list):
                        gout = out[igpu]
                    else:
                        gout = out[igpu * gsize_out : (igpu + 1) * gsize_out]
                if axis_out == 1:                    
                    gout = out[:, igpu * gsize_out : (igpu + 1) * gsize_out]
                if axis_out == 2:                    
                    gout = out[:, :, igpu * gsize_out : (igpu + 1) * gsize_out]
                
                if np.prod(gout.shape) == 0:
                    break
                if axis_inp == 0:
                    ginp = [
                        x[igpu * gsize : (igpu + 1) * gsize]
                        for x in inp[: proper + cproper]
                    ]
                    if len(inp[proper + cproper :]) > 0:
                        ginp.extend(inp[proper + cproper :])
                
                if axis_inp == 1:
                    ginp = [
                        x[:, igpu * gsize : (igpu + 1) * gsize]
                        for x in inp[: proper + cproper]
                    ]
                    if len(inp[proper + cproper :]) > 0:
                        ginp.extend(inp[proper + cproper :])

                if axis_inp == 2:
                    ginp = [
                        x[:,:, igpu * gsize : (igpu + 1) * gsize]
                        for x in inp[: proper + cproper]
                    ]
                    if len(inp[proper + cproper :]) > 0:
                        ginp.extend(inp[proper + cproper :])

                # regular parallelization case with the same dimension size for inp and out
                if size_out == size:
                    futures.append(
                        cl.pool.submit(
                            run,
                            gout,
                            ginp,
                            chunk,
                            proper,
                            cproper,
                            nonproper,
                            axis_out,
                            axis_inp,
                            cl,
                            func,
                            igpu,
                            ngpus_adj                                                        
                        )
                    )
                else:
                    futures.append(
                        cl.pool.submit(
                            run1,
                            gout,
                            ginp,
                            chunk,
                            proper,
                            cproper,
                            nonproper,
                            axis_inp,
                            cl,
                            func,
                            igpu,
                            ngpus_adj                                 
                        )
                    )   
            wait(futures)
            cp.cuda.Device(0).use()
        return inner

    return decorator


def run(
    out,
    inp,
    chunk,
    proper,
    cproper,
    nonproper,
    axis_out,
    axis_inp,
    cl,
    func,
    igpu,
    ngpus,
):
    """Run by chunks, the case where the size of chunking dimension is the same for inp and out"""

    # set gpu and get references to pinned and gpu memory, and streams
    cp.cuda.Device(igpu).use()
    
    pinned_mem = cl.pinned_mem[igpu]
    gpu_mem = cl.gpu_mem[igpu]
    stream = cl.stream[igpu]
    pool_inp = cl.pool_inp[igpu]
    pool_out = cl.pool_out[igpu]

    size = out.shape[axis_out]
    nchunk = int(np.ceil(size / chunk))
    out_shape0 = list(out.shape)
    out_shape0[axis_out] = chunk

    # take memory from the buffer
    out_gpu = cp.ndarray([2, *out_shape0], dtype=out.dtype, memptr=gpu_mem)
    out_pinned = np.frombuffer(
        pinned_mem, out.dtype, np.prod([2, *out_shape0])
    ).reshape([2, *out_shape0])

    # shift memory pointer
    offset = np.prod([2, *out_shape0]) * np.dtype(out.dtype).itemsize
    # determine the number of inputs and allocate memory for each input chunk
    inp_gpu = [[], []]
    inp_pinned = [[], []]

    for j in range(2):  # do it twice to assign memory pointers
        for k in range(proper):
            inp_shape0 = list(inp[k].shape)
            inp_shape0[axis_inp] = chunk

            # take memory from the buffers
            inp_gpu[j].append(
                cp.ndarray(inp_shape0, dtype=inp[k].dtype, memptr=gpu_mem + offset)
            )
            inp_pinned[j].append(
                np.frombuffer(
                    pinned_mem + offset, inp[k].dtype, np.prod(inp_shape0)
                ).reshape(inp_shape0)
            )

            # shift memory pointer
            offset += np.prod(inp_shape0) * np.dtype(inp[k].dtype).itemsize
        for k in range(proper, proper + cproper):
            inp_shape0 = list(inp[k].shape)
            inp_shape0[axis_inp] = chunk

            # take memory from the buffers
            inp_gpu[j].append(
                cp.ndarray(inp_shape0, dtype=inp[k].dtype, memptr=gpu_mem + offset)
            )

            # shift memory pointer
            offset += np.prod(inp_shape0) * np.dtype(inp[k].dtype).itemsize

    # if nonproper array is numpy, copy it to gpu
    for k in range(proper + cproper, proper + cproper + nonproper):
        inp[k] = cp.asarray(inp[k])
    
    # run by chunks
    for k in range(nchunk + 2):
        if k > 0 and k < nchunk + 1:
            with stream[1]:  # processing
                st, end = (k-1) * chunk, min(size, k * chunk)
                if axis_inp == 0:
                    inp_gpu_c = [a[:end-st] for a in inp_gpu[(k - 1) % 2]]                                    
                elif axis_inp == 1:
                    inp_gpu_c = [a[:,:end-st] for a in inp_gpu[(k - 1) % 2]]                
                elif axis_inp == 2:
                    inp_gpu_c = [a[:,:,:end-st] for a in inp_gpu[(k - 1) % 2]]      

                if axis_out == 0:                
                    out_gpu_c = out_gpu[(k - 1) % 2][:end-st]
                elif axis_out == 1:                
                    out_gpu_c = out_gpu[(k - 1) % 2][:,:end-st]
                elif axis_out == 2:                
                    out_gpu_c = out_gpu[(k - 1) % 2][:,:,:end-st]
                                    
                func(
                    cl,
                    out_gpu_c,
                    *inp_gpu_c,
                    *inp[proper + cproper :],
                )

        if k > 1:
            with stream[2]:  # gpu->cpu copy
                out_gpu[(k - 2) % 2].get(
                    out=out_pinned[(k - 2) % 2], blocking=False
                )  ####Note blocking parameter is not define for old cupy versions
        if k < nchunk:
            
            with stream[0]:  # copy to pinned memory
                st, end = k * chunk, min(size, (k + 1) * chunk)
                for j in range(proper):
                    if axis_inp == 0:
                        copy(inp[j][st:end], inp_pinned[k % 2][j][: end - st], pool_inp)
                    elif axis_inp == 1:
                        copy(
                            inp[j][:, st:end],
                            inp_pinned[k % 2][j][:, : end - st],
                            pool_inp,
                        )
                    elif axis_inp == 2:
                        copy(
                            inp[j][:, :, st:end],
                            inp_pinned[k % 2][j][:, :, : end - st],
                            pool_inp,
                        )
                    inp_gpu[k % 2][j].set(inp_pinned[k % 2][j])
                    #set tail to 0
                    # if axis_inp == 0:
                    #     inp_gpu[k % 2][j][end - st :] = 0
                    # elif axis_inp == 1:
                    #     inp_gpu[k % 2][j][:, end - st :] = 0

                for j in range(proper, cproper + proper):
                    if axis_inp == 0:
                        inp_gpu[k % 2][j][: end - st] = inp[j][st:end]
                        # inp_gpu[k % 2][j][end - st :] = 0
                    elif axis_inp == 1:
                        inp_gpu[k % 2][j][:, : end - st] = inp[j][:, st:end]
                        # inp_gpu[k % 2][j][:, end - st :] = 0
                    elif axis_inp == 2:
                        inp_gpu[k % 2][j][:, :, : end - st] = inp[j][:, :, st:end]

        stream[2].synchronize()
        if k > 1:
            st, end = (k - 2) * chunk, min(size, (k - 1) * chunk)
            if axis_out == 0:
                copy(out_pinned[(k - 2) % 2][: end - st], out[st:end], pool_out)
            if axis_out == 1:
                copy(out_pinned[(k - 2) % 2][:, : end - st], out[:, st:end], pool_out)
            if axis_out == 2:
                copy(out_pinned[(k - 2) % 2][:, :, : end - st], out[:, :, st:end], pool_out)
        stream[0].synchronize()
        stream[1].synchronize()
        

def run1(out, inp, chunk, proper, cproper, nonproper, axis_inp, cl, func, igpu, ngpus):
    """Run by chunks, the case where the size of chunking dimension is not the same for inp and out"""

    cp.cuda.Device(igpu).use()
    
    pinned_mem = cl.pinned_mem[igpu]
    gpu_mem = cl.gpu_mem[igpu]
    stream = cl.stream[igpu]
    pool_inp = cl.pool_inp[igpu]
    
    size = inp[0].shape[axis_inp]
    nchunk = int(np.ceil(size / chunk))    

    inp_gpu = [[], []]
    inp_pinned = [[], []]

    offset = 0
    for j in range(2):  # do it twice to assign memory pointers
        for k in range(proper):
            inp_shape0 = list(inp[k].shape)
            inp_shape0[axis_inp] = chunk

            # take memory from the buffers
            inp_gpu[j].append(
                cp.ndarray(inp_shape0, dtype=inp[k].dtype, memptr=gpu_mem + offset)
            )
            inp_pinned[j].append(
                np.frombuffer(
                    pinned_mem + offset, inp[k].dtype, np.prod(inp_shape0)
                ).reshape(inp_shape0)
            )

            # shift memory pointer
            offset += np.prod(inp_shape0) * np.dtype(inp[k].dtype).itemsize
        for k in range(proper, proper + cproper):
            inp_shape0 = list(inp[k].shape)
            inp_shape0[axis_inp] = chunk

            # take memory from the buffers
            inp_gpu[j].append(
                cp.ndarray(inp_shape0, dtype=inp[k].dtype, memptr=gpu_mem + offset)
            )

            # shift memory pointer
            offset += np.prod(inp_shape0) * np.dtype(inp[k].dtype).itemsize

    for k in range(proper + cproper, proper + cproper + nonproper):
        inp[k] = cp.asarray(inp[k])

    
    # run by chunks
    for k in range(nchunk + 2):
        if k > 0 and k < nchunk + 1:
            with stream[1]:  # processing
                st, end = (k-1) * chunk, min(size, k * chunk)
                if axis_inp == 0:
                    inp_gpu_c = [a[:end-st] for a in inp_gpu[(k - 1) % 2]]                
                elif axis_inp == 1:
                    inp_gpu_c = [a[:,:end-st] for a in inp_gpu[(k - 1) % 2]]                
                elif axis_inp == 2:
                    inp_gpu_c = [a[:,:,:end-st] for a in inp_gpu[(k - 1) % 2]]                
                func(cl, out, *inp_gpu_c, *inp[proper + cproper :])

        if k < nchunk:
            with stream[0]:  # copy to pinned memory
                st, end = k * chunk, min(size, (k + 1) * chunk)
                for j in range(proper):
                    if axis_inp == 0:
                        copy(inp[j][st:end], inp_pinned[k % 2][j][: end - st], pool_inp)
                    elif axis_inp == 1:
                        copy(
                            inp[j][:, st:end],
                            inp_pinned[k % 2][j][:, : end - st],
                            pool_inp,
                        )
                    elif axis_inp == 2:
                        copy(
                            inp[j][:, :, st:end],
                            inp_pinned[k % 2][j][:,:, : end - st],
                            pool_inp,
                        )

                    inp_gpu[k % 2][j].set(inp_pinned[k % 2][j])
                    # # set tail to 0
                    # if axis_inp == 0:
                    #     inp_gpu[k % 2][j][end - st :] = 0
                    # elif axis_inp == 1:
                    #     inp_gpu[k % 2][j][:, end - st :] = 0

                for j in range(proper, cproper + proper):
                    if axis_inp == 0:
                        inp_gpu[k % 2][j][: end - st] = inp[j][st:end]
                        # inp_gpu[k % 2][j][end - st :] = 0
                    elif axis_inp == 1:
                        inp_gpu[k % 2][j][:, : end - st] = inp[j][:, st:end]
                        # inp_gpu[k % 2][j][:, end - st :] = 0                    
                    elif axis_inp == 2:
                        inp_gpu[k % 2][j][:, :, : end - st] = inp[j][:, :, st:end]
                        # inp_gpu[k % 2][j][:, end - st :] = 0                    

        stream[2].synchronize()
        stream[0].synchronize()
        stream[1].synchronize()
    