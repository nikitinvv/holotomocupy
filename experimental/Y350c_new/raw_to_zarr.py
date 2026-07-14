"""
Convert /data2/maria/denoised_delta_average32bit_2106x2560x1995.raw
to a chunked zarr (v3) store, uncompressed.

Meant to run on tomodata2 (local /data2) to avoid NFS round-trips.
Streams in z-slabs so peak RAM stays around one chunk-row (~256 MB).
"""

import os
import time
import numpy as np
import zarr

RAW_PATH = "/data2/maria/denoised_delta_average32bit_2106x2560x1995.raw"
ZARR_PATH = "/data2/maria/denoised_delta_average32bit_2106x2560x1995.zarr"

SHAPE  = (2106, 2560, 1995)     # (z, y, x) as in filename
DTYPE  = np.float32
CHUNKS = (128, 256, 256)


def main():
    expected_bytes = int(np.prod(SHAPE)) * np.dtype(DTYPE).itemsize
    actual_bytes = os.path.getsize(RAW_PATH)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"size mismatch: {actual_bytes} bytes on disk vs {expected_bytes} expected "
            f"for shape {SHAPE} dtype {DTYPE}"
        )
    print(f"raw file: {RAW_PATH}  ({actual_bytes/1e9:.2f} GB)")
    print(f"zarr out: {ZARR_PATH}")
    print(f"shape={SHAPE} dtype={DTYPE.__name__} chunks={CHUNKS}")

    arr = zarr.create_array(
        store=ZARR_PATH,
        shape=SHAPE,
        dtype=DTYPE,
        chunks=CHUNKS,
        compressors=None,
        overwrite=True,
    )

    src = np.memmap(RAW_PATH, dtype=DTYPE, mode="r", shape=SHAPE)

    nz = SHAPE[0]
    z_step = CHUNKS[0]
    t0 = time.time()
    for zs in range(0, nz, z_step):
        ze = min(zs + z_step, nz)
        arr[zs:ze] = np.ascontiguousarray(src[zs:ze])
        dt = time.time() - t0
        gb = (ze * SHAPE[1] * SHAPE[2] * 4) / 1e9
        print(f"  wrote z=[{zs}:{ze}] / {nz}   {gb:.2f} GB in {dt:.1f}s "
              f"({gb/max(dt,1e-6):.1f} GB/s)")

    total_dt = time.time() - t0
    print(f"done in {total_dt:.1f}s")


if __name__ == "__main__":
    main()
