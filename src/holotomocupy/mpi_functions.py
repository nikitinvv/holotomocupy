
from mpi4py import MPI
import numpy as np
from .utils import timer


def get_local_chunk(total_size, rank, size):
    chunk_size = total_size // size
    remainder = total_size % size
    if rank < remainder:
        start_idx = rank * (chunk_size + 1)
        end_idx = start_idx + chunk_size + 1
    else:
        start_idx = remainder * (chunk_size + 1) + (rank - remainder) * chunk_size
        end_idx = start_idx + chunk_size
    return start_idx, end_idx


class MPIClass:
    """
    Cache MPI derived datatypes for repeated redistributions between:

      forward:
        src: (n_dst, local_n_src, nz)  -> dst: (local_n_dst, n_src, nz)

      backward:
        src: (local_n_dst, n_src, nz) -> dst: (n_dst, local_n_src, nz)

    where:
      - n_src is block-distributed (local_n_src) across ranks in the "src-slab" layout
      - n_dst is block-distributed (local_n_dst) across ranks in the "dst-slab" layout
      - nz is replicated everywhere
      - partitioning is defined by get_local_chunk()

    Only two lists of MPI datatypes are built (not four): forward-send and
    forward-recv types are reused as backward-recv and backward-send respectively,
    since the subarray layouts are identical by symmetry.
    """

    def __init__(self, comm: MPI.Comm, n_src: int, n_dst: int, nz: int, dtype):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        self.n_src = int(n_src)
        self.n_dst = int(n_dst)
        self.nz = int(nz)
        self.dtype = np.dtype(dtype)

        # local slabs
        self.st_src, self.end_src = get_local_chunk(self.n_src, self.rank, self.size)
        self.st_dst, self.end_dst = get_local_chunk(self.n_dst, self.rank, self.size)
        self.local_n_src = self.end_src - self.st_src
        self.local_n_dst = self.end_dst - self.st_dst

        # base MPI datatype
        try:
            self.base = MPI._typedict[self.dtype.char]
        except KeyError as e:
            raise TypeError(f"Unsupported dtype for direct MPI: {self.dtype}") from e

        # common arrays for Alltoallw (byte displacements)
        self.sendcounts = np.ones(self.size, dtype=np.int32)
        self.recvcounts = np.ones(self.size, dtype=np.int32)
        self.sdispls = np.zeros(self.size, dtype=np.int32)
        self.rdispls = np.zeros(self.size, dtype=np.int32)

        # Two type lists (forward send == backward recv, forward recv == backward send)
        self._types_dst = None  # subarray over the n_dst axis — fwd send / bwd recv
        self._types_src = None  # subarray over the n_src axis — fwd recv / bwd send

        self._build_types()

    def _build_types(self):
        src_shape_fwd = (self.n_dst, self.local_n_src, self.nz)
        dst_shape_fwd = (self.local_n_dst, self.n_src, self.nz)

        types_dst = []
        types_src = []

        for p in range(self.size):
            ds, de = get_local_chunk(self.n_dst, p, self.size)
            ss, se = get_local_chunk(self.n_src, p, self.size)

            # Subarray over the n_dst axis: used as fwd-send-to-p / bwd-recv-from-p
            t_dst = self.base.Create_subarray(
                sizes=src_shape_fwd,
                subsizes=(de - ds, self.local_n_src, self.nz),
                starts=(ds, 0, 0),
                order=MPI.ORDER_C,
            )
            t_dst.Commit()
            types_dst.append(t_dst)

            # Subarray over the n_src axis: used as fwd-recv-from-p / bwd-send-to-p
            t_src = self.base.Create_subarray(
                sizes=dst_shape_fwd,
                subsizes=(self.local_n_dst, se - ss, self.nz),
                starts=(0, ss, 0),
                order=MPI.ORDER_C,
            )
            t_src.Commit()
            types_src.append(t_src)

        self._types_dst = types_dst
        self._types_src = types_src

    def close(self):
        """Free committed MPI datatypes."""
        for lst in (self._types_dst, self._types_src):
            if lst is None:
                continue
            for t in lst:
                t.Free()
        self._types_dst = None
        self._types_src = None

    def __del__(self):
        # best-effort cleanup; explicit close() is preferred.
        # Guard against being called after MPI.Finalize().
        try:
            if not MPI.Is_finalized():
                self.close()
        except Exception:
            pass

    def forward(self, src: np.ndarray, dst: np.ndarray):
        """
        src: (n_dst, local_n_src, nz) -> dst: (local_n_dst, n_src, nz)
        """
        if src.dtype != self.dtype or dst.dtype != self.dtype:
            raise ValueError("dtype mismatch")
        if src.shape != (self.n_dst, self.local_n_src, self.nz):
            raise ValueError(f"src.shape={src.shape} expected {(self.n_dst, self.local_n_src, self.nz)}")
        if dst.shape != (self.local_n_dst, self.n_src, self.nz):
            raise ValueError(f"dst.shape={dst.shape} expected {(self.local_n_dst, self.n_src, self.nz)}")

        self.comm.Alltoallw(
            [src, self.sendcounts, self.sdispls, self._types_dst],
            [dst, self.recvcounts, self.rdispls, self._types_src],
        )

    def backward(self, src: np.ndarray, dst: np.ndarray):
        """
        src: (local_n_dst, n_src, nz) -> dst: (n_dst, local_n_src, nz)
        """
        if src.dtype != self.dtype or dst.dtype != self.dtype:
            raise ValueError("dtype mismatch")
        if src.shape != (self.local_n_dst, self.n_src, self.nz):
            raise ValueError(f"src.shape={src.shape} expected {(self.local_n_dst, self.n_src, self.nz)}")
        if dst.shape != (self.n_dst, self.local_n_src, self.nz):
            raise ValueError(f"dst.shape={dst.shape} expected {(self.n_dst, self.local_n_src, self.nz)}")

        self.comm.Alltoallw(
            [src, self.sendcounts, self.sdispls, self._types_src],
            [dst, self.recvcounts, self.rdispls, self._types_dst],
        )

    @timer
    def redist(self, src: np.ndarray, dst: np.ndarray, direction="forward"):
        if direction == "forward":
            return self.forward(src, dst)
        elif direction == "backward":
            return self.backward(src, dst)
        else:
            raise ValueError("direction must be 'forward' or 'backward'")

    def allreduce(self, arr):
        return self.comm.allreduce(arr, op=MPI.SUM)
