"""Background peak-memory sampler for the perf benchmarks.

`@timer` logs GPU and process memory at every timed call, so `parse_perf_log.py`
can report the largest value it happened to SEE. That misses anything that peaks
and is freed between two timed calls -- cuFFT plan work areas and the transient
buffers inside Shift/Tomo are exactly that. This polls instead, on its own
thread, and keeps the maxima.

Three numbers, all per rank:

  dev   cp.cuda.runtime.memGetInfo() -> total - free.  DEVICE-WIDE: on a shared
        GPU it also counts other processes, so `baseline` (what was already in
        use when the sampler started) is recorded and reported alongside.
  pool  cp.get_default_memory_pool().total_bytes() -- everything CuPy has taken
        from the driver, pool free-list included. Never shrinks, so it is a
        high-water mark by construction.
  rss   this process's resident set (psutil), i.e. pinned host memory + the rest.

Usage:
    s = PeakSampler(); s.start()
    ...                       # allocation
    setup = s.snapshot(); s.reset()
    ...                       # the part you care about
    bh = s.snapshot(); s.stop()
"""

import threading

import cupy as cp

try:
    import psutil
    _process = psutil.Process()
except Exception:                                  # psutil missing -> no RSS
    _process = None

GiB = 1024.0**3


class PeakSampler(threading.Thread):
    def __init__(self, period=0.05):
        super().__init__(daemon=True)
        self.period  = period
        self._done   = threading.Event()
        # A new thread starts on device 0; under one-GPU-per-rank pinning that
        # is already the right one, but be explicit for unpinned runs.
        self.dev_id  = cp.cuda.Device().id
        free, total  = cp.cuda.runtime.memGetInfo()
        self.total   = total
        self.baseline = total - free
        self.peak_dev = self.peak_pool = self.peak_rss = 0
        self.sample()

    # ------------------------------------------------------------------ core
    def sample(self):
        free, total = cp.cuda.runtime.memGetInfo()
        self.peak_dev  = max(self.peak_dev, total - free)
        self.peak_pool = max(self.peak_pool, cp.get_default_memory_pool().total_bytes())
        if _process is not None:
            self.peak_rss = max(self.peak_rss, _process.memory_info().rss)

    def run(self):
        with cp.cuda.Device(self.dev_id):
            while not self._done.wait(self.period):
                self.sample()

    def reset(self):
        """Drop the maxima to what is in use right now."""
        self.peak_dev = self.peak_pool = self.peak_rss = 0
        self.sample()

    def snapshot(self):
        """Current maxima, in GB."""
        self.sample()
        return {'dev':  self.peak_dev  / GiB,
                'pool': self.peak_pool / GiB,
                'rss':  self.peak_rss  / GiB,
                'baseline': self.baseline / GiB,
                'total':    self.total    / GiB}

    def stop(self):
        self._done.set()
        if self.is_alive():
            self.join(timeout=2)
        return self.snapshot()


def reduce_peaks(comm, snap):
    """Max of each field over all ranks (every rank must call this)."""
    from mpi4py import MPI
    return {k: comm.allreduce(v, op=MPI.MAX) for k, v in snap.items()}
