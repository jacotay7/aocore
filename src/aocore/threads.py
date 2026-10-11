"""One switch for every CPU thread pool the AO stack uses.

A simulation process can run five independent thread pools: OpenBLAS or MKL
(NumPy, SciPy), OpenMP, SciPy's FFT workers (sized by aocore), aocore's BLAS
cap for matrix Fourier transforms, and Numba's parallel loops (pyturb). Each
sizes itself from the core count by default, so a multiprocessing sweep of
16 workers on 16 cores can run 256 threads. :func:`set_threads` sets all of
them at once, for the libraries already loaded and for those loaded later.
See the "Threads and multiprocessing" page of the documentation.
"""

from __future__ import annotations

import os
import sys
from types import TracebackType
from typing import Any

__all__ = ["THREAD_VARIABLES", "ThreadLimits", "set_threads"]

#: Environment variables :func:`set_threads` sets. Libraries read them when they
#: load (OpenBLAS, MKL, OpenMP, Numba) or on every call (aocore's own).
THREAD_VARIABLES: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "AOCORE_FFT_WORKERS",
    "AOCORE_BLAS_THREADS",
)


class ThreadLimits:
    """The thread settings made by :func:`set_threads`, and how to undo them.

    Use it as a context manager to restore the previous settings on exit, or
    call :meth:`restore`. Without either the settings stay for the life of
    the process (what a worker initializer wants).
    """

    def __init__(self, n: int) -> None:
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError(f"the thread count must be a positive int, got {n!r}")
        self.n = n
        # Once Numba has started its threads, any change of NUMBA_NUM_THREADS
        # makes its next compilation raise, so the variable is left alone then.
        names = [v for v in THREAD_VARIABLES if v != "NUMBA_NUM_THREADS" or not _numba_loaded()]
        self._environ = {name: os.environ.get(name) for name in names}
        for name in names:
            os.environ[name] = str(n)
        self._pools = _limit_loaded_pools(n)
        self._numba = _set_numba_threads(n)

    def restore(self) -> None:
        """Put back the environment, the thread pools and Numba's thread count.

        ``NUMBA_NUM_THREADS`` stays at ``n`` if Numba was imported meanwhile:
        its pool has that size for good, and changing the variable would
        break its next compilation.
        """
        for name, value in self._environ.items():
            if name == "NUMBA_NUM_THREADS" and _numba_loaded():
                continue
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if self._pools is not None:
            self._pools.restore_original_limits()
            self._pools = None
        if self._numba is not None:
            _set_numba_threads(self._numba)
            self._numba = None

    def __enter__(self) -> ThreadLimits:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.restore()

    def __repr__(self) -> str:
        return f"ThreadLimits(n={self.n})"


def set_threads(n: int) -> ThreadLimits:
    """Run every CPU thread pool of the AO stack with ``n`` threads.

    In this process, right away:

    * the BLAS (OpenBLAS, MKL, BLIS) and OpenMP libraries already loaded are
      limited through threadpoolctl;
    * aocore's FFT workers and BLAS cap (``AOCORE_FFT_WORKERS``,
      ``AOCORE_BLAS_THREADS``, read on every call) become ``n``;
    * if Numba is already imported, ``numba.set_num_threads`` is called
      (capped at the size of its pool, which was fixed when it loaded).
      Numba is never imported for this.

    It also sets the environment variables of :data:`THREAD_VARIABLES` to
    ``n``, which libraries loaded later (Numba, a BLAS pulled in by another
    package) and child processes read at start-up. ``NUMBA_NUM_THREADS`` is
    left alone once Numba is imported: changing it after Numba has started
    its threads makes every later compilation raise.

    Use it as a function at the top of a worker (the settings then last for
    the process), or as a context manager::

        with aocore.set_threads(4):
            run_simulation()  # previous settings are restored afterwards

    Notes
    -----
    OpenMP and Numba thread counts are per calling thread: calling this from
    the main thread does not limit OpenMP regions started by other Python
    threads. In a ``multiprocessing`` or ``concurrent.futures`` pool, call it
    in every worker (``initializer=aocore.set_threads, initargs=(1,)``).

    Parameters
    ----------
    n:
        Threads per pool, ``>= 1``.
    """
    return ThreadLimits(n)


def _limit_loaded_pools(n: int) -> Any:
    """Limit every BLAS/OpenMP library loaded now; None without threadpoolctl."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - threadpoolctl is a dependency
        return None
    # A fresh inspection, not backend's cached controller: libraries loaded
    # since that controller was built (Numba's OpenMP layer, MKL) count too.
    return threadpool_limits(limits=n)


def _numba_loaded() -> bool:
    return "numba" in sys.modules


def _set_numba_threads(n: int) -> int | None:
    """Set Numba's thread count if it is loaded; return the previous count."""
    numba = sys.modules.get("numba")
    if numba is None:
        return None
    try:
        previous = int(numba.get_num_threads())
        numba.set_num_threads(max(1, min(n, int(numba.config.NUMBA_NUM_THREADS))))
    except Exception:  # pragma: no cover - partial or broken Numba installs
        return None
    return previous
