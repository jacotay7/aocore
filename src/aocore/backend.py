"""Array backends: NumPy/SciPy on the CPU and CuPy on CUDA GPUs.

Numerical routines in the AO stack take their array namespace from a
:class:`Backend`, so the same algorithm runs on either device. A backend fixes
three things:

* ``device`` — ``"cpu"`` (NumPy arrays, :mod:`scipy.fft` with all cores) or
  ``"gpu"`` (CuPy arrays, cuFFT/cuBLAS);
* ``precision`` — ``"double"`` (float64/complex128) or ``"single"``
  (float32/complex64). CPU defaults to double, GPU to single;
* the unitary FFT used by every propagator.

Arrays returned by a GPU solver stay on the GPU. :func:`to_numpy` is the one
explicit host boundary.
"""

from __future__ import annotations

import contextlib
import functools
import os
import threading
from dataclasses import dataclass
from typing import Any, Literal, Union

import numpy as np

DeviceName = Literal["cpu", "gpu", "auto"]
PrecisionName = Literal["single", "double"]

__all__ = [
    "Backend",
    "BackendLike",
    "backend_of",
    "get_backend",
    "gpu_available",
    "to_numpy",
]


@functools.lru_cache(maxsize=1)
def _cupy() -> Any:
    try:
        import cupy
    except ImportError:  # pragma: no cover - depends on the environment
        return None
    return cupy


@functools.lru_cache(maxsize=1)
def gpu_available() -> bool:
    """Return whether CuPy is importable and sees at least one CUDA device."""
    cupy = _cupy()
    if cupy is None:  # pragma: no cover - depends on the environment
        return False
    try:
        return bool(cupy.cuda.runtime.getDeviceCount() > 0)
    except Exception:  # pragma: no cover - driver/runtime missing
        return False


def _cpu_workers(size: int = 1 << 30) -> int:
    """SciPy FFT threads for a transform of ``size`` total elements.

    One thread per ~16k elements, capped at the core count: small transforms
    lose more to thread start-up and contention than they gain (a 48x48 FFT
    is ~10x slower on 16 threads than on one when the cores are busy).
    ``AOCORE_FFT_WORKERS`` (or the older ``SOLVEPHASE_FFT_WORKERS``) fixes
    the count.
    """
    env = os.environ.get("AOCORE_FFT_WORKERS") or os.environ.get("SOLVEPHASE_FFT_WORKERS")
    if env:
        return max(1, int(env))
    return max(1, min(os.cpu_count() or 1, size >> 14))


_scratch = threading.local()


def _workspace(name: str, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
    """A reusable host array of this thread (contents left from its last use).

    Large fresh arrays are new pages from the kernel: glibc returns freed
    blocks above its mmap threshold to the OS, and writing to the new ones
    costs a page fault per 4 KiB, which took as long as the FFT itself on an
    Arm server. A few arrays (up to 64 MiB each) are kept per thread.
    """
    buffers: dict[tuple[Any, ...], np.ndarray] | None = getattr(_scratch, "buffers", None)
    if buffers is None:
        buffers = _scratch.buffers = {}
    key = (name, shape, np.dtype(dtype).str)
    array = buffers.pop(key, None)
    if array is None:
        array = np.empty(shape, dtype=dtype)
        if array.nbytes > 64 * 2**20:
            return array
        while len(buffers) >= 6:
            buffers.pop(next(iter(buffers)))  # least recently used
    buffers[key] = array
    return array


def _line_bunches(lines: int, size: int) -> int:
    """``lines`` rounded up to a multiple of 16, at most ``size``.

    SciPy's FFTs transform lines in SIMD bunches (16 lines in ducc) and finish
    a remainder with code that can round differently, so a pruned pass over
    whole bunches keeps each line on the path the full transform takes.
    """
    return min(size, -(-lines // 16) * 16)


@functools.lru_cache(maxsize=1)
def _fft2_first_axis() -> int:
    """The axis SciPy's out-of-place ``fft2`` transforms first: -1 or -2.

    It depends on the SciPy version (pocketfft up to 1.17 follows ``axes``,
    ducc from 1.18 starts with the last axis), and the two orders round
    differently, so it is found once by comparing with explicit passes.
    """
    from scipy import fft

    rng = np.random.default_rng(0)
    probe = rng.standard_normal((48, 40)) + 1j * rng.standard_normal((48, 40))
    reference = fft.fft2(probe, workers=1)
    for axis in (-1, -2):
        passes = fft.fft(fft.fft(probe, axis=axis, workers=1), axis=-3 - axis, workers=1)
        if np.array_equal(passes, reference):
            return axis
    return -1  # pragma: no cover - neither order reproduces fft2 exactly


@functools.lru_cache(maxsize=1)
def _threadpool_controller() -> Any:
    try:
        from threadpoolctl import ThreadpoolController
    except ImportError:  # pragma: no cover - threadpoolctl is a dependency
        return None
    return ThreadpoolController()


def _blas_threads(work: float) -> int:
    """BLAS threads for a matrix product of about ``work`` multiply-adds.

    OpenBLAS defaults to one thread per logical CPU, which for the
    mid-sized complex products of the matrix Fourier transform is 2-4x slower
    than a few threads (hyper-threads and synchronization cost more than they
    give). ``AOCORE_BLAS_THREADS`` (or the older ``SOLVEPHASE_BLAS_THREADS``)
    fixes the count.
    """
    env = os.environ.get("AOCORE_BLAS_THREADS") or os.environ.get("SOLVEPHASE_BLAS_THREADS")
    if env:
        return max(1, int(env))
    cores = max(1, (os.cpu_count() or 2) // 2)
    return max(1, min(cores, 4 if work <= 2**28 else 8))


@dataclass(frozen=True)
class Backend:
    """One array namespace, device and floating-point precision.

    Use :func:`get_backend` rather than constructing this directly.

    Attributes
    ----------
    device:
        ``"cpu"`` or ``"gpu"``.
    precision:
        ``"double"`` or ``"single"``.
    """

    device: str
    precision: str

    # ------------------------------------------------------------------ basics
    @property
    def xp(self) -> Any:
        """The array namespace: :mod:`numpy` or :mod:`cupy`."""
        return _cupy() if self.device == "gpu" else np

    @property
    def is_gpu(self) -> bool:
        """Whether arrays live on a CUDA device."""
        return self.device == "gpu"

    @property
    def real_dtype(self) -> np.dtype[Any]:
        """float64 for double precision, float32 for single."""
        return np.dtype(np.float64 if self.precision == "double" else np.float32)

    @property
    def complex_dtype(self) -> np.dtype[Any]:
        """complex128 for double precision, complex64 for single."""
        return np.dtype(np.complex128 if self.precision == "double" else np.complex64)

    @property
    def eps(self) -> float:
        """Machine epsilon of :attr:`real_dtype`."""
        return float(np.finfo(self.real_dtype).eps)

    def __repr__(self) -> str:
        return f"Backend(device={self.device!r}, precision={self.precision!r})"

    # ---------------------------------------------------------------- creation
    def asarray(self, value: Any, dtype: Any = None) -> Any:
        """Move ``value`` onto this backend.

        ``dtype`` may be a NumPy dtype, ``"real"`` or ``"complex"`` (this
        backend's working precision), or ``None`` to promote real input to
        :attr:`real_dtype` and complex input to :attr:`complex_dtype`.
        Integer and boolean arrays keep their type when ``dtype`` is None.
        """
        if dtype == "real":
            dtype = self.real_dtype
        elif dtype == "complex":
            dtype = self.complex_dtype
        array = self.xp.asarray(value) if self.is_gpu else np.asarray(to_numpy(value))
        if dtype is None:
            kind = array.dtype.kind
            if kind == "c":
                dtype = self.complex_dtype
            elif kind == "f":
                dtype = self.real_dtype
            else:
                return array
        return array.astype(dtype, copy=False)

    def zeros(self, shape: Any, dtype: Any = "real") -> Any:
        """Zero-filled array in working precision (``"real"``/``"complex"``)."""
        return self.xp.zeros(shape, dtype=self._dtype(dtype))

    def empty(self, shape: Any, dtype: Any = "real") -> Any:
        """Uninitialized array in working precision (``"real"``/``"complex"``)."""
        return self.xp.empty(shape, dtype=self._dtype(dtype))

    def _dtype(self, dtype: Any) -> Any:
        if dtype == "real":
            return self.real_dtype
        if dtype == "complex":
            return self.complex_dtype
        return dtype

    # -------------------------------------------------------------------- FFTs
    def fft2(self, array: Any, *, axes: tuple[int, int] = (-2, -1)) -> Any:
        """Unitary (``norm="ortho"``) two-dimensional FFT, uncentred."""
        if self.is_gpu:
            return self.xp.fft.fft2(array, axes=axes, norm="ortho")
        from scipy import fft

        return fft.fft2(array, axes=axes, norm="ortho", workers=_cpu_workers(array.size))

    def ifft2(self, array: Any, *, axes: tuple[int, int] = (-2, -1)) -> Any:
        """Unitary inverse of :meth:`fft2` (also its adjoint)."""
        if self.is_gpu:
            return self.xp.fft.ifft2(array, axes=axes, norm="ortho")
        from scipy import fft

        return fft.ifft2(array, axes=axes, norm="ortho", workers=_cpu_workers(array.size))

    def padded_fft2(
        self,
        array: Any,
        shape: tuple[int, int],
        out_shape: tuple[int, int] | None = None,
        *,
        inverse: bool = False,
        weights: Any = None,
        out_weights: Any = None,
        out: Any = None,
    ) -> Any:
        """Unitary 2-D FFT of a zero-padded block, cropped to a corner.

        Returns ``fft2(grid)[..., :my, :mx] * out_weights`` (``ifft2`` when
        ``inverse``; ``out_weights`` defaults to 1) in working precision, where
        ``grid`` is a zero ``(..., *shape)`` array holding ``array * weights``
        (or ``array``) in its leading ``(ny, nx)`` corner and
        ``(my, mx) = out_shape`` (default ``shape``). The result is written to
        ``out`` when given.

        On the CPU the transform skips what the crop and the padding make
        redundant. A 2-D FFT is a pass of 1-D transforms along one axis, then
        the other; lines of zeros transform to zeros, so the first pass runs
        only over the lines that hold data (the ``ny`` rows when it runs along
        the last axis) and the second only over the lines that are kept (the
        ``mx`` columns). A half-filled grid cropped to half its width costs
        about half the FFT work and memory traffic, and the work arrays are
        reused between calls. Every value is computed by the same two 1-D
        passes, in the same order and with the unitary scale applied in the
        first, as :meth:`fft2` computes it; the results are bit-identical
        unless the full transform's thread partition puts a needed line in a
        short SIMD remainder group, where a value may differ in the last bit.
        On the GPU it is the full transform followed by the crop.
        """
        ny, nx = array.shape[-2:]
        big_y, big_x = (int(v) for v in shape)
        my, mx = (big_y, big_x) if out_shape is None else (int(v) for v in out_shape)
        if not (ny <= big_y and nx <= big_x and my <= big_y and mx <= big_x):
            raise ValueError(
                f"block {(ny, nx)} and crop {(my, mx)} must fit in the FFT grid {(big_y, big_x)}"
            )
        lead = array.shape[:-2]
        if weights is not None:
            lead = np.broadcast_shapes(lead, weights.shape[:-2])
        cdt = self.complex_dtype
        rows, cols = _line_bunches(ny, big_y), _line_bunches(mx, big_x)
        first_axis = _fft2_first_axis() if not self.is_gpu else -1
        if first_axis == -2:
            rows, cols = _line_bunches(my, big_y), _line_bunches(nx, big_x)
        if self.is_gpu or (rows == big_y and cols == big_x):
            grid = self.xp.zeros((*lead, big_y, big_x), dtype=cdt)
            self._fill(grid[..., :ny, :nx], array, weights)
            spectrum = self.ifft2(grid) if inverse else self.fft2(grid)
            return self._crop(spectrum[..., :my, :mx], out_weights, out, copy=False)
        from scipy import fft

        transform = fft.ifft if inverse else fft.fft
        # Pass order and scaling follow fft2 (see _fft2_first_axis): the
        # whole norm="ortho" factor 1 / sqrt(big_y * big_x), computed in long
        # double, is applied in the first pass. For a square grid that is the
        # 1-D "forward" (inverse: "backward") norm; otherwise it is applied
        # here, the same multiplication SciPy does after the unscaled pass.
        square = big_y == big_x
        unscaled = "forward" if inverse else "backward"
        if first_axis == -1:
            # Rows first: only the `rows` rows holding data, then only the
            # `cols` columns that are kept.
            first = _workspace("padded_fft2 a", (*lead, rows, big_x), cdt)
        else:
            # Columns first: only the `cols` columns holding data, then only
            # the `rows` rows that are kept.
            first = _workspace("padded_fft2 a", (*lead, big_y, cols), cdt)
        self._fill(first[..., :ny, :nx], array, weights)
        first[..., :ny, nx:] = 0
        first[..., ny:, :] = 0
        first = transform(
            first,
            axis=first_axis,
            norm=("backward" if inverse else "forward") if square else unscaled,
            overwrite_x=True,
            workers=_cpu_workers(first.size),
        )
        if not square:
            factor = 1 / np.sqrt(np.longdouble(big_y) * np.longdouble(big_x))
            first.view(self.real_dtype)[...] *= factor.astype(self.real_dtype)
        if first_axis == -1:
            second = _workspace("padded_fft2 b", (*lead, big_y, cols), cdt)
            second[..., :rows, :] = first[..., :cols]
            second[..., rows:, :] = 0
        else:
            second = _workspace("padded_fft2 b", (*lead, rows, big_x), cdt)
            second[..., :cols] = first[..., :rows, :]
            second[..., cols:] = 0
        second = transform(
            second,
            axis=-3 - first_axis,
            norm=unscaled,
            overwrite_x=True,
            workers=_cpu_workers(second.size),
        )
        return self._crop(second[..., :my, :mx], out_weights, out, copy=True)

    def _fill(self, target: Any, array: Any, weights: Any) -> None:
        """Write ``array`` (times ``weights``) into the view ``target``."""
        if weights is None:
            target[...] = array
        else:
            self.xp.multiply(array, weights, out=target)

    def _crop(self, crop: Any, weights: Any, out: Any, *, copy: bool) -> Any:
        """``crop * weights`` into ``out``; ``crop`` is copied if it is scratch space."""
        if weights is not None:
            return self.xp.multiply(crop, weights, out=out)
        if out is not None:
            out[...] = crop
            return out
        return crop.copy() if copy else crop

    def rfft2(self, array: Any, *, axes: tuple[int, int] = (-2, -1)) -> Any:
        """Unnormalized real-input 2-D FFT (used for convolutions)."""
        if self.is_gpu:
            return self.xp.fft.rfft2(array, axes=axes)
        from scipy import fft

        return fft.rfft2(array, axes=axes, workers=_cpu_workers(array.size))

    def irfft2(
        self, array: Any, shape: tuple[int, int], *, axes: tuple[int, int] = (-2, -1)
    ) -> Any:
        """Inverse of :meth:`rfft2` for an output of ``shape``."""
        if self.is_gpu:
            return self.xp.fft.irfft2(array, s=shape, axes=axes)
        from scipy import fft

        return fft.irfft2(array, s=shape, axes=axes, workers=_cpu_workers(array.size))

    def blas_limit(self, work: float) -> contextlib.AbstractContextManager[Any]:
        """Context limiting CPU BLAS threads for products of ``work`` multiply-adds.

        A no-op on the GPU. Entering and leaving costs a few microseconds.
        """
        controller = None if self.is_gpu else _threadpool_controller()
        if controller is None:
            return contextlib.nullcontext()
        return controller.limit(limits=_blas_threads(work), user_api="blas")

    # -------------------------------------------------------------- reductions
    def dot(self, a: Any, b: Any) -> float:
        """``Re(sum(conj(a) * b))`` over all elements, as a Python float.

        On the CPU this avoids BLAS level-1 calls: OpenBLAS threads ``ddot``
        above ~10k elements, and those threads stall (~1 ms per call) whenever
        the cores are busy. Use it for every scalar product in iterative loops.
        """
        a, b = a.reshape(-1), b.reshape(-1)
        if self.is_gpu:
            return float(self.xp.vdot(a, b).real)
        if a.dtype.kind == "c" or b.dtype.kind == "c":
            return float(np.einsum("i,i->", np.conj(a), b).real)
        return float(np.einsum("i,i->", a, b))

    # ---------------------------------------------------------- host boundary
    def to_numpy(self, array: Any) -> Any:
        """Copy an array to host NumPy (no-op on the CPU)."""
        return to_numpy(array)

    def scalar(self, value: Any) -> float:
        """Convert a 0-d array to a Python float (a device sync on GPU)."""
        return float(value)

    def synchronize(self) -> None:
        """Block until queued device work finishes (no-op on the CPU)."""
        if self.is_gpu:
            self.xp.cuda.get_current_stream().synchronize()

    def random(self, seed: Any) -> np.random.Generator:
        """A host :class:`numpy.random.Generator`.

        Random draws (initial guesses, noise) are made on the host and moved
        to the device so a seed gives the same numbers on CPU and GPU.
        """
        if isinstance(seed, np.random.Generator):
            return seed
        return np.random.default_rng(seed)


BackendLike = Union[Backend, str, None]  # noqa: UP007 (runtime alias on Python 3.10)


@functools.cache
def _make_backend(device: str, precision: str) -> Backend:
    return Backend(device=device, precision=precision)


def get_backend(device: BackendLike = "cpu", precision: str | None = None) -> Backend:
    """Return the :class:`Backend` for a device and precision.

    Parameters
    ----------
    device:
        ``"cpu"``, ``"gpu"`` (CuPy; raises if unavailable), ``"auto"`` (GPU when
        available, else CPU), or an existing :class:`Backend`, which is returned
        unchanged unless ``precision`` overrides it.
    precision:
        ``"double"`` or ``"single"``. Defaults to double on CPU, single on GPU.
    """
    if isinstance(device, Backend):
        if precision is None or precision == device.precision:
            return device
        return _make_backend(device.device, _check_precision(precision))
    name = "cpu" if device is None else str(device).lower()
    if name == "cuda":
        name = "gpu"
    if name == "auto":
        name = "gpu" if gpu_available() else "cpu"
    if name not in ("cpu", "gpu"):
        raise ValueError(f"device must be 'cpu', 'gpu' or 'auto', got {device!r}")
    if name == "gpu" and not gpu_available():
        raise RuntimeError(
            "device='gpu' needs CuPy and a CUDA device. Install the extra matching your "
            "driver (pip install 'aocore[cuda12]' or 'aocore[cuda13]'), or use "
            "device='auto' to fall back to the CPU."
        )
    if precision is None:
        precision = "single" if name == "gpu" else "double"
    return _make_backend(name, _check_precision(precision))


def _check_precision(precision: str) -> str:
    value = str(precision).lower()
    aliases = {"float32": "single", "float64": "double", "fp32": "single", "fp64": "double"}
    value = aliases.get(value, value)
    if value not in ("single", "double"):
        raise ValueError(f"precision must be 'single' or 'double', got {precision!r}")
    return value


def to_numpy(array: Any) -> Any:
    """Return ``array`` as a host NumPy array (copying from the GPU if needed).

    Non-array values (lists, scalars) go through :func:`numpy.asarray`.
    """
    cupy = _cupy()
    if cupy is not None and isinstance(array, cupy.ndarray):
        return cupy.asnumpy(array)
    return np.asarray(array)


def backend_of(array: Any, precision: str | None = None) -> Backend:
    """Infer a backend from an array (CuPy arrays give the GPU backend).

    Without ``precision``, single precision is chosen for float32/complex64
    input and double otherwise.
    """
    cupy = _cupy()
    device = "gpu" if cupy is not None and isinstance(array, cupy.ndarray) else "cpu"
    if precision is None:
        dtype = getattr(array, "dtype", None)
        single = dtype is not None and np.dtype(dtype) in (np.float32, np.complex64)
        precision = "single" if single else "double"
    return get_backend(device, precision)
