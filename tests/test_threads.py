"""aocore.set_threads: one switch for BLAS, OpenMP, aocore and Numba threads."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from threadpoolctl import threadpool_info

import aocore as ac
from aocore.backend import _blas_threads, _cpu_workers
from aocore.threads import THREAD_VARIABLES


@pytest.fixture
def clean_env(monkeypatch):
    for name in THREAD_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    return monkeypatch


def _blas_counts() -> set[int]:
    np.dot(np.ones((4, 4)), np.ones((4, 4)))  # make sure NumPy's BLAS is loaded
    return {info["num_threads"] for info in threadpool_info() if info["user_api"] == "blas"}


def test_context_manager_sets_and_restores_everything(clean_env) -> None:
    before = _blas_counts()
    with ac.set_threads(2) as limits:
        assert repr(limits) == "ThreadLimits(n=2)"
        assert all(os.environ[name] == "2" for name in THREAD_VARIABLES)
        assert _cpu_workers(1 << 30) == 2 and _blas_threads(1e12) == 2
        assert _blas_counts() <= {2}
    assert os.environ["OMP_NUM_THREADS"] == "7"
    assert all(name not in os.environ for name in THREAD_VARIABLES if name != "OMP_NUM_THREADS")
    assert _blas_counts() == before


def test_function_form_lasts_until_restored(clean_env) -> None:
    limits = ac.set_threads(1)
    assert os.environ["AOCORE_FFT_WORKERS"] == "1" and _cpu_workers(1 << 30) == 1
    assert _blas_counts() <= {1}
    limits.restore()
    limits.restore()  # idempotent
    assert "AOCORE_FFT_WORKERS" not in os.environ and os.environ["OMP_NUM_THREADS"] == "7"


@pytest.mark.parametrize("bad", [0, -2, 1.5, True, "4"])
def test_rejects_bad_counts(bad) -> None:
    with pytest.raises(ValueError, match="positive int"):
        ac.set_threads(bad)


def _run(code: str) -> str:
    env = {k: v for k, v in os.environ.items() if k not in THREAD_VARIABLES}
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


PARALLEL = """
import numba
import numpy as np

def make():
    @numba.njit(parallel=True)
    def total(a):
        s = 0.0
        for i in numba.prange(a.size):
            s += a[i]
        return s
    return total
"""


@pytest.mark.skipif(importlib.util.find_spec("numba") is None, reason="needs Numba")
def test_numba_already_running_is_limited_without_breaking_compilation() -> None:
    # Changing NUMBA_NUM_THREADS after Numba launched its threads makes every
    # later compilation raise; set_threads must use numba.set_num_threads instead.
    out = _run(
        PARALLEL
        + """
import os
import aocore
make()(np.ones(100))
pool = numba.config.NUMBA_NUM_THREADS
with aocore.set_threads(1):
    assert numba.get_num_threads() == 1
    assert "NUMBA_NUM_THREADS" not in os.environ
    make()(np.ones(100))
assert numba.get_num_threads() == pool
make()(np.ones(100))
print("ok")
"""
    )
    assert out == "ok"


@pytest.mark.skipif(importlib.util.find_spec("numba") is None, reason="needs Numba")
def test_numba_imported_later_starts_with_n_threads() -> None:
    out = _run(
        """
import os
import aocore
limits = aocore.set_threads(2)
"""
        + PARALLEL
        + """
make()(np.ones(100))
assert numba.config.NUMBA_NUM_THREADS == 2
limits.restore()
assert os.environ["NUMBA_NUM_THREADS"] == "2"  # kept: Numba's pool has that size now
make()(np.ones(100))
print("ok")
"""
    )
    assert out == "ok"


def test_worker_initializer_pins_spawned_processes() -> None:
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    context = multiprocessing.get_context("spawn")
    pin = {"initializer": ac.set_threads, "initargs": (1,)}
    with ProcessPoolExecutor(1, mp_context=context, **pin) as pool:
        assert pool.submit(os.getenv, "AOCORE_BLAS_THREADS").result() == "1"
        info = pool.submit(threadpool_info).result()
    # NumPy (and its BLAS) was loaded before the initializer ran, and is limited anyway.
    assert {entry["num_threads"] for entry in info} <= {1}
