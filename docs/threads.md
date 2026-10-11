# Threads and multiprocessing

A process that runs the AO stack can start five CPU thread pools:

| Pool | Used by | Default size | Set by |
|---|---|---|---|
| OpenBLAS / MKL / BLIS | NumPy, SciPy linear algebra | one per logical CPU | `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, threadpoolctl |
| OpenMP | compiled extensions, Numba's OpenMP layer | one per logical CPU | `OMP_NUM_THREADS`, threadpoolctl |
| SciPy FFT workers | every aocore FFT on the CPU | one per ~16k elements, up to the core count | `AOCORE_FFT_WORKERS` |
| aocore BLAS cap | `MFTPropagator` matrix products | 4-8 | `AOCORE_BLAS_THREADS` |
| Numba | pyturb's parallel loops | one per logical CPU | `NUMBA_NUM_THREADS`, `numba.set_num_threads` |

One simulation per machine should keep the defaults. A parameter sweep that
runs one simulation per process should not: 16 worker processes on 16 cores,
each starting 16 threads per pool, run hundreds of threads that mostly wait
for each other. In a closed-loop study this gave a load average of 170 and a
sweep several times slower than with one thread per worker.

## `aocore.set_threads`

`aocore.set_threads(n)` sets every pool at once:

- the BLAS and OpenMP libraries already loaded are limited with threadpoolctl;
- aocore's FFT workers and BLAS cap become `n` (they read `AOCORE_FFT_WORKERS`
  and `AOCORE_BLAS_THREADS` on every call);
- if Numba is already imported, `numba.set_num_threads(n)` is called (capped
  at the pool size Numba started with);
- the environment variables in `aocore.threads.THREAD_VARIABLES` are set to
  `n`, for libraries loaded later and for child processes.

Call it as a function to keep the setting for the rest of the process, or use
it as a context manager to restore the previous one:

```python
import aocore

aocore.set_threads(1)  # from here on, one thread per pool

with aocore.set_threads(4):  # four threads inside the block only
    run_simulation()
```

## Multiprocessing sweeps

Pin every worker with an initializer. It works with every start method
(`fork`, `spawn`, `forkserver`), because it limits the libraries that are
already loaded in the worker as well as those loaded later:

```python
from concurrent.futures import ProcessPoolExecutor

import aocore

with ProcessPoolExecutor(16, initializer=aocore.set_threads, initargs=(1,)) as pool:
    results = list(pool.map(run_one, configs))
```

The same goes for `multiprocessing.Pool(16, initializer=aocore.set_threads,
initargs=(1,))`. Choose `processes x threads` no larger than the number of
cores.

## Pitfalls

**Setting the variables after NumPy is imported does nothing.** OpenBLAS,
MKL and OpenMP read `*_NUM_THREADS` once, when they load, and NumPy loads
them on import. A script that runs `import numpy` (or imports any package that
does, aocore included) and then sets `os.environ["OPENBLAS_NUM_THREADS"] =
"1"` still runs 16 BLAS threads per process. A worker initializer that only
sets environment variables is too late as well: a forked worker inherits the
parent's loaded libraries, and a spawned one imports the initializer's module
(and NumPy with it) before running it. `set_threads` avoids this by limiting
the loaded libraries through threadpoolctl. If you set the variables yourself, do it
before the first import, at the very top of the entry point or in the shell:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
AOCORE_FFT_WORKERS=1 AOCORE_BLAS_THREADS=1 python sweep.py
```

**Numba's pool size is fixed when Numba starts its threads.** Importing Numba
with `NUMBA_NUM_THREADS=n` caps it at `n` for the life of the process, and
changing `NUMBA_NUM_THREADS` afterwards makes Numba's next compilation raise
`RuntimeError: Cannot set NUMBA_NUM_THREADS to a different value once the
threads have been launched`. So `set_threads` leaves that variable alone once
Numba is imported (it calls `numba.set_num_threads` instead), and does not put
it back on exit if Numba was imported inside the block.

**OpenMP and Numba limits are per thread.** Both apply to the thread that set
them. Call `set_threads` in the thread that does the work: in each worker
process, not only in the parent.

**Thread counts can change FFT results in the last bit.** SciPy partitions
FFT lines differently with 12 or more workers, so compare runs made with
different thread counts to rounding, not bit for bit. BLAS thread counts do
not change results.
