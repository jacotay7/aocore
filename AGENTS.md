# AGENTS.md

Guide for agents (and humans) working in `aocore`. Keep it accurate. Update it
in the same change that alters layout, commands or conventions, and add a
Gotchas entry when something led you astray.

## What this repository is

aocore is the foundation of the AO simulation stack. It holds:

- `CONVENTIONS.md`, the cross-package contract;
- `aocore.conformance`, the executable checks of that contract;
- the generic optics primitives that the other packages import instead of
  re-implementing (backend, pupils, propagators, metrics, unwrapping,
  binning, constants).

It must stay small, fast to import, and free of domain physics. Atmosphere
belongs to pyturb, detectors to getframes, wavefront sensors to makewfs,
bases to aobasis, phase retrieval to solvephase, and control to
shmpipeline-ao and pyRTC.

## Layout

```text
CONVENTIONS.md            the contract (versioned; see "Changing a convention")
src/aocore/conventions.py reference definitions (coordinates, units, RMS)
src/aocore/conformance.py check_* functions run by every package
src/aocore/backend.py     Backend (NumPy | CuPy, single | double)
src/aocore/propagation.py FFT/MFT/focal-plane/angular-spectrum propagators (exact adjoints)
src/aocore/pupil.py       Pupil: circular, segmented, presets
src/aocore/metrics.py     rms (+ rms_unweighted, rms_tiptilt_removed), remove_modes, wavefront_error, strehl
src/aocore/unwrap.py      weighted least-squares phase unwrapping
src/aocore/sampling.py    block_sum, block_mean
tests/                    pytest, including conformance self-checks
benchmarks/               timing scripts (python benchmarks/bench_block_sum.py [--gpu],
                          bench_propagation.py [--gpu]); results/ holds recorded runs
```

## Quality gate

```bash
ruff check . && ruff format --check .
python -m mypy
python -m pytest -q --cov=aocore
python -m pytest -q --run-gpu        # needs CuPy + a CUDA device
mkdocs build --strict
python -m build
```

## Changing a convention

1. Open a PR here that edits `CONVENTIONS.md`, the reference implementation
   and the conformance check together.
2. Bump the contract version (the "Status" line).
3. Name the downstream packages that must follow, and open their PRs.

Never loosen a check to make a package pass. Fix the package instead.

## Rules

- Every operator ships an exact adjoint and an adjoint test. Every check
  ships a test that it passes the reference and fails a realistic violation.
- Numerical code uses `Backend` (`backend.xp`, `Backend.fft2`, `Backend.dot`).
  Host transfers happen only at named boundaries.
- MIT-compatible code only; see CONVENTIONS 8.4.
- No planning or status files in the repo.

## Gotchas

- `Backend.dot` avoids threaded OpenBLAS level-1 calls, which stall about
  1 ms per call when the cores are busy. `Backend.blas_limit` caps BLAS
  threads for mid-sized matrix products, where 16 threads run 2-4x slower
  than 4 on a hyper-threaded x86. Without SMT (Arm servers) 8 threads are 2x
  faster than 4, so `_blas_threads` checks `/sys/devices/system/cpu/smt/active`.
  BLAS thread counts never change results; FFT thread counts can (below).
- `Backend.padded_fft2` must reproduce `fft2` value for value. SciPy's
  out-of-place `fft2` runs the last axis first with ducc (SciPy >= 1.18) but
  the first axis first with pocketfft (<= 1.17), so the pass order is probed
  once (`_fft2_first_axis`). The `ortho` factor is applied in the first pass.
  Lines are transformed in bunches of 16 and a remainder group can round
  differently, hence the 16-line padding; and SciPy's results already change
  in the last bit with the thread count (12+ threads partition lines
  differently), so never compare thread counts bit for bit. Check changes
  with a saved before/after matrix of outputs, not only the tests.
- Writing a fresh multi-megabyte NumPy array costs a page fault per 4 KiB
  (glibc hands freed large blocks back to the kernel), which took as long as
  the FFT itself on the Arm bench. Hot CPU paths reuse per-thread work arrays
  from `backend._workspace`; never return one of them to a caller.
- CuPy's `matmul` costs ~130 us more per call when it must broadcast an
  operand's batch axes. `MFTPropagator` expands its matrices to the field's
  batch shape (cached per shape) on the GPU; the results are bit-identical.
- `offset=-0.5` reproduces HCIPy's `fftshift` image centring (CONVENTIONS
  1.3).
- Binning speed: on NumPy, one `sum(axis=(-3, -1))` over a reshaped array is
  3-15x slower than strided adds per axis; on CuPy, a sum over the
  non-contiguous axis -2 is 10-50x slower than the two-axis sum, and a
  one-thread-per-block kernel beats both. `block_sum` takes the fast path
  for each; re-run `benchmarks/bench_block_sum.py --gpu` before changing it.
- `conformance.check_point_source_centring` needs an even window: for odd
  `n` the `fftshift` axis `n // 2` equals `(n - 1) / 2`.
