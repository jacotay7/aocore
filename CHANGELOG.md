# Changelog

All notable changes to `aocore` are documented here.

## [Unreleased]

## [0.1.4] - 2026-10-07

Faster propagators on CPU and GPU; results are unchanged up to last-bit
rounding.
Additive release; the contract stays at v1.1. Measured on an 80-core Arm
Neoverse-N1 pinned to 12 cores and an RTX 4060; see
`benchmarks/results/propagation-neoverse-n1.md` and
`benchmarks/bench_propagation.py`.

- **Faster `FFTPropagator` (and the FFT engine of `FocalPlanePropagator`) on
  the CPU: 2.4-3.0x for 64-256 pixel pupils, 1.6x for 512.** The padded
  transform skips the all-zero rows of the padded pupil and the columns the
  output window crops away (about half the FFT work), and reuses its work
  arrays between calls instead of faulting in fresh pages. In solvephase
  0.2.0 this makes `focal_lm` 1.2-1.4x, phase diversity up to 1.6x and LIFT
  up to 1.4x faster on the CPU. The values come from the same 1-D passes in
  the same order as before, so they are bit-identical for power-of-two grids
  up to 256 points; elsewhere they differ by at most 2.5 units in the last
  place of the largest value, as much as 0.1.3's own results change with
  the FFT thread count.
- **New: `Backend.padded_fft2(array, shape, out_shape, *, inverse, weights,
  out_weights, out)`**, the unitary FFT of a block zero-padded to `shape`,
  cropped to `out_shape`, which the propagators now use. It follows the pass
  order of the installed SciPy's `fft2` (pocketfft and ducc differ). On the
  GPU it is the full transform and crop, as before.
- **Faster `MFTPropagator` on the GPU: 3.5-4.3x for 128-pixel pupils, 1.4x
  for 256**, bit-identical. CuPy's `matmul` is slow when it broadcasts the
  matrices over leading field axes (diversity channels); the propagator now
  expands them to the field's batch shape once and keeps them (up to 32 MiB
  per shape, four shapes). Broadband focal-plane gradients in solvephase are
  1.3-1.4x faster on the GPU.
- **More BLAS threads on CPUs without simultaneous multithreading.** When
  Linux reports SMT inactive (Arm servers), matrix products above ~4M
  multiply-adds use 8 threads instead of 4, capped by the CPUs the process
  may run on: the CPU MFT is 1.8x faster for 128-pixel pupils. Hyper-threaded
  machines keep the 0.1.3 counts. The thread count does not change the
  results.
- **`FocalPlanePropagator` writes each wavelength's FFT result straight into
  one output array** instead of stacking copies (1.2x on the GPU at these
  sizes).
- **`Backend.dot` on the GPU reduces the elementwise product directly**,
  which rounds exactly as `cupy.vdot` (checked for all four dtypes) with
  ~20 us less overhead per call.
- `os.cpu_count()` is read once (it cost ~25 us per FFT on a busy Arm host).
- **Added: `benchmarks/results/block_sum-neoverse-n1.md`**, `block_sum` timings
  on an Arm Neoverse-N1 host with an RTX 4060 and an RTX A400.
- **Fixed: the backend threading test failed when `AOCORE_FFT_WORKERS` or
  `AOCORE_BLAS_THREADS` was set in the environment**, as on a benchmark host
  that pins thread counts. It now clears those overrides itself.
- Not adopted: pinned host staging buffers for `Backend.asarray` (10 us
  faster below 64 KiB, slower above 512 KiB on this host), threaded DCTs in
  `unwrap_phase` (2.5-3x faster preconditioner, but the thread partition
  changes the rounding and therefore the CG iterates), and pruned FFTs on
  the GPU (a CuPy 2-D FFT at these sizes costs mostly call overhead, which
  two 1-D passes would double).

## [0.1.3] - 2026-10-07

Additive release; the contract stays at v1.1. Gaps found while moving the
sibling packages onto aocore.

- **Fixed: CONVENTIONS 8.4 said every package is MIT-licensed.** pyRTC is
  GPL-3.0-or-later. The rule now says so, and states the direction that keeps
  the MIT packages clean: pyRTC may import them, but they never depend on
  pyRTC or copy code from it. No rule for code changes, so the contract stays
  at v1.1.
- **New: `rms_unweighted(opd, mask=None)` and `rms_tiptilt_removed(opd,
  pupil)`**, the RMS variants CONVENTIONS 4.1 names. `rms_unweighted` is the
  plain quadratic mean over the mask (or the whole array) with piston
  included, as makewfs reports it. `rms_tiptilt_removed` is
  intensity-weighted with piston, tip and tilt removed by weighted least
  squares (equal to `rms(opd, pupil, "tiptilt")`). Both run on NumPy or CuPy
  arrays without copying the OPD to the host, and ignore NaN outside the
  pupil. Rule 4.1 now names them.
- **Faster `block_sum`.** NumPy now reduces each axis separately, with
  strided adds for factors up to 4; CuPy runs one kernel with a thread per
  output pixel. It is 3-17x faster than 0.1.2 on NumPy and 1.4-4.7x on CuPy,
  and matches or beats the factor-2 fast path makewfs kept. Integer sums are
  exact and keep the dtype `sum` gives; float sums may differ in the last
  bits because the addition order changed. See
  `benchmarks/bench_block_sum.py`.
- **New: `block_mean(array, factor)`**, the block average. `Pupil.downsampled`
  uses it.
- **New: `dtype=` and `backend=` on `centered_coordinates` and
  `coordinate_grid`**, so callers can build coordinates on the GPU in their
  working precision. Without them the result is host float64 as before.
- **New conformance checks for rendered images:**
  - `check_point_source_centring(render)` and
    `check_point_source_flux(render)` take `render(shape) -> image`, for
    packages whose images are analytic (getframes) rather than OPD-driven.
    The OPD-based `check_image_centring` and `check_unit_flux` are
    unchanged.
  - `check_edge_flux_loss(render)` (rule 3.3) takes `render(shape,
    position)`. It checks that a source centred on a window edge deposits
    exactly the visible part of its light, compared with a larger window,
    and about half of what a central source deposits. It catches PSF
    renderers that renormalize clipped stamps or wrap light around the
    window.

## [0.1.2] - 2026-10-07

- **Changed: CONVENTIONS 7.1 (contract v1.1) recognizes both slope layouts in
  use.** shmpipeline-ao interleaves `(sx, sy)` per subaperture (its ADR 0002)
  and pyRTC concatenates all x-slopes, then all y-slopes. Slope vectors must
  now declare their layout, and `check_slope_sign(..., layout="interleaved" |
  "blocked")` checks against the declared one. A swapped layout or a flipped
  sign still fails.

## [0.1.1] - 2026-10-06

- **Fixed: `check_wind_motion` could report no motion for a correctly moving
  screen.** It located the shift with an FFT cross-correlation. Turbulence is
  red and the frames are not periodic, so the peak was dominated by the
  largest scales and could sit at zero shift (seen with pyturb, wind from 90
  degrees). The check now searches integer shifts by least squares over the
  frames' interior, and its test uses red, non-periodic screens.

## [0.1.0] - 2026-10-06

First release.

- `CONVENTIONS.md` (contract version 1) covers:
  - array axes and pixel centres;
  - SI units;
  - propagation sign and unitary normalization;
  - RMS and Strehl definitions;
  - Zernike ordering, normalization and orientation;
  - wind, direction and slope conventions;
  - device and precision vocabulary;
  - MIT-only licensing;
  - ownership of every shared primitive.
- `aocore.conformance` provides executable checks: coordinates, image
  centring, tilt direction, unit flux, RMS definition, Zernike basis, wind
  motion and slope sign. Each is tested against the reference and against
  the violations found in the stack.
- Primitives, moved from solvephase:
  - `Backend` (NumPy/CuPy, single/double, contention-safe CPU threading);
  - `Pupil` (anti-aliased circular, segmented hexagonal, Keck/JWST/VLT/HST
    presets);
  - FFT, MFT, focal-plane and angular-spectrum propagators with exact
    adjoints;
  - `rms`, `remove_modes`, `wavefront_error`, `strehl_from_rms`;
  - `unwrap_phase`.
- New `block_sum` and the unit constants `ARCSEC_TO_RAD` and `RAD_TO_ARCSEC`.
