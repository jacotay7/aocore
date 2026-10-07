# Changelog

All notable changes to `aocore` are documented here.

## [Unreleased]

- **Fixed: the backend threading test failed when `AOCORE_FFT_WORKERS` or
  `AOCORE_BLAS_THREADS` was set in the environment**, as on a benchmark host
  that pins thread counts. It now clears those overrides itself.

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
