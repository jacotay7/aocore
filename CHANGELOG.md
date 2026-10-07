# Changelog

All notable changes to `aocore` are documented here.

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
