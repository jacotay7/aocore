# Conformance checks

Each package runs the checks that apply to it in its own test suite. A
violation raises `aocore.conformance.ConformanceError`, which subclasses
`AssertionError`, so pytest reports it as a failure. The message names the
rule and the measured deviation.

| Check | Rule | Run it in |
|---|---|---|
| `check_coordinates` | 1.2 pixel centres | every package that builds grids |
| `check_image_centring` | 1.3 optical axis at (n-1)/2 | makewfs, solvephase, shmpipeline-ao (PSF) |
| `check_tilt_direction` | 1.1, 3.1 tilt sign and axes | makewfs, solvephase, shmpipeline-ao |
| `check_unit_flux` | 3.3 normalization | makewfs, solvephase |
| `check_rms` | 4.1 RMS definition | every package that reports RMS |
| `check_zernike_basis` | 5.1, 5.2 Noll basis | aobasis, shmpipeline-ao |
| `check_wind_motion` | 6.1 frozen flow | pyturb |
| `check_slope_sign` | 7.1, 7.2 slopes | shmpipeline-ao, pyRTC, shmpipeline |

::: aocore.conformance
