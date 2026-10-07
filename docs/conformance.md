# Conformance checks

Each package runs the checks that apply to it in its own test suite. A
violation raises `aocore.conformance.ConformanceError`, which subclasses
`AssertionError`, so pytest reports it as a failure. The message names the
rule and the measured deviation.

Packages that form images from an OPD (Fourier optics) pass an
`image_from_opd(opd)` callable. Packages that render analytic point sources
(Gaussian, Moffat, Airy profiles) use the `render` variants instead:
`render(shape)` for an on-axis source, and `render(shape, position)` for
`check_edge_flux_loss`, with `position = (y, x)` in pixels from the window
centre.

| Check | Rule | Run it in |
|---|---|---|
| `check_coordinates` | 1.2 pixel centres | every package that builds grids |
| `check_image_centring` | 1.3 optical axis at (n-1)/2 | makewfs, solvephase, shmpipeline-ao (PSF) |
| `check_point_source_centring` | 1.3, for analytic (rendered) images | getframes |
| `check_tilt_direction` | 1.1, 3.1 tilt sign and axes | makewfs, solvephase, shmpipeline-ao |
| `check_unit_flux` | 3.3 normalization | makewfs, solvephase |
| `check_point_source_flux` | 3.3 normalization, for analytic images | getframes |
| `check_edge_flux_loss` | 3.3 no renormalization of clipped light | getframes, makewfs |
| `check_rms` | 4.1 RMS definition | every package that reports RMS |
| `check_zernike_basis` | 5.1, 5.2 Noll basis | aobasis, shmpipeline-ao |
| `check_wind_motion` | 6.1 frozen flow | pyturb |
| `check_slope_sign` | 7.1, 7.2 slopes | shmpipeline-ao, pyRTC, shmpipeline |

::: aocore.conformance
