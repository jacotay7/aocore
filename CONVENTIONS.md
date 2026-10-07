# AO stack conventions

This is the contract shared by the adaptive-optics packages: aocore, aobasis,
pyturb, getframes, makewfs, solvephase, shmpipeline-ao, pyRTC and shmpipeline.
Each rule has a reference implementation in `aocore` and an executable check in
`aocore.conformance`, which every package runs in its own test suite. A rule
changes only through a pull request to aocore that also updates the checks.
The packages that consume it then follow.

Status: **adopted** (version 1.1).

## 1. Arrays and coordinates

1.1 **Index order.** 2-D arrays are indexed `[y, x]`: **x is axis 1
(columns), y is axis 0 (rows)**. Every x/y-labelled argument, return value
and docstring uses this frame. Examples are wind vectors, off-axis directions,
tip/tilt, slopes and actuator positions.

1.2 **Pixel centres.** Pixel `i` of an `n`-pixel axis sits at
`(i - (n - 1) / 2) * pitch`, so the grid is symmetric about the optical axis.
For even `n` the axis falls *between* the two central pixels.

1.3 **Image-plane centring.** The same rule holds in every image plane
(focal plane, Shack-Hartmann subaperture, pyramid detector). On a flat
wavefront the optical axis is at pixel coordinate `(n - 1) / 2`. A package
that interoperates with the `fftshift` convention (axis on pixel `n / 2`, as
in HCIPy or `numpy.fft.fftshift`) converts explicitly at its boundary, for
example with `offset = -0.5` pixel, and says so in its docs.

1.4 **Pupil sampling.** A pupil of diameter `D` sampled by `n` pixels across
its full width has pitch `D / n`, with pixel centres as in 1.2. Actuator
grids are a separate geometry: their pitch is whatever the DM defines.

1.5 **Angles.** Angles in the plane are measured counter-clockwise **from +x
toward +y**. They are in degrees in configuration and user-facing APIs, and
in radians internally.

## 2. Physical quantities and units

2.1 **SI units.** Lengths are in metres, including wavelengths, OPD, pupil
diameters and pixel pitches. Times are in seconds and angles on the sky in
radians. Arcseconds and nanometres appear only at user-facing boundaries
whose names say so (`*_arcsec`, `*_nm`).

2.2 **The wavefront is OPD in metres.** Phase is in radians and always
carries its wavelength: `phase = 2 pi OPD / lambda`.

2.3 **Constants.** Use `aocore.ARCSEC_TO_RAD` (`pi / 648000`) and
`aocore.RAD_TO_ARCSEC` rather than a local literal.

2.4 **Reflective surfaces.** A mirror surface displacement `s` produces OPD
`2 s` at normal incidence. An API that takes a surface says so (`surface_m`);
otherwise every quantity is OPD.

## 3. Propagation and flux

3.1 **Sign.** The Fraunhofer kernel is `exp(-2 pi i x . alpha / lambda)`.
An OPD ramp `a x` (metres per metre) moves the image to angle `+a`, towards
larger column index.

3.2 **Unitary transforms.** Discrete Fourier transforms between pupil and
image planes are unitary (`norm="ortho"`), so the total intensity in a window
that captures all the light equals the pupil's `sum |u|^2`.

3.3 **No hidden renormalization.** A finite detector window or field stop
loses light, and the loss is reported rather than renormalized away. A
normalized image ("PSF") sums to 1 for an unbounded detector.

3.4 **Photon budgets use the clear aperture.** Photons collected from a source
of given brightness scale with the transmitted area `sum |pupil|^2 pitch^2`.
Spiders, gaps and masks remove light.

## 4. Wavefront metrics

4.1 **RMS.** `rms(opd, pupil)` is weighted by pupil intensity, with piston
removed:
`sqrt(sum a^2 (opd - <opd>_a)^2 / sum a^2)`, where `<opd>_a` is the
intensity-weighted mean (`aocore.rms`). Variants say so in their name:
`aocore.rms_unweighted(opd, mask)` is the plain quadratic mean over the mask
with piston *included*, and `aocore.rms_tiptilt_removed(opd, pupil)` is
intensity-weighted with piston, tip and tilt removed by weighted least
squares.

4.2 **Strehl.** `strehl_marechal(rms, lambda) = exp(-(2 pi rms / lambda)^2)`
is the Maréchal approximation. A measured Strehl compares an image peak with
the peak of the unaberrated image under the same sampling and normalization,
and is named `strehl_peak`.

## 5. Modal bases

5.1 **Zernike ordering and normalization.** Zernikes use Noll ordering and
indexing (j = 1 is piston) with Noll normalization: unit RMS over the
continuous unit disc, or over the annulus for annular modes. Coefficients are
RMS OPD in metres.

5.2 **Orientation.** Noll j = 2 (tip) is `2 rho cos(theta)`, which varies
along **+x (columns)**. j = 3 (tilt) varies along +y. theta is measured as in
1.5.

5.3 **One implementation.** Bases are evaluated by aobasis. Other packages
call it rather than carry their own polynomials.

## 6. Atmosphere

6.1 **Wind.** `wind_direction` is the direction the wind blows **from**, in
degrees, measured as in 1.5. `wind_vector = (vx, vy)` points the same way.
The frozen-flow pattern moves along `-wind_vector`.

6.2 **Off-axis directions.** Direction tuples are `(theta_x, theta_y)`
along (x, y).

## 7. Wavefront sensing and control

7.1 **Slope layout.** Subapertures are ordered row-major over
`(subap_y, subap_x)`. Two layouts of slope vectors are in use, and every slope
vector, stream or calibration artifact declares which one it uses:

- **interleaved**, `[sx_1, sy_1, sx_2, sy_2, ...]` (shmpipeline-ao ADR 0002);
- **blocked**, `[sx_1 ... sx_N, sy_1 ... sy_N]` (pyRTC).

Each component uses the x/y of 1.1. Conversions between layouts are explicit.

7.2 **Slope sign.** A positive x-slope means the spot moved towards +x, which
is a positive OPD gradient along x (3.1).

## 8. Software

8.1 **Devices.** `device` accepts `"cpu"`, `"gpu"` (or `"gpu:N"`) and
`"auto"`. GPU results stay on the device. `to_numpy` is the explicit host
boundary.

8.2 **Precision.** `precision` accepts `"single"` and `"double"`, with
`"float32"` and `"float64"` as aliases.

8.3 **Randomness.** Randomness comes from an explicit seed or
`numpy.random.Generator`, never from global state.

8.4 **Licensing.** Every package is MIT-licensed except pyRTC, which is
GPL-3.0-or-later. pyRTC may import the MIT packages. The MIT packages never
depend on pyRTC and never copy code from it, or from any other GPL, LGPL,
non-commercial or CeCILL source. Never add a patent-encumbered algorithm
without the maintainer's explicit decision.

## 9. Ownership

| Primitive | Owner | Consumers must |
|---|---|---|
| Coordinates, constants, pupils, Fraunhofer/Fresnel propagators, OPD/phase helpers, RMS/Strehl, block-sum binning, array backend | **aocore** | import, not re-implement |
| Modal bases: Zernike, KL, Fourier, Hadamard, DM influence functions, M2C | **aobasis** | import |
| Atmosphere: phase screens, Cn² profiles, frozen flow, turbulence theory | **pyturb** | import |
| Detectors: QE, noise, gain, saturation, presets, radiometry of sources | **getframes** | import |
| Wavefront-sensor image formation (Shack-Hartmann, pyramid) | **makewfs** | import |
| Phase retrieval and focal-plane wavefront sensing | **solvephase** | import |
| Reconstruction, control, DM command models, AO timing | **shmpipeline-ao**, **pyRTC** | import |
| Shared-memory transport | **pyshmem** | import |

A bug in an owned primitive is fixed in its owner, never worked around
downstream.
