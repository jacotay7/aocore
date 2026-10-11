# aocore

[![CI](https://github.com/jacotay7/aocore/actions/workflows/ci.yml/badge.svg)](https://github.com/jacotay7/aocore/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Shared conventions, conformance checks and optics primitives for the
adaptive-optics simulation stack**: aobasis, pyturb, getframes, makewfs,
solvephase, shmpipeline-ao and pyRTC.

The packages of the stack hand each other OPD maps, images, wind vectors and
slopes. aocore makes sure they mean the same thing by them:

- **[CONVENTIONS.md](https://github.com/jacotay7/aocore/blob/main/CONVENTIONS.md)** is the contract: array axes, pixel
  centres, units, propagation sign and normalization, RMS and Strehl
  definitions, Zernike orientation, wind and slope conventions, and who owns
  which primitive.
- **`aocore.conformance`** holds executable checks of the contract that every
  package runs against its own implementation in its test suite.
- **The primitives themselves**, implemented once:
  - a NumPy/CuPy array backend;
  - anti-aliased and segmented pupils with telescope presets, and sub-pixel
    translated copies of them (`Pupil.shifted`) for pupil-shift and
    misregistration studies;
  - FFT, matrix-Fourier, focal-plane and angular-spectrum propagators with
    exact adjoints;
  - wavefront metrics (`rms`, `rms_unweighted`, `rms_tiptilt_removed`) and
    the measured Strehl ratio (`strehl_peak`);
  - perfect-coronagraph images and azimuthal profiles for contrast curves
    (`perfect_coronagraph_image`, `azimuthal_profile`);
  - least-squares phase unwrapping, fast pixel binning (`block_sum`,
    `block_mean`), periodic sub-pixel shifts (`fourier_shift`), coordinate
    grids built on either device, and unit constants;
  - `set_threads`, one switch for the BLAS, OpenMP, FFT and Numba thread
    pools, for multiprocessing sweeps.

```bash
pip install aocore              # CPU
pip install "aocore[cuda12]"    # + CuPy for CUDA 12.x
```

```python
import aocore as ac

pupil = ac.Pupil.vlt(128)
prop = ac.FocalPlanePropagator(pupil.shape, pupil.pitch, 1.6e-6, 1.6e-6 / 8 / 2, (64, 64))
print(ac.rms(opd, pupil))  # CONVENTIONS 4.1: weighted, piston removed
print(ac.rms_unweighted(opd, pupil.mask))  # plain quadratic mean, piston included
print(ac.rms_tiptilt_removed(opd, pupil))  # weighted, piston/tip/tilt removed
y, x = ac.coordinate_grid(pupil.shape, pupil.pitch, backend="auto", dtype="float32")

# Raw contrast behind a perfect coronagraph, and its azimuthal profile in lambda/D.
halo = ac.perfect_coronagraph_image(opd, pupil, prop)[0]
radii, contrast = ac.azimuthal_profile(halo, pixel_scale=1 / 2)  # 2 pixels per lambda/D

# In any package's tests:
from aocore import conformance

conformance.check_tilt_direction(
    my_image_from_opd, pupil_shape=(64, 64), pitch=..., pixel_scale=...
)
```

aocore depends only on NumPy, SciPy and threadpoolctl. CuPy is optional.

## License

MIT. Every package in the stack except pyRTC (GPL-3.0-or-later) is MIT, and none
of them may copy GPL code (see CONVENTIONS 8.4).
