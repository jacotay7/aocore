"""aocore: shared conventions and optics primitives for the AO simulation stack.

The contract is CONVENTIONS.md; this package implements it once so the other
packages can import rather than re-implement it:

* :mod:`aocore.conventions` - reference definitions (coordinates, units, RMS);
* :mod:`aocore.conformance` - executable checks every package runs;
* :mod:`aocore.backend` - NumPy/CuPy backend with precision handling;
* :mod:`aocore.pupil` - anti-aliased, segmented and preset pupils;
* :mod:`aocore.propagation` - FFT, MFT, focal-plane and angular-spectrum
  propagators with exact adjoints;
* :mod:`aocore.metrics`, :mod:`aocore.unwrap`, :mod:`aocore.sampling` -
  wavefront metrics, phase unwrapping, pixel binning and sub-pixel shifts;
* :mod:`aocore.contrast` - perfect-coronagraph images and contrast curves;
* :mod:`aocore.threads` - one switch for every CPU thread pool.
"""

from . import conformance, conventions
from .__about__ import __version__
from .backend import Backend, BackendLike, backend_of, get_backend, gpu_available, to_numpy
from .contrast import azimuthal_profile, perfect_coronagraph_field, perfect_coronagraph_image
from .conventions import (
    ARCSEC_TO_RAD,
    RAD_TO_ARCSEC,
    centered_coordinates,
    centroid,
    coordinate_grid,
    opd_to_phase,
    phase_to_opd,
    strehl_marechal,
    strehl_peak,
)
from .metrics import (
    remove_modes,
    rms,
    rms_tiptilt_removed,
    rms_unweighted,
    strehl_from_rms,
    wavefront_error,
)
from .propagation import (
    AngularSpectrumPropagator,
    FFTPropagator,
    FocalPlanePropagator,
    MFTPropagator,
    Propagator,
)
from .pupil import Pupil
from .sampling import block_mean, block_sum, fourier_shift
from .threads import ThreadLimits, set_threads
from .unwrap import unwrap_phase, wrap

__all__ = [
    "ARCSEC_TO_RAD",
    "RAD_TO_ARCSEC",
    "AngularSpectrumPropagator",
    "Backend",
    "BackendLike",
    "FFTPropagator",
    "FocalPlanePropagator",
    "MFTPropagator",
    "Propagator",
    "Pupil",
    "ThreadLimits",
    "__version__",
    "azimuthal_profile",
    "backend_of",
    "block_mean",
    "block_sum",
    "centered_coordinates",
    "centroid",
    "conformance",
    "conventions",
    "coordinate_grid",
    "fourier_shift",
    "get_backend",
    "gpu_available",
    "opd_to_phase",
    "perfect_coronagraph_field",
    "perfect_coronagraph_image",
    "phase_to_opd",
    "remove_modes",
    "rms",
    "rms_tiptilt_removed",
    "rms_unweighted",
    "set_threads",
    "strehl_from_rms",
    "strehl_marechal",
    "strehl_peak",
    "to_numpy",
    "unwrap_phase",
    "wavefront_error",
    "wrap",
]
