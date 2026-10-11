"""High-contrast imaging metrics: the perfect coronagraph and azimuthal profiles.

A *perfect coronagraph* (Cavarroc et al. 2006, A&A 447, 397) removes exactly
the light an unaberrated wavefront would put in the image: the coherent part of
the pupil field, its projection on the pupil amplitude. What is left, imaged,
is the residual halo that sets the raw contrast an AO system can reach with
any coronagraph, which makes it the standard yardstick of closed-loop
simulations. :func:`azimuthal_profile` turns the image into a contrast curve.

Images follow the focal-plane conventions of the propagators: unitary
transforms (CONVENTIONS 3.2), the optical axis at pixel ``(n - 1) / 2``
(1.3), and normalizations stated explicitly (3.3).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .backend import BackendLike, backend_of, get_backend
from .propagation import FocalPlanePropagator, Propagator
from .pupil import Pupil

__all__ = ["azimuthal_profile", "perfect_coronagraph_field", "perfect_coronagraph_image"]


def _amplitude(pupil: Pupil | Any) -> Any:
    return pupil.amplitude if isinstance(pupil, Pupil) else pupil


def perfect_coronagraph_field(
    field: Any, pupil: Pupil | Any, *, backend: BackendLike = None
) -> Any:
    """Pupil field with its coherent part removed, frame by frame.

    Returns ``E - c A`` with ``c = sum(A E) / sum(A^2)``: ``E`` minus its
    projection on the (real) pupil amplitude ``A``, which for a uniform
    pupil is the pupil-mean field. An unaberrated field ``E = A`` gives
    exactly zero. The operator is an orthogonal projection, so it is its own
    adjoint.

    Parameters
    ----------
    field:
        ``(..., ny, nx)`` complex pupil-plane field(s) including the pupil
        amplitude (for example ``A exp(2 pi i OPD / lambda)``), NumPy or
        CuPy. Each 2-D slice (frame, wavelength) is treated on its own.
    pupil:
        A :class:`~aocore.Pupil` or a ``(ny, nx)`` amplitude array.
    backend:
        Where to compute; default the field's device.
    """
    be = backend_of(field) if backend is None else get_backend(backend)
    xp = be.xp
    data = be.asarray(field, dtype="complex")
    amp = be.asarray(_amplitude(pupil), dtype="real")
    if data.ndim < 2 or data.shape[-2:] != amp.shape:
        raise ValueError(f"field {data.shape} does not end in the pupil shape {amp.shape}")
    weight = float(be.to_numpy(xp.sum(amp * amp)))
    if not weight > 0:
        raise ValueError("pupil has no transmission")
    coherent = xp.sum(data * amp, axis=(-2, -1), keepdims=True) / weight
    return data - coherent * amp


def perfect_coronagraph_image(
    field_or_opd: Any,
    pupil: Pupil | Any,
    propagator: Propagator,
    *,
    wavelength: Any = None,
    normalize: str | None = "peak",
) -> Any:
    """Image behind a perfect coronagraph (Cavarroc et al. 2006).

    Removes the coherent part of each pupil field with
    :func:`perfect_coronagraph_field` and returns
    ``|propagator(E - c A)|^2``, normalized as asked. A flat wavefront gives
    an image of zero; an aberrated one gives its incoherent halo.

    Parameters
    ----------
    field_or_opd:
        Either complex pupil field(s) ``(..., ny, nx)`` including the pupil
        amplitude, laid out as ``propagator`` expects (a
        :class:`~aocore.FocalPlanePropagator` wants ``(..., L, ny, nx)``), or
        a real OPD map(s) ``(..., ny, nx)`` in metres, which becomes
        ``A exp(2 pi i OPD / lambda)`` (with an ``L`` wavelength axis added
        before the last two for a :class:`~aocore.FocalPlanePropagator`).
    pupil:
        A :class:`~aocore.Pupil` or a ``(ny, nx)`` amplitude array, on the
        propagator's input grid.
    propagator:
        Pupil-to-image propagator (:class:`~aocore.FocalPlanePropagator`,
        :class:`~aocore.FFTPropagator` or :class:`~aocore.MFTPropagator`). It
        sets the sampling, the window and the device.
    wavelength:
        Wavelength in metres for an OPD input, required unless the
        propagator is a :class:`~aocore.FocalPlanePropagator`, whose own
        wavelengths are used (a value given must match them). Ignored for a
        complex field.
    normalize:
        * ``"peak"`` (default): divided by the peak of the unaberrated image
          without coronagraph (``|propagator(A)|^2``, per wavelength), so
          pixel values are raw contrast. The window must contain that peak.
        * ``"flux"``: divided by ``sum(A^2)``, the total of the unaberrated
          image on an unbounded detector, so the image is in units of the
          normalized PSF (CONVENTIONS 3.3).
        * ``None``: the raw ``|E|^2`` of the unitary propagator, in the
          units of ``|field|^2``.

    Returns
    -------
    The real image(s), on the propagator's device, with the propagator's
    output layout (one image per frame and wavelength; sum over the
    wavelength axis for a broadband image).
    """
    if normalize not in ("peak", "flux", None):
        raise ValueError(f"normalize must be 'peak', 'flux' or None, got {normalize!r}")
    be = propagator.backend
    xp = be.xp
    amp = be.asarray(_amplitude(pupil), dtype="real")
    if tuple(amp.shape) != tuple(propagator.in_shape):
        raise ValueError(
            f"pupil {amp.shape} does not match the propagator input {propagator.in_shape}"
        )
    # A FocalPlanePropagator takes (..., L, ny, nx): one slice per wavelength.
    stack = propagator.wavelengths if isinstance(propagator, FocalPlanePropagator) else None
    data = be.asarray(field_or_opd)
    if data.dtype.kind != "c":
        if stack is not None:
            if wavelength is not None and not np.allclose(np.ravel(wavelength), stack):
                raise ValueError("wavelength differs from the propagator's wavelengths")
            per_slice = be.asarray(2 * np.pi / stack, "real")[:, None, None]
            phase = data[..., None, :, :] * per_slice
        elif wavelength is None or np.size(wavelength) != 1:
            raise ValueError("an OPD input needs one wavelength (or a FocalPlanePropagator)")
        else:
            phase = data * (2 * np.pi / float(np.ravel(wavelength)[0]))
        data = amp * xp.exp(1j * phase.astype(be.real_dtype, copy=False))
    residual = perfect_coronagraph_field(data, amp, backend=be)
    image = xp.abs(propagator.forward(residual)) ** 2
    if normalize == "flux":
        return image / float(be.to_numpy(xp.sum(amp * amp)))
    if normalize == "peak":
        bare = amp if stack is None else xp.broadcast_to(amp, (stack.size, *amp.shape))
        reference = xp.abs(propagator.forward(bare.astype(be.complex_dtype))) ** 2
        return image / xp.max(reference, axis=(-2, -1), keepdims=True)
    return image


def azimuthal_profile(
    image: Any,
    center: Any = None,
    bin_width: float = 1.0,
    *,
    pixel_scale: float = 1.0,
    max_radius: float | None = None,
) -> tuple[Any, Any]:
    """Mean of an image in annuli about a centre, for contrast curves.

    Pixels are binned by the distance of their centre from ``center`` into
    annuli ``[k w, (k + 1) w)``, ``w = bin_width``, and averaged.

    Parameters
    ----------
    image:
        ``(..., ny, nx)`` image(s), NumPy or CuPy; leading axes are frames
        and each gets its own profile.
    center:
        ``(y, x)`` of the centre in pixels from the window centre, in the
        coordinates of CONVENTIONS 1.2 (as :func:`aocore.centroid` reports
        them). The default ``(0, 0)`` is the optical axis of an aocore image
        (1.3), pixel ``((ny - 1) / 2, (nx - 1) / 2)``.
    bin_width:
        Annulus width, in the units of the radii.
    pixel_scale:
        Size of one pixel in the units wanted for the radii: 1 (default) for
        pixels, ``pixel_scale_rad * D / wavelength`` for lambda/D, or the
        pixel scale in arcseconds for arcseconds.
    max_radius:
        Largest radius to profile (same units); pixels beyond it are left
        out. Default: the farthest pixel. There are ``ceil(max_radius / w)``
        bins, the last one closed at ``max_radius``.

    Returns
    -------
    radii, values:
        The bin centres ``(k + 1/2) w`` and the mean pixel value in each bin
        (``(..., K)``; NaN for a bin holding no pixel centre), on the image's
        device.
    """
    be = backend_of(image)
    xp = be.xp
    data = xp.asarray(image)
    if data.ndim < 2:
        raise ValueError(f"image must be at least 2-D, got shape {data.shape}")
    if not (bin_width > 0 and pixel_scale > 0):
        raise ValueError("bin_width and pixel_scale must be positive")
    cy, cx = (0.0, 0.0) if center is None else (float(center[0]), float(center[1]))
    ny, nx = data.shape[-2:]
    y = (np.arange(ny) - (ny - 1) / 2.0 - cy)[:, None]
    x = (np.arange(nx) - (nx - 1) / 2.0 - cx)[None, :]
    radius = np.hypot(x, y).ravel() * pixel_scale
    if max_radius is None:
        max_radius = float(radius.max())
    n_bins = max(1, math.ceil(max_radius / bin_width))
    # A pixel exactly at max_radius = n_bins * w joins the last bin.
    index = np.minimum(np.floor(radius / bin_width).astype(np.int64), n_bins - 1)
    keep = np.flatnonzero(radius <= max_radius)
    counts = np.bincount(index[keep], minlength=n_bins)[:n_bins].astype(np.float64)
    frames = data.reshape(-1, ny * nx)
    n_frames = frames.shape[0]
    # One bincount over all frames: frame f's bins sit at f * n_bins + k.
    labels = (np.arange(n_frames)[:, None] * n_bins + index[keep][None, :]).ravel()
    sums = xp.bincount(
        xp.asarray(labels),
        weights=frames[:, xp.asarray(keep)].ravel().astype(xp.float64),
        minlength=n_frames * n_bins,
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sums.reshape(n_frames, n_bins) / xp.asarray(np.where(counts > 0, counts, np.nan))
    radii = xp.asarray((np.arange(n_bins) + 0.5) * bin_width)
    return radii, mean.reshape(*data.shape[:-2], n_bins)
