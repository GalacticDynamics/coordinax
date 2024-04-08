"""Representation of coordinates in different systems."""

__all__: list[str] = []


import equinox as eqx

import quaxed.array_api as xp
from unxt import Quantity

from coordinax._typing import BatchableAngle, BatchableLength

_0m = Quantity(0, "meter")
_0d = Quantity(0, "rad")
_pid = Quantity(180, "deg")
_2pid = Quantity(360, "deg")


def check_r_non_negative(
    r: BatchableLength, /, _lower: Quantity["length"] = _0m
) -> BatchableLength:
    """Check that the radial distance is non-negative."""
    return eqx.error_if(
        r, xp.any(r < _lower), "The radial distance must be non-negative."
    )


def check_azimuth_range(
    phi: BatchableAngle,
    /,
    _lower: Quantity["angle"] = _0d,
    _upper: Quantity["angle"] = _2pid,
) -> BatchableAngle:
    """Check that the polar angle is in the range [0, 2pi)."""
    return eqx.error_if(
        phi,
        xp.any((phi < _lower) | (phi >= _upper)),
        "The azimuthal (polar) angle must be in the range [0, 2pi).",
    )


def check_polar_range(
    theta: BatchableAngle,
    /,
    _lower: Quantity["angle"] = _0d,
    _upper: Quantity["angle"] = _pid,
) -> BatchableAngle:
    """Check that the inclination angle is in the range [0, pi]."""
    return eqx.error_if(
        theta,
        xp.any(xp.logical_or((theta < _lower), (theta > _upper))),
        "The inclination angle must be in the range [0, pi].",
    )
