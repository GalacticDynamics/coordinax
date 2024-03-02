"""Representation of coordinates in different systems."""

__all__: list[str] = []


import equinox as eqx

import array_api_jax_compat as xp
from jax_quantity import Quantity

from coordinax._typing import BatchableAngle, BatchableLength

_0m = Quantity(0, "meter")
_0d = Quantity(0, "rad")
_pid = Quantity(180, "deg")
_2pid = Quantity(360, "deg")


def check_r_non_negative(r: BatchableLength) -> BatchableLength:
    """Check that the radial distance is non-negative."""
    return eqx.error_if(
        r,
        xp.any(r < _0m),
        "The radial distance must be non-negative.",
    )


def check_phi_range(phi: BatchableAngle) -> BatchableAngle:
    """Check that the polar angle is in the range [0, 2pi)."""
    return eqx.error_if(
        phi,
        xp.any((phi < _0d) | (phi >= _2pid)),
        "The azimuthal (polar) angle must be in the range [0, 2pi).",
    )


def check_theta_range(theta: BatchableAngle) -> BatchableAngle:
    """Check that the inclination angle is in the range [0, pi]."""
    return eqx.error_if(
        theta,
        xp.any((theta < _0d) | (theta > _pid)),
        "The inclination angle must be in the range [0, pi].",
    )
