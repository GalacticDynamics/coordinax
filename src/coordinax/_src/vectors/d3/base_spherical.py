"""Built-in vector classes."""

__all__ = [
    "AbstractSphericalAcc",
    "AbstractSphericalPos",
    "AbstractSphericalVel",
]


import unxt as u

from .base import AbstractAcc3D, AbstractPos3D, AbstractVel3D

_90d = u.Quantity(90, "deg")
_180d = u.Quantity(180, "deg")
_360d = u.Quantity(360, "deg")


class AbstractSphericalPos(AbstractPos3D):
    """Abstract spherical vector representation."""


class AbstractSphericalVel(AbstractVel3D):
    """Spherical differential representation."""


class AbstractSphericalAcc(AbstractAcc3D):
    """Spherical acceleration representation."""
