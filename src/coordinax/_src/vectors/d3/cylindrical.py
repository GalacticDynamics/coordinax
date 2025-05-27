"""Built-in vector classes."""

__all__ = [
    "CylindricalAcc",
    "CylindricalPos",
    "CylindricalVel",
]

import functools as ft
from typing import final
from typing_extensions import override

import equinox as eqx

import quaxed.numpy as jnp
import unxt as u
from dataclassish.converters import Unless

import coordinax._src.custom_types as ct
from .base import AbstractAcc3D, AbstractPos3D, AbstractVel3D
from coordinax._src.angles import Angle, BatchableAngle
from coordinax._src.distances import BBtLength
from coordinax._src.vectors.checks import check_r_non_negative
from coordinax._src.vectors.converters import converter_azimuth_to_range


@final
class CylindricalPos(AbstractPos3D):
    """Cylindrical vector representation.

    This adheres to ISO standard 31-11.

    """

    rho: BBtLength = eqx.field(converter=u.Quantity["length"].from_)
    r"""Cylindrical radial distance :math:`\rho \in [0,+\infty)`."""

    phi: BatchableAngle = eqx.field(
        converter=Unless(Angle, lambda x: converter_azimuth_to_range(Angle.from_(x)))
    )
    r"""Azimuthal angle, generally :math:`\phi \in [0,360)`."""

    z: BBtLength = eqx.field(converter=u.Quantity["length"].from_)
    r"""Height :math:`z \in (-\infty,+\infty)`."""

    def __check_init__(self) -> None:
        """Check the validity of the initialisation."""
        check_r_non_negative(self.rho)

    @override
    @ft.partial(eqx.filter_jit, inline=True)
    def norm(self) -> BBtLength:
        """Return the norm of the vector.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx
        >>> c = cx.vecs.CylindricalPos(rho=u.Quantity(3, "km"),
        ...                            phi=u.Quantity(0, "deg"),
        ...                            z=u.Quantity(4, "km"))
        >>> c.norm()
        Quantity(Array(5., dtype=float32, ...), unit='km')

        """
        return jnp.hypot(self.rho, self.z)


@final
class CylindricalVel(AbstractVel3D):
    """Cylindrical velocity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.vecs.CylindricalVel(rho=u.Quantity(1, "km/s"),
    ...                              phi=u.Quantity(2, "deg/s"),
    ...                              z=u.Quantity(3, "km/s"))
    >>> print(vec)
    <CylindricalVel: (rho[km / s], phi[deg / s], z[km / s])
        [1 2 3]>

    """

    rho: ct.BBtSpeed = eqx.field(converter=u.Quantity["speed"].from_)
    r"""Cyindrical radial speed :math:`d\rho/dt \in [-\infty, \infty]."""

    phi: ct.BBtAngularSpeed = eqx.field(converter=u.Quantity["angular speed"].from_)
    r"""Azimuthal speed :math:`d\phi/dt \in [-\infty, \infty]."""

    z: ct.BBtSpeed = eqx.field(converter=u.Quantity["speed"].from_)
    r"""Vertical speed :math:`dz/dt \in [-\infty, \infty]."""


@final
class CylindricalAcc(AbstractAcc3D):
    """Cylindrical acceleration representation.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.vecs.CylindricalAcc(rho=u.Quantity(1, "km/s2"),
    ...                              phi=u.Quantity(2, "deg/s2"),
    ...                              z=u.Quantity(3, "km/s2"))
    >>> print(vec)
    <CylindricalAcc: (rho[km / s2], phi[deg / s2], z[km / s2])
        [1 2 3]>

    """

    rho: ct.BBtAcc = eqx.field(converter=u.Quantity["acceleration"].from_)
    r"""Cyindrical radial acceleration :math:`d^2\rho/dt^2 \in [-\infty, \infty]."""

    phi: ct.BBtAngularAcc = eqx.field(
        converter=u.Quantity["angular acceleration"].from_
    )
    r"""Azimuthal acceleration :math:`d^2\phi/dt^2 \in [-\infty, \infty]."""

    z: ct.BBtAcc = eqx.field(converter=u.Quantity["acceleration"].from_)
    r"""Vertical acceleration :math:`d^2z/dt^2 \in [-\infty, \infty]."""
