"""Built-in vector classes."""

__all__ = [
    "CylindricalPosition",
    "CylindricalVelocity",
    "CylindricalAcceleration",
]

from functools import partial
from typing import final

import equinox as eqx
import jax

import quaxed.array_api as xp
from unxt import Quantity

import coordinax._coordinax.typing as ct
from .base import AbstractAcceleration3D, AbstractPosition3D, AbstractVelocity3D
from coordinax._coordinax.checks import check_azimuth_range, check_r_non_negative
from coordinax._coordinax.converters import converter_azimuth_to_range
from coordinax._coordinax.utils import classproperty


@final
class CylindricalPosition(AbstractPosition3D):
    """Cylindrical vector representation.

    This adheres to ISO standard 31-11.

    """

    rho: ct.BatchableLength = eqx.field(
        converter=partial(Quantity["length"].constructor, dtype=float)
    )
    r"""Cylindrical radial distance :math:`\rho \in [0,+\infty)`."""

    phi: ct.BatchableAngle = eqx.field(
        converter=lambda x: converter_azimuth_to_range(
            Quantity["angle"].constructor(x, dtype=float)  # pylint: disable=E1120
        )
    )
    r"""Azimuthal angle :math:`\phi \in [0,360)`."""

    z: ct.BatchableLength = eqx.field(
        converter=partial(Quantity["length"].constructor, dtype=float)
    )
    r"""Height :math:`z \in (-\infty,+\infty)`."""

    def __check_init__(self) -> None:
        """Check the validity of the initialisation."""
        check_r_non_negative(self.rho)
        check_azimuth_range(self.phi)

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["CylindricalVelocity"]:
        return CylindricalVelocity

    @partial(jax.jit, inline=True)
    def norm(self) -> ct.BatchableLength:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> c = cx.CylindricalPosition(rho=Quantity(3, "kpc"), phi=Quantity(0, "deg"),
        ...                       z=Quantity(4, "kpc"))
        >>> c.norm()
        Quantity['length'](Array(5., dtype=float32), unit='kpc')

        """
        return xp.sqrt(self.rho**2 + self.z**2)


@final
class CylindricalVelocity(AbstractVelocity3D):
    """Cylindrical differential representation."""

    d_rho: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["speed"].constructor, dtype=float)
    )
    r"""Cyindrical radial speed :math:`d\rho/dt \in [-\infty, \infty]."""

    d_phi: ct.BatchableAngularSpeed = eqx.field(
        converter=partial(Quantity["angular speed"].constructor, dtype=float)
    )
    r"""Azimuthal speed :math:`d\phi/dt \in [-\infty, \infty]."""

    d_z: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["speed"].constructor, dtype=float)
    )
    r"""Vertical speed :math:`dz/dt \in [-\infty, \infty]."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[CylindricalPosition]:
        return CylindricalPosition

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["CylindricalAcceleration"]:
        return CylindricalAcceleration


@final
class CylindricalAcceleration(AbstractAcceleration3D):
    """Cylindrical acceleration representation."""

    d2_rho: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["acceleration"].constructor, dtype=float)
    )
    r"""Cyindrical radial acceleration :math:`d^2\rho/dt^2 \in [-\infty, \infty]."""

    d2_phi: ct.BatchableAngularSpeed = eqx.field(
        converter=partial(Quantity["angular acceleration"].constructor, dtype=float)
    )
    r"""Azimuthal acceleration :math:`d^2\phi/dt^2 \in [-\infty, \infty]."""

    d2_z: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["acceleration"].constructor, dtype=float)
    )
    r"""Vertical acceleration :math:`d^2z/dt^2 \in [-\infty, \infty]."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[CylindricalVelocity]:
        return CylindricalVelocity
