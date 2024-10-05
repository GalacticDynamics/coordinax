"""Built-in vector classes."""

__all__ = [
    "CylindricalPos",
    "CylindricalVel",
    "CylindricalAcc",
]

from functools import partial
from typing import final
from typing_extensions import override

import equinox as eqx

import quaxed.numpy as xp
from unxt import Quantity

import coordinax._src.typing as ct
from .base import AbstractAcc3D, AbstractPos3D, AbstractVel3D
from coordinax._src.checks import check_azimuth_range, check_r_non_negative
from coordinax._src.converters import converter_azimuth_to_range
from coordinax._src.utils import classproperty


@final
class CylindricalPos(AbstractPos3D):
    """Cylindrical vector representation.

    This adheres to ISO standard 31-11.

    """

    rho: ct.BatchableLength = eqx.field(
        converter=partial(Quantity["length"].from_, dtype=float)
    )
    r"""Cylindrical radial distance :math:`\rho \in [0,+\infty)`."""

    phi: ct.BatchableAngle = eqx.field(
        converter=lambda x: converter_azimuth_to_range(
            Quantity["angle"].from_(x, dtype=float)  # pylint: disable=E1120
        )
    )
    r"""Azimuthal angle :math:`\phi \in [0,360)`."""

    z: ct.BatchableLength = eqx.field(
        converter=partial(Quantity["length"].from_, dtype=float)
    )
    r"""Height :math:`z \in (-\infty,+\infty)`."""

    def __check_init__(self) -> None:
        """Check the validity of the initialisation."""
        check_r_non_negative(self.rho)
        check_azimuth_range(self.phi)

    @override
    @classproperty
    @classmethod
    def differential_cls(cls) -> type["CylindricalVel"]:
        return CylindricalVel

    @override
    @partial(eqx.filter_jit, inline=True)
    def norm(self) -> ct.BatchableLength:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> c = cx.CylindricalPos(rho=Quantity(3, "kpc"), phi=Quantity(0, "deg"),
        ...                       z=Quantity(4, "kpc"))
        >>> c.norm()
        Quantity['length'](Array(5., dtype=float32), unit='kpc')

        """
        return xp.sqrt(self.rho**2 + self.z**2)


@final
class CylindricalVel(AbstractVel3D):
    """Cylindrical differential representation."""

    d_rho: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["speed"].from_, dtype=float)
    )
    r"""Cyindrical radial speed :math:`d\rho/dt \in [-\infty, \infty]."""

    d_phi: ct.BatchableAngularSpeed = eqx.field(
        converter=partial(Quantity["angular speed"].from_, dtype=float)
    )
    r"""Azimuthal speed :math:`d\phi/dt \in [-\infty, \infty]."""

    d_z: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["speed"].from_, dtype=float)
    )
    r"""Vertical speed :math:`dz/dt \in [-\infty, \infty]."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[CylindricalPos]:
        return CylindricalPos

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["CylindricalAcc"]:
        return CylindricalAcc


@final
class CylindricalAcc(AbstractAcc3D):
    """Cylindrical acceleration representation."""

    d2_rho: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["acceleration"].from_, dtype=float)
    )
    r"""Cyindrical radial acceleration :math:`d^2\rho/dt^2 \in [-\infty, \infty]."""

    d2_phi: ct.BatchableAngularSpeed = eqx.field(
        converter=partial(Quantity["angular acceleration"].from_, dtype=float)
    )
    r"""Azimuthal acceleration :math:`d^2\phi/dt^2 \in [-\infty, \infty]."""

    d2_z: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["acceleration"].from_, dtype=float)
    )
    r"""Vertical acceleration :math:`d^2z/dt^2 \in [-\infty, \infty]."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[CylindricalVel]:
        return CylindricalVel
