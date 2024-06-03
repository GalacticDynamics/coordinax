"""Built-in vector classes."""

__all__ = [
    # Position
    "CartesianPosition3D",
    "CylindricalPosition",
    # Differential
    "CartesianVelocity3D",
    "CylindricalVelocity",
]

from dataclasses import replace
from functools import partial
from typing import final

import equinox as eqx
import jax

import quaxed.array_api as xp
from unxt import Quantity

import coordinax._typing as ct
from .base import AbstractPosition3D, AbstractVelocity3D
from coordinax._base import AbstractVector
from coordinax._base_pos import AbstractPosition
from coordinax._base_vel import AdditionMixin
from coordinax._checks import check_azimuth_range, check_r_non_negative
from coordinax._converters import converter_azimuth_to_range
from coordinax._utils import classproperty

##############################################################################
# Position


@final
class CartesianPosition3D(AbstractPosition3D):
    """Cartesian vector representation."""

    x: ct.BatchableLength = eqx.field(
        converter=partial(Quantity["length"].constructor, dtype=float)
    )
    r"""X coordinate :math:`x \in (-\infty,+\infty)`."""

    y: ct.BatchableLength = eqx.field(
        converter=partial(Quantity["length"].constructor, dtype=float)
    )
    r"""Y coordinate :math:`y \in (-\infty,+\infty)`."""

    z: ct.BatchableLength = eqx.field(
        converter=partial(Quantity["length"].constructor, dtype=float)
    )
    r"""Z coordinate :math:`z \in (-\infty,+\infty)`."""

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["CartesianVelocity3D"]:
        return CartesianVelocity3D

    # -----------------------------------------------------
    # Unary operations

    def __neg__(self) -> "Self":
        """Negate the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CartesianPosition3D
        >>> q = CartesianPosition3D.constructor(Quantity([1, 2, 3], "kpc"))
        >>> (-q).x
        Quantity['length'](Array(-1., dtype=float32), unit='kpc')

        """
        return replace(self, x=-self.x, y=-self.y, z=-self.z)

    # -----------------------------------------------------
    # Binary operations

    @AbstractVector.__add__.dispatch  # type: ignore[misc]
    def __add__(
        self: "CartesianPosition3D", other: AbstractPosition, /
    ) -> "CartesianPosition3D":
        """Add two vectors.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CartesianPosition3D, SphericalPosition
        >>> q = CartesianPosition3D.constructor(Quantity([1, 2, 3], "kpc"))
        >>> s = SphericalPosition(r=Quantity(1, "kpc"), theta=Quantity(90, "deg"),
        ...                     phi=Quantity(0, "deg"))
        >>> (q + s).x
        Quantity['length'](Array(2., dtype=float32), unit='kpc')

        """
        cart = other.represent_as(CartesianPosition3D)
        return replace(self, x=self.x + cart.x, y=self.y + cart.y, z=self.z + cart.z)

    @AbstractVector.__sub__.dispatch  # type: ignore[misc]
    def __sub__(
        self: "CartesianPosition3D", other: AbstractPosition, /
    ) -> "CartesianPosition3D":
        """Subtract two vectors.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CartesianPosition3D, SphericalPosition
        >>> q = CartesianPosition3D.constructor(Quantity([1, 2, 3], "kpc"))
        >>> s = SphericalPosition(r=Quantity(1, "kpc"), theta=Quantity(90, "deg"),
        ...                     phi=Quantity(0, "deg"))
        >>> (q - s).x
        Quantity['length'](Array(0., dtype=float32), unit='kpc')

        """
        cart = other.represent_as(CartesianPosition3D)
        return replace(self, x=self.x - cart.x, y=self.y - cart.y, z=self.z - cart.z)

    @partial(jax.jit)
    def norm(self) -> ct.BatchableLength:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CartesianPosition3D
        >>> q = CartesianPosition3D.constructor(Quantity([1, 2, 3], "kpc"))
        >>> q.norm()
        Quantity['length'](Array(3.7416575, dtype=float32), unit='kpc')

        """
        return xp.sqrt(self.x**2 + self.y**2 + self.z**2)


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

    @partial(jax.jit)
    def norm(self) -> ct.BatchableLength:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CylindricalPosition
        >>> c = CylindricalPosition(rho=Quantity(3, "kpc"), phi=Quantity(0, "deg"),
        ...                       z=Quantity(4, "kpc"))
        >>> c.norm()
        Quantity['length'](Array(5., dtype=float32), unit='kpc')

        """
        return xp.sqrt(self.rho**2 + self.z**2)


##############################################################################
# Differential


@final
class CartesianVelocity3D(AbstractVelocity3D, AdditionMixin):
    """Cartesian differential representation."""

    d_x: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["speed"].constructor, dtype=float)
    )
    r"""X speed :math:`dx/dt \in [-\infty, \infty]."""

    d_y: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["speed"].constructor, dtype=float)
    )
    r"""Y speed :math:`dy/dt \in [-\infty, \infty]."""

    d_z: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["speed"].constructor, dtype=float)
    )
    r"""Z speed :math:`dz/dt \in [-\infty, \infty]."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[CartesianPosition3D]:
        return CartesianPosition3D

    @partial(jax.jit)
    def norm(self, _: AbstractPosition3D | None = None, /) -> ct.BatchableSpeed:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CartesianVelocity3D
        >>> c = CartesianVelocity3D(d_x=Quantity(1, "km/s"),
        ...                              d_y=Quantity(2, "km/s"),
        ...                              d_z=Quantity(3, "km/s"))
        >>> c.norm()
        Quantity['speed'](Array(3.7416575, dtype=float32), unit='km / s')

        """
        return xp.sqrt(self.d_x**2 + self.d_y**2 + self.d_z**2)


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
