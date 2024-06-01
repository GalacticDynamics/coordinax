"""Built-in vector classes."""

__all__ = [
    # Position
    "Cartesian3DVector",
    "CylindricalVector",
    # Differential
    "CartesianDifferential3D",
    "CylindricalDifferential",
]

from dataclasses import replace
from functools import partial
from typing import Any, final

import equinox as eqx
import jax

import quaxed.array_api as xp
from unxt import Quantity

import coordinax._typing as ct
from .base import Abstract3DVector, Abstract3DVectorDifferential
from coordinax._base_pos import AbstractPosition
from coordinax._base_vel import AdditionMixin
from coordinax._checks import check_azimuth_range, check_r_non_negative
from coordinax._converters import converter_azimuth_to_range
from coordinax._utils import classproperty

##############################################################################
# Position


@final
class Cartesian3DVector(Abstract3DVector):
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
    def differential_cls(cls) -> type["CartesianDifferential3D"]:
        return CartesianDifferential3D

    # -----------------------------------------------------
    # Unary operations

    def __neg__(self) -> "Self":
        """Negate the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import Cartesian3DVector
        >>> q = Cartesian3DVector.constructor(Quantity([1, 2, 3], "kpc"))
        >>> (-q).x
        Quantity['length'](Array(-1., dtype=float32), unit='kpc')

        """
        return replace(self, x=-self.x, y=-self.y, z=-self.z)

    # -----------------------------------------------------
    # Binary operations

    def __add__(self, other: Any, /) -> "Cartesian3DVector":
        """Add two vectors.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import Cartesian3DVector, SphericalVector
        >>> q = Cartesian3DVector.constructor(Quantity([1, 2, 3], "kpc"))
        >>> s = SphericalVector(r=Quantity(1, "kpc"), theta=Quantity(90, "deg"),
        ...                     phi=Quantity(0, "deg"))
        >>> (q + s).x
        Quantity['length'](Array(2., dtype=float32), unit='kpc')

        """
        if not isinstance(other, AbstractPosition):
            msg = f"Cannot add {self._cartesian_cls!r} and {type(other)!r}."
            raise TypeError(msg)

        cart = other.represent_as(Cartesian3DVector)
        return replace(self, x=self.x + cart.x, y=self.y + cart.y, z=self.z + cart.z)

    def __sub__(self, other: Any, /) -> "Cartesian3DVector":
        """Subtract two vectors.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import Cartesian3DVector, SphericalVector
        >>> q = Cartesian3DVector.constructor(Quantity([1, 2, 3], "kpc"))
        >>> s = SphericalVector(r=Quantity(1, "kpc"), theta=Quantity(90, "deg"),
        ...                     phi=Quantity(0, "deg"))
        >>> (q - s).x
        Quantity['length'](Array(0., dtype=float32), unit='kpc')

        """
        if not isinstance(other, AbstractPosition):
            msg = f"Cannot subtract {self._cartesian_cls!r} and {type(other)!r}."
            raise TypeError(msg)

        cart = other.represent_as(Cartesian3DVector)
        return replace(self, x=self.x - cart.x, y=self.y - cart.y, z=self.z - cart.z)

    @partial(jax.jit)
    def norm(self) -> ct.BatchableLength:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import Cartesian3DVector
        >>> q = Cartesian3DVector.constructor(Quantity([1, 2, 3], "kpc"))
        >>> q.norm()
        Quantity['length'](Array(3.7416575, dtype=float32), unit='kpc')

        """
        return xp.sqrt(self.x**2 + self.y**2 + self.z**2)


@final
class CylindricalVector(Abstract3DVector):
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
    def differential_cls(cls) -> type["CylindricalDifferential"]:
        return CylindricalDifferential

    @partial(jax.jit)
    def norm(self) -> ct.BatchableLength:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CylindricalVector
        >>> c = CylindricalVector(rho=Quantity(3, "kpc"), phi=Quantity(0, "deg"),
        ...                       z=Quantity(4, "kpc"))
        >>> c.norm()
        Quantity['length'](Array(5., dtype=float32), unit='kpc')

        """
        return xp.sqrt(self.rho**2 + self.z**2)


##############################################################################
# Differential


@final
class CartesianDifferential3D(Abstract3DVectorDifferential, AdditionMixin):
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
    def integral_cls(cls) -> type[Cartesian3DVector]:
        return Cartesian3DVector

    @partial(jax.jit)
    def norm(self, _: Abstract3DVector | None = None, /) -> ct.BatchableSpeed:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CartesianDifferential3D
        >>> c = CartesianDifferential3D(d_x=Quantity(1, "km/s"),
        ...                              d_y=Quantity(2, "km/s"),
        ...                              d_z=Quantity(3, "km/s"))
        >>> c.norm()
        Quantity['speed'](Array(3.7416575, dtype=float32), unit='km / s')

        """
        return xp.sqrt(self.d_x**2 + self.d_y**2 + self.d_z**2)


@final
class CylindricalDifferential(Abstract3DVectorDifferential):
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
    def integral_cls(cls) -> type[CylindricalVector]:
        return CylindricalVector
