"""Built-in vector classes."""

__all__ = [
    # Position
    "Cartesian3DVector",
    "SphericalVector",
    "CylindricalVector",
    # Differential
    "CartesianDifferential3D",
    "SphericalDifferential",
    "CylindricalDifferential",
]

from dataclasses import replace
from functools import partial
from typing import Any, final

import equinox as eqx
import jax

import array_api_jax_compat as xp
from jax_quantity import Quantity

from .base import Abstract3DVector, Abstract3DVectorDifferential
from coordinax._base import AbstractVector
from coordinax._checks import check_phi_range, check_r_non_negative, check_theta_range
from coordinax._typing import (
    BatchableAngle,
    BatchableAngularSpeed,
    BatchableLength,
    BatchableSpeed,
)
from coordinax._utils import classproperty

##############################################################################
# Position


@final
class Cartesian3DVector(Abstract3DVector):
    """Cartesian vector representation."""

    x: BatchableLength = eqx.field(
        converter=partial(Quantity["length"].constructor, dtype=float)
    )
    r"""X coordinate :math:`x \in (-\infty,+\infty)`."""

    y: BatchableLength = eqx.field(
        converter=partial(Quantity["length"].constructor, dtype=float)
    )
    r"""Y coordinate :math:`y \in (-\infty,+\infty)`."""

    z: BatchableLength = eqx.field(
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
        >>> from jax_quantity import Quantity
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
        >>> from jax_quantity import Quantity
        >>> from coordinax import Cartesian3DVector, SphericalVector
        >>> q = Cartesian3DVector.constructor(Quantity([1, 2, 3], "kpc"))
        >>> s = SphericalVector(r=Quantity(1, "kpc"), theta=Quantity(90, "deg"),
        ...                     phi=Quantity(0, "deg"))
        >>> (q + s).x
        Quantity['length'](Array(2., dtype=float32), unit='kpc')

        """
        if not isinstance(other, AbstractVector):
            msg = f"Cannot add {self._cartesian_cls!r} and {type(other)!r}."
            raise TypeError(msg)

        cart = other.represent_as(Cartesian3DVector)
        return replace(self, x=self.x + cart.x, y=self.y + cart.y, z=self.z + cart.z)

    def __sub__(self, other: Any, /) -> "Cartesian3DVector":
        """Subtract two vectors.

        Examples
        --------
        >>> from jax_quantity import Quantity
        >>> from coordinax import Cartesian3DVector, SphericalVector
        >>> q = Cartesian3DVector.constructor(Quantity([1, 2, 3], "kpc"))
        >>> s = SphericalVector(r=Quantity(1, "kpc"), theta=Quantity(90, "deg"),
        ...                     phi=Quantity(0, "deg"))
        >>> (q - s).x
        Quantity['length'](Array(0., dtype=float32), unit='kpc')

        """
        if not isinstance(other, AbstractVector):
            msg = f"Cannot subtract {self._cartesian_cls!r} and {type(other)!r}."
            raise TypeError(msg)

        cart = other.represent_as(Cartesian3DVector)
        return replace(self, x=self.x - cart.x, y=self.y - cart.y, z=self.z - cart.z)

    @partial(jax.jit)
    def norm(self) -> BatchableLength:
        """Return the norm of the vector.

        Examples
        --------
        >>> from jax_quantity import Quantity
        >>> from coordinax import Cartesian3DVector
        >>> q = Cartesian3DVector.constructor(Quantity([1, 2, 3], "kpc"))
        >>> q.norm()
        Quantity['length'](Array(3.7416575, dtype=float32), unit='kpc')

        """
        return xp.sqrt(self.x**2 + self.y**2 + self.z**2)


@final
class SphericalVector(Abstract3DVector):
    """Spherical vector representation."""

    r: BatchableLength = eqx.field(
        converter=partial(Quantity["length"].constructor, dtype=float)
    )
    r"""Radial distance :math:`r \in [0,+\infty)`."""

    theta: BatchableAngle = eqx.field(
        converter=partial(Quantity["angle"].constructor, dtype=float)
    )
    r"""Inclination angle :math:`\phi \in [0,180]`."""

    phi: BatchableAngle = eqx.field(
        converter=partial(Quantity["angle"].constructor, dtype=float)
    )
    r"""Azimuthal angle :math:`\phi \in [0,360)`."""

    def __check_init__(self) -> None:
        """Check the validity of the initialisation."""
        check_r_non_negative(self.r)
        check_theta_range(self.theta)
        check_phi_range(self.phi)

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["SphericalDifferential"]:
        return SphericalDifferential

    @partial(jax.jit)
    def norm(self) -> BatchableLength:
        """Return the norm of the vector.

        Examples
        --------
        >>> from jax_quantity import Quantity
        >>> from coordinax import SphericalVector
        >>> s = SphericalVector(r=Quantity(3, "kpc"), theta=Quantity(90, "deg"),
        ...                     phi=Quantity(0, "deg"))
        >>> s.norm()
        Quantity['length'](Array(3., dtype=float32), unit='kpc')

        """
        return self.r


@final
class CylindricalVector(Abstract3DVector):
    """Cylindrical vector representation."""

    rho: BatchableLength = eqx.field(
        converter=partial(Quantity["length"].constructor, dtype=float)
    )
    r"""Cylindrical radial distance :math:`\rho \in [0,+\infty)`."""

    phi: BatchableAngle = eqx.field(
        converter=partial(Quantity["angle"].constructor, dtype=float)
    )
    r"""Azimuthal angle :math:`\phi \in [0,360)`."""

    z: BatchableLength = eqx.field(
        converter=partial(Quantity["length"].constructor, dtype=float)
    )
    r"""Height :math:`z \in (-\infty,+\infty)`."""

    def __check_init__(self) -> None:
        """Check the validity of the initialisation."""
        check_r_non_negative(self.rho)
        check_phi_range(self.phi)

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["CylindricalDifferential"]:
        return CylindricalDifferential

    @partial(jax.jit)
    def norm(self) -> BatchableLength:
        """Return the norm of the vector.

        Examples
        --------
        >>> from jax_quantity import Quantity
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
class CartesianDifferential3D(Abstract3DVectorDifferential):
    """Cartesian differential representation."""

    d_x: BatchableSpeed = eqx.field(
        converter=partial(Quantity["speed"].constructor, dtype=float)
    )
    r"""X speed :math:`dx/dt \in [-\infty, \infty]."""

    d_y: BatchableSpeed = eqx.field(
        converter=partial(Quantity["speed"].constructor, dtype=float)
    )
    r"""Y speed :math:`dy/dt \in [-\infty, \infty]."""

    d_z: BatchableSpeed = eqx.field(
        converter=partial(Quantity["speed"].constructor, dtype=float)
    )
    r"""Z speed :math:`dz/dt \in [-\infty, \infty]."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[Cartesian3DVector]:
        return Cartesian3DVector

    @partial(jax.jit)
    def norm(self, _: Abstract3DVector | None = None, /) -> BatchableSpeed:
        """Return the norm of the vector.

        Examples
        --------
        >>> from jax_quantity import Quantity
        >>> from coordinax import CartesianDifferential3D
        >>> c = CartesianDifferential3D(d_x=Quantity(1, "km/s"),
        ...                              d_y=Quantity(2, "km/s"),
        ...                              d_z=Quantity(3, "km/s"))
        >>> c.norm()
        Quantity['speed'](Array(3.7416575, dtype=float32), unit='km / s')

        """
        return xp.sqrt(self.d_x**2 + self.d_y**2 + self.d_z**2)


@final
class SphericalDifferential(Abstract3DVectorDifferential):
    """Spherical differential representation."""

    d_r: BatchableSpeed = eqx.field(
        converter=partial(Quantity["speed"].constructor, dtype=float)
    )
    r"""Radial speed :math:`dr/dt \in [-\infty, \infty]."""

    d_theta: BatchableAngularSpeed = eqx.field(
        converter=partial(Quantity["angular speed"].constructor, dtype=float)
    )
    r"""Inclination speed :math:`d\theta/dt \in [-\infty, \infty]."""

    d_phi: BatchableAngularSpeed = eqx.field(
        converter=partial(Quantity["angular speed"].constructor, dtype=float)
    )
    r"""Azimuthal speed :math:`d\phi/dt \in [-\infty, \infty]."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[SphericalVector]:
        return SphericalVector


@final
class CylindricalDifferential(Abstract3DVectorDifferential):
    """Cylindrical differential representation."""

    d_rho: BatchableSpeed = eqx.field(
        converter=partial(Quantity["speed"].constructor, dtype=float)
    )
    r"""Cyindrical radial speed :math:`d\rho/dt \in [-\infty, \infty]."""

    d_phi: BatchableAngularSpeed = eqx.field(
        converter=partial(Quantity["angular speed"].constructor, dtype=float)
    )
    r"""Azimuthal speed :math:`d\phi/dt \in [-\infty, \infty]."""

    d_z: BatchableSpeed = eqx.field(
        converter=partial(Quantity["speed"].constructor, dtype=float)
    )
    r"""Vertical speed :math:`dz/dt \in [-\infty, \infty]."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[CylindricalVector]:
        return CylindricalVector
