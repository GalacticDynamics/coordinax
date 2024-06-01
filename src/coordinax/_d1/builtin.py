"""Carteisan vector."""

__all__ = [
    # Position
    "Cartesian1DVector",
    "RadialVector",
    # Differential
    "CartesianDifferential1D",
    "RadialDifferential",
]

from dataclasses import replace
from functools import partial
from typing import Any, final

import equinox as eqx
import jax

import quaxed.array_api as xp
from unxt import AbstractDistance, Distance, Quantity

import coordinax._typing as ct
from .base import Abstract1DVector, Abstract1DVectorDifferential
from coordinax._base_pos import AbstractPosition
from coordinax._checks import check_r_non_negative
from coordinax._utils import classproperty

##############################################################################
# Position


@final
class Cartesian1DVector(Abstract1DVector):
    """Cartesian vector representation."""

    x: ct.BatchableLength = eqx.field(
        converter=partial(Quantity["length"].constructor, dtype=float)
    )
    r"""X coordinate :math:`x \in (-\infty,+\infty)`."""

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["CartesianDifferential1D"]:
        return CartesianDifferential1D

    # -----------------------------------------------------
    # Unary operations

    def __neg__(self) -> "Self":
        """Negate the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import Cartesian1DVector
        >>> q = Cartesian1DVector.constructor(Quantity([1], "kpc"))
        >>> -q
        Cartesian1DVector(
           x=Quantity[PhysicalType('length')](value=f32[], unit=Unit("kpc"))
        )

        """
        return replace(self, x=-self.x)

    # -----------------------------------------------------
    # Binary operations

    def __add__(self, other: Any, /) -> "Cartesian1DVector":
        """Add two vectors.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import Cartesian1DVector, RadialVector

        >>> q = Cartesian1DVector.constructor(Quantity([1], "kpc"))
        >>> r = RadialVector.constructor(Quantity([1], "kpc"))
        >>> qpr = q + r
        >>> qpr
        Cartesian1DVector(
           x=Quantity[PhysicalType('length')](value=f32[], unit=Unit("kpc"))
        )
        >>> qpr.x
        Quantity['length'](Array(2., dtype=float32), unit='kpc')

        """
        if not isinstance(other, AbstractPosition):
            msg = f"Cannot add {Cartesian1DVector!r} and {type(other)!r}."
            raise TypeError(msg)

        cart = other.represent_as(Cartesian1DVector)
        return replace(self, x=self.x + cart.x)

    def __sub__(self, other: Any, /) -> "Cartesian1DVector":
        """Subtract two vectors.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import Cartesian1DVector, RadialVector

        >>> q = Cartesian1DVector.constructor(Quantity([1], "kpc"))
        >>> r = RadialVector.constructor(Quantity([1], "kpc"))
        >>> qmr = q - r
        >>> qmr
        Cartesian1DVector(
           x=Quantity[PhysicalType('length')](value=f32[], unit=Unit("kpc"))
        )
        >>> qmr.x
        Quantity['length'](Array(0., dtype=float32), unit='kpc')

        """
        if not isinstance(other, AbstractPosition):
            msg = f"Cannot subtract {Cartesian1DVector!r} and {type(other)!r}."
            raise TypeError(msg)

        cart = other.represent_as(Cartesian1DVector)
        return replace(self, x=self.x - cart.x)

    @partial(jax.jit)
    def norm(self) -> ct.BatchableLength:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import Cartesian1DVector, RadialVector

        >>> q = Cartesian1DVector.constructor(Quantity([-1], "kpc"))
        >>> q.norm()
        Quantity['length'](Array(1., dtype=float32), unit='kpc')

        """
        return xp.abs(self.x)


@final
class RadialVector(Abstract1DVector):
    """Radial vector representation."""

    r: ct.BatchableDistance = eqx.field(
        converter=lambda x: x
        if isinstance(x, AbstractDistance)
        else Distance.constructor(x, dtype=float)
    )
    r"""Radial distance :math:`r \in [0,+\infty)`."""

    def __check_init__(self) -> None:
        """Check the initialization."""
        check_r_non_negative(self.r)

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["RadialDifferential"]:
        return RadialDifferential


##############################################################################
# Velocity


@final
class CartesianDifferential1D(Abstract1DVectorDifferential):
    """Cartesian differential representation."""

    d_x: ct.BatchableSpeed = eqx.field(converter=Quantity["speed"].constructor)
    r"""X differential :math:`dx/dt \in (-\infty,+\infty`)`."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[Cartesian1DVector]:
        return Cartesian1DVector

    @partial(jax.jit)
    def norm(self, _: Abstract1DVector | None = None, /) -> ct.BatchableSpeed:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CartesianDifferential1D
        >>> q = CartesianDifferential1D.constructor(Quantity([-1], "km/s"))
        >>> q.norm()
        Quantity['speed'](Array(1, dtype=int32), unit='km / s')

        """
        return xp.abs(self.d_x)


@final
class RadialDifferential(Abstract1DVectorDifferential):
    """Radial differential representation."""

    d_r: ct.BatchableSpeed = eqx.field(converter=Quantity["speed"].constructor)
    r"""Radial speed :math:`dr/dt \in (-\infty,+\infty)`."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[RadialVector]:
        return RadialVector
