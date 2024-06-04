"""Carteisan vector."""

__all__ = [
    "CartesianPosition1D",
    "CartesianVelocity1D",
    "CartesianAcceleration1D",
]

from dataclasses import replace
from functools import partial
from typing import final

import equinox as eqx
import jax

import quaxed.array_api as xp
from unxt import Quantity

import coordinax._typing as ct
from .base import AbstractAcceleration1D, AbstractPosition1D, AbstractVelocity1D
from coordinax._base import AbstractVector
from coordinax._base_pos import AbstractPosition
from coordinax._utils import classproperty


@final
class CartesianPosition1D(AbstractPosition1D):
    """Cartesian vector representation."""

    x: ct.BatchableLength = eqx.field(
        converter=partial(Quantity["length"].constructor, dtype=float)
    )
    r"""X coordinate :math:`x \in (-\infty,+\infty)`."""

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["CartesianVelocity1D"]:
        return CartesianVelocity1D

    # -----------------------------------------------------
    # Unary operations

    def __neg__(self) -> "Self":
        """Negate the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CartesianPosition1D
        >>> q = CartesianPosition1D.constructor(Quantity([1], "kpc"))
        >>> -q
        CartesianPosition1D(
           x=Quantity[PhysicalType('length')](value=f32[], unit=Unit("kpc"))
        )

        """
        return replace(self, x=-self.x)

    # -----------------------------------------------------
    # Binary operations

    @AbstractVector.__add__.dispatch  # type: ignore[misc]
    def __add__(
        self: "CartesianPosition1D", other: AbstractPosition, /
    ) -> "CartesianPosition1D":
        """Add two vectors.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CartesianPosition1D, RadialPosition

        >>> q = CartesianPosition1D.constructor(Quantity([1], "kpc"))
        >>> r = RadialPosition.constructor(Quantity([1], "kpc"))
        >>> qpr = q + r
        >>> qpr
        CartesianPosition1D(
           x=Quantity[PhysicalType('length')](value=f32[], unit=Unit("kpc"))
        )
        >>> qpr.x
        Quantity['length'](Array(2., dtype=float32), unit='kpc')

        """
        cart = other.represent_as(CartesianPosition1D)
        return replace(self, x=self.x + cart.x)

    @AbstractVector.__sub__.dispatch  # type: ignore[misc]
    def __sub__(
        self: "CartesianPosition1D", other: AbstractPosition, /
    ) -> "CartesianPosition1D":
        """Subtract two vectors.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CartesianPosition1D, RadialPosition

        >>> q = CartesianPosition1D.constructor(Quantity([1], "kpc"))
        >>> r = RadialPosition.constructor(Quantity([1], "kpc"))
        >>> qmr = q - r
        >>> qmr
        CartesianPosition1D(
           x=Quantity[PhysicalType('length')](value=f32[], unit=Unit("kpc"))
        )
        >>> qmr.x
        Quantity['length'](Array(0., dtype=float32), unit='kpc')

        """
        cart = other.represent_as(CartesianPosition1D)
        return replace(self, x=self.x - cart.x)

    # -----------------------------------------------------
    # Methods

    @partial(jax.jit)
    def norm(self) -> ct.BatchableLength:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CartesianPosition1D, RadialPosition

        >>> q = CartesianPosition1D.constructor(Quantity([-1], "kpc"))
        >>> q.norm()
        Quantity['length'](Array(1., dtype=float32), unit='kpc')

        """
        return xp.abs(self.x)


@final
class CartesianVelocity1D(AbstractVelocity1D):
    """Cartesian differential representation."""

    d_x: ct.BatchableSpeed = eqx.field(converter=Quantity["speed"].constructor)
    r"""X differential :math:`dx/dt \in (-\infty,+\infty`)`."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[CartesianPosition1D]:
        return CartesianPosition1D

    @partial(jax.jit)
    def norm(self, _: AbstractPosition1D | None = None, /) -> ct.BatchableSpeed:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CartesianVelocity1D
        >>> q = CartesianVelocity1D.constructor(Quantity([-1], "km/s"))
        >>> q.norm()
        Quantity['speed'](Array(1, dtype=int32), unit='km / s')

        """
        return xp.abs(self.d_x)


@final
class CartesianAcceleration1D(AbstractAcceleration1D):
    """Cartesian differential representation."""

    d2_x: ct.BatchableAcc = eqx.field(converter=Quantity["acceleration"].constructor)
    r"""X differential :math:`d^2x/dt^2 \in (-\infty,+\infty`)`."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[CartesianVelocity1D]:
        return CartesianVelocity1D

    # -----------------------------------------------------
    # Binary operations

    @AbstractVector.__add__.dispatch  # type: ignore[misc]
    def __add__(
        self: "CartesianAcceleration1D", other: "CartesianAcceleration1D", /
    ) -> "CartesianAcceleration1D":
        """Add two Cartesian Accelerations."""
        return replace(self, d2_x=self.d2_x + other.d2_x)

    @AbstractVector.__sub__.dispatch  # type: ignore[misc]
    def __sub__(
        self: "CartesianAcceleration1D", other: "CartesianAcceleration1D", /
    ) -> "CartesianAcceleration1D":
        """Subtract two accelerations."""
        return replace(self, d2_x=self.d2_x - other.d2_x)

    # -----------------------------------------------------
    # Methods

    @partial(jax.jit)
    def norm(self, _: AbstractPosition1D | None = None, /) -> ct.BatchableAcc:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CartesianAcceleration1D
        >>> q = CartesianAcceleration1D.constructor(Quantity([-1], "km/s2"))
        >>> q.norm()
        Quantity['acceleration'](Array(1, dtype=int32), unit='km / s2')

        """
        return xp.abs(self.d2_x)
