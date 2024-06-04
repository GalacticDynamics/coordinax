"""Built-in vector classes."""

__all__ = [
    "CartesianPosition2D",
    "CartesianVelocity2D",
    "CartesianAcceleration2D",
]

from dataclasses import replace
from functools import partial
from typing import final

import equinox as eqx
import jax

import quaxed.array_api as xp
from unxt import Quantity

import coordinax._typing as ct
from .base import AbstractAcceleration2D, AbstractPosition2D, AbstractVelocity2D
from coordinax._base import AbstractVector
from coordinax._base_pos import AbstractPosition
from coordinax._utils import classproperty


@final
class CartesianPosition2D(AbstractPosition2D):
    """Cartesian vector representation."""

    x: ct.BatchableLength = eqx.field(
        converter=partial(Quantity["length"].constructor, dtype=float)
    )
    r"""X coordinate :math:`x \in (-\infty,+\infty)`."""

    y: ct.BatchableLength = eqx.field(
        converter=partial(Quantity["length"].constructor, dtype=float)
    )
    r"""Y coordinate :math:`y \in (-\infty,+\infty)`."""

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["CartesianVelocity2D"]:
        return CartesianVelocity2D

    # -----------------------------------------------------
    # Unary operations

    def __neg__(self) -> "Self":
        """Negate the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CartesianPosition2D

        >>> q = CartesianPosition2D.constructor(Quantity([1, 2], "kpc"))
        >>> (-q).x
        Quantity['length'](Array(-1., dtype=float32), unit='kpc')

        """
        return replace(self, x=-self.x, y=-self.y)

    # -----------------------------------------------------
    # Binary operations

    @AbstractVector.__add__.dispatch  # type: ignore[misc]
    def __add__(
        self: "CartesianPosition2D", other: AbstractPosition, /
    ) -> "CartesianPosition2D":
        """Add two vectors.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CartesianPosition2D, PolarPosition
        >>> cart = CartesianPosition2D.constructor(Quantity([1, 2], "kpc"))
        >>> polr = PolarPosition(r=Quantity(3, "kpc"), phi=Quantity(90, "deg"))

        >>> (cart + polr).x
        Quantity['length'](Array(0.9999999, dtype=float32), unit='kpc')

        """
        cart = other.represent_as(CartesianPosition2D)
        return replace(self, x=self.x + cart.x, y=self.y + cart.y)

    @AbstractVector.__sub__.dispatch  # type: ignore[misc]
    def __sub__(
        self: "CartesianPosition2D", other: AbstractPosition, /
    ) -> "CartesianPosition2D":
        """Subtract two vectors.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CartesianPosition2D, PolarPosition
        >>> cart = CartesianPosition2D.constructor(Quantity([1, 2], "kpc"))
        >>> polr = PolarPosition(r=Quantity(3, "kpc"), phi=Quantity(90, "deg"))

        >>> (cart - polr).x
        Quantity['length'](Array(1.0000001, dtype=float32), unit='kpc')

        """
        cart = other.represent_as(CartesianPosition2D)
        return replace(self, x=self.x - cart.x, y=self.y - cart.y)

    @partial(jax.jit)
    def norm(self) -> ct.BatchableLength:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CartesianPosition2D
        >>> q = CartesianPosition2D.constructor(Quantity([3, 4], "kpc"))
        >>> q.norm()
        Quantity['length'](Array(5., dtype=float32), unit='kpc')

        """
        return xp.sqrt(self.x**2 + self.y**2)


@final
class CartesianVelocity2D(AbstractVelocity2D):
    """Cartesian differential representation."""

    d_x: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["speed"].constructor, dtype=float)
    )
    r"""X coordinate differential :math:`\dot{x} \in (-\infty,+\infty)`."""

    d_y: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["speed"].constructor, dtype=float)
    )
    r"""Y coordinate differential :math:`\dot{y} \in (-\infty,+\infty)`."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[CartesianPosition2D]:
        return CartesianPosition2D

    @partial(jax.jit)
    def norm(self, _: AbstractPosition2D | None = None, /) -> ct.BatchableSpeed:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CartesianVelocity2D
        >>> v = CartesianVelocity2D.constructor(Quantity([3, 4], "km/s"))
        >>> v.norm()
        Quantity['speed'](Array(5., dtype=float32), unit='km / s')

        """
        return xp.sqrt(self.d_x**2 + self.d_y**2)


@final
class CartesianAcceleration2D(AbstractAcceleration2D):
    """Cartesian acceleration representation."""

    d2_x: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["acceleration"].constructor, dtype=float)
    )
    r"""X coordinate acceleration :math:`\frac{d^2 x}{dt^2} \in (-\infty,+\infty)`."""

    d2_y: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["acceleration"].constructor, dtype=float)
    )
    r"""Y coordinate acceleration :math:`\frac{d^2 y}{dt^2} \in (-\infty,+\infty)`."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[CartesianVelocity2D]:
        return CartesianVelocity2D

    @partial(jax.jit)
    def norm(self, _: AbstractVelocity2D | None = None, /) -> ct.BatchableAcc:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CartesianAcceleration2D
        >>> v = CartesianAcceleration2D.constructor(Quantity([3, 4], "km/s2"))
        >>> v.norm()
        Quantity['acceleration'](Array(5., dtype=float32), unit='km / s2')

        """
        return xp.sqrt(self.d2_x**2 + self.d2_y**2)
