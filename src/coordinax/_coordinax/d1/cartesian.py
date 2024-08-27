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
from jaxtyping import ArrayLike
from quax import register

import quaxed.array_api as xp
from quaxed import lax as qlax
from unxt import Quantity

import coordinax._coordinax.typing as ct
from .base import AbstractAcceleration1D, AbstractPosition1D, AbstractVelocity1D
from coordinax._coordinax.base_pos import AbstractPosition
from coordinax._coordinax.mixins import AvalMixin
from coordinax._coordinax.utils import classproperty


@final
class CartesianPosition1D(AbstractPosition1D):
    """Cartesian vector representation.

    Examples
    --------
    >>> import coordinax as cx

    >>> vec = cx.CartesianPosition1D.constructor([2], "m")
    >>> vec
    CartesianPosition1D(
      x=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m"))
    )

    Vectors support the basic math operations:

    >>> (vec + vec).x
    Quantity['length'](Array(4., dtype=float32), unit='m')

    >>> (vec - vec).x
    Quantity['length'](Array(0., dtype=float32), unit='m')

    >>> (3 * vec).x
    Quantity['length'](Array(6., dtype=float32), unit='m')

    """

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
        >>> import coordinax as cx
        >>> q = cx.CartesianPosition1D.constructor([1], "kpc")
        >>> -q
        CartesianPosition1D(
           x=Quantity[PhysicalType('length')](value=f32[], unit=Unit("kpc"))
        )

        """
        return replace(self, x=-self.x)


# -------------------------------------------------------------------
# Method dispatches


@register(jax.lax.add_p)  # type: ignore[misc]
def _add_qq(lhs: CartesianPosition1D, rhs: AbstractPosition, /) -> CartesianPosition1D:
    """Add a vector to a CartesianPosition1D.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> import coordinax as cx

    >>> q = cx.CartesianPosition1D.constructor([1], "kpc")
    >>> r = cx.RadialPosition.constructor([1], "kpc")

    >>> qpr = xp.add(q, r)
    >>> qpr
    CartesianPosition1D(
        x=Quantity[PhysicalType('length')](value=f32[], unit=Unit("kpc"))
    )
    >>> qpr.x
    Quantity['length'](Array(2., dtype=float32), unit='kpc')

    >>> (q + r).x
    Quantity['length'](Array(2., dtype=float32), unit='kpc')

    """
    cart = rhs.represent_as(CartesianPosition1D)
    return jax.tree.map(qlax.add, lhs, cart)


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_ac1(lhs: ArrayLike, rhs: CartesianPosition1D, /) -> CartesianPosition1D:
    """Scale a position by a scalar.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> v = cx.CartesianPosition1D(x=Quantity(1, "m"))
    >>> xp.multiply(2, v).x
    Quantity['length'](Array(2., dtype=float32), unit='m')

    >>> (2 * v).x
    Quantity['length'](Array(2., dtype=float32), unit='m')

    """
    # Validation
    lhs = eqx.error_if(
        lhs, any(jax.numpy.shape(lhs)), f"must be a scalar, not {type(lhs)}"
    )

    # Scale the components
    return replace(rhs, x=lhs * rhs.x)


@register(jax.lax.sub_p)  # type: ignore[misc]
def _sub_q1d_pos(
    self: CartesianPosition1D, other: AbstractPosition, /
) -> CartesianPosition1D:
    """Subtract two vectors.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> q = cx.CartesianPosition1D.constructor(Quantity([1], "kpc"))
    >>> r = cx.RadialPosition.constructor(Quantity([1], "kpc"))

    >>> qmr = xp.subtract(q, r)
    >>> qmr
    CartesianPosition1D(
       x=Quantity[PhysicalType('length')](value=f32[], unit=Unit("kpc"))
    )
    >>> qmr.x
    Quantity['length'](Array(0., dtype=float32), unit='kpc')

    >>> (q - r).x
    Quantity['length'](Array(0., dtype=float32), unit='kpc')

    """
    cart = other.represent_as(CartesianPosition1D)
    return jax.tree.map(qlax.sub, self, cart)


#####################################################################


@final
class CartesianVelocity1D(AvalMixin, AbstractVelocity1D):
    """Cartesian differential representation."""

    d_x: ct.BatchableSpeed = eqx.field(converter=Quantity["speed"].constructor)
    r"""X differential :math:`dx/dt \in (-\infty,+\infty`)`."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[CartesianPosition1D]:
        return CartesianPosition1D

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["CartesianAcceleration1D"]:
        return CartesianAcceleration1D

    @partial(jax.jit, inline=True)
    def norm(self, _: AbstractPosition1D | None = None, /) -> ct.BatchableSpeed:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> q = cx.CartesianVelocity1D.constructor([-1], "km/s")
        >>> q.norm()
        Quantity['speed'](Array(1, dtype=int32), unit='km / s')

        """
        return xp.abs(self.d_x)


# -------------------------------------------------------------------
# Method dispatches


@register(jax.lax.add_p)  # type: ignore[misc]
def _add_pp(
    lhs: CartesianVelocity1D, rhs: CartesianVelocity1D, /
) -> CartesianVelocity1D:
    """Add two Cartesian velocities.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> v = cx.CartesianVelocity1D.constructor([1], "km/s")
    >>> vec = xp.add(v, v)
    >>> vec
    CartesianVelocity1D(
       d_x=Quantity[...]( value=i32[], unit=Unit("km / s") )
    )
    >>> vec.d_x
    Quantity['speed'](Array(2, dtype=int32), unit='km / s')

    >>> (v + v).d_x
    Quantity['speed'](Array(2, dtype=int32), unit='km / s')

    """
    return jax.tree.map(qlax.add, lhs, rhs)


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_vcart(lhs: ArrayLike, rhs: CartesianVelocity1D, /) -> CartesianVelocity1D:
    """Scale a velocity by a scalar.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> v = cx.CartesianVelocity1D(d_x=Quantity(1, "m/s"))
    >>> vec = xp.multiply(2, v)
    >>> vec
    CartesianVelocity1D(
      d_x=Quantity[...]( value=i32[], unit=Unit("m / s") )
    )

    >>> vec.d_x
    Quantity['speed'](Array(2, dtype=int32, ...), unit='m / s')

    >>> (2 * v).d_x
    Quantity['speed'](Array(2, dtype=int32, ...), unit='m / s')

    """
    # Validation
    lhs = eqx.error_if(
        lhs, any(jax.numpy.shape(lhs)), f"must be a scalar, not {type(lhs)}"
    )

    # Scale the components
    return replace(rhs, d_x=lhs * rhs.d_x)


#####################################################################


@final
class CartesianAcceleration1D(AvalMixin, AbstractAcceleration1D):
    """Cartesian differential representation."""

    d2_x: ct.BatchableAcc = eqx.field(converter=Quantity["acceleration"].constructor)
    r"""X differential :math:`d^2x/dt^2 \in (-\infty,+\infty`)`."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[CartesianVelocity1D]:
        return CartesianVelocity1D

    # -----------------------------------------------------
    # Methods

    @partial(jax.jit, inline=True)
    def norm(self, _: AbstractPosition1D | None = None, /) -> ct.BatchableAcc:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> q = cx.CartesianAcceleration1D.constructor([-1], "km/s2")
        >>> q.norm()
        Quantity['acceleration'](Array(1, dtype=int32), unit='km / s2')

        """
        return xp.abs(self.d2_x)


@register(jax.lax.add_p)  # type: ignore[misc]
def _add_aa(
    lhs: CartesianAcceleration1D, rhs: CartesianAcceleration1D, /
) -> CartesianAcceleration1D:
    """Add two Cartesian accelerations.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> v = cx.CartesianAcceleration1D.constructor([1], "km/s2")
    >>> vec = xp.add(v, v)
    >>> vec
    CartesianAcceleration1D(
        d2_x=Quantity[...](value=i32[], unit=Unit("km / s2"))
    )
    >>> vec.d2_x
    Quantity['acceleration'](Array(2, dtype=int32), unit='km / s2')

    >>> (v + v).d2_x
    Quantity['acceleration'](Array(2, dtype=int32), unit='km / s2')

    """
    return jax.tree.map(qlax.add, lhs, rhs)


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_aq(lhs: ArrayLike, rhs: CartesianAcceleration1D, /) -> CartesianAcceleration1D:
    """Scale an acceleration by a scalar.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> v = cx.CartesianAcceleration1D(d2_x=Quantity(1, "m/s2"))
    >>> vec = xp.multiply(2, v)
    >>> vec
    CartesianAcceleration1D(
      d2_x=Quantity[...](value=i32[], unit=Unit("m / s2"))
    )

    >>> vec.d2_x
    Quantity['acceleration'](Array(2, dtype=int32, ...), unit='m / s2')

    >>> (2 * v).d2_x
    Quantity['acceleration'](Array(2, dtype=int32, ...), unit='m / s2')

    """
    # Validation
    lhs = eqx.error_if(
        lhs, any(jax.numpy.shape(lhs)), f"must be a scalar, not {type(lhs)}"
    )

    # Scale the components
    return replace(rhs, d2_x=lhs * rhs.d2_x)


@register(jax.lax.sub_p)  # type: ignore[misc]
def _sub_a1_a1(
    self: CartesianAcceleration1D, other: CartesianAcceleration1D, /
) -> CartesianAcceleration1D:
    """Subtract two 1-D cartesian accelerations.

    Examples
    --------
    >>> from quaxed import lax
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> v1 = cx.CartesianAcceleration1D(d2_x=Quantity(1, "m/s2"))
    >>> v2 = cx.CartesianAcceleration1D(d2_x=Quantity(2, "m/s2"))
    >>> vec = lax.sub(v1, v2)
    >>> vec
    CartesianAcceleration1D(
      d2_x=Quantity[...](value=i32[], unit=Unit("m / s2"))
    )

    >>> vec.d2_x
    Quantity['acceleration'](Array(-1, dtype=int32, ...), unit='m / s2')

    >>> (v1 - v2).d2_x
    Quantity['acceleration'](Array(-1, dtype=int32, ...), unit='m / s2')

    """
    return jax.tree.map(qlax.sub, self, other)
