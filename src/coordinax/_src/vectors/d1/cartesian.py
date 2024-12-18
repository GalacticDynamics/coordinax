"""Carteisan vector."""

__all__ = ["CartesianAcc1D", "CartesianPos1D", "CartesianVel1D"]

from dataclasses import replace
from functools import partial
from typing import final
from typing_extensions import override

import equinox as eqx
import jax
from jaxtyping import ArrayLike
from quax import register

import quaxed.numpy as jnp
import unxt as u
from quaxed import lax as qlax

import coordinax._src.typing as ct
from .base import AbstractAcc1D, AbstractPos1D, AbstractVel1D
from coordinax._src.distances import BatchableLength
from coordinax._src.utils import classproperty
from coordinax._src.vectors.base import AbstractPos
from coordinax._src.vectors.base.mixins import AvalMixin


@final
class CartesianPos1D(AbstractPos1D):
    """Cartesian vector representation.

    Examples
    --------
    >>> import coordinax as cx

    >>> vec = cx.vecs.CartesianPos1D.from_([2], "m")
    >>> vec
    CartesianPos1D(x=Quantity[...](value=i32[], unit=Unit("m")))

    Vectors support the basic math operations:

    >>> (vec + vec).x
    Quantity['length'](Array(4, dtype=int32), unit='m')

    >>> (vec - vec).x
    Quantity['length'](Array(0, dtype=int32), unit='m')

    >>> (3 * vec).x
    Quantity['length'](Array(6, dtype=int32), unit='m')

    """

    x: BatchableLength = eqx.field(converter=u.Quantity["length"].from_)
    r"""X coordinate :math:`x \in (-\infty,+\infty)`."""

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["CartesianVel1D"]:
        return CartesianVel1D


# -------------------------------------------------------------------
# Method dispatches


@register(jax.lax.add_p)  # type: ignore[misc]
def _add_qq(lhs: CartesianPos1D, rhs: AbstractPos, /) -> CartesianPos1D:
    """Add a vector to a CartesianPos1D.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    >>> q = cx.vecs.CartesianPos1D.from_([1], "km")
    >>> r = cx.vecs.RadialPos.from_([1], "km")

    >>> print(jnp.add(q, r))
    <CartesianPos1D (x[km])
        [2]>

    >>> print(q + r)
    <CartesianPos1D (x[km])
        [2]>

    """
    rhs = rhs.vconvert(CartesianPos1D)
    return jax.tree.map(jnp.add, lhs, rhs)


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_ac1(lhs: ArrayLike, rhs: CartesianPos1D, /) -> CartesianPos1D:
    """Scale a position by a scalar.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    >>> v = cx.vecs.CartesianPos1D.from_(1, "m")
    >>> jnp.multiply(2, v).x
    Quantity['length'](Array(2, dtype=int32, ...), unit='m')

    >>> (2 * v).x
    Quantity['length'](Array(2, dtype=int32, ...), unit='m')

    """
    # Validation
    lhs = eqx.error_if(
        lhs, any(jax.numpy.shape(lhs)), f"must be a scalar, not {type(lhs)}"
    )

    # Scale the components
    return replace(rhs, x=lhs * rhs.x)


@register(jax.lax.neg_p)  # type: ignore[misc]
def _neg_p_cart1d_pos(obj: CartesianPos1D, /) -> CartesianPos1D:
    """Negate the `coordinax.CartesianPos1D`.

    Examples
    --------
    >>> import coordinax as cx
    >>> q = cx.vecs.CartesianPos1D.from_([1], "km")
    >>> (-q).x
    Quantity['length'](Array(-1, dtype=int32), unit='km')

    """
    return jax.tree.map(qlax.neg, obj)


@register(jax.lax.sub_p)  # type: ignore[misc]
def _sub_q1d_pos(self: CartesianPos1D, other: AbstractPos, /) -> CartesianPos1D:
    """Subtract two vectors.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    >>> q = cx.vecs.CartesianPos1D.from_([1], "km")
    >>> r = cx.vecs.RadialPos.from_([1], "km")

    >>> print(jnp.subtract(q, r))
    <CartesianPos1D (x[km])
        [0]>

    >>> print(q - r)
    <CartesianPos1D (x[km])
        [0]>

    """
    cart = other.vconvert(CartesianPos1D)
    return jax.tree.map(jnp.subtract, self, cart)


#####################################################################


@final
class CartesianVel1D(AvalMixin, AbstractVel1D):
    """Cartesian differential representation."""

    d_x: ct.BatchableSpeed = eqx.field(converter=u.Quantity["speed"].from_)
    r"""X differential :math:`dx/dt \in (-\infty,+\infty`)`."""

    @override
    @classproperty
    @classmethod
    def integral_cls(cls) -> type[CartesianPos1D]:
        return CartesianPos1D

    @override
    @classproperty
    @classmethod
    def differential_cls(cls) -> type["CartesianAcc1D"]:
        return CartesianAcc1D

    @override
    @partial(eqx.filter_jit, inline=True)
    def norm(self, _: AbstractPos1D | None = None, /) -> ct.BatchableSpeed:
        """Return the norm of the vector.

        Examples
        --------
        >>> import coordinax as cx
        >>> q = cx.vecs.CartesianVel1D.from_([-1], "km/s")
        >>> q.norm()
        Quantity['speed'](Array(1, dtype=int32), unit='km / s')

        """
        return jnp.abs(self.d_x)


# -------------------------------------------------------------------
# Method dispatches


@register(jax.lax.add_p)  # type: ignore[misc]
def _add_pp(lhs: CartesianVel1D, rhs: CartesianVel1D, /) -> CartesianVel1D:
    """Add two Cartesian velocities.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    >>> v = cx.vecs.CartesianVel1D.from_([1], "km/s")
    >>> vec = jnp.add(v, v)
    >>> vec
    CartesianVel1D(
       d_x=Quantity[...]( value=i32[], unit=Unit("km / s") )
    )
    >>> vec.d_x
    Quantity['speed'](Array(2, dtype=int32), unit='km / s')

    >>> (v + v).d_x
    Quantity['speed'](Array(2, dtype=int32), unit='km / s')

    """
    return jax.tree.map(jnp.add, lhs, rhs)


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_vcart(lhs: ArrayLike, rhs: CartesianVel1D, /) -> CartesianVel1D:
    """Scale a velocity by a scalar.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    >>> v = cx.vecs.CartesianVel1D.from_(1, "m/s")
    >>> vec = jnp.multiply(2, v)
    >>> vec
    CartesianVel1D(
      d_x=Quantity[...]( value=...i32[], unit=Unit("m / s") )
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
class CartesianAcc1D(AvalMixin, AbstractAcc1D):
    """Cartesian differential representation."""

    d2_x: ct.BatchableAcc = eqx.field(converter=u.Quantity["acceleration"].from_)
    r"""X differential :math:`d^2x/dt^2 \in (-\infty,+\infty`)`."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[CartesianVel1D]:
        return CartesianVel1D

    # -----------------------------------------------------
    # Methods

    @override
    @partial(eqx.filter_jit, inline=True)
    def norm(self, _: AbstractPos1D | None = None, /) -> ct.BatchableAcc:
        """Return the norm of the vector.

        Examples
        --------
        >>> import coordinax as cx
        >>> q = cx.vecs.CartesianAcc1D.from_([-1], "km/s2")
        >>> q.norm()
        Quantity['acceleration'](Array(1, dtype=int32), unit='km / s2')

        """
        return jnp.abs(self.d2_x)


@register(jax.lax.add_p)  # type: ignore[misc]
def _add_aa(lhs: CartesianAcc1D, rhs: CartesianAcc1D, /) -> CartesianAcc1D:
    """Add two Cartesian accelerations.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    >>> v = cx.vecs.CartesianAcc1D.from_([1], "km/s2")
    >>> vec = jnp.add(v, v)
    >>> vec
    CartesianAcc1D(
        d2_x=Quantity[...](value=i32[], unit=Unit("km / s2"))
    )
    >>> vec.d2_x
    Quantity['acceleration'](Array(2, dtype=int32), unit='km / s2')

    >>> (v + v).d2_x
    Quantity['acceleration'](Array(2, dtype=int32), unit='km / s2')

    """
    return jax.tree.map(jnp.add, lhs, rhs)


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_aq(lhs: ArrayLike, rhs: CartesianAcc1D, /) -> CartesianAcc1D:
    """Scale an acceleration by a scalar.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    >>> v = cx.vecs.CartesianAcc1D.from_(1, "m/s2")
    >>> vec = jnp.multiply(2, v)
    >>> vec
    CartesianAcc1D( d2_x=... )

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
def _sub_a1_a1(self: CartesianAcc1D, other: CartesianAcc1D, /) -> CartesianAcc1D:
    """Subtract two 1-D cartesian accelerations.

    Examples
    --------
    >>> from quaxed import lax
    >>> import coordinax as cx

    >>> v1 = cx.vecs.CartesianAcc1D.from_(1, "m/s2")
    >>> v2 = cx.vecs.CartesianAcc1D.from_(2, "m/s2")
    >>> vec = lax.sub(v1, v2)
    >>> vec
    CartesianAcc1D( d2_x=... )

    >>> vec.d2_x
    Quantity['acceleration'](Array(-1, dtype=int32, ...), unit='m / s2')

    >>> (v1 - v2).d2_x
    Quantity['acceleration'](Array(-1, dtype=int32, ...), unit='m / s2')

    """
    return jax.tree.map(jnp.subtract, self, other)
