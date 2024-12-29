"""Built-in vector classes."""

__all__ = [
    "CartesianAcc2D",
    "CartesianPos2D",
    "CartesianVel2D",
]

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
from unxt.quantity import Quantity

import coordinax._src.typing as ct
from .base import AbstractAcc2D, AbstractPos2D, AbstractVel2D
from coordinax._src.distances import BatchableLength
from coordinax._src.utils import classproperty
from coordinax._src.vectors.base import AbstractPos
from coordinax._src.vectors.base.mixins import AvalMixin


@final
class CartesianPos2D(AbstractPos2D):
    """Cartesian 2D Position.

    Examples
    --------
    >>> import coordinax as cx

    >>> vec = cx.vecs.CartesianPos2D.from_([1, 2], "m")
    >>> print(vec)
    <CartesianPos2D (x[m], y[m])
        [1 2]>

    """

    x: BatchableLength = eqx.field(converter=u.Quantity["length"].from_)
    r"""X coordinate :math:`x \in (-\infty,+\infty)`."""

    y: BatchableLength = eqx.field(converter=u.Quantity["length"].from_)
    r"""Y coordinate :math:`y \in (-\infty,+\infty)`."""

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["CartesianVel2D"]:
        return CartesianVel2D


# -----------------------------------------------------


@register(jax.lax.add_p)  # type: ignore[misc]
def _add_cart2d_pos(lhs: CartesianPos2D, rhs: AbstractPos, /) -> CartesianPos2D:
    """Add two vectors.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> cart = cx.vecs.CartesianPos2D.from_([1, 2], "km")
    >>> polr = cx.vecs.PolarPos(r=u.Quantity(3, "km"), phi=u.Quantity(90, "deg"))
    >>> print(cart + polr)
    <CartesianPos2D (x[km], y[km])
        [1. 5.]>

    >>> print(jnp.add(cart, polr))
    <CartesianPos2D (x[km], y[km])
        [1. 5.]>

    """
    cart = rhs.vconvert(CartesianPos2D)
    return jax.tree.map(jnp.add, lhs, cart)


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_v_cart2d(lhs: ArrayLike, rhs: CartesianPos2D, /) -> CartesianPos2D:
    """Scale a cartesian 2D position by a scalar.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    >>> v = cx.vecs.CartesianPos2D.from_([3, 4], "m")
    >>> jnp.multiply(5, v).x
    Quantity['length'](Array(15, dtype=int32), unit='m')

    """
    # Validation
    lhs = eqx.error_if(
        lhs, any(jax.numpy.shape(lhs)), f"must be a scalar, not {type(lhs)}"
    )

    # Scale the components
    return replace(rhs, x=lhs * rhs.x, y=lhs * rhs.y)


@register(jax.lax.neg_p)  # type: ignore[misc]
def _neg_p_cart2d_pos(obj: CartesianPos2D, /) -> CartesianPos2D:
    """Negate the `coordinax.vecs.CartesianPos2D`.

    Examples
    --------
    >>> import coordinax as cx
    >>> q = cx.vecs.CartesianPos2D.from_([1, 2], "km")
    >>> (-q).x
    Quantity['length'](Array(-1, dtype=int32), unit='km')

    """
    return jax.tree.map(qlax.neg, obj)


@register(jax.lax.sub_p)  # type: ignore[misc]
def _sub_cart2d_pos2d(lhs: CartesianPos2D, rhs: AbstractPos, /) -> CartesianPos2D:
    """Subtract two vectors.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> cart = cx.vecs.CartesianPos2D.from_([1, 2], "km")
    >>> polr = cx.vecs.PolarPos(r=u.Quantity(3, "km"), phi=u.Quantity(90, "deg"))

    >>> print(cart - polr)
    <CartesianPos2D (x[km], y[km])
        [ 1. -1.]>

    """
    cart = rhs.vconvert(CartesianPos2D)
    return jax.tree.map(jnp.subtract, lhs, cart)


#####################################################################


@final
class CartesianVel2D(AvalMixin, AbstractVel2D):
    """Cartesian 2D Velocity.

    Examples
    --------
    >>> import coordinax as cx

    >>> vec = cx.vecs.CartesianVel2D.from_([1, 2], "m/s")
    >>> print(vec)
    <CartesianVel2D (d_x[m / s], d_y[m / s])
        [1 2]>

    """

    d_x: ct.BatchableSpeed = eqx.field(converter=u.Quantity["speed"].from_)
    r"""X coordinate differential :math:`\dot{x} \in (-\infty,+\infty)`."""

    d_y: ct.BatchableSpeed = eqx.field(converter=u.Quantity["speed"].from_)
    r"""Y coordinate differential :math:`\dot{y} \in (-\infty,+\infty)`."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[CartesianPos2D]:
        return CartesianPos2D

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["CartesianAcc2D"]:
        """Return the differential class.

        Examples
        --------
        >>> import coordinax as cx
        >>> print(cx.vecs.CartesianVel2D.differential_cls)
        <class 'coordinax...CartesianAcc2D'>

        """
        return CartesianAcc2D


# -----------------------------------------------------


@register(jax.lax.add_p)  # type: ignore[misc]
def _add_pp(lhs: CartesianVel2D, rhs: CartesianVel2D, /) -> CartesianVel2D:
    """Add two Cartesian velocities.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    >>> v = cx.vecs.CartesianVel2D.from_([1, 2], "km/s")
    >>> print(v + v)
    <CartesianVel2D (d_x[km / s], d_y[km / s])
        [2 4]>

    >>> print(jnp.add(v, v))
    <CartesianVel2D (d_x[km / s], d_y[km / s])
        [2 4]>

    """
    return jax.tree.map(qlax.add, lhs, rhs)


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_vp(lhs: ArrayLike, rhts: CartesianVel2D, /) -> CartesianVel2D:
    """Scale a cartesian 2D velocity by a scalar.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    >>> v = cx.vecs.CartesianVel2D.from_([3, 4], "m/s")
    >>> print(5 * v)
    <CartesianVel2D (d_x[m / s], d_y[m / s])
        [15 20]>

    >>> print(jnp.multiply(5, v))
    <CartesianVel2D (d_x[m / s], d_y[m / s])
        [15 20]>

    """
    # Validation
    lhs = eqx.error_if(
        lhs, any(jax.numpy.shape(lhs)), f"must be a scalar, not {type(lhs)}"
    )

    # Scale the components
    return replace(rhts, d_x=lhs * rhts.d_x, d_y=lhs * rhts.d_y)


#####################################################################


@final
class CartesianAcc2D(AvalMixin, AbstractAcc2D):
    """Cartesian Acceleration 3D.

    Examples
    --------
    >>> import coordinax as cx

    >>> vec = cx.vecs.CartesianAcc2D.from_([1, 2], "m/s2")
    >>> print(vec)
    <CartesianAcc2D (d2_x[m / s2], d2_y[m / s2])
        [1 2]>

    """

    d2_x: ct.BatchableAcc = eqx.field(converter=Quantity["acceleration"].from_)
    r"""X coordinate acceleration :math:`\frac{d^2 x}{dt^2} \in (-\infty,+\infty)`."""

    d2_y: ct.BatchableAcc = eqx.field(converter=Quantity["acceleration"].from_)
    r"""Y coordinate acceleration :math:`\frac{d^2 y}{dt^2} \in (-\infty,+\infty)`."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[CartesianVel2D]:
        return CartesianVel2D

    # -----------------------------------------------------

    @override
    @partial(eqx.filter_jit, inline=True)
    def norm(self, _: AbstractVel2D | None = None, /) -> ct.BatchableAcc:
        """Return the norm of the vector.

        Examples
        --------
        >>> import coordinax as cx
        >>> v = cx.vecs.CartesianAcc2D.from_([3, 4], "km/s2")
        >>> v.norm()
        Quantity['acceleration'](Array(5., dtype=float32), unit='km / s2')

        """
        return jnp.sqrt(self.d2_x**2 + self.d2_y**2)


# -----------------------------------------------------


@register(jax.lax.add_p)  # type: ignore[misc]
def _add_aa(lhs: CartesianAcc2D, rhs: CartesianAcc2D, /) -> CartesianAcc2D:
    """Add two Cartesian accelerations.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    >>> v = cx.vecs.CartesianAcc2D.from_([3, 4], "km/s2")
    >>> print(v + v)
    <CartesianAcc2D (d2_x[km / s2], d2_y[km / s2])
        [6 8]>

    >>> print(jnp.add(v, v))
    <CartesianAcc2D (d2_x[km / s2], d2_y[km / s2])
        [6 8]>

    """
    return jax.tree.map(jnp.add, lhs, rhs)


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_va(lhs: ArrayLike, rhts: CartesianAcc2D, /) -> CartesianAcc2D:
    """Scale a cartesian 2D acceleration by a scalar.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    >>> v = cx.vecs.CartesianAcc2D.from_([3, 4], "m/s2")
    >>> jnp.multiply(5, v).d2_x
    Quantity['acceleration'](Array(15, dtype=int32), unit='m / s2')

    >>> (5 * v).d2_x
    Quantity['acceleration'](Array(15, dtype=int32), unit='m / s2')

    """
    # Validation
    lhs = eqx.error_if(
        lhs, any(jax.numpy.shape(lhs)), f"must be a scalar, not {type(lhs)}"
    )

    # Scale the components
    return replace(rhts, d2_x=lhs * rhts.d2_x, d2_y=lhs * rhts.d2_y)
