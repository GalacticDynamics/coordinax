"""Register primitives for AbstractAcc."""

__all__: list[str] = []


from typing import cast

import jax
from quax import register

import quaxed.numpy as jnp
import unxt as u
from dataclassish import field_items
from quaxed import lax as qlax

from .core import AbstractAcc
from coordinax._src.vectors.base_pos import AbstractPos
from coordinax._src.vectors.base_vel import AbstractVel

# -----------------------------------------------


@register(jax.lax.mul_p)
def mul_p_acc_time(lhs: AbstractAcc, rhs: u.Quantity["time"], /) -> AbstractVel:
    """Multiply the vector by a `unxt.Quantity`.

    Examples
    --------
    >>> from quaxed import lax
    >>> import unxt as u
    >>> import coordinax as cx

    >>> d2r = cx.vecs.RadialAcc(u.Quantity(1, "m/s2"))
    >>> vec = lax.mul(d2r, u.Quantity(2, "s"))
    >>> print(vec)
    <RadialVel: (r) [m / s]
        [2]>

    """
    fs = {k: jnp.multiply(v, rhs) for k, v in field_items(lhs)}
    return cast(AbstractVel, lhs.time_antiderivative_cls.from_(fs))


@register(jax.lax.mul_p)
def mul_p_time_acc(lhs: u.Quantity["time"], rhs: AbstractAcc, /) -> AbstractVel:
    """Multiply a scalar by an acceleration.

    Examples
    --------
    >>> from quaxed import lax
    >>> import unxt as u
    >>> import coordinax as cx

    >>> d2r = cx.vecs.RadialAcc(u.Quantity(1, "m/s2"))
    >>> vec = lax.mul(u.Quantity(2, "s"), d2r)
    >>> print(vec)
    <RadialVel: (r) [m / s]
        [2]>

    """
    return cast(AbstractVel, qlax.mul(rhs, lhs))  # type: ignore[arg-type]  # pylint: disable=arguments-out-of-order


@register(jax.lax.mul_p)
def mul_p_acc_time2(lhs: AbstractAcc, rhs: u.Quantity["s2"], /) -> AbstractPos:
    """Multiply an acceleration by a scalar.

    Examples
    --------
    >>> from quaxed import lax
    >>> import unxt as u
    >>> import coordinax as cx

    >>> d2r = cx.vecs.RadialAcc(u.Quantity(1, "m/s2"))
    >>> vec = lax.mul(d2r, u.Quantity(2, "s2"))
    >>> print(vec)
    <RadialPos: (r) [m]
        [2]>

    >>> print(d2r * u.Quantity(2, "s2"))
    <RadialPos: (r) [m]
        [2]>

    """
    pos_cls = lhs.time_nth_derivative_cls(-2)
    fs = {k: v * rhs for k, v in field_items(lhs)}
    return cast(AbstractPos, pos_cls.from_(fs))


@register(jax.lax.mul_p)
def mul_p_time2_acc(lhs: u.Quantity["s2"], rhs: AbstractAcc, /) -> AbstractPos:
    """Multiply a scalar by an acceleration.

    Examples
    --------
    >>> from quaxed import lax
    >>> import unxt as u
    >>> import coordinax as cx

    >>> d2r = cx.vecs.RadialAcc(u.Quantity(1, "m/s2"))
    >>> vec = lax.mul(u.Quantity(2, "s2"), d2r)
    >>> print(vec)
    <RadialPos: (r) [m]
        [2]>

    """
    return qlax.mul(rhs, lhs)  # type: ignore[arg-type,return-value]  # pylint: disable=arguments-out-of-order


# -----------------------------------------------


@register(jax.lax.neg_p)
def neg_p_acc(vec: AbstractAcc, /) -> AbstractAcc:
    """Negate the vector.

    Examples
    --------
    >>> from quaxed import lax
    >>> import unxt as u
    >>> import coordinax as cx

    >>> d2r = cx.vecs.RadialAcc(u.Quantity(1, "m/s2"))
    >>> vec = lax.neg(d2r)
    >>> print(vec)
    <RadialAcc: (r) [m / s2]
        [-1]>

    """
    return jax.tree.map(jnp.negative, vec)
