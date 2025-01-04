"""Register primitives for AbstractAcc."""

__all__: list[str] = []


import jax
from quax import register

import quaxed.numpy as jnp
import unxt as u
from dataclassish import field_items
from quaxed import lax as qlax

from .core import AbstractAcc
from coordinax._src.vectors.base_pos import AbstractPos
from coordinax._src.vectors.base_vel import AbstractVel


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_acc_time(lhs: AbstractAcc, rhs: u.Quantity["time"]) -> AbstractVel:
    """Multiply the vector by a :class:`unxt.Quantity`.

    Examples
    --------
    >>> from quaxed import lax
    >>> import unxt as u
    >>> import coordinax as cx

    >>> d2r = cx.vecs.RadialAcc(u.Quantity(1, "m/s2"))
    >>> vec = lax.mul(d2r, u.Quantity(2, "s"))
    >>> vec
    RadialVel( d_r=Quantity[...]( value=...i32[], unit=Unit("m / s") ) )
    >>> vec.d_r
    Quantity['speed'](Array(2, dtype=int32, ...), unit='m / s')

    >>> (d2r * u.Quantity(2, "s")).d_r
    Quantity['speed'](Array(2, dtype=int32, ...), unit='m / s')

    """
    # TODO: better access to corresponding fields
    return lhs.integral_cls.from_(
        {k.replace("2", ""): jnp.multiply(v, rhs) for k, v in field_items(lhs)}
    )


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_time_acc(lhs: u.Quantity["time"], rhs: AbstractAcc) -> AbstractVel:
    """Multiply a scalar by an acceleration.

    Examples
    --------
    >>> from quaxed import lax
    >>> import unxt as u
    >>> import coordinax as cx

    >>> d2r = cx.vecs.RadialAcc(u.Quantity(1, "m/s2"))
    >>> vec = lax.mul(u.Quantity(2, "s"), d2r)
    >>> vec
    RadialVel( d_r=Quantity[...]( value=...i32[], unit=Unit("m / s") ) )
    >>> vec.d_r
    Quantity['speed'](Array(2, dtype=int32, ...), unit='m / s')

    """
    return qlax.mul(rhs, lhs)  # pylint: disable=arguments-out-of-order


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_acc_time2(lhs: AbstractAcc, rhs: u.Quantity["s2"]) -> AbstractPos:
    """Multiply an acceleration by a scalar.

    Examples
    --------
    >>> from quaxed import lax
    >>> import unxt as u
    >>> import coordinax as cx

    >>> d2r = cx.vecs.RadialAcc(u.Quantity(1, "m/s2"))
    >>> vec = lax.mul(d2r, u.Quantity(2, "s2"))
    >>> print(vec)
    <RadialPos (r[m])
        [2]>

    >>> print(d2r * u.Quantity(2, "s2"))
    <RadialPos (r[m])
        [2]>

    """
    # TODO: better access to corresponding fields
    return lhs.integral_cls.integral_cls.from_(
        {k.replace("d2_", ""): v * rhs for k, v in field_items(lhs)}
    )


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_time2_acc(lhs: u.Quantity["s2"], rhs: AbstractAcc) -> AbstractPos:
    """Multiply a scalar by an acceleration.

    Examples
    --------
    >>> from quaxed import lax
    >>> import unxt as u
    >>> import coordinax as cx

    >>> d2r = cx.vecs.RadialAcc(u.Quantity(1, "m/s2"))
    >>> vec = lax.mul(u.Quantity(2, "s2"), d2r)
    >>> print(vec)
    <RadialPos (r[m])
        [2]>

    """
    return qlax.mul(rhs, lhs)  # pylint: disable=arguments-out-of-order
