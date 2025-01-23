"""Register primitives for AbstractVel."""

__all__: list[str] = []

from typing import cast

import jax
from quax import register

import quaxed.numpy as jnp
import unxt as u
from dataclassish import field_items

from .core import AbstractVel
from coordinax._src.vectors.base_pos import AbstractPos

# ---------------------------------------------------------


@register(jax.lax.mul_p)
def _mul_vel_q(self: AbstractVel, other: u.Quantity["time"]) -> AbstractPos:
    """Multiply the vector by a time :class:`unxt.Quantity` to get a position.

    Examples
    --------
    >>> import quaxed.lax as qlax
    >>> import unxt as u
    >>> import coordinax as cx

    >>> dr = cx.vecs.RadialVel(u.Quantity(1, "m/s"))
    >>> vec = dr * u.Quantity(2, "s")
    >>> print(vec)
    <RadialPos (r[m])
        [2]>

    >>> print(qlax.mul(dr, u.Quantity(2, "s")))
    <RadialPos (r[m])
        [2]>

    """
    fs = {k[2:]: v * other for k, v in field_items(self)}
    return cast(AbstractPos, self.integral_cls.from_(fs))


# -----------------------------------------------


@register(jax.lax.neg_p)
def neg_vel(vec: AbstractVel, /) -> AbstractVel:
    """Negate the vector.

    Examples
    --------
    >>> from quaxed import lax
    >>> import unxt as u
    >>> import coordinax as cx

    >>> dr = cx.vecs.RadialVel.from_([1], "m/s")
    >>> print(lax.neg(dr))
    <RadialVel (d_r[m / s])
        [-1]>

    >>> -dr
    RadialVel( d_r=Quantity[...]( value=i32[], unit=Unit("m / s") ) )

    >>> dp = cx.vecs.PolarVel(u.Quantity(1, "m/s"), u.Quantity(1, "mas/yr"))
    >>> neg_dp = -dp
    >>> print(neg_dp)
    <PolarVel (d_r[m / s], d_phi[mas / yr])
        [-1 -1]>

    """
    return jax.tree.map(jnp.negative, vec)
