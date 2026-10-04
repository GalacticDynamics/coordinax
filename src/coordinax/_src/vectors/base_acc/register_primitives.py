"""Register primitives for AbstractAcc."""

__all__: tuple[str, ...] = ()


from typing import Any, cast

import jax
import quax
from quax import register

import quaxed.numpy as jnp
import unxt as u
from dataclassish import field_items

from .core import AbstractAcc
from coordinax._src.vectors.base import AbstractVector
from coordinax._src.vectors.base_pos import AbstractPos
from coordinax._src.vectors.base_vel import AbstractVel

mul_p_qbind = quax.quaxify(jax.lax.mul_p.bind)

# -----------------------------------------------


@register(jax.lax.mul_p)
def mul_p_acc_q(lhs: AbstractAcc, rhs: u.Q, /, **kw: Any) -> AbstractVel | AbstractPos:
    """Multiply an acceleration by a time or time-squared `unxt.Quantity`.

    Examples
    --------
    >>> from quaxed import lax
    >>> import unxt as u
    >>> import coordinax as cx

    >>> d2r = cx.vecs.RadialAcc(u.Q(1, "m/s2"))
    >>> print(lax.mul(d2r, u.Q(2, "s")))
    <RadialVel: (r) [m / s]
        [2]>

    >>> print(lax.mul(d2r, u.Q(2, "s2")))
    <RadialPos: (r) [m]
        [2]>

    >>> print(d2r * u.Q(2, "s2"))
    <RadialPos: (r) [m]
        [2]>

    Any other unit is an error:

    >>> try: d2r * u.Q(2, "m")
    ... except ValueError as e: print(e)
    Cannot multiply RadialAcc by a quantity in m.

    """
    # One rule branching on the unit, not one rule per dimension: quax caches
    # the rule by argument *type*, and every dimension is `unxt.Quantity`.
    # https://github.com/nstarman/quax/issues/257
    out_cls: type[AbstractVector]
    if u.is_unit_convertible("s", rhs):
        out_cls = lhs.time_antiderivative_cls
    elif u.is_unit_convertible("s2", rhs):
        out_cls = lhs.time_nth_derivative_cls(-2)
    else:
        msg = f"Cannot multiply {type(lhs).__name__} by a quantity in {rhs.unit}."
        raise ValueError(msg)
    fs = {k: mul_p_qbind(v, rhs, **kw) for k, v in field_items(lhs)}
    return cast("AbstractVel | AbstractPos", out_cls.from_(fs))


@register(jax.lax.mul_p)
def mul_p_q_acc(lhs: u.Q, rhs: AbstractAcc, /, **kw: Any) -> AbstractVel | AbstractPos:
    """Multiply a time or time-squared `unxt.Quantity` by an acceleration.

    Examples
    --------
    >>> from quaxed import lax
    >>> import unxt as u
    >>> import coordinax as cx

    >>> d2r = cx.vecs.RadialAcc(u.Q(1, "m/s2"))
    >>> print(lax.mul(u.Q(2, "s"), d2r))
    <RadialVel: (r) [m / s]
        [2]>

    >>> print(lax.mul(u.Q(2, "s2"), d2r))
    <RadialPos: (r) [m]
        [2]>

    """
    return mul_p_acc_q(rhs, lhs, **kw)  # pylint: disable=arguments-out-of-order


# -----------------------------------------------


@register(jax.lax.neg_p)
def neg_p_acc(vec: AbstractAcc, /) -> AbstractAcc:
    """Negate the vector.

    Examples
    --------
    >>> from quaxed import lax
    >>> import unxt as u
    >>> import coordinax as cx

    >>> d2r = cx.vecs.RadialAcc(u.Q(1, "m/s2"))
    >>> vec = lax.neg(d2r)
    >>> print(vec)
    <RadialAcc: (r) [m / s2]
        [-1]>

    """
    return jax.tree.map(jnp.negative, vec)
