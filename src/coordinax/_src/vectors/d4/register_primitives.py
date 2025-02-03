"""Built-in 4-vector classes."""

__all__: list[str] = []

from dataclasses import replace
from typing import cast

import jax
from quax import register

from .spacetime import FourVector
from coordinax._src.vectors.d3 import AbstractPos3D


@register(jax.lax.add_p)
def add_4v4v(self: FourVector, other: FourVector) -> FourVector:
    """Add two 4-vectors.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> w1 = cx.FourVector(t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
    >>> w2 = cx.FourVector(t=u.Quantity(2, "s"), q=u.Quantity([4, 5, 6], "m"))
    >>> w3 = w1 + w2
    >>> print(w3)
    <FourVector (t[s], q=(x[m], y[m], z[m]))
        [3 5 7 9]>

    """
    return replace(self, t=self.t + other.t, q=cast(AbstractPos3D, self.q + other.q))


@register(jax.lax.neg_p)
def neg_4v(self: FourVector) -> FourVector:
    """Negate the 4-vector.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> w = cx.FourVector(t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
    >>> print(-w)
    <FourVector (t[s], q=(x[m], y[m], z[m]))
        [-1 -1 -2 -3]>

    """
    return replace(self, t=-self.t, q=cast(AbstractPos3D, -self.q))


@register(jax.lax.sub_p)
def sub_4v_4v(lhs: FourVector, rhs: FourVector) -> FourVector:
    """Add two 4-vectors.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> w1 = cx.FourVector(t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
    >>> w2 = cx.FourVector(t=u.Quantity(2, "s"), q=u.Quantity([4, 5, 6], "m"))
    >>> w3 = w1 - w2
    >>> print(w3)
    <FourVector (t[s], q=(x[m], y[m], z[m]))
        [-1 -3 -3 -3]>

    """
    return replace(lhs, t=lhs.t - rhs.t, q=cast(AbstractPos3D, lhs.q - rhs.q))
