"""Register primitives."""

__all__: list[str] = []


import jax
from quax import register

import quaxed.numpy as jnp

from .generic import CartesianGeneric3D


@register(jax.lax.neg_p)
def neg_genericcart3d(obj: CartesianGeneric3D, /) -> CartesianGeneric3D:
    """Negate the `coordinax.vecs.CartesianGeneric3D`.

    Examples
    --------
    >>> import coordinax as cx
    >>> q = cx.vecs.CartesianGeneric3D.from_([1, 2, 3], "km")
    >>> print(-q)
    <CartesianGeneric3D (x[km], y[km], z[km])
    [-1 -2 -3]>

    """
    return jax.tree.map(jnp.negative, obj)
