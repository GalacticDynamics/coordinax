"""Register primitives for Space."""

__all__: list[str] = []


from typing import Any

import jax
from quax import register

import quaxed.numpy as jnp
from dataclassish import replace

from .core import Space


@register(jax.lax.broadcast_in_dim_p)
def broadcast_in_dim_p(obj: Space, *, shape: tuple[int, ...], **kwargs: Any) -> Space:
    """Broadcast in a dimension."""
    batch = shape[:-1]
    return replace(
        obj,
        **{
            k: jnp.broadcast_to(v, (*batch, v.aval().shape[-1])) for k, v in obj.items()
        },
    )


@register(jax.lax.neg_p)
def neg_space(space: Space, /) -> Space:
    """Negative of the vector.

    Examples
    --------
    >>> import coordinax as cx

    >>> w = cx.Space(
    ...     length=cx.CartesianPos3D.from_([[[1, 2, 3], [4, 5, 6]]], "m"),
    ...     speed=cx.CartesianVel3D.from_([[[1, 2, 3], [4, 5, 6]]], "m/s")
    ... )

    >>> print(-w)
    Space({
       'length': <CartesianPos3D (x[m], y[m], z[m])
           [[[-1 -2 -3]
             [-4 -5 -6]]]>,
       'speed': <CartesianVel3D (x[m / s], y[m / s], z[m / s])
           [[[-1 -2 -3]
             [-4 -5 -6]]]>
    })

    """
    return type(space)(**{k: -v for k, v in space.items()})
