"""Register primitives for Space."""

__all__: list[str] = []


from typing import Any

import jax
from quax import register

import quaxed.numpy as jnp
from dataclassish import replace

from .core import Space
from coordinax._src.custom_types import Shape


@register(jax.lax.broadcast_in_dim_p)
def broadcast_in_dim_p_space(obj: Space, /, *, shape: Shape, **kw: Any) -> Space:
    """Broadcast in a dimension."""
    batch = shape[:-1]
    return replace(
        obj,
        **{
            k: jnp.broadcast_to(v, (*batch, v.aval().shape[-1]))  # type: ignore[arg-type]
            for k, v in obj.items()
        },
    )


@register(jax.lax.neg_p)
def neg_p_space(space: Space, /) -> Space:
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
       'length': <CartesianPos3D: (x, y, z) [m]
           [[[-1 -2 -3]
             [-4 -5 -6]]]>,
       'speed': <CartesianVel3D: (x, y, z) [m / s]
           [[[-1 -2 -3]
             [-4 -5 -6]]]>
    })

    """
    return type(space)(**{k: -v for k, v in space.items()})
