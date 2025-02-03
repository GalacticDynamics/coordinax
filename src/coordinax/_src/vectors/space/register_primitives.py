"""Register primitives for Space."""

__all__: list[str] = []


import jax
from quax import register

from .core import Space


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
       'speed': <CartesianVel3D (d_x[m / s], d_y[m / s], d_z[m / s])
           [[[-1 -2 -3]
             [-4 -5 -6]]]>
    })

    """
    return type(space)(**{k: -v for k, v in space.items()})
