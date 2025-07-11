"""Register primitives."""

__all__: list[str] = []


import jax
from quax import register

from dataclassish import replace

from .coordinate import Coordinate


@register(jax.lax.neg_p)
def neg_p_coord(x: Coordinate, /) -> Coordinate:
    """Negate a coordinate.

    Examples
    --------
    >>> import coordinax as cx

    >>> data = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> coord = cx.Coordinate(data, cx.frames.ICRS())

    >>> print(-coord)
    Coordinate(
        {
           'length': <CartesianPos3D: (x, y, z) [kpc]
               [-1 -2 -3]>
        },
        frame=ICRS()
    )

    """
    return replace(x, data=-x.data)
