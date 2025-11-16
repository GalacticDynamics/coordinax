"""Register primitives."""

__all__: tuple[str, ...] = ()


import jax
from quax import register

from dataclassish import replace

from .coordinate import Coordinate
from coordinax._src.vectors.base_pos.core import AbstractPos


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


@register(jax.lax.add_p)
def add_p_coord_pos(x: Coordinate, y: AbstractPos, /) -> Coordinate:
    r"""Add a position vector to a coordinate.

    To understand this operation, let's consider a phase-space point $(x, v) \in
    \mathbb{R}^3\times\mathbb{R}^3$ consisting of a position and a velocity. A
    pure spatial translation is the map $T_{\Delta x} : (x,v) \mapsto (x+\Delta
    x,\ v)$, i.e. only the position is shifted; velocity is unchanged.

    """
    # Get the Cartesian class for the coordinate's position
    cart_cls = y.cartesian_type
    # Convert the coordinate to that class. This changes the position, but also
    # the other components, e.g. the velocity.
    data = dict(x.data.vconvert(cart_cls))
    # Now add the position vector to the position component only
    data = replace(data, length=data["length"] + y)
    # Transform back to the original vector types
    # data.vconvert()  # TODO: all original types
    # Reconstruct the Coordinate
    return Coordinate(data, frame=x.frame)
