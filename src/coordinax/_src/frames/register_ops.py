"""Register for `coordinax.ops`."""

__all__: list[str] = []


from dataclassish import replace

from .coordinate import Coordinate
from coordinax._src.operators import AbstractOperator


@AbstractOperator.__call__.dispatch  # type: ignore[misc]
def call(self: AbstractOperator, x: Coordinate, /) -> Coordinate:
    """Apply the operator to a coordinate.

    Examples
    --------
    >>> import coordinax as cx

    >>> coord = cx.Coordinate(cx.CartesianPos3D.from_([1, 2, 3], "kpc"),
    ...                       cx.frames.ICRS())
    >>> coord
    Coordinate(
        KinematicSpace({ 'length': CartesianPos3D( ... ) }),
        frame=ICRS()
    )

    >>> op = cx.ops.GalileanSpatialTranslation.from_([-1, -1, -1], "kpc")

    >>> new_coord = op(coord)
    >>> print(new_coord.data["length"])
    <CartesianPos3D: (x, y, z) [kpc]
        [0 1 2]>

    """
    # TODO: take the frame into account
    return replace(x, data=self(x.data))
