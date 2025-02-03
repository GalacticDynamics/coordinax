"""Register for `coordinax.ops`."""

__all__: list[str] = []


from dataclassish import replace

from .coordinate import Coordinate
from coordinax._src.operators import AbstractOperator


@AbstractOperator.__call__.dispatch  # type: ignore[misc]
def call(self: AbstractOperator, x: Coordinate, /) -> Coordinate:
    """Dispatch to the operator's `__call__` method.

    Examples
    --------
    >>> import coordinax as cx

    >>> coord = cx.Coordinate(cx.CartesianPos3D.from_([1, 2, 3], "kpc"),
    ...                       cx.frames.ICRS())
    >>> coord
    Coordinate(
        data=Space({ 'length': CartesianPos3D( ... ) }),
        frame=ICRS()
    )

    >>> op = cx.ops.GalileanSpatialTranslation.from_([-1, -1, -1], "kpc")

    >>> new_coord = op(coord)
    >>> print(new_coord.data["length"])
    <CartesianPos3D (x[kpc], y[kpc], z[kpc])
        [0 1 2]>

    """
    return replace(x, data=self(x.data))
