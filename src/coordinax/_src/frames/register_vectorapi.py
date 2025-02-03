"""Register dispatches for `unxt`."""

__all__: list[str] = []


from plum import dispatch

from dataclassish import replace

from .base import AbstractReferenceFrame
from .coordinate import Coordinate
from .xfm import TransformedReferenceFrame
from coordinax._src.operators import AbstractOperator
from coordinax._src.vectors.base_pos import AbstractPos
from coordinax._src.vectors.space.core import Space

# ===============================================================
# Constructors


@dispatch
def vector(
    cls: type[Coordinate],
    data: Space | AbstractPos,
    frame: AbstractReferenceFrame,
    /,
) -> Coordinate:
    """Construct a coordinate from data and a frame.

    Examples
    --------
    >>> import coordinax as cx

    >>> data = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> cx.Coordinate.from_(data, cx.frames.ICRS())
    Coordinate(
        data=Space({ 'length': CartesianPos3D( ... ) }),
        frame=ICRS()
    )

    """
    return cls(data=data, frame=frame)


@dispatch
def vector(
    cls: type[Coordinate],
    data: Space | AbstractPos,
    base_frame: AbstractReferenceFrame,
    ops: AbstractOperator,
    /,
) -> Coordinate:
    """Construct a coordinate from data and a frame.

    Examples
    --------
    >>> import coordinax as cx

    >>> data = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> cx.Coordinate.from_(data, cx.frames.ICRS(), cx.ops.Identity())
    Coordinate(
        data=Space({ 'length': CartesianPos3D( ... ) }),
        frame=TransformedReferenceFrame(base_frame=ICRS(), xop=Identity())
    )

    """
    frame = TransformedReferenceFrame(base_frame, ops)
    return cls(data=data, frame=frame)


# ===============================================================
# Vector conversion


@dispatch
def vconvert(target: type[AbstractPos], w: Coordinate, /) -> Coordinate:
    """Transform the vector representation of a coordinate.

    Examples
    --------
    >>> import coordinax as cx

    >>> frame = cx.frames.ICRS()
    >>> data = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> w = cx.Coordinate(data, frame)

    >>> cx.vconvert(cx.SphericalPos, w)
    Coordinate(
        data=Space({ 'length': SphericalPos( ... ) }),
        frame=ICRS()
    )

    """
    return replace(w, data=w.data.vconvert(target))
