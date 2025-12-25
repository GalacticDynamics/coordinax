"""Register dispatches for `unxt`."""

__all__: tuple[str, ...] = ()


from plum import dispatch

from dataclassish import replace

import coordinax.vecs as cxv
from .base import AbstractReferenceFrame
from .coordinate import Coordinate
from .xfm import TransformedReferenceFrame
from coordinax._src.operators import AbstractOperator

# ===============================================================
# Constructors


@dispatch  # TODO: KinematicSpace[PosT] -- plum#212
def vector(
    cls: type[Coordinate],
    data: cxv.KinematicSpace | cxv.Vector,
    frame: AbstractReferenceFrame,
    /,
) -> Coordinate:
    """Construct a coordinate from data and a frame.

    Examples
    --------
    >>> import coordinax as cx

    >>> data = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> cx.Coordinate.from_(data, cx.frames.ICRS())
    Coordinate( KinematicSpace({ 'length': CartesianPos3D(...) }),
                frame=ICRS() )

    """
    return cls(data=data, frame=frame)


@dispatch  # TODO: KinematicSpace[PosT] -- plum#212
def vector(
    cls: type[Coordinate],
    data: cxv.KinematicSpace | cxv.Vector,
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
        KinematicSpace({ 'length': CartesianPos3D(...) }),
        frame=TransformedReferenceFrame(base_frame=ICRS(), xop=Identity())
    )

    """
    frame = TransformedReferenceFrame(base_frame, ops)
    return cls(data=data, frame=frame)


# ===============================================================
# Vector conversion


@dispatch
def vconvert(target: type[cxv.Vector], w: Coordinate, /) -> Coordinate:
    """Transform the vector representation of a coordinate.

    Examples
    --------
    >>> import coordinax as cx

    >>> frame = cx.frames.NoFrame()
    >>> data = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> w = cx.Coordinate(data, frame)

    >>> cx.vconvert(cx.SphericalPos, w)
    Coordinate(
        KinematicSpace({ 'length': SphericalPos( ... ) }),
        frame=NoFrame()
    )

    """
    return replace(w, data=w.data.vconvert(target))
