"""Register dispatches for `coordinax.frames`."""

__all__: list[str] = []


from typing import NoReturn

from plum import dispatch

from .base import AbstractReferenceFrame
from .coordinate import AbstractCoordinate
from .errors import FrameTransformError
from .null import NoFrame


@dispatch
def frame_of(obj: AbstractReferenceFrame, /) -> AbstractReferenceFrame:
    """Get the frame of an `coordinax.frames.AbstractReferenceFrame`.

    Examples
    --------
    >>> import coordinax.frames as cxf
    >>> frame = cxf.ICRS()
    >>> frame_of(frame) is frame
    True

    """
    return obj


@dispatch
def frame_of(obj: AbstractCoordinate) -> AbstractReferenceFrame:
    """Return the frame of the coordinate.

    Examples
    --------
    >>> import coordinax as cx

    >>> coord = cx.Coordinate(cx.CartesianPos3D.from_([1, 2, 3], "kpc"),
    ...                       cx.frames.ICRS())
    >>> cx.frames.frame_of(coord)
    ICRS()

    """
    return obj.frame


# ===============================================================


@dispatch(precedence=1)
def frame_transform_op(
    from_frame: NoFrame, to_frame: AbstractReferenceFrame, /
) -> NoReturn:
    """Cannot transform from the null frame.

    Examples
    --------
    >>> import coordinax.frames as cxf

    >>> null = cxf.NoFrame()
    >>> alice = cxf.Alice()

    >>> try:
    ...     cxf.frame_transform_op(null, alice)
    ... except cxf.FrameTransformError as e:
    ...     print(e)
    Cannot transform from the null frame.

    """
    msg = "Cannot transform from the null frame."
    raise FrameTransformError(msg)


@dispatch
def frame_transform_op(
    from_frame: AbstractReferenceFrame, to_frame: NoFrame, /
) -> NoReturn:
    """Cannot transform to the null frame.

    Examples
    --------
    >>> import coordinax.frames as cxf

    >>> null = cxf.NoFrame()
    >>> alice = cxf.Alice()

    >>> try:
    ...     cxf.frame_transform_op(alice, null)
    ... except cxf.FrameTransformError as e:
    ...     print(e)
    Cannot transform to the null frame.

    """
    msg = "Cannot transform to the null frame."
    raise FrameTransformError(msg)
