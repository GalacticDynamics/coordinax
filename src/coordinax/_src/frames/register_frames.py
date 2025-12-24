"""Register dispatches for `coordinax.frames`."""

__all__: tuple[str, ...] = ()


from typing import NoReturn

import plum

from .base import AbstractReferenceFrame
from .errors import FrameTransformError
from .null import NoFrame


@plum.dispatch
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


# ===============================================================


@plum.dispatch(precedence=1)
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


@plum.dispatch
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
