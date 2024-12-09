"""Register frame transform operations."""

__all__: list[str] = []

from typing import NoReturn

from plum import dispatch

from .base import AbstractReferenceFrame
from .errors import FrameTransformError
from .null import NoFrame


@dispatch(precedence=1)
def frame_transform_op(
    from_frame: NoFrame, to_frame: AbstractReferenceFrame, /
) -> NoReturn:
    """Cannot transform from the null frame.

    Examples
    --------
    >>> import coordinax.frames as cxf

    >>> null = cxf.NoFrame()
    >>> icrs = cxf.ICRS()

    >>> try:
    ...     cxf.frame_transform_op(null, icrs)
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
    >>> icrs = cxf.ICRS()

    >>> try:
    ...     cxf.frame_transform_op(icrs, null)
    ... except cxf.FrameTransformError as e:
    ...     print(e)
    Cannot transform to the null frame.

    """
    msg = "Cannot transform to the null frame."
    raise FrameTransformError(msg)
