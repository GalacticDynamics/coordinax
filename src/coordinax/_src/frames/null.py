"""Base implementation of coordinate frames."""

__all__ = ["NoFrame"]


from typing import final

from .base import AbstractReferenceFrame


@final
class NoFrame(AbstractReferenceFrame):
    """A null reference frame.

    This is a reference frame that cannot be transformed to or from.

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

    >>> try:
    ...     cxf.frame_transform_op(icrs, null)
    ... except cxf.FrameTransformError as e:
    ...     print(e)
    Cannot transform to the null frame.

    """
