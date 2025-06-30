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
    >>> alice = cxf.Alice()

    >>> try:
    ...     cxf.frame_transform_op(null, alice)
    ... except cxf.FrameTransformError as e:
    ...     print(e)
    Cannot transform from the null frame.

    >>> try:
    ...     cxf.frame_transform_op(alice, null)
    ... except cxf.FrameTransformError as e:
    ...     print(e)
    Cannot transform to the null frame.

    """
