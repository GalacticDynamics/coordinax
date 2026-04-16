"""Base implementation of coordinate frames."""

__all__ = ("NoFrame", "noframe")


from typing import final

from .base import AbstractReferenceFrame


@final
class NoFrame(AbstractReferenceFrame):
    """A null reference frame.

    This is a reference frame that cannot be transformed to or from.

    Examples
    --------
    >>> import coordinax.frames as cxf

    >>> try:
    ...     cxf.frame_transition(cxf.noframe, cxf.alice)
    ... except cxf.FrameTransformError as e:
    ...     print(e)
    Cannot transform from the null frame.

    >>> try:
    ...     cxf.frame_transition(cxf.alice, cxf.noframe)
    ... except cxf.FrameTransformError as e:
    ...     print(e)
    Cannot transform to the null frame.

    """


noframe = NoFrame()  # instance of NoFrame
