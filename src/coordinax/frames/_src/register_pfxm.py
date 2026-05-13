"""Register dispatches for `coordinax.frames`."""

__all__: tuple[str, ...] = ()


from typing import NoReturn

import plum

import coordinax.transforms as cxfm
from .base import AbstractReferenceFrame
from .errors import FrameTransformError
from .null import NoFrame


@plum.dispatch(precedence=2)  # ty: ignore[no-matching-overload]
def frame_transition(from_frame: NoFrame, to_frame: NoFrame, /) -> cxfm.Identity:
    """Null-to-null frame transition is always the identity.

    When both source and target frames are :obj:`noframe` (i.e. the vector is
    frame-agnostic) there is nothing to transform, so the result is the
    identity operation.

    Examples
    --------
    >>> import coordinax.frames as cxf
    >>> import coordinax.transforms as cxfm

    >>> op = cxf.frame_transition(cxf.noframe, cxf.noframe)
    >>> isinstance(op, cxfm.Identity)
    True

    """
    return cxfm.identity


@plum.dispatch(precedence=1)  # ty: ignore[no-matching-overload]
def frame_transition(
    from_frame: NoFrame, to_frame: AbstractReferenceFrame, /
) -> NoReturn:
    """Cannot transform from the null frame.

    Examples
    --------
    >>> import coordinax.frames as cxf

    >>> try:
    ...     cxf.frame_transition(cxf.noframe, cxf.alice)
    ... except cxf.FrameTransformError as e:
    ...     print(e)
    Cannot transform from the null frame.

    """
    msg = "Cannot transform from the null frame."
    raise FrameTransformError(msg)


@plum.dispatch
def frame_transition(
    from_frame: AbstractReferenceFrame, to_frame: NoFrame, /
) -> NoReturn:
    """Cannot transform to the null frame.

    Examples
    --------
    >>> import coordinax.frames as cxf

    >>> try:
    ...     cxf.frame_transition(cxf.alice, cxf.noframe)
    ... except cxf.FrameTransformError as e:
    ...     print(e)
    Cannot transform to the null frame.

    """
    msg = "Cannot transform to the null frame."
    raise FrameTransformError(msg)
