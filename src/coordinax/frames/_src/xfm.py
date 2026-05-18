"""Transformations on Frames."""

__all__ = (
    "AbstractTransformedReferenceFrame",
    "TransformedReferenceFrame",
)

from typing import Generic, final
from typing_extensions import TypeVar

import plum

import coordinax.api.frames as cxfapi
import coordinax.transforms as cxfm
from .base import AbstractReferenceFrame
from coordinax.transforms import AbstractTransform

FrameT = TypeVar("FrameT", bound=AbstractReferenceFrame, default=AbstractReferenceFrame)


class AbstractTransformedReferenceFrame(AbstractReferenceFrame, Generic[FrameT]):
    r"""Transformations relative to a base reference frame.

    This class represents a reference frame that is defined relative to a base
    reference frame. The transformation from the base reference frame to this
    reference frame is stored as an `coordinax.operators.AbstractTransform`.

    ```{warning}

    The transformation operator ``ops`` is the transformation of the
    points, not a passive re-labeling of coordinates. Frame transitions in
    `coordinax` use active semantics: coordinates are transformed by direct
    application of the operator.

    ```

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.vectors as cxv
    >>> import coordinax.frames as cxf
    >>> import coordinax.transforms as cxfm
    >>> from coordinax.astro import ICRS

    >>> R = cxfm.Rotate(jnp.asarray([[0., -1, 0], [1, 0, 0], [0, 0, 1]]))
    >>> frame = cxf.TransformedReferenceFrame(ICRS(), R)
    >>> frame
    TransformedReferenceFrame(base_frame=ICRS(), xop=Rotate(R=f64[3,3]))

    Let's transform a position from the base frame to the transformed frame:

    >>> op = cxf.frame_transition(ICRS(), frame)

    >>> q_icrs = cxv.Point.from_([1, 0, 0], "kpc")
    >>> t = u.Q(1, "Myr")
    >>> q_frame = op(t, q_icrs)
    >>> print(q_frame)
    <Point: chart=Cart3D (x, y, z) [kpc]
        [0. 1. 0.]>

    >>> op.inverse(q_frame) == q_icrs
    Array(True, dtype=bool)

    """

    #: The base reference frame.
    base_frame: FrameT

    #: The transformation from the base frame to this frame.
    #: This is an active transformation. To transform coordinates from
    #: ``base_frame`` into this frame, apply this operator directly.
    xop: AbstractTransform


@final
class TransformedReferenceFrame(AbstractTransformedReferenceFrame[FrameT]):
    r"""Transformations relative to a base reference frame.

    This class represents a reference frame that is defined relative to a base
    reference frame. The transformation from the base reference frame to this
    reference frame is stored as an `coordinax.operators.AbstractTransform`.

    ```{warning}

    The transformation operator ``ops`` is the transformation of the
    points, not a passive re-labeling of coordinates. Frame transitions in
    `coordinax` use active semantics: coordinates are transformed by direct
    application of the operator.

    ```

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.vectors as cxv
    >>> import coordinax.frames as cxf
    >>> import coordinax.transforms as cxfm
    >>> from coordinax.astro import ICRS

    >>> R = cxfm.Rotate(jnp.asarray([[0., -1, 0], [1, 0, 0], [0, 0, 1]]))
    >>> frame = cxf.TransformedReferenceFrame(ICRS(), R)
    >>> frame
    TransformedReferenceFrame(base_frame=ICRS(), xop=Rotate(R=f64[3,3]))

    Let's transform a position from the base frame to the transformed frame:

    >>> op = cxf.frame_transition(ICRS(), frame)

    >>> q_icrs = cxv.Point.from_([1, 0, 0], "kpc")
    >>> t = u.Q(1, "Myr")
    >>> q_frame = op(t, q_icrs)
    >>> print(q_frame)
    <Point: chart=Cart3D (x, y, z) [kpc]
        [0. 1. 0.]>

    >>> op.inverse(q_frame) == q_icrs
    Array(True, dtype=bool)

    """

    #: The base reference frame.
    base_frame: FrameT

    #: The transformation from the base frame to this frame.
    #: This is an active transformation. To transform coordinates from
    #: ``base_frame`` into this frame, apply this operator directly.
    xop: AbstractTransform


@plum.dispatch
def frame_transition(
    from_frame: AbstractReferenceFrame, to_frame: TransformedReferenceFrame
) -> AbstractTransform:
    """Return a frame transform operator to a transformed frame.

    >>> import quaxed.numpy as jnp
    >>> import coordinax.vectors as cxv
    >>> import coordinax.frames as cxf
    >>> import coordinax.transforms as cxfm
    >>> from coordinax.astro import ICRS

    >>> R = cxfm.Rotate(jnp.asarray([[0., -1, 0], [1, 0, 0], [0, 0, 1]]))
    >>> frame = cxf.TransformedReferenceFrame(ICRS(), R)
    >>> frame
    TransformedReferenceFrame(base_frame=ICRS(), xop=Rotate(R=f64[3,3]))

    Let's transform a position from the base frame to the transformed frame:

    >>> op = cxf.frame_transition(ICRS(), frame)

    >>> q_icrs = cxv.Point.from_([1, 0, 0], "kpc")
    >>> q_frame = op(q_icrs)
    >>> print(q_frame)
    <Point: chart=Cart3D (x, y, z) [kpc]
        [0. 1. 0.]>

    """
    return cxfapi.frame_transition(from_frame, to_frame.base_frame) | to_frame.xop  # ty: ignore[unsupported-operator]


@plum.dispatch
def frame_transition(
    from_frame: TransformedReferenceFrame, to_frame: AbstractReferenceFrame
) -> AbstractTransform:
    """Return a frame transform operator from a transformed frame.

    >>> import quaxed.numpy as jnp
    >>> import coordinax.vectors as cxv
    >>> import coordinax.frames as cxf
    >>> import coordinax.transforms as cxfm
    >>> from coordinax.astro import ICRS

    >>> R = cxfm.Rotate(jnp.asarray([[0., -1, 0], [1, 0, 0], [0, 0, 1]]))
    >>> frame = cxf.TransformedReferenceFrame(ICRS(), R)
    >>> frame
    TransformedReferenceFrame(base_frame=ICRS(), xop=Rotate(R=f64[3,3]))

    Let's transform a position from the base frame to the transformed frame:

    >>> op = cxf.frame_transition(frame, ICRS())

    >>> q_icrs = cxv.Point.from_([0, 1, 0], "kpc")
    >>> q_frame = op(q_icrs)
    >>> print(q_frame)
    <Point: chart=Cart3D (x, y, z) [kpc]
        [1. 0. 0.]>

    """
    return from_frame.xop.inverse | cxfapi.frame_transition(
        from_frame.base_frame, to_frame
    )  # ty: ignore[unsupported-operator]


@plum.dispatch(precedence=1)  # ty: ignore[no-matching-overload]
def frame_transition(
    from_frame: TransformedReferenceFrame, to_frame: TransformedReferenceFrame
) -> AbstractTransform:
    """Return a frame transform operator between two transformed frames.

    When ``from_frame`` and ``to_frame`` are the same object the result is
    the identity transform.

    >>> import quaxed.numpy as jnp
    >>> import coordinax.vectors as cxv
    >>> import coordinax.frames as cxf
    >>> import coordinax.transforms as cxfm
    >>> from coordinax.astro import ICRS

    >>> R = cxfm.Rotate(jnp.asarray([[0., -1, 0], [1, 0, 0], [0, 0, 1]]))
    >>> frame1 = cxf.TransformedReferenceFrame(ICRS(), R)

    Same frame → identity:

    >>> cxf.frame_transition(frame1, frame1)
    Identity()

    >>> shift = cxfm.Translate.from_([1, 0, 0], "kpc")
    >>> frame2 = cxf.TransformedReferenceFrame(frame1, shift)

    >>> op1to2 = cxf.frame_transition(frame1, frame2)

    >>> q_frame1 = cxv.Point.from_([0, -1, 0], "kpc")
    >>> q_frame2 = op1to2(q_frame1)
    >>> print(q_frame2)
    <Point: chart=Cart3D (x, y, z) [kpc]
        [ 1. -1.  0.]>

    """
    if from_frame is to_frame:
        return cxfm.identity
    return (
        from_frame.xop.inverse  # ty: ignore[unsupported-operator]
        | cxfapi.frame_transition(from_frame.base_frame, to_frame.base_frame)
        | to_frame.xop
    )
