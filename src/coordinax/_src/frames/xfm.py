"""Transformations on Frames."""

__all__ = ("TransformedReferenceFrame",)

from typing import Generic, final
from typing_extensions import TypeVar

import plum

import coordinax._src.operators as cxo
from .base import AbstractReferenceFrame
from coordinax._src import api

FrameT = TypeVar("FrameT", bound=AbstractReferenceFrame, default=AbstractReferenceFrame)


@final
class TransformedReferenceFrame(AbstractReferenceFrame, Generic[FrameT]):
    r"""Transformations relative to a base reference frame.

    This class represents a reference frame that is defined relative to a base
    reference frame. The transformation from the base reference frame to this
    reference frame is stored as an `coordinax.operators.AbstractOperator`.

    :::{warning}

    The transformation operator ``ops`` is the transformation of the
    **frames**, not the coordinates. This is a passive (intrinsic) versus active
    (extrinsic) transformation. A frame transformation (passive) is applied to
    the coordinates by the inverse of the operator.

    :::

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import coordinax.frames as cxf

    >>> R = cx.ops.Rotate(jnp.asarray([[0., -1, 0], [1, 0, 0], [0, 0, 1]]))
    >>> frame = cxf.TransformedReferenceFrame(cxf.ICRS(), R)
    >>> frame
    TransformedReferenceFrame(base_frame=ICRS(), xop=Rotate(R=f64[3,3]))

    Let's transform a position from the base frame to the transformed frame:

    >>> op = cxf.frame_transform_op(cxf.ICRS(), frame)

    >>> q_icrs = cx.Vector.from_([1, 0, 0], "kpc")
    >>> t = u.Q(1, "Myr")
    >>> q_frame = op(t, q_icrs)
    >>> print(q_frame)
    <Vector: chart=Cart3D, role=Point (x, y, z) [kpc]
        [ 0. -1.  0.]>

    >>> op.inverse(q_frame) == q_icrs
    Array(True, dtype=bool)

    """

    #: The base reference frame.
    base_frame: FrameT

    #: The transformation from the base frame to this frame.
    #: This is a passive transformation, describing the transformation of the
    #: frame, not the coordinates. To transform the coordinates, apply the
    #: inverse of this operator.
    xop: cxo.AbstractOperator


@plum.dispatch
def frame_transform_op(
    from_frame: AbstractReferenceFrame, to_frame: TransformedReferenceFrame
) -> cxo.AbstractOperator:
    """Return a frame transform operator to a transformed frame.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx
    >>> import coordinax.frames as cxf

    >>> R = cx.ops.Rotate(jnp.asarray([[0., -1, 0], [1, 0, 0], [0, 0, 1]]))
    >>> frame = cxf.TransformedReferenceFrame(cxf.ICRS(), R)
    >>> frame
    TransformedReferenceFrame(base_frame=ICRS(), xop=Rotate(R=f64[3,3]))

    Let's transform a position from the base frame to the transformed frame:

    >>> op = cxf.frame_transform_op(cxf.ICRS(), frame)

    >>> q_icrs = cx.Vector.from_([1, 0, 0], "kpc")
    >>> q_frame = op(q_icrs)
    >>> print(q_frame)
    <Vector: chart=Cart3D, role=Point (x, y, z) [kpc]
        [ 0. -1.  0.]>

    """
    return (
        api.frame_transform_op(from_frame, to_frame.base_frame) | to_frame.xop.inverse
    )


@plum.dispatch
def frame_transform_op(
    from_frame: TransformedReferenceFrame, to_frame: AbstractReferenceFrame
) -> cxo.AbstractOperator:
    """Return a frame transform operator from a transformed frame.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx
    >>> import coordinax.frames as cxf

    >>> R = cx.ops.Rotate(jnp.asarray([[0., -1, 0], [1, 0, 0], [0, 0, 1]]))
    >>> frame = cxf.TransformedReferenceFrame(cxf.ICRS(), R)
    >>> frame
    TransformedReferenceFrame(base_frame=ICRS(), xop=Rotate(R=f64[3,3]))

    Let's transform a position from the base frame to the transformed frame:

    >>> op = cxf.frame_transform_op(frame, cxf.ICRS())

    >>> q_icrs = cx.Vector.from_([0, -1, 0], "kpc")
    >>> q_frame = op(q_icrs)
    >>> print(q_frame)
    <Vector: chart=Cart3D, role=Point (x, y, z) [kpc]
        [1. 0. 0.]>

    """
    return from_frame.xop | api.frame_transform_op(from_frame.base_frame, to_frame)


@plum.dispatch(precedence=1)
def frame_transform_op(
    from_frame: TransformedReferenceFrame, to_frame: TransformedReferenceFrame
) -> cxo.AbstractOperator:
    """Return a frame transform operator between two transformed frames.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx
    >>> import coordinax.frames as cxf

    >>> R = cx.ops.Rotate(jnp.asarray([[0., -1, 0], [1, 0, 0], [0, 0, 1]]))
    >>> frame1 = cxf.TransformedReferenceFrame(cxf.ICRS(), R)

    >>> shift = cx.ops.Translate.from_([1, 0, 0], "kpc")
    >>> frame2 = cxf.TransformedReferenceFrame(frame1, shift)

    >>> op1to2 = cxf.frame_transform_op(frame1, frame2)

    >>> q_frame1 = cx.Vector.from_([0, -1, 0], "kpc")
    >>> q_frame2 = op1to2(q_frame1)

    """
    return (
        from_frame.xop
        | api.frame_transform_op(from_frame.base_frame, to_frame.base_frame)
        | to_frame.xop.inverse
    )
