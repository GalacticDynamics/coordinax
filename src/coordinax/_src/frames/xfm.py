"""Transformations on Frames."""

__all__: list[str] = ["TransformedReferenceFrame"]

from typing import final

from plum import dispatch

from .base import AbstractReferenceFrame
from coordinax._src.operators.base import AbstractOperator


@final
class TransformedReferenceFrame(AbstractReferenceFrame):
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
    >>> import coordinax as cx
    >>> import coordinax.frames as cxf

    >>> R = cx.ops.GalileanRotation([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    >>> frame = cxf.TransformedReferenceFrame(cxf.ICRS(), R)
    >>> frame
    TransformedReferenceFrame(
      base_frame=ICRS(), xop=GalileanRotation(rotation=f32[3,3])
    )

    Let's transform a position from the base frame to the transformed frame:

    >>> op = cxf.frame_transform_op(cxf.ICRS(), frame)

    >>> q_icrs = cx.CartesianPos3D.from_([1, 0, 0], "kpc")
    >>> q_frame = op(q_icrs)
    >>> print(q_frame)
    <CartesianPos3D (x[kpc], y[kpc], z[kpc])
        [ 0. -1.  0.]>

    >>> op.inverse(q_frame) == q_icrs
    Array(True, dtype=bool)

    This can also transform a velocity:

    >>> v_icrs = cx.CartesianVel3D.from_([1, 0, 0], "km/s")
    >>> q_frame, v_frame = op(q_icrs, v_icrs)
    >>> print(q_frame, v_frame, sep="\n")
    <CartesianPos3D (x[kpc], y[kpc], z[kpc])
        [ 0. -1.  0.]>
    <CartesianVel3D (d_x[km / s], d_y[km / s], d_z[km / s])
        [ 0. -1.  0.]>

    >>> op.inverse(q_frame, v_frame) == (q_icrs, v_icrs)
    True

    """

    #: The base reference frame.
    base_frame: AbstractReferenceFrame

    #: The transformation from the base frame to this frame.
    #: This is a passive transformation, describing the transformation of the
    #: frame, not the coordinates. To transform the coordinates, apply the
    #: inverse of this operator.
    xop: AbstractOperator


@dispatch
def frame_transform_op(
    from_frame: AbstractReferenceFrame, to_frame: TransformedReferenceFrame
) -> AbstractOperator:
    """Return a frame transform operator to a transformed frame.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx
    >>> import coordinax.frames as cxf

    >>> R = cx.ops.GalileanRotation([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    >>> frame = cxf.TransformedReferenceFrame(cxf.ICRS(), R)
    >>> frame
    TransformedReferenceFrame(
      base_frame=ICRS(), xop=GalileanRotation(rotation=f32[3,3])
    )

    Let's transform a position from the base frame to the transformed frame:

    >>> op = cxf.frame_transform_op(cxf.ICRS(), frame)

    >>> q_icrs = cx.CartesianPos3D.from_([1, 0, 0], "kpc")
    >>> q_frame = op(q_icrs)
    >>> print(q_frame)
    <CartesianPos3D (x[kpc], y[kpc], z[kpc])
        [ 0. -1.  0.]>

    """
    return frame_transform_op(from_frame, to_frame.base_frame) | to_frame.xop.inverse


@dispatch
def frame_transform_op(
    from_frame: TransformedReferenceFrame, to_frame: AbstractReferenceFrame
) -> AbstractOperator:
    """Return a frame transform operator from a transformed frame.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx
    >>> import coordinax.frames as cxf

    >>> R = cx.ops.GalileanRotation([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    >>> frame = cxf.TransformedReferenceFrame(cxf.ICRS(), R)
    >>> frame
    TransformedReferenceFrame(
      base_frame=ICRS(), xop=GalileanRotation(rotation=f32[3,3])
    )

    Let's transform a position from the base frame to the transformed frame:

    >>> op = cxf.frame_transform_op(frame, cxf.ICRS())

    >>> q_icrs = cx.CartesianPos3D.from_([0, -1, 0], "kpc")
    >>> q_frame = op(q_icrs)
    >>> print(q_frame)
    <CartesianPos3D (x[kpc], y[kpc], z[kpc])
        [1. 0. 0.]>

    """
    return from_frame.xop | frame_transform_op(from_frame.base_frame, to_frame)


@dispatch(precedence=1)
def frame_transform_op(
    from_frame: TransformedReferenceFrame, to_frame: TransformedReferenceFrame
) -> AbstractOperator:
    """Return a frame transform operator between two transformed frames.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx
    >>> import coordinax.frames as cxf

    >>> R = cx.ops.GalileanRotation([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    >>> frame1 = cxf.TransformedReferenceFrame(cxf.ICRS(), R)

    >>> shift = cx.ops.GalileanSpatialTranslation.from_([1, 0, 0], "kpc")
    >>> frame2 = cxf.TransformedReferenceFrame(frame1, shift)

    >>> op1to2 = cxf.frame_transform_op(frame1, frame2)

    >>> q_frame1 = cx.CartesianPos3D.from_([0, -1, 0], "kpc")
    >>> q_frame2 = op1to2(q_icrs)

    """
    return (
        from_frame.xop
        | frame_transform_op(from_frame.base_frame, to_frame.base_frame)
        | to_frame.xop.inverse
    )
