r"""Reference frames and transformations between them.

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

__all__ = [
    # Frames
    "AbstractReferenceFrame",
    "FrameTransformError",
    "NoFrame",
    "TransformedReferenceFrame",
    "frame_transform_op",
    # Coordinates
    "AbstractCoordinate",
    "Coordinate",
]

from jaxtyping import install_import_hook

from .setup_package import RUNTIME_TYPECHECKER

with install_import_hook("coordinax.frames", RUNTIME_TYPECHECKER):
    from ._src.frames import (
        AbstractCoordinate,
        AbstractReferenceFrame,
        Coordinate,
        FrameTransformError,
        NoFrame,
        TransformedReferenceFrame,
        frame_transform_op,
    )

    # Register the frame transform operations
    # isort: split
    from ._src.frames.register_transforms import *  # noqa: F403

    # Frames from external packages
    # isort: split
    from . import _coordinax_space_frames
    from ._coordinax_space_frames import *  # noqa: F403


__all__ += _coordinax_space_frames.__all__

# clean up namespace
del _coordinax_space_frames, RUNTIME_TYPECHECKER, install_import_hook
