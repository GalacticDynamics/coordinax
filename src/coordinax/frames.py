r"""Reference frames and transformations between them.

Examples
--------
>>> import quaxed.numpy as jnp
>>> import unxt as u
>>> import coordinax as cx
>>> import coordinax.frames as cxf

>>> frame1 = cxf.Alice()
>>> frame2 = cxf.Bob()

Let's transform a position from Alice's frame to Bob's frame:

>>> op = cxf.frame_transform_op(frame1, frame2)
>>> op
Pipe((
    GalileanSpatialTranslation(CartesianPos3D( ... )),
    GalileanBoost(CartesianVel3D( ... ))
))

>>> q_alice = cx.CartesianPos3D.from_([0, 0, 0], "km")
>>> t = u.Quantity(2.5, "yr")
>>> _, q_bob = op(t, q_alice)
>>> print(q_bob)
<CartesianPos3D: (x, y, z) [km]
    [2.129e+13 1.000e+04 0.000e+00]>

Now let's create a new transformed frame and work with it:

>>> R = cx.ops.GalileanRotation([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
>>> frame = cxf.TransformedReferenceFrame(frame1, R)
>>> frame
TransformedReferenceFrame(
    base_frame=Alice(), xop=GalileanRotation(rotation=f32[3,3])
)

Let's transform a position from the base frame to the transformed frame:

>>> op = cxf.frame_transform_op(cxf.Alice(), frame)

>>> q_icrs = cx.CartesianPos3D.from_([1, 0, 0], "kpc")
>>> q_frame = op(q_icrs)
>>> print(q_frame)
<CartesianPos3D: (x, y, z) [kpc]
    [ 0. -1.  0.]>

>>> op.inverse(q_frame) == q_icrs
Array(True, dtype=bool)

This can also transform a velocity:

>>> v_icrs = cx.CartesianVel3D.from_([1, 0, 0], "km/s")
>>> q_frame, v_frame = op(q_icrs, v_icrs)
>>> print(q_frame, v_frame, sep="\n")
<CartesianPos3D: (x, y, z) [kpc]
    [ 0. -1.  0.]>
<CartesianVel3D: (x, y, z) [km / s]
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
    "frame_of",
    # Example frames
    "Alice",
    "FriendOfAlice",
    "Bob",
    # Coordinates
    "AbstractCoordinate",
    "Coordinate",
]

from .setup_package import install_import_hook

with install_import_hook("coordinax.frames"):
    from ._src.frames import (
        AbstractCoordinate,
        AbstractReferenceFrame,
        Alice,
        Bob,
        Coordinate,
        FrameTransformError,
        FriendOfAlice,
        NoFrame,
        TransformedReferenceFrame,
        frame_of,
        frame_transform_op,
    )

    # Frames from external packages
    # isort: split
    from . import _coordinax_space_frames
    from ._coordinax_space_frames import *  # noqa: F403


__all__ += _coordinax_space_frames.__all__

# clean up namespace
del _coordinax_space_frames, install_import_hook
