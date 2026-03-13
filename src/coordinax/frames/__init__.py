r"""Reference frames and transformations between them.

Examples
--------
>>> import quaxed.numpy as jnp
>>> import unxt as u
>>> import coordinax.core as cx
>>> import coordinax.ops as cxop
>>> import coordinax.frames as cxf

>>> frame1 = cxf.Alice()
>>> frame2 = cxf.Bob()

Let's transform a position from Alice's frame to Bob's frame:

>>> op = cxf.point_frame_transform(frame1, frame2)
>>> op
Pipe(( Translate(...), Boost(...) ))

>>> q_alice = cx.Vector.from_([0, 0, 0], "km")
>>> t = u.Q(2.5, "yr")
>>> q_bob = op(t, q_alice)
>>> print(q_bob)
<Vector: chart=Cart3D, role=Point (x, y, z) [km]
    [2.129e+13 1.000e+04 0.000e+00]>

Now let's create a new transformed frame and work with it:

>>> R = cxop.Rotate([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
>>> frame = cxf.TransformedReferenceFrame(frame1, R)
>>> frame
TransformedReferenceFrame(base_frame=Alice(), xop=Rotate(R=f64[3,3]))

Let's transform a position from the base frame to the transformed frame:

>>> op = cxf.point_frame_transform(cxf.Alice(), frame)

>>> q_icrs = cx.Vector.from_([1, 0, 0], "kpc")
>>> q_frame = op(q_icrs)
>>> print(q_frame)
<Vector: chart=Cart3D, role=Point (x, y, z) [kpc]
    [ 0. -1.  0.]>

>>> op.inverse(q_frame) == q_icrs
Array(True, dtype=bool)

"""

# Defined here b/c it's mutated by optional imports
__all__: tuple[str, ...] = (
    # Reference Frames
    "AbstractReferenceFrame",
    "FrameTransformError",
    "NoFrame",
    "TransformedReferenceFrame",
    "point_frame_transform",
    "frame_of",
    # Example frames
    "Alice",
    "FriendOfAlice",
    "Bob",
)


from ._setup_package import install_import_hook

with install_import_hook("coordinax.frames"):
    from ._src import (
        AbstractReferenceFrame,
        Alice,
        Bob,
        FrameTransformError,
        FriendOfAlice,
        NoFrame,
        TransformedReferenceFrame,
    )
    from coordinax.api.frames import frame_of, point_frame_transform


# clean up namespace
del install_import_hook
