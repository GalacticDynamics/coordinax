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
Pipe(( Translate(...), Boost(...) ))

>>> q_alice = cx.Vector.from_([0, 0, 0], "km")
>>> t = u.Q(2.5, "yr")
>>> q_bob = op(t, q_alice)
>>> print(q_bob)
<Vector: chart=Cart3D, role=Point (x, y, z) [km]
    [2.129e+13 1.000e+04 0.000e+00]>

Now let's create a new transformed frame and work with it:

>>> R = cx.ops.Rotate([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
>>> frame = cxf.TransformedReferenceFrame(frame1, R)
>>> frame
TransformedReferenceFrame(base_frame=Alice(), xop=Rotate(R=f64[3,3]))

Let's transform a position from the base frame to the transformed frame:

>>> op = cxf.frame_transform_op(cxf.Alice(), frame)

>>> q_icrs = cx.Vector.from_([1, 0, 0], "kpc")
>>> q_frame = op(q_icrs)
>>> print(q_frame)
<Vector: chart=Cart3D, role=Point (x, y, z) [kpc]
    [ 0. -1.  0.]>

>>> op.inverse(q_frame) == q_icrs
Array(True, dtype=bool)

"""

import sys

from .setup_package import install_import_hook

with install_import_hook("coordinax.frames"):
    from ._src.api import frame_of, frame_transform_op
    from ._src.frames import (
        AbstractReferenceFrame,
        Alice,
        Bob,
        FrameTransformError,
        FriendOfAlice,
        NoFrame,
        TransformedReferenceFrame,
    )


# Defined here b/c it's mutated by optional imports
__all__: list[str] = [
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
]

# Try to import astronomy frames - use lazy import to avoid circular dependency
try:
    import coordinax_astro as _cxastro  # noqa: ICN001

except ImportError:
    _cxastro = None  # type: ignore[assignment]
else:
    __all__ += list(_cxastro.__all__)


def __getattr__(name: str, /) -> object:
    """Lazy import for coordinax_astro frames."""
    if (_cxastro is None) or (name not in _cxastro.__all__):  # type: ignore[redundant-expr]
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)

    obj = getattr(_cxastro, name)  # get from coordinax_astro
    setattr(sys.modules[__name__], name, obj)  # cache in this module
    return obj


# clean up namespace
del install_import_hook
