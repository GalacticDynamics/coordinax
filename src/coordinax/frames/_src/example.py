"""Base implementation of coordinate frames."""

__all__ = ("Alice", "alice", "Alex", "alex")


from typing import cast, final

import plum

import unxt as u

import coordinax.api.frames as cxfapi
import coordinax.transforms as cxfm
from .base import AbstractReferenceFrame


@final
class Alice(AbstractReferenceFrame):
    """A stationary lab reference frame at the origin.

    ``Alice`` serves as the "home" reference frame for the example frame
    system.  Her frame is fixed at the origin with standard (Cartesian)
    orientation.  All other example frames — ``Alex`` and ``Bob`` — are
    defined by their transformations *relative to* Alice.

    Frame relationships:

    * **Alice → Alex**: translate +10 m along Alice's x-axis, then rotate
      +90 ° about the z-axis.
    * **Alice → Bob**: translate [100 000 km, 10 000 km, 0] from Alice's
      origin, then apply a Galilean velocity boost of ≈ 269 813 km s⁻¹
      (≈ 0.9 c) along Alice's x-axis.

    Examples
    --------
    >>> import coordinax.frames as cxf
    >>> import jax

    Identity transition back to Alice's own frame:

    >>> cxf.frame_transition(cxf.alice, cxf.alice)
    Identity()

    Transition to Alex's frame (translate then rotate):

    >>> op = cxf.frame_transition(cxf.alice, cxf.alex)
    >>> print(jax.tree.map(lambda x: x.round(2), op))
    Composed((
      Translate({'x': Q(10, 'm'), 'y': Q(0, 'm'), 'z': Q(0, 'm')},
                chart=Cart3D(M=Rn(3))),
      Rotate([[ 0. -1.  0.]
              [ 1.  0.  0.]
              [ 0.  0.  1.]])
    ))

    """


alice = Alice()  # instance of Alice


@final
class Alex(AbstractReferenceFrame):
    """A reference frame displaced and rotated relative to Alice's frame.

    Alex is an observer who is stationary (like Alice) but occupies a
    different location and orientation in space:

    * **Origin**: +10 m along Alice's x-axis.
    * **Orientation**: rotated +90 ° about the shared z-axis relative to
      Alice, so what Alice calls her y-axis points along Alex's x-axis.

    The transformation Alice → Alex is therefore a *translate-then-rotate*
    composition::

        Alice → Alex:  Translate([10, 0, 0] m) | Rotate(Z, +90°)
        Alex → Alice:  Rotate(Z, -90°) | Translate([-10, 0, 0] m)

    Examples
    --------
    >>> import coordinax.frames as cxf

    Identity transition within Alex's own frame:

    >>> cxf.frame_transition(cxf.alex, cxf.alex)
    Identity()

    """


alex = Alex()  # instance of Alex

# ===================================================================


@plum.dispatch.multi((Alice, Alice), (Alex, Alex))
def frame_transition(
    from_frame: AbstractReferenceFrame,
    to_frame: AbstractReferenceFrame,
    /,
) -> cxfm.Identity:
    """Return an identity operator for frames that are the same.

    Examples
    --------
    >>> import coordinax.frames as cxf

    >>> cxf.frame_transition(cxf.alice, cxf.alice)
    Identity()

    >>> cxf.frame_transition(cxf.alex, cxf.alex)
    Identity()

    """
    return cxfm.identity


@plum.dispatch
def frame_transition(from_frame: Alice, to_frame: Alex, /) -> cxfm.Composed:
    """Transform from Alice's frame to Alex's frame.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.main as cx

    >>> op = cxf.frame_transition(cxf.alice, cxf.alex)
    >>> print(op)
    Composed(( Translate(...), Rotate(...) ))

    """
    shift = cxfm.Translate.from_([10, 0, 0], "m")
    rotation = cxfm.Rotate.from_euler("Z", u.Q(90, "deg"))
    return shift | rotation  # ty: ignore[unsupported-operator]


@plum.dispatch.multi((Alex, Alice))
def frame_transition(
    from_frame: AbstractReferenceFrame, to_frame: AbstractReferenceFrame, /
) -> cxfm.Composed:
    """Transform back.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.main as cx

    >>> op = cxf.frame_transition(cxf.alex, cxf.alice)
    >>> print(op)
    Composed(( Rotate(...), Translate(...) ))

    """
    out = cxfapi.frame_transition(to_frame, from_frame).inverse  # pylint: disable=W1114  # ty: ignore[unresolved-attribute]
    return cast("cxfm.Composed", out)
