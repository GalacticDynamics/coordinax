"""Base implementation of coordinate frames."""

__all__ = ("Alice", "alice", "Alex", "alex", "Bob", "bob")


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
      Translate({'x': Q(10, 'm'), 'y': Q(0, 'm'), 'z': Q(0, 'm')}, chart=Cart3D()),
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


@final
class Bob(AbstractReferenceFrame):
    """An inertial frame moving at constant velocity relative to Alice.

    Bob is a non-rotating observer in uniform motion with respect to Alice.
    His frame is characterised by:

    * **Spatial offset** from Alice's origin:
      [100 000 km, 10 000 km, 0 km].
    * **Velocity** relative to Alice:
      ≈ 269 813 km s⁻¹ (≈ 0.9 c) along Alice's x-axis.

    Because Bob is non-rotating, no rotation operator is required; the
    transformation only needs a spatial translation and a Galilean boost.

    Examples
    --------
    >>> import coordinax.frames as cxf

    Identity transition within Bob's own frame:

    >>> cxf.frame_transition(cxf.bob, cxf.bob)
    Identity()

    Transition from Alice's frame to Bob's frame (translate then boost):

    >>> op = cxf.frame_transition(cxf.alice, cxf.bob)
    >>> print(op)
    Composed((
      Translate( {...}, chart=Cart3D() ),
      Boost( {...}, chart=Cart3D() )
    ))

    Applying the Alice -> Bob transform to a bare array is not supported,
    because the boost cannot infer whether the array represents a position or a
    velocity:

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.transforms as cxfm
    >>> x = jnp.asarray([0.0, 0.0, 0.0])
    >>> cxfm.act(
    ...     op, None, x, usys=u.unitsystems.si
    ... )
    Traceback (most recent call last):
    plum._resolver.NotFoundLookupError: ...

    Applying the Alice -> Bob transform to a ``Quantity`` works:

    >>> q = u.Q([0.0, 0.0, 0.0], "km")
    >>> op(None, q)
    Q([100000.,  10000.,      0.], 'km')

    Applying the same transform to a component dictionary (``cdict``) also
    works:

    >>> import coordinax.main as cx
    >>> d = cx.cdict(u.Q([0.0, 0.0, 0.0], "km"))
    >>> op(None, d)
    {'x': Q(100000., 'km'), 'y': Q(10000., 'km'), 'z': Q(0., 'km')}

    """


bob = Bob()


# ===================================================================


@plum.dispatch.multi((Alice, Alice), (Alex, Alex), (Bob, Bob))
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

    >>> cxf.frame_transition(cxf.bob, cxf.bob)
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


@plum.dispatch
def frame_transition(from_frame: Alice, to_frame: Bob, /) -> cxfm.Composed:
    r"""Transform from Alice's frame to Bob's frame.

    This is an example of a reference frame transformation that includes
    both a spatial translation (Translate) and a velocity boost (Boost).

    The Boost has well-defined actions on kinematic roles:

    - **Point**: identity (points are unchanged by a Galilean boost)
    - **Pos**: identity (displacements are Galilean invariant)
    - **Vel**: adds $v_0$
    - **PhysAcc**: identity (for constant boost)

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.main as cx
    >>> import coordinax.frames as cxf

    >>> alice = cxf.Alice()
    >>> bob = cxf.Bob()

    >>> op = cxf.frame_transition(alice, bob)
    >>> print(op)
    Composed(( Translate(...), Boost(...) ))

    """
    shift = cxfm.Translate.from_([100_000, 10_000, 0], "km")
    boost = cxfm.Boost.from_([269_813_212.2, 0, 0], "m/s")
    return shift | boost  # ty: ignore[unsupported-operator]


@plum.dispatch.multi((Alex, Alice), (Bob, Alice))
def frame_transition(
    from_frame: AbstractReferenceFrame, to_frame: AbstractReferenceFrame, /
) -> cxfm.Composed:
    """Transform back.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.main as cx

    >>> cxf.frame_transition(cxf.alex, cxf.alice)
    Composed(( Rotate(...), Translate(...) ))

    >>> cxf.frame_transition(cxf.bob, cxf.alice)
    Composed((
      Boost( {...}, chart=Cart3D() ),
      Translate( {...}, chart=Cart3D() )
    ))

    """
    out = cxfapi.frame_transition(to_frame, from_frame).inverse  # pylint: disable=W1114  # ty: ignore[unresolved-attribute]
    return cast("cxfm.Composed", out)
