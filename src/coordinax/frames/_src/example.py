"""Base implementation of coordinate frames."""

__all__ = ("Alice", "alice", "Alex", "alex", "Bob", "bob", "Carol", "carol")


from typing import cast, final

import plum

import unxt as u

import coordinax.charts as cxc
import coordinax.representations as cxr
import coordinax.transforms as cxfm
import coordinaxs.api.frames as cxfapi
from .base import AbstractReferenceFrame

BOB_BETA = 0.9
"""Bob's speed relative to Alice, as a fraction of ``c``.

Relativistic on purpose: at this speed a Galilean velocity kick gives answers
that exceed ``c``, which is what makes Bob the frame that has to be Lorentz.
"""

CAROL_SPEED = u.Q([30.0, 0.0, 0.0], "km/s")
"""Carol's velocity relative to Alice -- Earth's orbital speed, 1e-4 c.

Slow on purpose: this is where a Galilean velocity kick *is* the right physics,
and the correction Lorentz would add is one part in 1e8.
"""


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
    * **Alice → Bob**: on *spacetime*, translate the spatial origin then
      apply a Lorentz boost of β = 0.9 along Alice's x-axis.
    * **Alice → Carol**: translate [100 000 km, 10 000 km, 0] from Alice's
      origin, then apply a Galilean velocity kick of 30 km s⁻¹.

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


@final
class Bob(AbstractReferenceFrame):
    """A relativistic inertial frame, moving at 0.9 c relative to Alice.

    Bob is a non-rotating observer in uniform motion with respect to Alice:

    * **Spatial offset** from Alice's origin:
      [100 000 km, 10 000 km, 0 km].
    * **Velocity** relative to Alice: 0.9 c along Alice's x-axis.

    At that speed the transformation has to be a **Lorentz boost**, so Bob
    lives on spacetime -- `~coordinax.charts.minkowskict`, components
    ``(ct, x, y, z)`` -- rather than on a 3-D spatial chart. `Carol` is the
    slow counterpart, where a Galilean velocity kick is correct and ordinary
    3-D charts suffice.

    Examples
    --------
    >>> import coordinax.frames as cxf

    Identity transition within Bob's own frame:

    >>> cxf.frame_transition(cxf.bob, cxf.bob)
    Identity()

    Alice to Bob is a spacetime translation followed by the boost:

    >>> op = cxf.frame_transition(cxf.alice, cxf.bob)
    >>> [type(t).__name__ for t in op.transforms]
    ['Translate', 'LorentzBoost']

    **Why a boost and not a velocity kick.** A kick adds velocities, which at
    0.9 c gives answers faster than light. Take a particle already moving at
    0.3 c in Alice's frame:

    >>> c = 299792458.0
    >>> round((0.3 * c + 0.9 * c) / c, 4)          # adding them
    1.2

    That is 1.2 c -- not a small error but an impossible speed. Composing the
    velocities relativistically instead keeps the result below ``c``, as it
    must:

    >>> round((0.3 + 0.9) / (1 + 0.3 * 0.9), 4)    # composing them
    0.9449

    A boost needs a time component, so a purely spatial point has nothing to
    transform and is refused rather than silently mishandled:

    >>> import coordinax as cx
    >>> p = cx.Point.from_([0.0, 0.0, 0.0], "m")
    >>> try:
    ...     op(p)
    ... except Exception as e:
    ...     print(type(e).__name__)
    ManifoldMismatchError

    On spacetime it acts as expected -- Alice's origin, boosted, is somewhere
    else in both time and space:

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> ev = cx.Point.from_(
    ...     {"ct": u.Q(0.0, "m"), "x": u.Q(0.0, "m"),
    ...      "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")},
    ...     cxc.minkowskict, cx.point)
    >>> out = op(ev)
    >>> round(float(out["ct"].ustrip("m")), 1)
    206474160.5

    """


bob = Bob()


@final
class Carol(AbstractReferenceFrame):
    """An inertial frame in slow uniform motion relative to Alice.

    Carol is what Bob used to be: a non-rotating observer offset from Alice
    and moving at a constant velocity, so the transformation is a spatial
    `~coordinax.transforms.Translate` followed by a velocity kick (a
    ``Translate`` with ``semantic_kind=vel`` -- the fibre-only offset).

    * **Spatial offset** from Alice's origin: [100 000 km, 10 000 km, 0].
    * **Velocity** relative to Alice: 30 km s⁻¹ along Alice's x-axis.

    The speed is the point. At 30 km s⁻¹ -- Earth's orbital speed, 1e-4 c --
    adding velocities is correct to one part in 1e8, so the Galilean kick is
    the right physics and stays on ordinary 3-D charts. `Bob` moves at 0.9 c,
    where the same treatment gives velocities above ``c``, and so is a
    Lorentz boost on spacetime instead.

    Examples
    --------
    >>> import coordinax.frames as cxf

    >>> cxf.frame_transition(cxf.carol, cxf.carol)
    Identity()

    >>> op = cxf.frame_transition(cxf.alice, cxf.carol)
    >>> [type(t).__name__ for t in op.transforms]
    ['Translate', 'Translate']

    The kick is identity on positions and adds to velocities:

    >>> import unxt as u
    >>> q = u.Q([0.0, 0.0, 0.0], "km")
    >>> op(None, q)
    Q([100000.,  10000.,      0.], 'km')

    """


carol = Carol()
"""Canonical `Carol` singleton."""


# ===================================================================


@plum.dispatch.multi((Alice, Alice), (Alex, Alex), (Bob, Bob), (Carol, Carol))
def frame_transition(
    from_frame: AbstractReferenceFrame, to_frame: AbstractReferenceFrame, /
) -> cxfm.Identity:
    """Return an identity operator for frames that are the same.

    >>> import coordinax.frames as cxf
    >>> cxf.frame_transition(cxf.alice, cxf.alice)
    Identity()
    >>> cxf.frame_transition(cxf.alex, cxf.alex)
    Identity()

    >>> cxf.frame_transition(cxf.bob, cxf.bob)
    Identity()

    >>> cxf.frame_transition(cxf.carol, cxf.carol)
    Identity()

    """
    return cxfm.identity


@plum.dispatch
def frame_transition(from_frame: Alice, to_frame: Alex, /) -> cxfm.Composed:
    """Transform from Alice's frame to Alex's frame.

    >>> import unxt as u
    >>> import coordinax as cx
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

    A spacetime translation followed by a **Lorentz boost** of
    $\beta = 0.9$ along Alice's x-axis. Bob moves too fast for a Galilean
    velocity kick: adding 0.9 c to a particle already at 0.3 c would give
    1.2 c. See `Bob` for that comparison worked through, and `Carol` for the
    slow frame where the kick *is* the right physics.

    Because a boost mixes time into the spatial components, this transition
    acts on `~coordinax.charts.minkowskict` rather than a 3-D chart.

    Examples
    --------
    >>> import coordinax.frames as cxf

    >>> op = cxf.frame_transition(cxf.Alice(), cxf.Bob())
    >>> [type(t).__name__ for t in op.transforms]
    ['Translate', 'LorentzBoost']

    """
    shift = cxfm.Translate(
        cxc.cdict(u.Q([0.0, 1.0e8, 1.0e7, 0.0], "m"), cxc.minkowskict),
        chart=cxc.minkowskict,
    )
    return shift | cxfm.LorentzBoost([BOB_BETA, 0.0, 0.0])  # ty: ignore[unsupported-operator]


@plum.dispatch
def frame_transition(from_frame: Alice, to_frame: Carol, /) -> cxfm.Composed:
    r"""Transform from Alice's frame to Carol's frame.

    A spatial translation followed by a velocity kick -- the transformation
    `Bob` carried before he became relativistic. The kick has well-defined
    actions on each kinematic role:

    - **Point**: identity (a velocity kick does not move points)
    - **Displacement**: identity (Galilean invariant)
    - **Velocity**: adds $v_0$
    - **Acceleration**: identity (for a constant kick)

    Examples
    --------
    >>> import coordinax.frames as cxf

    >>> op = cxf.frame_transition(cxf.alice, cxf.carol)
    >>> [type(t).__name__ for t in op.transforms]
    ['Translate', 'Translate']

    """
    shift = cxfm.Translate.from_([100_000, 10_000, 0], "km")
    kick = cxfm.Translate(
        cxc.cdict(CAROL_SPEED, cxc.cart3d), chart=cxc.cart3d, semantic_kind=cxr.vel
    )
    return shift | kick  # ty: ignore[unsupported-operator]


@plum.dispatch.multi((Alex, Alice), (Bob, Alice), (Carol, Alice))
def frame_transition(
    from_frame: AbstractReferenceFrame, to_frame: AbstractReferenceFrame, /
) -> cxfm.Composed:
    """Transform back.

    >>> import unxt as u
    >>> import coordinax as cx

    >>> cxf.frame_transition(cxf.alex, cxf.alice)
    Composed(( Rotate(...), Translate(...) ))

    >>> [type(t).__name__ for t in cxf.frame_transition(cxf.bob, cxf.alice).transforms]
    ['LorentzBoost', 'Translate']

    >>> op = cxf.frame_transition(cxf.carol, cxf.alice)
    >>> [type(t).__name__ for t in op.transforms]
    ['Translate', 'Translate']

    """
    out = cxfapi.frame_transition(to_frame, from_frame).inverse  # pylint: disable=W1114  # ty: ignore[unresolved-attribute]
    return cast("cxfm.Composed", out)
