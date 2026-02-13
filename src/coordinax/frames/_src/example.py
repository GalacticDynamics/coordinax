"""Base implementation of coordinate frames."""

__all__ = ("Alice", "FriendOfAlice", "Bob")


from typing import final

import plum

import unxt as u

import coordinax.api as cxapi
import coordinax.ops as cxop
from .base import AbstractReferenceFrame


@final
class Alice(AbstractReferenceFrame):
    """A stationary lab reference frame.

    Examples
    --------
    >>> import coordinax.frames as cxf

    >>> alice = cxf.Alice()
    >>> bob = cxf.Bob()

    >>> op = cxf.frame_transform_op(alice, bob)
    >>> print(op)
    Pipe((
      Translate(...),
      Boost(...)
    ))

    """


@final
class FriendOfAlice(AbstractReferenceFrame):
    """A reference frame shifted from Alice's frame."""


@final
class Bob(AbstractReferenceFrame):
    """A moving reference frame.

    Examples
    --------
    >>> import coordinax.frames as cxf

    >>> alice = cxf.Alice()
    >>> bob = cxf.Bob()

    >>> op = cxf.frame_transform_op(alice, bob)
    >>> print(op)
    Pipe((
      Translate(...),
      Boost(...)
    ))

    """


# ===================================================================


@plum.dispatch.multi((Alice, Alice), (FriendOfAlice, FriendOfAlice), (Bob, Bob))
def frame_transform_op(
    from_frame: AbstractReferenceFrame,
    to_frame: AbstractReferenceFrame,
    /,
) -> cxop.Identity:
    """Return an identity operator for frames that are the same.

    Examples
    --------
    >>> import coordinax.frames as cxf

    >>> alice = cxf.Alice()
    >>> cxf.frame_transform_op(alice, alice)
    Identity()

    >>> friend = cxf.FriendOfAlice()
    >>> cxf.frame_transform_op(friend, friend)
    Identity()

    >>> bob = cxf.Bob()
    >>> cxf.frame_transform_op(bob, bob)
    Identity()

    """
    return cxop.Identity()


@plum.dispatch
def frame_transform_op(from_frame: Alice, to_frame: FriendOfAlice, /) -> cxop.Pipe:
    """Transform from Alice's frame to FriendOfAlice's frame.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> alice = cxf.Alice()
    >>> friend = cxf.FriendOfAlice()

    >>> op = cxf.frame_transform_op(alice, friend)
    >>> print(op)
    Pipe((
      Translate(...),
      Rotate(...)
    ))

    """
    shift = cxop.Translate.from_([10, 0, 0], "m")
    rotation = cxop.Rotate.from_euler("Z", u.Q(90, "deg"))
    return shift | rotation


@plum.dispatch
def frame_transform_op(from_frame: Alice, to_frame: Bob, /) -> cxop.Pipe:
    r"""Transform from Alice's frame to Bob's frame.

    This is an example of a reference frame transformation that includes
    both a spatial translation (Translate) and a velocity boost (Boost).

    The Boost has well-defined actions on kinematic roles:

    - **Point**: Not applicable (raises TypeError)
    - **Pos**: identity (displacements are Galilean invariant)
    - **Vel**: adds $v_0$
    - **PhysAcc**: identity (for constant boost)

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import coordinax.frames as cxf

    >>> alice = cxf.Alice()
    >>> bob = cxf.Bob()

    >>> op = cxf.frame_transform_op(alice, bob)
    >>> print(op)
    Pipe((
      Translate(...),
      Boost(...)
    ))

    """
    shift = cxop.Translate.from_([100_000, 10_000, 0], "km")
    boost = cxop.Boost.from_([269_813_212.2, 0, 0], "m/s")
    return shift | boost


@plum.dispatch.multi(
    (FriendOfAlice, Alice),
    (Bob, Alice),
)
def frame_transform_op(
    from_frame: AbstractReferenceFrame, to_frame: AbstractReferenceFrame, /
) -> cxop.Pipe:
    """Transform back.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> alice = cxf.Alice()
    >>> friend = cxf.FriendOfAlice()

    >>> op = cxf.frame_transform_op(friend, alice)
    >>> print(op)
    Pipe((
      Rotate(...),
      Translate(...)
    ))

    """
    return cxapi.frame_transform_op(to_frame, from_frame).inverse  # pylint: disable=W1114
