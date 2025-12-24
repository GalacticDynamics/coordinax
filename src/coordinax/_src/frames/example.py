"""Base implementation of coordinate frames."""

__all__ = ("Alice", "FriendOfAlice", "Bob")


from typing import final

import plum

import unxt as u

import coordinax._src.operators as cxo
from .base import AbstractReferenceFrame
from coordinax._src import api


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
) -> cxo.Identity:
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
    return cxo.Identity()


@plum.dispatch
def frame_transform_op(from_frame: Alice, to_frame: FriendOfAlice, /) -> cxo.Pipe:
    """Transform from Alice's frame to FriendOfAlice's frame.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> alice = cx.frames.Alice()
    >>> friend = cx.frames.FriendOfAlice()

    >>> op = cx.frames.frame_transform_op(alice, friend)
    >>> print(op)
    Pipe((
      Translate(...),
      Rotate(...)
    ))

    """
    shift = cxo.Translate.from_([10, 0, 0], "m")
    rotation = cxo.Rotate.from_euler("Z", u.Q(90, "deg"))
    return shift | rotation


@plum.dispatch
def frame_transform_op(from_frame: Alice, to_frame: Bob, /) -> cxo.Pipe:
    r"""Transform from Alice's frame to Bob's frame.

    This is an example of a reference frame transformation that includes
    both a spatial translation (Translate) and a velocity boost (Boost).

    The Boost has a well-defined action on all kinematic roles:

    - **Point**: translates by $v_0 \cdot (\tau - \tau_0)$
    - **Pos**: identity (displacements are invariant)
    - **Vel**: adds $v_0$
    - **Acc**: identity (constant boost)

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
    shift = cxo.Translate.from_([100_000, 10_000, 0], "km")
    boost = cxo.Boost.from_([269_813_212.2, 0, 0], "m/s")
    return shift | boost


@plum.dispatch.multi(
    (FriendOfAlice, Alice),
    (Bob, Alice),
)
def frame_transform_op(
    from_frame: AbstractReferenceFrame, to_frame: AbstractReferenceFrame, /
) -> cxo.Pipe:
    """Transform back.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> alice = cx.frames.Alice()
    >>> friend = cx.frames.FriendOfAlice()

    >>> op = cx.frames.frame_transform_op(friend, alice)
    >>> print(op)
    Pipe((
      Rotate(...),
      Translate(...)
    ))

    """
    return api.frame_transform_op(to_frame, from_frame).inverse  # pylint: disable=W1114
