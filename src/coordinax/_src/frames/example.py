"""Base implementation of coordinate frames."""

__all__ = ["Alice", "FriendOfAlice", "Bob"]


from typing import final

from plum import dispatch

import unxt as u

from .base import AbstractReferenceFrame
from coordinax._src.frames import api
from coordinax._src.operators import (
    GalileanBoost,
    GalileanRotation,
    GalileanSpatialTranslation,
    Identity,
    Pipe,
)


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
      GalileanSpatialTranslation(<CartesianPos3D: (x, y, z) [km]
          [100000  10000      0]>),
      GalileanBoost(<CartesianVel3D: (x, y, z) [m / s]
          [2.698e+08 0.000e+00 0.000e+00]>)
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
      GalileanSpatialTranslation(<CartesianPos3D: (x, y, z) [km]
          [100000  10000      0]>),
      GalileanBoost(<CartesianVel3D: (x, y, z) [m / s]
          [2.698e+08 0.000e+00 0.000e+00]>)
    ))

    """


# ===================================================================


@dispatch.multi((Alice, Alice), (FriendOfAlice, FriendOfAlice), (Bob, Bob))
def frame_transform_op(
    from_frame: AbstractReferenceFrame,
    to_frame: AbstractReferenceFrame,
    /,
) -> Identity:
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
    return Identity()


@dispatch
def frame_transform_op(from_frame: Alice, to_frame: FriendOfAlice, /) -> Pipe:
    """Transform from Alice's frame to Bob's frame.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> alice = cx.frames.Alice()
    >>> friend = cx.frames.FriendOfAlice()

    >>> op = cx.frames.frame_transform_op(alice, friend)
    >>> print(op)
    Pipe((
      GalileanSpatialTranslation(<CartesianPos3D: (x, y, z) [m]
          [10  0  0]>),
      GalileanRotation([[ 0.         -0.99999994  0.        ]
                        [ 0.99999994  0.          0.        ]
                        [ 0.          0.          0.99999994]])
    ))

    >>> q_alice = cx.vecs.CartesianPos3D.from_([0, 0, 0], "m")
    >>> q_friend = op(u.Quantity(1, "s"), q_alice)
    >>> q_friend
    (Quantity(Array(1, dtype=int32, weak_type=True), unit='s'),
     CartesianPos3D( x=Quantity(0., unit='m'), y=Quantity(9.999999, unit='m'),
                     z=Quantity(0., unit='m')
     ))

    """
    shift = GalileanSpatialTranslation.from_([10, 0, 0], "m")
    rotation = GalileanRotation.from_euler("Z", u.Quantity(90, "deg"))
    return shift | rotation


@dispatch
def frame_transform_op(from_frame: Alice, to_frame: Bob, /) -> Pipe:
    """Transform from Alice's frame to Bob's frame.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv
    >>> import coordinax.frames as cxf

    >>> alice = cxf.Alice()
    >>> bob = cxf.Bob()

    >>> op = cxf.frame_transform_op(alice, bob)
    >>> print(op)
    Pipe((
      GalileanSpatialTranslation(<CartesianPos3D: (x, y, z) [km]
          [100000  10000      0]>),
      GalileanBoost(<CartesianVel3D: (x, y, z) [m / s]
          [2.698e+08 0.000e+00 0.000e+00]>)
    ))

    >>> q = cxv.CartesianPos3D.from_([0, 0, 0], "m")

    >>> q_bob = op(u.Quantity(0, "s"), q)
    >>> q_bob
    (Quantity(Array(0, dtype=int32, weak_type=True), unit='s'),
     CartesianPos3D( x=Quantity(1.e+08, unit='m'), ... z=Quantity(0., unit='m')
    ))

    >>> q_bob = op(u.Quantity(1, "s"), q)
    >>> q_bob
    (Quantity(Array(1, dtype=int32, weak_type=True), unit='s'),
     CartesianPos3D( x=Quantity(3.6981322e+08, unit='m'), ... z=Quantity(0., unit='m')
    ))

    """
    shift = GalileanSpatialTranslation.from_([100_000, 10_000, 0], "km")
    boost = GalileanBoost.from_([269_813_212.2, 0, 0], "m/s")
    return shift | boost


@dispatch.multi(
    (FriendOfAlice, Alice),
    (Bob, Alice),
)
def frame_transform_op(
    from_frame: AbstractReferenceFrame, to_frame: AbstractReferenceFrame, /
) -> Pipe:
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
      GalileanRotation([[ 0.          0.99999994  0.        ]
                        [-0.99999994  0.          0.        ]
                        [ 0.          0.          0.99999994]]),
      GalileanSpatialTranslation(<CartesianPos3D: (x, y, z) [m]
          [-10   0   0]>)
    ))

    >>> q_friend = cx.vecs.CartesianPos3D.from_([0, 0, 0], "m")
    >>> q_alice = op(u.Quantity(1, "s"), q_friend)
    >>> q_alice
    (Quantity(Array(1, dtype=int32, weak_type=True), unit='s'),
     CartesianPos3D( x=Quantity(-10., unit='m'), y=Quantity(0., unit='m'),
                     z=Quantity(0., unit='m')
    ))

    """
    return api.frame_transform_op(to_frame, from_frame).inverse  # pylint: disable=W1114
