"""Representation of coordinates in different systems."""

__all__: list[str] = []

from collections.abc import Mapping
from typing import Any

from plum import dispatch

import quaxed.numpy as jnp

from .core import KinematicSpace
from coordinax._src.vectors.api import vconvert, vector
from coordinax._src.vectors.base_acc import AbstractAcc
from coordinax._src.vectors.base_pos import AbstractPos
from coordinax._src.vectors.base_vel import AbstractVel

# ===============================================================
# Constructor dispatches


@KinematicSpace.from_.dispatch(precedence=1)
def from_(  # TODO: KinematicSpace[PosT] for obj, return -- plum#212
    cls: type[KinematicSpace], obj: KinematicSpace, /
) -> KinematicSpace:
    """Construct a Space, returning the KinematicSpace.

    Examples
    --------
    >>> import coordinax as cx

    >>> q = cx.KinematicSpace.from_(cx.CartesianPos3D.from_([1, 2, 3], "m"))
    >>> cx.KinematicSpace.from_(q) is q
    True

    """
    return obj


@KinematicSpace.from_.dispatch  # TODO: KinematicSpace[PosT] for obj, return -- plum#212
def from_(cls: type[KinematicSpace], obj: AbstractPos, /) -> KinematicSpace:
    """Construct a `coordinax.KinematicSpace` from a `coordinax.AbstractPos`.

    Examples
    --------
    >>> import coordinax as cx

    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "m")
    >>> w = cx.KinematicSpace.from_(q)
    >>> w
    KinematicSpace({ 'length': CartesianPos3D( ... ) })

    """
    return KinematicSpace(length=obj)


@KinematicSpace.from_.dispatch  # TODO: KinematicSpace[PosT] for q, return -- plum#212
def from_(
    cls: type[KinematicSpace], q: AbstractPos, p: AbstractVel, /
) -> KinematicSpace:
    """Construct a `coordinax.KinematicSpace` from a `coordinax.AbstractPos`.

    Examples
    --------
    >>> import coordinax as cx

    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "m")
    >>> p = cx.CartesianVel3D.from_([4, 5, 6], "m/s")
    >>> w = cx.KinematicSpace.from_(q, p)
    >>> w
    KinematicSpace({ 'length': CartesianPos3D( ... ), 'speed': CartesianVel3D( ... ) })

    """
    return cls(length=q, speed=p)


@KinematicSpace.from_.dispatch  # TODO: KinematicSpace[PosT] for obj, return -- plum#212
def from_(
    cls: type[KinematicSpace], q: AbstractPos, p: AbstractVel, a: AbstractAcc, /
) -> KinematicSpace:
    """Construct a `coordinax.KinematicSpace` from a `coordinax.AbstractPos`.

    Examples
    --------
    >>> import coordinax as cx

    >>> q = cx.vecs.CartesianPos3D.from_([1, 2, 3], "m")
    >>> p = cx.vecs.CartesianVel3D.from_([4, 5, 6], "m/s")
    >>> a = cx.vecs.CartesianAcc3D.from_([7, 8, 9], "m/s2")
    >>> w = cx.KinematicSpace.from_(q, p, a)
    >>> w
    KinematicSpace({ 'length': CartesianPos3D( ... ),
            'speed': CartesianVel3D( ... ),
            'acceleration': CartesianAcc3D( ... ) })

    """
    return cls(length=q, speed=p, acceleration=a)


@KinematicSpace.from_.dispatch
def from_(cls: type[KinematicSpace], obj: Mapping[str, Any]) -> KinematicSpace:
    """Construct a KinematicSpace from a Mapping.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> space = cx.KinematicSpace.from_({ 'length': u.Quantity([1, 2, 3], "m") })
    >>> print(space)
    KinematicSpace({
       'length': <CartesianPos3D: (x, y, z) [m]
           [1 2 3]>
    })

    """
    return cls({k: vector(v) for k, v in obj.items()})


# ===============================================================
# Vector API dispatches


@dispatch
def vconvert(
    target: type[AbstractPos], current: AbstractPos, space: KinematicSpace, /
) -> AbstractPos:
    """Convert a position to the target type, with a KinematicSpace context.

    Examples
    --------
    >>> import coordinax as cx

    >>> space = cx.KinematicSpace(length=cx.CartesianPos3D.from_([1, 2, 3], "m"),
    ...                  speed=cx.CartesianVel3D.from_([4, 5, 6], "m/s"))

    >>> cx.vconvert(cx.SphericalPos, space["length"], space)
    SphericalPos( ... )

    """
    return vconvert(target, current)  # space is unnecessary


@dispatch
def vconvert(
    target: type[AbstractVel], current: AbstractVel, space: KinematicSpace, /
) -> AbstractVel:
    """Convert a velocity to the target type, with a KinematicSpace context.

    Examples
    --------
    >>> import coordinax as cx

    >>> space = cx.KinematicSpace(length=cx.CartesianPos3D.from_([1, 2, 3], "m"),
    ...                  speed=cx.CartesianVel3D.from_([4, 5, 6], "m/s"))

    >>> cx.vconvert(cx.SphericalVel, space["speed"], space)
    SphericalVel( ... )

    """
    return vconvert(target, current, space["length"])


@dispatch
def vconvert(
    target: type[AbstractAcc], current: AbstractAcc, space: KinematicSpace, /
) -> AbstractAcc:
    """Convert an acceleration to the target type, with a KinematicSpace context.

    Examples
    --------
    >>> import coordinax as cx

    >>> space = cx.KinematicSpace(length=cx.CartesianPos3D.from_([1, 2, 3], "m"),
    ...                  speed=cx.CartesianVel3D.from_([4, 5, 6], "m/s"),
    ...                  acceleration=cx.vecs.CartesianAcc3D.from_([7, 8, 9], "m/s2"))

    >>> cx.vconvert(cx.vecs.SphericalAcc, space["acceleration"], space)
    SphericalAcc( ... )

    """
    return vconvert(target, current, space["speed"], space["length"])


# =============================================================== Temporary
# These functions are very similar to `vconvert`, but I don't think this is
# the best API. Until we figure out a better way to do this, we'll keep these
# functions here.


@dispatch  # TODO: KinematicSpace[PosT] for all types -- plum#212
def vconvert(target: type[AbstractPos], space: KinematicSpace, /) -> KinematicSpace:
    """Represent the current vector to the target vector."""
    return type(space)({k: temp_vconvert(target, v, space) for k, v in space.items()})


# TODO: should this be moved to a different file?
@dispatch
def temp_vconvert(
    target: type[AbstractPos], current: AbstractPos, space: KinematicSpace, /
) -> AbstractPos:
    """Transform of Poss."""
    return vconvert(target, current)  # space is unnecessary


# TODO: should this be moved to a different file?
@dispatch
def temp_vconvert(
    target: type[AbstractPos], current: AbstractVel, space: KinematicSpace, /
) -> AbstractVel:
    """Transform of Velocities."""
    q, p = jnp.broadcast_arrays(space["length"], current)
    return vconvert(target.time_derivative_cls, p, q)


# TODO: should this be moved to a different file?
@dispatch
def temp_vconvert(
    target: type[AbstractPos], current: AbstractAcc, space: KinematicSpace, /
) -> AbstractAcc:
    """Transform of Accs."""
    q, p, a = jnp.broadcast_arrays(space["length"], space["speed"], current)
    return vconvert(target.time_nth_derivative_cls(2), a, p, q)
