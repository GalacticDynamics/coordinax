"""Representation of coordinates in different systems."""

__all__: list[str] = []

from collections.abc import Mapping
from typing import Any

from plum import dispatch

from .core import Space
from coordinax._src.vectors.api import vconvert
from coordinax._src.vectors.base import AbstractVector
from coordinax._src.vectors.base_acc import AbstractAcc
from coordinax._src.vectors.base_pos import AbstractPos
from coordinax._src.vectors.base_vel import AbstractVel

# ===============================================================
# Constructor dispatches


# This dispatch is needed because a Space is both a Map and a Vector.
@dispatch(precedence=1)
def vector(cls: type[Space], obj: Space, /) -> Space:
    """Construct a Space, returning the Space.

    Examples
    --------
    >>> import coordinax as cx

    >>> q = cx.Space.from_(cx.CartesianPos3D.from_([1, 2, 3], "m"))
    >>> cx.Space.from_(q) is q
    True

    """
    return obj


@dispatch
def vector(cls: type[Space], obj: AbstractPos, /) -> Space:
    """Construct a `coordinax.Space` from a `coordinax.AbstractPos`.

    Examples
    --------
    >>> import coordinax as cx

    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "m")
    >>> w = cx.Space.from_(q)
    >>> w
    Space({ 'length': CartesianPos3D( ... ) })

    """
    return cls(length=obj)


@dispatch
def vector(cls: type[Space], q: AbstractPos, p: AbstractVel, /) -> Space:
    """Construct a `coordinax.Space` from a `coordinax.AbstractPos`.

    Examples
    --------
    >>> import coordinax as cx

    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "m")
    >>> p = cx.CartesianVel3D.from_([4, 5, 6], "m/s")
    >>> w = cx.Space.from_(q, p)
    >>> w
    Space({ 'length': CartesianPos3D( ... ), 'speed': CartesianVel3D( ... ) })

    """
    return cls(length=q, speed=p)


@dispatch
def vector(
    cls: type[Space], q: AbstractPos, p: AbstractVel, a: AbstractAcc, /
) -> Space:
    """Construct a `coordinax.Space` from a `coordinax.AbstractPos`.

    Examples
    --------
    >>> import coordinax as cx

    >>> q = cx.vecs.CartesianPos3D.from_([1, 2, 3], "m")
    >>> p = cx.vecs.CartesianVel3D.from_([4, 5, 6], "m/s")
    >>> a = cx.vecs.CartesianAcc3D.from_([7, 8, 9], "m/s2")
    >>> w = cx.Space.from_(q, p, a)
    >>> w
    Space({ 'length': CartesianPos3D( ... ),
            'speed': CartesianVel3D( ... ),
            'acceleration': CartesianAcc3D( ... ) })

    """
    return cls(length=q, speed=p, acceleration=a)


@dispatch
def vector(
    cls: type[Space],
    obj: Mapping[str, Any],
) -> Space:
    """Construct a Space from a Mapping.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> space = cx.Space.from_({ 'length': u.Quantity([1, 2, 3], "m") })
    >>> print(space)
    Space({
       'length': <CartesianPos3D (x[m], y[m], z[m])
           [1 2 3]>
    })

    """
    return cls({k: vector(v) for k, v in obj.items()})


# ===============================================================
# Vector API dispatches


@dispatch
def vconvert(
    target: type[AbstractPos], current: AbstractPos, space: Space, /
) -> AbstractPos:
    """Convert a position to the target type, with a Space context.

    Examples
    --------
    >>> import coordinax as cx

    >>> space = cx.Space(length=cx.CartesianPos3D.from_([1, 2, 3], "m"),
    ...                  speed=cx.CartesianVel3D.from_([4, 5, 6], "m/s"))

    >>> cx.vconvert(cx.SphericalPos, space["length"], space)
    SphericalPos( ... )

    """
    return vconvert(target, current)  # space is unnecessary


@dispatch
def vconvert(
    target: type[AbstractVel], current: AbstractVel, space: Space, /
) -> AbstractVel:
    """Convert a velocty to the target type, with a Space context.

    Examples
    --------
    >>> import coordinax as cx

    >>> space = cx.Space(length=cx.CartesianPos3D.from_([1, 2, 3], "m"),
    ...                  speed=cx.CartesianVel3D.from_([4, 5, 6], "m/s"))

    >>> cx.vconvert(cx.SphericalVel, space["speed"], space)
    SphericalVel( ... )

    """
    return vconvert(target, current, space["length"])


@dispatch
def vconvert(
    target: type[AbstractAcc], current: AbstractAcc, space: Space, /
) -> AbstractAcc:
    """Convert an acceleration to the target type, with a Space context.

    Examples
    --------
    >>> import coordinax as cx

    >>> space = cx.Space(length=cx.CartesianPos3D.from_([1, 2, 3], "m"),
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


@dispatch
def vconvert(target: type[AbstractVector], space: Space, /) -> Space:
    """Represent the current vector to the target vector."""
    return type(space)({k: temp_vconvert(target, v, space) for k, v in space.items()})


# TODO: should this be moved to a different file?
@dispatch
def temp_vconvert(
    target: type[AbstractPos], current: AbstractPos, space: Space, /
) -> AbstractPos:
    """Transform of Poss."""
    return vconvert(target, current)  # space is unnecessary


# TODO: should this be moved to a different file?
@dispatch
def temp_vconvert(
    target: type[AbstractPos], current: AbstractVel, space: Space, /
) -> AbstractVel:
    """Transform of Velocities."""
    return vconvert(target.differential_cls, current, space["length"])


# TODO: should this be moved to a different file?
@dispatch
def temp_vconvert(
    target: type[AbstractPos], current: AbstractAcc, space: Space, /
) -> AbstractAcc:
    """Transform of Accs."""
    return vconvert(
        target.differential_cls.differential_cls,
        current,
        space["speed"],
        space["length"],
    )
