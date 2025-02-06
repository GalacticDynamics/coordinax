"""Copyright (c) 2023 coordinax maintainers. All rights reserved."""

__all__ = [
    "vector",
    "vconvert",
    "normalize_vector",
    "cartesian_vector_type",
]

from typing import TYPE_CHECKING, Any

from plum import dispatch

if TYPE_CHECKING:
    import coordinax.vecs


@dispatch.abstract
def vconvert(target: type[Any], /, *args: Any, **kwargs: Any) -> Any:
    """Transform the current vector to the target vector.

    See the dispatch implementations for more details. Not all transformations
    result in the target vector type, for example
    ``vconvert(type[Cartesian3DPos], FourVector)`` will return a
    `coordinax.vecs.FourVector` with the spatial part in Cartesian coordinates.
    Likewise, `coordinax.vconvert` on `coordinax.Coordinate` instances will
    transform the contained vectors to the target type, returning a
    `coordinax.Coordinate` instance.

    """
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def normalize_vector(x: Any, /) -> Any:
    """Return the unit vector."""
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def vector(*args: Any, **kwargs: Any) -> Any:
    """Construct a vector given the arguments."""
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def cartesian_vector_type(obj: Any, /) -> "type[coordinax.vecs.AbstractVector]":
    """Return the corresponding Cartesian vector type.

    Examples
    --------
    >>> import coordinax as cx

    >>> cx.vecs.cartesian_vector_type(cx.vecs.RadialPos)
    <class 'coordinax...CartesianPos1D'>

    >>> cx.vecs.cartesian_vector_type(cx.vecs.SphericalPos)
    <class 'coordinax...CartesianPos3D'>

    >>> cx.vecs.cartesian_vector_type(cx.vecs.RadialVel)
    <class 'coordinax...CartesianVel1D'>

    >>> cx.vecs.cartesian_vector_type(cx.vecs.TwoSphereAcc)
    <class 'coordinax...CartesianAcc2D'>

    >>> cx.vecs.cartesian_vector_type(cx.vecs.SphericalVel)
    <class 'coordinax...CartesianVel3D'>

    >>> cx.vecs.cartesian_vector_type(cx.vecs.CartesianAcc3D)
    <class 'coordinax...CartesianAcc3D'>

    >>> cx.vecs.cartesian_vector_type(cx.vecs.SphericalAcc)
    <class 'coordinax...CartesianAcc3D'>

    >>> cx.vecs.cartesian_vector_type(cx.vecs.FourVector)
    <class 'coordinax...CartesianPos3D'>

    >>> cx.vecs.cartesian_vector_type(cx.vecs.CartesianPosND)
    <class 'coordinax...CartesianPosND'>

    >>> cx.vecs.cartesian_vector_type(cx.vecs.CartesianVelND)
    <class 'coordinax...CartesianVelND'>

    >>> cx.vecs.cartesian_vector_type(cx.vecs.CartesianAccND)
    <class 'coordinax...CartesianAccND'>

    """
    raise NotImplementedError  # pragma: no cover
