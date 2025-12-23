"""Copyright (c) 2023 coordinax maintainers. All rights reserved."""

__all__ = (
    "vector",
    "normalize_vector",
    "cartesian_vector_type",
    "time_derivative_vector_type",
    "time_antiderivative_vector_type",
    "time_nth_derivative_vector_type",
)

from typing import TYPE_CHECKING, Any

from plum import dispatch

if TYPE_CHECKING:
    import coordinax.vecs  # noqa: ICN001


@dispatch.abstract
def vector(*args: Any, **kwargs: Any) -> Any:
    """Construct a vector given the arguments."""
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def normalize_vector(x: Any, /) -> Any:
    """Return the unit vector."""
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def cartesian_vector_type(obj: Any, /) -> "type[coordinax.vecs.AbstractVector]":
    """Return the corresponding Cartesian vector type.

    Examples
    --------
    >>> import coordinax.vecs as cxv

    >>> cxv.cartesian_vector_type(cxv.RadialPos)
    <class 'coordinax...CartesianPos1D'>

    >>> cxv.cartesian_vector_type(cxv.SphericalPos)
    <class 'coordinax...CartesianPos3D'>

    >>> cxv.cartesian_vector_type(cxv.RadialVel)
    <class 'coordinax...CartesianVel1D'>

    >>> cxv.cartesian_vector_type(cxv.TwoSphereAcc)
    <class 'coordinax...CartesianAcc2D'>

    >>> cxv.cartesian_vector_type(cxv.SphericalVel)
    <class 'coordinax...CartesianVel3D'>

    >>> cxv.cartesian_vector_type(cxv.CartesianAcc3D)
    <class 'coordinax...CartesianAcc3D'>

    >>> cxv.cartesian_vector_type(cxv.SphericalAcc)
    <class 'coordinax...CartesianAcc3D'>

    >>> cxv.cartesian_vector_type(cxv.FourVector)
    <class 'coordinax...CartesianPos3D'>

    >>> cxv.cartesian_vector_type(cxv.CartesianPosND)
    <class 'coordinax...CartesianPosND'>

    >>> cxv.cartesian_vector_type(cxv.CartesianVelND)
    <class 'coordinax...CartesianVelND'>

    >>> cxv.cartesian_vector_type(cxv.CartesianAccND)
    <class 'coordinax...CartesianAccND'>

    """
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def time_derivative_vector_type(obj: Any, /) -> "type[coordinax.vecs.AbstractVector]":
    """Return the corresponding time derivative vector type.

    Examples
    --------
    >>> import coordinax as cx

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.CartesianPos1D)
    <class 'coordinax...CartesianVel1D'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.CartesianVel1D)
    <class 'coordinax...CartesianAcc1D'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.RadialPos)
    <class 'coordinax...RadialVel'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.RadialVel)
    <class 'coordinax...RadialAcc'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.CartesianPos2D)
    <class 'coordinax...CartesianVel2D'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.CartesianVel2D)
    <class 'coordinax...CartesianAcc2D'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.PolarPos)
    <class 'coordinax...PolarVel'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.PolarVel)
    <class 'coordinax...PolarAcc'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.CartesianPos3D)
    <class 'coordinax...CartesianVel3D'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.CartesianVel3D)
    <class 'coordinax...CartesianAcc3D'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.CylindricalPos)
    <class 'coordinax...CylindricalVel'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.CylindricalVel)
    <class 'coordinax...CylindricalAcc'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.SphericalPos)
    <class 'coordinax...SphericalVel'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.SphericalVel)
    <class 'coordinax...SphericalAcc'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.MathSphericalPos)
    <class 'coordinax...MathSphericalVel'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.MathSphericalVel)
    <class 'coordinax...MathSphericalAcc'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.LonLatSphericalPos)
    <class 'coordinax...LonLatSphericalVel'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.LonLatSphericalVel)
    <class 'coordinax...LonLatSphericalAcc'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.LonCosLatSphericalVel)
    <class 'coordinax...LonLatSphericalAcc'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.ProlateSpheroidalPos)
    <class 'coordinax...ProlateSpheroidalVel'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.ProlateSpheroidalVel)
    <class 'coordinax...ProlateSpheroidalAcc'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.CartesianPosND)
    <class 'coordinax...CartesianVelND'>

    >>> cx.vecs.time_derivative_vector_type(cx.vecs.CartesianVelND)
    <class 'coordinax...CartesianAccND'>

    """
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def time_antiderivative_vector_type(
    obj: Any, /
) -> "type[coordinax.vecs.AbstractVector]":
    """Return the corresponding time antiderivative vector type.

    Examples
    --------
    >>> import coordinax as cx

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.CartesianVel1D)
    <class 'coordinax...CartesianPos1D'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.CartesianAcc1D)
    <class 'coordinax...CartesianVel1D'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.RadialVel)
    <class 'coordinax...RadialPos'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.RadialAcc)
    <class 'coordinax...RadialVel'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.CartesianVel2D)
    <class 'coordinax...CartesianPos2D'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.CartesianAcc2D)
    <class 'coordinax...CartesianVel2D'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.PolarVel)
    <class 'coordinax...PolarPos'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.PolarAcc)
    <class 'coordinax...PolarVel'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.CartesianVel3D)
    <class 'coordinax...CartesianPos3D'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.CartesianAcc3D)
    <class 'coordinax...CartesianVel3D'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.CylindricalVel)
    <class 'coordinax...CylindricalPos'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.CylindricalAcc)
    <class 'coordinax...CylindricalVel'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.SphericalVel)
    <class 'coordinax...SphericalPos'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.SphericalAcc)
    <class 'coordinax...SphericalVel'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.MathSphericalVel)
    <class 'coordinax...MathSphericalPos'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.MathSphericalAcc)
    <class 'coordinax...MathSphericalVel'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.LonLatSphericalVel)
    <class 'coordinax...LonLatSphericalPos'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.LonCosLatSphericalVel)
    <class 'coordinax...LonLatSphericalPos'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.LonLatSphericalAcc)
    <class 'coordinax...LonLatSphericalVel'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.ProlateSpheroidalVel)
    <class 'coordinax...ProlateSpheroidalPos'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.ProlateSpheroidalAcc)
    <class 'coordinax...ProlateSpheroidalVel'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.CartesianVelND)
    <class 'coordinax...CartesianPosND'>

    >>> cx.vecs.time_antiderivative_vector_type(cx.vecs.CartesianAccND)
    <class 'coordinax...CartesianVelND'>

    """
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def time_nth_derivative_vector_type(
    obj: Any, /, *, n: int
) -> "type[coordinax.vecs.AbstractVector]":
    """Return the corresponding time nth derivative vector type.

    Examples
    --------
    >>> import coordinax as cx

    >>> cx.vecs.RadialPos.time_nth_derivative_cls(n=0)
    <class 'coordinax...RadialPos'>

    >>> cx.vecs.RadialPos.time_nth_derivative_cls(n=1)
    <class 'coordinax...RadialVel'>

    >>> cx.vecs.RadialPos.time_nth_derivative_cls(n=2)
    <class 'coordinax...RadialAcc'>

    >>> cx.vecs.RadialVel.time_nth_derivative_cls(n=-1)
    <class 'coordinax...RadialPos'>

    >>> cx.vecs.RadialVel.time_nth_derivative_cls(n=0)
    <class 'coordinax...RadialVel'>

    >>> cx.vecs.RadialVel.time_nth_derivative_cls(n=1)
    <class 'coordinax...RadialAcc'>

    >>> cx.vecs.RadialAcc.time_nth_derivative_cls(n=-2)
    <class 'coordinax...RadialPos'>

    >>> cx.vecs.RadialAcc.time_nth_derivative_cls(n=-1)
    <class 'coordinax...RadialVel'>

    >>> cx.vecs.RadialAcc.time_nth_derivative_cls(n=0)
    <class 'coordinax...RadialAcc'>

    """
    raise NotImplementedError  # pragma: no cover
