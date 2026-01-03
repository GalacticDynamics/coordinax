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
    >>> import coordinax.vecs as cxv

    >>> cxv.time_derivative_vector_type(cxv.CartesianPos1D)
    <class 'coordinax...CartesianVel1D'>

    >>> cxv.time_derivative_vector_type(cxv.CartesianVel1D)
    <class 'coordinax...CartesianAcc1D'>

    >>> cxv.time_derivative_vector_type(cxv.RadialPos)
    <class 'coordinax...RadialVel'>

    >>> cxv.time_derivative_vector_type(cxv.RadialVel)
    <class 'coordinax...RadialAcc'>

    >>> cxv.time_derivative_vector_type(cxv.CartesianPos2D)
    <class 'coordinax...CartesianVel2D'>

    >>> cxv.time_derivative_vector_type(cxv.CartesianVel2D)
    <class 'coordinax...CartesianAcc2D'>

    >>> cxv.time_derivative_vector_type(cxv.PolarPos)
    <class 'coordinax...PolarVel'>

    >>> cxv.time_derivative_vector_type(cxv.PolarVel)
    <class 'coordinax...PolarAcc'>

    >>> cxv.time_derivative_vector_type(cxv.CartesianPos3D)
    <class 'coordinax...CartesianVel3D'>

    >>> cxv.time_derivative_vector_type(cxv.CartesianVel3D)
    <class 'coordinax...CartesianAcc3D'>

    >>> cxv.time_derivative_vector_type(cxv.CylindricalPos)
    <class 'coordinax...CylindricalVel'>

    >>> cxv.time_derivative_vector_type(cxv.CylindricalVel)
    <class 'coordinax...CylindricalAcc'>

    >>> cxv.time_derivative_vector_type(cxv.SphericalPos)
    <class 'coordinax...SphericalVel'>

    >>> cxv.time_derivative_vector_type(cxv.SphericalVel)
    <class 'coordinax...SphericalAcc'>

    >>> cxv.time_derivative_vector_type(cxv.MathSphericalPos)
    <class 'coordinax...MathSphericalVel'>

    >>> cxv.time_derivative_vector_type(cxv.MathSphericalVel)
    <class 'coordinax...MathSphericalAcc'>

    >>> cxv.time_derivative_vector_type(cxv.LonLatSphericalPos)
    <class 'coordinax...LonLatSphericalVel'>

    >>> cxv.time_derivative_vector_type(cxv.LonLatSphericalVel)
    <class 'coordinax...LonLatSphericalAcc'>

    >>> cxv.time_derivative_vector_type(cxv.LonCosLatSphericalVel)
    <class 'coordinax...LonLatSphericalAcc'>

    >>> cxv.time_derivative_vector_type(cxv.ProlateSpheroidalPos)
    <class 'coordinax...ProlateSpheroidalVel'>

    >>> cxv.time_derivative_vector_type(cxv.ProlateSpheroidalVel)
    <class 'coordinax...ProlateSpheroidalAcc'>

    >>> cxv.time_derivative_vector_type(cxv.CartesianPosND)
    <class 'coordinax...CartesianVelND'>

    >>> cxv.time_derivative_vector_type(cxv.CartesianVelND)
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
    >>> import coordinax.vecs as cxv

    >>> cxv.time_antiderivative_vector_type(cxv.CartesianVel1D)
    <class 'coordinax...CartesianPos1D'>

    >>> cxv.time_antiderivative_vector_type(cxv.CartesianAcc1D)
    <class 'coordinax...CartesianVel1D'>

    >>> cxv.time_antiderivative_vector_type(cxv.RadialVel)
    <class 'coordinax...RadialPos'>

    >>> cxv.time_antiderivative_vector_type(cxv.RadialAcc)
    <class 'coordinax...RadialVel'>

    >>> cxv.time_antiderivative_vector_type(cxv.CartesianVel2D)
    <class 'coordinax...CartesianPos2D'>

    >>> cxv.time_antiderivative_vector_type(cxv.CartesianAcc2D)
    <class 'coordinax...CartesianVel2D'>

    >>> cxv.time_antiderivative_vector_type(cxv.PolarVel)
    <class 'coordinax...PolarPos'>

    >>> cxv.time_antiderivative_vector_type(cxv.PolarAcc)
    <class 'coordinax...PolarVel'>

    >>> cxv.time_antiderivative_vector_type(cxv.CartesianVel3D)
    <class 'coordinax...CartesianPos3D'>

    >>> cxv.time_antiderivative_vector_type(cxv.CartesianAcc3D)
    <class 'coordinax...CartesianVel3D'>

    >>> cxv.time_antiderivative_vector_type(cxv.CylindricalVel)
    <class 'coordinax...CylindricalPos'>

    >>> cxv.time_antiderivative_vector_type(cxv.CylindricalAcc)
    <class 'coordinax...CylindricalVel'>

    >>> cxv.time_antiderivative_vector_type(cxv.SphericalVel)
    <class 'coordinax...SphericalPos'>

    >>> cxv.time_antiderivative_vector_type(cxv.SphericalAcc)
    <class 'coordinax...SphericalVel'>

    >>> cxv.time_antiderivative_vector_type(cxv.MathSphericalVel)
    <class 'coordinax...MathSphericalPos'>

    >>> cxv.time_antiderivative_vector_type(cxv.MathSphericalAcc)
    <class 'coordinax...MathSphericalVel'>

    >>> cxv.time_antiderivative_vector_type(cxv.LonLatSphericalVel)
    <class 'coordinax...LonLatSphericalPos'>

    >>> cxv.time_antiderivative_vector_type(cxv.LonCosLatSphericalVel)
    <class 'coordinax...LonLatSphericalPos'>

    >>> cxv.time_antiderivative_vector_type(cxv.LonLatSphericalAcc)
    <class 'coordinax...LonLatSphericalVel'>

    >>> cxv.time_antiderivative_vector_type(cxv.ProlateSpheroidalVel)
    <class 'coordinax...ProlateSpheroidalPos'>

    >>> cxv.time_antiderivative_vector_type(cxv.ProlateSpheroidalAcc)
    <class 'coordinax...ProlateSpheroidalVel'>

    >>> cxv.time_antiderivative_vector_type(cxv.CartesianVelND)
    <class 'coordinax...CartesianPosND'>

    >>> cxv.time_antiderivative_vector_type(cxv.CartesianAccND)
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
    >>> import coordinax.vecs as cxv

    >>> cxv.RadialPos.time_nth_derivative_cls(n=0)
    <class 'coordinax...RadialPos'>

    >>> cxv.RadialPos.time_nth_derivative_cls(n=1)
    <class 'coordinax...RadialVel'>

    >>> cxv.RadialPos.time_nth_derivative_cls(n=2)
    <class 'coordinax...RadialAcc'>

    >>> cxv.RadialVel.time_nth_derivative_cls(n=-1)
    <class 'coordinax...RadialPos'>

    >>> cxv.RadialVel.time_nth_derivative_cls(n=0)
    <class 'coordinax...RadialVel'>

    >>> cxv.RadialVel.time_nth_derivative_cls(n=1)
    <class 'coordinax...RadialAcc'>

    >>> cxv.RadialAcc.time_nth_derivative_cls(n=-2)
    <class 'coordinax...RadialPos'>

    >>> cxv.RadialAcc.time_nth_derivative_cls(n=-1)
    <class 'coordinax...RadialVel'>

    >>> cxv.RadialAcc.time_nth_derivative_cls(n=0)
    <class 'coordinax...RadialAcc'>

    """
    raise NotImplementedError  # pragma: no cover
