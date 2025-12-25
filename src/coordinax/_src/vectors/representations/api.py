"""Copyright (c) 2023 coordinax maintainers. All rights reserved."""

__all__ = (
    "cartesian_rep",
    "time_derivative_rep",
    "time_antiderivative_rep",
)

from typing import TYPE_CHECKING, Any

import plum

if TYPE_CHECKING:
    import coordinax.vecs  # noqa: ICN001


@plum.dispatch.abstract
def cartesian_rep(obj: Any, /) -> "coordinax.vecs.AbstractRepresentation":
    """Return the corresponding Cartesian vector type.

    Examples
    --------
    >>> import coordinax.r as cxr

    >>> cxr.cartesian_rep(cxr.CartPos1D)
    coordinax...CartPos1D

    >>> cxr.cartesian_rep(cxr.CartVel1D)
    coordinax...CartVel1D

    >>> cxr.cartesian_rep(cxr.CartAcc1D)
    coordinax...CartAcc1D

    >>> cxr.cartesian_rep(cxr.RadialPos)
    coordinax...CartPos1D

    >>> cxr.cartesian_rep(cxr.SphericalPos)
    coordinax...CartPos3D

    >>> cxr.cartesian_rep(cxr.RadialVel)
    coordinax...CartVel1D

    >>> cxr.cartesian_rep(cxr.TwoSphereAcc)
    coordinax...CartAcc2D

    >>> cxr.cartesian_rep(cxr.SphericalVel)
    coordinax...CartVel3D

    >>> cxr.cartesian_rep(cxr.CartAcc3D)
    coordinax...CartAcc3D

    >>> cxr.cartesian_rep(cxr.SphericalAcc)
    coordinax...CartAcc3D

    >>> cxr.cartesian_rep(cxr.FourVector)
    coordinax...CartPos3D

    >>> cxr.cartesian_rep(cxr.CartPosND)
    coordinax...CartPosND

    >>> cxr.cartesian_rep(cxr.CartVelND)
    coordinax...CartVelND

    >>> cxr.cartesian_rep(cxr.CartAccND)
    coordinax...CartAccND

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def time_derivative_rep(obj: Any, /) -> "coordinax.vecs.AbstractRepresentation":
    """Return the corresponding time derivative vector type.

    Examples
    --------
    >>> import coordinax as cx

    >>> cx.vecs.time_derivative_rep(cx.vecs.CartPos1D)
    coordinax...CartVel1D

    >>> cx.vecs.time_derivative_rep(cx.vecs.CartVel1D)
    coordinax...CartAcc1D

    >>> cx.vecs.time_derivative_rep(cx.vecs.RadialPos)
    coordinax...RadialVel

    >>> cx.vecs.time_derivative_rep(cx.vecs.RadialVel)
    coordinax...RadialAcc

    >>> cx.vecs.time_derivative_rep(cx.vecs.CartPos2D)
    coordinax...CartVel2D

    >>> cx.vecs.time_derivative_rep(cx.vecs.CartVel2D)
    coordinax...CartAcc2D

    >>> cx.vecs.time_derivative_rep(cx.vecs.PolarPos)
    coordinax...PolarVel

    >>> cx.vecs.time_derivative_rep(cx.vecs.PolarVel)
    coordinax...PolarAcc

    >>> cx.vecs.time_derivative_rep(cx.vecs.CartPos3D)
    coordinax...CartVel3D

    >>> cx.vecs.time_derivative_rep(cx.vecs.CartVel3D)
    coordinax...CartAcc3D

    >>> cx.vecs.time_derivative_rep(cx.vecs.CylindricalPos)
    coordinax...CylindricalVel

    >>> cx.vecs.time_derivative_rep(cx.vecs.CylindricalVel)
    coordinax...CylindricalAcc

    >>> cx.vecs.time_derivative_rep(cx.vecs.SphericalPos)
    coordinax...SphericalVel

    >>> cx.vecs.time_derivative_rep(cx.vecs.SphericalVel)
    coordinax...SphericalAcc

    >>> cx.vecs.time_derivative_rep(cx.vecs.MathSphericalPos)
    coordinax...MathSphericalVel

    >>> cx.vecs.time_derivative_rep(cx.vecs.MathSphericalVel)
    coordinax...MathSphericalAcc

    >>> cx.vecs.time_derivative_rep(cx.vecs.LonLatSphericalPos)
    coordinax...LonLatSphericalVel

    >>> cx.vecs.time_derivative_rep(cx.vecs.LonLatSphericalVel)
    coordinax...LonLatSphericalAcc

    >>> cx.vecs.time_derivative_rep(cx.vecs.LonCosLatSphericalVel)
    coordinax...LonLatSphericalAcc

    >>> cx.vecs.time_derivative_rep(cx.vecs.ProlateSpheroidalPos)
    coordinax...ProlateSpheroidalVel

    >>> cx.vecs.time_derivative_rep(cx.vecs.ProlateSpheroidalVel)
    coordinax...ProlateSpheroidalAcc

    >>> cx.vecs.time_derivative_rep(cx.vecs.CartesianND)
    coordinax...CartVelND

    >>> cx.vecs.time_derivative_rep(cx.vecs.CartVelND)
    coordinax...CartAccND

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def time_antiderivative_rep(obj: Any, /) -> "coordinax.vecs.AbstractRepresentation":
    """Return the corresponding time antiderivative vector type.

    Examples
    --------
    >>> import coordinax as cx

    >>> cx.vecs.time_antiderivative_rep(cx.vecs.CartVel1D)
    coordinax...CartPos1D

    >>> cx.vecs.time_antiderivative_rep(cx.vecs.CartAcc1D)
    coordinax...CartVel1D

    >>> cx.vecs.time_antiderivative_rep(cx.vecs.RadialVel)
    coordinax...RadialPos

    >>> cx.vecs.time_antiderivative_rep(cx.vecs.RadialAcc)
    coordinax...RadialVel

    >>> cx.vecs.time_antiderivative_rep(cx.vecs.CartVel2D)
    coordinax...CartPos2D

    >>> cx.vecs.time_antiderivative_rep(cx.vecs.CartAcc2D)
    coordinax...CartVel2D

    >>> cx.vecs.time_antiderivative_rep(cx.vecs.PolarVel)
    coordinax...PolarPos

    >>> cx.vecs.time_antiderivative_rep(cx.vecs.PolarAcc)
    coordinax...PolarVel

    >>> cx.vecs.time_antiderivative_rep(cx.vecs.CartVel3D)
    coordinax...CartPos3D

    >>> cx.vecs.time_antiderivative_rep(cx.vecs.CartAcc3D)
    coordinax...CartVel3D

    >>> cx.vecs.time_antiderivative_rep(cx.vecs.CylindricalVel)
    coordinax...CylindricalPos

    >>> cx.vecs.time_antiderivative_rep(cx.vecs.CylindricalAcc)
    coordinax...CylindricalVel

    >>> cx.vecs.time_antiderivative_rep(cx.vecs.SphericalVel)
    coordinax...SphericalPos

    >>> cx.vecs.time_antiderivative_rep(cx.vecs.SphericalAcc)
    coordinax...SphericalVel

    >>> cx.vecs.time_antiderivative_rep(cx.vecs.MathSphericalVel)
    coordinax...MathSphericalPos

    >>> cx.vecs.time_antiderivative_rep(cx.vecs.MathSphericalAcc)
    coordinax...MathSphericalVel

    >>> cx.vecs.time_antiderivative_rep(cx.vecs.LonLatSphericalVel)
    coordinax...LonLatSphericalPos

    >>> cx.vecs.time_antiderivative_rep(cx.vecs.LonCosLatSphericalVel)
    coordinax...LonLatSphericalPos

    >>> cx.vecs.time_antiderivative_rep(cx.vecs.LonLatSphericalAcc)
    coordinax...LonLatSphericalVel

    >>> cx.vecs.time_antiderivative_rep(cx.vecs.ProlateSpheroidalVel)
    coordinax...ProlateSpheroidalPos

    >>> cx.vecs.time_antiderivative_rep(cx.vecs.ProlateSpheroidalAcc)
    coordinax...ProlateSpheroidalVel

    >>> cx.vecs.time_antiderivative_rep(cx.vecs.CartVelND)
    coordinax...CartesianND

    >>> cx.vecs.time_antiderivative_rep(cx.vecs.CartAccND)
    coordinax...CartVelND

    """
    raise NotImplementedError  # pragma: no cover
