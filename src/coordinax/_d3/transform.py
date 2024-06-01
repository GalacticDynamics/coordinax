"""Representation of coordinates in different systems."""

__all__: list[str] = []

from typing import Any

from plum import dispatch

import quaxed.array_api as xp
from unxt import Quantity

from .base import Abstract3DVector, Abstract3DVectorDifferential
from .builtin import (
    Cartesian3DVector,
    CartesianDifferential3D,
    CylindricalDifferential,
    CylindricalVector,
)
from .sphere import (
    AbstractSphericalVector,
    LonCosLatSphericalDifferential,
    LonLatSphericalDifferential,
    LonLatSphericalVector,
    MathSphericalDifferential,
    MathSphericalVector,
    SphericalDifferential,
    SphericalVector,
)
from coordinax._base_pos import AbstractPosition

###############################################################################
# 3D


@dispatch
def represent_as(
    current: Abstract3DVector, target: type[Abstract3DVector], /, **kwargs: Any
) -> Abstract3DVector:
    """Abstract3DVector -> Cartesian3D -> Abstract3DVector."""
    return represent_as(represent_as(current, Cartesian3DVector), target)


@dispatch.multi(
    (Cartesian3DVector, type[Cartesian3DVector]),
    (CylindricalVector, type[CylindricalVector]),
    (SphericalVector, type[SphericalVector]),
    (LonLatSphericalVector, type[LonLatSphericalVector]),
    (MathSphericalVector, type[MathSphericalVector]),
)
def represent_as(
    current: Abstract3DVector, target: type[Abstract3DVector], /, **kwargs: Any
) -> Abstract3DVector:
    """Self transforms for 3D vectors.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    Cartesian to Cartesian:

    >>> vec = cx.Cartesian3DVector.constructor(Quantity([1, 2, 3], "kpc"))
    >>> cx.represent_as(vec, cx.Cartesian3DVector) is vec
    True

    Cylindrical to Cylindrical:

    >>> vec = cx.CylindricalVector(rho=Quantity(1, "kpc"), phi=Quantity(2, "deg"),
    ...                            z=Quantity(3, "kpc"))
    >>> cx.represent_as(vec, cx.CylindricalVector) is vec
    True

    Spherical to Spherical:

    >>> vec = cx.SphericalVector(r=Quantity(1, "kpc"), theta=Quantity(2, "deg"),
    ...                          phi=Quantity(3, "deg"))
    >>> cx.represent_as(vec, cx.SphericalVector) is vec
    True

    LonLatSpherical to LonLatSpherical:

    >>> vec = cx.LonLatSphericalVector(lon=Quantity(1, "deg"), lat=Quantity(2, "deg"),
    ...                                distance=Quantity(3, "kpc"))
    >>> cx.represent_as(vec, cx.LonLatSphericalVector) is vec
    True

    MathSpherical to MathSpherical:

    >>> vec = cx.MathSphericalVector(r=Quantity(1, "kpc"), theta=Quantity(2, "deg"),
    ...                              phi=Quantity(3, "deg"))
    >>> cx.represent_as(vec, cx.MathSphericalVector) is vec
    True

    """
    return current


@dispatch.multi(
    (CartesianDifferential3D, type[CartesianDifferential3D], AbstractPosition),
    (CylindricalDifferential, type[CylindricalDifferential], AbstractPosition),
    (SphericalDifferential, type[SphericalDifferential], AbstractPosition),
    (LonLatSphericalDifferential, type[LonLatSphericalDifferential], AbstractPosition),
    (
        LonCosLatSphericalDifferential,
        type[LonCosLatSphericalDifferential],
        AbstractPosition,
    ),
    (MathSphericalDifferential, type[MathSphericalDifferential], AbstractPosition),
)
def represent_as(
    current: Abstract3DVectorDifferential,
    target: type[Abstract3DVectorDifferential],
    position: AbstractPosition,
    /,
    **kwargs: Any,
) -> Abstract3DVectorDifferential:
    """Self transforms for 3D differentials.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    For these transformations the position does not matter since the
    self-transform returns the differential unchanged.

    >>> vec = cx.Cartesian3DVector.constructor(Quantity([1, 2, 3], "kpc"))

    Cartesian to Cartesian differential:

    >>> dif = cx.CartesianDifferential3D.constructor(Quantity([1, 2, 3], "km/s"))
    >>> cx.represent_as(dif, cx.CartesianDifferential3D, vec) is dif
    True

    Cylindrical to Cylindrical differential:

    >>> dif = cx.CylindricalDifferential(d_rho=Quantity(1, "km/s"),
    ...                                  d_phi=Quantity(2, "mas/yr"),
    ...                                  d_z=Quantity(3, "km/s"))
    >>> cx.represent_as(dif, cx.CylindricalDifferential, vec) is dif
    True

    Spherical to Spherical differential:

    >>> dif = cx.SphericalDifferential(d_r=Quantity(1, "km/s"),
    ...                                d_theta=Quantity(2, "mas/yr"),
    ...                                d_phi=Quantity(3, "mas/yr"))
    >>> cx.represent_as(dif, cx.SphericalDifferential, vec) is dif
    True

    LonLatSpherical to LonLatSpherical differential:

    >>> dif = cx.LonLatSphericalDifferential(d_lon=Quantity(1, "mas/yr"),
    ...                                      d_lat=Quantity(2, "mas/yr"),
    ...                                      d_distance=Quantity(3, "km/s"))
    >>> cx.represent_as(dif, cx.LonLatSphericalDifferential, vec) is dif
    True

    LonCosLatSpherical to LonCosLatSpherical differential:

    >>> dif = cx.LonCosLatSphericalDifferential(d_lon_coslat=Quantity(1, "mas/yr"),
    ...                                         d_lat=Quantity(2, "mas/yr"),
    ...                                         d_distance=Quantity(3, "km/s"))
    >>> cx.represent_as(dif, cx.LonCosLatSphericalDifferential, vec) is dif
    True

    MathSpherical to MathSpherical differential:

    >>> dif = cx.MathSphericalDifferential(d_r=Quantity(1, "km/s"),
    ...                                    d_theta=Quantity(2, "mas/yr"),
    ...                                    d_phi=Quantity(3, "mas/yr"))
    >>> cx.represent_as(dif, cx.MathSphericalDifferential, vec) is dif
    True

    """
    return current


# =============================================================================
# Cartesian3DVector


@dispatch
def represent_as(
    current: Cartesian3DVector, target: type[CylindricalVector], /, **kwargs: Any
) -> CylindricalVector:
    """Cartesian3DVector -> CylindricalVector.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.Cartesian3DVector.constructor(Quantity([1, 2, 3], "km"))
    >>> print(cx.represent_as(vec, cx.CylindricalVector))
    <CylindricalVector (rho[km], phi[rad], z[km])
        [2.236 1.107 3.   ]>

    """
    rho = xp.sqrt(current.x**2 + current.y**2)
    phi = xp.atan2(current.y, current.x)
    return target(rho=rho, phi=phi, z=current.z)


@dispatch
def represent_as(
    current: Cartesian3DVector, target: type[SphericalVector], /, **kwargs: Any
) -> SphericalVector:
    """Cartesian3DVector -> SphericalVector.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.Cartesian3DVector.constructor(Quantity([1, 2, 3], "km"))
    >>> print(cx.represent_as(vec, cx.SphericalVector))
    <SphericalVector (r[km], theta[rad], phi[rad])
        [3.742 0.641 1.107]>

    """
    r = xp.sqrt(current.x**2 + current.y**2 + current.z**2)
    theta = xp.acos(current.z / r)
    phi = xp.atan2(current.y, current.x)
    return target(r=r, theta=theta, phi=phi)


@dispatch.multi(
    (Cartesian3DVector, type[LonLatSphericalVector]),
    (Cartesian3DVector, type[MathSphericalVector]),
)
def represent_as(
    current: Cartesian3DVector, target: type[AbstractSphericalVector], /, **kwargs: Any
) -> AbstractSphericalVector:
    """Cartesian3DVector -> AbstractSphericalVector.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.Cartesian3DVector.constructor(Quantity([1, 2, 3], "km"))

    >>> print(cx.represent_as(vec, cx.LonLatSphericalVector))
    <LonLatSphericalVector (lon[rad], lat[deg], distance[km])
        [ 1.107 53.301  3.742]>

    >>> print(cx.represent_as(vec, cx.MathSphericalVector))
    <MathSphericalVector (r[km], theta[rad], phi[rad])
        [3.742 1.107 0.641]>

    """
    return represent_as(represent_as(current, SphericalVector), target)


# =============================================================================
# CylindricalVector


@dispatch
def represent_as(
    current: CylindricalVector, target: type[Cartesian3DVector], /, **kwargs: Any
) -> Cartesian3DVector:
    """CylindricalVector -> Cartesian3DVector.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.CylindricalVector(rho=Quantity(1., "kpc"), phi=Quantity(90, "deg"),
    ...                            z=Quantity(1, "kpc"))
    >>> print(cx.represent_as(vec, cx.Cartesian3DVector))
    <Cartesian3DVector (x[kpc], y[kpc], z[kpc])
        [-4.371e-08  1.000e+00  1.000e+00]>

    """
    x = current.rho * xp.cos(current.phi)
    y = current.rho * xp.sin(current.phi)
    z = current.z
    return target(x=x, y=y, z=z)


@dispatch
def represent_as(
    current: CylindricalVector, target: type[SphericalVector], /, **kwargs: Any
) -> SphericalVector:
    """CylindricalVector -> SphericalVector.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.CylindricalVector(rho=Quantity(1., "kpc"), phi=Quantity(90, "deg"),
    ...                            z=Quantity(1, "kpc"))
    >>> print(cx.represent_as(vec, cx.SphericalVector))
    <SphericalVector (r[kpc], theta[rad], phi[deg])
        [ 1.414  0.785 90.   ]>

    """
    r = xp.sqrt(current.rho**2 + current.z**2)
    theta = xp.acos(current.z / r)
    return target(r=r, theta=theta, phi=current.phi)


@dispatch.multi(
    (CylindricalVector, type[LonLatSphericalVector]),
    (CylindricalVector, type[MathSphericalVector]),
)
def represent_as(
    current: CylindricalVector, target: type[AbstractSphericalVector], /, **kwargs: Any
) -> AbstractSphericalVector:
    """CylindricalVector -> AbstractSphericalVector.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.CylindricalVector(rho=Quantity(1., "kpc"), phi=Quantity(90, "deg"),
    ...                            z=Quantity(1, "kpc"))

    >>> print(cx.represent_as(vec, cx.LonLatSphericalVector))
    <LonLatSphericalVector (lon[deg], lat[deg], distance[kpc])
        [90.    45.     1.414]>

    >>> print(cx.represent_as(vec, cx.MathSphericalVector))
    <MathSphericalVector (r[kpc], theta[deg], phi[rad])
        [ 1.414 90.     0.785]>

    """
    return represent_as(represent_as(current, SphericalVector), target)


# =============================================================================
# SphericalVector


@dispatch
def represent_as(
    current: SphericalVector, target: type[Cartesian3DVector], /, **kwargs: Any
) -> Cartesian3DVector:
    """SphericalVector -> Cartesian3DVector.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.SphericalVector(r=Quantity(1., "kpc"), theta=Quantity(90, "deg"),
    ...                          phi=Quantity(90, "deg"))
    >>> print(cx.represent_as(vec, cx.Cartesian3DVector))
    <Cartesian3DVector (x[kpc], y[kpc], z[kpc])
        [-4.371e-08  1.000e+00 -4.371e-08]>

    """
    x = current.r.distance * xp.sin(current.theta) * xp.cos(current.phi)
    y = current.r.distance * xp.sin(current.theta) * xp.sin(current.phi)
    z = current.r.distance * xp.cos(current.theta)
    return target(x=x, y=y, z=z)


@dispatch
def represent_as(
    current: SphericalVector, target: type[CylindricalVector], /, **kwargs: Any
) -> CylindricalVector:
    """SphericalVector -> CylindricalVector.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.SphericalVector(r=Quantity(1., "kpc"), theta=Quantity(90, "deg"),
    ...                          phi=Quantity(90, "deg"))
    >>> print(cx.represent_as(vec, cx.CylindricalVector))
    <CylindricalVector (rho[kpc], phi[deg], z[kpc])
        [ 1.000e+00  9.000e+01 -4.371e-08]>

    """
    rho = xp.abs(current.r.distance * xp.sin(current.theta))
    z = current.r.distance * xp.cos(current.theta)
    return target(rho=rho, phi=current.phi, z=z)


@dispatch
def represent_as(
    current: SphericalVector, target: type[LonLatSphericalVector], /, **kwargs: Any
) -> LonLatSphericalVector:
    """SphericalVector -> LonLatSphericalVector.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.SphericalVector(r=Quantity(1., "kpc"), theta=Quantity(90, "deg"),
    ...                          phi=Quantity(90, "deg"))
    >>> print(cx.represent_as(vec, cx.LonLatSphericalVector))
    <LonLatSphericalVector (lon[deg], lat[deg], distance[kpc])
        [90.  0.  1.]>

    """
    return target(
        lon=current.phi, lat=Quantity(90, "deg") - current.theta, distance=current.r
    )


@dispatch
def represent_as(
    current: SphericalVector, target: type[MathSphericalVector], /, **kwargs: Any
) -> MathSphericalVector:
    """SphericalVector -> MathSphericalVector.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.SphericalVector(r=Quantity(1., "kpc"), theta=Quantity(90, "deg"),
    ...                          phi=Quantity(90, "deg"))
    >>> print(cx.represent_as(vec, cx.MathSphericalVector))
    <MathSphericalVector (r[kpc], theta[deg], phi[deg])
        [ 1. 90. 90.]>

    """
    return target(r=current.r, theta=current.phi, phi=current.theta)


# =============================================================================
# LonLatSphericalVector


@dispatch
def represent_as(
    current: LonLatSphericalVector, target: type[Cartesian3DVector], /, **kwargs: Any
) -> Cartesian3DVector:
    """LonLatSphericalVector -> Cartesian3DVector.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.LonLatSphericalVector(lon=Quantity(90, "deg"), lat=Quantity(0, "deg"),
    ...                                distance=Quantity(1., "kpc"))
    >>> print(cx.represent_as(vec, cx.Cartesian3DVector))
    <Cartesian3DVector (x[kpc], y[kpc], z[kpc])
        [-4.371e-08  1.000e+00 -4.371e-08]>

    """
    return represent_as(represent_as(current, SphericalVector), Cartesian3DVector)


@dispatch
def represent_as(
    current: LonLatSphericalVector, target: type[CylindricalVector], /, **kwargs: Any
) -> CylindricalVector:
    """LonLatSphericalVector -> CylindricalVector.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.LonLatSphericalVector(lon=Quantity(90, "deg"), lat=Quantity(0, "deg"),
    ...                                distance=Quantity(1., "kpc"))
    >>> print(cx.represent_as(vec, cx.CylindricalVector))
    <CylindricalVector (rho[kpc], phi[deg], z[kpc])
        [ 1.000e+00  9.000e+01 -4.371e-08]>

    """
    return represent_as(represent_as(current, SphericalVector), target)


@dispatch
def represent_as(
    current: LonLatSphericalVector, target: type[SphericalVector], /, **kwargs: Any
) -> SphericalVector:
    """LonLatSphericalVector -> SphericalVector.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.LonLatSphericalVector(lon=Quantity(90, "deg"), lat=Quantity(0, "deg"),
    ...                                distance=Quantity(1., "kpc"))
    >>> print(cx.represent_as(vec, cx.SphericalVector))
    <SphericalVector (r[kpc], theta[deg], phi[deg])
        [ 1. 90. 90.]>

    """
    return target(
        r=current.distance, theta=Quantity(90, "deg") - current.lat, phi=current.lon
    )


# =============================================================================
# MathSphericalVector


@dispatch
def represent_as(
    current: MathSphericalVector, target: type[Cartesian3DVector], /, **kwargs: Any
) -> Cartesian3DVector:
    """MathSphericalVector -> Cartesian3DVector.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.MathSphericalVector(r=Quantity(1., "kpc"), theta=Quantity(90, "deg"),
    ...                              phi=Quantity(90, "deg"))
    >>> print(cx.represent_as(vec, cx.Cartesian3DVector))
    <Cartesian3DVector (x[kpc], y[kpc], z[kpc])
        [-4.371e-08  1.000e+00 -4.371e-08]>

    """
    x = current.r.distance * xp.sin(current.phi) * xp.cos(current.theta)
    y = current.r.distance * xp.sin(current.phi) * xp.sin(current.theta)
    z = current.r.distance * xp.cos(current.phi)
    return target(x=x, y=y, z=z)


@dispatch
def represent_as(
    current: MathSphericalVector, target: type[CylindricalVector], /, **kwargs: Any
) -> CylindricalVector:
    """MathSphericalVector -> CylindricalVector.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.MathSphericalVector(r=Quantity(1., "kpc"), theta=Quantity(90, "deg"),
    ...                              phi=Quantity(90, "deg"))
    >>> print(cx.represent_as(vec, cx.CylindricalVector))
    <CylindricalVector (rho[kpc], phi[deg], z[kpc])
        [ 1.000e+00  9.000e+01 -4.371e-08]>

    """
    rho = xp.abs(current.r.distance * xp.sin(current.phi))
    z = current.r.distance * xp.cos(current.phi)
    return target(rho=rho, phi=current.theta, z=z)


@dispatch
def represent_as(
    current: MathSphericalVector, target: type[SphericalVector], /, **kwargs: Any
) -> SphericalVector:
    """MathSphericalVector -> SphericalVector.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.MathSphericalVector(r=Quantity(1., "kpc"), theta=Quantity(90, "deg"),
    ...                              phi=Quantity(90, "deg"))
    >>> print(cx.represent_as(vec, cx.SphericalVector))
    <SphericalVector (r[kpc], theta[deg], phi[deg])
        [ 1. 90. 90.]>

    """
    return target(r=current.r, theta=current.phi, phi=current.theta)


# =============================================================================
# LonLatSphericalDifferential


@dispatch
def represent_as(
    current: Abstract3DVectorDifferential,
    target: type[LonCosLatSphericalDifferential],
    position: AbstractPosition | Quantity["length"],
    /,
    **kwargs: Any,
) -> LonCosLatSphericalDifferential:
    """Abstract3DVectorDifferential -> LonCosLatSphericalDifferential.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.LonLatSphericalVector(lon=Quantity(15, "deg"), lat=Quantity(10, "deg"),
    ...                                distance=Quantity(1.5, "kpc"))
    >>> dif = cx.LonLatSphericalDifferential(d_lon=Quantity(7, "mas/yr"),
    ...                                      d_lat=Quantity(0, "deg/Gyr"),
    ...                                      d_distance=Quantity(-5, "km/s"))
    >>> newdif = cx.represent_as(dif, cx.LonCosLatSphericalDifferential, vec)
    >>> newdif
    LonCosLatSphericalDifferential(
      d_lon_coslat=Quantity[...]( value=f32[], unit=Unit("mas / yr") ),
      d_lat=Quantity[...]( value=f32[], unit=Unit("deg / Gyr") ),
      d_distance=Quantity[...]( value=f32[], unit=Unit("km / s") )
    )

    >>> newdif.d_lon_coslat / xp.cos(vec.lat)  # float32 imprecision
    Quantity['angular frequency'](Array(6.9999995, dtype=float32), unit='mas / yr')

    """
    # Parse the position to an AbstractPosition
    if isinstance(position, AbstractPosition):
        posvec = position
    else:  # Q -> Cart<X>D
        posvec = current.integral_cls._cartesian_cls.constructor(  # noqa: SLF001
            position
        )

    # Transform the differential to LonLatSphericalDifferential
    current = represent_as(current, LonLatSphericalDifferential, posvec)

    # Transform the position to the required type
    posvec = represent_as(posvec, current.integral_cls)

    # Calculate the differential in the new system
    return target(
        d_lon_coslat=current.d_lon * xp.cos(posvec.lat),
        d_lat=current.d_lat,
        d_distance=current.d_distance,
    )


@dispatch
def represent_as(
    current: LonCosLatSphericalDifferential,
    target: type[LonLatSphericalDifferential],
    position: AbstractPosition | Quantity["length"],
    /,
    **kwargs: Any,
) -> LonLatSphericalDifferential:
    """LonCosLatSphericalDifferential -> LonLatSphericalDifferential."""
    # Parse the position to an AbstractPosition
    if isinstance(position, AbstractPosition):
        posvec = position
    else:  # Q -> Cart<X>D
        posvec = current.integral_cls._cartesian_cls.constructor(  # noqa: SLF001
            position
        )

    # Transform the position to the required type
    posvec = represent_as(posvec, current.integral_cls)

    # Calculate the differential in the new system
    return target(
        d_lon=current.d_lon_coslat / xp.cos(posvec.lat),
        d_lat=current.d_lat,
        d_distance=current.d_distance,
    )


@dispatch
def represent_as(
    current: LonCosLatSphericalDifferential,
    target: type[Abstract3DVectorDifferential],
    position: AbstractPosition | Quantity["length"],
    /,
    **kwargs: Any,
) -> Abstract3DVectorDifferential:
    """LonCosLatSphericalDifferential -> Abstract3DVectorDifferential."""
    # Parse the position to an AbstractPosition
    if isinstance(position, AbstractPosition):
        posvec = position
    else:  # Q -> Cart<X>D
        posvec = current.integral_cls._cartesian_cls.constructor(  # noqa: SLF001
            position
        )
    # Transform the differential to LonLatSphericalDifferential
    current = represent_as(current, LonLatSphericalDifferential, posvec)
    # Transform the position to the required type
    return represent_as(current, target, posvec)
