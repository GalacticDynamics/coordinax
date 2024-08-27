"""Representation of coordinates in different systems."""

__all__: list[str] = []

from typing import Any

from plum import dispatch

import quaxed.array_api as xp
from unxt import Quantity

from .base import AbstractPosition3D, AbstractVelocity3D
from .cartesian import CartesianAcceleration3D, CartesianPosition3D, CartesianVelocity3D
from .cylindrical import CylindricalPosition, CylindricalVelocity
from .spherical import (
    AbstractSphericalPosition,
    LonCosLatSphericalVelocity,
    LonLatSphericalPosition,
    LonLatSphericalVelocity,
    MathSphericalPosition,
    MathSphericalVelocity,
    SphericalPosition,
    SphericalVelocity,
)
from coordinax._coordinax.base_pos import AbstractPosition

###############################################################################
# 3D


@dispatch
def represent_as(
    current: AbstractPosition3D, target: type[AbstractPosition3D], /, **kwargs: Any
) -> AbstractPosition3D:
    """AbstractPosition3D -> Cartesian3D -> AbstractPosition3D."""
    return represent_as(represent_as(current, CartesianPosition3D), target)


@dispatch.multi(
    (CartesianPosition3D, type[CartesianPosition3D]),
    (CylindricalPosition, type[CylindricalPosition]),
    (SphericalPosition, type[SphericalPosition]),
    (LonLatSphericalPosition, type[LonLatSphericalPosition]),
    (MathSphericalPosition, type[MathSphericalPosition]),
)
def represent_as(
    current: AbstractPosition3D, target: type[AbstractPosition3D], /, **kwargs: Any
) -> AbstractPosition3D:
    """Self transforms for 3D vectors.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    Cartesian to Cartesian:

    >>> vec = cx.CartesianPosition3D.constructor([1, 2, 3], "kpc")
    >>> cx.represent_as(vec, cx.CartesianPosition3D) is vec
    True

    Cylindrical to Cylindrical:

    >>> vec = cx.CylindricalPosition(rho=Quantity(1, "kpc"), phi=Quantity(2, "deg"),
    ...                            z=Quantity(3, "kpc"))
    >>> cx.represent_as(vec, cx.CylindricalPosition) is vec
    True

    Spherical to Spherical:

    >>> vec = cx.SphericalPosition(r=Quantity(1, "kpc"), theta=Quantity(2, "deg"),
    ...                          phi=Quantity(3, "deg"))
    >>> cx.represent_as(vec, cx.SphericalPosition) is vec
    True

    LonLatSpherical to LonLatSpherical:

    >>> vec = cx.LonLatSphericalPosition(lon=Quantity(1, "deg"), lat=Quantity(2, "deg"),
    ...                                distance=Quantity(3, "kpc"))
    >>> cx.represent_as(vec, cx.LonLatSphericalPosition) is vec
    True

    MathSpherical to MathSpherical:

    >>> vec = cx.MathSphericalPosition(r=Quantity(1, "kpc"), theta=Quantity(2, "deg"),
    ...                              phi=Quantity(3, "deg"))
    >>> cx.represent_as(vec, cx.MathSphericalPosition) is vec
    True

    """
    return current


@dispatch.multi(
    (CartesianVelocity3D, type[CartesianVelocity3D], AbstractPosition),
    (CylindricalVelocity, type[CylindricalVelocity], AbstractPosition),
    (SphericalVelocity, type[SphericalVelocity], AbstractPosition),
    (LonLatSphericalVelocity, type[LonLatSphericalVelocity], AbstractPosition),
    (
        LonCosLatSphericalVelocity,
        type[LonCosLatSphericalVelocity],
        AbstractPosition,
    ),
    (MathSphericalVelocity, type[MathSphericalVelocity], AbstractPosition),
)
def represent_as(
    current: AbstractVelocity3D,
    target: type[AbstractVelocity3D],
    position: AbstractPosition,
    /,
    **kwargs: Any,
) -> AbstractVelocity3D:
    """Self transforms for 3D velocity.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    For these transformations the position does not matter since the
    self-transform returns the velocity unchanged.

    >>> vec = cx.CartesianPosition3D.constructor([1, 2, 3], "kpc")

    Cartesian to Cartesian velocity:

    >>> dif = cx.CartesianVelocity3D.constructor([1, 2, 3], "km/s")
    >>> cx.represent_as(dif, cx.CartesianVelocity3D, vec) is dif
    True

    Cylindrical to Cylindrical velocity:

    >>> dif = cx.CylindricalVelocity(d_rho=Quantity(1, "km/s"),
    ...                                  d_phi=Quantity(2, "mas/yr"),
    ...                                  d_z=Quantity(3, "km/s"))
    >>> cx.represent_as(dif, cx.CylindricalVelocity, vec) is dif
    True

    Spherical to Spherical velocity:

    >>> dif = cx.SphericalVelocity(d_r=Quantity(1, "km/s"),
    ...                                d_theta=Quantity(2, "mas/yr"),
    ...                                d_phi=Quantity(3, "mas/yr"))
    >>> cx.represent_as(dif, cx.SphericalVelocity, vec) is dif
    True

    LonLatSpherical to LonLatSpherical velocity:

    >>> dif = cx.LonLatSphericalVelocity(d_lon=Quantity(1, "mas/yr"),
    ...                                      d_lat=Quantity(2, "mas/yr"),
    ...                                      d_distance=Quantity(3, "km/s"))
    >>> cx.represent_as(dif, cx.LonLatSphericalVelocity, vec) is dif
    True

    LonCosLatSpherical to LonCosLatSpherical velocity:

    >>> dif = cx.LonCosLatSphericalVelocity(d_lon_coslat=Quantity(1, "mas/yr"),
    ...                                         d_lat=Quantity(2, "mas/yr"),
    ...                                         d_distance=Quantity(3, "km/s"))
    >>> cx.represent_as(dif, cx.LonCosLatSphericalVelocity, vec) is dif
    True

    MathSpherical to MathSpherical velocity:

    >>> dif = cx.MathSphericalVelocity(d_r=Quantity(1, "km/s"),
    ...                                    d_theta=Quantity(2, "mas/yr"),
    ...                                    d_phi=Quantity(3, "mas/yr"))
    >>> cx.represent_as(dif, cx.MathSphericalVelocity, vec) is dif
    True

    """
    return current


# =============================================================================
# CartesianPosition3D


@dispatch
def represent_as(
    current: CartesianPosition3D, target: type[CylindricalPosition], /, **kwargs: Any
) -> CylindricalPosition:
    """CartesianPosition3D -> CylindricalPosition.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.CartesianPosition3D.constructor([1, 2, 3], "km")
    >>> print(cx.represent_as(vec, cx.CylindricalPosition))
    <CylindricalPosition (rho[km], phi[rad], z[km])
        [2.236 1.107 3.   ]>

    """
    rho = xp.sqrt(current.x**2 + current.y**2)
    phi = xp.atan2(current.y, current.x)
    return target(rho=rho, phi=phi, z=current.z)


@dispatch
def represent_as(
    current: CartesianPosition3D, target: type[SphericalPosition], /, **kwargs: Any
) -> SphericalPosition:
    """CartesianPosition3D -> SphericalPosition.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.CartesianPosition3D.constructor([1, 2, 3], "km")
    >>> print(cx.represent_as(vec, cx.SphericalPosition))
    <SphericalPosition (r[km], theta[rad], phi[rad])
        [3.742 0.641 1.107]>

    """
    r = xp.sqrt(current.x**2 + current.y**2 + current.z**2)
    theta = xp.acos(current.z / r)
    phi = xp.atan2(current.y, current.x)
    return target(r=r, theta=theta, phi=phi)


@dispatch.multi(
    (CartesianPosition3D, type[LonLatSphericalPosition]),
    (CartesianPosition3D, type[MathSphericalPosition]),
)
def represent_as(
    current: CartesianPosition3D,
    target: type[AbstractSphericalPosition],
    /,
    **kwargs: Any,
) -> AbstractSphericalPosition:
    """CartesianPosition3D -> AbstractSphericalPosition.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.CartesianPosition3D.constructor([1, 2, 3], "km")

    >>> print(cx.represent_as(vec, cx.LonLatSphericalPosition))
    <LonLatSphericalPosition (lon[rad], lat[deg], distance[km])
        [ 1.107 53.301  3.742]>

    >>> print(cx.represent_as(vec, cx.MathSphericalPosition))
    <MathSphericalPosition (r[km], theta[rad], phi[rad])
        [3.742 1.107 0.641]>

    """
    return represent_as(represent_as(current, SphericalPosition), target)


# =============================================================================
# CylindricalPosition


@dispatch
def represent_as(
    current: CylindricalPosition, target: type[CartesianPosition3D], /, **kwargs: Any
) -> CartesianPosition3D:
    """CylindricalPosition -> CartesianPosition3D.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.CylindricalPosition(rho=Quantity(1., "kpc"), phi=Quantity(90, "deg"),
    ...                              z=Quantity(1, "kpc"))
    >>> print(cx.represent_as(vec, cx.CartesianPosition3D))
    <CartesianPosition3D (x[kpc], y[kpc], z[kpc])
        [-4.371e-08  1.000e+00  1.000e+00]>

    """
    x = current.rho * xp.cos(current.phi)
    y = current.rho * xp.sin(current.phi)
    z = current.z
    return target(x=x, y=y, z=z)


@dispatch
def represent_as(
    current: CylindricalPosition, target: type[SphericalPosition], /, **kwargs: Any
) -> SphericalPosition:
    """CylindricalPosition -> SphericalPosition.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.CylindricalPosition(rho=Quantity(1., "kpc"), phi=Quantity(90, "deg"),
    ...                            z=Quantity(1, "kpc"))
    >>> print(cx.represent_as(vec, cx.SphericalPosition))
    <SphericalPosition (r[kpc], theta[rad], phi[deg])
        [ 1.414  0.785 90.   ]>

    """
    r = xp.sqrt(current.rho**2 + current.z**2)
    theta = xp.acos(current.z / r)
    return target(r=r, theta=theta, phi=current.phi)


@dispatch.multi(
    (CylindricalPosition, type[LonLatSphericalPosition]),
    (CylindricalPosition, type[MathSphericalPosition]),
)
def represent_as(
    current: CylindricalPosition,
    target: type[AbstractSphericalPosition],
    /,
    **kwargs: Any,
) -> AbstractSphericalPosition:
    """CylindricalPosition -> AbstractSphericalPosition.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.CylindricalPosition(rho=Quantity(1., "kpc"), phi=Quantity(90, "deg"),
    ...                            z=Quantity(1, "kpc"))

    >>> print(cx.represent_as(vec, cx.LonLatSphericalPosition))
    <LonLatSphericalPosition (lon[deg], lat[deg], distance[kpc])
        [90.    45.     1.414]>

    >>> print(cx.represent_as(vec, cx.MathSphericalPosition))
    <MathSphericalPosition (r[kpc], theta[deg], phi[rad])
        [ 1.414 90.     0.785]>

    """
    return represent_as(represent_as(current, SphericalPosition), target)


# =============================================================================
# SphericalPosition


@dispatch
def represent_as(
    current: SphericalPosition, target: type[CartesianPosition3D], /, **kwargs: Any
) -> CartesianPosition3D:
    """SphericalPosition -> CartesianPosition3D.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.SphericalPosition(r=Quantity(1., "kpc"), theta=Quantity(90, "deg"),
    ...                          phi=Quantity(90, "deg"))
    >>> print(cx.represent_as(vec, cx.CartesianPosition3D))
    <CartesianPosition3D (x[kpc], y[kpc], z[kpc])
        [-4.371e-08  1.000e+00 -4.371e-08]>

    """
    x = current.r.distance * xp.sin(current.theta) * xp.cos(current.phi)
    y = current.r.distance * xp.sin(current.theta) * xp.sin(current.phi)
    z = current.r.distance * xp.cos(current.theta)
    return target(x=x, y=y, z=z)


@dispatch
def represent_as(
    current: SphericalPosition, target: type[CylindricalPosition], /, **kwargs: Any
) -> CylindricalPosition:
    """SphericalPosition -> CylindricalPosition.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.SphericalPosition(r=Quantity(1., "kpc"), theta=Quantity(90, "deg"),
    ...                          phi=Quantity(90, "deg"))
    >>> print(cx.represent_as(vec, cx.CylindricalPosition))
    <CylindricalPosition (rho[kpc], phi[deg], z[kpc])
        [ 1.000e+00  9.000e+01 -4.371e-08]>

    """
    rho = xp.abs(current.r.distance * xp.sin(current.theta))
    z = current.r.distance * xp.cos(current.theta)
    return target(rho=rho, phi=current.phi, z=z)


@dispatch
def represent_as(
    current: SphericalPosition, target: type[LonLatSphericalPosition], /, **kwargs: Any
) -> LonLatSphericalPosition:
    """SphericalPosition -> LonLatSphericalPosition.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.SphericalPosition(r=Quantity(1., "kpc"), theta=Quantity(90, "deg"),
    ...                          phi=Quantity(90, "deg"))
    >>> print(cx.represent_as(vec, cx.LonLatSphericalPosition))
    <LonLatSphericalPosition (lon[deg], lat[deg], distance[kpc])
        [90.  0.  1.]>

    """
    return target(
        lon=current.phi, lat=Quantity(90, "deg") - current.theta, distance=current.r
    )


@dispatch
def represent_as(
    current: SphericalPosition, target: type[MathSphericalPosition], /, **kwargs: Any
) -> MathSphericalPosition:
    """SphericalPosition -> MathSphericalPosition.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.SphericalPosition(r=Quantity(1., "kpc"), theta=Quantity(90, "deg"),
    ...                          phi=Quantity(90, "deg"))
    >>> print(cx.represent_as(vec, cx.MathSphericalPosition))
    <MathSphericalPosition (r[kpc], theta[deg], phi[deg])
        [ 1. 90. 90.]>

    """
    return target(r=current.r, theta=current.phi, phi=current.theta)


# =============================================================================
# LonLatSphericalPosition


@dispatch
def represent_as(
    current: LonLatSphericalPosition,
    target: type[CartesianPosition3D],
    /,
    **kwargs: Any,
) -> CartesianPosition3D:
    """LonLatSphericalPosition -> CartesianPosition3D.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.LonLatSphericalPosition(lon=Quantity(90, "deg"),
    ...                                  lat=Quantity(0, "deg"),
    ...                                  distance=Quantity(1., "kpc"))
    >>> print(cx.represent_as(vec, cx.CartesianPosition3D))
    <CartesianPosition3D (x[kpc], y[kpc], z[kpc])
        [-4.371e-08  1.000e+00 -4.371e-08]>

    """
    return represent_as(represent_as(current, SphericalPosition), CartesianPosition3D)


@dispatch
def represent_as(
    current: LonLatSphericalPosition,
    target: type[CylindricalPosition],
    /,
    **kwargs: Any,
) -> CylindricalPosition:
    """LonLatSphericalPosition -> CylindricalPosition.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.LonLatSphericalPosition(lon=Quantity(90, "deg"),
    ...                                  lat=Quantity(0, "deg"),
    ...                                  distance=Quantity(1., "kpc"))
    >>> print(cx.represent_as(vec, cx.CylindricalPosition))
    <CylindricalPosition (rho[kpc], phi[deg], z[kpc])
        [ 1.000e+00  9.000e+01 -4.371e-08]>

    """
    return represent_as(represent_as(current, SphericalPosition), target)


@dispatch
def represent_as(
    current: LonLatSphericalPosition, target: type[SphericalPosition], /, **kwargs: Any
) -> SphericalPosition:
    """LonLatSphericalPosition -> SphericalPosition.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.LonLatSphericalPosition(lon=Quantity(90, "deg"),
    ...                                  lat=Quantity(0, "deg"),
    ...                                  distance=Quantity(1., "kpc"))
    >>> print(cx.represent_as(vec, cx.SphericalPosition))
    <SphericalPosition (r[kpc], theta[deg], phi[deg])
        [ 1. 90. 90.]>

    """
    return target(
        r=current.distance, theta=Quantity(90, "deg") - current.lat, phi=current.lon
    )


# =============================================================================
# MathSphericalPosition


@dispatch
def represent_as(
    current: MathSphericalPosition, target: type[CartesianPosition3D], /, **kwargs: Any
) -> CartesianPosition3D:
    """MathSphericalPosition -> CartesianPosition3D.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.MathSphericalPosition(r=Quantity(1., "kpc"), theta=Quantity(90, "deg"),
    ...                              phi=Quantity(90, "deg"))
    >>> print(cx.represent_as(vec, cx.CartesianPosition3D))
    <CartesianPosition3D (x[kpc], y[kpc], z[kpc])
        [-4.371e-08  1.000e+00 -4.371e-08]>

    """
    x = current.r.distance * xp.sin(current.phi) * xp.cos(current.theta)
    y = current.r.distance * xp.sin(current.phi) * xp.sin(current.theta)
    z = current.r.distance * xp.cos(current.phi)
    return target(x=x, y=y, z=z)


@dispatch
def represent_as(
    current: MathSphericalPosition, target: type[CylindricalPosition], /, **kwargs: Any
) -> CylindricalPosition:
    """MathSphericalPosition -> CylindricalPosition.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.MathSphericalPosition(r=Quantity(1., "kpc"), theta=Quantity(90, "deg"),
    ...                              phi=Quantity(90, "deg"))
    >>> print(cx.represent_as(vec, cx.CylindricalPosition))
    <CylindricalPosition (rho[kpc], phi[deg], z[kpc])
        [ 1.000e+00  9.000e+01 -4.371e-08]>

    """
    rho = xp.abs(current.r.distance * xp.sin(current.phi))
    z = current.r.distance * xp.cos(current.phi)
    return target(rho=rho, phi=current.theta, z=z)


@dispatch
def represent_as(
    current: MathSphericalPosition, target: type[SphericalPosition], /, **kwargs: Any
) -> SphericalPosition:
    """MathSphericalPosition -> SphericalPosition.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.MathSphericalPosition(r=Quantity(1., "kpc"), theta=Quantity(90, "deg"),
    ...                              phi=Quantity(90, "deg"))
    >>> print(cx.represent_as(vec, cx.SphericalPosition))
    <SphericalPosition (r[kpc], theta[deg], phi[deg])
        [ 1. 90. 90.]>

    """
    return target(r=current.r, theta=current.phi, phi=current.theta)


# =============================================================================
# LonLatSphericalVelocity


@dispatch
def represent_as(
    current: AbstractVelocity3D,
    target: type[LonCosLatSphericalVelocity],
    position: AbstractPosition | Quantity["length"],
    /,
    **kwargs: Any,
) -> LonCosLatSphericalVelocity:
    """AbstractVelocity3D -> LonCosLatSphericalVelocity.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.LonLatSphericalPosition(lon=Quantity(15, "deg"),
    ...                                  lat=Quantity(10, "deg"),
    ...                                  distance=Quantity(1.5, "kpc"))
    >>> dif = cx.LonLatSphericalVelocity(d_lon=Quantity(7, "mas/yr"),
    ...                                      d_lat=Quantity(0, "deg/Gyr"),
    ...                                      d_distance=Quantity(-5, "km/s"))
    >>> newdif = cx.represent_as(dif, cx.LonCosLatSphericalVelocity, vec)
    >>> newdif
    LonCosLatSphericalVelocity(
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

    # Transform the differential to LonLatSphericalVelocity
    current = represent_as(current, LonLatSphericalVelocity, posvec)

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
    current: LonCosLatSphericalVelocity,
    target: type[LonLatSphericalVelocity],
    position: AbstractPosition | Quantity["length"],
    /,
    **kwargs: Any,
) -> LonLatSphericalVelocity:
    """LonCosLatSphericalVelocity -> LonLatSphericalVelocity."""
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
    current: LonCosLatSphericalVelocity,
    target: type[AbstractVelocity3D],
    position: AbstractPosition | Quantity["length"],
    /,
    **kwargs: Any,
) -> AbstractVelocity3D:
    """LonCosLatSphericalVelocity -> AbstractVelocity3D."""
    # Parse the position to an AbstractPosition
    if isinstance(position, AbstractPosition):
        posvec = position
    else:  # Q -> Cart<X>D
        posvec = current.integral_cls._cartesian_cls.constructor(  # noqa: SLF001
            position
        )
    # Transform the differential to LonLatSphericalVelocity
    current = represent_as(current, LonLatSphericalVelocity, posvec)
    # Transform the position to the required type
    return represent_as(current, target, posvec)


# =============================================================================
# CartesianVelocity3D


@dispatch
def represent_as(
    current: CartesianVelocity3D, target: type[CartesianVelocity3D], /
) -> CartesianVelocity3D:
    """CartesianVelocity3D -> CartesianVelocity3D with no position.

    Cartesian coordinates are an affine coordinate system and so the
    transformation of an n-th order derivative vector in this system do not
    require lower-order derivatives to be specified. See
    https://en.wikipedia.org/wiki/Tensors_in_curvilinear_coordinates for more
    information. This mixin provides a corresponding implementation of the
    `coordinax.represent_as` method for Cartesian velocities.

    Examples
    --------
    >>> import coordinax as cx
    >>> v = cx.CartesianVelocity3D.constructor([1, 1, 1], "m/s")
    >>> cx.represent_as(v, cx.CartesianVelocity3D) is v
    True

    """
    return current


# =============================================================================
# CartesianAcceleration3D


@dispatch
def represent_as(
    current: CartesianAcceleration3D, target: type[CartesianAcceleration3D], /
) -> CartesianAcceleration3D:
    """CartesianAcceleration3D -> CartesianAcceleration3D with no position.

    Cartesian coordinates are an affine coordinate system and so the
    transformation of an n-th order derivative vector in this system do not
    require lower-order derivatives to be specified. See
    https://en.wikipedia.org/wiki/Tensors_in_curvilinear_coordinates for more
    information. This mixin provides a corresponding implementation of the
    `coordinax.represent_as` method for Cartesian vectors.

    Examples
    --------
    >>> import coordinax as cx
    >>> a = cx.CartesianAcceleration3D.constructor([1, 1, 1], "m/s2")
    >>> cx.represent_as(a, cx.CartesianAcceleration3D) is a
    True

    """
    return current
