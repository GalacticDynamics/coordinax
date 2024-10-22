"""Representation of coordinates in different systems."""

__all__: list[str] = []

from typing import Any

from plum import dispatch

import quaxed.numpy as xp
from unxt import Quantity

from .base import AbstractPos3D, AbstractVel3D
from .base_spherical import AbstractSphericalPos
from .cartesian import CartesianAcc3D, CartesianPos3D, CartesianVel3D
from .cylindrical import CylindricalPos, CylindricalVel
from .lonlatspherical import (
    LonCosLatSphericalVel,
    LonLatSphericalPos,
    LonLatSphericalVel,
)
from .mathspherical import MathSphericalPos, MathSphericalVel
from .spherical import SphericalPos, SphericalVel
from coordinax._src.base import AbstractPos

###############################################################################
# 3D


@dispatch
def represent_as(
    current: AbstractPos3D, target: type[AbstractPos3D], /, **kwargs: Any
) -> AbstractPos3D:
    """AbstractPos3D -> Cartesian3D -> AbstractPos3D."""
    return represent_as(represent_as(current, CartesianPos3D), target)


@dispatch.multi(
    (CartesianPos3D, type[CartesianPos3D]),
    (CylindricalPos, type[CylindricalPos]),
    (SphericalPos, type[SphericalPos]),
    (LonLatSphericalPos, type[LonLatSphericalPos]),
    (MathSphericalPos, type[MathSphericalPos]),
)
def represent_as(
    current: AbstractPos3D, target: type[AbstractPos3D], /, **kwargs: Any
) -> AbstractPos3D:
    """Self transforms for 3D vectors.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    Cartesian to Cartesian:

    >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> cx.represent_as(vec, cx.CartesianPos3D) is vec
    True

    Cylindrical to Cylindrical:

    >>> vec = cx.CylindricalPos(rho=Quantity(1, "kpc"), phi=Quantity(2, "deg"),
    ...                         z=Quantity(3, "kpc"))
    >>> cx.represent_as(vec, cx.CylindricalPos) is vec
    True

    Spherical to Spherical:

    >>> vec = cx.SphericalPos(r=Quantity(1, "kpc"), theta=Quantity(2, "deg"),
    ...                       phi=Quantity(3, "deg"))
    >>> cx.represent_as(vec, cx.SphericalPos) is vec
    True

    LonLatSpherical to LonLatSpherical:

    >>> vec = cx.LonLatSphericalPos(lon=Quantity(1, "deg"), lat=Quantity(2, "deg"),
    ...                             distance=Quantity(3, "kpc"))
    >>> cx.represent_as(vec, cx.LonLatSphericalPos) is vec
    True

    MathSpherical to MathSpherical:

    >>> vec = cx.MathSphericalPos(r=Quantity(1, "kpc"), theta=Quantity(2, "deg"),
    ...                           phi=Quantity(3, "deg"))
    >>> cx.represent_as(vec, cx.MathSphericalPos) is vec
    True

    """
    return current


@dispatch.multi(
    (CartesianVel3D, type[CartesianVel3D], AbstractPos),
    (CylindricalVel, type[CylindricalVel], AbstractPos),
    (SphericalVel, type[SphericalVel], AbstractPos),
    (LonLatSphericalVel, type[LonLatSphericalVel], AbstractPos),
    (
        LonCosLatSphericalVel,
        type[LonCosLatSphericalVel],
        AbstractPos,
    ),
    (MathSphericalVel, type[MathSphericalVel], AbstractPos),
)
def represent_as(
    current: AbstractVel3D,
    target: type[AbstractVel3D],
    position: AbstractPos,
    /,
    **kwargs: Any,
) -> AbstractVel3D:
    """Self transforms for 3D velocity.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    For these transformations the position does not matter since the
    self-transform returns the velocity unchanged.

    >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "kpc")

    Cartesian to Cartesian velocity:

    >>> dif = cx.CartesianVel3D.from_([1, 2, 3], "km/s")
    >>> cx.represent_as(dif, cx.CartesianVel3D, vec) is dif
    True

    Cylindrical to Cylindrical velocity:

    >>> dif = cx.CylindricalVel(d_rho=Quantity(1, "km/s"),
    ...                         d_phi=Quantity(2, "mas/yr"),
    ...                         d_z=Quantity(3, "km/s"))
    >>> cx.represent_as(dif, cx.CylindricalVel, vec) is dif
    True

    Spherical to Spherical velocity:

    >>> dif = cx.SphericalVel(d_r=Quantity(1, "km/s"),
    ...                       d_theta=Quantity(2, "mas/yr"),
    ...                       d_phi=Quantity(3, "mas/yr"))
    >>> cx.represent_as(dif, cx.SphericalVel, vec) is dif
    True

    LonLatSpherical to LonLatSpherical velocity:

    >>> dif = cx.LonLatSphericalVel(d_lon=Quantity(1, "mas/yr"),
    ...                             d_lat=Quantity(2, "mas/yr"),
    ...                             d_distance=Quantity(3, "km/s"))
    >>> cx.represent_as(dif, cx.LonLatSphericalVel, vec) is dif
    True

    LonCosLatSpherical to LonCosLatSpherical velocity:

    >>> dif = cx.LonCosLatSphericalVel(d_lon_coslat=Quantity(1, "mas/yr"),
    ...                                d_lat=Quantity(2, "mas/yr"),
    ...                                d_distance=Quantity(3, "km/s"))
    >>> cx.represent_as(dif, cx.LonCosLatSphericalVel, vec) is dif
    True

    MathSpherical to MathSpherical velocity:

    >>> dif = cx.MathSphericalVel(d_r=Quantity(1, "km/s"),
    ...                           d_theta=Quantity(2, "mas/yr"),
    ...                           d_phi=Quantity(3, "mas/yr"))
    >>> cx.represent_as(dif, cx.MathSphericalVel, vec) is dif
    True

    """
    return current


# =============================================================================
# CartesianPos3D


@dispatch
def represent_as(
    current: CartesianPos3D, target: type[CylindricalPos], /, **kwargs: Any
) -> CylindricalPos:
    """CartesianPos3D -> CylindricalPos.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "km")
    >>> print(cx.represent_as(vec, cx.CylindricalPos))
    <CylindricalPos (rho[km], phi[rad], z[km])
        [2.236 1.107 3.   ]>

    """
    rho = xp.sqrt(current.x**2 + current.y**2)
    phi = xp.atan2(current.y, current.x)
    return target(rho=rho, phi=phi, z=current.z)


@dispatch
def represent_as(
    current: CartesianPos3D, target: type[SphericalPos], /, **kwargs: Any
) -> SphericalPos:
    """CartesianPos3D -> SphericalPos.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "km")
    >>> print(cx.represent_as(vec, cx.SphericalPos))
    <SphericalPos (r[km], theta[rad], phi[rad])
        [3.742 0.641 1.107]>

    """
    r = xp.sqrt(current.x**2 + current.y**2 + current.z**2)
    theta = xp.acos(current.z / r)
    phi = xp.atan2(current.y, current.x)
    return target(r=r, theta=theta, phi=phi)


@dispatch.multi(
    (CartesianPos3D, type[LonLatSphericalPos]),
    (CartesianPos3D, type[MathSphericalPos]),
)
def represent_as(
    current: CartesianPos3D,
    target: type[AbstractSphericalPos],
    /,
    **kwargs: Any,
) -> AbstractSphericalPos:
    """CartesianPos3D -> AbstractSphericalPos.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "km")

    >>> print(cx.represent_as(vec, cx.LonLatSphericalPos))
    <LonLatSphericalPos (lon[rad], lat[deg], distance[km])
        [ 1.107 53.301  3.742]>

    >>> print(cx.represent_as(vec, cx.MathSphericalPos))
    <MathSphericalPos (r[km], theta[rad], phi[rad])
        [3.742 1.107 0.641]>

    """
    return represent_as(represent_as(current, SphericalPos), target)


# =============================================================================
# CylindricalPos


@dispatch
def represent_as(
    current: CylindricalPos, target: type[CartesianPos3D], /, **kwargs: Any
) -> CartesianPos3D:
    """CylindricalPos -> CartesianPos3D.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.CylindricalPos(rho=Quantity(1., "kpc"), phi=Quantity(90, "deg"),
    ...                         z=Quantity(1, "kpc"))
    >>> print(cx.represent_as(vec, cx.CartesianPos3D))
    <CartesianPos3D (x[kpc], y[kpc], z[kpc])
        [-4.371e-08  1.000e+00  1.000e+00]>

    """
    x = current.rho * xp.cos(current.phi)
    y = current.rho * xp.sin(current.phi)
    z = current.z
    return target(x=x, y=y, z=z)


@dispatch
def represent_as(
    current: CylindricalPos, target: type[SphericalPos], /, **kwargs: Any
) -> SphericalPos:
    """CylindricalPos -> SphericalPos.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.CylindricalPos(rho=Quantity(1., "kpc"), phi=Quantity(90, "deg"),
    ...                         z=Quantity(1, "kpc"))
    >>> print(cx.represent_as(vec, cx.SphericalPos))
    <SphericalPos (r[kpc], theta[rad], phi[deg])
        [ 1.414  0.785 90.   ]>

    """
    r = xp.sqrt(current.rho**2 + current.z**2)
    theta = xp.acos(current.z / r)
    return target(r=r, theta=theta, phi=current.phi)


@dispatch.multi(
    (CylindricalPos, type[LonLatSphericalPos]),
    (CylindricalPos, type[MathSphericalPos]),
)
def represent_as(
    current: CylindricalPos,
    target: type[AbstractSphericalPos],
    /,
    **kwargs: Any,
) -> AbstractSphericalPos:
    """CylindricalPos -> AbstractSphericalPos.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.CylindricalPos(rho=Quantity(1., "kpc"), phi=Quantity(90, "deg"),
    ...                         z=Quantity(1, "kpc"))

    >>> print(cx.represent_as(vec, cx.LonLatSphericalPos))
    <LonLatSphericalPos (lon[deg], lat[deg], distance[kpc])
        [90.    45.     1.414]>

    >>> print(cx.represent_as(vec, cx.MathSphericalPos))
    <MathSphericalPos (r[kpc], theta[deg], phi[rad])
        [ 1.414 90.     0.785]>

    """
    return represent_as(represent_as(current, SphericalPos), target)


# =============================================================================
# SphericalPos


@dispatch
def represent_as(
    current: SphericalPos, target: type[CartesianPos3D], /, **kwargs: Any
) -> CartesianPos3D:
    """SphericalPos -> CartesianPos3D.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.SphericalPos(r=Quantity(1., "kpc"), theta=Quantity(90, "deg"),
    ...                       phi=Quantity(90, "deg"))
    >>> print(cx.represent_as(vec, cx.CartesianPos3D))
    <CartesianPos3D (x[kpc], y[kpc], z[kpc])
        [-4.371e-08  1.000e+00 -4.371e-08]>

    """
    x = current.r.distance * xp.sin(current.theta) * xp.cos(current.phi)
    y = current.r.distance * xp.sin(current.theta) * xp.sin(current.phi)
    z = current.r.distance * xp.cos(current.theta)
    return target(x=x, y=y, z=z)


@dispatch
def represent_as(
    current: SphericalPos, target: type[CylindricalPos], /, **kwargs: Any
) -> CylindricalPos:
    """SphericalPos -> CylindricalPos.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.SphericalPos(r=Quantity(1., "kpc"), theta=Quantity(90, "deg"),
    ...                       phi=Quantity(90, "deg"))
    >>> print(cx.represent_as(vec, cx.CylindricalPos))
    <CylindricalPos (rho[kpc], phi[deg], z[kpc])
        [ 1.000e+00  9.000e+01 -4.371e-08]>

    """
    rho = xp.abs(current.r.distance * xp.sin(current.theta))
    z = current.r.distance * xp.cos(current.theta)
    return target(rho=rho, phi=current.phi, z=z)


@dispatch
def represent_as(
    current: SphericalPos, target: type[LonLatSphericalPos], /, **kwargs: Any
) -> LonLatSphericalPos:
    """SphericalPos -> LonLatSphericalPos.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.SphericalPos(r=Quantity(1., "kpc"), theta=Quantity(90, "deg"),
    ...                       phi=Quantity(90, "deg"))
    >>> print(cx.represent_as(vec, cx.LonLatSphericalPos))
    <LonLatSphericalPos (lon[deg], lat[deg], distance[kpc])
        [90.  0.  1.]>

    """
    return target(
        lon=current.phi, lat=Quantity(90, "deg") - current.theta, distance=current.r
    )


@dispatch
def represent_as(
    current: SphericalPos, target: type[MathSphericalPos], /, **kwargs: Any
) -> MathSphericalPos:
    """SphericalPos -> MathSphericalPos.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.SphericalPos(r=Quantity(1., "kpc"), theta=Quantity(90, "deg"),
    ...                       phi=Quantity(90, "deg"))
    >>> print(cx.represent_as(vec, cx.MathSphericalPos))
    <MathSphericalPos (r[kpc], theta[deg], phi[deg])
        [ 1. 90. 90.]>

    """
    return target(r=current.r, theta=current.phi, phi=current.theta)


# =============================================================================
# LonLatSphericalPos


@dispatch
def represent_as(
    current: LonLatSphericalPos,
    target: type[CartesianPos3D],
    /,
    **kwargs: Any,
) -> CartesianPos3D:
    """LonLatSphericalPos -> CartesianPos3D.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.LonLatSphericalPos(lon=Quantity(90, "deg"),
    ...                             lat=Quantity(0, "deg"),
    ...                             distance=Quantity(1., "kpc"))
    >>> print(cx.represent_as(vec, cx.CartesianPos3D))
    <CartesianPos3D (x[kpc], y[kpc], z[kpc])
        [-4.371e-08  1.000e+00 -4.371e-08]>

    """
    return represent_as(represent_as(current, SphericalPos), CartesianPos3D)


@dispatch
def represent_as(
    current: LonLatSphericalPos,
    target: type[CylindricalPos],
    /,
    **kwargs: Any,
) -> CylindricalPos:
    """LonLatSphericalPos -> CylindricalPos.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.LonLatSphericalPos(lon=Quantity(90, "deg"),
    ...                             lat=Quantity(0, "deg"),
    ...                             distance=Quantity(1., "kpc"))
    >>> print(cx.represent_as(vec, cx.CylindricalPos))
    <CylindricalPos (rho[kpc], phi[deg], z[kpc])
        [ 1.000e+00  9.000e+01 -4.371e-08]>

    """
    return represent_as(represent_as(current, SphericalPos), target)


@dispatch
def represent_as(
    current: LonLatSphericalPos, target: type[SphericalPos], /, **kwargs: Any
) -> SphericalPos:
    """LonLatSphericalPos -> SphericalPos.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.LonLatSphericalPos(lon=Quantity(90, "deg"),
    ...                             lat=Quantity(0, "deg"),
    ...                             distance=Quantity(1., "kpc"))
    >>> print(cx.represent_as(vec, cx.SphericalPos))
    <SphericalPos (r[kpc], theta[deg], phi[deg])
        [ 1. 90. 90.]>

    """
    return target(
        r=current.distance, theta=Quantity(90, "deg") - current.lat, phi=current.lon
    )


# =============================================================================
# MathSphericalPos


@dispatch
def represent_as(
    current: MathSphericalPos, target: type[CartesianPos3D], /, **kwargs: Any
) -> CartesianPos3D:
    """MathSphericalPos -> CartesianPos3D.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.MathSphericalPos(r=Quantity(1., "kpc"), theta=Quantity(90, "deg"),
    ...                           phi=Quantity(90, "deg"))
    >>> print(cx.represent_as(vec, cx.CartesianPos3D))
    <CartesianPos3D (x[kpc], y[kpc], z[kpc])
        [-4.371e-08  1.000e+00 -4.371e-08]>

    """
    x = current.r.distance * xp.sin(current.phi) * xp.cos(current.theta)
    y = current.r.distance * xp.sin(current.phi) * xp.sin(current.theta)
    z = current.r.distance * xp.cos(current.phi)
    return target(x=x, y=y, z=z)


@dispatch
def represent_as(
    current: MathSphericalPos, target: type[CylindricalPos], /, **kwargs: Any
) -> CylindricalPos:
    """MathSphericalPos -> CylindricalPos.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.MathSphericalPos(r=Quantity(1., "kpc"), theta=Quantity(90, "deg"),
    ...                           phi=Quantity(90, "deg"))
    >>> print(cx.represent_as(vec, cx.CylindricalPos))
    <CylindricalPos (rho[kpc], phi[deg], z[kpc])
        [ 1.000e+00  9.000e+01 -4.371e-08]>

    """
    rho = xp.abs(current.r.distance * xp.sin(current.phi))
    z = current.r.distance * xp.cos(current.phi)
    return target(rho=rho, phi=current.theta, z=z)


@dispatch
def represent_as(
    current: MathSphericalPos, target: type[SphericalPos], /, **kwargs: Any
) -> SphericalPos:
    """MathSphericalPos -> SphericalPos.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.MathSphericalPos(r=Quantity(1., "kpc"), theta=Quantity(90, "deg"),
    ...                           phi=Quantity(90, "deg"))
    >>> print(cx.represent_as(vec, cx.SphericalPos))
    <SphericalPos (r[kpc], theta[deg], phi[deg])
        [ 1. 90. 90.]>

    """
    return target(r=current.r, theta=current.phi, phi=current.theta)


# =============================================================================
# LonLatSphericalVel


@dispatch
def represent_as(
    current: AbstractVel3D,
    target: type[LonCosLatSphericalVel],
    position: AbstractPos | Quantity["length"],
    /,
    **kwargs: Any,
) -> LonCosLatSphericalVel:
    """AbstractVel3D -> LonCosLatSphericalVel.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.LonLatSphericalPos(lon=Quantity(15, "deg"),
    ...                             lat=Quantity(10, "deg"),
    ...                             distance=Quantity(1.5, "kpc"))
    >>> dif = cx.LonLatSphericalVel(d_lon=Quantity(7, "mas/yr"),
    ...                             d_lat=Quantity(0, "deg/Gyr"),
    ...                             d_distance=Quantity(-5, "km/s"))
    >>> newdif = cx.represent_as(dif, cx.LonCosLatSphericalVel, vec)
    >>> newdif
    LonCosLatSphericalVel(
      d_lon_coslat=Quantity[...]( value=f32[], unit=Unit("mas / yr") ),
      d_lat=Quantity[...]( value=f32[], unit=Unit("deg / Gyr") ),
      d_distance=Quantity[...]( value=f32[], unit=Unit("km / s") )
    )

    >>> newdif.d_lon_coslat / jnp.cos(vec.lat)  # float32 imprecision
    Quantity['angular frequency'](Array(6.9999995, dtype=float32), unit='mas / yr')

    """
    # Parse the position to an AbstractPos
    if isinstance(position, AbstractPos):
        posvec = position
    else:  # Q -> Cart<X>D
        posvec = current.integral_cls._cartesian_cls.from_(  # noqa: SLF001
            position
        )

    # Transform the differential to LonLatSphericalVel
    current = represent_as(current, LonLatSphericalVel, posvec)

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
    current: LonCosLatSphericalVel,
    target: type[LonLatSphericalVel],
    position: AbstractPos | Quantity["length"],
    /,
    **kwargs: Any,
) -> LonLatSphericalVel:
    """LonCosLatSphericalVel -> LonLatSphericalVel."""
    # Parse the position to an AbstractPos
    if isinstance(position, AbstractPos):
        posvec = position
    else:  # Q -> Cart<X>D
        posvec = current.integral_cls._cartesian_cls.from_(  # noqa: SLF001
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
    current: LonCosLatSphericalVel,
    target: type[AbstractVel3D],
    position: AbstractPos | Quantity["length"],
    /,
    **kwargs: Any,
) -> AbstractVel3D:
    """LonCosLatSphericalVel -> AbstractVel3D."""
    # Parse the position to an AbstractPos
    if isinstance(position, AbstractPos):
        posvec = position
    else:  # Q -> Cart<X>D
        posvec = current.integral_cls._cartesian_cls.from_(  # noqa: SLF001
            position
        )
    # Transform the differential to LonLatSphericalVel
    current = represent_as(current, LonLatSphericalVel, posvec)
    # Transform the position to the required type
    return represent_as(current, target, posvec)


# =============================================================================
# CartesianVel3D


@dispatch
def represent_as(
    current: CartesianVel3D, target: type[CartesianVel3D], /
) -> CartesianVel3D:
    """CartesianVel3D -> CartesianVel3D with no position.

    Cartesian coordinates are an affine coordinate system and so the
    transformation of an n-th order derivative vector in this system do not
    require lower-order derivatives to be specified. See
    https://en.wikipedia.org/wiki/Tensors_in_curvilinear_coordinates for more
    information. This mixin provides a corresponding implementation of the
    `coordinax.represent_as` method for Cartesian velocities.

    Examples
    --------
    >>> import coordinax as cx
    >>> v = cx.CartesianVel3D.from_([1, 1, 1], "m/s")
    >>> cx.represent_as(v, cx.CartesianVel3D) is v
    True

    """
    return current


# =============================================================================
# CartesianAcc3D


@dispatch
def represent_as(
    current: CartesianAcc3D, target: type[CartesianAcc3D], /
) -> CartesianAcc3D:
    """CartesianAcc3D -> CartesianAcc3D with no position.

    Cartesian coordinates are an affine coordinate system and so the
    transformation of an n-th order derivative vector in this system do not
    require lower-order derivatives to be specified. See
    https://en.wikipedia.org/wiki/Tensors_in_curvilinear_coordinates for more
    information. This mixin provides a corresponding implementation of the
    `coordinax.represent_as` method for Cartesian vectors.

    Examples
    --------
    >>> import coordinax as cx
    >>> a = cx.CartesianAcc3D.from_([1, 1, 1], "m/s2")
    >>> cx.represent_as(a, cx.CartesianAcc3D) is a
    True

    """
    return current
