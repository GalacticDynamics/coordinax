"""Representation of coordinates in different systems."""
# ruff: noqa: N803, N806

__all__: list[str] = []

from functools import partial
from typing import Any

import equinox as eqx
from plum import dispatch

import quaxed.lax as qlax
import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AbstractQuantity

from .base import AbstractAcc3D, AbstractPos3D, AbstractVel3D
from .base_spherical import AbstractSphericalPos, _90d, _180d, _360d
from .cartesian import CartesianAcc3D, CartesianPos3D, CartesianVel3D
from .cylindrical import CylindricalPos, CylindricalVel
from .generic import CartesianGeneric3D
from .lonlatspherical import (
    LonCosLatSphericalVel,
    LonLatSphericalPos,
    LonLatSphericalVel,
)
from .mathspherical import MathSphericalPos, MathSphericalVel
from .spherical import SphericalPos, SphericalVel
from .spheroidal import ProlateSpheroidalPos, ProlateSpheroidalVel
from coordinax._src.vectors.base_pos import AbstractPos

###############################################################################


@dispatch(precedence=1)
def vector(cls: type[AbstractPos3D], obj: AbstractPos3D, /) -> AbstractPos3D:
    """Construct from a 3D position.

    Examples
    --------
    >>> import coordinax as cx

    >>> cart = cx.CartesianPos3D.from_([1, 2, 3], "km")
    >>> cx.vecs.AbstractPos3D.from_(cart) is cart
    True

    >>> sph = cart.vconvert(cx.SphericalPos)
    >>> cx.vecs.AbstractPos3D.from_(sph) is sph
    True

    >>> cyl = cart.vconvert(cx.vecs.CylindricalPos)
    >>> cx.vecs.AbstractPos3D.from_(cyl) is cyl
    True

    """
    return obj


# ---------------------------------------------------------


@dispatch(precedence=1)
def vector(cls: type[AbstractVel3D], obj: AbstractVel3D, /) -> AbstractVel3D:
    """Construct from a 3D velocity.

    Examples
    --------
    >>> import coordinax as cx

    >>> q = cx.CartesianPos3D.from_([1, 1, 1], "km")

    >>> cart = cx.CartesianVel3D.from_([1, 2, 3], "km/s")
    >>> cx.vecs.AbstractVel3D.from_(cart) is cart
    True

    >>> sph = cart.vconvert(cx.SphericalVel, q)
    >>> cx.vecs.AbstractVel3D.from_(sph) is sph
    True

    >>> cyl = cart.vconvert(cx.vecs.CylindricalVel, q)
    >>> cx.vecs.AbstractVel3D.from_(cyl) is cyl
    True

    """
    return obj


# ---------------------------------------------------------


@dispatch(precedence=1)
def vector(cls: type[AbstractAcc3D], obj: AbstractAcc3D, /) -> AbstractAcc3D:
    """Construct from a 3D velocity.

    Examples
    --------
    >>> import coordinax as cx

    >>> q = cx.CartesianPos3D.from_([1, 1, 1], "km")
    >>> p = cx.CartesianVel3D.from_([1, 1, 1], "km/s")

    >>> cart = cx.vecs.CartesianAcc3D.from_([1, 2, 3], "km/s2")
    >>> cx.vecs.AbstractAcc3D.from_(cart) is cart
    True

    >>> sph = cart.vconvert(cx.vecs.SphericalAcc, p, q)
    >>> cx.vecs.AbstractAcc3D.from_(sph) is sph
    True

    >>> cyl = cart.vconvert(cx.vecs.CylindricalAcc, p, q)
    >>> cx.vecs.AbstractAcc3D.from_(cyl) is cyl
    True

    """
    return obj


# ---------------------------------------------------------


@dispatch
def vector(
    cls: type[SphericalPos],
    *,
    r: AbstractQuantity,
    theta: AbstractQuantity,
    phi: AbstractQuantity,
) -> SphericalPos:
    """Construct SphericalPos, allowing for out-of-range values.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    Let's start with a valid input:

    >>> vec = cx.SphericalPos.from_(r=u.Quantity(3, "km"),
    ...                             theta=u.Quantity(90, "deg"),
    ...                             phi=u.Quantity(0, "deg"))
    >>> print(vec)
    <SphericalPos (r[km], theta[deg], phi[deg])
        [ 3 90  0]>

    The radial distance can be negative, which wraps the azimuthal angle by 180
    degrees and flips the polar angle:

    >>> vec = cx.SphericalPos.from_(r=u.Quantity(-3, "km"),
    ...                             theta=u.Quantity(45, "deg"),
    ...                             phi=u.Quantity(0, "deg"))
    >>> print(vec)
    <SphericalPos (r[km], theta[deg], phi[deg])
        [  3 135 180]>

    The polar angle can be outside the [0, 180] deg range, causing the azimuthal
    angle to be shifted by 180 degrees:

    >>> vec = cx.SphericalPos.from_(r=u.Quantity(3, "km"),
    ...                             theta=u.Quantity(190, "deg"),
    ...                             phi=u.Quantity(0, "deg"))
    >>> print(vec)
    <SphericalPos (r[km], theta[deg], phi[deg])
        [  3 170 180]>

    The azimuth can be outside the [0, 360) deg range. This is wrapped to the
    [0, 360) deg range (actually the base from_ does this):

    >>> vec = cx.SphericalPos.from_(r=u.Quantity(3, "km"),
    ...                             theta=u.Quantity(90, "deg"),
    ...                             phi=u.Quantity(365, "deg"))
    >>> vec.phi
    Angle(Array(5, dtype=int32, ...), unit='deg')

    """
    # 1) Convert the inputs
    fields = SphericalPos.__dataclass_fields__
    r = fields["r"].metadata["converter"](r)
    theta = fields["theta"].metadata["converter"](theta)
    phi = fields["phi"].metadata["converter"](phi)

    # 2) handle negative distances
    r_pred = r < jnp.zeros_like(r)
    r = jnp.where(r_pred, -r, r)
    phi = jnp.where(r_pred, phi + _180d, phi)
    theta = jnp.where(r_pred, _180d - theta, theta)

    # 3) Handle polar angle outside of [0, 180] degrees
    theta = jnp.mod(theta, _360d)  # wrap to [0, 360) deg
    theta_pred = theta < _180d
    theta = jnp.where(theta_pred, theta, _360d - theta)
    phi = jnp.where(theta_pred, phi, phi + _180d)

    # 4) Construct. This also handles the azimuthal angle wrapping
    return cls(r=r, theta=theta, phi=phi)


# ---------------------------------------------------------


@dispatch
def vector(
    cls: type[LonLatSphericalPos],
    *,
    lon: AbstractQuantity,
    lat: AbstractQuantity,
    distance: AbstractQuantity,
) -> LonLatSphericalPos:
    """Construct LonLatSphericalPos, allowing for out-of-range values.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    Let's start with a valid input:

    >>> vec = cx.vecs.LonLatSphericalPos.from_(lon=u.Quantity(0, "deg"),
    ...                                        lat=u.Quantity(0, "deg"),
    ...                                        distance=u.Quantity(3, "km"))
    >>> print(vec)
    <LonLatSphericalPos (lon[deg], lat[deg], distance[km])
        [0 0 3]>

    The distance can be negative, which wraps the longitude by 180 degrees and
    flips the latitude:

    >>> vec = cx.vecs.LonLatSphericalPos.from_(lon=u.Quantity(0, "deg"),
    ...                                        lat=u.Quantity(45, "deg"),
    ...                                        distance=u.Quantity(-3, "km"))
    >>> print(vec)
    <LonLatSphericalPos (lon[deg], lat[deg], distance[km])
        [180 -45   3]>

    The latitude can be outside the [-90, 90] deg range, causing the longitude
    to be shifted by 180 degrees:

    >>> vec = cx.vecs.LonLatSphericalPos.from_(lon=u.Quantity(0, "deg"),
    ...                                        lat=u.Quantity(-100, "deg"),
    ...                                        distance=u.Quantity(3, "km"))
    >>> print(vec)
    <LonLatSphericalPos (lon[deg], lat[deg], distance[km])
        [180 -80   3]>

    >>> vec = cx.vecs.LonLatSphericalPos.from_(lon=u.Quantity(0, "deg"),
    ...                                        lat=u.Quantity(100, "deg"),
    ...                                        distance=u.Quantity(3, "km"))
    >>> print(vec)
    <LonLatSphericalPos (lon[deg], lat[deg], distance[km])
        [180  80   3]>

    The longitude can be outside the [0, 360) deg range. This is wrapped to the
    [0, 360) deg range (actually the base constructor does this):

    >>> vec = cx.vecs.LonLatSphericalPos.from_(lon=u.Quantity(365, "deg"),
    ...                                        lat=u.Quantity(0, "deg"),
    ...                                        distance=u.Quantity(3, "km"))
    >>> vec.lon
    Angle(Array(5, dtype=int32, ...), unit='deg')

    """
    # 1) Convert the inputs
    fields = LonLatSphericalPos.__dataclass_fields__
    lon = fields["lon"].metadata["converter"](lon)
    lat = fields["lat"].metadata["converter"](lat)
    distance = fields["distance"].metadata["converter"](distance)

    # 2) handle negative distances
    distance_pred = distance < jnp.zeros_like(distance)
    distance = qlax.select(distance_pred, -distance, distance)
    lon = qlax.select(distance_pred, lon + _180d, lon)
    lat = qlax.select(distance_pred, -lat, lat)

    # 3) Handle latitude outside of [-90, 90] degrees
    # TODO: fix when lat < -180, lat > 180
    lat_pred = lat < -_90d
    lat = qlax.select(lat_pred, -_180d - lat, lat)
    lon = qlax.select(lat_pred, lon + _180d, lon)

    lat_pred = lat > _90d
    lat = qlax.select(lat_pred, _180d - lat, lat)
    lon = qlax.select(lat_pred, lon + _180d, lon)

    # 4) Construct. This also handles the longitude wrapping
    return cls(lon=lon, lat=lat, distance=distance)


# ---------------------------------------------------------


@dispatch
def vector(
    cls: type[MathSphericalPos],
    *,
    r: AbstractQuantity,
    theta: AbstractQuantity,
    phi: AbstractQuantity,
) -> MathSphericalPos:
    """Construct MathSphericalPos, allowing for out-of-range values.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    Let's start with a valid input:

    >>> vec = cx.vecs.MathSphericalPos.from_(r=u.Quantity(3, "km"),
    ...                                      theta=u.Quantity(90, "deg"),
    ...                                      phi=u.Quantity(0, "deg"))
    >>> print(vec)
    <MathSphericalPos (r[km], theta[deg], phi[deg])
        [ 3 90  0]>

    The radial distance can be negative, which wraps the azimuthal angle by 180
    degrees and flips the polar angle:

    >>> vec = cx.vecs.MathSphericalPos.from_(r=u.Quantity(-3, "km"),
    ...                                      theta=u.Quantity(100, "deg"),
    ...                                      phi=u.Quantity(45, "deg"))
    >>> print(vec)
    <MathSphericalPos (r[km], theta[deg], phi[deg])
        [  3 280 135]>

    The polar angle can be outside the [0, 180] deg range, causing the azimuthal
    angle to be shifted by 180 degrees:

    >>> vec = cx.vecs.MathSphericalPos.from_(r=u.Quantity(3, "km"),
    ...                                      theta=u.Quantity(0, "deg"),
    ...                                      phi=u.Quantity(190, "deg"))
    >>> print(vec)
    <MathSphericalPos (r[km], theta[deg], phi[deg])
        [  3 180 170]>

    The azimuth can be outside the [0, 360) deg range. This is wrapped to the
    [0, 360) deg range (actually the base constructor does this):

    >>> vec = cx.vecs.MathSphericalPos.from_(r=u.Quantity(3, "km"),
    ...                                      theta=u.Quantity(365, "deg"),
    ...                                      phi=u.Quantity(90, "deg"))
    >>> vec.theta
    Angle(Array(5, dtype=int32, ...), unit='deg')

    """
    # 1) Convert the inputs
    fields = MathSphericalPos.__dataclass_fields__
    r = fields["r"].metadata["converter"](r)
    theta = fields["theta"].metadata["converter"](theta)
    phi = fields["phi"].metadata["converter"](phi)

    # 2) handle negative distances
    r_pred = r < jnp.zeros_like(r)
    r = jnp.where(r_pred, -r, r)
    theta = jnp.where(r_pred, theta + _180d, theta)
    phi = jnp.where(r_pred, _180d - phi, phi)

    # 3) Handle polar angle outside of [0, 180] degrees
    phi = jnp.mod(phi, _360d)  # wrap to [0, 360) deg
    phi_pred = phi < _180d
    phi = jnp.where(phi_pred, phi, _360d - phi)
    theta = jnp.where(phi_pred, theta, theta + _180d)

    # 4) Construct. This also handles the azimuthal angle wrapping
    return cls(r=r, theta=theta, phi=phi)


###############################################################################
# 3D


@dispatch
def vconvert(
    target: type[AbstractPos3D], current: AbstractPos3D, /, **kwargs: Any
) -> AbstractPos3D:
    """AbstractPos3D -> Cartesian3D -> AbstractPos3D."""
    return vconvert(target, vconvert(CartesianPos3D, current))


@dispatch.multi(
    (type[CartesianPos3D], CartesianPos3D),
    (type[CylindricalPos], CylindricalPos),
    (type[SphericalPos], SphericalPos),
    (type[LonLatSphericalPos], LonLatSphericalPos),
    (type[MathSphericalPos], MathSphericalPos),
)
def vconvert(
    target: type[AbstractPos3D], current: AbstractPos3D, /, **kwargs: Any
) -> AbstractPos3D:
    """Self transforms for 3D vectors.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    Cartesian to Cartesian:

    >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "km")
    >>> cx.vconvert(cx.CartesianPos3D, vec) is vec
    True

    Cylindrical to Cylindrical:

    >>> vec = cx.vecs.CylindricalPos(rho=u.Quantity(1, "km"),
    ...                              phi=u.Quantity(2, "deg"),
    ...                              z=u.Quantity(3, "km"))
    >>> cx.vconvert(cx.vecs.CylindricalPos, vec) is vec
    True

    Spherical to Spherical:

    >>> vec = cx.SphericalPos(r=u.Quantity(1, "km"),
    ...                       theta=u.Quantity(2, "deg"),
    ...                       phi=u.Quantity(3, "deg"))
    >>> cx.vconvert(cx.SphericalPos, vec) is vec
    True

    LonLatSpherical to LonLatSpherical:

    >>> vec = cx.vecs.LonLatSphericalPos(lon=u.Quantity(1, "deg"),
    ...                                  lat=u.Quantity(2, "deg"),
    ...                                  distance=u.Quantity(3, "km"))
    >>> cx.vconvert(cx.vecs.LonLatSphericalPos, vec) is vec
    True

    MathSpherical to MathSpherical:

    >>> vec = cx.vecs.MathSphericalPos(r=u.Quantity(1, "km"),
    ...                                theta=u.Quantity(2, "deg"),
    ...                                phi=u.Quantity(3, "deg"))
    >>> cx.vconvert(cx.vecs.MathSphericalPos, vec) is vec
    True

    """
    return current


@dispatch.multi(
    (type[CartesianVel3D], CartesianVel3D, AbstractPos),
    (type[CylindricalVel], CylindricalVel, AbstractPos),
    (type[SphericalVel], SphericalVel, AbstractPos),
    (type[LonLatSphericalVel], LonLatSphericalVel, AbstractPos),
    (
        type[LonCosLatSphericalVel],
        LonCosLatSphericalVel,
        AbstractPos,
    ),
    (type[MathSphericalVel], MathSphericalVel, AbstractPos),
    (type[ProlateSpheroidalVel], ProlateSpheroidalVel, AbstractPos),
)
def vconvert(
    target: type[AbstractVel3D],
    current: AbstractVel3D,
    position: AbstractPos,
    /,
    **kwargs: Any,
) -> AbstractVel3D:
    """Self transforms for 3D velocity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    For these transformations the position does not matter since the
    self-transform returns the velocity unchanged.

    >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "km")

    Cartesian to Cartesian velocity:

    >>> dif = cx.CartesianVel3D.from_([1, 2, 3], "km/s")
    >>> cx.vconvert(cx.CartesianVel3D, dif, vec) is dif
    True

    Cylindrical to Cylindrical velocity:

    >>> dif = cx.vecs.CylindricalVel(rho=u.Quantity(1, "km/s"),
    ...                              phi=u.Quantity(2, "mas/yr"),
    ...                              z=u.Quantity(3, "km/s"))
    >>> cx.vconvert(cx.vecs.CylindricalVel, dif, vec) is dif
    True

    Spherical to Spherical velocity:

    >>> dif = cx.vecs.SphericalVel(r=u.Quantity(1, "km/s"),
    ...                            theta=u.Quantity(2, "mas/yr"),
    ...                            phi=u.Quantity(3, "mas/yr"))
    >>> cx.vconvert(cx.SphericalVel, dif, vec) is dif
    True

    LonLatSpherical to LonLatSpherical velocity:

    >>> dif = cx.vecs.LonLatSphericalVel(lon=u.Quantity(1, "mas/yr"),
    ...                                  lat=u.Quantity(2, "mas/yr"),
    ...                                  distance=u.Quantity(3, "km/s"))
    >>> cx.vconvert(cx.vecs.LonLatSphericalVel, dif, vec) is dif
    True

    LonCosLatSpherical to LonCosLatSpherical velocity:

    >>> dif = cx.vecs.LonCosLatSphericalVel(lon_coslat=u.Quantity(1, "mas/yr"),
    ...                                     lat=u.Quantity(2, "mas/yr"),
    ...                                     distance=u.Quantity(3, "km/s"))
    >>> cx.vconvert(cx.vecs.LonCosLatSphericalVel, dif, vec) is dif
    True

    MathSpherical to MathSpherical velocity:

    >>> dif = cx.vecs.MathSphericalVel(r=u.Quantity(1, "km/s"),
    ...                                theta=u.Quantity(2, "mas/yr"),
    ...                                phi=u.Quantity(3, "mas/yr"))
    >>> cx.vconvert(cx.vecs.MathSphericalVel, dif, vec) is dif
    True

    """
    return current


# =============================================================================
# CartesianPos3D


@dispatch
def vconvert(
    target: type[CylindricalPos], current: CartesianPos3D, /, **kwargs: Any
) -> CylindricalPos:
    """CartesianPos3D -> CylindricalPos.

    Examples
    --------
    >>> import coordinax as cx

    >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "km")
    >>> print(cx.vconvert(cx.vecs.CylindricalPos, vec))
    <CylindricalPos (rho[km], phi[rad], z[km])
        [2.236 1.107 3.   ]>

    """
    rho = jnp.sqrt(current.x**2 + current.y**2)
    phi = jnp.atan2(current.y, current.x)
    return target(rho=rho, phi=phi, z=current.z)


@dispatch
def vconvert(
    target: type[SphericalPos], current: CartesianPos3D, /, **kwargs: Any
) -> SphericalPos:
    """CartesianPos3D -> SphericalPos.

    Examples
    --------
    >>> import coordinax as cx

    >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "km")
    >>> print(cx.vconvert(cx.SphericalPos, vec))
    <SphericalPos (r[km], theta[rad], phi[rad])
        [3.742 0.641 1.107]>

    """
    r = jnp.sqrt(current.x**2 + current.y**2 + current.z**2)
    theta = jnp.acos(current.z / r)
    phi = jnp.atan2(current.y, current.x)
    return target(r=r, theta=theta, phi=phi)


@dispatch.multi(
    (type[LonLatSphericalPos], CartesianPos3D),
    (type[MathSphericalPos], CartesianPos3D),
)
def vconvert(
    target: type[AbstractSphericalPos],
    current: CartesianPos3D,
    /,
    **kwargs: Any,
) -> AbstractSphericalPos:
    """CartesianPos3D -> AbstractSphericalPos.

    Examples
    --------
    >>> import coordinax as cx

    >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "km")

    >>> print(cx.vconvert(cx.vecs.LonLatSphericalPos, vec))
    <LonLatSphericalPos (lon[rad], lat[deg], distance[km])
        [ 1.107 53.301  3.742]>

    >>> print(cx.vconvert(cx.vecs.MathSphericalPos, vec))
    <MathSphericalPos (r[km], theta[rad], phi[rad])
        [3.742 1.107 0.641]>

    """
    return vconvert(target, vconvert(SphericalPos, current))


# =============================================================================
# CylindricalPos


@dispatch
def vconvert(
    target: type[CartesianPos3D], current: CylindricalPos, /, **kwargs: Any
) -> CartesianPos3D:
    """CylindricalPos -> CartesianPos3D.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.vecs.CylindricalPos(rho=u.Quantity(1., "km"),
    ...                              phi=u.Quantity(90, "deg"),
    ...                              z=u.Quantity(1, "km"))
    >>> print(cx.vconvert(cx.CartesianPos3D, vec))
    <CartesianPos3D (x[km], y[km], z[km])
        [-4.371e-08  1.000e+00  1.000e+00]>

    """
    x = current.rho * jnp.cos(current.phi)
    y = current.rho * jnp.sin(current.phi)
    z = current.z
    return target(x=x, y=y, z=z)


@dispatch
def vconvert(
    target: type[SphericalPos], current: CylindricalPos, /, **kwargs: Any
) -> SphericalPos:
    """CylindricalPos -> SphericalPos.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.vecs.CylindricalPos(rho=u.Quantity(1., "km"),
    ...                              phi=u.Quantity(90, "deg"),
    ...                              z=u.Quantity(1, "km"))
    >>> print(cx.vconvert(cx.SphericalPos, vec))
    <SphericalPos (r[km], theta[rad], phi[deg])
        [ 1.414  0.785 90.   ]>

    """
    r = jnp.sqrt(current.rho**2 + current.z**2)
    theta = jnp.acos(current.z / r)
    return target(r=r, theta=theta, phi=current.phi)


@dispatch.multi(
    (type[LonLatSphericalPos], CylindricalPos),
    (type[MathSphericalPos], CylindricalPos),
)
def vconvert(
    target: type[AbstractSphericalPos],
    current: CylindricalPos,
    /,
    **kwargs: Any,
) -> AbstractSphericalPos:
    """CylindricalPos -> AbstractSphericalPos.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.vecs.CylindricalPos(rho=u.Quantity(1., "km"),
    ...                              phi=u.Quantity(90, "deg"),
    ...                              z=u.Quantity(1, "km"))

    >>> print(cx.vconvert(cx.vecs.LonLatSphericalPos, vec))
    <LonLatSphericalPos (lon[deg], lat[deg], distance[km])
        [90.    45.     1.414]>

    >>> print(cx.vconvert(cx.vecs.MathSphericalPos, vec))
    <MathSphericalPos (r[km], theta[deg], phi[rad])
        [ 1.414 90.     0.785]>

    """
    return vconvert(target, vconvert(SphericalPos, current))


# =============================================================================
# SphericalPos


@dispatch
def vconvert(
    target: type[CartesianPos3D], current: SphericalPos, /, **kwargs: Any
) -> CartesianPos3D:
    """SphericalPos -> CartesianPos3D.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.SphericalPos(r=u.Quantity(1., "km"),
    ...                       theta=u.Quantity(90, "deg"),
    ...                       phi=u.Quantity(90, "deg"))
    >>> print(cx.vconvert(cx.CartesianPos3D, vec))
    <CartesianPos3D (x[km], y[km], z[km])
        [-4.371e-08  1.000e+00 -4.371e-08]>

    """
    x = current.r.distance * jnp.sin(current.theta) * jnp.cos(current.phi)
    y = current.r.distance * jnp.sin(current.theta) * jnp.sin(current.phi)
    z = current.r.distance * jnp.cos(current.theta)
    return target(x=x, y=y, z=z)


@dispatch
def vconvert(
    target: type[CylindricalPos], current: SphericalPos, /, **kwargs: Any
) -> CylindricalPos:
    """SphericalPos -> CylindricalPos.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.SphericalPos(r=u.Quantity(1., "km"),
    ...                       theta=u.Quantity(90, "deg"),
    ...                       phi=u.Quantity(90, "deg"))
    >>> print(cx.vconvert(cx.vecs.CylindricalPos, vec))
    <CylindricalPos (rho[km], phi[deg], z[km])
        [ 1.000e+00  9.000e+01 -4.371e-08]>

    """
    rho = jnp.abs(current.r.distance * jnp.sin(current.theta))
    z = current.r.distance * jnp.cos(current.theta)
    return target(rho=rho, phi=current.phi, z=z)


@dispatch
def vconvert(
    target: type[LonLatSphericalPos], current: SphericalPos, /, **kwargs: Any
) -> LonLatSphericalPos:
    """SphericalPos -> LonLatSphericalPos.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.SphericalPos(r=u.Quantity(1., "km"),
    ...                       theta=u.Quantity(90, "deg"),
    ...                       phi=u.Quantity(90, "deg"))
    >>> print(cx.vconvert(cx.vecs.LonLatSphericalPos, vec))
    <LonLatSphericalPos (lon[deg], lat[deg], distance[km])
        [90.  0.  1.]>

    """
    return target(
        lon=current.phi, lat=u.Quantity(90, "deg") - current.theta, distance=current.r
    )


@dispatch
def vconvert(
    target: type[MathSphericalPos], current: SphericalPos, /, **kwargs: Any
) -> MathSphericalPos:
    """SphericalPos -> MathSphericalPos.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.SphericalPos(r=u.Quantity(1., "km"),
    ...                       theta=u.Quantity(90, "deg"),
    ...                       phi=u.Quantity(90, "deg"))
    >>> print(cx.vconvert(cx.vecs.MathSphericalPos, vec))
    <MathSphericalPos (r[km], theta[deg], phi[deg])
        [ 1. 90. 90.]>

    """
    return target(r=current.r, theta=current.phi, phi=current.theta)


# =============================================================================
# LonLatSphericalPos


@dispatch
def vconvert(
    target: type[CartesianPos3D],
    current: LonLatSphericalPos,
    /,
    **kwargs: Any,
) -> CartesianPos3D:
    """LonLatSphericalPos -> CartesianPos3D.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.vecs.LonLatSphericalPos(lon=u.Quantity(90, "deg"),
    ...                                  lat=u.Quantity(0, "deg"),
    ...                                  distance=u.Quantity(1., "km"))
    >>> print(cx.vconvert(cx.CartesianPos3D, vec))
    <CartesianPos3D (x[km], y[km], z[km])
        [-4.371e-08  1.000e+00 -4.371e-08]>

    """
    return vconvert(CartesianPos3D, vconvert(SphericalPos, current))


@dispatch
def vconvert(
    target: type[CylindricalPos],
    current: LonLatSphericalPos,
    /,
    **kwargs: Any,
) -> CylindricalPos:
    """LonLatSphericalPos -> CylindricalPos.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.vecs.LonLatSphericalPos(lon=u.Quantity(90, "deg"),
    ...                                  lat=u.Quantity(0, "deg"),
    ...                                  distance=u.Quantity(1., "km"))
    >>> print(cx.vconvert(cx.vecs.CylindricalPos, vec))
    <CylindricalPos (rho[km], phi[deg], z[km])
        [ 1.000e+00  9.000e+01 -4.371e-08]>

    """
    return vconvert(target, vconvert(SphericalPos, current))


@dispatch
def vconvert(
    target: type[SphericalPos], current: LonLatSphericalPos, /, **kwargs: Any
) -> SphericalPos:
    """LonLatSphericalPos -> SphericalPos.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.vecs.LonLatSphericalPos(lon=u.Quantity(90, "deg"),
    ...                                  lat=u.Quantity(0, "deg"),
    ...                                  distance=u.Quantity(1., "km"))
    >>> print(cx.vconvert(cx.SphericalPos, vec))
    <SphericalPos (r[km], theta[deg], phi[deg])
        [ 1. 90. 90.]>

    """
    return target(
        r=current.distance, theta=u.Quantity(90, "deg") - current.lat, phi=current.lon
    )


# =============================================================================
# MathSphericalPos


@dispatch
def vconvert(
    target: type[CartesianPos3D], current: MathSphericalPos, /, **kwargs: Any
) -> CartesianPos3D:
    """MathSphericalPos -> CartesianPos3D.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.vecs.MathSphericalPos(r=u.Quantity(1., "km"),
    ...                                theta=u.Quantity(90, "deg"),
    ...                                phi=u.Quantity(90, "deg"))
    >>> print(cx.vconvert(cx.CartesianPos3D, vec))
    <CartesianPos3D (x[km], y[km], z[km])
        [-4.371e-08  1.000e+00 -4.371e-08]>

    """
    x = current.r.distance * jnp.sin(current.phi) * jnp.cos(current.theta)
    y = current.r.distance * jnp.sin(current.phi) * jnp.sin(current.theta)
    z = current.r.distance * jnp.cos(current.phi)
    return target(x=x, y=y, z=z)


@dispatch
def vconvert(
    target: type[CylindricalPos], current: MathSphericalPos, /, **kwargs: Any
) -> CylindricalPos:
    """MathSphericalPos -> CylindricalPos.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.vecs.MathSphericalPos(r=u.Quantity(1., "km"),
    ...                                theta=u.Quantity(90, "deg"),
    ...                                phi=u.Quantity(90, "deg"))
    >>> print(cx.vconvert(cx.vecs.CylindricalPos, vec))
    <CylindricalPos (rho[km], phi[deg], z[km])
        [ 1.000e+00  9.000e+01 -4.371e-08]>

    """
    rho = jnp.abs(current.r.distance * jnp.sin(current.phi))
    z = current.r.distance * jnp.cos(current.phi)
    return target(rho=rho, phi=current.theta, z=z)


@dispatch
def vconvert(
    target: type[SphericalPos], current: MathSphericalPos, /, **kwargs: Any
) -> SphericalPos:
    """MathSphericalPos -> SphericalPos.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.vecs.MathSphericalPos(r=u.Quantity(1., "km"),
    ...                                theta=u.Quantity(90, "deg"),
    ...                                phi=u.Quantity(90, "deg"))
    >>> print(cx.vconvert(cx.SphericalPos, vec))
    <SphericalPos (r[km], theta[deg], phi[deg])
        [ 1. 90. 90.]>

    """
    return target(r=current.r, theta=current.phi, phi=current.theta)


# =============================================================================
# ProlateSpheroidalPos


@dispatch
def vconvert(
    target: type[CylindricalPos], current: ProlateSpheroidalPos, /, **kwargs: Any
) -> CylindricalPos:
    """ProlateSpheroidalPos -> CylindricalPos.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.vecs.ProlateSpheroidalPos(
    ...     mu=u.Quantity(1., "km2"),
    ...     nu=u.Quantity(0.2, "km2"),
    ...     phi=u.Quantity(90, "deg"),
    ...     Delta=u.Quantity(0.5, "km")
    ... )
    >>> print(cx.vconvert(cx.vecs.CylindricalPos, vec))
    <CylindricalPos (rho[km], phi[deg], z[km])
        [ 0.387 90.     0.894]>

    """
    Delta2 = current.Delta**2
    nu_D2 = jnp.abs(current.nu) / Delta2

    R = jnp.sqrt((current.mu - Delta2) * (1 - nu_D2))
    z = jnp.sqrt(current.mu * nu_D2) * jnp.sign(current.nu)

    return target(rho=R, phi=current.phi, z=z)


@dispatch
def vconvert(
    target: type[ProlateSpheroidalPos],
    current: CylindricalPos,
    /,
    **kwargs: Any,
) -> ProlateSpheroidalPos:
    """CylindricalPos -> ProlateSpheroidalPos.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.vecs.CylindricalPos(
    ...     rho=u.Quantity(1., "km"),
    ...     phi=u.Quantity(90, "deg"),
    ...     z=u.Quantity(1, "km")
    ... )
    >>> print(vec.vconvert(cx.vecs.ProlateSpheroidalPos,
    ...                    Delta=u.Quantity(0.5, "km")))
    <ProlateSpheroidalPos (mu[km2], nu[km2], phi[deg])
        [ 2.133  0.117 90.   ]>

    """
    Delta = eqx.error_if(
        kwargs.get("Delta"),
        "Delta" not in kwargs,
        "Delta must be provided for ProlateSpheroidalPos.",
    )
    R2 = current.rho**2
    z2 = current.z**2
    Delta2 = Delta**2

    sum_ = R2 + z2 + Delta2
    diff_ = R2 + z2 - Delta2

    # compute D = sqrt((R² + z² - Δ²)² + 4R²Δ²)
    D = jnp.sqrt(diff_**2 + 4 * R2 * Delta2)

    # handle special cases for R=0 or z=0
    D = jnp.where(current.z == 0, sum_, D)
    D = jnp.where(current.rho == 0, jnp.abs(diff_), D)

    # compute mu and nu depending on sign of diff_ - avoids dividing by a small number
    pos_mu_minus_delta = 0.5 * (D + diff_)
    pos_delta_minus_nu = Delta2 * R2 / pos_mu_minus_delta

    neg_delta_minus_nu = 0.5 * (D - diff_)
    neg_mu_minus_delta = Delta2 * R2 / neg_delta_minus_nu

    # Select based on condition
    mu_minus_delta = jnp.where(diff_ >= 0, pos_mu_minus_delta, neg_mu_minus_delta)
    delta_minus_nu = jnp.where(diff_ >= 0, pos_delta_minus_nu, neg_delta_minus_nu)

    # compute mu and nu:
    mu = Delta2 + mu_minus_delta
    abs_nu = 2 * Delta2 / (sum_ + D) * z2

    # for numerical stability when Delta^2-|nu| is small
    abs_nu = jnp.where(abs_nu * 2 > Delta2, Delta2 - delta_minus_nu, abs_nu)

    nu = abs_nu * jnp.sign(current.z)

    return target(mu=mu, nu=nu, phi=current.phi, Delta=Delta)


@dispatch
def vconvert(
    target: type[AbstractPos3D], current: ProlateSpheroidalPos, /, **kwargs: Any
) -> AbstractPos3D:
    """ProlateSpheroidalPos -> AbstractPos3D.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.vecs.ProlateSpheroidalPos(
    ...     mu=u.Quantity(1., "km2"),
    ...     nu=u.Quantity(0.2, "km2"),
    ...     phi=u.Quantity(90, "deg"),
    ...     Delta=u.Quantity(0.5, "km")
    ... )
    >>> print(cx.vconvert(cx.CartesianPos3D, vec))
    <CartesianPos3D (x[km], y[km], z[km])
        [-1.693e-08  3.873e-01  8.944e-01]>

    """
    return current.vconvert(CylindricalPos).vconvert(target)


@dispatch
def vconvert(
    target: type[ProlateSpheroidalPos],
    current: ProlateSpheroidalPos,
    /,
    Delta: u.Quantity["length"] | None = None,
    **kwargs: Any,
) -> ProlateSpheroidalPos:
    """ProlateSpheroidalPos -> ProlateSpheroidalPos.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    Self-transforms can change the focal length:

    >>> vec = cx.vecs.ProlateSpheroidalPos(
    ...     mu=u.Quantity(1., "km2"),
    ...     nu=u.Quantity(0.2, "km2"),
    ...     phi=u.Quantity(90, "deg"),
    ...     Delta=u.Quantity(0.5, "km")
    ... )
    >>> print(cx.vconvert(cx.vecs.ProlateSpheroidalPos, vec,
    ...                   Delta=u.Quantity(0.8, "km")))
    <ProlateSpheroidalPos...>

    Without changing the focal length, no transform is done:

    >>> vec2 = cx.vconvert(cx.vecs.ProlateSpheroidalPos, vec)
    >>> vec is vec2
    True

    """
    if Delta is None:
        return current
    return current.vconvert(CylindricalPos).vconvert(target, Delta=Delta)


@dispatch
def vconvert(
    target: type[ProlateSpheroidalPos],
    current: AbstractPos3D,
    /,
    **kwargs: Any,
) -> ProlateSpheroidalPos:
    """AbstractPos3D -> ProlateSpheroidalPos.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.vecs.ProlateSpheroidalPos(
    ...     mu=u.Quantity(1., "km2"),
    ...     nu=u.Quantity(0.2, "km2"),
    ...     phi=u.Quantity(90, "deg"),
    ...     Delta=u.Quantity(0.5, "km")
    ... )
    >>> print(cx.vconvert(cx.CartesianPos3D, vec))
    <CartesianPos3D (x[km], y[km], z[km])
        [-1.693e-08  3.873e-01  8.944e-01]>

    Self-transforms also work to change the focal length:

    >>> vec = cx.vecs.ProlateSpheroidalPos(
    ...     mu=u.Quantity(1., "km2"),
    ...     nu=u.Quantity(0.2, "km2"),
    ...     phi=u.Quantity(90, "deg"),
    ...     Delta=u.Quantity(0.5, "km")
    ... )
    >>> print(cx.vconvert(cx.vecs.ProlateSpheroidalPos, vec,
    ...                   Delta=u.Quantity(0.8, "km")))
    <ProlateSpheroidalPos...>

    """
    Delta = eqx.error_if(
        kwargs.get("Delta"),
        "Delta" not in kwargs,
        "Delta must be provided for ProlateSpheroidalPos.",
    )
    cyl = vconvert(CylindricalPos, current)
    return vconvert(target, cyl, Delta=Delta)


# =============================================================================
# LonLatSphericalVel


@dispatch
def vconvert(
    target: type[LonCosLatSphericalVel],
    current: AbstractVel3D,
    position: AbstractPos | u.Quantity["length"],
    /,
    **kwargs: Any,
) -> LonCosLatSphericalVel:
    """AbstractVel3D -> LonCosLatSphericalVel.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> q = cx.vecs.LonLatSphericalPos(lon=u.Quantity(15, "deg"),
    ...                                lat=u.Quantity(10, "deg"),
    ...                                distance=u.Quantity(1.5, "km"))
    >>> p = cx.vecs.LonLatSphericalVel(lon=u.Quantity(7, "mas/yr"),
    ...                                lat=u.Quantity(0, "deg/Gyr"),
    ...                                distance=u.Quantity(-5, "km/s"))
    >>> newp = cx.vconvert(cx.vecs.LonCosLatSphericalVel, p, q)
    >>> print(newp)
    <LonCosLatSphericalVel (lon_coslat[mas / yr], lat[deg / Gyr], distance[km / s])
        [ 6.894  0.    -5.   ]>

    """
    # Parse the position to an AbstractPos
    if isinstance(position, AbstractPos):
        posvec = position
    else:  # Q -> Cart<X>D
        posvec = current.integral_cls._cartesian_cls.from_(  # noqa: SLF001
            position
        )

    # Transform the differential to LonLatSphericalVel
    current = vconvert(LonLatSphericalVel, current, posvec)

    # Transform the position to the required type
    posvec = vconvert(current.integral_cls, posvec)

    # Calculate the differential in the new system
    return target(
        lon_coslat=current.lon * jnp.cos(posvec.lat),
        lat=current.lat,
        distance=current.distance,
    )


@dispatch
def vconvert(
    target: type[LonLatSphericalVel],
    current: LonCosLatSphericalVel,
    position: AbstractPos | u.Quantity["length"],
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
    posvec = vconvert(current.integral_cls, posvec)

    # Calculate the differential in the new system
    return target(
        lon=current.lon_coslat / jnp.cos(posvec.lat),
        lat=current.lat,
        distance=current.distance,
    )


@dispatch
def vconvert(
    target: type[AbstractVel3D],
    current: LonCosLatSphericalVel,
    position: AbstractPos | u.Quantity["length"],
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
    current = vconvert(LonLatSphericalVel, current, posvec)
    # Transform the position to the required type
    return vconvert(target, current, posvec)


# =============================================================================
# CartesianVel3D


@dispatch
def vconvert(
    target: type[CartesianVel3D], current: CartesianVel3D, /
) -> CartesianVel3D:
    """CartesianVel3D -> CartesianVel3D with no position.

    Cartesian coordinates are an affine coordinate system and so the
    transformation of an n-th order derivative vector in this system do not
    require lower-order derivatives to be specified. See
    https://en.wikipedia.org/wiki/Tensors_in_curvilinear_coordinates for more
    information. This mixin provides a corresponding implementation of the
    `coordinax.vconvert` method for Cartesian velocities.

    Examples
    --------
    >>> import coordinax as cx
    >>> v = cx.CartesianVel3D.from_([1, 1, 1], "m/s")
    >>> cx.vconvert(cx.CartesianVel3D, v) is v
    True

    """
    return current


# =============================================================================
# CartesianAcc3D


@dispatch
def vconvert(
    target: type[CartesianAcc3D], current: CartesianAcc3D, /
) -> CartesianAcc3D:
    """CartesianAcc3D -> CartesianAcc3D with no position.

    Cartesian coordinates are an affine coordinate system and so the
    transformation of an n-th order derivative vector in this system do not
    require lower-order derivatives to be specified. See
    https://en.wikipedia.org/wiki/Tensors_in_curvilinear_coordinates for more
    information. This mixin provides a corresponding implementation of the
    `coordinax.vconvert` method for Cartesian vectors.

    Examples
    --------
    >>> import coordinax as cx
    >>> a = cx.vecs.CartesianAcc3D.from_([1, 1, 1], "m/s2")
    >>> cx.vconvert(cx.vecs.CartesianAcc3D, a) is a
    True

    """
    return current


#####################################################################


# from coordinax.vectors.funcs
@dispatch
@partial(eqx.filter_jit, inline=True)
def normalize_vector(obj: CartesianPos3D, /) -> CartesianGeneric3D:
    """Return the norm of the vector.

    This has length 1.

    .. note::

        The unit vector is dimensionless, even if the input vector has units.
        This is because the unit vector is a ratio of two quantities: each
        component and the norm of the vector.

    Returns
    -------
    CartesianGeneric3D
        The norm of the vector.

    Examples
    --------
    >>> import coordinax as cx
    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "km")
    >>> print(cx.vecs.normalize_vector(q))
    <CartesianGeneric3D (x[], y[], z[])
        [0.267 0.535 0.802]>

    """
    norm: AbstractQuantity = obj.norm()  # type: ignore[misc]
    return CartesianGeneric3D(x=obj.x / norm, y=obj.y / norm, z=obj.z / norm)
