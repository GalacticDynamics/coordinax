"""Representation of coordinates in different systems."""
# ruff: noqa: N803, N806

__all__: list[str] = []

from typing import Any

import equinox as eqx
from plum import dispatch

import quaxed.numpy as xp
import unxt as u

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
from .spheroidal import ProlateSpheroidalPos, ProlateSpheroidalVel
from coordinax._src.vectors.base import AbstractPos

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

    >>> dif = cx.vecs.CylindricalVel(d_rho=u.Quantity(1, "km/s"),
    ...                              d_phi=u.Quantity(2, "mas/yr"),
    ...                              d_z=u.Quantity(3, "km/s"))
    >>> cx.vconvert(cx.vecs.CylindricalVel, dif, vec) is dif
    True

    Spherical to Spherical velocity:

    >>> dif = cx.SphericalVel(d_r=u.Quantity(1, "km/s"),
    ...                       d_theta=u.Quantity(2, "mas/yr"),
    ...                       d_phi=u.Quantity(3, "mas/yr"))
    >>> cx.vconvert(cx.SphericalVel, dif, vec) is dif
    True

    LonLatSpherical to LonLatSpherical velocity:

    >>> dif = cx.vecs.LonLatSphericalVel(d_lon=u.Quantity(1, "mas/yr"),
    ...                                  d_lat=u.Quantity(2, "mas/yr"),
    ...                                  d_distance=u.Quantity(3, "km/s"))
    >>> cx.vconvert(cx.vecs.LonLatSphericalVel, dif, vec) is dif
    True

    LonCosLatSpherical to LonCosLatSpherical velocity:

    >>> dif = cx.vecs.LonCosLatSphericalVel(d_lon_coslat=u.Quantity(1, "mas/yr"),
    ...                                     d_lat=u.Quantity(2, "mas/yr"),
    ...                                     d_distance=u.Quantity(3, "km/s"))
    >>> cx.vconvert(cx.vecs.LonCosLatSphericalVel, dif, vec) is dif
    True

    MathSpherical to MathSpherical velocity:

    >>> dif = cx.vecs.MathSphericalVel(d_r=u.Quantity(1, "km/s"),
    ...                                d_theta=u.Quantity(2, "mas/yr"),
    ...                                d_phi=u.Quantity(3, "mas/yr"))
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
    rho = xp.sqrt(current.x**2 + current.y**2)
    phi = xp.atan2(current.y, current.x)
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
    r = xp.sqrt(current.x**2 + current.y**2 + current.z**2)
    theta = xp.acos(current.z / r)
    phi = xp.atan2(current.y, current.x)
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
    x = current.rho * xp.cos(current.phi)
    y = current.rho * xp.sin(current.phi)
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
    r = xp.sqrt(current.rho**2 + current.z**2)
    theta = xp.acos(current.z / r)
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
    x = current.r.distance * xp.sin(current.theta) * xp.cos(current.phi)
    y = current.r.distance * xp.sin(current.theta) * xp.sin(current.phi)
    z = current.r.distance * xp.cos(current.theta)
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
    rho = xp.abs(current.r.distance * xp.sin(current.theta))
    z = current.r.distance * xp.cos(current.theta)
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
    x = current.r.distance * xp.sin(current.phi) * xp.cos(current.theta)
    y = current.r.distance * xp.sin(current.phi) * xp.sin(current.theta)
    z = current.r.distance * xp.cos(current.phi)
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
    rho = xp.abs(current.r.distance * xp.sin(current.phi))
    z = current.r.distance * xp.cos(current.phi)
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
    nu_D2 = xp.abs(current.nu) / Delta2

    R = xp.sqrt((current.mu - Delta2) * (1 - nu_D2))
    z = xp.sqrt(current.mu * nu_D2) * xp.sign(current.nu)

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
    D = xp.sqrt(diff_**2 + 4 * R2 * Delta2)

    # handle special cases for R=0 or z=0
    # TODO: quaxed.numpy.select doesn't work with Quantity's?
    # D = xp.select(
    #     [current.z == 0, current.rho == 0, xp.full(current.rho.shape, 1, dtype=bool)],
    #     [
    #         sum_,  # z=0 case
    #         xp.abs(diff_),  # R=0 case
    #         D,  # otherwise
    #     ],
    # )
    D = xp.where(current.z == 0, sum_, D)
    D = xp.where(current.rho == 0, xp.abs(diff_), D)

    # compute mu and nu depending on sign of diff_ - avoids dividing by a small number
    pos_mu_minus_delta = 0.5 * (D + diff_)
    pos_delta_minus_nu = Delta2 * R2 / pos_mu_minus_delta

    neg_delta_minus_nu = 0.5 * (D - diff_)
    neg_mu_minus_delta = Delta2 * R2 / neg_delta_minus_nu

    # Select based on condition
    mu_minus_delta = xp.where(diff_ >= 0, pos_mu_minus_delta, neg_mu_minus_delta)
    delta_minus_nu = xp.where(diff_ >= 0, pos_delta_minus_nu, neg_delta_minus_nu)

    # compute mu and nu:
    mu = Delta2 + mu_minus_delta
    abs_nu = 2 * Delta2 / (sum_ + D) * z2

    # for numerical stability when Delta^2-|nu| is small
    abs_nu = xp.where(abs_nu * 2 > Delta2, Delta2 - delta_minus_nu, abs_nu)

    nu = abs_nu * xp.sign(current.z)

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
    >>> p = cx.vecs.LonLatSphericalVel(d_lon=u.Quantity(7, "mas/yr"),
    ...                                d_lat=u.Quantity(0, "deg/Gyr"),
    ...                                d_distance=u.Quantity(-5, "km/s"))
    >>> newp = cx.vconvert(cx.vecs.LonCosLatSphericalVel, p, q)
    >>> print(newp)
    <LonCosLatSphericalVel (d_lon_coslat[mas / yr], d_lat[deg / Gyr], d_distance[km / s])
        [ 6.894  0.    -5.   ]>

    """  # noqa: E501
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
        d_lon_coslat=current.d_lon * xp.cos(posvec.lat),
        d_lat=current.d_lat,
        d_distance=current.d_distance,
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
        d_lon=current.d_lon_coslat / xp.cos(posvec.lat),
        d_lat=current.d_lat,
        d_distance=current.d_distance,
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
