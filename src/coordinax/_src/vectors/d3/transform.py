"""Representation of coordinates in different systems."""

__all__: list[str] = []

from typing import Any

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
from coordinax._src.distance import AbstractDistance
from coordinax._src.vectors.base import AbstractPos

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
    >>> import unxt as u
    >>> import coordinax as cx

    Cartesian to Cartesian:

    >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> cx.represent_as(vec, cx.CartesianPos3D) is vec
    True

    Cylindrical to Cylindrical:

    >>> vec = cx.CylindricalPos(rho=u.Quantity(1, "kpc"), phi=u.Quantity(2, "deg"),
    ...                         z=u.Quantity(3, "kpc"))
    >>> cx.represent_as(vec, cx.CylindricalPos) is vec
    True

    Spherical to Spherical:

    >>> vec = cx.SphericalPos(r=u.Quantity(1, "kpc"), theta=u.Quantity(2, "deg"),
    ...                       phi=u.Quantity(3, "deg"))
    >>> cx.represent_as(vec, cx.SphericalPos) is vec
    True

    LonLatSpherical to LonLatSpherical:

    >>> vec = cx.LonLatSphericalPos(lon=u.Quantity(1, "deg"), lat=u.Quantity(2, "deg"),
    ...                             distance=u.Quantity(3, "kpc"))
    >>> cx.represent_as(vec, cx.LonLatSphericalPos) is vec
    True

    MathSpherical to MathSpherical:

    >>> vec = cx.MathSphericalPos(r=u.Quantity(1, "kpc"), theta=u.Quantity(2, "deg"),
    ...                           phi=u.Quantity(3, "deg"))
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
    (ProlateSpheroidalVel, type[ProlateSpheroidalVel], AbstractPos),
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
    >>> import unxt as u
    >>> import coordinax as cx

    For these transformations the position does not matter since the
    self-transform returns the velocity unchanged.

    >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "kpc")

    Cartesian to Cartesian velocity:

    >>> dif = cx.CartesianVel3D.from_([1, 2, 3], "km/s")
    >>> cx.represent_as(dif, cx.CartesianVel3D, vec) is dif
    True

    Cylindrical to Cylindrical velocity:

    >>> dif = cx.CylindricalVel(d_rho=u.Quantity(1, "km/s"),
    ...                         d_phi=u.Quantity(2, "mas/yr"),
    ...                         d_z=u.Quantity(3, "km/s"))
    >>> cx.represent_as(dif, cx.CylindricalVel, vec) is dif
    True

    Spherical to Spherical velocity:

    >>> dif = cx.SphericalVel(d_r=u.Quantity(1, "km/s"),
    ...                       d_theta=u.Quantity(2, "mas/yr"),
    ...                       d_phi=u.Quantity(3, "mas/yr"))
    >>> cx.represent_as(dif, cx.SphericalVel, vec) is dif
    True

    LonLatSpherical to LonLatSpherical velocity:

    >>> dif = cx.LonLatSphericalVel(d_lon=u.Quantity(1, "mas/yr"),
    ...                             d_lat=u.Quantity(2, "mas/yr"),
    ...                             d_distance=u.Quantity(3, "km/s"))
    >>> cx.represent_as(dif, cx.LonLatSphericalVel, vec) is dif
    True

    LonCosLatSpherical to LonCosLatSpherical velocity:

    >>> dif = cx.LonCosLatSphericalVel(d_lon_coslat=u.Quantity(1, "mas/yr"),
    ...                                d_lat=u.Quantity(2, "mas/yr"),
    ...                                d_distance=u.Quantity(3, "km/s"))
    >>> cx.represent_as(dif, cx.LonCosLatSphericalVel, vec) is dif
    True

    MathSpherical to MathSpherical velocity:

    >>> dif = cx.MathSphericalVel(d_r=u.Quantity(1, "km/s"),
    ...                           d_theta=u.Quantity(2, "mas/yr"),
    ...                           d_phi=u.Quantity(3, "mas/yr"))
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
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.CylindricalPos(rho=u.Quantity(1., "kpc"), phi=u.Quantity(90, "deg"),
    ...                         z=u.Quantity(1, "kpc"))
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
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.CylindricalPos(rho=u.Quantity(1., "kpc"), phi=u.Quantity(90, "deg"),
    ...                         z=u.Quantity(1, "kpc"))
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
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.CylindricalPos(rho=u.Quantity(1., "kpc"), phi=u.Quantity(90, "deg"),
    ...                         z=u.Quantity(1, "kpc"))

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
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.SphericalPos(r=u.Quantity(1., "kpc"), theta=u.Quantity(90, "deg"),
    ...                       phi=u.Quantity(90, "deg"))
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
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.SphericalPos(r=u.Quantity(1., "kpc"), theta=u.Quantity(90, "deg"),
    ...                       phi=u.Quantity(90, "deg"))
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
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.SphericalPos(r=u.Quantity(1., "kpc"), theta=u.Quantity(90, "deg"),
    ...                       phi=u.Quantity(90, "deg"))
    >>> print(cx.represent_as(vec, cx.LonLatSphericalPos))
    <LonLatSphericalPos (lon[deg], lat[deg], distance[kpc])
        [90.  0.  1.]>

    """
    return target(
        lon=current.phi, lat=u.Quantity(90, "deg") - current.theta, distance=current.r
    )


@dispatch
def represent_as(
    current: SphericalPos, target: type[MathSphericalPos], /, **kwargs: Any
) -> MathSphericalPos:
    """SphericalPos -> MathSphericalPos.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.SphericalPos(r=u.Quantity(1., "kpc"), theta=u.Quantity(90, "deg"),
    ...                       phi=u.Quantity(90, "deg"))
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
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.LonLatSphericalPos(lon=u.Quantity(90, "deg"),
    ...                             lat=u.Quantity(0, "deg"),
    ...                             distance=u.Quantity(1., "kpc"))
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
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.LonLatSphericalPos(lon=u.Quantity(90, "deg"),
    ...                             lat=u.Quantity(0, "deg"),
    ...                             distance=u.Quantity(1., "kpc"))
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
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.LonLatSphericalPos(lon=u.Quantity(90, "deg"),
    ...                             lat=u.Quantity(0, "deg"),
    ...                             distance=u.Quantity(1., "kpc"))
    >>> print(cx.represent_as(vec, cx.SphericalPos))
    <SphericalPos (r[kpc], theta[deg], phi[deg])
        [ 1. 90. 90.]>

    """
    return target(
        r=current.distance, theta=u.Quantity(90, "deg") - current.lat, phi=current.lon
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
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.MathSphericalPos(r=u.Quantity(1., "kpc"), theta=u.Quantity(90, "deg"),
    ...                           phi=u.Quantity(90, "deg"))
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
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.MathSphericalPos(r=u.Quantity(1., "kpc"), theta=u.Quantity(90, "deg"),
    ...                           phi=u.Quantity(90, "deg"))
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
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.MathSphericalPos(r=u.Quantity(1., "kpc"), theta=u.Quantity(90, "deg"),
    ...                           phi=u.Quantity(90, "deg"))
    >>> print(cx.represent_as(vec, cx.SphericalPos))
    <SphericalPos (r[kpc], theta[deg], phi[deg])
        [ 1. 90. 90.]>

    """
    return target(r=current.r, theta=current.phi, phi=current.theta)


# =============================================================================
# ProlateSpheroidalPos


@dispatch
def represent_as(
    current: ProlateSpheroidalPos, target: type[CylindricalPos], /, **kwargs: Any
) -> CylindricalPos:
    """ProlateSpheroidalPos -> CylindricalPos.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.ProlateSpheroidalPos(
    ...     mu=Quantity(1., "kpc2"),
    ...     nu=Quantity(0.2, "kpc2"),
    ...     phi=Quantity(90, "deg"),
    ...     Delta=Quantity(0.5, "kpc")
    ... )
    >>> print(cx.represent_as(vec, cx.CylindricalPos))
    TODO: add

    """
    Delta2 = current.Delta**2
    absnu = xp.abs(current.nu)

    mu_minus_delta = current.mu - Delta2
    # nu_minus_delta = absnu - Delta2

    R = xp.sqrt(mu_minus_delta * (1 - absnu / Delta2))
    z = xp.sqrt(current.mu * absnu / Delta2) * xp.sign(current.nu)

    return target(rho=R, phi=current.phi, z=z)


@dispatch
def represent_as(
    current: CylindricalPos,
    target: type[ProlateSpheroidalPos],
    *,
    Delta: AbstractDistance | Quantity["length"],  # noqa: N803
    **kwargs: Any,
) -> ProlateSpheroidalPos:
    """CylindricalPos -> ProlateSpheroidalPos.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.CylindricalPos(
    ...     rho=Quantity(1., "kpc"),
    ...     phi=Quantity(90, "deg"),
    ...     z=Quantity(1, "kpc")
    ... )
    >>> print(vec.represent_as(cx.ProlateSpheroidalPos, Delta=Quantity(0.5, "kpc")))
    <ProlateSpheroidalPos (mu[kpc2], nu[kpc2], phi[deg], Delta[kpc])
        [ 2.133  0.117 90.     0.5  ]>

    """
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
def represent_as(
    current: ProlateSpheroidalPos, target: type[CartesianPos3D], /, **kwargs: Any
) -> CartesianPos3D:
    """ProlateSpheroidalPos -> CartesianPos3D."""
    cyl = represent_as(current, CylindricalPos)
    return represent_as(cyl, target)


@dispatch
def represent_as(
    current: CartesianPos3D,
    target: type[ProlateSpheroidalPos],
    *,
    Delta: AbstractDistance | Quantity["length"],  # noqa: N803
    **kwargs: Any,
) -> ProlateSpheroidalPos:
    """CartesianPos3D -> ProlateSpheroidalPos."""
    cyl = represent_as(current, CylindricalPos)
    return represent_as(cyl, target, Delta=Delta)


@dispatch
def represent_as(
    current: ProlateSpheroidalPos, target: type[ProlateSpheroidalPos], /
) -> ProlateSpheroidalPos:
    """ProlateSpheroidalPos -> ProlateSpheroidalPos."""
    return current


@dispatch
def represent_as(
    current: ProlateSpheroidalPos,
    target: type[ProlateSpheroidalPos],
    *,
    Delta: AbstractDistance | Quantity["length"],  # noqa: N803
    **kwargs: Any,
) -> ProlateSpheroidalPos:
    """ProlateSpheroidalPos -> ProlateSpheroidalPos."""
    cyl = represent_as(current, CylindricalPos)
    return represent_as(cyl, target, Delta=Delta)


# =============================================================================
# LonLatSphericalVel


@dispatch
def represent_as(
    current: AbstractVel3D,
    target: type[LonCosLatSphericalVel],
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

    >>> vec = cx.LonLatSphericalPos(lon=u.Quantity(15, "deg"),
    ...                             lat=u.Quantity(10, "deg"),
    ...                             distance=u.Quantity(1.5, "kpc"))
    >>> dif = cx.LonLatSphericalVel(d_lon=u.Quantity(7, "mas/yr"),
    ...                             d_lat=u.Quantity(0, "deg/Gyr"),
    ...                             d_distance=u.Quantity(-5, "km/s"))
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
