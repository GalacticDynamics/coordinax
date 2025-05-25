"""Representation of coordinates in different systems."""
# ruff: noqa: N803, N806

__all__: list[str] = []

import functools as ft
from typing import Any

import equinox as eqx
import jax
from plum import dispatch

import quaxed.lax as qlax
import quaxed.numpy as jnp
import unxt as u

import coordinax._src.vectors.custom_types as ct
from .base import AbstractAcc3D, AbstractPos3D, AbstractVel3D
from .base_spherical import AbstractSphericalPos, _90d, _180d, _360d
from .cartesian import CartesianAcc3D, CartesianPos3D, CartesianVel3D
from .cylindrical import CylindricalAcc, CylindricalPos, CylindricalVel
from .generic import Cartesian3D
from .lonlatspherical import (
    LonCosLatSphericalVel,
    LonLatSphericalAcc,
    LonLatSphericalPos,
    LonLatSphericalVel,
)
from .mathspherical import MathSphericalAcc, MathSphericalPos, MathSphericalVel
from .spherical import SphericalAcc, SphericalPos, SphericalVel
from .spheroidal import ProlateSpheroidalAcc, ProlateSpheroidalPos, ProlateSpheroidalVel
from coordinax._src.vectors.base import AbstractVector
from coordinax._src.vectors.base_pos import AbstractPos
from coordinax._src.vectors.private_api import combine_aux, wrap_vconvert_impl_params

###############################################################################


@dispatch(precedence=1)
def vector(cls: type[AbstractPos3D], obj: AbstractPos3D, /) -> AbstractPos3D:
    """Construct from a 3D position.

    Examples
    --------
    >>> import coordinax.vecs as cxv

    >>> cart = cxv.CartesianPos3D.from_([1, 2, 3], "km")
    >>> cxv.AbstractPos3D.from_(cart) is cart
    True

    >>> sph = cart.vconvert(cxv.SphericalPos)
    >>> cxv.AbstractPos3D.from_(sph) is sph
    True

    >>> cyl = cart.vconvert(cxv.CylindricalPos)
    >>> cxv.AbstractPos3D.from_(cyl) is cyl
    True

    """
    return obj


# ---------------------------------------------------------


@dispatch(precedence=1)
def vector(cls: type[AbstractVel3D], obj: AbstractVel3D, /) -> AbstractVel3D:
    """Construct from a 3D velocity.

    Examples
    --------
    >>> import coordinax.vecs as cxv

    >>> q = cxv.CartesianPos3D.from_([1, 1, 1], "km")

    >>> cart = cxv.CartesianVel3D.from_([1, 2, 3], "km/s")
    >>> cxv.AbstractVel3D.from_(cart) is cart
    True

    >>> sph = cart.vconvert(cxv.SphericalVel, q)
    >>> cxv.AbstractVel3D.from_(sph) is sph
    True

    >>> cyl = cart.vconvert(cxv.CylindricalVel, q)
    >>> cxv.AbstractVel3D.from_(cyl) is cyl
    True

    """
    return obj


# ---------------------------------------------------------


@dispatch(precedence=1)
def vector(cls: type[AbstractAcc3D], obj: AbstractAcc3D, /) -> AbstractAcc3D:
    """Construct from a 3D velocity.

    Examples
    --------
    >>> import coordinax.vecs as cxv

    >>> q = cxv.CartesianPos3D.from_([1, 1, 1], "km")
    >>> p = cxv.CartesianVel3D.from_([1, 1, 1], "km/s")

    >>> cart = cxv.CartesianAcc3D.from_([1, 2, 3], "km/s2")
    >>> cxv.AbstractAcc3D.from_(cart) is cart
    True

    >>> sph = cart.vconvert(cxv.SphericalAcc, p, q)
    >>> cxv.AbstractAcc3D.from_(sph) is sph
    True

    >>> cyl = cart.vconvert(cxv.CylindricalAcc, p, q)
    >>> cxv.AbstractAcc3D.from_(cyl) is cyl
    True

    """
    return obj


# ---------------------------------------------------------


@dispatch
def vector(
    cls: type[SphericalPos],
    *,
    r: u.AbstractQuantity,
    theta: u.AbstractQuantity,
    phi: u.AbstractQuantity,
) -> SphericalPos:
    """Construct SphericalPos, allowing for out-of-range values.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    Let's start with a valid input:

    >>> vec = cxv.SphericalPos.from_(r=u.Quantity(3, "km"),
    ...                              theta=u.Quantity(90, "deg"),
    ...                              phi=u.Quantity(0, "deg"))
    >>> print(vec)
    <SphericalPos: (r[km], theta[deg], phi[deg])
        [ 3 90  0]>

    The radial distance can be negative, which wraps the azimuthal angle by 180
    degrees and flips the polar angle:

    >>> vec = cxv.SphericalPos.from_(r=u.Quantity(-3, "km"),
    ...                              theta=u.Quantity(45, "deg"),
    ...                              phi=u.Quantity(0, "deg"))
    >>> print(vec)
    <SphericalPos: (r[km], theta[deg], phi[deg])
        [  3 135 180]>

    The polar angle can be outside the [0, 180] deg range, causing the azimuthal
    angle to be shifted by 180 degrees:

    >>> vec = cxv.SphericalPos.from_(r=u.Quantity(3, "km"),
    ...                              theta=u.Quantity(190, "deg"),
    ...                              phi=u.Quantity(0, "deg"))
    >>> print(vec)
    <SphericalPos: (r[km], theta[deg], phi[deg])
        [  3 170 180]>

    The azimuth can be outside the [0, 360) deg range. This is wrapped to the
    [0, 360) deg range (actually the base from_ does this):

    >>> vec = cxv.SphericalPos.from_(r=u.Quantity(3, "km"),
    ...                              theta=u.Quantity(90, "deg"),
    ...                              phi=u.Quantity(365, "deg"))
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
    lon: u.AbstractQuantity,
    lat: u.AbstractQuantity,
    distance: u.AbstractQuantity,
) -> LonLatSphericalPos:
    """Construct LonLatSphericalPos, allowing for out-of-range values.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    Let's start with a valid input:

    >>> vec = cxv.LonLatSphericalPos.from_(lon=u.Quantity(0, "deg"),
    ...                                    lat=u.Quantity(0, "deg"),
    ...                                    distance=u.Quantity(3, "km"))
    >>> print(vec)
    <LonLatSphericalPos: (lon[deg], lat[deg], distance[km])
        [0 0 3]>

    The distance can be negative, which wraps the longitude by 180 degrees and
    flips the latitude:

    >>> vec = cxv.LonLatSphericalPos.from_(lon=u.Quantity(0, "deg"),
    ...                                    lat=u.Quantity(45, "deg"),
    ...                                    distance=u.Quantity(-3, "km"))
    >>> print(vec)
    <LonLatSphericalPos: (lon[deg], lat[deg], distance[km])
        [180 -45   3]>

    The latitude can be outside the [-90, 90] deg range, causing the longitude
    to be shifted by 180 degrees:

    >>> vec = cxv.LonLatSphericalPos.from_(lon=u.Quantity(0, "deg"),
    ...                                    lat=u.Quantity(-100, "deg"),
    ...                                    distance=u.Quantity(3, "km"))
    >>> print(vec)
    <LonLatSphericalPos: (lon[deg], lat[deg], distance[km])
        [180 -80   3]>

    >>> vec = cxv.LonLatSphericalPos.from_(lon=u.Quantity(0, "deg"),
    ...                                    lat=u.Quantity(100, "deg"),
    ...                                    distance=u.Quantity(3, "km"))
    >>> print(vec)
    <LonLatSphericalPos: (lon[deg], lat[deg], distance[km])
        [180  80   3]>

    The longitude can be outside the [0, 360) deg range. This is wrapped to the
    [0, 360) deg range (actually the base constructor does this):

    >>> vec = cxv.LonLatSphericalPos.from_(lon=u.Quantity(365, "deg"),
    ...                                    lat=u.Quantity(0, "deg"),
    ...                                    distance=u.Quantity(3, "km"))
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
    r: u.AbstractQuantity,
    theta: u.AbstractQuantity,
    phi: u.AbstractQuantity,
) -> MathSphericalPos:
    """Construct MathSphericalPos, allowing for out-of-range values.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    Let's start with a valid input:

    >>> vec = cxv.MathSphericalPos.from_(r=u.Quantity(3, "km"),
    ...                                  theta=u.Quantity(90, "deg"),
    ...                                  phi=u.Quantity(0, "deg"))
    >>> print(vec)
    <MathSphericalPos: (r[km], theta[deg], phi[deg])
        [ 3 90  0]>

    The radial distance can be negative, which wraps the azimuthal angle by 180
    degrees and flips the polar angle:

    >>> vec = cxv.MathSphericalPos.from_(r=u.Quantity(-3, "km"),
    ...                                  theta=u.Quantity(100, "deg"),
    ...                                  phi=u.Quantity(45, "deg"))
    >>> print(vec)
    <MathSphericalPos: (r[km], theta[deg], phi[deg])
        [  3 280 135]>

    The polar angle can be outside the [0, 180] deg range, causing the azimuthal
    angle to be shifted by 180 degrees:

    >>> vec = cxv.MathSphericalPos.from_(r=u.Quantity(3, "km"),
    ...                                  theta=u.Quantity(0, "deg"),
    ...                                  phi=u.Quantity(190, "deg"))
    >>> print(vec)
    <MathSphericalPos: (r[km], theta[deg], phi[deg])
        [  3 180 170]>

    The azimuth can be outside the [0, 360) deg range. This is wrapped to the
    [0, 360) deg range (actually the base constructor does this):

    >>> vec = cxv.MathSphericalPos.from_(r=u.Quantity(3, "km"),
    ...                                  theta=u.Quantity(365, "deg"),
    ...                                  phi=u.Quantity(90, "deg"))
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
# Vector Transformation

# =============================================================================
# `vconvert_impl`


@dispatch
def vconvert(
    to_vector: type[AbstractPos3D],
    from_vector: type[AbstractPos3D],
    params: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """AbstractPos -> CartesianPos3D -> AbstractPos."""
    params, aux = vconvert(
        CartesianPos3D, from_vector, params, in_aux=in_aux, out_aux=None, units=units
    )
    params, aux = vconvert(
        to_vector, CartesianPos3D, params, in_aux=aux, out_aux=out_aux, units=units
    )
    return params, aux


@dispatch
@ft.partial(jax.jit, static_argnums=(0, 1), static_argnames=("units",))
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[CylindricalPos],
    from_vector: type[CartesianPos3D],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """CartesianPos3D -> CylindricalPos.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> cart = {"x": 1, "y": 2, "z": 3}
    >>> cxv.vconvert(cxv.CylindricalPos, cxv.CartesianPos3D, cart)
    ({'phi': Array(1.1071488, dtype=float32, ...),
      'rho': Array(2.236068, dtype=float32, ...),
      'z': Array(3, dtype=int32, ...)},
     {})

    >>> cart = {"x": u.Quantity(1, "km"), "y": u.Quantity(2, "km"),
    ...         "z": u.Quantity(3, "km")}
    >>> cxv.vconvert(cxv.CylindricalPos, cxv.CartesianPos3D, cart)
    ({'phi': Quantity(Array(1.1071488, dtype=float32, ...), unit='rad'),
      'rho': Quantity(Array(2.236068, dtype=float32, ...), unit='km'),
      'z': Quantity(Array(3, dtype=int32, ...), unit='km')},
     {})

    """
    rho = jnp.hypot(p["x"], p["y"])
    phi = jnp.atan2(p["y"], p["x"])
    return {"rho": rho, "phi": phi, "z": p["z"]}, combine_aux(in_aux, out_aux)


@dispatch
@ft.partial(jax.jit, static_argnums=(0, 1), static_argnames=("units",))
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[SphericalPos],
    from_vector: type[CartesianPos3D],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """CartesianPos3D -> SphericalPos.

    Examples
    --------
    >>> import coordinax.vecs as cxv
    >>> cart = {"x": 1, "y": 2, "z": 3}
    >>> cxv.vconvert(cxv.SphericalPos, cxv.CartesianPos3D, cart)
    ({'phi': Array(1.1071488, dtype=float32, ...),
      'r': Array(3.7416575, dtype=float32, ...),
      'theta': Array(0.64052236, dtype=float32)},
     {})

    The origin is a special case, where the angles are set to 0 by convention:

    >>> cart = {"x": 0, "y": 0, "z": 0}
    >>> cxv.vconvert(cxv.SphericalPos, cxv.CartesianPos3D, cart)
    ({'phi': Array(0., dtype=float32, ...),
      'r': Array(0., dtype=float32, ...),
      'theta': Array(0., dtype=float32)},
     {})

    """
    del units  # unused
    r = jnp.sqrt(p["x"] ** 2 + p["y"] ** 2 + p["z"] ** 2)
    # Avoid division by zero: when r == 0, set theta = 0 by convention
    theta = jnp.acos(jnp.where(r == 0, jnp.ones(r.shape), p["z"] / r))
    # atan2 handles the case when x = y = 0, returning phi = 0
    phi = jnp.atan2(p["y"], p["x"])
    return {"r": r, "theta": theta, "phi": phi}, combine_aux(in_aux, out_aux)


@dispatch.multi(
    (type[LonLatSphericalPos], type[CartesianPos3D], ct.ParamsDict),
    (type[MathSphericalPos], type[CartesianPos3D], ct.ParamsDict),
)
def vconvert(
    to_vector: type[AbstractSphericalPos],
    from_vector: type[CartesianPos3D],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """CartesianPos3D -> AbstractSphericalPos.

    Examples
    --------
    >>> import coordinax.vecs as cxv
    >>> params = {"x": 1, "y": 2, "z": 3}
    >>> usys = u.unitsystem("km", "deg")

    >>> cxv.vconvert(cxv.LonLatSphericalPos, cxv.CartesianPos3D,
    ...              params, units=usys)
    ({'distance': Array(3.7416575, dtype=float32, ...),
      'lat': Array(53.300774, dtype=float32),
      'lon': Array(63.43495, dtype=float32, ...)},
     {})

    >>> cxv.vconvert(cxv.MathSphericalPos, cxv.CartesianPos3D,
    ...              params, units=usys)
    ({'phi': Array(36.69923, dtype=float32),
      'r': Array(3.7416575, dtype=float32, ...),
      'theta': Array(63.43495, dtype=float32, ...)},
     {})

    """
    p, aux = vconvert(
        SphericalPos, from_vector, p, in_aux=in_aux, out_aux=None, units=units
    )
    p, aux = vconvert(
        to_vector, SphericalPos, p, in_aux=aux, out_aux=out_aux, units=units
    )
    return p, aux


@dispatch
@ft.partial(jax.jit, static_argnums=(0, 1), static_argnames=("units",))
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[CartesianPos3D],
    from_vector: type[CylindricalPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """CylindricalPos -> CartesianPos3D.

    Examples
    --------
    >>> import coordinax.vecs as cxv
    >>> cyl = {"rho": 1, "phi": 90, "z": 1}
    >>> usys = u.unitsystem("km", "deg")

    >>> cxv.vconvert(cxv.CartesianPos3D, cxv.CylindricalPos, cyl, units=usys)
    ({'x': Array(-4.371139e-08, dtype=float32, ...),
      'y': Array(1., dtype=float32, ...),
      'z': Array(1, dtype=int32, ...)},
     {})

    """
    x = p["rho"] * jnp.cos(p["phi"])
    y = p["rho"] * jnp.sin(p["phi"])
    return {"x": x, "y": y, "z": p["z"]}, combine_aux(in_aux, out_aux)


@dispatch
@ft.partial(jax.jit, static_argnums=(0, 1), static_argnames=("units",))
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[SphericalPos],
    from_vector: type[CylindricalPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """CylindricalPos -> SphericalPos.

    Examples
    --------
    >>> import coordinax.vecs as cxv
    >>> cyl = {"rho": 1, "phi": 90, "z": 1}
    >>> usys = u.unitsystem("km", "deg")

    >>> cxv.vconvert(cxv.SphericalPos, cxv.CylindricalPos, cyl, units=usys)
    ({'phi': Array(90., dtype=float32, ...),
      'r': Array(1.4142135, dtype=float32, ...),
      'theta': Array(45..., dtype=float32)},
     {})

    The origin is a special case, where the angles are set to 0 by convention:

    >>> cyl = {"rho": 0, "phi": 0, "z": 0}
    >>> cxv.vconvert(cxv.SphericalPos, cxv.CylindricalPos, cyl, units=usys)
    ({'phi': Array(0., dtype=float32, ...),
      'r': Array(0., dtype=float32, ...),
      'theta': Array(0., dtype=float32)},
     {})

    """
    r = jnp.hypot(p["rho"], p["z"])
    # Avoid division by zero: when r == 0, set theta = 0 by convention
    theta = jnp.acos(jnp.where(r == 0, jnp.ones(r.shape), p["z"] / r))
    return {"r": r, "theta": theta, "phi": p["phi"]}, combine_aux(in_aux, out_aux)


@dispatch.multi(
    (type[LonLatSphericalPos], type[CylindricalPos], ct.ParamsDict),
    (type[MathSphericalPos], type[CylindricalPos], ct.ParamsDict),
)
@ft.partial(jax.jit, static_argnums=(0, 1), static_argnames=("units",))
def vconvert(
    to_vector: type[AbstractSphericalPos],
    from_vector: type[CylindricalPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """CylindricalPos -> AbstractSphericalPos.

    Examples
    --------
    >>> import coordinax.vecs as cxv
    >>> cyl = {"rho": 1, "phi": 90, "z": 1}
    >>> usys = u.unitsystem("km", "deg")

    >>> cxv.vconvert(cxv.LonLatSphericalPos, cxv.CylindricalPos, cyl, units=usys)
    ({'distance': Array(1.4142135, dtype=float32, ...),
      'lat': Array(45., dtype=float32),
      'lon': Array(90., dtype=float32, ...)},
     {})

    >>> cxv.vconvert(cxv.MathSphericalPos, cxv.CylindricalPos, cyl, units=usys)
    ({'phi': Array(45..., dtype=float32),
      'r': Array(1.4142135, dtype=float32, ...),
      'theta': Array(90., dtype=float32, ...)},
     {})

    """
    params, aux = vconvert(
        SphericalPos, from_vector, p, in_aux=in_aux, out_aux=out_aux, units=units
    )
    params, aux = vconvert(
        to_vector, SphericalPos, params, in_aux=aux, out_aux=out_aux, units=units
    )
    return params, aux


@dispatch
@ft.partial(jax.jit, static_argnums=(0, 1), static_argnames=("units",))
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[CartesianPos3D],
    from_vector: type[SphericalPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """CylindricalPos -> CartesianPos3D.

    Examples
    --------
    >>> import coordinax.vecs as cxv
    >>> sph = {"r": 1, "theta": 90, "phi": 90}
    >>> usys = u.unitsystem("km", "deg")

    >>> cxv.vconvert(cxv.CartesianPos3D, cxv.SphericalPos, sph, units=usys)
    ({'x': Array(-4.371139e-08, dtype=float32, ...),
      'y': Array(1., dtype=float32, ...),
      'z': Array(-4.371139e-08, dtype=float32, ...)},
     {})

    """
    r, theta, phi = p["r"], p["theta"], p["phi"]
    x = r * jnp.sin(theta) * jnp.cos(phi)
    y = r * jnp.sin(theta) * jnp.sin(phi)
    z = r * jnp.cos(theta)
    return {"x": x, "y": y, "z": z}, combine_aux(in_aux, out_aux)


@dispatch
@ft.partial(jax.jit, static_argnums=(0, 1), static_argnames=("units",))
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[CylindricalPos],
    from_vector: type[SphericalPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """SphericalPos -> CylindricalPos.

    Examples
    --------
    >>> import coordinax.vecs as cxv
    >>> sph = {"r": 1, "theta": 90, "phi": 90}
    >>> usys = u.unitsystem("km", "deg")

    >>> cxv.vconvert(cxv.CylindricalPos, cxv.SphericalPos, sph, units=usys)
    ({'phi': Array(90., dtype=float32, ...),
      'rho': Array(1., dtype=float32, ...),
      'z': Array(-4.371139e-08, dtype=float32, ...)},
     {})

    """
    rho = jnp.abs(p["r"]) * jnp.sin(p["theta"])
    phi = p["phi"]
    z = p["r"] * jnp.cos(p["theta"])
    return {"rho": rho, "phi": phi, "z": z}, combine_aux(in_aux, out_aux)


@dispatch
@ft.partial(jax.jit, static_argnums=(0, 1), static_argnames=("units",))
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[LonLatSphericalPos],
    from_vector: type[SphericalPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """SphericalPos -> LonLatSphericalPos.

    Examples
    --------
    >>> import coordinax.vecs as cxv
    >>> sph = {"r": 1, "theta": 90, "phi": 90}
    >>> usys = u.unitsystem("km", "deg")

    >>> cxv.vconvert(cxv.LonLatSphericalPos, cxv.SphericalPos, sph, units=usys)
    ({'distance': Array(1, dtype=int32, ...),
      'lat': Array(3.2016512e-06, dtype=float32, ...),
      'lon': Array(90., dtype=float32, ...)},
     {})

    """
    if isinstance(p["theta"], u.AbstractQuantity):
        lat = u.Quantity(90, "deg") - p["theta"]
    else:
        lat = jnp.pi / 2 - p["theta"]

    aux = combine_aux(in_aux, out_aux)
    return {"lon": p["phi"], "lat": lat, "distance": p["r"]}, aux


@dispatch.multi(
    (type[LonLatSphericalVel], type[SphericalVel], ct.ParamsDict),
    (type[LonLatSphericalAcc], type[SphericalAcc], ct.ParamsDict),
)
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[AbstractVector],
    from_vector: type[AbstractVector],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """SphericalVel/Acc -> LonLatSphericalVel/Acc.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> p = {"r": u.Quantity(1, "km/s"),
    ...      "theta": u.Quantity(10, "deg/s"),
    ...      "phi": u.Quantity(20, "deg/s")}

    >>> cxv.vconvert(cxv.LonLatSphericalVel, cxv.SphericalVel, p)
    ({'distance': Quantity(Array(1, dtype=int32, ...), unit='km / s'),
      'lat': Quantity(Array(-10, dtype=int32, ...), unit='deg / s'),
      'lon': Quantity(Array(20, dtype=int32, ...), unit='deg / s')},
     {})

    >>> x = cxv.SphericalVel(**p)
    >>> y = cxv.vconvert(cxv.LonLatSphericalVel, x)
    >>> print(y)
    <LonLatSphericalVel: (lon[deg / s], lat[deg / s], distance[km / s])
        [ 20 -10   1]>

    >>> p = {"r": u.Quantity(1, "km/s2"),
    ...      "theta": u.Quantity(10, "deg/s2"),
    ...      "phi": u.Quantity(20, "deg/s2")}

    >>> cxv.vconvert(cxv.LonLatSphericalAcc, cxv.SphericalAcc, p)
    ({'distance': Quantity(Array(1, dtype=int32, ...), unit='km / s2'),
      'lat': Quantity(Array(-10, dtype=int32, ...), unit='deg / s2'),
      'lon': Quantity(Array(20, dtype=int32, ...), unit='deg / s2')},
     {})

    >>> x = cxv.SphericalAcc(**p)
    >>> y = cxv.vconvert(cxv.LonLatSphericalAcc, x)
    >>> print(y)
    <LonLatSphericalAcc: (lon[deg / s2], lat[deg / s2], distance[km / s2])
        [ 20 -10   1]>

    """
    del to_vector, from_vector, in_aux, units
    newp = {"distance": p["r"], "lat": -p["theta"], "lon": p["phi"]}
    return newp, (out_aux or {})


@dispatch.multi(
    (type[MathSphericalPos], type[SphericalPos], ct.ParamsDict),
    (type[SphericalPos], type[MathSphericalPos], ct.ParamsDict),
    (type[MathSphericalVel], type[SphericalVel], ct.ParamsDict),
    (type[SphericalVel], type[MathSphericalVel], ct.ParamsDict),
    (type[MathSphericalAcc], type[SphericalAcc], ct.ParamsDict),
    (type[SphericalAcc], type[MathSphericalAcc], ct.ParamsDict),
)
@ft.partial(jax.jit, static_argnums=(0, 1), static_argnames=("units",))
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[AbstractVector],
    from_vector: type[AbstractVector],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """SphericalPos <-> MathSphericalPos.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    Position:

    >>> p = {"r": 1, "theta": 90, "phi": 90}
    >>> usys = u.unitsystem("km", "deg")

    >>> cxv.vconvert(cxv.MathSphericalPos, cxv.SphericalPos, p, units=usys)
    ({'phi': Array(90., dtype=float32, ...),
      'r': Array(1, dtype=int32, ...),
      'theta': Array(90., dtype=float32, ...)},
     {})

    >>> cxv.vconvert(cxv.SphericalPos, cxv.MathSphericalPos, p, units=usys)
    ({'phi': Array(90., dtype=float32, ...),
      'r': Array(1, dtype=int32, ...),
      'theta': Array(90., dtype=float32, ...)},
     {})

    Velocity:

    >>> p = {"r": u.Quantity(1, "km/s"),
    ...      "theta": u.Quantity(10, "deg/s"),
    ...      "phi": u.Quantity(20, "deg/s")}

    >>> p, aux = cxv.vconvert(cxv.MathSphericalVel, cxv.SphericalVel, p)
    >>> p, aux
    ({'phi': Quantity(Array(10, dtype=int32, ...), unit='deg / s'),
      'r': Quantity(Array(1, dtype=int32, ...), unit='km / s'),
      'theta': Quantity(Array(20, dtype=int32, ...), unit='deg / s')}, {})

    >>> cxv.vconvert(cxv.SphericalVel, cxv.MathSphericalVel, p)
    ({'phi': Quantity(Array(20, dtype=int32, ...), unit='deg / s'),
      'r': Quantity(Array(1, dtype=int32, ...), unit='km / s'),
      'theta': Quantity(Array(10, dtype=int32, ...), unit='deg / s')},
     {})

    >>> x = cxv.SphericalVel(r=u.Quantity(1, "km/s"),
    ...                      theta=u.Quantity(10, "deg/s"),
    ...                      phi=u.Quantity(20, "deg/s"))
    >>> y = cxv.vconvert(cxv.MathSphericalVel, x)
    >>> print(y)
    <MathSphericalVel: (r[km / s], theta[deg / s], phi[deg / s])
        [ 1 20 10]>

    >>> x = cxv.vconvert(cxv.SphericalVel, y)
    >>> print(x)
    <SphericalVel: (r[km / s], theta[deg / s], phi[deg / s])
        [ 1 10 20]>

    Acceleration:

    >>> p = {"r": u.Quantity(1, "km/s2"),
    ...      "theta": u.Quantity(10, "deg/s2"),
    ...      "phi": u.Quantity(20, "deg/s2")}

    >>> p, aux = cxv.vconvert(cxv.MathSphericalAcc, cxv.SphericalAcc, p)
    >>> p, aux
    ({'phi': Quantity(Array(10, dtype=int32, ...), unit='deg / s2'),
      'r': Quantity(Array(1, dtype=int32, ...), unit='km / s2'),
      'theta': Quantity(Array(20, dtype=int32, ...), unit='deg / s2')},
     {})

    >>> cxv.vconvert(cxv.SphericalAcc, cxv.MathSphericalAcc, p)
    ({'phi': Quantity(Array(20, dtype=int32, ...), unit='deg / s2'),
      'r': Quantity(Array(1, dtype=int32, ...), unit='km / s2'),
      'theta': Quantity(Array(10, dtype=int32, ...), unit='deg / s2')},
     {})

    >>> x = cxv.SphericalAcc(r=u.Quantity(1, "km/s2"),
    ...                      theta=u.Quantity(10, "deg/s2"),
    ...                      phi=u.Quantity(20, "deg/s2"))
    >>> y = cxv.vconvert(cxv.MathSphericalAcc, x)
    >>> print(y)
    <MathSphericalAcc: (r[km / s2], theta[deg / s2], phi[deg / s2])
        [ 1 20 10]>

    >>> x = cxv.vconvert(cxv.SphericalAcc, y)
    >>> print(x)
    <SphericalAcc: (r[km / s2], theta[deg / s2], phi[deg / s2])
        [ 1 10 20]>

    """
    outp = {"r": p["r"], "theta": p["phi"], "phi": p["theta"]}
    return outp, combine_aux(in_aux, out_aux)


@dispatch
@ft.partial(jax.jit, static_argnums=(0, 1), static_argnames=("units",))
def vconvert(
    to_vector: type[CartesianPos3D],
    from_vector: type[LonLatSphericalPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """LonLatSphericalPos -> CartesianPos3D.

    Examples
    --------
    >>> import coordinax.vecs as cxv
    >>> vec = {"lon": 90, "lat": 0, "distance": 1}
    >>> usys = u.unitsystem("km", "deg")

    >>> cxv.vconvert(cxv.CartesianPos3D, cxv.LonLatSphericalPos, vec, units=usys)
    ({'x': Array(-4.371139e-08, dtype=float32, ...),
      'y': Array(1., dtype=float32, ...),
      'z': Array(-4.371139e-08, dtype=float32, ...)},
     {})

    """
    p, aux = vconvert(
        SphericalPos, from_vector, p, in_aux=in_aux, out_aux=out_aux, units=units
    )
    p, aux = vconvert(
        to_vector, SphericalPos, p, in_aux=aux, out_aux=out_aux, units=units
    )
    return p, aux


@dispatch
@ft.partial(jax.jit, static_argnums=(0, 1), static_argnames=("units",))
def vconvert(
    to_vector: type[CylindricalPos],
    from_vector: type[LonLatSphericalPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """LonLatSphericalPos -> CylindricalPos.

    Examples
    --------
    >>> import coordinax.vecs as cxv
    >>> sph = {"lon": 90, "lat": 0, "distance": 1}
    >>> usys = u.unitsystem("km", "deg")

    >>> cxv.vconvert(cxv.CylindricalPos, cxv.LonLatSphericalPos, sph, units=usys)
    ({'phi': Array(90., dtype=float32, ...),
      'rho': Array(1., dtype=float32, ...),
      'z': Array(-4.371139e-08, dtype=float32, ...)},
     {})

    """
    p, aux = vconvert(
        SphericalPos, from_vector, p, in_aux=in_aux, out_aux=None, units=units
    )
    p, aux = vconvert(
        to_vector, SphericalPos, p, in_aux=aux, out_aux=out_aux, units=units
    )
    return p, aux


@dispatch
@ft.partial(jax.jit, static_argnums=(0, 1), static_argnames=("units",))
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[SphericalPos],
    from_vector: type[LonLatSphericalPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """LonLatSphericalPos -> SphericalPos.

    Examples
    --------
    >>> import coordinax.vecs as cxv
    >>> vec = {"lon": 90, "lat": 0, "distance": 1}
    >>> usys = u.unitsystem("km", "deg")

    >>> cxv.vconvert(cxv.SphericalPos, cxv.LonLatSphericalPos, vec, units=usys)
    ({'phi': Array(90., dtype=float32, ...,
      'r': Array(1, dtype=int32, ...,
      'theta': Array(90., dtype=float32, ...},
     {})

    """
    if isinstance(p["lat"], u.AbstractQuantity):
        theta = u.Quantity(90, "deg") - p["lat"]
    else:
        theta = jnp.pi / 2 - p["lat"]

    outp = {"r": p["distance"], "theta": theta, "phi": p["lon"]}
    return outp, combine_aux(in_aux, out_aux)


@dispatch.multi(
    (type[SphericalVel], type[LonLatSphericalVel], ct.ParamsDict),
    (type[SphericalAcc], type[LonLatSphericalAcc], ct.ParamsDict),
)
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[AbstractVector],
    from_vector: type[AbstractVector],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """LonLatSphericalVel -> SphericalVel.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> p = {"lon": u.Quantity(90, "deg/s"),
    ...      "lat": u.Quantity(0, "deg/s"),
    ...      "distance": u.Quantity(1, "km/s")}
    >>> cxv.vconvert(cxv.SphericalVel, cxv.LonLatSphericalVel, p)
    ({'r': Quantity(Array(1, dtype=int32, ...), unit='km / s'),
      'theta': Quantity(Array(0, dtype=int32, ...), unit='deg / s'),
      'phi': Quantity(Array(90, dtype=int32, ...), unit='deg / s')},
     {})

    >>> x = cxv.LonLatSphericalVel(**p)
    >>> y = cxv.vconvert(cxv.SphericalVel, x)
    >>> print(y)
    <SphericalVel: (r[km / s], theta[deg / s], phi[deg / s])
        [ 1  0 90]>

    >>> p = {"lon": u.Quantity(90, "deg/s2"),
    ...      "lat": u.Quantity(0, "deg/s2"),
    ...      "distance": u.Quantity(1, "km/s2")}
    >>> cxv.vconvert(cxv.SphericalAcc, cxv.LonLatSphericalAcc, p)
    ({'r': Quantity(Array(1, dtype=int32, ...), unit='km / s2'),
      'theta': Quantity(Array(0, dtype=int32, ...), unit='deg / s2'),
      'phi': Quantity(Array(90, dtype=int32, ...), unit='deg / s2')},
     {})

    >>> x = cxv.LonLatSphericalAcc(**p)
    >>> y = cxv.vconvert(cxv.SphericalAcc, x)
    >>> print(y)
    <SphericalAcc: (r[km / s2], theta[deg / s2], phi[deg / s2])
        [ 1  0 90]>

    """
    newp = {"r": p["distance"], "theta": -p["lat"], "phi": p["lon"]}
    return newp, (out_aux or {})


@dispatch
@ft.partial(jax.jit, static_argnums=(0, 1), static_argnames=("units",))
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[CartesianPos3D],
    from_vector: type[MathSphericalPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """MathSphericalPos -> CartesianPos3D.

    Examples
    --------
    >>> import coordinax.vecs as cxv
    >>> vec = {"r": 1, "theta": 90, "phi": 90}
    >>> usys = u.unitsystem("km", "deg")

    >>> cxv.vconvert(cxv.CartesianPos3D, cxv.MathSphericalPos, vec, units=usys)
    ({'x': Array(-4.371139e-08, dtype=float32, ...),
      'y': Array(1., dtype=float32, ...),
      'z': Array(-4.371139e-08, dtype=float32, ...)},
     {})

    """
    x = p["r"] * jnp.sin(p["phi"]) * jnp.cos(p["theta"])
    y = p["r"] * jnp.sin(p["phi"]) * jnp.sin(p["theta"])
    z = p["r"] * jnp.cos(p["phi"])
    return {"x": x, "y": y, "z": z}, combine_aux(in_aux, out_aux)


@dispatch
@ft.partial(jax.jit, static_argnums=(0, 1), static_argnames=("units",))
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[CylindricalPos],
    from_vector: type[MathSphericalPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """MathSphericalPos -> CylindricalPos.

    Examples
    --------
    >>> import coordinax.vecs as cxv
    >>> vec = {"r": 1, "theta": 90, "phi": 90}
    >>> usys = u.unitsystem("km", "deg")

    >>> cxv.vconvert(cxv.CylindricalPos, cxv.MathSphericalPos, vec, units=usys)
    ({'phi': Array(90., dtype=float32, ...),
      'rho': Array(1., dtype=float32, ...),
      'z': Array(-4.371139e-08, dtype=float32, ...)},
     {})

    """
    rho = jnp.abs(p["r"]) * jnp.sin(p["phi"])
    z = p["r"] * jnp.cos(p["phi"])
    return {"rho": rho, "phi": p["theta"], "z": z}, combine_aux(in_aux, out_aux)


@dispatch
@ft.partial(jax.jit, static_argnums=(0, 1), static_argnames=("units",))
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[CylindricalPos],
    from_vector: type[ProlateSpheroidalPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """ProlateSpheroidalPos -> CylindricalPos.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv
    >>> vec = {"mu": 1, "nu": 0.2, "phi": 90}
    >>> in_aux = {"Delta": 0.5}
    >>> usys = u.unitsystem("km", "deg")

    >>> cxv.vconvert(cxv.CylindricalPos, cxv.ProlateSpheroidalPos,
    ...                   vec, in_aux=in_aux, units=usys)
    ({'phi': Array(90., dtype=float32, ...),
      'rho': Array(0.38729832, dtype=float32, ...),
      'z': Array(0.8944272, dtype=float32, ...)},
     {})

    TODO: example with Delta as a Quantity

    """
    Delta2 = in_aux["Delta"] ** 2
    nu_D2 = jnp.abs(p["nu"]) / Delta2
    rho = jnp.sqrt((p["mu"] - Delta2) * (1 - nu_D2))
    z = jnp.sqrt(p["mu"] * nu_D2) * jnp.sign(p["nu"])
    return {"rho": rho, "phi": p["phi"], "z": z}, {}


@dispatch
@ft.partial(jax.jit, static_argnums=(0, 1), static_argnames=("units",))
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[ProlateSpheroidalPos],
    from_vector: type[CylindricalPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """CylindricalPos -> ProlateSpheroidalPos.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> vec = {"rho": 1, "phi": 90, "z": 1}
    >>> out_aux = {"Delta": 0.5}
    >>> usys = u.unitsystem("km", "deg")

    >>> cxv.vconvert(cxv.ProlateSpheroidalPos, cxv.CylindricalPos,
    ...                   vec, out_aux=out_aux, units=usys)
    ({'mu': Array(2.1327822, dtype=float32, ...),
      'nu': Array(0.11721778, dtype=float32, ...),
      'phi': Array(90., dtype=float32, ...)},
     {'Delta': Array(0.5, dtype=float32, ...)})

    # <ProlateSpheroidalPos: (mu[km2], nu[km2], phi[deg])
    #     [ 2.133  0.117 90.   ]>

    TODO: example with Delta as a Quantity

    """
    Delta = eqx.error_if(
        out_aux.get("Delta"),
        "Delta" not in out_aux,
        "Delta must be provided for ProlateSpheroidalPos.",
    )
    R2 = p["rho"] ** 2
    z2 = p["z"] ** 2
    Delta2 = Delta**2

    sum_ = R2 + z2 + Delta2
    diff_ = R2 + z2 - Delta2

    # compute D = sqrt((R² + z² - Δ²)² + 4R²Δ²)
    D = jnp.sqrt(diff_**2 + 4 * R2 * Delta2)

    # handle special cases for R=0 or z=0
    D = jnp.where(p["z"] == 0, sum_, D)
    D = jnp.where(p["rho"] == 0, jnp.abs(diff_), D)

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

    nu = abs_nu * jnp.sign(p["z"])

    return {"mu": mu, "nu": nu, "phi": p["phi"]}, out_aux


@dispatch
@ft.partial(jax.jit, static_argnums=(0, 1), static_argnames=("units",))
def vconvert(
    to_vector: type[ProlateSpheroidalPos],
    from_vector: type[ProlateSpheroidalPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.AuxDict,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """ProlateSpheroidalPos -> ProlateSpheroidalPos.

    Examples
    --------
    >>> import coordinax.vecs as cxv
    >>> vec = {"mu": 1, "nu": 0.2, "phi": 90}
    >>> in_aux = {"Delta": 0.5}
    >>> usys = u.unitsystem("km", "deg")

    Self-transforms can change the focal length:

    >>> out_aux = {"Delta": 0.8}
    >>> cxv.vconvert(cxv.ProlateSpheroidalPos, cxv.ProlateSpheroidalPos,
    ...                   vec, in_aux=in_aux, out_aux=out_aux, units=usys)
    ({'mu': Array(1.1414464, dtype=float32, ...),
      'nu': Array(0.44855377, dtype=float32, ...),
      'phi': Array(90., dtype=float32, ...)},
     {'Delta': Array(0.8, dtype=float32, ...)})

    Without changing the focal length, no transform is done:

    >>> cxv.vconvert(cxv.ProlateSpheroidalPos, cxv.ProlateSpheroidalPos,
    ...                   vec, in_aux=in_aux, units=usys)
    ({'mu': Array(1, dtype=int32, ...),
      'nu': Array(0.2, dtype=float32, ...),
      'phi': Array(90, dtype=int32, ...)},
     {'Delta': Array(0.5, dtype=float32, ...)})

    """
    out_aux = out_aux or {}
    if out_aux.get("Delta", None) is None:
        return p, combine_aux(in_aux, out_aux)

    # If Delta is provided, we can proceed with the conversion
    p, aux = vconvert(
        CylindricalPos, from_vector, p, in_aux=in_aux, out_aux=None, units=units
    )
    p, aux = vconvert(
        to_vector, CylindricalPos, p, in_aux=aux, out_aux=out_aux, units=units
    )
    return p, aux


@dispatch
@ft.partial(jax.jit, static_argnums=(0, 1), static_argnames=("units",))
def vconvert(
    to_vector: type[ProlateSpheroidalPos],
    from_vector: type[AbstractPos3D,],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.AuxDict,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """AbstractPos3D -> ProlateSpheroidalPos.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> vec = {"x": 1, "y": 2, "z": 3}
    >>> out_aux = {"Delta": 0.5}
    >>> usys = u.unitsystem("km", "deg")

    >>> cxv.vconvert(cxv.ProlateSpheroidalPos, cxv.CartesianPos3D,
    ...                   vec, out_aux=out_aux, units=usys)
    ({'mu': Array(14.090316, dtype=float32, ...),
      'nu': Array(0.15968415, dtype=float32, ...),
      'phi': Array(63.43495, dtype=float32, ...)},
     {'Delta': Array(0.5, dtype=float32, ...)})

    """
    p, aux = vconvert(
        CylindricalPos, from_vector, p, in_aux=in_aux, out_aux=None, units=units
    )
    p, aux = vconvert(
        to_vector, CylindricalPos, p, in_aux=aux, out_aux=out_aux, units=units
    )
    return p, aux


# =============================================================================
# LonLatSphericalVel


@dispatch
def vconvert(
    target: type[LonCosLatSphericalVel],
    current: AbstractVel3D,
    position: AbstractPos,
    /,
    **kwargs: Any,
) -> LonCosLatSphericalVel:
    """AbstractVel3D -> LonCosLatSphericalVel.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> q = cxv.LonLatSphericalPos(lon=u.Quantity(15, "deg"),
    ...                            lat=u.Quantity(10, "deg"),
    ...                            distance=u.Quantity(1.5, "km"))
    >>> p = cxv.LonLatSphericalVel(lon=u.Quantity(7, "mas/yr"),
    ...                            lat=u.Quantity(0, "deg/Gyr"),
    ...                            distance=u.Quantity(-5, "km/s"))
    >>> newp = cxv.vconvert(cxv.LonCosLatSphericalVel, p, q)
    >>> print(newp)
    <LonCosLatSphericalVel: (lon_coslat[mas / yr], lat[deg / Gyr], distance[km / s])
        [ 6.894  0.    -5.   ]>

    """
    del kwargs  # unused

    # Transform the differential to LonLatSphericalVel
    current = vconvert(LonLatSphericalVel, current, position)

    # Transform the position to the required type
    position = vconvert(current.time_antiderivative_cls, position)

    # Calculate the differential in the new system
    return target(
        lon_coslat=current.lon * jnp.cos(position.lat),
        lat=current.lat,
        distance=current.distance,
    )


@dispatch
def vconvert(
    target: type[LonLatSphericalVel],
    current: LonCosLatSphericalVel,
    position: AbstractPos,
    /,
    **kwargs: Any,
) -> LonLatSphericalVel:
    """LonCosLatSphericalVel -> LonLatSphericalVel."""
    del kwargs  # unused
    # Transform the position to the required type
    position = vconvert(current.time_antiderivative_cls, position)
    # Calculate the differential in the new system
    return target(
        lon=current.lon_coslat / jnp.cos(position.lat),
        lat=current.lat,
        distance=current.distance,
    )


@dispatch
def vconvert(
    target: type[AbstractVel3D],
    current: LonCosLatSphericalVel,
    position: AbstractPos,
    /,
    **kwargs: Any,
) -> AbstractVel3D:
    """LonCosLatSphericalVel -> AbstractVel3D."""
    del kwargs  # unused
    # Transform the differential to LonLatSphericalVel
    current = vconvert(LonLatSphericalVel, current, position)
    # Transform the position to the required type
    return vconvert(target, current, position)


#####################################################################


# from coordinax.vectors.funcs
@dispatch
@ft.partial(eqx.filter_jit, inline=True)
def normalize_vector(obj: CartesianPos3D, /) -> Cartesian3D:
    """Return the norm of the vector.

    This has length 1.

    .. note::

        The unit vector is dimensionless, even if the input vector has units.
        This is because the unit vector is a ratio of two quantities: each
        component and the norm of the vector.

    Returns
    -------
    Cartesian3D
        The norm of the vector.

    Examples
    --------
    >>> import coordinax.vecs as cxv
    >>> q = cxv.CartesianPos3D.from_([1, 2, 3], "km")
    >>> print(cxv.normalize_vector(q))
    <Cartesian3D: (x, y, z) []
        [0.267 0.535 0.802]>

    """
    norm: u.AbstractQuantity = obj.norm()  # type: ignore[misc]
    return Cartesian3D(x=obj.x / norm, y=obj.y / norm, z=obj.z / norm)


###############################################################################
# Corresponding Cartesian classes


@dispatch
def cartesian_vector_type(
    obj: type[AbstractPos3D] | AbstractPos3D, /
) -> type[CartesianPos3D]:
    """Return the corresponding Cartesian class."""
    return CartesianPos3D


@dispatch
def cartesian_vector_type(
    obj: type[AbstractVel3D] | AbstractVel3D, /
) -> type[CartesianVel3D]:
    """Return the corresponding Cartesian class."""
    return CartesianVel3D


@dispatch
def cartesian_vector_type(
    obj: type[AbstractAcc3D] | AbstractAcc3D, /
) -> type[CartesianAcc3D]:
    """Return the corresponding Cartesian class."""
    return CartesianAcc3D


###############################################################################
# Corresponding time derivative classes

# -----------------------------------------------
# Position -> Velocity


@dispatch
def time_derivative_vector_type(
    obj: type[CartesianPos3D] | CartesianPos3D, /
) -> type[CartesianVel3D]:
    """Return the corresponding time derivative class."""
    return CartesianVel3D


@dispatch
def time_derivative_vector_type(
    obj: type[CylindricalPos] | CylindricalPos, /
) -> type[CylindricalVel]:
    """Return the corresponding time derivative class."""
    return CylindricalVel


@dispatch
def time_derivative_vector_type(
    obj: type[SphericalPos] | SphericalPos, /
) -> type[SphericalVel]:
    """Return the corresponding time derivative class."""
    return SphericalVel


@dispatch
def time_derivative_vector_type(
    obj: type[MathSphericalPos] | MathSphericalPos, /
) -> type[MathSphericalVel]:
    """Return the corresponding time derivative class."""
    return MathSphericalVel


@dispatch
def time_derivative_vector_type(
    obj: type[LonLatSphericalPos] | LonLatSphericalPos, /
) -> type[LonLatSphericalVel]:
    """Return the corresponding time derivative class."""
    return LonLatSphericalVel


@dispatch
def time_derivative_vector_type(
    obj: type[ProlateSpheroidalPos] | ProlateSpheroidalPos, /
) -> type[ProlateSpheroidalVel]:
    """Return the corresponding time derivative class."""
    return ProlateSpheroidalVel


# -----------------------------------------------
# Velocity -> Position


@dispatch
def time_antiderivative_vector_type(
    obj: type[CartesianVel3D] | CartesianVel3D, /
) -> type[CartesianPos3D]:
    """Return the corresponding time antiderivative class."""
    return CartesianPos3D


@dispatch
def time_antiderivative_vector_type(
    obj: type[CylindricalVel] | CylindricalVel, /
) -> type[CylindricalPos]:
    """Return the corresponding time antiderivative class."""
    return CylindricalPos


@dispatch
def time_antiderivative_vector_type(
    obj: type[SphericalVel] | SphericalVel, /
) -> type[SphericalPos]:
    """Return the corresponding time antiderivative class."""
    return SphericalPos


@dispatch
def time_antiderivative_vector_type(
    obj: type[MathSphericalVel] | MathSphericalVel, /
) -> type[MathSphericalPos]:
    """Return the corresponding time derivative class."""
    return MathSphericalPos


@dispatch
def time_antiderivative_vector_type(
    obj: type[LonLatSphericalVel] | LonLatSphericalVel, /
) -> type[LonLatSphericalPos]:
    """Return the corresponding time antiderivative class."""
    return LonLatSphericalPos


@dispatch
def time_antiderivative_vector_type(
    obj: type[LonCosLatSphericalVel] | LonCosLatSphericalVel, /
) -> type[LonLatSphericalPos]:
    """Return the corresponding time antiderivative class."""
    return LonLatSphericalPos


@dispatch
def time_antiderivative_vector_type(
    obj: type[ProlateSpheroidalVel] | ProlateSpheroidalVel, /
) -> type[ProlateSpheroidalPos]:
    """Return the corresponding time antiderivative class."""
    return ProlateSpheroidalPos


# -----------------------------------------------
# Velocity -> Acceleration


@dispatch
def time_derivative_vector_type(
    obj: type[CartesianVel3D] | CartesianVel3D, /
) -> type[CartesianAcc3D]:
    """Return the corresponding time derivative class."""
    return CartesianAcc3D


@dispatch
def time_derivative_vector_type(
    obj: type[CylindricalVel] | CylindricalVel, /
) -> type[CylindricalAcc]:
    """Return the corresponding time derivative class."""
    return CylindricalAcc


@dispatch
def time_derivative_vector_type(
    obj: type[SphericalVel] | SphericalVel, /
) -> type[SphericalAcc]:
    """Return the corresponding time derivative class."""
    return SphericalAcc


@dispatch
def time_derivative_vector_type(
    obj: type[MathSphericalVel] | MathSphericalVel, /
) -> type[MathSphericalAcc]:
    """Return the corresponding time derivative class."""
    return MathSphericalAcc


@dispatch
def time_derivative_vector_type(
    obj: type[LonLatSphericalVel] | LonLatSphericalVel, /
) -> type[LonLatSphericalAcc]:
    """Return the corresponding time derivative class."""
    return LonLatSphericalAcc


@dispatch
def time_derivative_vector_type(
    obj: type[LonCosLatSphericalVel] | LonCosLatSphericalVel, /
) -> type[LonLatSphericalAcc]:
    """Return the corresponding time derivative class."""
    return LonLatSphericalAcc


@dispatch
def time_derivative_vector_type(
    obj: type[ProlateSpheroidalVel] | ProlateSpheroidalVel, /
) -> type[ProlateSpheroidalAcc]:
    """Return the corresponding time derivative class."""
    return ProlateSpheroidalAcc


# -----------------------------------------------
# Acceleration -> Velocity


@dispatch
def time_antiderivative_vector_type(
    obj: type[CartesianAcc3D] | CartesianAcc3D, /
) -> type[CartesianVel3D]:
    """Return the corresponding time antiderivative class."""
    return CartesianVel3D


@dispatch
def time_antiderivative_vector_type(
    obj: type[CylindricalAcc] | CylindricalAcc, /
) -> type[CylindricalVel]:
    """Return the corresponding time antiderivative class."""
    return CylindricalVel


@dispatch
def time_antiderivative_vector_type(
    obj: type[SphericalAcc] | SphericalAcc, /
) -> type[SphericalVel]:
    """Return the corresponding time antiderivative class."""
    return SphericalVel


@dispatch
def time_antiderivative_vector_type(
    obj: type[MathSphericalAcc] | MathSphericalAcc, /
) -> type[MathSphericalVel]:
    """Return the corresponding time antiderivative class."""
    return MathSphericalVel


@dispatch
def time_antiderivative_vector_type(
    obj: type[LonLatSphericalAcc] | LonLatSphericalAcc, /
) -> type[LonLatSphericalVel]:
    """Return the corresponding time antiderivative class."""
    return LonLatSphericalVel


@dispatch
def time_antiderivative_vector_type(
    obj: type[ProlateSpheroidalAcc] | ProlateSpheroidalAcc, /
) -> type[ProlateSpheroidalVel]:
    """Return the corresponding time antiderivative class."""
    return ProlateSpheroidalVel
