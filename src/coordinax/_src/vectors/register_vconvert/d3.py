"""Transformations between representations."""

__all__: list[str] = []

import functools as ft
from warnings import warn

import jax
from plum import dispatch

import quaxed.numpy as jnp

import coordinax._src.vectors.custom_types as ct
from coordinax._src.vectors import d1, d2, d3
from coordinax._src.vectors.base_pos import AbstractPos
from coordinax._src.vectors.exceptions import IrreversibleDimensionChange
from coordinax._src.vectors.private_api import wrap_vconvert_impl_params

# =============================================================================
# CartesianPos3D


# -----------------------------------------------
# 1D


@dispatch
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[d1.RadialPos],
    from_vector: type[d3.CartesianPos3D],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """CartesianPos3D -> RadialPos.

    Examples
    --------
    >>> import warnings
    >>> import coordinax.vecs as cxv

    >>> params = {"x": 1, "y": 2, "z": 3}

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     cxv.vconvert(cxv.RadialPos, cxv.CartesianPos3D, params)
    ({'r': Array(3.7416575, dtype=float32, ...)}, {})

    >>> x = cxv.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     y = cxv.vconvert(cxv.RadialPos, x)
    >>> print(y)
    <RadialPos: (r) [km]
        [3.742]>

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    r = jnp.sqrt(p["x"] ** 2 + p["y"] ** 2 + p["z"] ** 2)
    return {"r": r}, (out_aux or {})


# -----------------------------------------------
# 2D


@dispatch
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[d2.PolarPos],
    from_vector: type[d3.CartesianPos3D],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """CartesianPos3D -> Cartesian2D -> AbstractPos2D.

    Examples
    --------
    >>> import warnings
    >>> import coordinax as cx

    >>> params = {"x": 1, "y": 2, "z": 3}
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     cx.vecs.vconvert(cx.vecs.PolarPos, cx.vecs.CartesianPos3D, params)
    ({'r': Array(2.236068, dtype=float32, ...),
      'phi': Array(1.1071488, dtype=float32, ...)},
     {})

    >>> x = cx.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     y = cx.vconvert(cx.vecs.PolarPos, x)
    >>> print(y)
    <PolarPos: (r[km], phi[rad])
        [2.236 1.107]>

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    r = jnp.hypot(p["x"], p["y"])
    phi = jnp.arctan2(p["y"], p["x"])
    return {"r": r, "phi": phi}, (out_aux or {})


# =============================================================================
# CylindricalPos


# -----------------------------------------------
# 1D


@dispatch
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[d1.CartesianPos1D],
    from_vector: type[d3.CylindricalPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """CylindricalPos -> CartesianPos1D.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> params = {"rho": 1, "phi": 10, "z": 14}
    >>> usys = u.unitsystem("km", "deg")

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     cxv.vconvert(cxv.CartesianPos1D, cxv.CylindricalPos,
    ...                       params, units=usys)
    ({'x': Array(0.9848077, dtype=float32, ...)}, {})

    >>> x = cx.vecs.CylindricalPos(rho=u.Quantity(1.0, "km"),
    ...                            phi=u.Quantity(10.0, "deg"),
    ...                            z=u.Quantity(14, "km"))
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     y = cx.vconvert(cx.vecs.CartesianPos1D, x)
    >>> print(y)
    <CartesianPos1D: (x) [km]
        [0.985]>

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return {"x": p["rho"] * jnp.cos(p["phi"])}, (out_aux or {})


@dispatch
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[d1.RadialPos],
    from_vector: type[d3.CylindricalPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """CylindricalPos -> RadialPos.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> params = {"rho": 1, "phi": 10, "z": 14}
    >>> usys = u.unitsystem("km", "deg")

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     cxv.vconvert(cxv.RadialPos, cxv.CylindricalPos, params, units=usys)
    ({'r': Array(1, dtype=int32, ...)}, {})

    >>> x = cxv.CylindricalPos(rho=u.Quantity(1, "km"),
    ...                        phi=u.Quantity(10, "deg"),
    ...                        z=u.Quantity(14, "km"))
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     y = cx.vconvert(cxv.RadialPos, x)
    >>> print(y)
    <RadialPos: (r) [km]
        [1]>

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return {"r": p["rho"]}, (out_aux or {})


# -----------------------------------------------
# 2D


@dispatch
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[d2.CartesianPos2D],
    from_vector: type[d3.CylindricalPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """CylindricalPos -> CartesianPos2D.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> params = {"rho": 1, "phi": 10, "z": 14}
    >>> usys = u.unitsystem("km", "deg")

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     cxv.vconvert(cxv.CartesianPos2D, cxv.CylindricalPos,
    ...                       params, units=usys)
    ({'x': Array(0.9848077, dtype=float32, ...),
      'y': Array(0.17364818, dtype=float32, ...)},
     {})

    >>> x = cxv.CylindricalPos(rho=u.Quantity(1, "km"),
    ...                        phi=u.Quantity(10, "deg"),
    ...                        z=u.Quantity(14, "km"))
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     y = cxv.vconvert(cxv.CartesianPos2D, x)
    >>> print(y)
    <CartesianPos2D: (x, y) [km]
        [0.985 0.174]>

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    x = p["rho"] * jnp.cos(p["phi"])
    y = p["rho"] * jnp.sin(p["phi"])
    return {"x": x, "y": y}, (out_aux or {})


@dispatch
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[d2.PolarPos],
    from_vector: type[d3.CylindricalPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """CylindricalPos -> PolarPos.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> params = {"rho": 1, "phi": 10, "z": 14}
    >>> usys = u.unitsystem("km", "deg")

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     cxv.vconvert(cxv.PolarPos, cxv.CylindricalPos,
    ...                       params, units=usys)
    ({'r': Array(1, dtype=int32, ...),
      'phi': Array(10., dtype=float32, ...)}, {})

    >>> x = cxv.CylindricalPos(rho=u.Quantity(1, "km"),
    ...                        phi=u.Quantity(10, "deg"),
    ...                        z=u.Quantity(14, "km"))
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     y = cxv.vconvert(cxv.PolarPos, x)
    >>> print(y)
    <PolarPos: (r[km], phi[deg])
        [ 1 10]>

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return {"r": p["rho"], "phi": p["phi"]}, (out_aux or {})


# =============================================================================
# SphericalPos


# -----------------------------------------------
# 1D


@dispatch
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[d1.CartesianPos1D],
    from_vector: type[d3.SphericalPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """SphericalPos -> CartesianPos1D.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> params = {"r": 1, "theta": 14, "phi": 10}
    >>> usys = u.unitsystem("km", "deg")

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     cxv.vconvert(cxv.CartesianPos1D, cxv.SphericalPos,
    ...                       params, units=usys)
    ({'x': Array(0.23824656, dtype=float32, ...)}, {})

    >>> x = cxv.SphericalPos(r=u.Quantity(1, "km"),
    ...                     theta=u.Quantity(14, "deg"),
    ...                     phi=u.Quantity(10, "deg"))
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     y = cxv.vconvert(cxv.CartesianPos1D, x)
    >>> print(y)
    <CartesianPos1D: (x) [km]
        [0.238]>

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    x = p["r"] * jnp.sin(p["theta"]) * jnp.cos(p["phi"])
    return {"x": x}, (out_aux or {})


# -----------------------------------------------
# 2D


@dispatch
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[d2.CartesianPos2D],
    from_vector: type[d3.SphericalPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """SphericalPos -> CartesianPos2D.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> params = {"r": 1, "theta": 14, "phi": 10}
    >>> usys = u.unitsystem("km", "deg")

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     cxv.vconvert(cxv.CartesianPos2D, cxv.SphericalPos,
    ...                       params, units=usys)
    ({'x': Array(0.23824656, dtype=float32, ...),
      'y': Array(0.0420093, dtype=float32, ...)},
     {})

    >>> x = cxv.SphericalPos(r=u.Quantity(1, "km"),
    ...                     theta=u.Quantity(14, "deg"),
    ...                     phi=u.Quantity(10, "deg"))
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     y = cxv.vconvert(cxv.CartesianPos2D, x)
    >>> print(y)
    <CartesianPos2D: (x, y) [km]
        [0.238 0.042]>

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    x = p["r"] * jnp.sin(p["theta"]) * jnp.cos(p["phi"])
    y = p["r"] * jnp.sin(p["theta"]) * jnp.sin(p["phi"])
    return {"x": x, "y": y}, (out_aux or {})


@dispatch
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[d2.PolarPos],
    from_vector: type[d3.SphericalPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """SphericalPos -> PolarPos.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> params = {"r": 1, "theta": 14, "phi": 10}
    >>> usys = u.unitsystem("km", "deg")

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     cxv.vconvert(cxv.PolarPos, cxv.SphericalPos,
    ...                       params, units=usys)
    ({'r': Array(0.2419219, dtype=float32, ...),
     'phi': Array(10., dtype=float32, ...)},
     {})

    >>> x = cxv.SphericalPos(r=u.Quantity(1, "km"),
    ...                     theta=u.Quantity(14, "deg"),
    ...                     phi=u.Quantity(10, "deg"))
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     y = cxv.vconvert(cxv.PolarPos, x)
    >>> print(y)
    <PolarPos: (r[km], phi[deg])
        [ 0.242 10.   ]>

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    r = p["r"] * jnp.sin(p["theta"])
    return {"r": r, "phi": p["phi"]}, (out_aux or {})


# =============================================================================
# MathSphericalPos


# -----------------------------------------------
# 1D


@dispatch
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[d1.CartesianPos1D],
    from_vector: type[d3.MathSphericalPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """MathSphericalPos -> CartesianPos1D.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> params = {"r": 1, "theta": 10, "phi": 14}
    >>> usys = u.unitsystem("km", "deg")

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     cxv.vconvert(cxv.CartesianPos1D, cxv.MathSphericalPos,
    ...                       params, units=usys)
    ({'x': Array(0.23824656, dtype=float32, ...)}, {})

    >>> x = cx.vecs.MathSphericalPos(r=u.Quantity(1, "km"),
    ...                              theta=u.Quantity(10, "deg"),
    ...                              phi=u.Quantity(14, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     y = cx.vconvert(cx.vecs.CartesianPos1D, x)
    >>> print(y)
    <CartesianPos1D: (x) [km]
        [0.238]>

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    x = p["r"] * jnp.sin(p["phi"]) * jnp.cos(p["theta"])
    return {"x": x}, (out_aux or {})


# -----------------------------------------------
# 2D


@dispatch
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[d2.CartesianPos2D],
    from_vector: type[d3.MathSphericalPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """MathSphericalPos -> CartesianPos2D.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> params = {"r": 1, "theta": 10, "phi": 14}
    >>> usys = u.unitsystem("km", "deg")

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     cxv.vconvert(cxv.CartesianPos2D, cxv.MathSphericalPos,
    ...                       params, units=usys)
    ({'x': Array(0.23824656, dtype=float32, ...),
      'y': Array(0.0420093, dtype=float32, ...)},
     {})

    >>> x = cx.vecs.MathSphericalPos(r=u.Quantity(1, "km"),
    ...                              theta=u.Quantity(10, "deg"),
    ...                              phi=u.Quantity(14, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     y = cx.vconvert(cx.vecs.CartesianPos2D, x)
    >>> print(y)
    <CartesianPos2D: (x, y) [km]
        [0.238 0.042]>

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    x = p["r"] * jnp.sin(p["phi"]) * jnp.cos(p["theta"])
    y = p["r"] * jnp.sin(p["phi"]) * jnp.sin(p["theta"])
    return {"x": x, "y": y}, (out_aux or {})


@dispatch
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[d2.PolarPos],
    from_vector: type[d3.MathSphericalPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """MathSphericalPos -> PolarPos.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> params = {"r": 1, "theta": 10, "phi": 14}
    >>> usys = u.unitsystem("km", "deg")

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     cxv.vconvert(cxv.PolarPos, cxv.MathSphericalPos,
    ...                       params, units=usys)
    ({'r': Array(0.2419219, dtype=float32, ...),
      'phi': Array(10., dtype=float32, ...)},
     {})

    >>> x = cxv.MathSphericalPos(r=u.Quantity(1, "km"),
    ...                          theta=u.Quantity(10, "deg"),
    ...                          phi=u.Quantity(14, "deg"))
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     y = cxv.vconvert(cxv.PolarPos, x)
    >>> print(y)
    <PolarPos: (r[km], phi[deg])
        [ 0.242 10.   ]>

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    r = p["r"] * jnp.sin(p["phi"])
    return {"r": r, "phi": p["theta"]}, (out_aux or {})


# =============================================================================
# ProlateSpheroidalPos


# -----------------------------------------------
# 1D


@dispatch.multi(
    (type[d1.AbstractPos1D], type[d3.ProlateSpheroidalPos], ct.ParamsDict),
    (type[d2.AbstractPos2D], type[d3.ProlateSpheroidalPos], ct.ParamsDict),
    (type[d3.AbstractPos3D], type[d3.ProlateSpheroidalPos], ct.ParamsDict),
)
@ft.partial(jax.jit, static_argnums=(0, 1), static_argnames=("units",))
def vconvert(
    to_vector: type[AbstractPos],
    from_vector: type[d3.ProlateSpheroidalPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """ProlateSpheroidalPos -> AbstractPos.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax as cx

    >>> x = cx.vecs.ProlateSpheroidalPos(
    ...     mu=u.Quantity(2, "km2"),
    ...     nu=u.Quantity(0.5, "km2"),
    ...     phi=u.Quantity(0.5, "rad"),
    ...     Delta=u.Quantity(1, "km"),
    ... )

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     y = cx.vconvert(cx.vecs.CartesianPos1D, x)
    >>> print(y)
    <CartesianPos1D: (x) [km]
        [0.621]>

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     y = cx.vconvert(cx.vecs.RadialPos, x)
    >>> print(y)
    <RadialPos: (r) [km]
        [0.707]>

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.vconvert(cx.vecs.CartesianPos2D, x)
    >>> print(x2)
    <CartesianPos2D: (x, y) [km]
        [0.621 0.339]>

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cx.vconvert(cx.vecs.PolarPos, x)
    >>> print(x2)
    <PolarPos: (r[km], phi[rad])
        [0.707 0.5  ]>

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

    >>> cxv.vconvert(cxv.CartesianPos3D, cxv.ProlateSpheroidalPos,
    ...                   vec, in_aux=in_aux, units=usys)
    ({'x': Array(-1.6929347e-08, dtype=float32, ...),
      'y': Array(0.38729832, dtype=float32, ...),
      'z': Array(0.8944272, dtype=float32, ...)},
     {})

    """
    p, aux = vconvert(
        d3.CylindricalPos, from_vector, p, in_aux=in_aux, out_aux=None, units=units
    )
    p, aux = vconvert(
        to_vector, d3.CylindricalPos, p, in_aux=aux, out_aux=out_aux, units=units
    )
    return p, aux
