"""Transformations between representations."""

__all__: list[str] = []

from warnings import warn

from plum import dispatch

import quaxed.numpy as jnp

import coordinax._src.vectors.custom_types as ct
from coordinax._src.vectors import d1, d2, d3
from coordinax._src.vectors.exceptions import IrreversibleDimensionChange
from coordinax._src.vectors.private_api import wrap_vconvert_impl_params


@dispatch.multi(
    (type[d3.CylindricalPos], type[d2.CartesianPos2D], ct.ParamsDict),
    (type[d3.SphericalPos], type[d2.CartesianPos2D], ct.ParamsDict),
    (type[d3.MathSphericalPos], type[d2.CartesianPos2D], ct.ParamsDict),
)
def vconvert(
    to_vector: type[d3.AbstractPos3D],
    from_vector: type[d2.CartesianPos2D],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """AbstractPos2D -> Cartesian2D -> Cartesian3D -> AbstractPos3D.

    The 2D vector is in the xy plane. The `z` coordinate is a keyword argument and
    defaults to 0.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> x = cxv.CartesianPos2D.from_([1, 2], "km")

    >>> y = cxv.vconvert(cxv.CylindricalPos, x, z=u.Quantity(14, "km"))
    >>> print(y)
    <CylindricalPos: (rho[km], phi[rad], z[km])
        [ 2.236  1.107 14.   ]>

    >>> y = cxv.vconvert(cxv.SphericalPos, x, z=u.Quantity(14, "km"))
    >>> print(y)
    <SphericalPos: (r[km], theta[rad], phi[rad])
        [14.177  0.158  1.107]>

    >>> y = cxv.vconvert(cxv.MathSphericalPos, x, z=u.Quantity(14, "km"))
    >>> print(y)
    <MathSphericalPos: (r[km], theta[rad], phi[rad])
        [14.177  1.107  0.158]>

    """
    p, aux = vconvert(d3.CartesianPos3D, from_vector, p, in_aux=in_aux, out_aux=None)
    # The z coordinate needs to be provided for the total conversion, however it
    # can either be consumed in the previous sub-conversion or appear in
    # out_aux, so we need to handle both cases.
    p["z"] = out_aux.pop("z", p["z"])
    p, aux = vconvert(
        to_vector, d3.CartesianPos3D, p, in_aux=aux, out_aux=out_aux, units=units
    )
    return p, aux


@dispatch
def vconvert(
    to_vector: type[d3.CartesianPos3D],
    from_vector: type[d2.PolarPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """AbstractPos2D -> PolarPos -> Cylindrical -> AbstractPos3D.

    The 2D vector is in the xy plane. The `z` coordinate is a keyword argument and
    defaults to 0.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> x = cxv.PolarPos(r=u.Quantity(1, "km"), phi=u.Quantity(10, "deg"))

    >>> x2 = cxv.vconvert(cxv.CartesianPos3D, x, z=u.Quantity(14, "km"))
    >>> print(x2)
    <CartesianPos3D: (x, y, z) [km]
        [ 0.985  0.174 14.   ]>

    """
    newp = {
        "x": p["r"] * jnp.cos(p["phi"]),
        "y": p["r"] * jnp.sin(p["phi"]),
        "z": p.get("z", 0 * p["r"]),
    }
    return newp, (out_aux or {})


# =============================================================================
# CartesianPos2D


# -----------------------------------------------
# 1D


@dispatch
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[d1.RadialPos],
    from_vector: type[d2.CartesianPos2D],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """CartesianPos2D -> RadialPos.

    The `x` and `y` coordinates are converted to the radial coordinate `r`.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> x = cxv.CartesianPos2D.from_([1, 2], "km")

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     y = cxv.vconvert(cxv.RadialPos, x)
    >>> print(y)
    <RadialPos: (r) [km]
        [2.236]>

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    newp = {"r": jnp.hypot(p["x"], p["y"])}
    return newp, (out_aux or {})


# -----------------------------------------------
# 3D


@dispatch
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[d3.CartesianPos3D],
    from_vector: type[d2.CartesianPos2D],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """CartesianPos2D -> CartesianPos3D.

    The `x` and `y` coordinates are converted to the `x` and `y` coordinates of
    the 3D system.  The `z` coordinate is a keyword argument and defaults to 0.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> x = cxv.CartesianPos2D.from_([1, 2], "km")

    >>> x2 = cxv.vconvert(cxv.CartesianPos3D, x, z=u.Quantity(14, "km"))
    >>> print(x2)
    <CartesianPos3D: (x, y, z) [km]
        [ 1  2 14]>

    """
    newp = {"x": p["x"], "y": p["y"], "z": p.get("z", 0 * p["x"])}
    return newp, (out_aux or {})


# =============================================================================
# PolarPos

# -----------------------------------------------
# 1D


@dispatch
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[d1.CartesianPos1D],
    from_vector: type[d2.PolarPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """PolarPos -> CartesianPos1D.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> x = cxv.PolarPos(r=u.Quantity(1, "km"), phi=u.Quantity(10, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     x2 = cxv.vconvert(cxv.CartesianPos1D, x)
    >>> print(x2)
    <CartesianPos1D: (x) [km]
        [0.985]>

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    newp = {"x": p["r"] * jnp.cos(p["phi"])}
    return newp, (out_aux or {})


# -----------------------------------------------
# 3D


@dispatch
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[d3.SphericalPos],
    from_vector: type[d2.PolarPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """PolarPos -> SphericalPos.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> x = cxv.PolarPos(r=u.Quantity(1, "km"), phi=u.Quantity(10, "deg"))

    >>> x2 = cxv.vconvert(cxv.SphericalPos, x, theta=u.Quantity(14, "deg"))
    >>> print(x2)
    <SphericalPos: (r[km], theta[deg], phi[deg])
        [ 1 14 10]>

    """
    newp = {"r": p["r"], "theta": p.get("theta", 0 * p["phi"]), "phi": p["phi"]}
    return newp, (out_aux or {})


@dispatch
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[d3.MathSphericalPos],
    from_vector: type[d2.PolarPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """PolarPos -> MathSphericalPos.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> x = cxv.PolarPos(r=u.Quantity(1, "km"), phi=u.Quantity(10, "deg"))

    >>> y = cxv.vconvert(cxv.MathSphericalPos, x, theta=u.Quantity(14, "deg"))
    >>> print(y)
    <MathSphericalPos: (r[km], theta[deg], phi[deg])
        [ 1 10 14]>

    """
    newp = {"r": p["r"], "theta": p["phi"], "phi": p.get("theta", 0 * p["phi"])}
    return newp, (out_aux or {})


@dispatch
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[d3.CylindricalPos],
    from_vector: type[d2.PolarPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """PolarPos -> CylindricalPos.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> x = cxv.PolarPos(r=u.Quantity(1, "km"), phi=u.Quantity(10, "deg"))

    >>> x2 = cxv.vconvert(cxv.CylindricalPos, x, z=u.Quantity(14, "km"))
    >>> print(x2)
    <CylindricalPos: (rho[km], phi[deg], z[km])
        [ 1 10 14]>

    >>> x2 = cxv.vconvert(cxv.CylindricalPos, x)
    >>> print(x2)
    <CylindricalPos: (rho[km], phi[deg], z[km])
        [ 1 10 0]>

    """
    newp = {"rho": p["r"], "phi": p["phi"], "z": p.get("z", 0 * p["r"])}
    return newp, (out_aux or {})
