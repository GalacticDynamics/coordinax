"""Transformations from 1D."""

__all__: list[str] = []


from plum import dispatch

import unxt as u
from unxt.quantity import is_any_quantity

import coordinax._src.vectors.custom_types as ct
from coordinax._src.vectors import d1, d2, d3
from coordinax._src.vectors.private_api import wrap_vconvert_impl_params

# =============================================================================
# CartesianPos1D


# -----------------------------------------------
# to 2D


@dispatch
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[d2.CartesianPos2D],
    from_vector: type[d1.CartesianPos1D],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """CartesianPos1D -> CartesianPos2D.

    The `x` coordinate is converted to the `x` coordinate of the 2D system.
    The `y` coordinate is a keyword argument and defaults to 0.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> x = cxv.CartesianPos1D(x=u.Quantity(1.0, "km"))
    >>> x2 = cxv.vconvert(cxv.CartesianPos2D, x)
    >>> print(x2)
    <CartesianPos2D: (x, y) [km]
        [1. 0.]>

    >>> x3 = cxv.vconvert(cxv.CartesianPos3D, x, y=u.Quantity(14, "km"))
    >>> print(x3)
    <CartesianPos3D: (x, y, z) [km]
        [ 1. 14.  0.]>

    """
    x = p["x"]
    return {"x": x, "y": p.get("y", 0 * x)}, (out_aux or {})


@dispatch
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[d2.PolarPos],
    from_vector: type[d1.CartesianPos1D],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """CartesianPos1D -> PolarPos.

    The `x` coordinate is converted to the radial coordinate `r`.
    The `phi` coordinate is a keyword argument and defaults to 0.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> x = cxv.CartesianPos1D(x=u.Quantity(1.0, "km"))
    >>> x2 = cxv.vconvert(cxv.PolarPos, x)
    >>> print(x2)
    <PolarPos: (r[km], phi[rad])
        [1. 0.]>

    >>> x3 = cxv.vconvert(cxv.PolarPos, x, phi=u.Quantity(14, "deg"))
    >>> print(x3)
    <PolarPos: (r[km], phi[deg])
        [ 1. 14.]>

    """
    r = p["x"]
    phi = p.get("phi", u.Quantity(0, "rad") if is_any_quantity(r) else 0)
    return {"r": r, "phi": phi}, (p.get("out_aux", {}) or {})


# -----------------------------------------------
# to 3D


@dispatch
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[d3.CartesianPos3D],
    from_vector: type[d1.CartesianPos1D],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """CartesianPos1D -> CartesianPos3D.

    The `x` coordinate is converted to the `x` coordinate of the 3D system.
    The `y` and `z` coordinates are keyword arguments and default to 0.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> x = cxv.CartesianPos1D(x=u.Quantity(1, "km"))
    >>> x2 = cxv.vconvert(cxv.CartesianPos3D, x)
    >>> print(x2)
    <CartesianPos3D: (x, y, z) [km]
        [1 0 0]>

    >>> x3 = cxv.vconvert(cxv.CartesianPos3D, x, y=u.Quantity(14, "km"))
    >>> print(x3)
    <CartesianPos3D: (x, y, z) [km]
        [ 1 14  0]>

    """
    x = p["x"]
    return {"x": x, "y": p.get("y", 0 * x), "z": p.get("z", 0 * x)}, (out_aux or {})


@dispatch.multi(
    (type[d3.SphericalPos], type[d1.CartesianPos1D], ct.ParamsDict),
    (type[d3.MathSphericalPos], type[d1.CartesianPos1D], ct.ParamsDict),
)
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[d3.AbstractPos3D],
    from_vector: type[d1.CartesianPos1D],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """CartesianPos1D -> SphericalPos | MathSphericalPos.

    The `x` coordinate is converted to the radial coordinate `r`.
    The `theta` and `phi` coordinates are keyword arguments and default to 0.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    SphericalPos:

    >>> x = cxv.CartesianPos1D(x=u.Quantity(1, "km"))
    >>> x2 = cxv.vconvert(cxv.SphericalPos, x)
    >>> print(x2)
    <SphericalPos: (r[km], theta[rad], phi[rad])
        [1. 0. 0.]>

    >>> x3 = cxv.vconvert(cxv.SphericalPos, x, phi=u.Quantity(14, "deg"))
    >>> print(x3)
    <SphericalPos: (r[km], theta[rad], phi[deg])
        [ 1  0 14]>

    MathSphericalPos:
    Note that ``theta`` and ``phi`` have different meanings in this context.

    >>> x2 = cxv.vconvert(cxv.MathSphericalPos, x)
    >>> print(x2)
    <MathSphericalPos: (r[km], theta[rad], phi[rad])
        [1. 0. 0.]>

    >>> x3 = cxv.vconvert(cxv.MathSphericalPos, x, phi=u.Quantity(14, "deg"))
    >>> print(x3)
    <MathSphericalPos: (r[km], theta[rad], phi[deg])
        [ 1.  0. 14.]>

    """
    r = p["x"]
    theta = p.get("theta", u.Quantity(0, "rad") if is_any_quantity(r) else 0)
    phi = p.get("phi", u.Quantity(0, "rad") if is_any_quantity(r) else 0)
    return {"r": r, "theta": theta, "phi": phi}, (out_aux or {})


@dispatch
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[d3.CylindricalPos],
    from_vector: type[d1.CartesianPos1D],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """RadialPos -> CartesianPos2D.

    The `x` coordinate is converted to the radial coordinate `rho`.
    The `phi` and `z` coordinates are keyword arguments and default to 0.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> x = cxv.CartesianPos1D(x=u.Quantity(1, "km"))
    >>> x2 = cxv.vconvert(cxv.CylindricalPos, x)
    >>> print(x2)
    <CylindricalPos: (rho[km], phi[rad], z[km])
        [1. 0. 0.]>

    >>> x3 = cxv.vconvert(cxv.CylindricalPos, x, phi=u.Quantity(14, "deg"))
    >>> print(x3)
    <CylindricalPos: (rho[km], phi[deg], z[km])
        [ 1 14  0]>

    """
    rho = p["x"]
    phi = p.get("phi", u.Quantity(0, "rad") if is_any_quantity(rho) else 0)
    z = p.get("z", 0 * rho)
    return {"rho": rho, "phi": phi, "z": z}, (out_aux or {})


# =============================================================================
# RadialPos

# -----------------------------------------------
# 2D


@dispatch
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[d2.CartesianPos2D],
    from_vector: type[d1.RadialPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """RadialPos -> CartesianPos2D.

    The `r` coordinate is converted to the cartesian coordinate `x`.
    The `y` coordinate is a keyword argument and defaults to 0.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> x = cxv.RadialPos(r=u.Quantity(1, "km"))
    >>> x2 = cxv.vconvert(cxv.CartesianPos2D, x)
    >>> print(x2)
    <CartesianPos2D: (x, y) [km]
        [1 0]>

    >>> x3 = cxv.vconvert(cxv.CartesianPos2D, x, y=u.Quantity(14, "km"))
    >>> print(x3)
    <CartesianPos2D: (x, y) [km]
        [ 1 14]>

    """
    r = p["r"]
    return {"x": r, "y": p.get("y", 0 * r)}, (out_aux or {})


@dispatch
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[d2.PolarPos],
    from_vector: type[d1.RadialPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """RadialPos -> PolarPos.

    The `r` coordinate is converted to the radial coordinate `r`.
    The `phi` coordinate is a keyword argument and defaults to 0.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> x = cxv.RadialPos(r=u.Quantity(1, "km"))
    >>> x2 = cxv.vconvert(cxv.PolarPos, x)
    >>> print(x2)
    <PolarPos: (r[km], phi[rad])
        [1. 0.]>

    >>> x3 = cxv.vconvert(cxv.PolarPos, x, phi=u.Quantity(14, "deg"))
    >>> print(x3)
    <PolarPos: (r[km], phi[deg])
        [ 1 14]>

    """
    newp = {
        "r": p["r"],
        "phi": p.get("phi", u.Quantity(0, "rad") if is_any_quantity(p["r"]) else 0),
    }
    return newp, (out_aux or {})


# -----------------------------------------------
# 3D


# TODO: I don't like this rule. C3->R1 takes the norm. This places it on the x
# axis. What about giving theta, phi instead?
@dispatch
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[d3.CartesianPos3D],
    from_vector: type[d1.RadialPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """RadialPos -> CartesianPos3D.

    The `r` coordinate is converted to the `x` coordinate of the 3D system.
    The `y` and `z` coordinates are keyword arguments and default to 0.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> x = cxv.RadialPos(r=u.Quantity(1.0, "km"))
    >>> y = cxv.vconvert(cxv.CartesianPos3D, x)
    >>> print(y)
    <CartesianPos3D: (x, y, z) [km]
        [1. 0. 0.]>

    >>> y = cxv.vconvert(cxv.CartesianPos3D, x, y=u.Quantity(14, "km"))
    >>> print(y)
    <CartesianPos3D: (x, y, z) [km]
        [ 1. 14.  0.]>

    """
    r = p["r"]
    newp = {"x": r, "y": p.get("y", 0 * r), "z": p.get("z", 0 * r)}
    return newp, (out_aux or {})


@dispatch.multi(
    (type[d3.SphericalPos], type[d1.RadialPos], ct.ParamsDict),
    (type[d3.MathSphericalPos], type[d1.RadialPos], ct.ParamsDict),
)
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[d3.AbstractPos3D],
    from_vector: type[d1.RadialPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """RadialPos -> SphericalPos | MathSphericalPos.

    The `r` coordinate is converted to the radial coordinate `r`.
    The `theta` and `phi` coordinates are keyword arguments and default to 0.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> x = cxv.RadialPos(r=u.Quantity(1, "km"))

    SphericalPos:

    >>> y = cxv.vconvert(cxv.SphericalPos, x)
    >>> print(y)
    <SphericalPos: (r[km], theta[rad], phi[rad])
        [1. 0. 0.]>

    >>> y = cxv.vconvert(cxv.SphericalPos, x, phi=u.Quantity(14, "deg"))
    >>> print(y)
    <SphericalPos: (r[km], theta[rad], phi[deg])
        [ 1  0 14]>

    MathSphericalPos:

    >>> y = cxv.vconvert(cxv.MathSphericalPos, x)
    >>> print(y)
    <MathSphericalPos: (r[km], theta[rad], phi[rad])
        [1. 0. 0.]>

    >>> y = cxv.vconvert(cxv.MathSphericalPos, x, phi=u.Quantity(14, "deg"))
    >>> print(y)
    <MathSphericalPos: (r[km], theta[rad], phi[deg])
        [ 1.  0. 14.]>

    """
    r = p["r"]
    newp = {
        "r": r,
        "theta": p.get("theta", u.Quantity(0, "rad") if is_any_quantity(r) else 0),
        "phi": p.get("phi", u.Quantity(0, "rad") if is_any_quantity(r) else 0),
    }
    return newp, (out_aux or {})


@dispatch
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[d3.CylindricalPos],
    from_vector: type[d1.RadialPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """RadialPos -> CylindricalPos.

    The `r` coordinate is converted to the radial coordinate `rho`.
    The `phi` and `z` coordinates are keyword arguments and default to 0.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> x = cxv.RadialPos(r=u.Quantity(1.0, "km"))
    >>> y = cxv.vconvert(cxv.CylindricalPos, x)
    >>> print(y)
    <CylindricalPos: (rho[km], phi[rad], z[km])
        [1. 0. 0.]>

    >>> y = cxv.vconvert(cxv.CylindricalPos, x, phi=u.Quantity(14, "deg"))
    >>> print(y)
    <CylindricalPos: (rho[km], phi[deg], z[km])
        [ 1. 14.  0.]>

    """
    r = p["r"]
    newp = {
        "rho": r,
        "phi": p.get("phi", u.Quantity(0, "rad") if is_any_quantity(r) else 0),
        "z": p.get("z", 0 * r),
    }
    return newp, (out_aux or {})
