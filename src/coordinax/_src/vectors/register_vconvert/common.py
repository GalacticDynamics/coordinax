"""Transformations for Vectors."""

__all__: list[str] = []

from typing import Any, cast
from warnings import warn

import equinox as eqx
from plum import dispatch

import coordinax._src.vectors.custom_types as ct
from coordinax._src.vectors import d1, d2, d3
from coordinax._src.vectors.api import vconvert_impl
from coordinax._src.vectors.base_pos import AbstractPos
from coordinax._src.vectors.exceptions import IrreversibleDimensionChange
from coordinax._src.vectors.private_api import wrap_vconvert_impl_params


# TODO: move this to universal location
@dispatch
def vconvert(
    target: type[AbstractPos], current: AbstractPos, /, **out_aux: Any
) -> AbstractPos:
    """AbstractPos -> vconvert_impl -> AbstractPos.

    This is the base case for the transformation of position vectors.

    """
    # Get the parameters and auxiliary data
    p_and_aux = cast("dict[str, Any]", current.asdict())
    # Separate the parameters from the auxiliary data
    comps = current.components
    p = {k: p_and_aux.pop(k) for k in tuple(p_and_aux) if k in comps}
    in_aux = p_and_aux  # popped all the params out

    # Parameters can be passed by kwarg, so we need to filter them out
    # from the auxiliary data. E.g. for RadialPos -> CartesianPos2D
    # we need to specify `y` by kwarg. It is a param, not an aux.
    p_by_kw = {k: out_aux.pop(k) for k in tuple(out_aux) if k in target.components}
    p_by_kw = eqx.error_if(p_by_kw, any(set(p_by_kw) & set(p)), "overlap in params.")
    p = p | p_by_kw

    # Convert the parameters, using the auxiliary data
    p, aux = vconvert_impl(target, type(current), p, in_aux=in_aux, out_aux=out_aux)

    # Build the new vector
    return target(**(aux or {}), **p)


# =============================================================================
# CartesianND -> Cartesian1D


@dispatch.multi(
    (type[d1.CartesianPos1D], type[d2.CartesianPos2D], ct.ParamsDict),
    (type[d1.CartesianPos1D], type[d3.CartesianPos3D], ct.ParamsDict),
    (type[d1.RadialPos], type[d2.PolarPos], ct.ParamsDict),
    (type[d2.CartesianPos2D], type[d3.CartesianPos3D], ct.ParamsDict),
    (type[d1.RadialPos], type[d3.SphericalPos], ct.ParamsDict),
    (type[d1.RadialPos], type[d3.MathSphericalPos], ct.ParamsDict),
)
@wrap_vconvert_impl_params
def vconvert_impl(
    to_vector: type[AbstractPos],
    from_vector: type[AbstractPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """CartesianPos2D -> CartesianPos1D.

    The `y` and `z` coordinates are dropped.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    ## 2D:

    >>> params = {"x": 1, "y": 2}

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     cxv.vconvert_impl(cxv.CartesianPos1D, cxv.CartesianPos2D, params)
    ({'x': 1}, {})

    >>> x = cxv.CartesianPos2D.from_([1, 2], "km")
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     y = cxv.vconvert(cxv.CartesianPos1D, x)
    >>> print(y)
    <CartesianPos1D (x[km])
        [1]>

    >>> x = cxv.PolarPos(r=u.Quantity(1, "km"), phi=u.Quantity(10, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     y = cxv.vconvert(cxv.RadialPos, x)
    >>> print(y)
    <RadialPos (r[km])
        [1]>

    ## 3D:

    >>> params = {"x": 1, "y": 2, "z": 3}

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     cxv.vconvert_impl(cxv.CartesianPos1D, cxv.CartesianPos3D, params)
    ({'x': 1}, {})

    >>> x = cxv.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     y = cxv.vconvert(cxv.CartesianPos1D, x)
    >>> print(y)
    <CartesianPos1D (x[km])
        [1.]>

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     cxv.vconvert_impl(cxv.CartesianPos2D, cxv.CartesianPos3D, params)
    ({'x': 1, 'y': 2}, {})

    >>> x = cxv.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     y = cxv.vconvert(cxv.CartesianPos2D, x)
    >>> print(y)
    <CartesianPos2D (x[km], y[km])
        [1. 2.]>

    >>> params = {"r": 1, "theta": 14, "phi": 10}
    >>> usys = u.unitsystem("km", "deg")

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     cxv.vconvert_impl(cxv.RadialPos, cxv.SphericalPos,
    ...                       params, units=usys)
    ({'r': Array(1, dtype=int32, ...)}, {})

    >>> x = cxv.SphericalPos(r=u.Quantity(1, "km"),
    ...                     theta=u.Quantity(14, "deg"),
    ...                     phi=u.Quantity(10, "deg"))
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     y = cxv.vconvert(cxv.RadialPos, x)
    >>> print(y)
    <RadialPos (r[km])
        [1]>

    >>> params = {"r": 1, "theta": 10, "phi": 14}
    >>> usys = u.unitsystem("km", "deg")

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     cxv.vconvert_impl(cxv.RadialPos, cxv.MathSphericalPos,
    ...                       params, units=usys)
    ({'r': Array(1, dtype=int32, ...)}, {})

    >>> x = cxv.MathSphericalPos(r=u.Quantity(1, "km"),
    ...                          theta=u.Quantity(10, "deg"),
    ...                          phi=u.Quantity(14, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     y = cxv.vconvert(cxv.RadialPos, x)
    >>> print(y)
    <RadialPos (r[km])
        [1]>

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return {k: p[k] for k in to_vector.components}, (out_aux or {})
