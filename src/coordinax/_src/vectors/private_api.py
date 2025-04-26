"""Copyright (c) 2023 coordinax maintainers. All rights reserved."""

__all__ = ["spatial_component"]

import functools as ft
from typing import Any, TypeAlias

from jaxtyping import Array, ArrayLike
from plum import dispatch

import unxt as u
from unxt.quantity import is_any_quantity

from . import custom_types as ct
from .base import AbstractVector
from .base_pos import AbstractPos


@dispatch.abstract
def spatial_component(x: Any, /) -> Any:
    """Return the spatial component of the vector."""
    raise NotImplementedError  # pragma: no cover


@dispatch
def spatial_component(x: AbstractPos, /) -> AbstractPos:
    """Return the spatial component of the vector."""
    return x


# ============================================================================

DimsDict: TypeAlias = dict[str, u.dims.AbstractDimension]

internal_units = {u.dimension("angle"): u.unit("radian")}


def ustrip_to_internal_units(
    v: ArrayLike, k: str, dims: DimsDict, usys: u.AbstractUnitSystem
) -> Array:
    dim = dims[k]
    from_unit = usys[dim]
    to_unit = internal_units.get(dim, from_unit)
    return u.Quantity(v, from_unit).ustrip(to_unit)


def vconvert_parse_input(
    params: ct.ParamsDict, dims: DimsDict, usys: ct.OptUSys, /
) -> ct.ParamsDict:
    """Parse input parameters.

    The parameters ``params`` can be either array-valued or Quantity-valued. If
    they are Quantity-valued then this function does nothing. If they are
    array-valued then they are potentially values in the wrong unit system and
    need to be converted.

    Parameters
    ----------
    params
        Current set of parameters, which can either be array-valued or
        Quantity-valued.
    dims
        Dictionary mapping parameter names to dimensions. This is needed for
        getting the correct units from the unitsystem.
    usys
        The unitsystem that represents the units of the parameters. This is only
        needed if the parameters are not Quantity-valued. If it's None then the
        parameters are assumed to be in the internal units used by
        `coordinax.vecs.vconvert_imp`, e.g. radians for angles. If provided then
        array parameters are converted to Quantity object with units from `usys`
        and then stripped to the internal units.

    Examples
    --------
    >>> import unxt as u
    >>> from coordinax._src.vectors.private_api import vconvert_parse_input

    >>> usys = u.unitsystem("kpc", "deg")
    >>> params = {"x": 1, "phi": 180/3.1415926}  # in [usys]
    >>> dims = {"x": u.dimension("length"), "phi": u.dimension("angle")}

    >>> vconvert_parse_input(params, dims, usys)
    {'x': Array(1, dtype=int32, ...), 'phi': Array(1., dtype=float32, ...)}

    Note how `x` is unchanged since the internal unit system doesn't have a
    distance scale, but the `phi` parameter is converted to radians since all
    the numpy functions expect angles in radians.

    """
    if usys is None:
        return params
    return {
        k: (v if is_any_quantity(v) else ustrip_to_internal_units(v, k, dims, usys))
        for k, v in params.items()
    }


def ustrip_from_internal_units(
    v: ArrayLike, k: str, dims: DimsDict, usys: u.AbstractUnitSystem
) -> Array:
    dim = dims[k]
    to_unit = usys[dim]
    from_unit = internal_units.get(dim, to_unit)
    return u.Quantity(v, from_unit).ustrip(to_unit)


def vconvert_parse_output(
    params: ct.ParamsDict, dims: DimsDict, usys: ct.OptUSys, /
) -> ct.ParamsDict:
    """Parse output parameters."""
    if usys is None:
        return params
    return {
        k: (
            v.uconvert(usys[u.dimension_of(v)])
            if is_any_quantity(v)
            else ustrip_from_internal_units(v, k, dims, usys)
        )
        for k, v in params.items()
    }


def combine_aux(in_aux: ct.OptAuxDict, out_aux: ct.OptAuxDict, /) -> ct.AuxDict:
    """Combine auxiliary dictionaries."""
    return (in_aux or {}) | (out_aux or {})


def wrap_vconvert_impl_params(func: Any) -> Any:
    """Wrap a function that implements a vector conversion."""

    @ft.wraps(func)
    def wrapper(
        to_vector: type[AbstractVector],
        from_vector: type[AbstractVector],
        p: ct.ParamsDict,
        /,
        *args: Any,
        units: ct.OptUSys = None,
        **kwargs: Any,
    ) -> tuple[ct.ParamsDict, ct.AuxDict]:
        p = vconvert_parse_input(p, from_vector.dimensions, units)
        outp, aux = func(to_vector, from_vector, p, *args, units=units, **kwargs)
        outp = vconvert_parse_output(outp, to_vector.dimensions, units)
        return outp, aux

    return wrapper
