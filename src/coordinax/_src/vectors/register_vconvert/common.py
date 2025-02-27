"""Transformations for Vectors."""

__all__: list[str] = []

from typing import Any, cast

from plum import dispatch

from coordinax._src.vectors.api import vconvert_impl
from coordinax._src.vectors.base_pos import AbstractPos


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

    # # Parameters can be passed by kwarg, so we need to filter them out
    # # from the auxiliary data. E.g. for RadialPos -> CartesianPos2D
    # # we need to specify `y` by kwarg. It is a param, not an aux.
    # p = p | {k: out_aux.pop(k) for k in tuple(out_aux) if k in target}

    # Convert the parameters, using the auxiliary data
    p, aux = vconvert_impl(target, type(current), p, in_aux=in_aux, out_aux=out_aux)

    # Build the new vector
    return target(**(aux or {}), **p)
