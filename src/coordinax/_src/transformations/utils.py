"""Utilities."""

__all__: tuple[str, ...] = ("pack_uniform_unit", "unpack_with_unit")

from jaxtyping import ArrayLike
from typing import Any, overload

import jax.tree as jtu

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

from coordinax._src.custom_types import CDict, ComponentsKey, CsDict


@overload
def pack_uniform_unit(
    p: dict[ComponentsKey, Any], /, keys: tuple[str, ...]
) -> tuple[jnp.ndarray, u.AbstractUnit]: ...


@overload
def pack_uniform_unit(
    p: dict[ComponentsKey, ArrayLike], /, keys: tuple[str, ...]
) -> tuple[jnp.ndarray, None]: ...


def pack_uniform_unit(
    p: CsDict, /, keys: tuple[str, ...]
) -> tuple[jnp.ndarray, u.AbstractUnit | None]:
    """Pack dict-of-quantities into an array.

    Converting all entries to a common unit.
    """
    # Choose a reference unit from the first key.
    v0 = p[keys[0]]
    unitful = isinstance(v0, u.AbstractQuantity)
    unit = v0.unit if unitful else u.unit("")
    vals = [u.ustrip(AllowValue, unit, p[k]) for k in keys]
    return jnp.stack(vals, axis=-1), unit if unitful else None


def unpack_with_unit(
    vals: jnp.ndarray, unit: u.AbstractUnit | None, keys: tuple[str, ...]
) -> CDict:
    """Unpack an array into dict-of-quantities with a shared unit."""
    data = {k: vals[..., i] for i, k in enumerate(keys)}
    if unit is None:
        return data
    return jtu.map(lambda v: u.Q(v, unit), data)
