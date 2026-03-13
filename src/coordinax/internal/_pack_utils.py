"""Utilities for packing/unpacking coordinate dicts with uniform units."""

__all__ = (
    "pack_uniform_unit",
    "unpack_with_unit",
    "cdict_units",
    "pack_nonuniform_unit",
)

from jaxtyping import ArrayLike
from typing import Any, Final, overload

import jax.tree as jtu
import plum

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

from .custom_types import CDict, CKey

DMLS: Final = u.unit("")


@overload
def pack_uniform_unit(
    p: dict[CKey, Any], /, keys: tuple[CKey, ...]
) -> tuple[jnp.ndarray, u.AbstractUnit]: ...


@overload
def pack_uniform_unit(
    p: dict[CKey, ArrayLike], /, keys: tuple[CKey, ...]
) -> tuple[jnp.ndarray, None]: ...


def pack_uniform_unit(
    p: CDict, /, keys: tuple[CKey, ...]
) -> tuple[jnp.ndarray, u.AbstractUnit | None]:
    """Pack dict-of-quantities into an array.

    Converting all entries to a common unit.
    """
    # Choose a reference unit from the first key.
    v0 = p[keys[0]]
    unitful = isinstance(v0, u.AbstractQuantity)
    unit = v0.unit if unitful else DMLS
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


@plum.dispatch
def cdict_units(
    p: CDict, keys: tuple[str, ...], /
) -> tuple[u.AbstractUnit | None, ...]:
    """Extract the units from a dict of quantities.

    Examples
    --------
    >>> import unxt as u

    >>> d = {'x': u.Q(1.0, 'm'), 'y': 2.0, 'z': u.Q(3.0, 'kg')}
    >>> cdict_units(d, ('x', 'y', 'z'))
    (Unit("m"), None, Unit("kg"))

    """
    # `unit_of()` returns None for non-quantities, so this works for both cases.
    return tuple(u.unit_of(p[k]) for k in keys)


def pack_nonuniform_unit(
    p: CDict, keys: tuple[str, ...]
) -> tuple[jnp.ndarray, tuple[u.AbstractUnit | None, ...]]:
    """Pack dict-of-quantities into an array, allowing different units."""
    units = cdict_units(p, keys)
    vals = [
        u.ustrip(AllowValue, unit, p[k]) for k, unit in zip(keys, units, strict=True)
    ]
    return jnp.stack(vals, axis=-1), units
