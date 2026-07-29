"""Helpers for packing component dictionaries into arrays while tracking units.

The helpers in this module are used when chart- or vector-like data is most
conveniently manipulated as a stacked JAX array, but unit metadata still needs
to be preserved across the pack/unpack boundary.

Two packing modes are supported:

- uniform-unit packing, where all entries are converted into one shared unit
- non-uniform packing, where each entry keeps its own unit metadata
"""

__all__ = (
    "pack_uniform_unit",
    "pack_nonuniform_unit",
)

from jaxtyping import Array, ArrayLike
from typing import Any, Final, overload

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
    """Pack a component dictionary into one array using a shared unit.

    The first requested key chooses the reference unit when the data is
    quantity-valued. Remaining entries are converted into that unit before the
    values are stacked along the trailing axis. If the entries are plain arrays
    or scalars, the returned unit is `None`.

    Parameters
    ----------
    p
        Component dictionary to pack.
    keys
        Ordered keys to extract and stack.

    Returns
    -------
    tuple[jnp.ndarray, u.AbstractUnit | None]
        Packed values together with the shared unit used for conversion, or
        `None` for unitless inputs.

    Examples
    --------
    >>> import unxt as u
    >>> from coordinax.internal import pack_uniform_unit

    >>> p = {"x": u.Q(1.0, "km"), "y": u.Q(200.0, "m")}
    >>> vals, unit = pack_uniform_unit(p, ("x", "y"))
    >>> unit
    Unit("km")

    """
    # Choose a reference unit from the first key.
    v0 = p[keys[0]]
    unitful = isinstance(v0, u.AbstractQuantity)
    unit = v0.unit if unitful else DMLS
    vals = [u.ustrip(AllowValue, unit, p[k]) for k in keys]
    return jnp.stack(vals, axis=-1), unit if unitful else None  # ty: ignore[invalid-return-type]


def pack_nonuniform_unit(
    p: CDict, /, keys: tuple[str, ...]
) -> tuple[Array, tuple[u.AbstractUnit | None, ...]]:
    """Pack a component dictionary into an array with per-component units.

    Unlike `pack_uniform_unit`, this helper does not choose a single reference
    unit. Each requested component is stripped in its own native unit and the
    resulting unit tuple is returned alongside the stacked values.

    This is the appropriate packing mode when different coordinates naturally
    have different physical dimensions or when preserving the original unit of
    each component is important.
    """
    units = tuple(u.unit_of(p[k]) for k in keys)
    vals = [
        u.ustrip(AllowValue, unit, p[k]) for k, unit in zip(keys, units, strict=True)
    ]
    return jnp.stack(vals, axis=-1), units  # ty: ignore[invalid-return-type]
