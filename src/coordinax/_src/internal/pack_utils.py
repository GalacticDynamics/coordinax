"""Helpers for packing component dictionaries into arrays while tracking units.

`pack_uniform_unit` converts all entries of a component dict into one shared
unit and returns a stacked JAX array, preserving the ``None``-for-unitless
convention that keeps genuinely unitless components as raw arrays. For
non-uniform (per-component) unit packing into a ``QuantityMatrix``, use the
:func:`coordinaxs.api.charts.carray` dispatch.
"""

__all__ = ("pack_uniform_unit",)

from jaxtyping import ArrayLike
from typing import Any, Final, overload

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

from coordinax._src.custom_types import CDict, CKey

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
