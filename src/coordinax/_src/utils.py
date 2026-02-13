"""Representation of coordinates in different systems."""

from unxt.quantity import AllowValue

__all__: tuple[str, ...] = (
    "uconvert_to_rad",
    "ustrip_value",
    "pack_uniform_unit",
    "unpack_with_unit",
)

from jaxtyping import ArrayLike
from typing import Any, Final, TypeVar, overload

import jax.tree as jtu

import quaxed.numpy as jnp
import unxt as u
from unxt import AbstractQuantity as ABCQ  # noqa: N814
from unxt.quantity import AllowValue

from coordinax.api import CDict, ComponentsKey, CsDict

V = TypeVar("V", bound=ABCQ | ArrayLike)

RAD: Final = u.unit("rad")


def uconvert_to_rad(value: V, usys: u.AbstractUnitSystem | None, /) -> V:
    """Convert an angle value to radians, handling no-usys case."""
    return u.uconvert_value(RAD, RAD if usys is None else usys["angle"], value)


def ustrip_value(
    uto: u.AbstractUnit,
    usysfrom: u.AbstractUnitSystem | None,
    dfrom: str,
    x: u.AbstractQuantity | ArrayLike,
    /,
) -> ArrayLike:
    if not isinstance(x, u.AbstractQuantity):
        if usysfrom is None:
            msg = "Unit system must be provided."
            raise ValueError(msg)
        ufrom = usysfrom[dfrom]
    else:
        ufrom = x.unit
    return u.ustrip(AllowValue, u.uconvert_value(uto, ufrom, x))


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
