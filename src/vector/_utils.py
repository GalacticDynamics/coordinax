"""Representation of coordinates in different systems."""

__all__: list[str] = []

from collections.abc import Iterator
from dataclasses import fields, replace
from functools import singledispatch
from typing import TYPE_CHECKING, Any

import array_api_jax_compat as xp
import astropy.units as u
import jax.numpy as jnp
from jax.dtypes import canonicalize_dtype
from jax_quantity import Quantity
from jaxtyping import Array, Float, Shaped

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


@singledispatch
def converter_quantity_array(x: Any, /) -> Float[Quantity, "*shape"]:
    """Convert to a batched vector."""
    msg = f"Cannot convert {type(x)} to a batched Quantity."
    raise NotImplementedError(msg)


@converter_quantity_array.register
def _convert_quantity(x: Quantity, /) -> Float[Quantity, "*shape"]:
    """Convert to a batched vector."""
    dtype = jnp.promote_types(x.dtype, canonicalize_dtype(float))
    return xp.asarray(x, dtype=dtype)


@converter_quantity_array.register(jnp.ndarray)
def _convert_jax_array(x: Shaped[Array, "*shape"], /) -> Float[Quantity, "*shape"]:
    """Convert to a batched vector."""
    dtype = jnp.promote_types(x.dtype, canonicalize_dtype(float))
    x = xp.asarray(x, dtype=dtype)
    return Quantity(x, u.one)


# ============================================================================


def dataclass_values(obj: "DataclassInstance") -> Iterator[Any]:
    """Return the values of a dataclass instance."""
    yield from (getattr(obj, f.name) for f in fields(obj))


def dataclass_items(obj: "DataclassInstance") -> Iterator[tuple[str, Any]]:
    """Return the field names and values of a dataclass instance."""
    yield from ((f.name, getattr(obj, f.name)) for f in fields(obj))


def full_shaped(obj: "AbstractVectorBase", /) -> "AbstractVectorBase":
    """Return the vector, fully broadcasting all components."""
    arrays = xp.broadcast_arrays(*dataclass_values(obj))
    return replace(obj, **dict(zip(obj.components, arrays, strict=True)))
