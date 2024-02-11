"""Representation of coordinates in different systems."""

__all__: list[str] = []

from functools import singledispatch
from typing import Any

import array_api_jax_compat as xp
import astropy.units as u
import jax.numpy as jnp
from jax_quantity import Quantity
from jaxtyping import Array, Float, Shaped


@singledispatch
def converter_quantity_array(x: Any, /) -> Float[Quantity, "*shape"]:
    """Convert to a batched vector."""
    msg = f"Cannot convert {type(x)} to a batched Quantity."
    raise NotImplementedError(msg)


@converter_quantity_array.register
def _convert_quantity(x: Quantity, /) -> Float[Quantity, "*shape"]:
    """Convert to a batched vector."""
    return xp.asarray(x, dtype=jnp.promote_types(x.dtype, float))


@converter_quantity_array.register
def _convert_jax_array(x: Shaped[Array, "*shape"], /) -> Float[Quantity, "*shape"]:
    """Convert to a batched vector."""
    x = xp.asarray(x, dtype=jnp.promote_types(x.dtype, float))
    return Quantity(x, u.one)
