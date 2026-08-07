"""Dtype strategies that agree with the active JAX x64 setting."""

__all__: tuple[str, ...] = ()

import functools as ft

from typing import Any

import hypothesis.strategies as st
import jax.numpy as jnp


@ft.cache
def _is_honoured(dtype: Any, /) -> bool:
    """Whether JAX produces ``dtype`` rather than silently narrowing it."""
    return jnp.empty(0, dtype=dtype).dtype == jnp.dtype(dtype)


def honoured_dtypes(dtypes: st.SearchStrategy[Any], /) -> st.SearchStrategy[Any]:
    """Drop dtypes JAX narrows under the active x64 setting.

    With ``jax_enable_x64`` off, ``jnp.asarray(x, dtype=float64)`` returns a
    float32 array and hypothesis rejects the mismatch with `InvalidArgument`.
    """
    return dtypes.filter(_is_honoured)
