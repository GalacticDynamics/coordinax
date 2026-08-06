"""Dtype strategies that agree with the active JAX x64 setting.

`hypothesis.extra.array_api.make_strategies_namespace` reads the dtypes a
namespace *declares*, and ``jax.numpy`` declares the 64-bit ones unconditionally.
With ``jax_enable_x64`` off -- the JAX default -- they are not honoured:
``jnp.asarray(x, dtype=jnp.float64)`` silently returns a **float32** array. So
hypothesis draws elements from the float64 range, builds the array, finds it
came back float32, and raises `InvalidArgument`.

That is a hard error, not a discard, so it fails the test outright. It does not
bite coordinax's own suite, which sets ``JAX_ENABLE_X64=1`` in ``pyproject.toml``,
but it does bite any downstream project importing these strategies without that
setting: measured 43 failures in 200 draws of ``charts()``.

The fix is to offer only dtypes that survive a round-trip through ``jnp``, which
makes the strategies correct under either setting rather than only under x64.
"""

__all__: tuple[str, ...] = ()

import functools as ft

from typing import Any

import hypothesis.strategies as st
import jax.numpy as jnp


@ft.cache
def _is_honoured(dtype: Any, /) -> bool:
    """Whether JAX produces ``dtype`` rather than silently narrowing it."""
    return jnp.empty(0, dtype=dtype).dtype == jnp.dtype(dtype)


def jax_honoured(dtypes: st.SearchStrategy[Any], /) -> st.SearchStrategy[Any]:
    """Restrict a dtype strategy to dtypes JAX honours under the active setting.

    With ``jax_enable_x64`` on this filters nothing; with it off it drops the
    64-bit dtypes that would otherwise produce arrays hypothesis rejects.
    """
    return dtypes.filter(_is_honoured)
