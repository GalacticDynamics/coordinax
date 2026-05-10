"""Core operator API functions.

This module defines helpers for operator implementations.
"""

__all__: tuple[str, ...] = ("Neg",)

import dataclasses

from typing import Any, final

import jax.numpy as jnp
import jax.tree as jtu


@final
@dataclasses.dataclass(slots=True)
class Neg:
    """A parameter that negates another parameter."""

    param: Any
    """The parameter to negate."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the parameter and negate the result."""
        return jtu.map(jnp.negative, self.param(*args, **kwargs))

    def __neg__(self) -> Any:
        """Return the original parameter."""
        return self.param
