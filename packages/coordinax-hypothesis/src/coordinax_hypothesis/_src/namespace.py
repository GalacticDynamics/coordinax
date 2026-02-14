__all__ = ("xps",)

import jax.numpy as jnp
from hypothesis.extra.array_api import make_strategies_namespace

xps = make_strategies_namespace(jnp)
