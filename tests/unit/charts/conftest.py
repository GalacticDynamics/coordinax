"""Hypothesis strategies for coordinax charts."""

from typing import Final

import hypothesis.strategies as st
import jax.numpy as jnp
from hypothesis.extra.array_api import make_strategies_namespace

import coordinax.charts as cxc

xps = make_strategies_namespace(jnp)

SHAPE_CART_MAP: Final[dict[int, cxc.AbstractChart]] = {
    1: cxc.cart1d,
    2: cxc.cart2d,
    3: cxc.cart3d,
}


@st.composite
def shapes_ending_in_123(draw, *, max_dims: int = 5) -> tuple[int, ...]:
    """Generate arbitrary shapes whose last dimension is 1, 2, or 3.

    - Total rank is in [1, max_dims]
    - Last axis is one of {1, 2, 3}
    """
    last = draw(st.sampled_from((1, 2, 3)))
    return draw(xps.array_shapes(max_dims=max_dims - 1).map(lambda s: (*s, last)))
