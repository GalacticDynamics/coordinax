"""Hypothesis strategies for Distance quantities."""

__all__ = ("make_nonnegative",)


from collections.abc import Mapping
from typing import Any, assert_never

import jax.numpy as jnp
from hypothesis import strategies as st

from coordinax_hypothesis._src.namespace import xps


def make_nonnegative(draw: st.DrawFn, /, **kwargs: Any) -> dict[str, Any]:
    """Adjust kwargs to ensure non-negative values in generated quantities.

    This helper function modifies the `elements` parameter in kwargs to ensure
    that generated values are non-negative. It handles three cases:

    1. If `elements` is a Mapping (dict), updates `min_value` to be at least 0
    2. If `elements` is a SearchStrategy, applies absolute value mapping
    3. If `elements` is not provided, creates a default strategy with min_value=0

    Parameters
    ----------
    draw
        Hypothesis draw function for drawing from strategies.
    **kwargs
        Keyword arguments that will be passed to `unxt_hypothesis.quantities`.
        The `elements` parameter will be modified if present, or created if absent.

    Returns
    -------
    dict[str, Any]
        Modified kwargs with adjusted `elements` parameter to ensure non-negative
        values.

    Examples
    --------
    >>> from hypothesis import strategies as st
    >>> import jax.numpy as jnp
    >>> # With custom elements
    >>> kwargs = {"elements": {"min_value": -10, "max_value": 100}}
    >>> # modified = make_nonnegative(draw, **kwargs)
    >>> # modified["elements"]["min_value"] will be 0

    >>> # Without elements, creates default non-negative strategy
    >>> kwargs = {"dtype": jnp.float32}
    >>> # modified = make_nonnegative(draw, **kwargs)
    >>> # modified will have "elements" key with min_value=0

    """
    if "elements" in kwargs:
        # User provided elements strategy - need to check if it's a float
        # strategy and adjust min_value if necessary
        elements: Mapping[str, Any] | st.SearchStrategy = kwargs["elements"]
        if isinstance(elements, Mapping):
            elements = dict(elements)
            elements["min_value"] = max(0.0, elements.get("min_value", 0.0))
        elif isinstance(elements, st.SearchStrategy):
            elements = elements.map(abs)  # Simple way to ensure non-negative
        else:
            assert_never(elements)

        kwargs["elements"] = elements

    else:
        # No elements provided, we need to set a default with min_value=0
        # Get dtype if specified, otherwise use default
        dtype = kwargs.get("dtype")
        if dtype is not None:
            dtype = draw(dtype) if isinstance(dtype, st.SearchStrategy) else dtype
        else:
            dtype = jnp.float32  # Default dtype

        # Create elements strategy with min_value=0
        kwargs["elements"] = xps.from_dtype(dtype, min_value=0.0)  # TODO: other kwargs?

    return kwargs
