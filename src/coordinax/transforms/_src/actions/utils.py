"""Core operator API functions.

This module defines helpers for operator implementations.
"""

__all__: tuple[str, ...] = ("Neg", "is_flat_chart")

import dataclasses

from typing import Any, final

import jax.numpy as jnp
import jax.tree as jtu


def is_flat_chart(chart: Any, /) -> bool:
    """Whether ``chart`` is a Cartesian-type chart (its own canonical Cartesian).

    In such charts a componentwise offset IS a translation of the flat
    ambient space (Jacobian = identity, no base-point dependence). In any
    other chart an offset must be pushed through the chart Jacobian at the
    point, so additive fast paths do not apply.

    A chart with no global Cartesian chart (e.g. ``PoincarePolar6D``) is not
    flat: this predicate returns `False` rather than propagating
    `~coordinax.charts.NoGlobalCartesianChartError`.
    """
    from coordinax._src.exceptions import (  # noqa: PLC0415 - avoid cycle
        NoGlobalCartesianChartError,
    )

    try:
        cart = chart.cartesian
    except NoGlobalCartesianChartError:
        return False
    return isinstance(chart, type(cart))


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
