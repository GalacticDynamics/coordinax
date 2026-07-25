"""Core operator API functions.

This module defines helpers for operator implementations.
"""

__all__: tuple[str, ...] = (
    "ComposedR",
    "Neg",
    "is_componentwise_offset",
    "is_flat_chart",
)

import dataclasses

from typing import Any, final

import jax.numpy as jnp
import jax.tree as jtu

import coordinax.representations as cxr
from coordinax._src.exceptions import NoGlobalCartesianChartError


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


@final
@dataclasses.dataclass(slots=True)
class ComposedR:
    """Matrix product of two rotation parameters, ``second @ first``.

    Each of ``first`` / ``second`` is either a constant matrix or a callable of
    ``tau``. Evaluating at ``tau`` returns ``eval(second, tau) @ eval(first,
    tau)``, i.e. the matrix that applies ``first`` and then ``second`` — the
    time-dependent generalization of ``Rotate.__matmul__`` for composed
    rotations.
    """

    first: Any
    """The rotation applied first (an array, or a callable of ``tau``)."""

    second: Any
    """The rotation applied second (an array, or a callable of ``tau``)."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Evaluate the composed matrix at ``tau``: ``second(tau) @ first(tau)``."""
        a = self.first(*args, **kwargs) if callable(self.first) else self.first
        b = self.second(*args, **kwargs) if callable(self.second) else self.second
        return b @ a


def is_componentwise_offset(op: Any, chart: Any, /) -> bool:
    """Whether an additive offset acts componentwise on data in ``chart``.

    True for fibre-only offsets (ladder order k >= 1 — their point action is
    the identity, so the componentwise rule is definitional), and for k = 0
    offsets whose ``delta`` and data share the same Cartesian-type (flat)
    chart (a true ambient translation). Everything else is base-point
    dependent and must go through the generic engine.

    This is THE routing predicate for the additive family — `act`,
    `pushforward`, and `prolong` must all use it so the fast paths stay
    provably consistent with the generic prolongation.
    """
    k = getattr(op, "semantic_kind", cxr.dpl).order
    return k != 0 or (chart == op.chart and is_flat_chart(chart))
