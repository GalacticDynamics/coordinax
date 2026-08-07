"""Core operator API functions.

This module defines helpers for operator implementations.
"""

__all__: tuple[str, ...] = (
    "is_componentwise_offset",
    "is_flat_chart",
    "require_matching_keys",
)

from collections.abc import Iterable
from typing import Any

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


def is_componentwise_offset(op: Any, chart: Any, /) -> bool:
    """Whether an additive offset acts componentwise on data in ``chart``.

    True for fibre-only offsets (ladder order k >= 1 — their point action is
    the identity, so the componentwise rule is definitional), and for k = 0
    offsets whose ``delta`` and data share the same Cartesian-type (flat)
    chart (a true ambient translation). Everything else is base-point
    dependent and must go through the generic engine.

    This is THE routing predicate for the additive family — `act`,
    `pushforward`, and `act_jet` must all use it so the fast paths stay
    provably consistent with the generic prolongation.
    """
    k = getattr(op, "semantic_kind", cxr.dpl).order
    return k != 0 or (chart == op.chart and is_flat_chart(chart))


def require_matching_keys(
    actual: Iterable[str], expected: Iterable[str], message: str, /
) -> None:
    """Raise ``TypeError(message + missing/unexpected keys)`` if keys differ.

    Shared by `act`, `pushforward`, and `prolong` so component-mismatch errors
    report the missing and unexpected keys in one consistent format.
    """
    got, exp = set(actual), set(expected)
    if got != exp:
        miss, extra = sorted(exp - got), sorted(got - exp)
        raise TypeError(
            message
            + (f"; missing {miss}" if miss else "")
            + (f"; unexpected {extra}" if extra else "")
            + "."
        )
