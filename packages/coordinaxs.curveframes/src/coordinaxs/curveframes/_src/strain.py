r"""How fast a slice's geometry deforms.

The extrinsic-curvature slot of the ADM decomposition,
$K_{ij} = \tfrac{1}{2\alpha}\left(\partial_t\gamma_{ij} - (\mathcal{L}_\beta
\gamma)_{ij}\right)$, reduces here to $\tfrac12\partial_t\gamma_{ij}$: the
lapse is identically 1 under absolute time, and the Lie-drag term vanishes
because a chart coordinate *is* what the time derivative holds fixed, so the
coordinate shift is zero by construction. See
`coordinaxs.curveframes.velocity` for the other ADM piece, which is the
ambient motion of the frame origin and does not vanish.
"""

__all__ = ("rate_of_strain",)

from collections.abc import Callable
from typing import Any, cast

import jax
import unxts.linalg as ul

import unxt as u
from coordinaxs.api.manifolds import metric_matrix

_MSG_BARE_TIME = (
    "`rate_of_strain` differentiates with respect to `t`, so `t` must carry a "
    "unit: nothing else states what the rate is per. Pass a `Quantity`, e.g. "
    "`u.Q(1.0, 's')`."
)


def rate_of_strain(
    chart_at_time: Callable[[Any], Any], point: dict, t: Any, /
) -> ul.QuantityMatrix:
    r"""Return $K_{ij} = \tfrac12\,\partial_t\gamma_{ij}$ at ``point``.

    Parameters
    ----------
    chart_at_time
        The slice family: given a time, the chart of the tube at that time.
        Typically ``lambda t: TubularChart(BishopBuilder(AtTime(curve, t), ...))``.
    point
        Chart coordinates at which to evaluate, e.g.
        ``{"tau": ..., "n1": ..., "n2": ...}``.
    t
        The time to differentiate at. Must carry a unit.

    Notes
    -----
    The value depends on the *labelling*, and correctly so. A material
    parametrisation shows the tube genuinely stretching; wrapping the same
    curve in `ArcLength` holds the metric near unit-speed, so its rate of
    strain is near zero. That is the same choice `velocity` reports, made once
    by the caller's curve -- see the `ArcLength` docs.

    Takes the family rather than one chart because $\partial_t$ needs
    neighbouring slices, and a `TubularChart` is a single one.

    """
    t_unit = u.unit_of(t)
    if t_unit is None:
        raise TypeError(_MSG_BARE_TIME)

    def gamma(t_val: Any) -> Any:
        chart = chart_at_time(u.Q(t_val, t_unit))
        return cast("Any", metric_matrix(chart.M, point, chart)).matrix.value

    here = chart_at_time(t)
    unit = cast("Any", metric_matrix(here.M, point, here)).matrix.unit / t_unit

    d_gamma = jax.jacfwd(gamma)(t.ustrip(t_unit))
    return ul.QuantityMatrix(0.5 * d_gamma, unit=unit)
