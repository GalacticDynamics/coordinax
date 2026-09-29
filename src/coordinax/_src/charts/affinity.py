r"""Whether a chart transition is affine.

A transition $\psi$ between two charts is *affine* exactly when
$\partial^2\psi \equiv 0$, so the Jacobian pushforward is the complete
transformation law at every order rather than only at order 1. That is the
condition under which an acceleration may be carried between charts without
its velocity: the curvature term $\partial^2\psi(v, v)$ is identically zero
and there is nothing for the lower slot to feed (see gh#936).

Affinity cannot be read off the class hierarchy. `Spherical3D`,
`MathSpherical3D` and `LonLatSpherical3D` are the same parameterisation
written three ways -- $\mathrm{lat} = \pi/2 - \theta$ and friends -- so their
transitions are affine, while `LonCosLatSpherical3D` sits under the same
`AbstractSpherical3D` base and is *not*: its ``lon_coslat`` carries a
$\cos(\mathrm{lat})$ factor, which makes the Jacobian base-point dependent
like any other curvilinear map. The relationship is a fact about a *pair*,
and has to be declared. The concrete declarations live next to the charts
they are about -- see `register_affinity` and `coordinax._src.spherical`.
"""

__all__: tuple[str, ...] = ("is_flat_chart",)

from typing import Any

import plum

from coordinax._src.base.charts import AbstractChart
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


@plum.dispatch  # type: ignore[no-redef]
def is_affine_transition(
    from_chart: AbstractChart,  # type: ignore[type-arg]
    to_chart: AbstractChart,  # type: ignore[type-arg]
    /,
) -> bool:
    """Conservative default: assume the transition curves.

    A chart with itself is the identity, and two Cartesian-type charts differ
    by at most a linear relabelling of flat space. Everything else must say so
    for itself, because a wrong `True` here is silent -- an order-2 fibre would
    take the cheap path and lose its curvature term -- while a wrong `False`
    only costs a prolongation that was not strictly needed.
    """
    if from_chart == to_chart:
        return True
    return is_flat_chart(from_chart) and is_flat_chart(to_chart)
