"""The generated domains must agree with the bounds core actually enforces.

Core validates a chart's angular components at construction -- `Spherical3D`
rejects a polar angle outside ``[0, 180 deg]``, `LonLatSpherical3D` a latitude
outside ``[-90, 90 deg]``. `coordinaxs.hypothesis` separately declares
`POLAR`, `LATITUDE` and `AZIMUTH` so it can generate valid points.

Those are two statements of one fact, in two packages, with no link between
them. Drift is silent in the direction that matters: widen a bound in core and
the strategies keep generating a narrower range, so the new values are never
exercised; narrow one and the strategies generate points core rejects, which
surfaces as unrelated-looking failures in whatever test drew them.

This does not unify them -- that is a larger change, and core's validation is
on the construction path for every chart. It pins them equal so the drift is
caught rather than absorbed.
"""

import math

import pytest

import unxt as u

import coordinax.charts as cxc
from coordinaxs.hypothesis.charts import component_domains

#: (chart, component, expected bounds in radians). The bounds are those core
#: enforces, read off the `checks.polar_range` calls in the chart definitions.
CORE_BOUNDS = [
    (cxc.sph3d, "theta", (0.0, math.pi)),
    (cxc.lonlat_sph3d, "lat", (-math.pi / 2, math.pi / 2)),
    (cxc.math_sph3d, "phi", (0.0, math.pi)),
    (cxc.sph2, "theta", (0.0, math.pi)),
    (cxc.lonlat_sph2, "lat", (-math.pi / 2, math.pi / 2)),
    (cxc.math_sph2, "phi", (0.0, math.pi)),
]


@pytest.mark.parametrize(("chart", "component", "bounds"), CORE_BOUNDS)
def test_generated_domain_matches_core_bounds(chart, component, bounds) -> None:
    """The strategy's interval must be the interval core enforces."""
    interval = component_domains(chart)[component]
    lo, hi = bounds
    got_lo = u.ustrip("rad", u.Q(interval.min, interval.unit))
    got_hi = u.ustrip("rad", u.Q(interval.max, interval.unit))
    assert math.isclose(float(got_lo), lo, abs_tol=1e-12)
    assert math.isclose(float(got_hi), hi, abs_tol=1e-12)


@pytest.mark.parametrize(("chart", "component", "bounds"), CORE_BOUNDS)
def test_core_rejects_outside_the_generated_domain(chart, component, bounds) -> None:
    """A value past the declared bound must actually be refused by core.

    Pins the direction the equality above cannot: that the numbers are the
    bounds core *enforces*, not merely numbers both sides happen to store.
    """
    _lo, hi = bounds
    point = {
        k: u.Angle(0.1, "rad") if d == "angle" else u.Q(1.0, "m")
        for k, d in zip(chart.components, chart.coord_dimensions, strict=False)
    }
    point[component] = u.Angle(hi + 0.5, "rad")
    with pytest.raises(Exception, match=r"(?i)must be|range|between"):
        chart.check_data(point, keys=False, values=True)
