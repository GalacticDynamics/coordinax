"""What a chart *declares* and what it *enforces* must be the same interval.

`coordinax.charts.component_domains` declares each component's legal interval;
`check_data` refuses values outside it at construction. Those were separate
statements of one fact -- core's bounds were written inline at the
`checks.polar_range` calls, and ``coordinaxs.hypothesis`` declared its own copy
so its strategies could generate valid points. #772 pinned the two equal with a
test; the declaration has since moved into core and both sides read it, so what
is left to check is that the enforcement really uses it.

Drift here is silent in the direction that matters: widen a bound and the
strategies keep generating the narrower range, so the new values are never
exercised; narrow one and they generate points core rejects, surfacing as
unrelated-looking failures in whatever test drew them.
"""

import math

import equinox as eqx
import pytest

import unxt as u

import coordinax.charts as cxc
from coordinax.charts import component_domains

#: (chart, component, expected bounds in radians). Written out rather than read
#: from `component_domains`, so that a wrong constant in that table cannot make
#: the test agree with it. Covers both spherical conventions, which is the case
#: the dispatch exists for.
CORE_BOUNDS = [
    (cxc.sph3d, "theta", (0.0, math.pi)),
    (cxc.lonlat_sph3d, "lat", (-math.pi / 2, math.pi / 2)),
    (cxc.loncoslat_sph3d, "lat", (-math.pi / 2, math.pi / 2)),
    (cxc.math_sph3d, "phi", (0.0, math.pi)),
    (cxc.sph2, "theta", (0.0, math.pi)),
    (cxc.lonlat_sph2, "lat", (-math.pi / 2, math.pi / 2)),
    (cxc.loncoslat_sph2, "lat", (-math.pi / 2, math.pi / 2)),
    (cxc.math_sph2, "phi", (0.0, math.pi)),
]


@pytest.mark.parametrize(("chart", "component", "bounds"), CORE_BOUNDS)
def test_declared_domain_is_the_physical_interval(chart, component, bounds) -> None:
    """The declared interval is the one the geometry calls for."""
    interval = component_domains(chart)[component]
    lo, hi = bounds
    got_lo, got_hi = (
        float(u.ustrip("rad", u.Q(v, interval.unit)))
        for v in (interval.min, interval.max)
    )
    assert math.isclose(got_lo, lo, abs_tol=1e-12)
    assert math.isclose(got_hi, hi, abs_tol=1e-12)


@pytest.mark.parametrize("side", ["below", "above"])
@pytest.mark.parametrize(("chart", "component", "bounds"), CORE_BOUNDS)
def test_core_rejects_outside_the_declared_domain(
    chart, component, bounds, side
) -> None:
    """A value past either bound must actually be refused at construction.

    Pins the direction the equality above cannot: that the declared numbers are
    the ones `check_data` enforces, not merely numbers it happens to store.

    Both sides, because they are separately enforceable. Probing only the
    upper one would miss core dropping the lower -- which for a latitude is
    the whole of ``-pi/2``, not a degenerate endpoint.
    """
    lo, hi = bounds
    point = {
        k: u.Angle(0.1, "rad") if d == "angle" else u.Q(1.0, "m")
        # `strict=True`: these are the chart's own parallel declarations, so a
        # length mismatch is a broken chart. Dropping components silently would
        # fail this test on a missing key rather than on the bound it is about.
        for k, d in zip(chart.components, chart.coord_dimensions, strict=True)
    }
    point[component] = u.Angle(lo - 0.5 if side == "below" else hi + 0.5, "rad")
    # The pair `tests/unit/charts/test_checks.py` uses: `eqx.error_if` raises
    # `EquinoxRuntimeError` from inside JIT and `ValueError` outside it.
    with pytest.raises(
        (eqx.EquinoxRuntimeError, ValueError), match="must be in the range"
    ):
        chart.check_data(point, keys=False, values=True)


def test_strategies_read_the_core_declaration() -> None:
    """``coordinaxs.hypothesis`` re-exports core's table rather than its own.

    The merge is only real while this holds -- a second table in the strategy
    package would put the drift straight back.
    """
    from coordinaxs.hypothesis.charts import component_domains as hyp_domains

    assert hyp_domains is component_domains
