"""`TubularChart` is generatable by `coordinaxs.hypothesis`, and usable once drawn.

`TubularChart` is the first chart class defined outside core `coordinax`, and
by default nothing tells `coordinaxs.hypothesis` how to draw one -- its
`builder: AbstractCurveFrameBuilder` and `tau_bounds: tuple[Any, Any]` fields
have no annotation-derived strategy (see
`coordinaxs.hypothesis.curveframes._tubular`). These tests exercise the fix
from the consumer side: `coordinaxs.hypothesis.main.charts()`, not the
strategy module directly.
"""

import coordinaxs.hypothesis.main as cxst
import hypothesis.strategies as st
import jax.numpy as jnp
from coordinaxs.hypothesis.charts import Interval, component_domains
from hypothesis import HealthCheck, given, settings

import coordinax.charts as cxc

import coordinaxs.curveframes as cxfc


@given(chart_cls=cxst.chart_classes(filter=cxfc.TubularChart))
def test_tubular_chart_is_a_registered_chart_class(chart_cls) -> None:
    """`TubularChart` is enumerable, not just constructible by hand."""
    assert chart_cls is cxfc.TubularChart


@given(chart=cxst.charts(filter=cxfc.TubularChart))
def test_charts_can_draw_a_tubular_chart(chart) -> None:
    """`charts(filter=TubularChart)` draws a usable instance, not just a class."""
    assert isinstance(chart, cxfc.TubularChart)
    assert chart.components == ("tau", "n1", "n2")
    assert chart.ndim == 3


@settings(deadline=None)  # the inverse solve recompiles per draw (new curve/builder)
@given(chart=cxst.charts(filter=cxfc.TubularChart))
def test_a_drawn_tubular_chart_round_trips_a_point(chart) -> None:
    """A drawn chart is not just constructible -- it can carry a point.

    The midpoint of `tau_bounds` is interior to the range for every curve in
    the strategy's fixed set (never on the closed curve's seam), and a point
    *on* the curve (n1 = n2 = 0) is always inside the reach whatever the
    builder or curve, so this needs no per-draw reach reasoning.
    """
    lo, hi = chart.tau_bounds
    tau = 0.5 * (lo + hi)
    on_curve = chart.builder.location(tau)
    data = {k: on_curve[i] for i, k in enumerate(("x", "y", "z"))}

    back = cxc.pt_map(data, chart.M, cxc.cart3d, chart.M, chart)

    assert jnp.allclose(
        back["tau"].ustrip(chart.builder.tau_unit),
        tau.ustrip(chart.builder.tau_unit),
        atol=1e-5,
    )
    assert jnp.allclose(back["n1"].ustrip("km"), 0.0, atol=1e-5)
    assert jnp.allclose(back["n2"].ustrip("km"), 0.0, atol=1e-5)


@given(chart=cxst.charts(filter=cxfc.TubularChart))
def test_component_domains_are_free_for_every_component(chart) -> None:
    """`TubularChart`'s own `component_domains` coverage.

    Mirror of `coordinaxs.hypothesis`'s `test_every_concrete_chart_is_covered`,
    which is scoped to core `coordinax`'s own charts and so cannot see
    `TubularChart` (see that test's docstring) -- this is this package's half
    of the split.

    No `component_domains` overload is registered for `TubularChart`, so it
    falls through to the generic `AbstractChart` rule at `domains.py:124`:
    unconstrained (`Interval()`, i.e. `FREE`) for every component. That is the
    sensible answer here, not just the default one: `n1`/`n2`'s legal range is
    the curve's local reach, which depends on curvature at a *point*, not a
    fixed constant that could hold for every curve a `TubularChart` might
    wrap -- there is no tighter domain to state.
    """
    domains = component_domains(chart)
    assert set(domains) == set(chart.components) == {"tau", "n1", "n2"}
    assert all(interval == Interval() for interval in domains.values())


@settings(deadline=None, suppress_health_check=list(HealthCheck))
@given(chart=cxst.charts(filter=cxfc.TubularChart), data=st.data())
def test_drawn_points_fall_inside_the_component_domains(chart, data) -> None:
    """The payoff, mirroring `TestCDictsRespectsDomains` in `coordinaxs.hypothesis`.

    Vacuous in the sense that `FREE` accepts anything finite -- the domain
    puts no constraint on `n1`/`n2` (see the test above) -- but it still
    catches a real regression: a `component_domains` overload added later for
    `TubularChart` that disagrees with `cdicts`, or a component renamed in one
    place and not the other.
    """
    domains = component_domains(chart)
    point = data.draw(cxst.cdicts(chart))
    for name, q in point.items():
        interval = domains[name]
        assert interval.unit is None  # FREE: nothing to check the value against
        assert jnp.isfinite(q.ustrip(q.unit))
