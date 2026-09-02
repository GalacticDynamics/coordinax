"""What a chart *declares* and what it *enforces* must be the same interval.

`coordinax.charts.component_domains` declares each component's legal interval;
`check_data` refuses values outside it at construction. Both must be the same
numbers, so what is checked here is that the declared bounds are the enforced
ones -- and, separately, that they are the intervals the geometry calls for.

Drift is silent in the direction that matters, because the declaration also
drives the ``coordinaxs.hypothesis`` strategies: widen a bound and they keep
generating the narrower range, so the new values are never exercised; narrow
one and they generate points core rejects, surfacing as unrelated-looking
failures in whatever test drew them.
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

    One declaration only holds while this does -- a second table in the
    strategy package would reintroduce the drift.
    """
    from coordinaxs.hypothesis.charts import component_domains as hyp_domains

    assert hyp_domains is component_domains


def test_prolate_spheroidal_declares_its_focal_bounds() -> None:
    """The one chart whose domain follows from a parameter, not just a type.

    `check_data` enforces ``mu >= Delta^2`` and ``|nu| <= Delta^2``. A lookup
    keyed on the class alone cannot say that, so the declaration has to read
    `Delta` off the instance -- and must track it, not a fixed number.
    """
    for delta, bound in ((2.0, 4.0), (3.0, 9.0)):
        chart = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(delta, "kpc"))
        domains = component_domains(chart)
        assert domains["mu"].bounds_in(u.unit("kpc2")) == (bound, None)
        assert domains["nu"].bounds_in(u.unit("kpc2")) == (-bound, bound)


@pytest.mark.parametrize(("component", "value"), [("mu", 1.0), ("nu", 9.0)])
def test_prolate_spheroidal_rejects_outside_its_focal_bounds(component, value) -> None:
    """A value the declaration excludes is refused, as for the fixed domains.

    ``Delta = 2 kpc`` puts both bounds at ``4 kpc2``, so ``mu = 1`` is under
    its floor and ``nu = 9`` over its ceiling.
    """
    chart = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, "kpc"))
    point = {"mu": u.Q(5.0, "kpc2"), "nu": u.Q(0.1, "kpc2"), "phi": u.Angle(0.1, "rad")}
    point[component] = u.Q(value, "kpc2")
    with pytest.raises((eqx.EquinoxRuntimeError, ValueError)):
        chart.check_data(point, keys=False, values=True)


def test_every_enforced_bound_is_declared() -> None:
    """No chart may enforce a bound its domain does not mention.

    The invariant the module docstring states, checked rather than asserted:
    enforcement is a *subset* of the declaration. `ProlateSpheroidal3D` broke
    it -- `check_data` refused values its all-`FREE` domain called legal --
    and only a per-chart audit like this one catches the next such chart.

    Approximated by construction: a chart that enforces a bound rejects
    *something*, so a chart whose every component is declared `FREE` must
    accept every dimensionally-valid point.
    """
    unconstrained = [
        chart
        for chart in (cxc.cart3d, cxc.cart2d, cxc.cart1d, cxc.cyl3d, cxc.polar2d)
        if all(iv.unit is None for iv in component_domains(chart).values())
    ]
    for chart in unconstrained:
        point = {
            k: u.Angle(4.0, "rad") if d == "angle" else u.Q(-7.0, "m")
            for k, d in zip(chart.components, chart.coord_dimensions, strict=True)
        }
        # Values well outside every interval this module declares; a chart
        # declaring nothing must take them.
        chart.check_data(point, keys=False, values=True)


@pytest.mark.parametrize(
    ("delta", "reason"),
    [
        (u.StaticQuantity(2.0, "s"), "not convertible to the components' area"),
        (u.StaticQuantity(1e200, "m"), "squares past the float ceiling"),
    ],
)
def test_prolate_spheroidal_declares_nothing_for_an_unusable_delta(
    delta, reason
) -> None:
    """`Delta` is only constrained positive and scalar, so it can be unusable.

    A non-length `Delta` gives a bound in a unit `mu` cannot be compared
    against, and an enormous one squares to infinity. Neither is a bound, so
    neither is declared -- the alternative is a domain a caller cannot convert
    or a generator cannot draw from.
    """
    domains = component_domains(cxc.ProlateSpheroidal3D(Delta=delta))
    assert domains["mu"].unit is None, reason
    assert domains["nu"].unit is None, reason
