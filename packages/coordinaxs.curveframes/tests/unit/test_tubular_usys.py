"""The raw-array route through `TubularChart`'s `pt_map`.

`pt_map` accepts three shapes of coordinate throughout `coordinax`: a
`unxt.Quantity`, a raw array interpreted through a `usys`, and mixtures of
the two. `TubularChart` advertised the same signature but opened its body
with ``del usys`` and then called ``p["n1"].ustrip(...)``, so a raw
coordinate died on the missing method rather than being interpreted.

Raw in gives raw out, matching every other chart: a caller who came in on
the cheap route wants to leave on it.
"""

import jax.numpy as jnp
import pytest

import coordinax.charts as cxc
import unxt as u

import coordinaxs.curveframes as cxfc

USYS = u.unitsystem("km", "s", "kg", "rad")


def circle(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), "km")


@pytest.fixture
def chart() -> cxfc.TubularChart:
    return cxfc.TubularChart(
        cxfc.BishopBuilder(circle, "s"),
        tau_bounds=(u.Q(0.0, "s"), u.Q(2 * jnp.pi, "s")),
    )


def test_the_raw_route_agrees_with_the_quantity_route(chart) -> None:
    """Same numbers either way; only the wrapper differs."""
    pq = {"tau": u.Q(0.5, "s"), "n1": u.Q(0.1, "km"), "n2": u.Q(0.0, "km")}
    pa = {"tau": jnp.asarray(0.5), "n1": jnp.asarray(0.1), "n2": jnp.asarray(0.0)}

    fq = cxc.pt_map(pq, chart.M, chart, chart.M, cxc.cart3d)
    fa = cxc.pt_map(pa, chart.M, chart, chart.M, cxc.cart3d, usys=USYS)

    for k in ("x", "y", "z"):
        assert u.unit_of(fa[k]) is None, f"{k} came back wrapped"
        assert jnp.allclose(fq[k].ustrip("km"), fa[k]), k


def test_the_raw_route_round_trips(chart) -> None:
    """Cart3D -> Tubular -> Cart3D returns the coordinates it was given."""
    pa = {"tau": jnp.asarray(0.5), "n1": jnp.asarray(0.1), "n2": jnp.asarray(0.0)}
    fwd = cxc.pt_map(pa, chart.M, chart, chart.M, cxc.cart3d, usys=USYS)
    back = cxc.pt_map(fwd, chart.M, cxc.cart3d, chart.M, chart, usys=USYS)

    for k, want in pa.items():
        assert u.unit_of(back[k]) is None, f"{k} came back wrapped"
        assert jnp.allclose(back[k], want, atol=1e-5), (k, back[k], want)


def test_a_raw_coordinate_without_a_usys_says_what_is_missing(chart) -> None:
    """The one unserveable combination, named rather than crashed into.

    Without a `usys` the numbers mean nothing, and before this the failure was
    an `AttributeError` on a missing `ustrip`, several frames down.
    """
    pa = {"tau": jnp.asarray(0.5), "n1": jnp.asarray(0.1), "n2": jnp.asarray(0.0)}
    with pytest.raises(ValueError, match="needs a `usys=`"):
        cxc.pt_map(pa, chart.M, chart, chart.M, cxc.cart3d)


def test_quantities_and_raw_values_can_be_mixed(chart) -> None:
    """A `Quantity` states its own unit, so it is passed through untouched.

    Which means a partly-wrapped dict works, and a `Quantity` in a unit the
    `usys` does not use is still honoured -- 100 m rather than 0.1 km.
    """
    mixed = {"tau": jnp.asarray(0.5), "n1": u.Q(100.0, "m"), "n2": jnp.asarray(0.0)}
    allraw = {"tau": jnp.asarray(0.5), "n1": jnp.asarray(0.1), "n2": jnp.asarray(0.0)}

    got = cxc.pt_map(mixed, chart.M, chart, chart.M, cxc.cart3d, usys=USYS)
    want = cxc.pt_map(allraw, chart.M, chart, chart.M, cxc.cart3d, usys=USYS)
    for k in ("x", "y", "z"):
        assert jnp.allclose(got[k], want[k]), k


def test_raw_tau_bounds_are_where_the_raw_route_stops() -> None:
    """`tau_bounds` states the chart's own tau range, so it must carry a unit.

    A `usys` interprets the coordinates of a *call*; `tau_bounds` is
    structural -- read by `coord_dimensions` with no call in sight, so there
    is no `usys` in scope to consult. The raw route stops here, with the
    builder's own message rather than an `AttributeError`.

    Worth pinning now that raw coordinates work everywhere else in this
    chart: passing raw bounds too is the natural next assumption.

    `nearest_tau` is deliberately not exercised here -- its ``bounds`` is
    annotated `tuple[AbstractQuantity, AbstractQuantity]`, so beartype
    rejects a raw pair before any of this runs. `TubularChart.tau_bounds` is
    `tuple[Any, Any]`, which is why the chart is the path that reaches it.
    """
    chart = cxfc.TubularChart(
        cxfc.BishopBuilder(circle),  # inferring, so nothing declared
        tau_bounds=(0.0, 2 * jnp.pi),
    )
    with pytest.raises(TypeError, match="carries no unit"):
        _ = chart.coord_dimensions

    # Declaring the unit is the way through, exactly as for a raw parameter.
    declared = cxfc.TubularChart(
        cxfc.BishopBuilder(circle, "s"), tau_bounds=(0.0, 2 * jnp.pi)
    )
    assert declared.coord_dimensions == ("time", "length", "length")
