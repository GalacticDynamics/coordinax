"""A station-pinned one-argument builder cannot parameterise a tubular chart.

Legitimate on the builder, degenerate as a chart -- so the chart refuses it and
the builder does not.
"""

import jax.numpy as jnp
import pytest

import unxt as u

import coordinaxs.curveframes as cxfc

BOUNDS = (u.Q(0.0, "s"), u.Q(3.0, "s"))
STATION = u.Q(0.5, "s")


def circle(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), "km")


def two_arg(s: u.AbstractQuantity, t: u.AbstractQuantity) -> u.AbstractQuantity:
    sv, tv = s.ustrip("s"), t.ustrip("s")
    return u.Q(
        jnp.stack([sv * (1.0 + 0.5 * tv), 0.1 * tv * sv**2, jnp.zeros_like(sv)]), "km"
    )


@pytest.mark.parametrize("builder", [cxfc.BishopBuilder, cxfc.FrenetSerretBuilder])
def test_a_chart_over_a_pinned_one_argument_builder_is_refused(builder) -> None:
    """Every `tau` mapped to `gamma(station)`, and nothing said so."""
    b = builder(circle, "s", station=STATION)
    with pytest.raises(ValueError, match="nothing left to vary"):
        cxfc.TubularChart(b, tau_bounds=BOUNDS)


@pytest.mark.parametrize("builder", [cxfc.BishopBuilder, cxfc.FrenetSerretBuilder])
def test_the_builder_itself_still_accepts_a_pinned_station(builder) -> None:
    """The frame field is the documented use and must survive the guard."""
    b = builder(circle, "s", station=STATION)
    # tau-independent by construction: two different call parameters, one frame.
    a = b.rotation_matrix(u.Q(1.0, "s"))
    c = b.rotation_matrix(u.Q(2.5, "s"))
    assert jnp.allclose(jnp.asarray(a), jnp.asarray(c))


def test_an_unpinned_one_argument_builder_still_charts() -> None:
    """The control: dropping `station=` is the remedy the message names."""
    ch = cxfc.TubularChart(cxfc.BishopBuilder(circle, "s"), tau_bounds=BOUNDS)
    assert ch.components == ("tau", "n1", "n2")


def test_a_worldtube_still_charts() -> None:
    """A station on a *two*-argument curve is required, not refused."""
    b = cxfc.BishopBuilder(two_arg, "s", station=STATION)
    ch = cxfc.TubularChart(b, tau_bounds=BOUNDS)
    assert ch.is_time_dependent
