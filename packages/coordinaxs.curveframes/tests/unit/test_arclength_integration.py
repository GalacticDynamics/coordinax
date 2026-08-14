"""An arc-length curve is a curve: builders and charts take it unchanged."""

import jax.numpy as jnp
from coordinaxs.api.manifolds import metric_matrix

import coordinax.charts as cxc
import unxt as u

import coordinaxs.curveframes as cxfc


def helix(tau):
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), 0.3 * t]), "km")


def _chart(builder_cls):
    # NOTE the length unit: an arc-length curve's parameter IS a length.
    arc = cxfc.ArcLength(helix, "s")
    return cxfc.TubularChart(
        builder_cls(arc, "km"), tau_bounds=(u.Q(0.0, "km"), u.Q(5.0, "km"))
    )


def test_a_builder_accepts_an_arc_length_curve_unchanged() -> None:
    b = cxfc.BishopBuilder(cxfc.ArcLength(helix, "s"), "km")
    R = b.rotation_matrix(u.Q(1.0, "km"))
    assert jnp.allclose(jnp.linalg.norm(R[0]), 1.0, atol=1e-6)


def test_the_chart_round_trips_over_an_arc_length_curve() -> None:
    ch = _chart(cxfc.BishopBuilder)
    p = {"tau": u.Q(1.3, "km"), "n1": u.Q(0.1, "km"), "n2": u.Q(-0.05, "km")}
    xyz = cxc.pt_map(p, ch.M, ch, ch.M, cxc.cart3d)
    back = cxc.pt_map(xyz, ch.M, cxc.cart3d, ch.M, ch)
    assert jnp.allclose(back["tau"].ustrip("km"), 1.3, atol=1e-5)
    assert jnp.allclose(back["n1"].ustrip("km"), 0.1, atol=1e-5)


def test_unit_speed_collapses_g_ss_to_the_textbook_form() -> None:
    """The payoff PR #699 could not have: no speed factor in g_ss.

    For a unit-speed curve the Bishop metric is
    ``g = (1 - k1 n1 - k2 n2)^2 ds^2 + dn1^2 + dn2^2``, so with n1 = n2 = 0
    the ss-component is exactly 1 -- not the curve's speed squared.
    """
    ch = _chart(cxfc.BishopBuilder)
    at = {"tau": u.Q(1.3, "km"), "n1": u.Q(0.0, "km"), "n2": u.Q(0.0, "km")}
    g = metric_matrix(ch.M, at, ch).matrix
    assert jnp.allclose(g[0, 0].ustrip(""), 1.0, atol=1e-6), g[0, 0]
