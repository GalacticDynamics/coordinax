"""`TubularChart` is a chart, on the parameterized branch."""

import jax
import jax.numpy as jnp
import pytest

import coordinax.charts as cxc
import unxt as u
from coordinax._src.base.charts import AbstractParameterizedChart

import coordinaxs.curveframes as cxfc


def circle(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), "km")


# Exactly ONE period of the circle. Bounds spanning more than a period make
# the inverse genuinely ambiguous: gamma(tau) and gamma(tau + 2*pi) are the
# same point, so the nearest-point solve faces an exact tie (measured:
# |gamma(0.7) - gamma(0.7 + 2*pi)| = 0.0). That is a property of closed
# curves, not a solver defect.
BOUNDS = (u.Q(0.0, "s"), u.Q(2 * jnp.pi, "s"))


def _chart(**kw):
    return cxfc.TubularChart(cxfc.BishopBuilder(circle), tau_bounds=BOUNDS, **kw)


def test_is_on_the_parameterized_branch() -> None:
    assert issubclass(cxfc.TubularChart, AbstractParameterizedChart)


def test_components_and_dimensions() -> None:
    ch = _chart()
    assert ch.components == ("tau", "n1", "n2")
    assert ch.coord_dimensions == ("time", "length", "length")


def test_dimension_follows_the_curve_parameter() -> None:
    """A curve parameterised by length reports 'length', not 'time'."""

    def by_length(tau):
        s = tau.ustrip("km")
        return u.Q(jnp.stack([s, jnp.zeros_like(s), jnp.zeros_like(s)]), "km")

    ch = cxfc.TubularChart(
        cxfc.BishopBuilder(by_length, "km"), tau_bounds=(u.Q(0.0, "km"), u.Q(1.0, "km"))
    )
    assert ch.coord_dimensions == ("length", "length", "length")


def test_cartesian_is_cart3d() -> None:
    assert isinstance(_chart().cartesian, cxc.Cart3D)


def test_the_chart_carries_the_builders_leaves() -> None:
    """The chart is always dynamic, and this pins why.

    `AbstractCurveFrameBuilder` holds a live `tau_0` array, so a builder has
    two leaves whatever the curve is -- a plain function contributes itself as
    a (non-array) leaf, an `equinox.Module` curve contributes its parameters.
    The chart adds its own `tau_bounds`. Measured, not assumed.
    """
    ch = _chart()
    assert len(jax.tree.leaves(ch.builder)) == 2
    assert len(jax.tree.leaves(ch)) == 4  # builder 2 + tau_bounds 2


def test_static_bounds_drop_their_leaves() -> None:
    """`tau_bounds` follows the usual opt-in rule for chart parameters."""
    ch = cxfc.TubularChart(
        cxfc.BishopBuilder(circle),
        tau_bounds=(u.StaticQuantity(-1.0, "s"), u.StaticQuantity(7.0, "s")),
    )
    assert len(jax.tree.leaves(ch)) == 2  # builder only


def test_the_reach_guard_fires_past_the_focal_distance() -> None:
    ch = _chart()
    at = {"tau": u.Q(0.0, "s"), "n1": u.Q(-1.6, "km"), "n2": u.Q(0.0, "km")}
    with pytest.raises(ValueError, match="outside the reach"):
        ch.check_data(at, values=True)


def test_the_reach_guard_also_fires_under_jit() -> None:
    """The eager path and the traced path are different code; test both.

    A bare `eqx.error_if` whose result is unused is dead-code-eliminated, so
    the traced branch can pass silently while the eager branch works.
    """
    ch = _chart()

    @jax.jit
    def run(v):
        at = {"tau": u.Q(0.0, "s"), "n1": u.Q(v, "km"), "n2": u.Q(0.0, "km")}
        return ch.check_data(at, values=True)["n1"].ustrip("km")

    with pytest.raises(jax.errors.JaxRuntimeError, match="outside the reach"):
        run(-1.6)


def test_forward_matches_the_frame_construction() -> None:
    """Forward is gamma + n1*U1 + n2*U2, by definition."""
    ch = _chart()
    b = ch.builder
    tau, n1, n2 = u.Q(0.7, "s"), u.Q(0.13, "km"), u.Q(-0.21, "km")
    got = cxc.pt_map({"tau": tau, "n1": n1, "n2": n2}, ch.M, ch, ch.M, cxc.cart3d)

    R = b.rotation_matrix(tau)
    g = b.location(tau).ustrip("km")
    want = g + 0.13 * R[1] - 0.21 * R[2]
    assert jnp.allclose(jnp.stack([got[k].ustrip("km") for k in "xyz"]), want)


def test_round_trip_cartesian_to_tubular_and_back() -> None:
    # BOUNDS is exactly one period of `circle` (see its definition above), so
    # tau=0.7 has no periodic alias inside the scan range and the inverse is
    # well-posed: there is exactly one nearest point for nearest_tau to find.
    ch = _chart()
    p = {"tau": u.Q(0.7, "s"), "n1": u.Q(0.13, "km"), "n2": u.Q(-0.21, "km")}
    xyz = cxc.pt_map(p, ch.M, ch, ch.M, cxc.cart3d)
    back = cxc.pt_map(xyz, ch.M, cxc.cart3d, ch.M, ch)
    assert jnp.allclose(back["tau"].ustrip("s"), 0.7, atol=1e-6)
    assert jnp.allclose(back["n1"].ustrip("km"), 0.13, atol=1e-6)
    assert jnp.allclose(back["n2"].ustrip("km"), -0.21, atol=1e-6)


def test_a_point_on_the_curve_has_zero_offsets() -> None:
    ch = _chart()
    on_curve = ch.builder.location(u.Q(1.1, "s"))
    d = {k: on_curve[i] for i, k in enumerate(("x", "y", "z"))}
    got = cxc.pt_map(d, ch.M, cxc.cart3d, ch.M, ch)
    assert jnp.allclose(got["n1"].ustrip("km"), 0.0, atol=1e-6)
    assert jnp.allclose(got["n2"].ustrip("km"), 0.0, atol=1e-6)


def test_identity_between_two_distinct_but_equal_charts() -> None:
    """Distinct chart objects decline to the Cartesian round trip.

    Parameterized charts compare conservatively -- equal only when identical --
    so two independently built charts take the fallback path, not the fast one.
    """
    ch1, ch2 = _chart(), _chart()
    assert ch1 is not ch2
    p = {"tau": u.Q(0.7, "s"), "n1": u.Q(0.13, "km"), "n2": u.Q(-0.21, "km")}
    got = cxc.pt_map(p, ch1.M, ch1, ch2.M, ch2)
    assert jnp.allclose(got["tau"].ustrip("s"), 0.7, atol=1e-6)
    assert jnp.allclose(got["n1"].ustrip("km"), 0.13, atol=1e-6)
    assert jnp.allclose(got["n2"].ustrip("km"), -0.21, atol=1e-6)


def test_identity_falls_back_through_cartesian_for_different_charts() -> None:
    """Two charts on *different* curves must give different coordinates.

    The same-curve fallback test above passes trivially for a fallback that
    returns its input unchanged or a copy of it, since the round trip lands
    back on the same numbers either way. Pin the fallback against a chart
    whose curve is displaced along z by 1 km: the offset picked up by that
    displacement must show up in the recovered coordinates. Values below are
    measured (run and printed), not hand-derived.
    """

    def shifted(tau: u.AbstractQuantity) -> u.AbstractQuantity:
        t = tau.ustrip("s")
        return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), jnp.ones_like(t)]), "km")

    ch1 = _chart()
    ch2 = cxfc.TubularChart(cxfc.BishopBuilder(shifted), tau_bounds=BOUNDS)
    p = {"tau": u.Q(0.7, "s"), "n1": u.Q(0.13, "km"), "n2": u.Q(-0.21, "km")}
    got = cxc.pt_map(p, ch1.M, ch1, ch2.M, ch2)

    # Measured: tau and n1 are unchanged (the shift is along U2 for this
    # planar curve's Bishop frame), n2 shifts by exactly the 1 km offset.
    assert jnp.allclose(got["tau"].ustrip("s"), 0.7, atol=1e-6)
    assert jnp.allclose(got["n1"].ustrip("km"), 0.13, atol=1e-6)
    assert jnp.allclose(got["n2"].ustrip("km"), 0.79, atol=1e-6)
    # And the requirement this test exists to enforce: at least one component
    # differs substantially from the input, so a no-op fallback would fail.
    assert not jnp.allclose(got["n2"].ustrip("km"), p["n2"].ustrip("km"), atol=1e-3)
