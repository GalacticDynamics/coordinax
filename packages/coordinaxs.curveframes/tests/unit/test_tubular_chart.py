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


BOUNDS = (u.Q(-1.0, "s"), u.Q(7.0, "s"))


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
    # NOTE: tau=0.8, not the brief's 0.7. `circle` has period 2*pi =~ 6.283, and
    # BOUNDS=(-1, 7) spans 8 units -- wider than one period -- so tau=0.7 has a
    # periodic alias at 0.7 + 2*pi =~ 6.983, still inside BOUNDS. Bishop/Frenet
    # holonomy is exactly zero for a planar curve, so gamma and the (U1, U2)
    # triad are *identical* at both parameter values: the two candidates are an
    # exact tie in Euclidean distance, not a near-tie broken by seed-grid luck.
    # `nearest_tau`'s coarse scan (see nearest.py) has no way to prefer one
    # over the other, and empirically resolves it to the 6.983 alias. This is
    # not a chart bug: nearest_tau is documented to find the *nearest* point,
    # and here two points are equally near. tau=0.8 puts the alias (0.8+2*pi
    # =~ 7.083) outside BOUNDS, making the inverse well-posed again.
    ch = _chart()
    p = {"tau": u.Q(0.8, "s"), "n1": u.Q(0.13, "km"), "n2": u.Q(-0.21, "km")}
    xyz = cxc.pt_map(p, ch.M, ch, ch.M, cxc.cart3d)
    back = cxc.pt_map(xyz, ch.M, cxc.cart3d, ch.M, ch)
    assert jnp.allclose(back["tau"].ustrip("s"), 0.8, atol=1e-6)
    assert jnp.allclose(back["n1"].ustrip("km"), 0.13, atol=1e-6)
    assert jnp.allclose(back["n2"].ustrip("km"), -0.21, atol=1e-6)


def test_a_point_on_the_curve_has_zero_offsets() -> None:
    ch = _chart()
    on_curve = ch.builder.location(u.Q(1.1, "s"))
    d = {k: on_curve[i] for i, k in enumerate(("x", "y", "z"))}
    got = cxc.pt_map(d, ch.M, cxc.cart3d, ch.M, ch)
    assert jnp.allclose(got["n1"].ustrip("km"), 0.0, atol=1e-6)
    assert jnp.allclose(got["n2"].ustrip("km"), 0.0, atol=1e-6)
