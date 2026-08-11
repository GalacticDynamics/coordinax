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
