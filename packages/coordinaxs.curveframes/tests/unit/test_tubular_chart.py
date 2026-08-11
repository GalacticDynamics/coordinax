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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Unreachable given the landed BishopBuilder: `tau_0` is always a real "
        "Quantity leaf by design (see "
        "test_bishop.py::TestBishopTau0::test_tau_0_is_a_pytree_leaf), and the "
        "plain `circle` function is itself an opaque pytree leaf. `builder` "
        "cannot be a static TubularChart field either, since that would also "
        "hide an eqx.Module curve's differentiable parameters. See "
        "task-2-report.md for the full analysis."
    ),
)
def test_a_static_curve_gives_no_leaves() -> None:
    """A plain-function curve closes over constants: nothing to differentiate."""
    assert jax.tree.leaves(_chart()) == []
