"""The point of the exercise: a fitted curve's parameters stay differentiable.

The inverse transition runs a root-find (and, for Bishop, an ODE solve inside
it). `optimistix` differentiates the root-find implicitly, so a gradient with
respect to a curve parameter flows all the way through.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

import coordinax.charts as cxc
import coordinax.manifolds as cxm
import unxt as u

import coordinaxs.curveframes as cxfc


class Helix(eqx.Module):
    """A curve whose radius is a live, fittable parameter."""

    radius: u.AbstractQuantity

    def __call__(self, tau: u.AbstractQuantity) -> u.AbstractQuantity:
        t = tau.ustrip("s")
        r = self.radius.ustrip("km")
        return u.Q(jnp.stack([r * jnp.cos(t), r * jnp.sin(t), 0.3 * t]), "km")


BOUNDS = (u.Q(-1.0, "s"), u.Q(6.0, "s"))


def _chart(radius_km: float) -> cxfc.TubularChart:
    curve = Helix(radius=u.Q(radius_km, "km"))
    return cxfc.TubularChart(cxfc.BishopBuilder(curve), tau_bounds=BOUNDS)


def test_a_live_curve_parameter_is_a_leaf() -> None:
    """Radius + the builder's tau_0 + two tau_bounds = 4."""
    assert len(jax.tree.leaves(_chart(1.0))) == 4


def test_grad_through_the_inverse_solve_matches_finite_differences() -> None:
    x = {"x": u.Q(1.1, "km"), "y": u.Q(0.4, "km"), "z": u.Q(0.2, "km")}

    def n1_of_radius(radius_km):
        return cxc.pt_map(x, cxm.R3, cxc.cart3d, cxm.R3, _chart(radius_km))[
            "n1"
        ].ustrip("km")

    analytic = jax.grad(n1_of_radius)(1.0)
    h = 1e-5
    numeric = (n1_of_radius(1.0 + h) - n1_of_radius(1.0 - h)) / (2 * h)
    assert jnp.allclose(analytic, numeric, rtol=1e-4), (analytic, numeric)


def test_the_transition_is_jittable() -> None:
    x = {"x": u.Q(1.1, "km"), "y": u.Q(0.4, "km"), "z": u.Q(0.2, "km")}
    ch = _chart(1.0)
    eager = cxc.pt_map(x, ch.M, cxc.cart3d, ch.M, ch)["n1"]
    jitted = eqx.filter_jit(lambda c: cxc.pt_map(x, c.M, cxc.cart3d, c.M, c)["n1"])(ch)
    assert jnp.allclose(eager.ustrip("km"), jitted.ustrip("km"), atol=1e-8)


def test_jit_retraces_once_across_radii() -> None:
    """The radius is a leaf, so it must not be part of the trace key."""
    traces = []
    x = {"x": u.Q(1.1, "km"), "y": u.Q(0.4, "km"), "z": u.Q(0.2, "km")}

    @eqx.filter_jit
    def f(chart):
        traces.append(1)
        return cxc.pt_map(x, chart.M, cxc.cart3d, chart.M, chart)["n1"]

    f(_chart(1.0))
    f(_chart(1.3))
    assert len(traces) == 1, f"retraced {len(traces)} times"
