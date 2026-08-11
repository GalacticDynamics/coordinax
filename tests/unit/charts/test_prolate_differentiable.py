"""Delta is differentiable when passed as a dynamic Quantity."""

import jax
import jax.numpy as jnp

import unxt as u

import coordinax.charts as cxc

# The chart's validity domain is ``mu >= Delta**2`` and ``|nu| <= Delta**2``.
# ``mu`` must clear the largest ``Delta`` used below (3 m), or the transition
# takes the square root of a negative number and every value is NaN.
Q_IN = {"mu": u.Q(12.0, "m2"), "nu": u.Q(0.5, "m2"), "phi": u.Q(0.3, "rad")}


def test_prolate_still_has_fixed_components():
    """The invariant the shared-base split exists to protect.

    `AbstractFixedComponentsChart` is used as a *filter* by `guess_chart_cls`,
    `GalileanCT`, and the hypothesis strategies -- if prolate ever fell out of
    it, every one of them would silently skip it and nothing would go red.
    """
    assert issubclass(cxc.ProlateSpheroidal3D, cxc.AbstractFixedComponentsChart)


def test_static_delta_is_unchanged():
    """Every existing call site passes StaticQuantity and must be unaffected."""
    c = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, "m"))
    twin = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, "m"))
    assert len(jax.tree.leaves(c)) == 0
    assert c == twin  # by value, not identity -- dict/cache keys depend on it
    assert hash(c) == hash(twin)
    assert c != cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(3.0, "m"))


def test_dynamic_delta_gives_a_leaf():
    c = cxc.ProlateSpheroidal3D(Delta=u.Q(2.0, "m"))
    assert len(jax.tree.leaves(c)) == 1


def _x_of_delta(delta_value):
    chart = cxc.ProlateSpheroidal3D(Delta=u.Q(delta_value, "m"))
    return cxc.pt_map(Q_IN, chart, cxc.cart3d)["x"].ustrip("m")


def _x_of_chart(chart):
    return cxc.pt_map(Q_IN, chart, cxc.cart3d)["x"].ustrip("m")


def test_grad_wrt_delta_matches_finite_differences():
    """Differentiate w.r.t. the *chart*, which only works if Delta is a leaf.

    Differentiating `_x_of_delta` instead would pass even on the static branch:
    the chart is built inside the traced function there, so the tracer flows
    through the arithmetic regardless of how the chart flattens.
    """
    d0 = 2.0
    chart = cxc.ProlateSpheroidal3D(Delta=u.Q(d0, "m"))
    analytic = float(jax.grad(_x_of_chart)(chart).Delta.ustrip("m"))
    h = 1e-4
    numeric = (_x_of_delta(d0 + h) - _x_of_delta(d0 - h)) / (2 * h)
    assert jnp.allclose(analytic, numeric, rtol=1e-4), (analytic, numeric)


def test_jit_retraces_once_across_delta_values():
    """The chart is passed *in*, so Delta must be a leaf, not part of the key.

    Building the chart inside the traced function would pass either way -- the
    tracer would simply be swallowed by a static chart -- so the chart has to
    cross the `jit` boundary for this to mean anything.
    """
    traces = []

    @jax.jit
    def f(chart):
        traces.append(1)
        return cxc.pt_map(Q_IN, chart, cxc.cart3d)["x"].ustrip("m")

    f(cxc.ProlateSpheroidal3D(Delta=u.Q(2.0, "m")))
    f(cxc.ProlateSpheroidal3D(Delta=u.Q(3.0, "m")))
    assert len(traces) == 1, (
        f"retraced {len(traces)} times; Delta is being treated as static"
    )

    # A static Delta still keys the cache by value: equal charts share a trace.
    static_traces = []

    @jax.jit
    def g(chart):
        static_traces.append(1)
        return cxc.pt_map(Q_IN, chart, cxc.cart3d)["x"].ustrip("m")

    g(cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, "m")))
    g(cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, "m")))
    assert len(static_traces) == 1
