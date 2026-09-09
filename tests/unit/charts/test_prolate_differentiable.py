"""Delta is differentiable when passed as a dynamic Quantity."""

import jax
import jax.numpy as jnp
import plum

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


def _routes_through_cylindrical(p, frm, to):
    """Whether the transition hands `Cylindrical3D` to a nested `pt_map`.

    The converting branch is *defined* by that hop, so this asks the question
    the guard cares about directly rather than inferring it from a call count.
    """
    seen = False
    original = plum.Function.__call__

    def watching(self, *args, **kwargs):
        nonlocal seen
        if self.__name__ == "pt_map" and any(
            isinstance(a, cxc.Cylindrical3D) for a in args
        ):
            seen = True
        return original(self, *args, **kwargs)

    plum.Function.__call__ = watching
    try:
        cxc.pt_map(p, frm, to)
    finally:
        plum.Function.__call__ = original
    return seen


def test_same_delta_skips_the_conversion_branch():
    """An identity transition does not pay for the branch it does not take.

    `jax.lax.cond` traces *both* branches, so routing a same-`Delta` pair
    through it cost a full round-trip to cylindrical and back -- as expensive
    as actually converting. A concrete `Delta` decides in Python instead.

    The conversion branch is exactly the hop through `Cylindrical3D`, so that
    is what is asserted on. A call *count* would be the weaker question: an
    identity that still traced the dead branch would make more calls than it
    should and fewer than a real conversion, and any `n_same < n_conv` test
    would pass regardless.
    """
    same = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, "m"))
    twin = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, "m"))
    other = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(3.0, "m"))

    assert not _routes_through_cylindrical(Q_IN, same, twin)
    # The converting case is the control: it must still take that route.
    assert _routes_through_cylindrical(Q_IN, same, other)
