"""Time-dependent curves: binding the evaluation time, and the Eulerian reading."""

import jax.numpy as jnp

import unxt as u

import coordinaxs.curveframes as cxfc

A = 0.5


def stretch(tau: u.AbstractQuantity, t: u.AbstractQuantity) -> u.AbstractQuantity:
    """A line stretching uniformly: gamma(tau, t) = (tau (1 + A t), 0, 0)."""
    x = tau.ustrip("s") * (1.0 + A * t.ustrip("s"))
    z = jnp.zeros_like(x)
    return u.Q(jnp.stack([x, z, z]), "km")


def test_at_time_binds_the_evaluation_time() -> None:
    curve = cxfc.AtTime(stretch, u.Q(1.0, "s"))
    got = curve(u.Q(2.0, "s")).ustrip("km")
    assert jnp.allclose(got, jnp.array([2.0 * 1.5, 0.0, 0.0]))


def test_arclength_over_a_time_dependent_curve_measures_the_current_slice() -> None:
    """Eulerian: `s` is arc length on the slice being evaluated.

    At t=1 the curve is stretched by 1.5, so the point at arc length 1.5 is
    the material point tau=1, at x=1.5.
    """
    arc = cxfc.ArcLength(stretch)  # still two-argument
    at_t1 = cxfc.AtTime(arc, u.Q(1.0, "s"))
    got = at_t1(u.Q(1.5, "km")).ustrip("km")
    assert jnp.allclose(got, jnp.array([1.5, 0.0, 0.0]), atol=1e-6)


def test_binding_time_first_is_still_supported_for_static_use() -> None:
    """`ArcLength(AtTime(...))` is legal -- it is just the frozen slice."""
    frozen = cxfc.ArcLength(cxfc.AtTime(stretch, u.Q(0.0, "s")))
    got = frozen(u.Q(1.0, "km")).ustrip("km")
    assert jnp.allclose(got, jnp.array([1.0, 0.0, 0.0]), atol=1e-6)


def test_a_live_time_is_a_leaf() -> None:
    import jax

    dyn = cxfc.AtTime(stretch, u.Q(1.0, "s"))
    sta = cxfc.AtTime(stretch, u.StaticQuantity(1.0, "s"))
    assert len(jax.tree.leaves(dyn)) == 1
    assert jax.tree.leaves(sta) == []
