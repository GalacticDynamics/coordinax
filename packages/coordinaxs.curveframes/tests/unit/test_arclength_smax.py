"""The optional `s_max` dense-interpolation precompute.

Behaviour must be identical with and without `s_max` -- that agreement is
the whole correctness claim of this path. See
`.superpowers/sdd/2026-08-13-arc-length/task-5b-report.md` for the
before/after timing measurements this test file does not itself take.
"""

import jax
import jax.numpy as jnp
import pytest

import unxt as u

import coordinaxs.curveframes as cxfc


def helix(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    """Helix of speed sqrt(1 + 0.3**2) = 1.0440306508910550 km/s."""
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), 0.3 * t]), "km")


def parabola(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    """Non-constant speed: gamma(tau) = (tau, tau**2, 0)."""
    t = tau.ustrip("s")
    return u.Q(jnp.stack([t, t**2, jnp.zeros_like(t)]), "km")


def stretch(tau: u.AbstractQuantity, t: u.AbstractQuantity) -> u.AbstractQuantity:
    """A line stretching uniformly: gamma(tau, t) = (tau (1 + 0.5 t), 0, 0)."""
    x = tau.ustrip("s") * (1.0 + 0.5 * t.ustrip("s"))
    z = jnp.zeros_like(x)
    return u.Q(jnp.stack([x, z, z]), "km")


# --------------------------------------------------------------------------
# Behaviour is identical with and without s_max.


def test_arclength_agrees_with_and_without_s_max() -> None:
    plain = cxfc.ArcLength(helix)
    cached = cxfc.ArcLength(helix, s_max=u.Q(5.0, "km"))
    for s_val in (0.0, 0.5, 1.0, 3.0, 5.0):
        got_plain = plain(u.Q(s_val, "km")).ustrip("km")
        got_cached = cached(u.Q(s_val, "km")).ustrip("km")
        assert jnp.allclose(got_plain, got_cached, atol=1e-8), (
            s_val,
            got_plain,
            got_cached,
        )


def test_arclength_agrees_for_nonconstant_speed() -> None:
    plain = cxfc.ArcLength(parabola)
    cached = cxfc.ArcLength(parabola, s_max=u.Q(4.0, "km"))
    for s_val in (0.0, 0.3, 1.7, 4.0):
        got_plain = plain(u.Q(s_val, "km")).ustrip("km")
        got_cached = cached(u.Q(s_val, "km")).ustrip("km")
        assert jnp.allclose(got_plain, got_cached, atol=1e-6), (
            s_val,
            got_plain,
            got_cached,
        )


def test_lagrangian_agrees_with_and_without_s_max() -> None:
    plain = cxfc.LagrangianArcLength(stretch, u.Q(0.0, "s"))
    cached = cxfc.LagrangianArcLength(stretch, u.Q(0.0, "s"), s_max=u.Q(5.0, "km"))
    for s_val, t_val in ((0.0, 0.0), (1.0, 0.0), (2.5, 1.0), (5.0, 2.0)):
        got_plain = plain(u.Q(s_val, "km"), u.Q(t_val, "s")).ustrip("km")
        got_cached = cached(u.Q(s_val, "km"), u.Q(t_val, "s")).ustrip("km")
        assert jnp.allclose(got_plain, got_cached, atol=1e-8), (
            s_val,
            t_val,
            got_plain,
            got_cached,
        )


def test_arclength_precompute_stays_differentiable() -> None:
    """The precomputed path is still smooth and gives the same gradient."""
    plain = cxfc.ArcLength(helix)
    cached = cxfc.ArcLength(helix, s_max=u.Q(5.0, "km"))

    def z_of_s(arc, s_val):
        return arc(u.Q(s_val, "km")).ustrip("km")[2]

    g_plain = jax.grad(lambda s: z_of_s(plain, s))(2.0)
    g_cached = jax.grad(lambda s: z_of_s(cached, s))(2.0)
    assert jnp.allclose(g_plain, g_cached, rtol=1e-6)


# --------------------------------------------------------------------------
# ArcLength over a two-argument curve rejects s_max at construction.


def test_s_max_raises_for_a_two_argument_curve() -> None:
    with pytest.raises(ValueError, match="two-argument"):
        cxfc.ArcLength(stretch, s_max=u.Q(5.0, "km"))


def test_s_max_is_fine_after_binding_time_with_at_time() -> None:
    frozen = cxfc.AtTime(stretch, u.Q(0.0, "s"))
    cached = cxfc.ArcLength(frozen, s_max=u.Q(5.0, "km"))
    plain = cxfc.ArcLength(frozen)
    got_plain = plain(u.Q(2.0, "km")).ustrip("km")
    got_cached = cached(u.Q(2.0, "km")).ustrip("km")
    assert jnp.allclose(got_plain, got_cached, atol=1e-8)


# --------------------------------------------------------------------------
# The interpolation's domain is [0, s_max]; outside it, raise.


def test_s_outside_s_max_raises() -> None:
    cached = cxfc.ArcLength(helix, s_max=u.Q(3.0, "km"))
    with pytest.raises(Exception, match="precomputed domain"):
        cached(u.Q(3.5, "km"))


def test_s_negative_raises() -> None:
    cached = cxfc.ArcLength(helix, s_max=u.Q(3.0, "km"))
    with pytest.raises(Exception, match="precomputed domain"):
        cached(u.Q(-0.1, "km"))


def test_lagrangian_s_outside_s_max_raises() -> None:
    cached = cxfc.LagrangianArcLength(stretch, u.Q(0.0, "s"), s_max=u.Q(2.0, "km"))
    with pytest.raises(Exception, match="precomputed domain"):
        cached(u.Q(2.5, "km"), u.Q(0.0, "s"))


# --------------------------------------------------------------------------
# #693: a Quantity s_max is a leaf; a StaticQuantity s_max is not.


def test_a_quantity_s_max_is_a_leaf() -> None:
    dyn = cxfc.ArcLength(helix, s_max=u.Q(5.0, "km"))
    sta = cxfc.ArcLength(helix, s_max=u.StaticQuantity(5.0, "km"))
    assert len(jax.tree.leaves(dyn)) > len(jax.tree.leaves(sta))
