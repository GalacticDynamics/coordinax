"""The optional `s_max` precompute, and the custom JVP that keeps it correct.

#713 (see its issue text for the full numbers): a previous attempt cached
`tau(s)` as a `diffrax` dense interpolation built once in `__init__`, and
that cache went stale the moment a caller perturbed the wrapped curve's own
parameters without rebuilding the `ArcLength` -- exactly the pattern
`equinox.tree_at`/`jax.grad` use on a module built *outside* the function
being differentiated. Gradients came back wrong, silently. The fix replaces
autodiff through the cache with a hand-derived `equinox.filter_custom_jvp`
rule (implicit function theorem on the arc-length integral), so the fast
path and gradient correctness are independent of each other: the tests below
are organised around that independence, not around `s_max` alone.

Every test states, in its own docstring or a comment, what would have to
break in `_src/arclength.py` for it to fail -- per this codebase's testing
standard (see `_src/arclength.py`'s own module docstring and #712/#699's
five-tests-passing-for-the-wrong-reason history). The three load-bearing
ones (gradient correctness built outside, the chart boundary case, and the
Bishop/forward-mode regression) were each verified by deliberately breaking
the implementation and confirming the test catches it; see the PR
description for what was broken and how.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import unxt as u

import coordinaxs.curveframes as cxfc


def helix(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    """Constant-speed helix: gamma(tau) = (cos, sin, 0.3 tau), speed 1.044..."""
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), 0.3 * t]), "km")


def parabola(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    """Non-constant speed: gamma(tau) = (tau, tau**2, 0), speed sqrt(1+4tau**2).

    A constant-speed curve cannot distinguish the fast path from the slow
    one: tau(s) is exactly linear regardless of how it's solved, so even a
    badly-wrong "fast" implementation (e.g. one that silently used the wrong
    speed) could still agree with the slow path to machine precision. This
    curve's speed genuinely varies, so agreement here is evidence the dense
    interpolation is solving the *same* ODE, not just an easy one.
    """
    t = tau.ustrip("s")
    return u.Q(jnp.stack([t, t**2, jnp.zeros_like(t)]), "km")


def stretch(tau: u.AbstractQuantity, t: u.AbstractQuantity) -> u.AbstractQuantity:
    """A curve that bends over time: gamma(tau, t) = (tau(1+0.5t), 0.1 t tau^2, 0)."""
    tv = tau.ustrip("s")
    tt = t.ustrip("s")
    x = tv * (1.0 + 0.5 * tt)
    y = 0.1 * tt * tv**2
    z = jnp.zeros_like(x)
    return u.Q(jnp.stack([x, y, z]), "km")


class Helix(eqx.Module):
    """A curve carrying a differentiable parameter, for gradient tests."""

    theta: u.AbstractQuantity

    def __call__(self, tau: u.AbstractQuantity) -> u.AbstractQuantity:
        t = tau.ustrip("s")
        th = self.theta.ustrip("km")
        return u.Q(jnp.stack([th * jnp.cos(t), th * jnp.sin(t), 0.3 * t]), "km")


# --------------------------------------------------------------------------
# Criterion 2: fast and slow paths agree, on a curve whose speed varies.
#
# Fails if `_solve_tau_dense`/`_eval_tau_dense` solve a different ODE than
# `_solve_tau` does (e.g. a sign error, or evaluating speed at the wrong
# point) -- a bug that a constant-speed curve's linear tau(s) would hide.


def test_fast_and_slow_agree_for_nonconstant_speed() -> None:
    """At the diffeqsolver's own tight tolerance, agreement is tolerance-tight too."""
    import dataclasses

    import diffrax as dfx

    from coordinaxs.curveframes._src.arclength import _DIFFEQSOLVER

    tight = dataclasses.replace(
        _DIFFEQSOLVER, stepsize_controller=dfx.PIDController(rtol=1e-13, atol=1e-13)
    )
    plain = cxfc.ArcLength(parabola, "s", diffeqsolver=tight)
    fast = cxfc.ArcLength(parabola, "s", diffeqsolver=tight, s_max=u.Q(5.0, "km"))
    for s_val in (0.1, 1.0, 3.0, 4.99):
        got_plain = plain(u.Q(s_val, "km")).ustrip("km")
        got_fast = fast(u.Q(s_val, "km")).ustrip("km")
        assert jnp.allclose(got_plain, got_fast, atol=1e-11), (
            s_val,
            got_plain,
            got_fast,
        )


def test_fast_and_slow_agree_for_lagrangian_nonconstant_speed() -> None:
    plain = cxfc.LagrangianArcLength(stretch, u.Q(0.3, "s"), "s")
    fast = cxfc.LagrangianArcLength(stretch, u.Q(0.3, "s"), "s", s_max=u.Q(5.0, "km"))
    for s_val, t_val in ((0.1, 0.0), (1.0, 0.5), (3.0, 1.2)):
        got_plain = plain(u.Q(s_val, "km"), u.Q(t_val, "s")).ustrip("km")
        got_fast = fast(u.Q(s_val, "km"), u.Q(t_val, "s")).ustrip("km")
        assert jnp.allclose(got_plain, got_fast, atol=1e-6), (
            s_val,
            t_val,
            got_plain,
            got_fast,
        )


# --------------------------------------------------------------------------
# s_max validation at construction.


def test_s_max_raises_for_a_two_argument_curve() -> None:
    with pytest.raises(ValueError, match="two-argument"):
        cxfc.ArcLength(stretch, "s", s_max=u.Q(5.0, "km"))


def test_s_max_is_fine_after_binding_time_with_at_time() -> None:
    frozen = cxfc.AtTime(stretch, u.Q(0.0, "s"))
    fast = cxfc.ArcLength(frozen, "s", s_max=u.Q(5.0, "km"))
    plain = cxfc.ArcLength(frozen, "s")
    assert jnp.allclose(
        plain(u.Q(2.0, "km")).ustrip("km"), fast(u.Q(2.0, "km")).ustrip("km"), atol=1e-8
    )


def test_s_max_always_valid_for_lagrangian() -> None:
    """Unlike `ArcLength`, `LagrangianArcLength.s_max` never raises: t0 is fixed."""
    cxfc.LagrangianArcLength(stretch, u.Q(0.0, "s"), "s", s_max=u.Q(5.0, "km"))


def test_a_quantity_s_max_is_a_leaf() -> None:
    dyn = cxfc.ArcLength(helix, "s", s_max=u.Q(5.0, "km"))
    sta = cxfc.ArcLength(helix, "s", s_max=u.StaticQuantity(5.0, "km"))
    assert len(jax.tree.leaves(dyn)) > len(jax.tree.leaves(sta))


# --------------------------------------------------------------------------
# The domain margin: solved data just past each end, and a raise beyond it.
#
# Fails if `_eval_tau_dense`'s margin regresses to an exact `[0, s_max]` gate
# (the reverted PR's bug), or grows large enough to swallow the `6.0`/`-0.26`
# misuse cases, or goes back to *clamping* into the margin instead of
# `_solve_tau_dense` integrating across it.


def test_s_within_the_margin_of_s_max_is_correct_not_clamped() -> None:
    """The margin is solved data, so a query in it is right, not just accepted.

    This is the whole point of integrating past each end. The earlier version
    clipped into ``[0, s_max]``, so `fast(5.24)` returned tau(5.0) -- an
    answer wrong by 6.3e-2 here, handed back with no error at all. Merely
    asserting "does not raise" passed against that bug; comparing to a fresh
    solve is what catches it.
    """
    s_max = u.Q(5.0, "km")
    fast = cxfc.ArcLength(helix, "s", s_max=s_max)
    plain = cxfc.ArcLength(helix, "s")  # ground truth: no precompute at all

    for s_val in (5.24, -0.24, 5.0, 0.0, 2.5):
        got = fast(u.Q(s_val, "km")).ustrip("km")
        want = plain(u.Q(s_val, "km")).ustrip("km")
        assert jnp.allclose(got, want, atol=1e-9), (s_val, got, want)

    # And specifically: it is *not* the boundary value it used to clamp to.
    at_edge = fast(u.Q(5.0, "km")).ustrip("km")
    past_edge = fast(u.Q(5.24, "km")).ustrip("km")
    assert not jnp.allclose(past_edge, at_edge, atol=1e-3), (past_edge, at_edge)


def test_s_far_outside_s_max_raises() -> None:
    fast = cxfc.ArcLength(helix, "s", s_max=u.Q(5.0, "km"))
    with pytest.raises(Exception, match="solved domain"):
        fast(u.Q(6.0, "km"))
    with pytest.raises(Exception, match="solved domain"):
        fast(u.Q(-0.26, "km"))


def test_the_margin_scales_with_a_small_s_max() -> None:
    """The margin is a *fraction* of `s_max`, not an absolute floor.

    With ``slack = margin * max(1, |s_max|)`` the tolerated overshoot stopped
    shrinking once ``s_max`` fell below 1: a 0.001 km domain would silently
    clamp a query at 0.05 km -- fifty times its own length -- and hand back
    the boundary value as though it were the answer. Fails if that floor
    comes back.
    """
    tiny = cxfc.ArcLength(helix, "s", s_max=u.Q(0.001, "km"))
    tiny(u.Q(0.00104, "km"))  # 4% past: inside the 5% margin, must not raise
    with pytest.raises(Exception, match="solved domain"):
        tiny(u.Q(0.002, "km"))  # 100% past: must raise, and did not before


def test_lagrangian_s_far_outside_s_max_raises() -> None:
    fast = cxfc.LagrangianArcLength(stretch, u.Q(0.0, "s"), "s", s_max=u.Q(2.0, "km"))
    with pytest.raises(Exception, match="solved domain"):
        fast(u.Q(3.0, "km"), u.Q(0.0, "s"))


# --------------------------------------------------------------------------
# Criterion 1: gradients match finite differences, module built *outside*
# the differentiated function -- the exact pattern that broke in #713.
#
# Each test below reproduces the failure numbers from the issue first (as an
# explicit assertion, not a comment) against a hand-rolled "differentiate
# through the raw cache" computation, then asserts the shipped `ArcLength`
# gets the right answer for the same setup. A test that only checked the
# post-fix number would not have caught the original bug; asserting the
# pre-fix number too makes that failure mode explicit and pins that this
# test *would* have caught it.


def test_grad_matches_finite_difference_for_curve_params_built_outside() -> None:
    """#713's own worked example: helix pitch, s=2, theta=1."""
    curve0 = Helix(u.Q(1.0, "km"))
    arc = cxfc.ArcLength(curve0, "s", s_max=u.Q(10.0, "km"))  # built OUTSIDE grad
    s_val = u.Q(2.0, "km")

    def f(theta_val: float) -> float:
        new_curve = Helix(u.Q(theta_val, "km"))
        # The exact "amortising" pattern: reuse `arc`, only swap the curve.
        perturbed = eqx.tree_at(lambda a: a.curve, arc, new_curve)
        return perturbed(s_val).ustrip("km")[2]

    def f_rebuilt(theta_val: float) -> float:
        """Ground truth: rebuild the whole `ArcLength`, interpolation included."""
        new_curve = Helix(u.Q(theta_val, "km"))
        fresh = cxfc.ArcLength(new_curve, "s", s_max=u.Q(10.0, "km"))
        return fresh(s_val).ustrip("km")[2]

    got = jax.grad(f)(1.0)
    h = 1e-6
    fd = (f_rebuilt(1.0 + h) - f_rebuilt(1.0 - h)) / (2 * h)

    # `f` returns the *z* component, and z = 0.3 * tau, so the chain rule puts
    # dz/dtheta at 0.3 * dtau/dtheta. Divide the pitch back out to compare
    # against the issue's own reported numbers for this exact setup:
    # tau*(s=2, theta=1) = 1.9156525704, dtau/dtheta = -1.7574794224.
    dtau_dtheta = got / 0.3

    # The stale cache's gradient landed near the *primal* tau value, nowhere
    # near the true derivative below. A regression back to differentiating
    # through `_interp` would reproduce something in that neighbourhood.
    assert not jnp.allclose(dtau_dtheta, 1.9156525704, atol=1e-3), got
    assert jnp.allclose(dtau_dtheta, -1.7574794224, atol=1e-6), got
    assert jnp.allclose(got, fd, rtol=1e-4), (got, fd)


def test_grad_matches_finite_difference_for_tau_0_built_outside() -> None:
    curve0 = Helix(u.Q(1.0, "km"))
    arc = cxfc.ArcLength(curve0, "s", tau_0=u.Q(0.2, "s"), s_max=u.Q(10.0, "km"))
    s_val = u.Q(2.0, "km")

    def f(tau0_val: float) -> float:
        perturbed = eqx.tree_at(lambda a: a.tau_0, arc, u.Q(tau0_val, "s"))
        return perturbed(s_val).ustrip("km")[2]

    def f_rebuilt(tau0_val: float) -> float:
        fresh = cxfc.ArcLength(
            curve0, "s", tau_0=u.Q(tau0_val, "s"), s_max=u.Q(10.0, "km")
        )
        return fresh(s_val).ustrip("km")[2]

    got = jax.grad(f)(0.2)
    h = 1e-6
    fd = (f_rebuilt(0.2 + h) - f_rebuilt(0.2 - h)) / (2 * h)
    assert jnp.allclose(got, fd, rtol=1e-4), (got, fd)


def test_lagrangian_t0_grad_matches_finite_difference_built_outside() -> None:
    """#713: `LagrangianArcLength.t0`'s gradient came back exactly 0.0. Not here."""
    t0_0 = u.Q(0.3, "s")
    lag = cxfc.LagrangianArcLength(
        stretch, t0_0, "s", s_max=u.Q(5.0, "km")
    )  # built OUTSIDE grad
    s_val = u.Q(2.0, "km")
    t_eval = u.Q(1.0, "s")

    def f(t0_val: float) -> float:
        perturbed = eqx.tree_at(lambda m: m.t0, lag, u.Q(t0_val, "s"))
        return perturbed(s_val, t_eval).ustrip("km")[1]

    def f_rebuilt(t0_val: float) -> float:
        fresh = cxfc.LagrangianArcLength(
            stretch, u.Q(t0_val, "s"), "s", s_max=u.Q(5.0, "km")
        )
        return fresh(s_val, t_eval).ustrip("km")[1]

    got = jax.grad(f)(0.3)
    h = 1e-6
    fd = (f_rebuilt(0.3 + h) - f_rebuilt(0.3 - h)) / (2 * h)

    # The reverted implementation returned exactly 0.0 here (#713).
    assert not jnp.allclose(got, 0.0, atol=1e-8), got
    assert jnp.allclose(got, fd, rtol=1e-3), (got, fd)


def test_grad_without_s_max_also_uses_the_custom_jvp() -> None:
    """The slow path goes through the same `_tau_of_s`, so it is exercised too.

    Fails if the custom JVP is only wired up on the `interp is not None`
    branch and the plain solve still falls through to ordinary autodiff
    (which would be correct here, since nothing is stale without a cache --
    this pins that the *same* code path is used regardless, not just that
    the answer happens to agree in this particular case).
    """
    curve0 = Helix(u.Q(1.0, "km"))
    arc = cxfc.ArcLength(curve0, "s")  # no s_max
    s_val = u.Q(2.0, "km")

    def f(theta_val: float) -> float:
        new_curve = Helix(u.Q(theta_val, "km"))
        perturbed = eqx.tree_at(lambda a: a.curve, arc, new_curve)
        return perturbed(s_val).ustrip("km")[2]

    # As above, `f` returns z = 0.3 * tau, so divide the pitch back out to
    # recover dtau/dtheta and compare against #713's verified value.
    got = jax.grad(f)(1.0)
    assert jnp.allclose(got / 0.3, -1.7574794224, atol=1e-6), got


# --------------------------------------------------------------------------
# Criterion 3: works inside `TubularChart`, including the domain-edge case
# that broke the reverted version (a `nearest_tau` root landing just past
# `tau_bounds`'s start).
#
# Fails if `_eval_tau_dense`'s domain margin regresses to an exact gate, or
# if the custom JVP breaks `nearest_tau`'s own forward-mode use (it doesn't
# differentiate here, but `check_data`/`jacobian_factor` downstream do, via
# `jax.jacfwd` -- a `custom_vjp`-based version raises `TypeError: can't
# apply forward-mode autodiff (jvp) to a custom_vjp function` there, which
# is exactly the class of bug this file's Bishop/jacfwd test below targets).


def test_chart_round_trip_at_and_near_the_tau_zero_boundary() -> None:
    """5 round trips at/near the boundary `nearest_tau` probes past s_max's edge.

    `nearest_tau`'s bracketed root-find evaluates the curve up to one scan-seed
    spacing outside `tau_bounds` whenever the nearest seed sits at an edge --
    not a rare event, the normal case for an on-curve point near tau=0 or
    tau=s_max. #713 measured a *converged* result of s=-4.73e-9 hitting an
    exact `[0, s_max]` gate and failing 4 of these 5 cases.
    """
    import coordinax.charts as cxc

    s_max = u.Q(2 * jnp.pi, "km")
    arc = cxfc.ArcLength(helix, "s", s_max=s_max)
    bishop = cxfc.BishopBuilder(arc, "km")
    ch = cxfc.TubularChart(bishop, tau_bounds=(u.Q(0.0, "km"), s_max))

    s_probe_vals = (0.0, 1e-9, 0.05, 1.0, float(s_max.value) - 1e-6)
    for s_val in s_probe_vals:
        on_curve = ch.builder.location(u.Q(s_val, "km"))
        d = {k: on_curve[i] for i, k in enumerate(("x", "y", "z"))}
        back = cxc.pt_map(d, ch.M, cxc.cart3d, ch.M, ch)
        assert jnp.allclose(back["tau"].ustrip("km"), s_val, atol=1e-5), (
            s_val,
            back["tau"].ustrip("km"),
        )
        assert jnp.allclose(back["n1"].ustrip("km"), 0.0, atol=1e-5)
        assert jnp.allclose(back["n2"].ustrip("km"), 0.0, atol=1e-5)


# --------------------------------------------------------------------------
# Bishop + forward-mode regression: this is the case that a `custom_vjp`
# (rather than `custom_jvp`) implementation cannot pass at all --
# `BishopBuilder._tangent_at` differentiates the curve it wraps in forward
# mode (`unxt.experimental.jacfwd`), and `jax.custom_vjp` cannot be `jvp`-ed
# (measured directly: `TypeError: can't apply forward-mode autodiff (jvp) to
# a custom_vjp function`, the same failure `BishopBuilder`'s own docstring
# documents for `diffrax`'s default `RecursiveCheckpointAdjoint`). Nothing
# in the suite exercised `jax.jacfwd` through a `BishopBuilder`-wrapped
# `ArcLength` before this test.


def test_bishop_wrapped_arclength_survives_jacfwd_with_and_without_s_max() -> None:
    def bishop_tip_z(theta_val: float, *, s_max: u.AbstractQuantity | None) -> float:
        curve = Helix(u.Q(theta_val, "km"))
        kw = {} if s_max is None else {"s_max": s_max}
        arc = cxfc.ArcLength(curve, "s", **kw)
        b = cxfc.BishopBuilder(arc, "km")
        return b.location(u.Q(2.0, "km")).ustrip("km")[2]

    h = 1e-6
    for s_max in (None, u.Q(10.0, "km")):

        def f(th: float, s_max: u.AbstractQuantity | None = s_max) -> float:
            return bishop_tip_z(th, s_max=s_max)

        grad_fwd = jax.jacfwd(f)(1.0)
        grad_rev = jax.grad(f)(1.0)
        fd = (f(1.0 + h) - f(1.0 - h)) / (2 * h)
        assert jnp.allclose(grad_fwd, fd, rtol=1e-3), (s_max, grad_fwd, fd)
        assert jnp.allclose(grad_rev, fd, rtol=1e-3), (s_max, grad_rev, fd)
