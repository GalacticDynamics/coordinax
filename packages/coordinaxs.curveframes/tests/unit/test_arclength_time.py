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


def bend(tau: u.AbstractQuantity, t: u.AbstractQuantity) -> u.AbstractQuantity:
    """A curve that *bends* with time: gamma(tau, t) = (tau, t tau^2, 0).

    `stretch` above cannot detect the Eulerian property: it is a straight line
    through the origin, so its arc length is identically its x-displacement on
    every slice. This one is straight at t=0 and increasingly parabolic after,
    so arc length genuinely depends on which slice measures it.
    """
    x = tau.ustrip("s")
    return u.Q(jnp.stack([x, t.ustrip("s") * x**2, jnp.zeros_like(x)]), "km")


def _arclen_by_quadrature(tau_end: float, t_val: float) -> float:
    """Arc length of `bend` from 0 to ``tau_end`` on slice ``t_val``.

    Independent of the ODE under test: speed is ``sqrt(1 + (2 t tau)^2)``.
    """
    grid = jnp.linspace(0.0, tau_end, 200_001)
    return float(jnp.trapezoid(jnp.sqrt(1.0 + (2 * t_val * grid) ** 2), grid))


def test_at_time_binds_the_evaluation_time() -> None:
    curve = cxfc.AtTime(stretch, u.Q(1.0, "s"))
    got = curve(u.Q(2.0, "s")).ustrip("km")
    assert jnp.allclose(got, jnp.array([2.0 * 1.5, 0.0, 0.0]))


def test_arclength_over_a_time_dependent_curve_measures_the_current_slice() -> None:
    """Eulerian: `s` is arc length on the slice being evaluated.

    On `bend`, the same label lands at x = 1.500, 1.009, 0.777 for t = 0, 1,
    2, so the slice genuinely matters. The target arc length comes from
    `_arclen_by_quadrature`, a reference the ODE never sees.
    """
    arc = cxfc.ArcLength(bend)  # still two-argument
    for t_val in (0.0, 1.0, 2.0):
        s_val = _arclen_by_quadrature(1.2, t_val)
        got = cxfc.AtTime(arc, u.Q(t_val, "s"))(u.Q(s_val, "km")).ustrip("km")
        assert jnp.allclose(got[0], 1.2, atol=1e-6), (t_val, got)


def test_the_slice_actually_changes_the_answer() -> None:
    """Guard the fixture itself.

    A straight-line curve would make the test above pass for the wrong reason,
    so pin that `bend` is genuinely slice-sensitive.
    """
    arc = cxfc.ArcLength(bend)
    xs = [
        float(cxfc.AtTime(arc, u.Q(t, "s"))(u.Q(1.5, "km")).ustrip("km")[0])
        for t in (0.0, 1.0, 2.0)
    ]
    assert xs[0] > xs[1] > xs[2], xs


def test_binding_time_first_is_still_supported_for_static_use() -> None:
    """`ArcLength(AtTime(...))` is legal -- it is just the frozen slice."""
    frozen = cxfc.ArcLength(cxfc.AtTime(stretch, u.Q(0.0, "s")))
    got = frozen(u.Q(1.0, "km")).ustrip("km")
    assert jnp.allclose(got, jnp.array([1.0, 0.0, 0.0]), atol=1e-6)


def test_a_live_time_is_a_leaf() -> None:
    """A live `Quantity` `t` contributes exactly one more leaf than a static one."""
    import jax

    dyn = cxfc.AtTime(stretch, u.Q(1.0, "s"))
    sta = cxfc.AtTime(stretch, u.StaticQuantity(1.0, "s"))
    assert len(jax.tree.leaves(dyn)) - len(jax.tree.leaves(sta)) == 1


def test_lagrangian_labels_ride_with_the_material_point() -> None:
    """The label is arc length at t0, and thereafter names the same point.

    The material point at tau=1 is at x=1 when t=0 and x=1.5 when t=1. Its
    Lagrangian label stays 1.0 throughout.
    """
    lag = cxfc.LagrangianArcLength(stretch, u.Q(0.0, "s"))
    assert jnp.allclose(
        cxfc.AtTime(lag, u.Q(0.0, "s"))(u.Q(1.0, "km")).ustrip("km"),
        jnp.array([1.0, 0.0, 0.0]),
        atol=1e-6,
    )
    assert jnp.allclose(
        cxfc.AtTime(lag, u.Q(1.0, "s"))(u.Q(1.0, "km")).ustrip("km"),
        jnp.array([1.5, 0.0, 0.0]),
        atol=1e-6,
    )


def test_the_two_readings_differ_under_stretching() -> None:
    """Separate the two readings, at label 1 evaluated at t=1.

    Eulerian gives x=1.0 (one unit along the *current* curve); Lagrangian
    gives x=1.5 (the material point, which has moved). A rigid motion would
    NOT separate them -- it leaves every arc length untouched -- so the curve
    must stretch.
    """
    eul = cxfc.AtTime(cxfc.ArcLength(stretch), u.Q(1.0, "s"))
    lag = cxfc.AtTime(cxfc.LagrangianArcLength(stretch, u.Q(0.0, "s")), u.Q(1.0, "s"))
    x_eul = eul(u.Q(1.0, "km")).ustrip("km")[0]
    x_lag = lag(u.Q(1.0, "km")).ustrip("km")[0]
    assert jnp.allclose(x_eul, 1.0, atol=1e-6), x_eul
    assert jnp.allclose(x_lag, 1.5, atol=1e-6), x_lag


def test_a_rigid_motion_does_not_separate_them() -> None:
    """Pins why the discriminating test must stretch rather than rotate."""

    def rot(tau: u.AbstractQuantity, t: u.AbstractQuantity) -> u.AbstractQuantity:
        a = tau.ustrip("s") + t.ustrip("s")
        return u.Q(jnp.stack([jnp.cos(a), jnp.sin(a), jnp.zeros_like(a)]), "km")

    eul = cxfc.AtTime(cxfc.ArcLength(rot), u.Q(1.0, "s"))
    lag = cxfc.AtTime(cxfc.LagrangianArcLength(rot, u.Q(0.0, "s")), u.Q(1.0, "s"))
    assert jnp.allclose(
        eul(u.Q(0.7, "km")).ustrip("km"), lag(u.Q(0.7, "km")).ustrip("km"), atol=1e-6
    )
