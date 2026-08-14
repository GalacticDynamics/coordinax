"""Arc-length reparametrisation of a static curve."""

import jax
import jax.numpy as jnp

import unxt as u

import coordinaxs.curveframes as cxfc


def helix(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    """Helix of speed sqrt(1 + 0.3**2) = 1.0440306508910550 km/s."""
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), 0.3 * t]), "km")


SPEED = (1.0 + 0.09) ** 0.5


def test_reparametrised_curve_has_unit_speed() -> None:
    """The defining property: |d(gamma)/ds| == 1 everywhere."""
    arc = cxfc.ArcLength(helix, "s")

    def pos(s_val):
        return arc(u.Q(s_val, "km")).ustrip("km")

    for s_val in (0.5, 1.0, 3.0):
        speed = jnp.linalg.norm(jax.jacfwd(pos)(s_val))
        assert jnp.allclose(speed, 1.0, atol=1e-8), (s_val, speed)


def test_matches_the_closed_form() -> None:
    """For a constant-speed curve, tau(s) = s / speed exactly."""
    arc = cxfc.ArcLength(helix, "s")
    got = arc(u.Q(2.0, "km")).ustrip("km")
    want = helix(u.Q(2.0 / SPEED, "s")).ustrip("km")
    assert jnp.allclose(got, want, atol=1e-8), (got, want)


def test_differentiable_through_the_solve() -> None:
    """d(tau)/ds = 1/speed, obtained through the ODE by implicit diff."""
    arc = cxfc.ArcLength(helix, "s")

    def z_of_s(s_val):
        return arc(u.Q(s_val, "km")).ustrip("km")[2]

    # z = 0.3 * tau(s), so dz/ds = 0.3 / speed
    assert jnp.allclose(jax.grad(z_of_s)(2.0), 0.3 / SPEED, rtol=1e-6)


def test_s_zero_sits_at_tau_0() -> None:
    arc = cxfc.ArcLength(helix, "s")
    assert jnp.allclose(
        arc(u.Q(0.0, "km")).ustrip("km"), helix(u.Q(0.0, "s")).ustrip("km")
    )


def parabola(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    """Non-constant speed: gamma(tau) = (tau, tau**2, 0), speed sqrt(1+4tau**2)."""
    t = tau.ustrip("s")
    return u.Q(jnp.stack([t, t**2, jnp.zeros_like(t)]), "km")


def test_matches_independent_quadrature_for_nonconstant_speed() -> None:
    """Non-constant-speed curve, checked against a reference the ODE never sees.

    A frozen-speed shortcut (evaluating ||gamma'|| once at tau_0 instead of at
    the ODE's live state) would still pass every other test in this file,
    because they all use a constant-speed helix. Here the arc length ``s_val``
    fed to `ArcLength` is itself computed independently, by `jnp.trapezoid`
    quadrature of the true speed from 0 to ``tau_target`` -- not by the ODE
    under test -- so a frozen-speed implementation gets the wrong ``s_val`` ->
    ``tau`` mapping and this test catches it.
    """
    tau_target = 1.3
    t_grid = jnp.linspace(0.0, tau_target, 200_001)
    speed_grid = jnp.sqrt(1.0 + 4.0 * t_grid**2)
    s_val = jnp.trapezoid(speed_grid, t_grid)

    arc = cxfc.ArcLength(parabola, "s")
    got = arc(u.Q(s_val, "km")).ustrip("km")
    want = parabola(u.Q(tau_target, "s")).ustrip("km")
    assert jnp.allclose(got, want, atol=1e-6), (got, want)


def test_tau_unit_other_than_s() -> None:
    """`tau_unit` need not be "s"; here the wrapped curve's parameter is yr."""

    def helix_yr(tau: u.AbstractQuantity) -> u.AbstractQuantity:
        t = tau.ustrip("yr")
        return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), 0.3 * t]), "km")

    arc = cxfc.ArcLength(helix_yr, tau_unit="yr")
    assert arc.tau_unit == u.unit("yr")
    got = arc(u.Q(2.0, "km")).ustrip("km")
    want = helix_yr(u.Q(2.0 / SPEED, "yr")).ustrip("km")
    assert jnp.allclose(got, want, atol=1e-8), (got, want)


def test_custom_tau_0_shifts_the_origin() -> None:
    """s=0 sits at the custom tau_0, and unit speed still holds there."""
    arc = cxfc.ArcLength(helix, "s", tau_0=u.Q(1.0, "s"))
    assert jnp.allclose(
        arc(u.Q(0.0, "km")).ustrip("km"), helix(u.Q(1.0, "s")).ustrip("km")
    )

    def pos(s_val):
        return arc(u.Q(s_val, "km")).ustrip("km")

    speed = jnp.linalg.norm(jax.jacfwd(pos)(1.0))
    assert jnp.allclose(speed, 1.0, atol=1e-8)
