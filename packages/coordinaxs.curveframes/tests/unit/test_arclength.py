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
    arc = cxfc.ArcLength(helix)

    def pos(s_val):
        return arc(u.Q(s_val, "km")).ustrip("km")

    for s_val in (0.5, 1.0, 3.0):
        speed = jnp.linalg.norm(jax.jacfwd(pos)(s_val))
        assert jnp.allclose(speed, 1.0, atol=1e-8), (s_val, speed)


def test_matches_the_closed_form() -> None:
    """For a constant-speed curve, tau(s) = s / speed exactly."""
    arc = cxfc.ArcLength(helix)
    got = arc(u.Q(2.0, "km")).ustrip("km")
    want = helix(u.Q(2.0 / SPEED, "s")).ustrip("km")
    assert jnp.allclose(got, want, atol=1e-8), (got, want)


def test_differentiable_through_the_solve() -> None:
    """d(tau)/ds = 1/speed, obtained through the ODE by implicit diff."""
    arc = cxfc.ArcLength(helix)

    def z_of_s(s_val):
        return arc(u.Q(s_val, "km")).ustrip("km")[2]

    # z = 0.3 * tau(s), so dz/ds = 0.3 / speed
    assert jnp.allclose(jax.grad(z_of_s)(2.0), 0.3 / SPEED, rtol=1e-6)


def test_s_zero_sits_at_tau_0() -> None:
    arc = cxfc.ArcLength(helix)
    assert jnp.allclose(
        arc(u.Q(0.0, "km")).ustrip("km"), helix(u.Q(0.0, "s")).ustrip("km")
    )
