r"""The frame velocity comes from the curve, and only from the curve.

A time-dependent curve frame's connection is $\partial\gamma/\partial t$ at
fixed first argument. Nothing in the library selects the Eulerian or the
Lagrangian reading; the parametrisation the caller hands in already fixes it,
and these tests pin that the library never overrides it.
"""

import jax
import jax.numpy as jnp

import unxt as u

import coordinaxs.curveframes as cxfc

S0 = 1.3
T0 = 1.0


def stretch_and_bend(
    sigma: u.AbstractQuantity, t: u.AbstractQuantity
) -> u.AbstractQuantity:
    """gamma(sigma, t) = (sigma (1 + t/2), t sigma^2 / 10, 0).

    `sigma` is a *material* label: it names the same piece of the curve on
    every slice. The curve both stretches and bends, so arc length is not a
    reparametrisation-invariant relabelling of it.
    """
    sv, tv = sigma.ustrip("km"), t.ustrip("s")
    x = sv * (1.0 + 0.5 * tv)
    y = 0.1 * tv * sv**2
    return u.Q(jnp.stack([x, y, jnp.zeros_like(x)]), "km")


#: d/dt of `stretch_and_bend` at fixed `sigma`, in km/s.
MATERIAL_VELOCITY = jnp.array([0.5 * S0, 0.1 * S0**2, 0.0])


def dt_at_fixed_label(curve, label: float, t: float) -> jax.Array:
    """d(curve)/dt holding the first argument fixed -- whatever it labels."""

    def f(tv: float) -> jax.Array:
        return curve(u.Q(label, "km"), u.Q(tv, "s")).ustrip("km")

    return jax.jacfwd(f)(t)


def test_plain_curve_is_material() -> None:
    """An unwrapped curve's first argument is whatever the caller made it."""
    got = dt_at_fixed_label(stretch_and_bend, S0, T0)
    assert jnp.allclose(got, MATERIAL_VELOCITY)


def test_arclength_is_eulerian() -> None:
    """Wrapping in `ArcLength` re-measures per slice, so the label advects."""
    arc = cxfc.ArcLength(stretch_and_bend, "km")
    got = dt_at_fixed_label(arc, S0, T0)
    assert not jnp.allclose(got, MATERIAL_VELOCITY)


def test_lagrangian_arclength_restores_the_material_velocity() -> None:
    """`LagrangianArcLength`'s label names a material point, so it does not."""
    lag = cxfc.LagrangianArcLength(stretch_and_bend, u.Q(0.0, "s"), "km")
    got = dt_at_fixed_label(lag, S0, T0)
    assert jnp.allclose(got, MATERIAL_VELOCITY, atol=1e-6)
