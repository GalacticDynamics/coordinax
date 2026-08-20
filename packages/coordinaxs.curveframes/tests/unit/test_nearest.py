"""The nearest-point solve, independent of any chart."""

import jax
import jax.numpy as jnp
import pytest

import unxt as u

import coordinaxs.curveframes as cxfc
from coordinaxs.curveframes import nearest_tau


def helix(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), 0.3 * t]), "km")


BOUNDS = (u.Q(-1.0, "s"), u.Q(6.0, "s"))


def _point_at(builder, tau_v, n1, n2):
    """Build an ambient point exactly on the normal plane at ``tau_v``."""
    tau = u.Q(tau_v, "s")
    R = builder.rotation_matrix(tau)
    g = builder.location(tau).ustrip("km")
    return u.Q(g + n1 * R[1] + n2 * R[2], "km")


@pytest.mark.parametrize("builder_cls", [cxfc.BishopBuilder, cxfc.FrenetSerretBuilder])
def test_recovers_the_generating_tau(builder_cls) -> None:
    b = builder_cls(helix, "s")
    x = _point_at(b, 0.7, 0.13, -0.21)
    got = nearest_tau(b, x, bounds=BOUNDS)
    assert jnp.allclose(got.ustrip("s"), 0.7, atol=1e-6)


def test_finds_the_nearest_root_not_merely_a_stationary_one() -> None:
    """A bare Newton converges to whichever root its guess falls into.

    The helix has stationary points of the distance near tau = 3.99 and
    -2.87; the seeded scan must still return 0.7.
    """
    b = cxfc.BishopBuilder(helix, "s")
    x = _point_at(b, 0.7, 0.13, -0.21)
    got = nearest_tau(b, x, bounds=(u.Q(-4.0, "s"), u.Q(12.0, "s")), n_seed=128)
    assert jnp.allclose(got.ustrip("s"), 0.7, atol=1e-6)


def test_is_jittable() -> None:
    b = cxfc.BishopBuilder(helix, "s")
    x = _point_at(b, 0.7, 0.13, -0.21)
    got = jax.jit(lambda xx: nearest_tau(b, xx, bounds=BOUNDS))(x)
    assert jnp.allclose(got.ustrip("s"), 0.7, atol=1e-6)


def test_nonconvergence_raises_eagerly() -> None:
    """An unreachable tolerance forces `max_steps_reached`, not a silent tau.

    `rtol`/`atol` this tight are unreachable at any dtype, so this is a
    deterministic way to exercise the non-convergence guard independent of
    the default-tolerance dtype scaling.
    """
    b = cxfc.BishopBuilder(helix, "s")
    x = _point_at(b, 0.7, 0.13, -0.21)
    with pytest.raises(RuntimeError, match="did not converge"):
        nearest_tau(b, x, bounds=BOUNDS, rtol=1e-300, atol=1e-300)


def test_polish_does_not_walk_onto_a_local_maximum() -> None:
    """Regression for the counterexample in `nearest_tau`'s docstring.

    An unconstrained polish can leave the scan's basin for a local maximum
    instead of the minimiser.
    """

    def wavy(tau: u.AbstractQuantity) -> u.AbstractQuantity:
        t = tau.ustrip("s")
        return u.Q(jnp.stack([t, 0.6 * jnp.sin(6 * t), jnp.zeros_like(t)]), "km")

    b = cxfc.FrenetSerretBuilder(wavy, "s")
    x = u.Q(jnp.array([4.4, -1.4, 0.0]), "km")
    got = nearest_tau(b, x, bounds=(u.Q(-5.0, "s"), u.Q(5.0, "s")), n_seed=64)

    assert jnp.allclose(got.ustrip("s"), 3.95275, atol=1e-4)
    assert not jnp.allclose(got.ustrip("s"), 3.37962, atol=1e-2)


def test_nonconvergence_raises_under_jit() -> None:
    """The eager and traced guards are different code paths; test both.

    A bare `eqx.error_if` whose result is unused is dead-code-eliminated, so
    the traced branch can pass silently while the eager branch still works.
    """
    b = cxfc.BishopBuilder(helix, "s")
    x = _point_at(b, 0.7, 0.13, -0.21)

    @jax.jit
    def run(xx):
        return nearest_tau(b, xx, bounds=BOUNDS, rtol=1e-300, atol=1e-300)

    with pytest.raises(jax.errors.JaxRuntimeError, match="did not converge"):
        run(x)
