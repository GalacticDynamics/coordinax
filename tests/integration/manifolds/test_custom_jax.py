"""Integration tests for CustomManifold under JAX transforms."""

__all__: tuple[str, ...] = ()

import jax
import jax.numpy as jnp
import pytest

import unxt as u

import coordinax.charts as cxc


def _cart2d_to_polar_with_custom(x: u.AbstractQuantity, y: u.AbstractQuantity):
    p = {"x": x, "y": y}
    out = cxc.pt_map(p, cxc.cart2d, cxc.polar2d)
    return out["r"], out["theta"]


class TestCustomManifoldJAX:
    """JAX transform compatibility for custom manifold wrappers."""

    def test_jit_matches_eager(self) -> None:
        """Jit over manifold transition map matches eager execution."""
        x = u.Q(3.0, "m")
        y = u.Q(4.0, "m")

        r_eager, theta_eager = _cart2d_to_polar_with_custom(x, y)
        r_jit, theta_jit = jax.jit(_cart2d_to_polar_with_custom)(x, y)

        assert u.ustrip("m", r_jit) == pytest.approx(u.ustrip("m", r_eager), rel=1e-6)
        assert u.ustrip("rad", theta_jit) == pytest.approx(
            u.ustrip("rad", theta_eager), rel=1e-6
        )

    def test_vmap_batch_radius(self) -> None:
        """Vmap over manifold transition map yields expected radii."""
        xs = u.Q(jnp.array([1.0, 0.0, 3.0]), "m")
        ys = u.Q(jnp.array([0.0, 2.0, 4.0]), "m")

        r_batch, _theta_batch = jax.vmap(_cart2d_to_polar_with_custom)(xs, ys)

        expected = jnp.sqrt(xs.value**2 + ys.value**2)
        assert jnp.allclose(r_batch.value, expected, rtol=1e-6)
