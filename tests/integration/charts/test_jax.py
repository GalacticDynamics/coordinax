"""Integration tests for coordinax.charts with JAX transforms.

Key behavioural contracts verified here:

* ``jax.jit`` traces through coordinate transformations without error, and the
  compiled result agrees with the eager result.
* ``jax.vmap`` maps a per-point transformation over a batch of coordinates and
  produces results consistent with transforming each point individually.
* ``jax.grad`` differentiates through coordinate transformations; the computed
  Jacobian entries match known analytical values.
* The three JAX transforms compose: ``jit(vmap(fn))`` and
  ``jit(grad(fn))`` both work correctly.
"""

__all__: tuple[str, ...] = ()

import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, settings, strategies as st

import unxt as u

import coordinax.charts as cxc

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_pos_floats = st.floats(min_value=0.5, max_value=5, width=32, allow_nan=False)
_any_floats = st.floats(min_value=-5, max_value=5, width=32, allow_nan=False)


def _cart3d_to_sph3d(x, y, z):
    """Apply cart3d → sph3d on scalar component values."""
    p = {"x": x, "y": y, "z": z}
    result = cxc.pt_map(p, cxc.cart3d, cxc.sph3d)
    return result["r"], result["theta"], result["phi"]


def _cart3d_to_cyl3d(x, y, z):
    """Apply cart3d → cyl3d on scalar component values."""
    p = {"x": x, "y": y, "z": z}
    result = cxc.pt_map(p, cxc.cart3d, cxc.cyl3d)
    return result["rho"], result["phi"], result["z"]


def _cart2d_to_polar(x, y):
    """Apply cart2d → polar2d on scalar component values."""
    p = {"x": x, "y": y}
    result = cxc.pt_map(p, cxc.cart2d, cxc.polar2d)
    return result["r"], result["theta"]


# ---------------------------------------------------------------------------
# jit compatibility
# ---------------------------------------------------------------------------


class TestJITCompatibility:
    """jax.jit traces through coordinate transformations correctly."""

    def test_jit_cart3d_to_sph3d_matches_eager(self) -> None:
        """jit(cart3d → sph3d) gives the same result as the eager call."""
        x, y, z = u.Q(3, "m"), u.Q(4, "m"), u.Q(0, "m")

        r_eager, theta_eager, phi_eager = _cart3d_to_sph3d(x, y, z)
        r_jit, theta_jit, phi_jit = jax.jit(_cart3d_to_sph3d)(x, y, z)

        assert u.ustrip("m", r_jit) == pytest.approx(u.ustrip("m", r_eager), rel=1e-6)
        assert u.ustrip("rad", theta_jit) == pytest.approx(
            u.ustrip("rad", theta_eager), rel=1e-6
        )
        assert u.ustrip("rad", phi_jit) == pytest.approx(
            u.ustrip("rad", phi_eager), rel=1e-6
        )

    def test_jit_cart3d_to_cyl3d_matches_eager(self) -> None:
        """jit(cart3d → cyl3d) gives the same result as the eager call."""
        x, y, z = u.Q(3, "m"), u.Q(4, "m"), u.Q(5, "m")

        rho_eager, phi_eager, z_eager = _cart3d_to_cyl3d(x, y, z)
        rho_jit, phi_jit, z_jit = jax.jit(_cart3d_to_cyl3d)(x, y, z)

        assert u.ustrip("m", rho_jit) == pytest.approx(
            u.ustrip("m", rho_eager), rel=1e-6
        )
        assert u.ustrip("rad", phi_jit) == pytest.approx(
            u.ustrip("rad", phi_eager), rel=1e-6
        )
        assert u.ustrip("m", z_jit) == pytest.approx(u.ustrip("m", z_eager), rel=1e-6)

    def test_jit_identity_map(self) -> None:
        """Jit over the identity map (cart3d → cart3d) works."""

        def identity(x, y, z):
            p = {"x": x, "y": y, "z": z}
            return cxc.pt_map(p, cxc.cart3d, cxc.cart3d)

        x, y, z = u.Q(1, "m"), u.Q(2, "m"), u.Q(3, "m")
        result = jax.jit(identity)(x, y, z)
        assert u.ustrip("m", result["x"]) == pytest.approx(1)
        assert u.ustrip("m", result["y"]) == pytest.approx(2)
        assert u.ustrip("m", result["z"]) == pytest.approx(3)

    @given(x=_any_floats, y=_any_floats, z=_pos_floats)
    @settings(deadline=None)
    def test_jit_agrees_with_eager_property(self, x: float, y: float, z: float) -> None:
        """Property: jit gives the same r as eager for cart3d → sph3d."""
        x_q, y_q, z_q = u.Q(x, "m"), u.Q(y, "m"), u.Q(z, "m")
        r_eager, _, _ = _cart3d_to_sph3d(x_q, y_q, z_q)
        r_jit, _, _ = jax.jit(_cart3d_to_sph3d)(x_q, y_q, z_q)
        assert u.ustrip("m", r_jit) == pytest.approx(u.ustrip("m", r_eager), rel=1e-5)


# ---------------------------------------------------------------------------
# vmap compatibility
# ---------------------------------------------------------------------------


class TestVmapCompatibility:
    """jax.vmap maps per-point transformations over batches correctly."""

    def test_vmap_cart3d_to_sph3d_matches_individual(self) -> None:
        """vmap(cart3d → sph3d) over a batch matches element-wise results."""
        xs = u.Q(jnp.array([1, 0, 0]), "m")
        ys = u.Q(jnp.array([0, 1, 0]), "m")
        zs = u.Q(jnp.array([0, 0, 1]), "m")

        r_batch, _, _ = jax.vmap(_cart3d_to_sph3d)(xs, ys, zs)

        # Each of the standard basis vectors has r=1
        assert jnp.allclose(r_batch.value, jnp.ones(3), atol=1e-6)

    def test_vmap_cart3d_to_cyl3d_shape(self) -> None:
        """Vmap output has the correct batch shape."""
        n = 5
        xs = u.Q(jnp.ones(n), "m")
        ys = u.Q(jnp.zeros(n), "m")
        zs = u.Q(jnp.zeros(n), "m")

        rho_batch, phi_batch, z_batch = jax.vmap(_cart3d_to_cyl3d)(xs, ys, zs)

        assert rho_batch.shape == (n,)
        assert phi_batch.shape == (n,)
        assert z_batch.shape == (n,)

    def test_vmap_matches_looped_individual(self) -> None:
        """Vmap result == list of individual calls stacked together."""
        xs = u.Q(jnp.array([3, 0, 0]), "m")
        ys = u.Q(jnp.array([0, 4, 0]), "m")
        zs = u.Q(jnp.array([0, 0, 5]), "m")

        r_vmap, theta_vmap, _phi_vmap = jax.vmap(_cart3d_to_sph3d)(xs, ys, zs)

        for i in range(3):
            r_i, theta_i, _phi_i = _cart3d_to_sph3d(
                u.Q(xs.value[i], "m"), u.Q(ys.value[i], "m"), u.Q(zs.value[i], "m")
            )
            assert u.ustrip("m", r_vmap)[i] == pytest.approx(
                u.ustrip("m", r_i), rel=1e-5
            )
            assert u.ustrip("rad", theta_vmap)[i] == pytest.approx(
                u.ustrip("rad", theta_i), rel=1e-5
            )

    @given(
        xs=st.lists(_any_floats, min_size=3, max_size=3),
        ys=st.lists(_any_floats, min_size=3, max_size=3),
        zs=st.lists(_pos_floats, min_size=3, max_size=3),
    )
    @settings(deadline=None)
    def test_vmap_r_equals_norm(self, xs: list, ys: list, zs: list) -> None:
        """vmap: r == ||xyz|| for every point in the batch."""
        xs_q = u.Q(jnp.array(xs, dtype=jnp.float32), "m")
        ys_q = u.Q(jnp.array(ys, dtype=jnp.float32), "m")
        zs_q = u.Q(jnp.array(zs, dtype=jnp.float32), "m")

        r_batch, _, _ = jax.vmap(_cart3d_to_sph3d)(xs_q, ys_q, zs_q)
        expected_r = jnp.sqrt(
            jnp.array(xs, dtype=jnp.float32) ** 2
            + jnp.array(ys, dtype=jnp.float32) ** 2
            + jnp.array(zs, dtype=jnp.float32) ** 2
        )
        assert jnp.allclose(r_batch.value, expected_r, rtol=1e-4)


# ---------------------------------------------------------------------------
# grad compatibility
# ---------------------------------------------------------------------------


class TestGradCompatibility:
    """jax.grad differentiates through coordinate transformations correctly."""

    # dr/dx = x/r.  At (x=1,y=0,z=0): r=1, so dr/dx = 1.
    def test_grad_r_wrt_x_at_unit_x(self) -> None:
        """dr/dx = 1 at (1, 0, 0) m — agrees with the analytic Jacobian."""

        def r_value(x_val):
            p = {"x": u.Q(x_val, "m"), "y": u.Q(0, "m"), "z": u.Q(0, "m")}
            return cxc.pt_map(p, cxc.cart3d, cxc.sph3d)["r"].value

        dr_dx = jax.grad(r_value)(1.0)
        assert float(dr_dx) == pytest.approx(1, rel=1e-5)

    # drho/dx = x/rho.  At (3,4,0): rho=5, so drho/dx = 3/5 = 0.6.
    def test_grad_rho_wrt_x_at_known_point(self) -> None:
        """drho/dx = x/rho — checked at (3, 4, 0) m."""

        def rho_value(x_val):
            p = {"x": u.Q(x_val, "m"), "y": u.Q(4, "m"), "z": u.Q(0, "m")}
            return cxc.pt_map(p, cxc.cart3d, cxc.cyl3d)["rho"].value

        drho_dx = jax.grad(rho_value)(3.0)
        assert float(drho_dx) == pytest.approx(0.6, rel=1e-5)

    # r_2d = sqrt(x^2 + y^2).  dr/dx = x/r.  At (1,0): r=1, dr/dx=1.
    def test_grad_polar_r_wrt_x(self) -> None:
        """d(polar r)/dx = 1 at (1, 0) m."""

        def r_polar(x_val):
            p = {"x": u.Q(x_val, "m"), "y": u.Q(0, "m")}
            return cxc.pt_map(p, cxc.cart2d, cxc.polar2d)["r"].value

        dr_dx = jax.grad(r_polar)(1.0)
        assert float(dr_dx) == pytest.approx(1, rel=1e-5)

    @given(x=_pos_floats, z=_pos_floats)
    @settings(deadline=None)
    def test_grad_r_equals_x_over_r_property(self, x: float, z: float) -> None:
        """Property: dr/dx = x/r (analytical Jacobian of spherical r)."""

        def r_value(x_val):
            p = {"x": u.Q(x_val, "m"), "y": u.Q(0, "m"), "z": u.Q(z, "m")}
            return cxc.pt_map(p, cxc.cart3d, cxc.sph3d)["r"].value

        dr_dx = jax.grad(r_value)(x)
        expected = x / jnp.sqrt(x**2 + z**2)
        assert float(dr_dx) == pytest.approx(float(expected), rel=1e-4)


# ---------------------------------------------------------------------------
# Composed transforms: jit + vmap, jit + grad
# ---------------------------------------------------------------------------


class TestComposedTransforms:
    """JAX transforms compose correctly with coordinate transformations."""

    def test_jit_vmap(self) -> None:
        """jit(vmap(fn)) works and gives the same result as vmap(fn)."""
        xs = u.Q(jnp.array([1, 2, 3]), "m")
        ys = u.Q(jnp.zeros(3), "m")
        zs = u.Q(jnp.zeros(3), "m")

        r_vmap, _, _ = jax.vmap(_cart3d_to_sph3d)(xs, ys, zs)
        r_jit_vmap, _, _ = jax.jit(jax.vmap(_cart3d_to_sph3d))(xs, ys, zs)

        assert jnp.allclose(r_vmap.value, r_jit_vmap.value, rtol=1e-5)

    def test_jit_grad(self) -> None:
        """jit(grad(fn)) works and gives the same result as grad(fn)."""

        def r_value(x_val):
            p = {"x": u.Q(x_val, "m"), "y": u.Q(0, "m"), "z": u.Q(0, "m")}
            return cxc.pt_map(p, cxc.cart3d, cxc.sph3d)["r"].value

        dr_dx_grad = jax.grad(r_value)(1.0)
        dr_dx_jit_grad = jax.jit(jax.grad(r_value))(1.0)

        assert float(dr_dx_jit_grad) == pytest.approx(float(dr_dx_grad), rel=1e-5)
