"""Usage tests for ``jac_pt_map`` in ``coordinax.charts``.

Key behavioural contracts verified here:

* **Curried-form workflow**: ``jac_pt_map(from_chart, to_chart, usys=si)``
  returns a callable that can be reused across many base points and produces
  the same result as the direct call.
* **JIT compatibility**: the curried form can be wrapped in ``jax.jit`` and
  still returns the correct result.
* **Vmap compatibility**: ``jax.vmap`` over a batch of base points works with
  the curried form.
* **Chain rule (composition)**: for invertible chart pairs, the product
  ``J_{B→A}(p_B) @ J_{A→B}(p_A)`` is the identity matrix, verified via
  Hypothesis property-based tests.
"""

__all__: tuple[str, ...] = ()

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings
from strategies import (
    any_angle_rad as _any_angle_rad,
    any_m as _any_m,
    polar_rad as _angle_rad,
    pos_m as _pos_m,
)

import quaxed.numpy as qnp
import unxt as u

import coordinax.charts as cxc
from coordinax.internal import QuantityMatrix

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_jacobian_approx(
    J1: QuantityMatrix, J2: QuantityMatrix, *, atol: float = 1e-5
) -> None:
    """Assert two Jacobians agree entry-wise (values only)."""
    np.testing.assert_allclose(
        np.asarray(J1.value),
        np.asarray(J2.value),
        atol=atol,
        err_msg="Jacobian values differ",
    )


# ===========================================================================
# 1. Curried-form workflow
# ===========================================================================


class TestCurriedWorkflow:
    """Test the curried form ``jac_fn = jac_pt_map(from, to, usys=si)``."""

    def test_curried_matches_direct_cart3d_sph3d(self) -> None:
        """Curried Jacobian matches direct call for Cart3D→Sph3D."""
        at = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
        jac_fn = cxc.jac_pt_map(cxc.cart3d, cxc.sph3d, usys=u.unitsystems.si)
        _assert_jacobian_approx(
            jac_fn(at),
            cxc.jac_pt_map(at, cxc.cart3d, cxc.sph3d),
        )

    def test_curried_reuse_across_points(self) -> None:
        """A single curried function can be called at multiple base points."""
        jac_fn = cxc.jac_pt_map(cxc.cart2d, cxc.polar2d, usys=u.unitsystems.si)
        for x, y in [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]:
            at = {"x": u.Q(x, "m"), "y": u.Q(y, "m")}
            J = jac_fn(at)
            assert isinstance(J, QuantityMatrix)
            assert J.value.shape == (2, 2)

    def test_none_partial_matches_direct(self) -> None:
        """None-partial form also matches the direct call."""
        at = {"x": u.Q(3.0, "m"), "y": u.Q(4.0, "m"), "z": u.Q(0.0, "m")}
        fn = cxc.jac_pt_map(None, cxc.cart3d, cxc.sph3d, usys=u.unitsystems.si)
        _assert_jacobian_approx(
            fn(at),
            cxc.jac_pt_map(at, cxc.cart3d, cxc.sph3d),
        )


# ===========================================================================
# 2. JIT compatibility
# ===========================================================================


class TestJITCompatibility:
    """jac_pt_map must work inside jax.jit via the curried form."""

    def test_jit_curried_cart3d_sph3d(self) -> None:
        """JIT the curried form and verify the result."""
        at = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
        jac_fn = jax.jit(cxc.jac_pt_map(cxc.cart3d, cxc.sph3d, usys=u.unitsystems.si))
        J = jac_fn(at)
        assert isinstance(J, QuantityMatrix)
        assert J.value.shape == (3, 3)
        # At (1,0,0): ∂r/∂x = 1
        np.testing.assert_allclose(J.value[0, 0], 1.0, atol=1e-6)

    def test_jit_curried_cart2d_polar2d(self) -> None:
        """JIT the curried form for Cart2D→Polar2D."""
        at = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m")}
        jac_fn = jax.jit(cxc.jac_pt_map(cxc.cart2d, cxc.polar2d, usys=u.unitsystems.si))
        J = jac_fn(at)
        np.testing.assert_allclose(J.value, jnp.eye(2), atol=1e-6)

    def test_jit_direct_call(self) -> None:
        """Direct call JIT-compiled correctly."""
        at = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}

        @jax.jit
        def jitted(at):
            return cxc.jac_pt_map(at, cxc.cart3d, cxc.sph3d)

        J = jitted(at)
        assert isinstance(J, QuantityMatrix)
        np.testing.assert_allclose(J.value[0, 0], 1.0, atol=1e-6)


# ===========================================================================
# 3. Vmap compatibility
# ===========================================================================


class TestVmapCompatibility:
    """jac_pt_map must work within jax.vmap for batched evaluation."""

    def test_vmap_cart3d_sph3d(self) -> None:
        """Vmap over a batch of Cart3D→Sph3D base points."""
        jac_fn = cxc.jac_pt_map(cxc.cart3d, cxc.sph3d, usys=u.unitsystems.si)

        def single(x, y, z):
            return jac_fn({"x": u.Q(x, "m"), "y": u.Q(y, "m"), "z": u.Q(z, "m")})

        xs = jnp.array([1.0, 0.0, 3.0])
        ys = jnp.array([0.0, 1.0, 4.0])
        zs = jnp.array([0.0, 0.0, 0.0])
        Js = jax.vmap(single)(xs, ys, zs)
        assert Js.value.shape == (3, 3, 3)

    def test_vmap_cart2d_polar2d(self) -> None:
        """Vmap over Cart2D→Polar2D points produces batched (N, 2, 2) Jacobians."""
        jac_fn = cxc.jac_pt_map(cxc.cart2d, cxc.polar2d, usys=u.unitsystems.si)

        def single(x, y):
            return jac_fn({"x": u.Q(x, "m"), "y": u.Q(y, "m")})

        xs = jnp.array([1.0, 0.0, 1.0])
        ys = jnp.array([0.0, 1.0, 1.0])
        Js = jax.vmap(single)(xs, ys)
        assert Js.value.shape == (3, 2, 2)
        # At (1, 0): identity
        np.testing.assert_allclose(Js.value[0], jnp.eye(2), atol=1e-6)


# ===========================================================================
# 4. Chain rule: J_inv @ J_fwd = I (using curried form)
# ===========================================================================


class TestChainRuleViaCurriedForm:
    """Using curried Jacobians, J_{B→A} @ J_{A→B} = I for invertible pairs."""

    def _check_composition_identity(self, c1, c2, at_c1, *, atol=1e-5):
        at_c2 = cxc.pt_map(at_c1, c1, c2)
        jac_fwd = cxc.jac_pt_map(c1, c2, usys=u.unitsystems.si)
        jac_inv = cxc.jac_pt_map(c2, c1, usys=u.unitsystems.si)
        J_fwd = jac_fwd(at_c1)
        J_inv = jac_inv(at_c2)
        result = qnp.matmul(J_inv, J_fwd)
        n = len(c1.components)
        np.testing.assert_allclose(result.value, jnp.eye(n), atol=atol)

    def test_cart3d_sph3d_at_x1_y0_z0(self) -> None:
        self._check_composition_identity(
            cxc.cart3d,
            cxc.sph3d,
            {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")},
        )

    def test_cart3d_cyl3d_at_x1_y0_z0(self) -> None:
        self._check_composition_identity(
            cxc.cart3d,
            cxc.cyl3d,
            {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")},
        )

    def test_cart2d_polar2d_at_1_0(self) -> None:
        self._check_composition_identity(
            cxc.cart2d,
            cxc.polar2d,
            {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m")},
        )

    @given(r=_pos_m, theta=_angle_rad, phi=_any_angle_rad)
    @settings(deadline=None)
    def test_cart3d_sph3d_property(self, r, theta, phi) -> None:
        """Property: J_inv @ J_fwd = I for any non-singular Sph3D point."""
        self._check_composition_identity(
            cxc.sph3d,
            cxc.cart3d,
            {"r": r, "theta": theta, "phi": phi},
            atol=1e-4,
        )

    @given(r=_pos_m, phi=_any_angle_rad, z=_any_m)
    @settings(deadline=None)
    def test_cart3d_cyl3d_property(self, r, phi, z) -> None:
        """Property: J_inv @ J_fwd = I for any non-singular Cyl3D point."""
        self._check_composition_identity(
            cxc.cyl3d,
            cxc.cart3d,
            {"rho": r, "phi": phi, "z": z},
            atol=1e-4,
        )
