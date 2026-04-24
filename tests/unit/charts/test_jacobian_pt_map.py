"""Tests for ``jac_pt_map`` in ``coordinax.charts``."""

__all__: tuple[str, ...] = ()

import jaxtyping

import jax
import jax.numpy as jnp
import numpy as np
import pytest
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


def _jac_via_autodiff(from_chart, to_chart, at_qty):
    """Reference: compute Jacobian via jax.jacfwd applied to pt_map.

    Returns a nested dict  jac[out_k][in_k] of plain JAX scalars (units stripped).
    We strip units from ``at_qty`` for the plain-array reference path.
    """
    at_plain = {k: v.value for k, v in at_qty.items()}

    def pt_fn(q):
        return {
            k: v.value
            for k, v in cxc.pt_map(
                {kk: u.Q(vv, at_qty[kk].unit) for kk, vv in q.items()},
                from_chart,
                to_chart,
            ).items()
        }

    return jax.jacfwd(pt_fn)(at_plain)


# ===========================================================================
# 1. Importability
# ===========================================================================


class TestJacobianPtMapImportable:
    """jac_pt_map is importable from coordinax.charts."""

    def test_importable_from_charts(self) -> None:
        assert hasattr(cxc, "jac_pt_map")
        assert callable(cxc.jac_pt_map)


# ===========================================================================
# 2. Return type and shape
# ===========================================================================


class TestJacobianPtMapReturnType:
    """Returns a 2-D QuantityMatrix with shape (n_to, n_from)."""

    @pytest.mark.parametrize(
        ("from_chart", "to_chart", "at", "expected_shape"),
        [
            (
                cxc.cart2d,
                cxc.polar2d,
                {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m")},
                (2, 2),
            ),
            (
                cxc.cart3d,
                cxc.sph3d,
                {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")},
                (3, 3),
            ),
            (
                cxc.cart3d,
                cxc.cyl3d,
                {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")},
                (3, 3),
            ),
            (
                cxc.sph3d,
                cxc.cart3d,
                {
                    "r": u.Q(1.0, "m"),
                    "theta": u.Q(jnp.pi / 2, "rad"),
                    "phi": u.Q(0.0, "rad"),
                },
                (3, 3),
            ),
        ],
    )
    def test_returns_QuantityMatrix(
        self, from_chart, to_chart, at, expected_shape
    ) -> None:
        J = cxc.jac_pt_map(at, from_chart, to_chart)
        assert isinstance(J, QuantityMatrix)
        assert J.ndim == 2
        assert J.value.shape == expected_shape


# ===========================================================================
# 3. Unit structure
# ===========================================================================


class TestJacobianPtMapUnits:
    """J[j, i].unit = to_chart_dim_j / from_chart_dim_i."""

    def test_cart3d_to_sph3d_row0_dimensionless(self) -> None:
        """J[r, *] : m/m → dimensionless (r row, x/y/z columns)."""
        at = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
        J = cxc.jac_pt_map(at, cxc.cart3d, cxc.sph3d)
        # r has units m, x/y/z have units m → m/m = dimensionless
        for i in range(3):
            assert J.unit[0, i] == u.unit("") or J.unit[0, i] == u.unit("m/m"), (
                f"J[r, {i}] unit should be dimensionless, got {J.unit[0, i]}"
            )

    def test_cart3d_to_sph3d_rows1_and_2_are_rad_per_m(self) -> None:
        """J[θ, *] and J[φ, *] : rad/m (angle output, length input)."""
        at = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
        J = cxc.jac_pt_map(at, cxc.cart3d, cxc.sph3d)
        rad_per_m = u.unit("rad/m")
        for i in range(3):
            assert J.unit[1, i] == rad_per_m, (
                f"J[θ, {i}] unit: expected rad/m, got {J.unit[1, i]}"
            )
            assert J.unit[2, i] == rad_per_m, (
                f"J[φ, {i}] unit: expected rad/m, got {J.unit[2, i]}"
            )

    def test_cart3d_to_cyl3d_phi_row_is_rad_per_m(self) -> None:
        """J[φ, *] for Cyl3D : rad/m."""
        at = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
        J = cxc.jac_pt_map(at, cxc.cart3d, cxc.cyl3d)
        # cyl3d components: (rho=m, phi=rad, z=m); cart3d: (x=m, y=m, z=m)
        # Row 1 (phi): rad output / m input → rad/m
        for i in range(3):
            assert J.unit[1, i] == u.unit("rad/m"), (
                f"J[φ, {i}] should be rad/m, got {J.unit[1, i]}"
            )

    def test_sph3d_to_cart3d_theta_col_is_m_per_rad(self) -> None:
        """J[*, θ] for Sph3D → Cart3D : m/rad (length output / angle input)."""
        at = {
            "r": u.Q(1.0, "m"),
            "theta": u.Q(jnp.pi / 2, "rad"),
            "phi": u.Q(0.0, "rad"),
        }
        J = cxc.jac_pt_map(at, cxc.sph3d, cxc.cart3d)
        # cart3d: (x=m, y=m, z=m); sph3d: (r=m, theta=rad, phi=rad)
        # Column 1 (theta): m output / rad input → m/rad
        for j in range(3):
            assert J.unit[j, 1] == u.unit("m/rad"), (
                f"J[{j}, θ] should be m/rad, got {J.unit[j, 1]}"
            )


# ===========================================================================
# 4. Known values: Cart2D → Polar2D
# ===========================================================================


class TestJacobianPtMapCart2dToPolar2d:
    r"""Analytical Jacobian: Cart2D → Polar2D.

    Coordinate maps: r = sqrt(x²+y²),  θ = atan2(y, x).

    Jacobian  J = [[∂r/∂x,  ∂r/∂y ],
                   [∂θ/∂x,  ∂θ/∂y ]]

              = [[ x/r,           y/r         ],
                 [ -y/(x²+y²),   x/(x²+y²)   ]]

    Specific evaluations (all rows: r, θ; all columns: x, y):

        At (1, 0): r=1, θ=0
          J = [[1,  0],
               [0,  1]]          identity

        At (0, 1): r=1, θ=π/2
          J = [[0,  1],
               [-1, 0]]          90° rotation (negative)

        At (1, 1): r=√2, θ=π/4
          J = [[1/√2,  1/√2],
               [-1/2,  1/2 ]]
    """

    def test_at_1_0_identity(self) -> None:
        """At (x=1, y=0) the Jacobian is the 2x2 identity."""
        at = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m")}
        J = cxc.jac_pt_map(at, cxc.cart2d, cxc.polar2d)
        np.testing.assert_allclose(J.value[0, 0], 1.0, atol=1e-6)  # ∂r/∂x
        np.testing.assert_allclose(J.value[0, 1], 0.0, atol=1e-6)  # ∂r/∂y
        np.testing.assert_allclose(J.value[1, 0], 0.0, atol=1e-6)  # ∂θ/∂x
        np.testing.assert_allclose(J.value[1, 1], 1.0, atol=1e-6)  # ∂θ/∂y

    def test_at_0_1(self) -> None:
        """At (x=0, y=1) J = [[0, 1], [-1, 0]]."""
        at = {"x": u.Q(0.0, "m"), "y": u.Q(1.0, "m")}
        J = cxc.jac_pt_map(at, cxc.cart2d, cxc.polar2d)
        np.testing.assert_allclose(J.value[0, 0], 0.0, atol=1e-6)  # ∂r/∂x
        np.testing.assert_allclose(J.value[0, 1], 1.0, atol=1e-6)  # ∂r/∂y
        np.testing.assert_allclose(J.value[1, 0], -1.0, atol=1e-6)  # ∂θ/∂x
        np.testing.assert_allclose(J.value[1, 1], 0.0, atol=1e-6)  # ∂θ/∂y

    def test_at_1_1(self) -> None:
        """At (x=1, y=1) J = [[1/√2, 1/√2], [-1/2, 1/2]]."""
        at = {"x": u.Q(1.0, "m"), "y": u.Q(1.0, "m")}
        J = cxc.jac_pt_map(at, cxc.cart2d, cxc.polar2d)
        invsq2 = float(jnp.sqrt(0.5))
        np.testing.assert_allclose(J.value[0, 0], invsq2, atol=1e-6)  # ∂r/∂x = 1/√2
        np.testing.assert_allclose(J.value[0, 1], invsq2, atol=1e-6)  # ∂r/∂y = 1/√2
        np.testing.assert_allclose(J.value[1, 0], -0.5, atol=1e-6)  # ∂θ/∂x = -1/2
        np.testing.assert_allclose(J.value[1, 1], 0.5, atol=1e-6)  # ∂θ/∂y = 1/2


# ===========================================================================
# 5. Known values: Polar2D → Cart2D
# ===========================================================================


class TestJacobianPtMapPolar2dToCart2d:
    r"""Analytical Jacobian: Polar2D → Cart2D.

    Coordinate maps: x = r cos(θ),  y = r sin(θ).

    Jacobian  J = [[∂x/∂r,  ∂x/∂θ ],
                   [∂y/∂r,  ∂y/∂θ ]]

              = [[ cos(θ),  -r sin(θ) ],
                 [ sin(θ),   r cos(θ) ]]

    At (r=1, θ=0) = (x=1, y=0):
        J = [[1,  0],
             [0,  1]]             identity

    At (r=1, θ=π/2) = (x=0, y=1):
        J = [[0, -1],
             [1,  0]]

    At (r=2, θ=π/4) = (x=√2, y=√2):
        J = [[1/√2,  -√2],
             [1/√2,   √2]]
    """

    def test_at_r1_theta0_identity(self) -> None:
        """At (r=1, θ=0) J is the 2x2 identity."""
        at = {"r": u.Q(1.0, "m"), "theta": u.Q(0.0, "rad")}
        J = cxc.jac_pt_map(at, cxc.polar2d, cxc.cart2d)
        np.testing.assert_allclose(J.value[0, 0], 1.0, atol=1e-6)  # ∂x/∂r
        np.testing.assert_allclose(J.value[0, 1], 0.0, atol=1e-6)  # ∂x/∂θ
        np.testing.assert_allclose(J.value[1, 0], 0.0, atol=1e-6)  # ∂y/∂r
        np.testing.assert_allclose(J.value[1, 1], 1.0, atol=1e-6)  # ∂y/∂θ

    def test_at_r1_theta_pi2(self) -> None:
        """At (r=1, θ=π/2) J = [[0, -1], [1, 0]]."""
        at = {"r": u.Q(1.0, "m"), "theta": u.Q(float(jnp.pi / 2), "rad")}
        J = cxc.jac_pt_map(at, cxc.polar2d, cxc.cart2d)
        np.testing.assert_allclose(J.value[0, 0], 0.0, atol=1e-6)  # cos(π/2) ≈ 0
        np.testing.assert_allclose(J.value[0, 1], -1.0, atol=1e-6)  # -r sin(π/2) = -1
        np.testing.assert_allclose(J.value[1, 0], 1.0, atol=1e-6)  # sin(π/2) = 1
        np.testing.assert_allclose(J.value[1, 1], 0.0, atol=1e-6)  # r cos(π/2) ≈ 0

    def test_at_r2_theta_pi4(self) -> None:
        """At (r=2, θ=π/4): J = [[1/√2, -√2], [1/√2, √2]]."""
        at = {"r": u.Q(2.0, "m"), "theta": u.Q(float(jnp.pi / 4), "rad")}
        J = cxc.jac_pt_map(at, cxc.polar2d, cxc.cart2d)
        invsq2 = float(jnp.sqrt(0.5))
        sq2 = float(jnp.sqrt(2.0))
        np.testing.assert_allclose(J.value[0, 0], invsq2, atol=1e-6)  # cos(π/4)
        np.testing.assert_allclose(J.value[0, 1], -sq2, atol=1e-6)  # -2 sin(π/4) = -√2
        np.testing.assert_allclose(J.value[1, 0], invsq2, atol=1e-6)  # sin(π/4)
        np.testing.assert_allclose(J.value[1, 1], sq2, atol=1e-6)  # 2 cos(π/4) = √2


# ===========================================================================
# 6. Known values: Cart3D → Sph3D
# ===========================================================================


class TestJacobianPtMapCart3dToSph3d:
    r"""Analytical Jacobian: Cart3D → Sph3D.

    Physics convention: x = r sinθ cosφ,  y = r sinθ sinφ,  z = r cosθ.

    Inverse: r = sqrt(x²+y²+z²),  θ = arccos(z/r),  φ = atan2(y, x).

    Jacobian rows: (r, θ, φ);  columns: (x, y, z).

        ∂r/∂x = x/r,   ∂r/∂y = y/r,   ∂r/∂z = z/r

        Let ρ = sqrt(x²+y²) (cylindrical radius):
        ∂θ/∂x = x z/(r² ρ),   ∂θ/∂y = y z/(r² ρ),   ∂θ/∂z = -ρ/r²

        ∂φ/∂x = -y/(x²+y²),   ∂φ/∂y = x/(x²+y²),   ∂φ/∂z = 0

    At (x=1, y=0, z=0)  →  r=1, θ=π/2, φ=0,  ρ=1:
        J = [[1,  0,   0],
             [0,  0,  -1],
             [0,  1,   0]]

    At (x=0, y=1, z=0)  →  r=1, θ=π/2, φ=π/2,  ρ=1:
        J = [[0,  1,   0],
             [0,  0,  -1],
             [-1, 0,   0]]
    """

    def _check_jac(self, at, expected):
        J = cxc.jac_pt_map(at, cxc.cart3d, cxc.sph3d)
        np.testing.assert_allclose(J.value, expected, atol=1e-6)

    def test_at_x1_y0_z0(self) -> None:
        """At (1, 0, 0): J = [[1,0,0],[0,0,-1],[0,1,0]]."""
        at = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
        expected = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        self._check_jac(at, expected)

    def test_at_x0_y1_z0(self) -> None:
        """At (0, 1, 0): J = [[0,1,0],[0,0,-1],[-1,0,0]]."""
        at = {"x": u.Q(0.0, "m"), "y": u.Q(1.0, "m"), "z": u.Q(0.0, "m")}
        expected = jnp.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]])
        self._check_jac(at, expected)

    def test_at_general_point(self) -> None:
        """At (3, 4, 0): r=5, θ=π/2, φ=atan2(4,3).

        ∂r/∂x = 3/5 = 0.6,   ∂r/∂y = 4/5 = 0.8,   ∂r/∂z = 0
        ρ = 5 (since z=0)
        ∂θ/∂x = xz/(r²ρ) = 0,  ∂θ/∂y = yz/(r²ρ) = 0,  ∂θ/∂z = -ρ/r² = -5/25 = -0.2
        ∂φ/∂x = -y/(x²+y²) = -4/25 = -0.16
        ∂φ/∂y = x/(x²+y²)  =  3/25 =  0.12
        ∂φ/∂z = 0
        """
        at = {"x": u.Q(3.0, "m"), "y": u.Q(4.0, "m"), "z": u.Q(0.0, "m")}
        expected = jnp.array(
            [
                [0.6, 0.8, 0.0],
                [0.0, 0.0, -0.2],
                [-0.16, 0.12, 0.0],
            ]
        )
        self._check_jac(at, expected)


# ===========================================================================
# 7. Known values: Sph3D → Cart3D
# ===========================================================================


class TestJacobianPtMapSph3dToCart3d:
    r"""Analytical Jacobian: Sph3D → Cart3D.

    Forward map: x = r sinθ cosφ,  y = r sinθ sinφ,  z = r cosθ.

    Jacobian rows: (x, y, z);  columns: (r, θ, φ).

        ∂x/∂r = sinθ cosφ,  ∂x/∂θ = r cosθ cosφ,  ∂x/∂φ = -r sinθ sinφ
        ∂y/∂r = sinθ sinφ,  ∂y/∂θ = r cosθ sinφ,  ∂y/∂φ =  r sinθ cosφ
        ∂z/∂r = cosθ,        ∂z/∂θ = -r sinθ,       ∂z/∂φ = 0

    At (r=1, θ=π/2, φ=0)  →  (x=1, y=0, z=0):
        J = [[ 1,   0,  0],
             [ 0,   0,  1],
             [ 0,  -1,  0]]
    """

    def test_at_r1_theta_pi2_phi0(self) -> None:
        """At (r=1, θ=π/2, φ=0): J = [[1,0,0],[0,0,1],[0,-1,0]]."""
        at = {
            "r": u.Q(1.0, "m"),
            "theta": u.Q(float(jnp.pi / 2), "rad"),
            "phi": u.Q(0.0, "rad"),
        }
        J = cxc.jac_pt_map(at, cxc.sph3d, cxc.cart3d)
        expected = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
        np.testing.assert_allclose(J.value, expected, atol=1e-5)

    def test_at_general_point(self) -> None:
        """At (r=1, θ=π/3, φ=π/4): verify via forward formula.

        sinθ = √3/2, cosθ = 1/2, cosφ = 1/√2, sinφ = 1/√2.

        ∂x/∂r = (√3/2)(1/√2) = √3/(2√2)
        ∂x/∂θ = 1 · (1/2)(1/√2) = 1/(2√2)
        ∂x/∂φ = -1 · (√3/2)(1/√2) = -√3/(2√2)
        """
        t, p = float(jnp.pi / 3), float(jnp.pi / 4)
        at = {"r": u.Q(1.0, "m"), "theta": u.Q(t, "rad"), "phi": u.Q(p, "rad")}
        J = cxc.jac_pt_map(at, cxc.sph3d, cxc.cart3d)
        exp_dxdr = float(jnp.sin(t) * jnp.cos(p))
        exp_dxdtheta = float(1.0 * jnp.cos(t) * jnp.cos(p))
        exp_dxdphi = float(-1.0 * jnp.sin(t) * jnp.sin(p))
        np.testing.assert_allclose(J.value[0, 0], exp_dxdr, atol=1e-5)
        np.testing.assert_allclose(J.value[0, 1], exp_dxdtheta, atol=1e-5)
        np.testing.assert_allclose(J.value[0, 2], exp_dxdphi, atol=1e-5)


# ===========================================================================
# 8. Known values: Cart3D → Cyl3D
# ===========================================================================


class TestJacobianPtMapCart3dToCyl3d:
    r"""Analytical Jacobian: Cart3D → Cyl3D.

    Maps: ρ = sqrt(x²+y²),  φ = atan2(y, x),  z = z.

    Jacobian rows: (ρ, φ, z);  columns: (x, y, z).

        ∂ρ/∂x = x/ρ,   ∂ρ/∂y = y/ρ,   ∂ρ/∂z = 0
        ∂φ/∂x = -y/ρ², ∂φ/∂y = x/ρ²,  ∂φ/∂z = 0
        ∂z/∂x = 0,      ∂z/∂y = 0,      ∂z/∂z = 1

    At (x=1, y=0, z=0): ρ=1, φ=0.
        J = [[1,  0,  0],
             [0,  1,  0],
             [0,  0,  1]]              identity

    At (x=0, y=1, z=2): ρ=1, φ=π/2.
        J = [[0,  1,  0],
             [-1, 0,  0],
             [0,  0,  1]]
    """

    def test_at_x1_y0_z0_is_identity(self) -> None:
        """At (1, 0, 0) the Jacobian is the 3x3 identity."""
        at = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
        J = cxc.jac_pt_map(at, cxc.cart3d, cxc.cyl3d)
        np.testing.assert_allclose(J.value, jnp.eye(3), atol=1e-6)

    def test_at_x0_y1_z2(self) -> None:
        """At (0, 1, 2): J = [[0,1,0],[-1,0,0],[0,0,1]]."""
        at = {"x": u.Q(0.0, "m"), "y": u.Q(1.0, "m"), "z": u.Q(2.0, "m")}
        J = cxc.jac_pt_map(at, cxc.cart3d, cxc.cyl3d)
        expected = jnp.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(J.value, expected, atol=1e-6)


# ===========================================================================
# 9. Known values: Cyl3D → Cart3D
# ===========================================================================


class TestJacobianPtMapCyl3dToCart3d:
    r"""Analytical Jacobian: Cyl3D → Cart3D.

    Forward map: x = ρ cosφ,  y = ρ sinφ,  z = z.

    Jacobian rows: (x, y, z);  columns: (ρ, φ, z).

        ∂x/∂ρ =  cosφ,  ∂x/∂φ = -ρ sinφ,  ∂x/∂z = 0
        ∂y/∂ρ =  sinφ,  ∂y/∂φ =  ρ cosφ,  ∂y/∂z = 0
        ∂z/∂ρ = 0,       ∂z/∂φ = 0,         ∂z/∂z = 1

    At (ρ=1, φ=0, z=0) = (x=1, y=0, z=0):
        J = [[1,  0,  0],
             [0,  1,  0],
             [0,  0,  1]]              identity

    At (ρ=1, φ=π/2, z=2) = (x=0, y=1, z=2):
        J = [[0, -1,  0],
             [1,  0,  0],
             [0,  0,  1]]
    """

    def test_at_rho1_phi0_z0_is_identity(self) -> None:
        """At (ρ=1, φ=0, z=0) J is the 3x3 identity."""
        at = {"rho": u.Q(1.0, "m"), "phi": u.Q(0.0, "rad"), "z": u.Q(0.0, "m")}
        J = cxc.jac_pt_map(at, cxc.cyl3d, cxc.cart3d)
        np.testing.assert_allclose(J.value, jnp.eye(3), atol=1e-6)

    def test_at_rho1_phi_pi2_z2(self) -> None:
        """At (ρ=1, φ=π/2, z=2): J = [[0,-1,0],[1,0,0],[0,0,1]]."""
        at = {
            "rho": u.Q(1.0, "m"),
            "phi": u.Q(float(jnp.pi / 2), "rad"),
            "z": u.Q(2.0, "m"),
        }
        J = cxc.jac_pt_map(at, cxc.cyl3d, cxc.cart3d)
        expected = jnp.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(J.value, expected, atol=1e-5)


# ===========================================================================
# 10. Property: composition = identity
# ===========================================================================


class TestJacobianPtMapCompositionProperty:
    r"""Property: J_{C2→C1}(p_{C2}) @ J_{C1→C2}(p_{C1}) = I.

    This is the chain rule: the Jacobian of the round-trip is the identity.
    Uses QuantityMatrix matmul (quaxed) which tracks units through the product.
    The result has all-dimensionless units and values equal to the nxn identity.
    """

    def _check_composition_identity(self, c1, c2, at_c1):
        """Helper: check J_{c2→c1} @ J_{c1→c2} ≈ I."""
        at_c2 = cxc.pt_map(at_c1, c1, c2)
        J_fwd = cxc.jac_pt_map(at_c1, c1, c2)
        J_inv = cxc.jac_pt_map(at_c2, c2, c1)
        result = qnp.matmul(J_inv, J_fwd)
        n = len(c1.components)
        np.testing.assert_allclose(
            result.value,
            jnp.eye(n),
            atol=1e-5,
            err_msg=f"J_{{{c2}→{c1}}} @ J_{{{c1}→{c2}}} ≠ I",
        )

    def test_cart2d_polar2d_at_1_0(self) -> None:
        """Cart2D ↔ Polar2D at (1, 0): composition = I."""
        self._check_composition_identity(
            cxc.cart2d,
            cxc.polar2d,
            {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m")},
        )

    def test_cart2d_polar2d_at_1_1(self) -> None:
        """Cart2D ↔ Polar2D at (1, 1): composition = I."""
        self._check_composition_identity(
            cxc.cart2d,
            cxc.polar2d,
            {"x": u.Q(1.0, "m"), "y": u.Q(1.0, "m")},
        )

    def test_cart3d_sph3d_at_x1_y0_z0(self) -> None:
        """Cart3D ↔ Sph3D at (1, 0, 0): composition = I."""
        self._check_composition_identity(
            cxc.cart3d,
            cxc.sph3d,
            {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")},
        )

    def test_cart3d_cyl3d_at_x1_y0_z0(self) -> None:
        """Cart3D ↔ Cyl3D at (1, 0, 0): composition = I."""
        self._check_composition_identity(
            cxc.cart3d,
            cxc.cyl3d,
            {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")},
        )

    @given(r=_pos_m, theta=_angle_rad, phi=_any_angle_rad)
    @settings(deadline=None)
    def test_cart3d_sph3d_property(self, r, theta, phi) -> None:
        """Property: J_{Sph→Cart} @ J_{Cart→Sph} = I for any non-singular point."""
        p_sph = {"r": r, "theta": theta, "phi": phi}
        p_cart = cxc.pt_map(p_sph, cxc.sph3d, cxc.cart3d)
        J_fwd = cxc.jac_pt_map(p_cart, cxc.cart3d, cxc.sph3d)
        J_inv = cxc.jac_pt_map(p_sph, cxc.sph3d, cxc.cart3d)
        result = qnp.matmul(J_inv, J_fwd)
        np.testing.assert_allclose(result.value, jnp.eye(3), atol=1e-4)

    @given(r=_pos_m, phi=_any_angle_rad, z=_any_m)
    @settings(deadline=None)
    def test_cart3d_cyl3d_property(self, r, phi, z) -> None:
        """Property: J_{Cyl→Cart} @ J_{Cart→Cyl} = I for any non-singular point."""
        p_cyl = {"rho": r, "phi": phi, "z": z}
        p_cart = cxc.pt_map(p_cyl, cxc.cyl3d, cxc.cart3d)
        J_fwd = cxc.jac_pt_map(p_cart, cxc.cart3d, cxc.cyl3d)
        J_inv = cxc.jac_pt_map(p_cyl, cxc.cyl3d, cxc.cart3d)
        result = qnp.matmul(J_inv, J_fwd)
        np.testing.assert_allclose(result.value, jnp.eye(3), atol=1e-4)

    @given(r=_pos_m, theta=_any_angle_rad)
    @settings(deadline=None)
    def test_cart2d_polar2d_property(self, r, theta) -> None:
        """Property: J_{Polar→Cart} @ J_{Cart→Polar} = I for r > 0."""
        p_polar = {"r": r, "theta": theta}
        p_cart = cxc.pt_map(p_polar, cxc.polar2d, cxc.cart2d)
        J_fwd = cxc.jac_pt_map(p_cart, cxc.cart2d, cxc.polar2d)
        J_inv = cxc.jac_pt_map(p_polar, cxc.polar2d, cxc.cart2d)
        result = qnp.matmul(J_inv, J_fwd)
        np.testing.assert_allclose(result.value, jnp.eye(2), atol=1e-4)


# ===========================================================================
# 11. Property: agrees with jax.jacfwd applied to pt_map
# ===========================================================================


class TestJacobianPtMapAgreesWithJacfwd:
    """Values of jac_pt_map must match jax.jacfwd(pt_map) numerically.

    This is the gold-standard check: any hand-coded Jacobian must agree
    with the automatic-differentiation reference at the same point.
    """

    def _check_agrees(self, from_chart, to_chart, at_qty, *, atol=1e-5):
        J = cxc.jac_pt_map(at_qty, from_chart, to_chart)
        ref = _jac_via_autodiff(from_chart, to_chart, at_qty)
        out_keys = list(to_chart.components)
        in_keys = list(from_chart.components)
        for j, ok in enumerate(out_keys):
            for i, ik in enumerate(in_keys):
                np.testing.assert_allclose(
                    float(J.value[j, i]),
                    float(ref[ok][ik]),
                    atol=atol,
                    err_msg=f"J[{ok}, {ik}] mismatch vs jacfwd",
                )

    def test_cart2d_to_polar2d_at_1_0(self) -> None:
        self._check_agrees(
            cxc.cart2d, cxc.polar2d, {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m")}
        )

    def test_cart2d_to_polar2d_at_1_1(self) -> None:
        self._check_agrees(
            cxc.cart2d, cxc.polar2d, {"x": u.Q(1.0, "m"), "y": u.Q(1.0, "m")}
        )

    def test_polar2d_to_cart2d_at_r1_theta_pi3(self) -> None:
        self._check_agrees(
            cxc.polar2d,
            cxc.cart2d,
            {"r": u.Q(1.0, "m"), "theta": u.Q(float(jnp.pi / 3), "rad")},
        )

    def test_cart3d_to_sph3d_at_x1_y0_z0(self) -> None:
        self._check_agrees(
            cxc.cart3d,
            cxc.sph3d,
            {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")},
        )

    def test_cart3d_to_sph3d_at_x3_y4_z0(self) -> None:
        self._check_agrees(
            cxc.cart3d,
            cxc.sph3d,
            {"x": u.Q(3.0, "m"), "y": u.Q(4.0, "m"), "z": u.Q(0.0, "m")},
        )

    def test_sph3d_to_cart3d_at_r1_theta_pi2_phi0(self) -> None:
        self._check_agrees(
            cxc.sph3d,
            cxc.cart3d,
            {
                "r": u.Q(1.0, "m"),
                "theta": u.Q(float(jnp.pi / 2), "rad"),
                "phi": u.Q(0.0, "rad"),
            },
        )

    def test_cart3d_to_cyl3d_at_x0_y1_z2(self) -> None:
        self._check_agrees(
            cxc.cart3d,
            cxc.cyl3d,
            {"x": u.Q(0.0, "m"), "y": u.Q(1.0, "m"), "z": u.Q(2.0, "m")},
        )

    def test_cyl3d_to_cart3d_at_rho1_phi_pi4_z1(self) -> None:
        self._check_agrees(
            cxc.cyl3d,
            cxc.cart3d,
            {
                "rho": u.Q(1.0, "m"),
                "phi": u.Q(float(jnp.pi / 4), "rad"),
                "z": u.Q(1.0, "m"),
            },
        )

    @given(r=_pos_m, theta=_angle_rad, phi=_any_angle_rad)
    @settings(deadline=None)
    def test_cart3d_sph3d_property(self, r, theta, phi) -> None:
        """Analytical J agrees with jacfwd for any non-singular sph3d point."""
        p_sph = {"r": r, "theta": theta, "phi": phi}
        p_cart = cxc.pt_map(p_sph, cxc.sph3d, cxc.cart3d)
        self._check_agrees(cxc.cart3d, cxc.sph3d, p_cart, atol=1e-4)

    @given(r=_pos_m, phi=_any_angle_rad, z=_any_m)
    @settings(deadline=None)
    def test_cart3d_cyl3d_property(self, r, phi, z) -> None:
        """Analytical J agrees with jacfwd for any non-singular cyl3d point."""
        p_cyl = {"rho": r, "phi": phi, "z": z}
        p_cart = cxc.pt_map(p_cyl, cxc.cyl3d, cxc.cart3d)
        self._check_agrees(cxc.cart3d, cxc.cyl3d, p_cart, atol=1e-4)


# ===========================================================================
# 12. JAX compatibility
# ===========================================================================


class TestJacobianPtMapJAXCompatibility:
    """jac_pt_map must be usable inside jax.jit and jax.vmap."""

    def test_jit_cart3d_to_sph3d(self) -> None:
        """JIT compilation: jac_pt_map is traceable."""
        at = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}

        @jax.jit
        def jitted(at):
            return cxc.jac_pt_map(at, cxc.cart3d, cxc.sph3d)

        J = jitted(at)
        assert isinstance(J, QuantityMatrix)
        np.testing.assert_allclose(J.value[0, 0], 1.0, atol=1e-6)  # ∂r/∂x at (1,0,0)

    def test_jit_cart2d_to_polar2d(self) -> None:
        """JIT compilation: 2D case."""
        at = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m")}

        @jax.jit
        def jitted(at):
            return cxc.jac_pt_map(at, cxc.cart2d, cxc.polar2d)

        J = jitted(at)
        np.testing.assert_allclose(J.value, jnp.eye(2), atol=1e-6)

    def test_vmap_over_base_points_cart2d_polar2d(self) -> None:
        """Vmap over a batch of base points — all produce valid 2x2 Jacobians."""
        xs = jnp.array([1.0, 0.0, 1.0])
        ys = jnp.array([0.0, 1.0, 1.0])

        def single(x, y):
            at = {"x": u.Q(x, "m"), "y": u.Q(y, "m")}
            return cxc.jac_pt_map(at, cxc.cart2d, cxc.polar2d)

        batched = jax.vmap(single)(xs, ys)
        assert batched.value.shape == (3, 2, 2)
        # At (1, 0): identity; check first element of batch
        np.testing.assert_allclose(batched.value[0], jnp.eye(2), atol=1e-6)


# ===========================================================================
# 13. Curried and None-partial forms
# ===========================================================================


class TestJacobianPtMapCurriedForms:
    """Curried and None-partial forms match direct call."""

    def test_curried_returns_callable(self) -> None:
        """jac_pt_map(from_chart, to_chart, usys=si) returns a callable."""
        fn = cxc.jac_pt_map(cxc.cart3d, cxc.sph3d, usys=u.unitsystems.si)
        assert callable(fn)

    def test_curried_result_matches_direct(self) -> None:
        """Curried form result matches direct call."""
        at = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
        fn = cxc.jac_pt_map(cxc.cart3d, cxc.sph3d, usys=u.unitsystems.si)
        J_curried = fn(at)
        J_direct = cxc.jac_pt_map(at, cxc.cart3d, cxc.sph3d)
        np.testing.assert_allclose(J_curried.value, J_direct.value, atol=1e-6)

    def test_none_partial_returns_callable(self) -> None:
        """jac_pt_map(None, from_chart, to_chart, usys=si) returns a callable."""
        fn = cxc.jac_pt_map(None, cxc.cart3d, cxc.sph3d, usys=u.unitsystems.si)
        assert callable(fn)

    def test_none_partial_result_matches_direct(self) -> None:
        """None-partial form result matches direct call."""
        at = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
        fn = cxc.jac_pt_map(None, cxc.cart3d, cxc.sph3d, usys=u.unitsystems.si)
        J_partial = fn(at)
        J_direct = cxc.jac_pt_map(at, cxc.cart3d, cxc.sph3d)
        np.testing.assert_allclose(J_partial.value, J_direct.value, atol=1e-6)

    def test_curried_2d(self) -> None:
        """Curried form works for 2D chart pair."""
        at = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m")}
        fn = cxc.jac_pt_map(cxc.cart2d, cxc.polar2d, usys=u.unitsystems.si)
        J = fn(at)
        assert isinstance(J, QuantityMatrix)
        assert J.value.shape == (2, 2)


# ===========================================================================
# 14. Plain Array dispatch (requires usys)
# ===========================================================================


class TestJacobianPtMapArrayInput:
    """Plain Array input dispatches to an Array output (dispatch 3)."""

    def test_array_input_returns_array(self) -> None:
        """Plain array in → plain array out with usys."""
        at = jnp.array([1.0, 0.0, 0.0])
        J = cxc.jac_pt_map(at, cxc.cart3d, cxc.sph3d, usys=u.unitsystems.si)
        assert isinstance(J, jnp.ndarray)
        assert J.shape == (3, 3)

    def test_array_input_values_match_direct(self) -> None:
        """Array dispatch values agree with CDict quantity dispatch."""
        at_arr = jnp.array([1.0, 0.0, 0.0])
        at_qty = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
        J_arr = cxc.jac_pt_map(at_arr, cxc.cart3d, cxc.sph3d, usys=u.unitsystems.si)
        J_qty = cxc.jac_pt_map(at_qty, cxc.cart3d, cxc.sph3d)
        np.testing.assert_allclose(J_arr, J_qty.value, atol=1e-6)

    def test_array_2d(self) -> None:
        """Plain array dispatch works for Cart2D → Polar2D."""
        at = jnp.array([1.0, 0.0])
        J = cxc.jac_pt_map(at, cxc.cart2d, cxc.polar2d, usys=u.unitsystems.si)
        assert isinstance(J, jnp.ndarray)
        assert J.shape == (2, 2)


# ===========================================================================
# 15. CDict is_array branch (plain array values)
# ===========================================================================


class TestJacobianPtMapCDictArrayBranch:
    """CDict with plain array values — the is_array=True branch in the generic dispatch.

    The generic CDict dispatch (for chart pairs without a dedicated CDict overload,
    e.g. Cart3D→Sph3D) branches on whether values are quantities or plain arrays.
    Plain-array CDicts require usys because the branch forwards to the plain-Array
    dispatch which requires a unit system.

    Note: Cart2D→Polar2D has its own CDict dispatch that always calls pack_to_qmatrix,
    so plain-array CDicts for that pair fail even before the is_array check.
    """

    def test_generic_pair_with_usys(self) -> None:
        """Cart3D→Sph3D CDict with plain floats and usys provided → Array output."""
        at = {"x": jnp.array(1.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}
        J = cxc.jac_pt_map(at, cxc.cart3d, cxc.sph3d, usys=u.unitsystems.si)
        assert J.shape == (3, 3)

    def test_generic_pair_no_usys_fails(self) -> None:
        """Cart3D→Sph3D CDict with plain floats and no usys raises an error.

        The CDict is_array=True branch forwards to the Array dispatch which
        requires usys.  For generic chart pairs with no analytical Array
        dispatch this is a known limitation.
        """
        at = {"x": jnp.array(1.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}
        with pytest.raises(jaxtyping.TypeCheckError, match="usys"):
            cxc.jac_pt_map(at, cxc.cart3d, cxc.sph3d)
