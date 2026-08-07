"""Tests for ``jac_pt_map`` in ``coordinax.charts``."""

__all__: tuple[str, ...] = ()

import itertools
import math

import jaxtyping

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from numpy.testing import assert_allclose

import quaxed.numpy as qnp
import unxt as u
import unxts.linalg as ul

import coordinax.charts as cxc
import coordinaxs.hypothesis.main as cxst

usys_si = u.unitsystems.si

#: Chart pairs whose Jacobians are checked in both directions.
#:
#: Points are drawn in the *curvilinear* member and mapped into whichever chart
#: the Jacobian is taken from. Drawing in the Cartesian member instead would
#: need a rejection filter for the origin and the polar axis, which the
#: curvilinear domains exclude by construction.
CHART_PAIRS = [
    pytest.param(cxc.cart2d, cxc.polar2d, id="cart2d-polar2d"),
    pytest.param(cxc.cart3d, cxc.sph3d, id="cart3d-sph3d"),
    pytest.param(cxc.cart3d, cxc.cyl3d, id="cart3d-cyl3d"),
]

#: Keeps points well conditioned for a derivative comparison.
#:
#: The upper bound is the one that matters for `jacfwd` agreement at
#: ``atol=1e-4``: a float32 ULP at ``1.8e19 m`` is ~2e12, so the assertion
#: would be meaningless there. The lower bound matters just as much for the
#: curvilinear charts, where Jacobian entries scale like ``1/r`` and are
#: unusable as ``r`` approaches the origin.
WELL_CONDITIONED = (0.5, 8.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jac_via_autodiff(from_chart, to_chart, at_qty):
    """Reference: compute Jacobian via jax.jacfwd applied to pt_map.

    Returns a nested dict  jac[out_k][in_k] of plain JAX scalars (units stripped).
    We strip units from ``at_qty`` for the plain-array reference path.
    """
    at_plain = {k: jnp.asarray(v.value, dtype=float) for k, v in at_qty.items()}

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
        ("from_chart", "to_chart", "at", "exp_shape"),
        [
            (cxc.cart2d, cxc.polar2d, {"x": u.Q(1, "m"), "y": u.Q(0, "m")}, (2, 2)),
            (
                cxc.cart3d,
                cxc.sph3d,
                {"x": u.Q(1, "m"), "y": u.Q(0, "m"), "z": u.Q(0, "m")},
                (3, 3),
            ),
            (
                cxc.cart3d,
                cxc.cyl3d,
                {"x": u.Q(1, "m"), "y": u.Q(0, "m"), "z": u.Q(0, "m")},
                (3, 3),
            ),
            (
                cxc.sph3d,
                cxc.cart3d,
                {
                    "r": u.Q(1, "m"),
                    "theta": u.Q(jnp.pi / 2, "rad"),
                    "phi": u.Q(0, "rad"),
                },
                (3, 3),
            ),
        ],
    )
    def test_returns_QuantityMatrix(self, from_chart, to_chart, at, exp_shape) -> None:
        J = cxc.jac_pt_map(at, from_chart, to_chart)
        assert isinstance(J, ul.QuantityMatrix)
        assert J.ndim == 2
        assert J.value.shape == exp_shape


# ===========================================================================
# 3. Unit structure
# ===========================================================================


class TestJacobianPtMapUnits:
    """``J.unit[i, j] == to_chart_unit_i / from_chart_unit_j``, every cell.

    The four hand-written checks this replaces each pinned one row or one
    column of one pair at one hardcoded point -- so ``cart2d <-> polar2d`` had
    no unit coverage at all, and no reverse direction but one was checked.

    The expected unit is read off `pt_map`'s output rather than written out by
    hand, so the assertion stays honest without a table of units to maintain:
    `pt_map` and `jac_pt_map` do their unit bookkeeping independently.
    """

    @pytest.mark.parametrize(("chart_a", "chart_b"), CHART_PAIRS)
    @pytest.mark.parametrize("forward", [True, False], ids=["fwd", "rev"])
    @given(data=st.data())
    @settings(deadline=None, max_examples=10)
    def test_every_cell_is_out_over_in(
        self, chart_a, chart_b, forward, data: st.DataObject
    ) -> None:
        curv_pt = data.draw(cxst.cdicts(chart_b, magnitude=WELL_CONDITIONED))
        from_chart, to_chart = (chart_a, chart_b) if forward else (chart_b, chart_a)
        at = cxc.pt_map(curv_pt, chart_b, from_chart)

        J = cxc.jac_pt_map(at, from_chart, to_chart)
        out = cxc.pt_map(at, from_chart, to_chart)

        for i, out_comp in enumerate(to_chart.components):
            for j, in_comp in enumerate(from_chart.components):
                expected = out[out_comp].unit / at[in_comp].unit
                assert J.unit[i, j] == expected, (
                    f"J[{out_comp}, {in_comp}]: expected {expected}, got {J.unit[i, j]}"
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
        at = {"x": u.Q(1, "m"), "y": u.Q(0, "m")}
        J = cxc.jac_pt_map(at, cxc.cart2d, cxc.polar2d)
        assert_allclose(J.value[0, 0], 1, atol=1e-6)  # ∂r/∂x
        assert_allclose(J.value[0, 1], 0, atol=1e-6)  # ∂r/∂y
        assert_allclose(J.value[1, 0], 0, atol=1e-6)  # ∂θ/∂x
        assert_allclose(J.value[1, 1], 1, atol=1e-6)  # ∂θ/∂y

    def test_at_0_1(self) -> None:
        """At (x=0, y=1) J = [[0, 1], [-1, 0]]."""
        at = {"x": u.Q(0, "m"), "y": u.Q(1, "m")}
        J = cxc.jac_pt_map(at, cxc.cart2d, cxc.polar2d)
        assert_allclose(J.value[0, 0], 0, atol=1e-6)  # ∂r/∂x
        assert_allclose(J.value[0, 1], 1, atol=1e-6)  # ∂r/∂y
        assert_allclose(J.value[1, 0], -1, atol=1e-6)  # ∂θ/∂x
        assert_allclose(J.value[1, 1], 0, atol=1e-6)  # ∂θ/∂y

    def test_at_1_1(self) -> None:
        """At (x=1, y=1) J = [[1/√2, 1/√2], [-1/2, 1/2]]."""
        at = {"x": u.Q(1, "m"), "y": u.Q(1, "m")}
        J = cxc.jac_pt_map(at, cxc.cart2d, cxc.polar2d)
        invsq2 = jnp.sqrt(0.5)
        assert_allclose(J.value[0, 0], invsq2, atol=1e-6)  # ∂r/∂x = 1/√2
        assert_allclose(J.value[0, 1], invsq2, atol=1e-6)  # ∂r/∂y = 1/√2
        assert_allclose(J.value[1, 0], -0.5, atol=1e-6)  # ∂θ/∂x = -1/2
        assert_allclose(J.value[1, 1], 0.5, atol=1e-6)  # ∂θ/∂y = 1/2


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
        at = {"r": u.Q(1, "m"), "theta": u.Q(0, "rad")}
        J = cxc.jac_pt_map(at, cxc.polar2d, cxc.cart2d)
        assert_allclose(J.value[0, 0], 1, atol=1e-6)  # ∂x/∂r
        assert_allclose(J.value[0, 1], 0, atol=1e-6)  # ∂x/∂θ
        assert_allclose(J.value[1, 0], 0, atol=1e-6)  # ∂y/∂r
        assert_allclose(J.value[1, 1], 1, atol=1e-6)  # ∂y/∂θ

    def test_at_r1_theta_pi2(self) -> None:
        """At (r=1, θ=π/2) J = [[0, -1], [1, 0]]."""
        at = {"r": u.Q(1, "m"), "theta": u.Q(jnp.pi / 2, "rad")}
        J = cxc.jac_pt_map(at, cxc.polar2d, cxc.cart2d)
        assert_allclose(J.value[0, 0], 0, atol=1e-6)  # cos(π/2) ≈ 0
        assert_allclose(J.value[0, 1], -1, atol=1e-6)  # -r sin(π/2) = -1
        assert_allclose(J.value[1, 0], 1, atol=1e-6)  # sin(π/2) = 1
        assert_allclose(J.value[1, 1], 0, atol=1e-6)  # r cos(π/2) ≈ 0

    def test_at_r2_theta_pi4(self) -> None:
        """At (r=2, θ=π/4): J = [[1/√2, -√2], [1/√2, √2]]."""
        at = {"r": u.Q(2, "m"), "theta": u.Q(jnp.pi / 4, "rad")}
        J = cxc.jac_pt_map(at, cxc.polar2d, cxc.cart2d)
        invsq2 = jnp.sqrt(0.5)
        sq2 = jnp.sqrt(2)
        assert_allclose(J.value[0, 0], invsq2, atol=1e-6)  # cos(π/4)
        assert_allclose(J.value[0, 1], -sq2, atol=1e-6)  # -2 sin(π/4) = -√2
        assert_allclose(J.value[1, 0], invsq2, atol=1e-6)  # sin(π/4)
        assert_allclose(J.value[1, 1], sq2, atol=1e-6)  # 2 cos(π/4) = √2


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

    def _check_jac(self, at, exp):
        J = cxc.jac_pt_map(at, cxc.cart3d, cxc.sph3d)
        assert_allclose(J.value, exp, atol=1e-6)

    def test_at_x1_y0_z0(self) -> None:
        """At (1, 0, 0): J = [[1,0,0],[0,0,-1],[0,1,0]]."""
        at = {"x": u.Q(1, "m"), "y": u.Q(0, "m"), "z": u.Q(0, "m")}
        exp = jnp.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        self._check_jac(at, exp)

    def test_at_x0_y1_z0(self) -> None:
        """At (0, 1, 0): J = [[0,1,0],[0,0,-1],[-1,0,0]]."""
        at = {"x": u.Q(0, "m"), "y": u.Q(1, "m"), "z": u.Q(0, "m")}
        exp = jnp.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
        self._check_jac(at, exp)

    def test_at_general_point(self) -> None:
        """At (3, 4, 0): r=5, θ=π/2, φ=atan2(4,3).

        ∂r/∂x = 3/5 = 0.6,   ∂r/∂y = 4/5 = 0.8,   ∂r/∂z = 0
        ρ = 5 (since z=0)
        ∂θ/∂x = xz/(r²ρ) = 0,  ∂θ/∂y = yz/(r²ρ) = 0,  ∂θ/∂z = -ρ/r² = -5/25 = -0.2
        ∂φ/∂x = -y/(x²+y²) = -4/25 = -0.16
        ∂φ/∂y = x/(x²+y²)  =  3/25 =  0.12
        ∂φ/∂z = 0
        """
        at = {"x": u.Q(3, "m"), "y": u.Q(4, "m"), "z": u.Q(0, "m")}
        exp = jnp.array([[0.6, 0.8, 0], [0, 0, -0.2], [-0.16, 0.12, 0]])
        self._check_jac(at, exp)


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
        at = {"r": u.Q(1, "m"), "theta": u.Q(jnp.pi / 2, "rad"), "phi": u.Q(0, "rad")}
        J = cxc.jac_pt_map(at, cxc.sph3d, cxc.cart3d)
        exp = jnp.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        assert_allclose(J.value, exp, atol=1e-5)

    def test_at_general_point(self) -> None:
        """At (r=1, θ=π/3, φ=π/4): verify via forward formula.

        sinθ = √3/2, cosθ = 1/2, cosφ = 1/√2, sinφ = 1/√2.

        ∂x/∂r = (√3/2)(1/√2) = √3/(2√2)
        ∂x/∂θ = 1 · (1/2)(1/√2) = 1/(2√2)
        ∂x/∂φ = -1 · (√3/2)(1/√2) = -√3/(2√2)
        """
        t, p = jnp.pi / 3, jnp.pi / 4
        at = {"r": u.Q(1, "m"), "theta": u.Q(t, "rad"), "phi": u.Q(p, "rad")}
        J = cxc.jac_pt_map(at, cxc.sph3d, cxc.cart3d)
        exp_dxdr = jnp.sin(t) * jnp.cos(p)
        exp_dxdtheta = jnp.cos(t) * jnp.cos(p)
        exp_dxdphi = -jnp.sin(t) * jnp.sin(p)
        assert_allclose(J.value[0, 0], exp_dxdr, atol=1e-5)
        assert_allclose(J.value[0, 1], exp_dxdtheta, atol=1e-5)
        assert_allclose(J.value[0, 2], exp_dxdphi, atol=1e-5)


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
        at = {"x": u.Q(1, "m"), "y": u.Q(0, "m"), "z": u.Q(0, "m")}
        J = cxc.jac_pt_map(at, cxc.cart3d, cxc.cyl3d)
        assert_allclose(J.value, jnp.eye(3), atol=1e-6)

    def test_at_x0_y1_z2(self) -> None:
        """At (0, 1, 2): J = [[0,1,0],[-1,0,0],[0,0,1]]."""
        at = {"x": u.Q(0, "m"), "y": u.Q(1, "m"), "z": u.Q(2, "m")}
        J = cxc.jac_pt_map(at, cxc.cart3d, cxc.cyl3d)
        exp = jnp.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        assert_allclose(J.value, exp, atol=1e-6)


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
        at = {"rho": u.Q(1, "m"), "phi": u.Q(0, "rad"), "z": u.Q(0, "m")}
        J = cxc.jac_pt_map(at, cxc.cyl3d, cxc.cart3d)
        assert_allclose(J.value, jnp.eye(3), atol=1e-6)

    def test_at_rho1_phi_pi2_z2(self) -> None:
        """At (ρ=1, φ=π/2, z=2): J = [[0,-1,0],[1,0,0],[0,0,1]]."""
        at = {"rho": u.Q(1, "m"), "phi": u.Q(jnp.pi / 2, "rad"), "z": u.Q(2, "m")}
        J = cxc.jac_pt_map(at, cxc.cyl3d, cxc.cart3d)
        exp = jnp.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        assert_allclose(J.value, exp, atol=1e-5)


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
        assert_allclose(
            result.value,
            jnp.eye(n),
            atol=1e-5,
            err_msg=f"J_{{{c2}→{c1}}} @ J_{{{c1}→{c2}}} ≠ I",
        )

    @pytest.mark.parametrize(("cart", "curv"), CHART_PAIRS)
    @given(data=st.data())
    @settings(deadline=None)
    def test_composition_is_the_identity(
        self, cart: cxc.AbstractChart, curv: cxc.AbstractChart, data: st.DataObject
    ) -> None:
        """J_{curv->cart} @ J_{cart->curv} = I at an arbitrary point.

        One parametrized property in place of three that each hardcoded a
        single pair, drawing from `cdicts` for the same reason as
        `test_agrees_with_jacfwd`: the chart's own domain excludes the
        singular directions, so no filtering is needed.
        """
        p_curv = data.draw(cxst.cdicts(curv, magnitude=WELL_CONDITIONED))
        p_cart = cxc.pt_map(p_curv, curv, cart)

        j_fwd = cxc.jac_pt_map(p_cart, cart, curv)
        j_inv = cxc.jac_pt_map(p_curv, curv, cart)

        assert_allclose(qnp.matmul(j_inv, j_fwd).value, jnp.eye(curv.ndim), atol=1e-4)


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
        for (j, ok), (i, ik) in itertools.product(
            enumerate(out_keys), enumerate(in_keys)
        ):
            assert_allclose(
                J.value[j, i],
                ref[ok][ik],
                atol=atol,
                err_msg=f"J[{ok}, {ik}] mismatch vs jacfwd",
            )

    @pytest.mark.parametrize(("cart", "curv"), CHART_PAIRS)
    @pytest.mark.parametrize("forward", [True, False], ids=["cart->curv", "curv->cart"])
    @given(data=st.data())
    @settings(deadline=None)
    def test_agrees_with_jacfwd(
        self,
        cart: cxc.AbstractChart,
        curv: cxc.AbstractChart,
        forward: bool,
        data: st.DataObject,
    ) -> None:
        """The analytic Jacobian matches jacfwd at an arbitrary bounded point.

        Replaces eight single-point tests. Those covered six chart pairs but
        only ever at one hand-picked point each, and the two property tests
        that sat beside them covered just `cart3d -> sph3d` and
        `cart3d -> cyl3d` -- so `cart2d <-> polar2d` and the three reverse
        directions had no arbitrary-point coverage at all. This closes that:
        every pair is checked in both directions.

        Points come from `coordinaxs.hypothesis.cdicts`, which knows each
        chart's domain -- r > 0, colatitude off both poles -- so no filtering
        is needed and the singular *directions* are excluded by construction.
        `WELL_CONDITIONED` bounds the scale on top of that.

        "Arbitrary" therefore means arbitrary within that box, not across the
        whole non-singular domain: extreme *scales* remain untested, and would
        need a scale-aware tolerance rather than a wider strategy.
        """
        point = data.draw(cxst.cdicts(curv, magnitude=WELL_CONDITIONED))
        if forward:
            self._check_agrees(cart, curv, cxc.pt_map(point, curv, cart), atol=1e-4)
        else:
            self._check_agrees(curv, cart, point, atol=1e-4)


# ===========================================================================
# 12. JAX compatibility
# ===========================================================================


class TestJacobianPtMapJAXCompatibility:
    """jac_pt_map must be usable inside jax.jit and jax.vmap."""

    def test_jit_cart3d_to_sph3d(self) -> None:
        """JIT compilation: jac_pt_map is traceable."""
        at = {"x": u.Q(1, "m"), "y": u.Q(0, "m"), "z": u.Q(0, "m")}

        @jax.jit
        def jitted(at):
            return cxc.jac_pt_map(at, cxc.cart3d, cxc.sph3d)

        J = jitted(at)
        assert isinstance(J, ul.QuantityMatrix)
        assert_allclose(J.value[0, 0], 1, atol=1e-6)  # ∂r/∂x at (1,0,0)

    def test_jit_cart2d_to_polar2d(self) -> None:
        """JIT compilation: 2D case."""
        at = {"x": u.Q(1, "m"), "y": u.Q(0, "m")}

        @jax.jit
        def jitted(at):
            return cxc.jac_pt_map(at, cxc.cart2d, cxc.polar2d)

        J = jitted(at)
        assert_allclose(J.value, jnp.eye(2), atol=1e-6)

    def test_vmap_over_base_points_cart2d_polar2d(self) -> None:
        """Vmap over a batch of base points — all produce valid 2x2 Jacobians."""
        xs = jnp.array([1, 0, 1])
        ys = jnp.array([0, 1, 1])

        def single(x, y):
            at = {"x": u.Q(x, "m"), "y": u.Q(y, "m")}
            return cxc.jac_pt_map(at, cxc.cart2d, cxc.polar2d)

        batched = jax.vmap(single)(xs, ys)
        assert batched.value.shape == (3, 2, 2)
        # At (1, 0): identity; check first element of batch
        assert_allclose(batched.value[0], jnp.eye(2), atol=1e-6)


# ===========================================================================
# 13. Curried and None-partial forms
# ===========================================================================


class TestJacobianPtMapCurriedForms:
    """Curried and None-partial forms match direct call."""

    def test_curried_returns_callable(self) -> None:
        """jac_pt_map(from_chart, to_chart, usys=si) returns a callable."""
        fn = cxc.jac_pt_map(cxc.cart3d, cxc.sph3d, usys=usys_si)
        assert callable(fn)

    def test_curried_result_matches_direct(self) -> None:
        """Curried form result matches direct call."""
        at = {"x": u.Q(1, "m"), "y": u.Q(0, "m"), "z": u.Q(0, "m")}
        fn = cxc.jac_pt_map(cxc.cart3d, cxc.sph3d, usys=usys_si)
        J_curried = fn(at)
        J_direct = cxc.jac_pt_map(at, cxc.cart3d, cxc.sph3d)
        assert_allclose(J_curried.value, J_direct.value, atol=1e-6)

    def test_none_partial_returns_callable(self) -> None:
        """jac_pt_map(None, from_chart, to_chart, usys=si) returns a callable."""
        fn = cxc.jac_pt_map(None, cxc.cart3d, cxc.sph3d, usys=usys_si)
        assert callable(fn)

    def test_none_partial_result_matches_direct(self) -> None:
        """None-partial form result matches direct call."""
        at = {"x": u.Q(1, "m"), "y": u.Q(0, "m"), "z": u.Q(0, "m")}
        fn = cxc.jac_pt_map(None, cxc.cart3d, cxc.sph3d, usys=usys_si)
        J_partial = fn(at)
        J_direct = cxc.jac_pt_map(at, cxc.cart3d, cxc.sph3d)
        assert_allclose(J_partial.value, J_direct.value, atol=1e-6)

    def test_curried_2d(self) -> None:
        """Curried form works for 2D chart pair."""
        at = {"x": u.Q(1, "m"), "y": u.Q(0, "m")}
        fn = cxc.jac_pt_map(cxc.cart2d, cxc.polar2d, usys=usys_si)
        J = fn(at)
        assert isinstance(J, ul.QuantityMatrix)
        assert J.value.shape == (2, 2)


# ===========================================================================
# 14. Plain Array dispatch (requires usys)
# ===========================================================================


class TestJacobianPtMapArrayInput:
    """Plain Array input dispatches to an Array output (dispatch 3)."""

    def test_array_input_returns_array(self) -> None:
        """Plain array in → plain array out with usys."""
        at = jnp.array([1, 0, 0])
        J = cxc.jac_pt_map(at, cxc.cart3d, cxc.sph3d, usys=usys_si)
        assert isinstance(J, jnp.ndarray)
        assert J.shape == (3, 3)

    def test_int_array_input_is_promoted_and_supported(self) -> None:
        """Integer plain-array input is promoted and produces the correct Jacobian."""
        at_int = jnp.array([1, 0, 0])
        at_float = jnp.array([1, 0, 0], dtype=float)

        J_int = cxc.jac_pt_map(at_int, cxc.cart3d, cxc.sph3d, usys=usys_si)
        J_float = cxc.jac_pt_map(at_float, cxc.cart3d, cxc.sph3d, usys=usys_si)

        assert isinstance(J_int, jnp.ndarray)
        assert J_int.dtype == jax.dtypes.canonicalize_dtype(jnp.float_)
        assert_allclose(J_int, J_float, atol=1e-6)

    def test_bool_array_input_is_promoted_and_supported(self) -> None:
        """Boolean plain-array input is promoted and produces the correct Jacobian."""
        at_bool = jnp.array([True, False, False], dtype=jnp.bool_)
        at_float = jnp.array([1, 0, 0], dtype=float)

        J_bool = cxc.jac_pt_map(at_bool, cxc.cart3d, cxc.sph3d, usys=usys_si)
        J_float = cxc.jac_pt_map(at_float, cxc.cart3d, cxc.sph3d, usys=usys_si)

        assert isinstance(J_bool, jnp.ndarray)
        assert J_bool.dtype == jax.dtypes.canonicalize_dtype(jnp.float_)
        assert_allclose(J_bool, J_float, atol=1e-6)

    def test_complex_array_input_raises_and_is_not_silently_cast(self) -> None:
        """Complex plain-array input raises rather than dropping imaginary parts."""
        at_complex = jnp.array([1 + 2j, 0 + 0j, 0 + 0j], dtype=jnp.complex64)

        with pytest.raises(TypeError, match="real-valued inputs"):
            cxc.jac_pt_map(at_complex, cxc.cart3d, cxc.sph3d, usys=usys_si)

    def test_array_input_values_match_direct(self) -> None:
        """Array dispatch values agree with CDict quantity dispatch."""
        at_arr = jnp.array([1, 0, 0])
        at_qty = {"x": u.Q(1, "m"), "y": u.Q(0, "m"), "z": u.Q(0, "m")}
        J_arr = cxc.jac_pt_map(at_arr, cxc.cart3d, cxc.sph3d, usys=usys_si)
        J_qty = cxc.jac_pt_map(at_qty, cxc.cart3d, cxc.sph3d)
        assert_allclose(J_arr, J_qty.value, atol=1e-6)

    def test_array_2d(self) -> None:
        """Plain array dispatch works for Cart2D → Polar2D."""
        at = jnp.array([1, 0])
        J = cxc.jac_pt_map(at, cxc.cart2d, cxc.polar2d, usys=usys_si)
        assert isinstance(J, jnp.ndarray)
        assert J.shape == (2, 2)


# ===========================================================================
# 15. CDict is_array branch (plain array values)
# ===========================================================================


class TestJacobianPtMapCDictArrayBranch:
    """CDict with plain array values — the is_array=True branch in the generic dispatch.

    The CDict dispatch branches on whether values are quantities or plain arrays.
    The plain-array branch stacks the values and forwards to the Array dispatch,
    so whether usys is required is decided by that dispatch, not by this branch.

    Note: for pairs with an analytical Array dispatch (e.g. Cart2D→Polar2D) a
    plain-array CDict works without usys; for pairs that fall through to the
    generic Array path (e.g. Cart3D→Sph3D) usys must be supplied.
    """

    def test_generic_pair_with_usys(self) -> None:
        """Cart3D→Sph3D CDict with plain floats and usys provided → Array output."""
        at = {"x": jnp.array(1), "y": jnp.array(0), "z": jnp.array(0)}
        J = cxc.jac_pt_map(at, cxc.cart3d, cxc.sph3d, usys=usys_si)
        assert J.shape == (3, 3)

    def test_generic_pair_int_arrays_are_promoted_and_supported(self) -> None:
        """Cart3D→Sph3D integer CDict values are promoted via Array dispatch."""
        at_int = {"x": jnp.array(1), "y": jnp.array(0), "z": jnp.array(0)}
        at_float = {
            "x": jnp.array(1, dtype=float),
            "y": jnp.array(0, dtype=float),
            "z": jnp.array(0, dtype=float),
        }

        J_int = cxc.jac_pt_map(at_int, cxc.cart3d, cxc.sph3d, usys=usys_si)
        J_float = cxc.jac_pt_map(at_float, cxc.cart3d, cxc.sph3d, usys=usys_si)

        assert isinstance(J_int, jnp.ndarray)
        assert J_int.dtype == jax.dtypes.canonicalize_dtype(jnp.float_)
        assert_allclose(J_int, J_float, atol=1e-6)

    def test_generic_pair_no_usys_fails(self) -> None:
        """Cart3D→Sph3D CDict with plain floats and no usys raises an error.

        The CDict is_array=True branch forwards to the Array dispatch which
        requires usys.  For generic chart pairs with no analytical Array
        dispatch this is a known limitation.
        """
        at = {"x": jnp.array(1), "y": jnp.array(0), "z": jnp.array(0)}
        with pytest.raises((jaxtyping.TypeCheckError, ValueError), match="usys"):
            cxc.jac_pt_map(at, cxc.cart3d, cxc.sph3d)


# ===========================================================================
# Extreme scales
# ===========================================================================


#: Magnitudes the scale-free properties are checked at, spanning 34 decades.
#:
#: `WELL_CONDITIONED` deliberately stays near 1 so an *absolute* tolerance on
#: Jacobian entries means something, which leaves the arithmetic untested at
#: the magnitudes real data arrives in -- parsecs, or metres between atoms.
#:
#: The range is bounded by float32, not by the charts, and the binding limit is
#: *underflow at the top*: entries of ``J_{cart->sph}`` scale like ``1/r**2``,
#: and ``1/r**2`` reaches float32's smallest normal at
#: ``r = sqrt(1/tiny) = 9.2e18`` -- measured to fail there, well before
#: ``r**2`` would overflow at ``sqrt(max) = 1.8e19``. At the bottom ``r**2``
#: underflows below ``sqrt(tiny) = 1.1e-19``, and near-pole colatitudes shrink
#: ``x`` and ``y`` further, so the floor sits above that at 1e-16.
#:
#: Each entry is drawn over ``[mag, 10*mag]``, so the tops and bottoms of those
#: windows are what must stay inside [1e-16, 1e18]. Measured worst case across
#: the whole angular domain is 2.6e-5, against ``atol=1e-4``.
#:
#: Not asserted as a boundary test: the limits move if JAX x64 is enabled, so
#: pinning them would encode the dtype of the environment rather than the maths.
EXTREME_MAGNITUDES = [
    pytest.param(1e-16, id="1e-16"),
    pytest.param(1e-11, id="1e-11"),
    pytest.param(1e-6, id="1e-6"),
    pytest.param(1e6, id="1e6"),
    pytest.param(1e11, id="1e11"),
    pytest.param(1e17, id="1e17"),
]


class TestJacobianPtMapAtExtremeScales:
    r"""The scale-free properties, checked far from unit scale.

    Every other Jacobian test here pins points near 1 because it compares
    entries against an absolute tolerance, and entries of these Jacobians
    scale like ``r`` and ``1/r``. That leaves the arithmetic untested at the
    magnitudes real data arrives in -- parsecs, or metres between atoms.

    Both properties below are *dimensionless*, so they sidestep that: the
    identity matrix is the identity matrix at any scale, and a relative error
    is scale-free by construction. One flat tolerance therefore works across
    all 36 decades, with no rtol/atol tradeoff to tune.
    """

    @pytest.mark.parametrize("magnitude", EXTREME_MAGNITUDES)
    @pytest.mark.parametrize(("cart", "curv"), CHART_PAIRS)
    @given(data=st.data())
    @settings(deadline=None, max_examples=5)
    def test_composition_is_the_identity(
        self,
        cart: cxc.AbstractChart,
        curv: cxc.AbstractChart,
        magnitude: float,
        data: st.DataObject,
    ) -> None:
        """``J_{curv->cart} @ J_{cart->curv} = I``, at any magnitude.

        The chain rule does not care how big the coordinates are, so this is
        the same assertion `TestJacobianPtMapCompositionProperty` makes -- run
        where the numbers are extreme rather than convenient.
        """
        p_curv = data.draw(cxst.cdicts(curv, magnitude=(magnitude, magnitude * 10.0)))
        p_cart = cxc.pt_map(p_curv, curv, cart)

        j_fwd = cxc.jac_pt_map(p_cart, cart, curv)
        j_inv = cxc.jac_pt_map(p_curv, curv, cart)

        assert_allclose(qnp.matmul(j_inv, j_fwd).value, jnp.eye(curv.ndim), atol=1e-4)

    @pytest.mark.parametrize("magnitude", EXTREME_MAGNITUDES)
    def test_analytic_dispatch_agrees_with_jacfwd(self, magnitude: float) -> None:
        """The closed-form `Cart2D -> Polar2D` Jacobian matches autodiff.

        This is the only pair with a hand-written Jacobian, and it is reached
        only on *packed* input -- a cdict resolves to the generic branch, which
        is ``jax.jacfwd(pt_map)`` itself and so cannot disagree with it. Passing
        an array is what makes this an independent check rather than a
        tautology.

        Compared relatively: entries here scale like ``1/r``, so an absolute
        tolerance would be vacuous at 1e18 and unmeetable at 1e-18.
        """
        theta = 0.7
        at = jnp.asarray(
            [magnitude * math.cos(theta), magnitude * math.sin(theta)],
            dtype=jnp.float32,
        )

        got = np.asarray(cxc.jac_pt_map(at, cxc.cart2d, cxc.polar2d, usys=usys_si))
        expected = np.asarray(
            jax.jacfwd(cxc.pt_map(None, cxc.cart2d, cxc.polar2d, usys=usys_si))(at)
        )

        assert np.all(np.isfinite(got))
        nonzero = expected != 0
        assert_allclose(got[nonzero], expected[nonzero], rtol=1e-5)
