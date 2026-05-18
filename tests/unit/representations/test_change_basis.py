"""Tests for change_basis function."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import unxt as u

import coordinax.charts as cxc
import coordinax.main as cx
import coordinax.manifolds as cxm
import coordinax.representations as cxr
from coordinax.internal import QMatrix, UnitsMatrix
from coordinax.representations._src.basis_change import _qm_triangular_solve


def tree_equal(lhs: Any, rhs: Any) -> bool:
    """Return True when two pytrees are elementwise equal."""
    eq_tree = jax.tree_util.tree_map(jnp.equal, lhs, rhs)
    leaves = jax.tree_util.tree_leaves(eq_tree)
    if not leaves:
        return True
    reduced = [jnp.all(jnp.asarray(leaf)) for leaf in leaves]
    return bool(jnp.all(jnp.stack(reduced)))


usys = u.unitsystems.si


class TestChangeBasisExistence:
    """Import-surface checks for the public API."""

    def test_importable_from_representations(self):
        assert hasattr(cxr, "change_basis")
        assert callable(cxr.change_basis)

    def test_importable_from_main(self):
        assert hasattr(cx, "change_basis")
        assert callable(cx.change_basis)


class TestChangeBasisDispatch:
    """Dispatch behavior for basis and representation overloads."""

    def test_basis_overload_identity(self):
        v = {"x": jnp.array(1), "y": jnp.array(2)}
        at = {"x": jnp.array(0), "y": jnp.array(0)}
        out = cxr.change_basis(v, cxc.cart2d, cxr.coord_basis, cxr.coord_basis, at=at)
        assert tree_equal(out, v)

    def test_basis_overload_coord_to_phys_cartesian(self):
        v = {"x": jnp.array(1), "y": jnp.array(0)}

        at = {"x": jnp.array(1), "y": jnp.array(0)}
        out = cxr.change_basis(v, cxc.cart2d, cxr.coord_basis, cxr.phys_basis, at=at)
        np.testing.assert_allclose(out["x"], v["x"])
        np.testing.assert_allclose(out["y"], v["y"])

    def test_basis_overload_phys_to_coord_cartesian(self):
        v = {"x": jnp.array(0), "y": jnp.array(2)}
        at = {"x": jnp.array(1), "y": jnp.array(0)}
        out = cxr.change_basis(v, cxc.cart2d, cxr.phys_basis, cxr.coord_basis, at=at)
        np.testing.assert_allclose(out["x"], v["x"])
        np.testing.assert_allclose(out["y"], v["y"])

    def test_rep_overload_identity(self):
        v = {"x": jnp.array(1), "y": jnp.array(2)}
        at = {"x": jnp.array(0), "y": jnp.array(0)}
        rep = cxr.coord_disp
        out = cxr.change_basis(v, cxc.cart2d, rep, rep, at=at)
        assert tree_equal(out, v)

    def test_rep_overload_coord_to_phys(self):
        v = {"x": jnp.array(1), "y": jnp.array(0)}
        at = {"x": jnp.array(1), "y": jnp.array(0)}
        rep_from = cxr.coord_disp
        rep_to = cxr.phys_disp
        out = cxr.change_basis(v, cxc.cart2d, rep_from, rep_to, at=at)
        np.testing.assert_allclose(out["x"], v["x"])
        np.testing.assert_allclose(out["y"], v["y"])

    def test_rep_overload_phys_to_coord(self):
        v = {"x": jnp.array(0), "y": jnp.array(2)}
        at = {"x": jnp.array(1), "y": jnp.array(0)}
        rep_from = cxr.phys_disp
        rep_to = cxr.coord_disp
        out = cxr.change_basis(v, cxc.cart2d, rep_from, rep_to, at=at)
        np.testing.assert_allclose(out["x"], v["x"])
        np.testing.assert_allclose(out["y"], v["y"])

    def test_round_trip_cartesian(self):
        v = {"x": jnp.array(1.5), "y": jnp.array(-2)}
        at = {"x": jnp.array(0), "y": jnp.array(0)}
        v_phys = cxr.change_basis(v, cxc.cart2d, cxr.coord_basis, cxr.phys_basis, at=at)
        v_back = cxr.change_basis(
            v_phys, cxc.cart2d, cxr.phys_basis, cxr.coord_basis, at=at
        )
        np.testing.assert_allclose(v_back["x"], v["x"])
        np.testing.assert_allclose(v_back["y"], v["y"])

    def test_round_trip_spherical_non_cartesian(self):
        v = {"r": u.Q(5, "m/s"), "theta": u.Q(1, "rad/s"), "phi": u.Q(1, "rad/s")}
        at = {"r": u.Q(2, "m"), "theta": u.Q(jnp.pi / 2, "rad"), "phi": u.Q(0, "rad")}

        v_phys = cxr.change_basis(v, cxc.sph3d, cxr.coord_basis, cxr.phys_basis, at=at)
        np.testing.assert_allclose(u.ustrip("m/s", v_phys["r"]), 5)
        np.testing.assert_allclose(u.ustrip("m/s", v_phys["theta"]), 2)
        np.testing.assert_allclose(u.ustrip("m/s", v_phys["phi"]), 2)

        v_back = cxr.change_basis(
            v_phys, cxc.sph3d, cxr.phys_basis, cxr.coord_basis, at=at
        )
        np.testing.assert_allclose(
            u.ustrip("m/s", v_back["r"]), u.ustrip("m/s", v["r"])
        )
        np.testing.assert_allclose(
            u.ustrip("rad/s", v_back["theta"]), u.ustrip("rad/s", v["theta"])
        )
        np.testing.assert_allclose(
            u.ustrip("rad/s", v_back["phi"]), u.ustrip("rad/s", v["phi"])
        )

    def test_round_trip_diagonal_metric(self):
        M = cxm.R3
        v = {"r": u.Q(4, "m/s"), "theta": u.Q(0.5, "rad/s"), "phi": u.Q(0.25, "rad/s")}
        at = {"r": u.Q(3, "m"), "theta": u.Q(0.5, "rad"), "phi": u.Q(0, "rad")}

        v_phys = cxr.change_basis(
            v, cxc.sph3d, M, cxr.coord_basis, cxr.phys_basis, at=at
        )
        v_back = cxr.change_basis(
            v_phys, cxc.sph3d, M, cxr.phys_basis, cxr.coord_basis, at=at
        )

        np.testing.assert_allclose(
            u.ustrip("m/s", v_back["r"]), u.ustrip("m/s", v["r"])
        )
        np.testing.assert_allclose(
            u.ustrip("rad/s", v_back["theta"]), u.ustrip("rad/s", v["theta"])
        )
        np.testing.assert_allclose(
            u.ustrip("rad/s", v_back["phi"]), u.ustrip("rad/s", v["phi"])
        )

    def test_round_trip_general_metric(self):
        M = cxm.EmbeddedManifold(
            intrinsic=cxm.S2,
            ambient=cxm.R3,
            embed_map=cxm.TwoSphereIn3D(radius=u.Q(1, "km")),
        )
        v = {"theta": u.Q(1, "rad/s"), "phi": u.Q(2, "rad/s")}
        at = {"theta": u.Q(jnp.pi / 3, "rad"), "phi": u.Q(0, "rad")}

        v_phys = cxr.change_basis(
            v, cxc.sph2, M, cxr.coord_basis, cxr.phys_basis, at=at
        )
        v_back = cxr.change_basis(
            v_phys, cxc.sph2, M, cxr.phys_basis, cxr.coord_basis, at=at
        )

        np.testing.assert_allclose(
            u.ustrip("rad/s", v_back["theta"]), u.ustrip("rad/s", v["theta"])
        )
        np.testing.assert_allclose(
            u.ustrip("rad/s", v_back["phi"]), u.ustrip("rad/s", v["phi"])
        )


class TestChangeBasisErrors:
    """Unsupported basis and representation cases."""

    def test_no_basis_to_coord_is_identity(self):
        v = {"x": jnp.array(1)}
        at = {"x": jnp.array(0)}
        out = cxr.change_basis(v, cxc.cart1d, cxr.no_basis, cxr.coord_basis, at=at)
        np.testing.assert_allclose(out["x"], v["x"])

    def test_no_basis_to_phys_is_identity_for_uniform_dimensions(self):
        v = {"x": u.Q(1, "m/s"), "y": u.Q(2, "km/s")}
        at = {"x": jnp.array(0), "y": jnp.array(0)}
        out = cxr.change_basis(v, cxc.cart2d, cxr.no_basis, cxr.phys_basis, at=at)
        np.testing.assert_allclose(u.ustrip("m/s", out["x"]), 1)
        np.testing.assert_allclose(u.ustrip("km/s", out["y"]), 2)

    def test_no_basis_to_phys_rejects_mixed_dimensions(self):
        v = {"x": u.Q(1, "m/s"), "y": u.Q(2, "m")}
        at = {"x": jnp.array(0), "y": jnp.array(0)}
        with pytest.raises(ValueError, match="same dimension"):
            cxr.change_basis(v, cxc.cart2d, cxr.no_basis, cxr.phys_basis, at=at)

    def test_point_rep_rejected(self):
        v = {"x": jnp.array(1)}
        at = {"x": jnp.array(0)}
        with pytest.raises((TypeError, ValueError)):
            cxr.change_basis(v, cxc.cart1d, cxr.point, cxr.coord_disp, at=at)

    def test_missing_at_raises(self):
        v = {"r": jnp.array(1), "theta": jnp.array(0.2), "phi": jnp.array(0.1)}
        with pytest.raises((TypeError, ValueError)):
            cxr.change_basis(v, cxc.sph3d, cxr.coord_basis, cxr.phys_basis)


class TestChangeBasisJAX:
    """JAX transformation compatibility for change_basis."""

    def test_jit(self):
        v = {"x": jnp.array(1), "y": jnp.array(0)}
        at = {"x": jnp.array(1), "y": jnp.array(0)}

        @jax.jit
        def run(v, at):
            return cxr.change_basis(
                v, cxc.cart2d, cxr.coord_basis, cxr.phys_basis, at=at
            )

        out = run(v, at)
        np.testing.assert_allclose(out["x"], v["x"])
        np.testing.assert_allclose(out["y"], v["y"])

    def test_vmap(self):
        vs = {"x": jnp.ones(3), "y": jnp.zeros(3)}
        ats = {"x": jnp.array([1, 2, 3]), "y": jnp.zeros(3)}

        def single(v: dict[str, Any], at: dict[str, Any]) -> dict[str, Any]:
            return cxr.change_basis(
                v, cxc.cart2d, cxr.coord_basis, cxr.phys_basis, at=at
            )

        batched = jax.vmap(single)(vs, ats)
        np.testing.assert_allclose(batched["x"], 1)
        np.testing.assert_allclose(batched["y"], 0)


class TestChangeBasisManifold:
    """Tests for the manifold-based change_basis dispatches (Phase 3d).

    Covers the new overloads added in Phase 3d:
    - Cartesian + any manifold → identity (precedence=1)
    - AbstractChart + AbstractManifold, diagonal metric path (scale factors)
    - AbstractChart + AbstractManifold, general Cholesky path
    - Spherical3D + EuclideanManifold → delegates to chart-specific dispatch
    """

    def test_cartesian_with_manifold_coord_to_phys_is_identity(self):
        """Dispatch 7: Cartesian + any manifold, coord→phys = identity."""
        v = {"x": u.Q(3.0, "m/s"), "y": u.Q(4.0, "m/s")}
        at = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m")}
        out = cxr.change_basis(
            v, cxc.cart2d, cxm.R2, cxr.coord_basis, cxr.phys_basis, at=at
        )
        np.testing.assert_allclose(u.ustrip("m/s", out["x"]), 3.0)
        np.testing.assert_allclose(u.ustrip("m/s", out["y"]), 4.0)

    def test_cartesian_with_manifold_phys_to_coord_is_identity(self):
        """Dispatch 7: Cartesian + any manifold, phys→coord = identity."""
        v = {"x": u.Q(3.0, "m/s"), "y": u.Q(4.0, "m/s")}
        out = cxr.change_basis(v, cxc.cart2d, cxm.R2, cxr.phys_basis, cxr.coord_basis)
        np.testing.assert_allclose(u.ustrip("m/s", out["x"]), 3.0)
        np.testing.assert_allclose(u.ustrip("m/s", out["y"]), 4.0)

    def test_euclidean_sph3d_manifold_matches_no_metric(self):
        """Dispatch 9 delegates to 8: EuclideanManifold+sph3d with no-metric."""
        v = {"r": u.Q(5, "m/s"), "theta": u.Q(1, "rad/s"), "phi": u.Q(2, "rad/s")}
        at = {"r": u.Q(3, "m"), "theta": u.Q(0.5, "rad"), "phi": u.Q(0, "rad")}

        out_with = cxr.change_basis(
            v, cxc.sph3d, cxm.R3, cxr.coord_basis, cxr.phys_basis, at=at
        )
        out_without = cxr.change_basis(
            v, cxc.sph3d, cxr.coord_basis, cxr.phys_basis, at=at
        )

        np.testing.assert_allclose(
            u.ustrip("m/s", out_with["r"]), u.ustrip("m/s", out_without["r"])
        )
        np.testing.assert_allclose(
            u.ustrip("m/s", out_with["theta"]), u.ustrip("m/s", out_without["theta"])
        )
        np.testing.assert_allclose(
            u.ustrip("m/s", out_with["phi"]), u.ustrip("m/s", out_without["phi"])
        )

    def test_diagonal_path_coord_to_phys_values(self):
        """Diagonal path: verify scale-factor multiplication at theta=pi/2."""
        # h_r=1, h_theta=r=2, h_phi=r*sin(pi/2)=2
        v = {"r": u.Q(5, "m/s"), "theta": u.Q(1, "rad/s"), "phi": u.Q(2, "rad/s")}
        at = {"r": u.Q(2, "m"), "theta": u.Q(jnp.pi / 2, "rad"), "phi": u.Q(0, "rad")}
        out = cxr.change_basis(
            v, cxc.sph3d, cxm.R3, cxr.coord_basis, cxr.phys_basis, at=at
        )
        np.testing.assert_allclose(u.ustrip("m/s", out["r"]), 5.0)
        np.testing.assert_allclose(u.ustrip("m/s", out["theta"]), 2.0)  # 1 * 2
        np.testing.assert_allclose(u.ustrip("m/s", out["phi"]), 4.0)  # 2 * 2

    def test_diagonal_path_phys_to_coord_values(self):
        """Diagonal path: verify inverse scale-factor division at theta=pi/2."""
        # h_r=1, h_theta=2, h_phi=2
        v_phys = {
            "r": u.Q(5.0, "m/s"),
            "theta": u.Q(2.0, "m/s"),
            "phi": u.Q(4.0, "m/s"),
        }
        at = {"r": u.Q(2, "m"), "theta": u.Q(jnp.pi / 2, "rad"), "phi": u.Q(0, "rad")}
        out = cxr.change_basis(
            v_phys, cxc.sph3d, cxm.R3, cxr.phys_basis, cxr.coord_basis, at=at
        )
        np.testing.assert_allclose(u.ustrip("m/s", out["r"]), 5.0)
        np.testing.assert_allclose(u.ustrip("rad/s", out["theta"]), 1.0)  # 2 / 2
        np.testing.assert_allclose(u.ustrip("rad/s", out["phi"]), 2.0)  # 4 / 2

    def test_cholesky_path_output_keys_and_units(self):
        """Cholesky path (PullbackMetric): correct keys and speed dimension."""
        manifold = cxm.EmbeddedManifold(
            intrinsic=cxm.S2,
            ambient=cxm.R3,
            embed_map=cxm.TwoSphereIn3D(radius=u.Q(1.0, "km")),
        )
        v = {"theta": u.Q(1.0, "rad/s"), "phi": u.Q(2.0, "rad/s")}
        at = {"theta": u.Q(jnp.pi / 3, "rad"), "phi": u.Q(0.0, "rad")}
        out = cxr.change_basis(
            v, cxc.sph2, manifold, cxr.coord_basis, cxr.phys_basis, at=at
        )
        assert set(out.keys()) == {"theta", "phi"}
        assert u.dimension_of(out["theta"]) == u.dimension("speed")
        assert u.dimension_of(out["phi"]) == u.dimension("speed")


class TestChangeBasisManifoldJAX:
    """JAX transformation compatibility for manifold-based change_basis dispatches."""

    def test_jit_diagonal_path(self):
        v = {"r": u.Q(5.0, "m/s"), "theta": u.Q(1.0, "rad/s"), "phi": u.Q(1.0, "rad/s")}
        at = {
            "r": u.Q(2.0, "m"),
            "theta": u.Q(jnp.pi / 2, "rad"),
            "phi": u.Q(0.0, "rad"),
        }

        @jax.jit
        def run(v, at):
            return cxr.change_basis(
                v, cxc.sph3d, cxm.R3, cxr.coord_basis, cxr.phys_basis, at=at
            )

        out = run(v, at)
        np.testing.assert_allclose(u.ustrip("m/s", out["r"]), 5.0)
        np.testing.assert_allclose(u.ustrip("m/s", out["theta"]), 2.0)  # h=2

    def test_jit_cholesky_path(self):
        manifold = cxm.EmbeddedManifold(
            intrinsic=cxm.S2,
            ambient=cxm.R3,
            embed_map=cxm.TwoSphereIn3D(radius=u.Q(1.0, "km")),
        )
        v = {"theta": u.Q(1.0, "rad/s"), "phi": u.Q(2.0, "rad/s")}
        at = {"theta": u.Q(jnp.pi / 3, "rad"), "phi": u.Q(0.0, "rad")}

        @jax.jit
        def run(v, at):
            return cxr.change_basis(
                v, cxc.sph2, manifold, cxr.coord_basis, cxr.phys_basis, at=at
            )

        out = run(v, at)
        assert set(out.keys()) == {"theta", "phi"}

    def test_jit_cartesian_with_manifold(self):
        v = {"x": u.Q(3.0, "m/s"), "y": u.Q(4.0, "m/s")}
        at = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m")}

        @jax.jit
        def run(v, at):
            return cxr.change_basis(
                v, cxc.cart2d, cxm.R2, cxr.coord_basis, cxr.phys_basis, at=at
            )

        out = run(v, at)
        np.testing.assert_allclose(u.ustrip("m/s", out["x"]), 3.0)
        np.testing.assert_allclose(u.ustrip("m/s", out["y"]), 4.0)


class TestTriangularSolveBatching:
    """Regression tests for batched triangular solves used by basis conversion."""

    def test_qm_triangular_solve_batched_rows_scaled_correctly(self):
        e_val = jnp.array([[[2, 1], [0, 4]], [[3, 2], [0, 5]]])
        e_unit = UnitsMatrix(((u.unit("m"), u.unit("m")), (u.unit("m"), u.unit("m"))))
        e = QMatrix(e_val, unit=e_unit)

        b_val = jnp.array([[5, 8], [7, 10]])
        b_unit = UnitsMatrix((u.unit("m/s"), u.unit("m/s")))
        b = QMatrix(b_val, unit=b_unit)

        out = _qm_triangular_solve(e, b)

        expected = jnp.stack(
            [
                jax.scipy.linalg.solve_triangular(e_val[0], b_val[0], lower=False),
                jax.scipy.linalg.solve_triangular(e_val[1], b_val[1], lower=False),
            ]
        )
        np.testing.assert_allclose(out.value, expected)
        assert out.unit == UnitsMatrix((u.unit("1 / s"), u.unit("1 / s")))
