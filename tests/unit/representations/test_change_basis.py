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
from coordinax.internal import QuantityMatrix, UnitsMatrix
from coordinax.representations._src.basis_change import _qm_triangular_solve


def tree_equal(lhs: Any, rhs: Any) -> bool:
    """Return True when two pytrees are elementwise equal."""
    eq_tree = jax.tree_util.tree_map(lambda x, y: jnp.equal(x, y), lhs, rhs)
    leaves = jax.tree_util.tree_leaves(eq_tree)
    if not leaves:
        return True
    reduced = [jnp.all(jnp.asarray(leaf)) for leaf in leaves]
    return bool(jnp.all(jnp.stack(reduced)))


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
        metric = cxm.EuclideanMetric(3)
        v = {"r": u.Q(4, "m/s"), "theta": u.Q(0.5, "rad/s"), "phi": u.Q(0.25, "rad/s")}
        at = {"r": u.Q(3, "m"), "theta": u.Q(0.5, "rad"), "phi": u.Q(0, "rad")}

        v_phys = cxr.change_basis(
            v, cxc.sph3d, metric, cxr.coord_basis, cxr.phys_basis, at=at
        )
        v_back = cxr.change_basis(
            v_phys, cxc.sph3d, metric, cxr.phys_basis, cxr.coord_basis, at=at
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
        metric = cxm.InducedMetric(
            cxm.TwoSphereIn3D(radius=u.Q(1, "km")), cxm.EuclideanMetric(3)
        )
        v = {"theta": u.Q(1, "rad/s"), "phi": u.Q(2, "rad/s")}
        at = {"theta": u.Q(jnp.pi / 3, "rad"), "phi": u.Q(0, "rad")}

        v_phys = cxr.change_basis(
            v, cxc.sph2, metric, cxr.coord_basis, cxr.phys_basis, at=at
        )
        v_back = cxr.change_basis(
            v_phys, cxc.sph2, metric, cxr.phys_basis, cxr.coord_basis, at=at
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


class TestTriangularSolveBatching:
    """Regression tests for batched triangular solves used by basis conversion."""

    def test_qm_triangular_solve_batched_rows_scaled_correctly(self):
        e_val = jnp.array([[[2, 1], [0, 4]], [[3, 2], [0, 5]]])
        e_unit = UnitsMatrix(((u.unit("m"), u.unit("m")), (u.unit("m"), u.unit("m"))))
        e = QuantityMatrix(e_val, unit=e_unit)

        b_val = jnp.array([[5, 8], [7, 10]])
        b_unit = UnitsMatrix((u.unit("m/s"), u.unit("m/s")))
        b = QuantityMatrix(b_val, unit=b_unit)

        out = _qm_triangular_solve(e, b)

        expected = jnp.stack(
            [
                jax.scipy.linalg.solve_triangular(e_val[0], b_val[0], lower=False),
                jax.scipy.linalg.solve_triangular(e_val[1], b_val[1], lower=False),
            ]
        )
        np.testing.assert_allclose(out.value, expected)
        assert out.unit == UnitsMatrix((u.unit("1 / s"), u.unit("1 / s")))
