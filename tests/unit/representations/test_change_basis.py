"""Tests for change_basis function."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import unxt as u

import coordinax.charts as cxc
import coordinax.main as cx
import coordinax.representations as cxr

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
        v = {"x": jnp.array(1.0), "y": jnp.array(2.0)}
        at = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
        out = cxr.change_basis(v, cxc.cart2d, cxr.coord_basis, cxr.coord_basis, at=at)
        assert out == v

    def test_basis_overload_coord_to_phys_cartesian(self):
        v = {"x": jnp.array(1.0), "y": jnp.array(0.0)}

        at = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
        out = cxr.change_basis(v, cxc.cart2d, cxr.coord_basis, cxr.phys_basis, at=at)
        np.testing.assert_allclose(out["x"], v["x"])
        np.testing.assert_allclose(out["y"], v["y"])

    def test_basis_overload_phys_to_coord_cartesian(self):
        v = {"x": jnp.array(0.0), "y": jnp.array(2.0)}
        at = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
        out = cxr.change_basis(v, cxc.cart2d, cxr.phys_basis, cxr.coord_basis, at=at)
        np.testing.assert_allclose(out["x"], v["x"])
        np.testing.assert_allclose(out["y"], v["y"])

    def test_rep_overload_identity(self):
        v = {"x": jnp.array(1.0), "y": jnp.array(2.0)}
        at = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
        rep = cxr.coord_disp
        out = cxr.change_basis(v, cxc.cart2d, rep, rep, at=at)
        assert out == v

    def test_rep_overload_coord_to_phys(self):
        v = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
        at = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
        rep_from = cxr.coord_disp
        rep_to = cxr.phys_disp
        out = cxr.change_basis(v, cxc.cart2d, rep_from, rep_to, at=at)
        np.testing.assert_allclose(out["x"], v["x"])
        np.testing.assert_allclose(out["y"], v["y"])

    def test_rep_overload_phys_to_coord(self):
        v = {"x": jnp.array(0.0), "y": jnp.array(2.0)}
        at = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
        rep_from = cxr.phys_disp
        rep_to = cxr.coord_disp
        out = cxr.change_basis(v, cxc.cart2d, rep_from, rep_to, at=at)
        np.testing.assert_allclose(out["x"], v["x"])
        np.testing.assert_allclose(out["y"], v["y"])

    def test_round_trip_cartesian(self):
        v = {"x": jnp.array(1.5), "y": jnp.array(-2.0)}
        at = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
        v_phys = cxr.change_basis(v, cxc.cart2d, cxr.coord_basis, cxr.phys_basis, at=at)
        v_back = cxr.change_basis(
            v_phys, cxc.cart2d, cxr.phys_basis, cxr.coord_basis, at=at
        )
        np.testing.assert_allclose(v_back["x"], v["x"])
        np.testing.assert_allclose(v_back["y"], v["y"])


class TestChangeBasisErrors:
    """Unsupported basis and representation cases."""

    def test_no_basis_to_coord_is_identity(self):
        v = {"x": jnp.array(1.0)}
        at = {"x": jnp.array(0.0)}
        out = cxr.change_basis(v, cxc.cart1d, cxr.no_basis, cxr.coord_basis, at=at)
        np.testing.assert_allclose(out["x"], v["x"])

    def test_no_basis_to_phys_is_identity_for_uniform_dimensions(self):
        v = {"x": u.Q(1.0, "m/s"), "y": u.Q(2.0, "km/s")}
        at = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
        out = cxr.change_basis(v, cxc.cart2d, cxr.no_basis, cxr.phys_basis, at=at)
        np.testing.assert_allclose(u.ustrip("m/s", out["x"]), 1.0)
        np.testing.assert_allclose(u.ustrip("km/s", out["y"]), 2.0)

    def test_no_basis_to_phys_rejects_mixed_dimensions(self):
        v = {"x": u.Q(1.0, "m/s"), "y": u.Q(2.0, "m")}
        at = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
        with pytest.raises(ValueError, match="same dimension"):
            cxr.change_basis(v, cxc.cart2d, cxr.no_basis, cxr.phys_basis, at=at)

    def test_point_rep_rejected(self):
        v = {"x": jnp.array(1.0)}
        at = {"x": jnp.array(0.0)}
        with pytest.raises((TypeError, ValueError)):
            cxr.change_basis(v, cxc.cart1d, cxr.point, cxr.coord_disp, at=at)

    def test_missing_at_raises(self):
        v = {"r": jnp.array(1.0), "theta": jnp.array(0.2), "phi": jnp.array(0.1)}
        with pytest.raises((TypeError, ValueError)):
            cxr.change_basis(v, cxc.sph3d, cxr.coord_basis, cxr.phys_basis)


class TestChangeBasisJAX:
    """JAX transformation compatibility for change_basis."""

    def test_jit(self):
        v = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
        at = {"x": jnp.array(1.0), "y": jnp.array(0.0)}

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
        ats = {"x": jnp.array([1.0, 2.0, 3.0]), "y": jnp.zeros(3)}

        def single(v: dict[str, Any], at: dict[str, Any]) -> dict[str, Any]:
            return cxr.change_basis(
                v, cxc.cart2d, cxr.coord_basis, cxr.phys_basis, at=at
            )

        batched = jax.vmap(single)(vs, ats)
        np.testing.assert_allclose(batched["x"], 1.0)
        np.testing.assert_allclose(batched["y"], 0.0)
