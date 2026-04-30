"""Tests for cconvert dispatching to tangent_map when source is TangentGeometry."""

__all__: tuple[str, ...] = ()

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import unxt as u

import coordinax.charts as cxc
import coordinax.representations as cxr

usys = u.unitsystems.si


class TestCconvertTangentGeometry:
    """cconvert with TangentGeometry representation dispatches to tangent_map."""

    def test_same_chart_noncartesian_matches_change_basis(self) -> None:
        """Same-chart tangent conversion should reduce to basis conversion."""
        v = {"r": jnp.array(5.0), "theta": jnp.array(1.0), "phi": jnp.array(2.0)}
        at = {"r": jnp.array(3.0), "theta": jnp.array(0.5), "phi": jnp.array(0.0)}

        result = cxr.cconvert(
            v, cxc.sph3d, cxr.coord_disp, cxc.sph3d, cxr.phys_disp, at=at, usys=usys
        )
        expected = cxr.change_basis(
            v, cxc.sph3d, cxr.coord_basis, cxr.phys_basis, at=at, usys=usys
        )

        np.testing.assert_allclose(result["r"], expected["r"])
        np.testing.assert_allclose(result["theta"], expected["theta"])
        np.testing.assert_allclose(result["phi"], expected["phi"])

    def test_cart2d_to_polar2d_coord_disp(self) -> None:
        """Cconvert with coord_disp routes through tangent_map (Jacobian)."""
        v = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
        at = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
        result = cxr.cconvert(
            v, cxc.cart2d, cxr.coord_disp, cxc.polar2d, cxr.coord_disp, at=at, usys=usys
        )
        np.testing.assert_allclose(result["r"], 1.0, atol=1e-6)
        np.testing.assert_allclose(result["theta"], 0.0, atol=1e-6)

    def test_same_chart_identity(self) -> None:
        """Cconvert with same chart + TangentGeometry returns input unchanged."""
        v = {"x": jnp.array(2.0), "y": jnp.array(3.0)}
        at = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
        result = cxr.cconvert(
            v, cxc.cart2d, cxr.coord_disp, cxc.cart2d, cxr.coord_disp, at=at, usys=usys
        )
        np.testing.assert_allclose(result["x"], 2.0)
        np.testing.assert_allclose(result["y"], 3.0)

    def test_same_chart_cartesian_without_at(self) -> None:
        """Cartesian same-chart basis conversion should not require `at`."""
        v = {"x": jnp.array(1.0), "y": jnp.array(2.0)}
        result = cxr.cconvert(
            v, cxc.cart2d, cxr.coord_disp, cxc.cart2d, cxr.phys_disp, usys=usys
        )
        np.testing.assert_allclose(result["x"], v["x"])
        np.testing.assert_allclose(result["y"], v["y"])

    def test_cart3d_to_sph3d_coord_vel(self) -> None:
        """Cconvert with coord_vel representation uses tangent_map semantics."""
        v = {"x": jnp.array(1.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}
        at = {"x": jnp.array(1.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}
        result = cxr.cconvert(
            v, cxc.cart3d, cxr.coord_vel, cxc.sph3d, cxr.coord_vel, at=at, usys=usys
        )
        # Purely radial result
        np.testing.assert_allclose(result["r"], 1.0, atol=1e-6)
        np.testing.assert_allclose(result["theta"], 0.0, atol=1e-6)
        np.testing.assert_allclose(result["phi"], 0.0, atol=1e-6)

    def test_same_chart_respects_tangent_semantic_kind(self) -> None:
        """Displacement and velocity variants should follow the same basis map."""
        v = {"r": jnp.array(5.0), "theta": jnp.array(1.0), "phi": jnp.array(2.0)}
        at = {"r": jnp.array(3.0), "theta": jnp.array(0.5), "phi": jnp.array(0.0)}

        out_disp = cxr.cconvert(
            v, cxc.sph3d, cxr.coord_disp, cxc.sph3d, cxr.phys_disp, at=at, usys=usys
        )
        out_vel = cxr.cconvert(
            v, cxc.sph3d, cxr.coord_vel, cxc.sph3d, cxr.phys_vel, at=at, usys=usys
        )

        np.testing.assert_allclose(out_disp["r"], out_vel["r"])
        np.testing.assert_allclose(out_disp["theta"], out_vel["theta"])
        np.testing.assert_allclose(out_disp["phi"], out_vel["phi"])

    def test_jit_compatible(self) -> None:
        """Cconvert with TangentGeometry is JIT-compatible."""
        v = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
        at = {"x": jnp.array(1.0), "y": jnp.array(0.0)}

        @jax.jit
        def run(v, at):
            return cxr.cconvert(
                v,
                cxc.cart2d,
                cxr.coord_disp,
                cxc.polar2d,
                cxr.coord_disp,
                at=at,
                usys=usys,
            )

        result = run(v, at)
        np.testing.assert_allclose(result["r"], 1.0, atol=1e-6)

    def test_round_trip(self) -> None:
        """Cconvert tangent round trip: cart2d → polar2d → cart2d is identity."""
        v_cart = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
        at_cart = {"x": jnp.array(1.0), "y": jnp.array(0.0)}

        # cart → polar
        v_polar = cxr.cconvert(
            v_cart,
            cxc.cart2d,
            cxr.coord_disp,
            cxc.polar2d,
            cxr.coord_disp,
            at=at_cart,
            usys=usys,
        )

        # at in polar coords
        at_polar = cxr.cconvert(at_cart, cxc.cart2d, cxr.point, cxc.polar2d, usys=usys)

        # polar → cart
        v_cart_back = cxr.cconvert(
            v_polar,
            cxc.polar2d,
            cxr.coord_disp,
            cxc.cart2d,
            cxr.coord_disp,
            at=at_polar,
            usys=usys,
        )

        np.testing.assert_allclose(v_cart_back["x"], v_cart["x"], atol=1e-6)
        np.testing.assert_allclose(v_cart_back["y"], v_cart["y"], atol=1e-6)


class TestCconvertAtRequired:
    """cconvert with TangentGeometry requires the `at` keyword argument."""

    def test_at_required_for_nonlinear_charts(self) -> None:
        """Missing `at` raises informative error for non-Cartesian charts."""
        v = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
        with pytest.raises((TypeError, ValueError)):
            cxr.cconvert(
                v, cxc.cart2d, cxr.coord_disp, cxc.polar2d, cxr.coord_disp, usys=usys
            )
