"""Tests for coordinate realization functions (register_realize.py).

cartesian_chart, pt_map.
"""

import jax.numpy as jnp
import plum
import pytest
from hypothesis import given

import unxt as u

import coordinax.charts as cxc
import coordinaxs.hypothesis.main as cxst

# =============================================================================
# cartesian_chart
# =============================================================================


class TestCartesianChartFunction:
    """Tests for cartesian_chart function."""

    @pytest.mark.parametrize(
        ("chart", "expected_cartesian"),
        [
            (cxc.cart0d, cxc.cart0d),
            (cxc.cart1d, cxc.cart1d),
            (cxc.radial1d, cxc.cart1d),
            (cxc.cart2d, cxc.cart2d),
            (cxc.polar2d, cxc.cart2d),
            (cxc.cart3d, cxc.cart3d),
            (cxc.sph3d, cxc.cart3d),
            (cxc.lonlat_sph3d, cxc.cart3d),
            (cxc.loncoslat_sph3d, cxc.cart3d),
            (cxc.cyl3d, cxc.cart3d),
            (cxc.cartnd, cxc.cartnd),
        ],
    )
    def test_cartesian_chart_examples(self, chart, expected_cartesian):
        """Test that cartesian_chart returns the expected cartesian chart."""
        assert cxc.cartesian_chart(chart) == expected_cartesian

    @given(chart=cxst.charts())
    def test_cartesian_chart_idempotent(self, chart):
        """Property test: cartesian_chart is idempotent."""
        try:
            cart1 = cxc.cartesian_chart(chart)
            cart2 = cxc.cartesian_chart(cart1)
        except (cxc.NoGlobalCartesianChartError, plum.NotFoundLookupError):
            pass
        else:
            assert cart1 == cart2


# =============================================================================
# cartesian_chart for product charts
# =============================================================================


class TestCartesianChartProductCharts:
    """Test cartesian_chart dispatch for product charts."""

    def test_namespaced_product_cartesian_chart(self) -> None:
        """cartesian_chart should convert factors while preserving factor_names."""
        phase_sph = cxc.CartesianProductChart((cxc.sph3d, cxc.sph3d), ("q", "p"))
        phase_cart = cxc.cartesian_chart(phase_sph)
        assert isinstance(phase_cart.factors[0], cxc.Cart3D)
        assert isinstance(phase_cart.factors[1], cxc.Cart3D)
        assert phase_cart.factor_names == ("q", "p")

    def test_cartesian_chart_idempotent(self) -> None:
        """cartesian_chart applied twice should return same object."""
        phase_sph = cxc.CartesianProductChart((cxc.sph3d, cxc.sph3d), ("q", "p"))
        cart1 = cxc.cartesian_chart(phase_sph)
        cart2 = cxc.cartesian_chart(cart1)
        assert cart1 is cart2


# =============================================================================
# pt_map with product charts
# =============================================================================


class TestPointTransformProductCharts:
    """Test ``pt_map`` works correctly with product charts."""

    def test_namespaced_phase_space_transform(self) -> None:
        """pt_map should work with namespaced CartesianProductChart."""
        phase_cart = cxc.CartesianProductChart((cxc.cart3d, cxc.cart3d), ("q", "p"))
        phase_sph = cxc.CartesianProductChart((cxc.sph3d, cxc.sph3d), ("q", "p"))
        p = {
            "q.x": u.Q(1, "m"),
            "q.y": u.Q(0, "m"),
            "q.z": u.Q(0, "m"),
            "p.x": u.Q(0, "m"),
            "p.y": u.Q(1, "m"),
            "p.z": u.Q(0, "m"),
        }
        result = cxc.pt_map(p, phase_cart, phase_sph)
        assert u.ustrip("m", result["q.r"]) == pytest.approx(1)
        assert u.ustrip("rad", result["q.phi"]) == pytest.approx(0)
        assert u.ustrip("m", result["p.r"]) == pytest.approx(1)
        assert u.ustrip("rad", result["p.phi"]) == pytest.approx(jnp.pi / 2)


class TestPointTransformProlate:
    """``pt_map`` into ProlateSpheroidal3D routes via Cylindrical3D (no recursion)."""

    def test_cart3d_to_prolate_roundtrips(self) -> None:
        prolate = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, "m"))
        p = {"x": u.Q(0.5, "m"), "y": u.Q(1.5, "m"), "z": u.Q(3.0, "m")}
        out = cxc.pt_map(p, cxc.cart3d, prolate)
        assert set(out) == {"mu", "nu", "phi"}
        back = cxc.pt_map(out, prolate, cxc.cart3d)
        for k in ("x", "y", "z"):
            assert u.ustrip("m", back[k]) == pytest.approx(u.ustrip("m", p[k]))

    def test_spherical_to_prolate(self) -> None:
        # Routes via the generic fallback: Spherical3D -> Cart3D -> Cyl -> Prolate.
        prolate = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, "m"))
        p = {"r": u.Q(3.0, "m"), "theta": u.Q(0.6, "rad"), "phi": u.Q(0.4, "rad")}
        out = cxc.pt_map(p, cxc.sph3d, prolate)
        assert set(out) == {"mu", "nu", "phi"}
