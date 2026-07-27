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


class TestPointTransformCartND:
    """``pt_map`` from CartND reads components on the last axis (batch-safe)."""

    def test_cartnd_to_cart3d_batched(self) -> None:
        # A batch of 2 points in 3D: the dimensionality guard must read the
        # component axis (last), not the batch axis (leading), and not raise.
        p = {"q": u.Q(jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), "m")}
        out = cxc.pt_map(p, cxc.cartnd, cxc.cart3d)
        assert set(out) == {"x", "y", "z"}
        assert u.ustrip("m", out["x"]) == pytest.approx([1.0, 4.0])
        assert u.ustrip("m", out["y"]) == pytest.approx([2.0, 5.0])
        assert u.ustrip("m", out["z"]) == pytest.approx([3.0, 6.0])


_TWO_SPHERE_PTS = {
    "lonlat": (cxc.lonlat_sph2, {"lon": u.Q(45, "deg"), "lat": u.Q(30, "deg")}),
    "math": (cxc.math_sph2, {"theta": u.Q(50, "deg"), "phi": u.Q(35, "deg")}),
    "loncoslat": (
        cxc.loncoslat_sph2,
        {"lon_coslat": u.Q(0.4, "rad"), "lat": u.Q(0.5, "rad")},
    ),
}


class TestPointTransformTwoSphereCrossChart:
    """Non-canonical two-sphere charts convert to each other via SphericalTwoSphere."""

    @pytest.mark.parametrize("src", ["lonlat", "math", "loncoslat"])
    @pytest.mark.parametrize("dst", ["lonlat", "math", "loncoslat"])
    def test_cross_chart_matches_route_via_canonical(self, src, dst):
        if src == dst:
            return
        a, p = _TWO_SPHERE_PTS[src]
        b, _ = _TWO_SPHERE_PTS[dst]
        out = cxc.pt_map(p, a, b)
        ref = cxc.pt_map(cxc.pt_map(p, a, cxc.sph2), cxc.sph2, b)
        assert set(out) == set(ref)
        # Compare in a shared canonical unit (rad) so mismatched unit metadata
        # can't slip through equal magnitudes with different units.
        for k in out:
            assert u.ustrip("rad", out[k]) == pytest.approx(u.ustrip("rad", ref[k]))
