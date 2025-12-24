"""Tests for Cartesian product charts with namespaced and flat keys."""

import jax.numpy as jnp
import pytest

import unxt as u

import coordinax.charts as cxc
import coordinax.transforms as cxt

# =============================================================================
# CartesianProductChart construction


class TestCartesianProductChartConstruction:
    """Test CartesianProductChart construction with factor_names."""

    def test_namespaced_construction(self) -> None:
        """CartesianProductChart requires both factors and factor_names."""
        cart3d = cxc.cart3d
        product = cxc.CartesianProductChart((cart3d, cart3d), ("q", "p"))
        assert product.factors == (cart3d, cart3d)
        assert product.factor_names == ("q", "p")

    def test_components_are_namespaced_tuples(self) -> None:
        """Components should be (factor_name, component_name) tuples."""
        cart3d = cxc.cart3d
        product = cxc.CartesianProductChart((cart3d, cart3d), ("q", "p"))
        expected = (
            ("q", "x"),
            ("q", "y"),
            ("q", "z"),
            ("p", "x"),
            ("p", "y"),
            ("p", "z"),
        )
        assert product.components == expected

    def test_ndim_is_sum_of_factors(self) -> None:
        """Ndim should equal sum of factor dimensions."""
        cart3d = cxc.cart3d
        cart2d = cxc.cart2d
        product = cxc.CartesianProductChart((cart3d, cart2d), ("a", "b"))
        assert product.ndim == 3 + 2

    def test_factor_names_count_must_match_factors(self) -> None:
        """factor_names length must match factors length."""
        cart3d = cxc.cart3d
        with pytest.raises(ValueError, match="same length"):
            cxc.CartesianProductChart((cart3d, cart3d), ("q",))

    def test_factor_names_must_be_unique(self) -> None:
        """factor_names must be unique."""
        cart3d = cxc.cart3d
        with pytest.raises(ValueError, match="unique"):
            cxc.CartesianProductChart((cart3d, cart3d), ("q", "q"))


# =============================================================================
# split_components and merge_components


class TestNamespacedSplitMerge:
    """Test split_components and merge_components for namespaced products."""

    @pytest.fixture
    def phase_space(self) -> cxc.CartesianProductChart:
        """Create a phase space chart with (q, p) factors."""
        return cxc.CartesianProductChart((cxc.cart3d, cxc.cart3d), ("q", "p"))

    def test_split_components_extracts_by_prefix(
        self, phase_space: cxc.CartesianProductChart
    ) -> None:
        """split_components should extract keys by prefix and strip it."""
        p = {
            ("q", "x"): 1.0,
            ("q", "y"): 2.0,
            ("q", "z"): 3.0,
            ("p", "x"): 4.0,
            ("p", "y"): 5.0,
            ("p", "z"): 6.0,
        }
        parts = phase_space.split_components(p)
        assert len(parts) == 2
        assert parts[0] == {"x": 1.0, "y": 2.0, "z": 3.0}
        assert parts[1] == {"x": 4.0, "y": 5.0, "z": 6.0}

    def test_merge_components_reattaches_prefix(
        self, phase_space: cxc.CartesianProductChart
    ) -> None:
        """merge_components should re-add namespace prefix."""
        parts = (
            {"x": 1.0, "y": 2.0, "z": 3.0},
            {"x": 4.0, "y": 5.0, "z": 6.0},
        )
        merged = phase_space.merge_components(parts)
        expected = {
            ("q", "x"): 1.0,
            ("q", "y"): 2.0,
            ("q", "z"): 3.0,
            ("p", "x"): 4.0,
            ("p", "y"): 5.0,
            ("p", "z"): 6.0,
        }
        assert merged == expected

    def test_split_merge_roundtrip(
        self, phase_space: cxc.CartesianProductChart
    ) -> None:
        """Split followed by merge should recover original dict."""
        original = {
            ("q", "x"): 1.0,
            ("q", "y"): 2.0,
            ("q", "z"): 3.0,
            ("p", "x"): 4.0,
            ("p", "y"): 5.0,
            ("p", "z"): 6.0,
        }
        parts = phase_space.split_components(original)
        recovered = phase_space.merge_components(parts)
        assert recovered == original


# =============================================================================
# SpaceTimeCT (flat keys)


class TestSpaceTimeCTFlatKeys:
    """Test SpaceTimeCT uses flat string keys."""

    def test_factor_names_is_none(self) -> None:
        """SpaceTimeCT.factor_names should be None (flat-key specialization)."""
        st = cxc.SpaceTimeCT(cxc.cart3d)
        assert st.factor_names is None

    def test_components_are_flat_strings(self) -> None:
        """SpaceTimeCT components should be flat strings like 'ct', 'x', etc."""
        st = cxc.SpaceTimeCT(cxc.cart3d)
        assert st.components == ("ct", "x", "y", "z")

    def test_split_components_preserves_ct(self) -> None:
        """split_components should keep 'ct' key for time factor."""
        st = cxc.SpaceTimeCT(cxc.cart3d)
        p = {"ct": 1.0, "x": 2.0, "y": 3.0, "z": 4.0}
        parts = st.split_components(p)
        assert parts[0] == {"ct": 1.0}
        assert parts[1] == {"x": 2.0, "y": 3.0, "z": 4.0}

    def test_merge_components_preserves_ct(self) -> None:
        """merge_components should preserve 'ct' key."""
        st = cxc.SpaceTimeCT(cxc.cart3d)
        parts = ({"ct": 1.0}, {"x": 2.0, "y": 3.0, "z": 4.0})
        merged = st.merge_components(parts)
        assert merged == {"ct": 1.0, "x": 2.0, "y": 3.0, "z": 4.0}

    def test_split_merge_roundtrip(self) -> None:
        """Split followed by merge should recover original dict."""
        st = cxc.SpaceTimeCT(cxc.cart3d)
        original = {"ct": 1.0, "x": 2.0, "y": 3.0, "z": 4.0}
        parts = st.split_components(original)
        recovered = st.merge_components(parts)
        assert recovered == original


# =============================================================================
# SpaceTimeEuclidean (flat keys)


class TestSpaceTimeEuclideanFlatKeys:
    """Test SpaceTimeEuclidean uses flat string keys."""

    def test_factor_names_is_none(self) -> None:
        """SpaceTimeEuclidean.factor_names should be None."""
        ste = cxc.SpaceTimeEuclidean(cxc.cart3d)
        assert ste.factor_names is None

    def test_components_are_flat_strings(self) -> None:
        """SpaceTimeEuclidean components should be flat strings."""
        ste = cxc.SpaceTimeEuclidean(cxc.cart3d)
        assert ste.components == ("ct", "x", "y", "z")

    def test_split_merge_roundtrip(self) -> None:
        """Split followed by merge should recover original dict."""
        ste = cxc.SpaceTimeEuclidean(cxc.cart3d)
        original = {"ct": 1.0, "x": 2.0, "y": 3.0, "z": 4.0}
        parts = ste.split_components(original)
        recovered = ste.merge_components(parts)
        assert recovered == original


# =============================================================================
# point_transform with product charts


class TestPointTransformProductCharts:
    """Test point_transform works correctly with product charts."""

    def test_spacetime_ct_cart_to_sph(self) -> None:
        """point_transform works between SpaceTimeCT(cart3d) and SpaceTimeCT(sph3d)."""
        st_cart = cxc.SpaceTimeCT(cxc.cart3d)
        st_sph = cxc.SpaceTimeCT(cxc.sph3d)
        p = {
            "ct": u.Q(1.0, "s"),
            "x": u.Q(1.0, "m"),
            "y": u.Q(0.0, "m"),
            "z": u.Q(0.0, "m"),
        }
        result = cxt.point_transform(st_sph, st_cart, p)
        # Check time is preserved
        assert u.ustrip("s", result["ct"]) == pytest.approx(1.0)
        # Check spatial transform
        assert u.ustrip("m", result["r"]) == pytest.approx(1.0)

    def test_spacetime_euclidean_cart_to_sph(self) -> None:
        """point_transform should work between SpaceTimeEuclidean variants."""
        ste_cart = cxc.SpaceTimeEuclidean(cxc.cart3d)
        ste_sph = cxc.SpaceTimeEuclidean(cxc.sph3d)
        p = {
            "ct": u.Q(2.0, "km"),
            "x": u.Q(3.0, "m"),
            "y": u.Q(4.0, "m"),
            "z": u.Q(0.0, "m"),
        }
        result = cxt.point_transform(ste_sph, ste_cart, p)
        # Check time is preserved
        assert u.ustrip("km", result["ct"]) == pytest.approx(2.0)
        # Check spatial r = sqrt(3^2 + 4^2) = 5
        assert u.ustrip("m", result["r"]) == pytest.approx(5.0)

    def test_namespaced_phase_space_transform(self) -> None:
        """point_transform should work with namespaced CartesianProductChart."""
        phase_cart = cxc.CartesianProductChart((cxc.cart3d, cxc.cart3d), ("q", "p"))
        phase_sph = cxc.CartesianProductChart((cxc.sph3d, cxc.sph3d), ("q", "p"))
        p = {
            ("q", "x"): u.Q(1.0, "m"),
            ("q", "y"): u.Q(0.0, "m"),
            ("q", "z"): u.Q(0.0, "m"),
            ("p", "x"): u.Q(0.0, "m"),
            ("p", "y"): u.Q(1.0, "m"),
            ("p", "z"): u.Q(0.0, "m"),
        }
        result = cxt.point_transform(phase_sph, phase_cart, p)
        # Check q factor: (1, 0, 0) -> r=1, theta=pi/2, phi=0
        assert u.ustrip("m", result[("q", "r")]) == pytest.approx(1.0)
        assert u.ustrip("rad", result[("q", "phi")]) == pytest.approx(0.0)
        # Check p factor: (0, 1, 0) -> r=1, theta=pi/2, phi=pi/2
        assert u.ustrip("m", result[("p", "r")]) == pytest.approx(1.0)

        assert u.ustrip("rad", result[("p", "phi")]) == pytest.approx(jnp.pi / 2)


# =============================================================================
# cartesian_chart for product charts


class TestCartesianChartProductCharts:
    """Test cartesian_chart dispatch for product charts."""

    def test_spacetime_ct_cartesian_chart(self) -> None:
        """cartesian_chart should convert SpaceTimeCT spatial to Cart3D."""
        st_sph = cxc.SpaceTimeCT(cxc.sph3d)
        st_cart = cxc.cartesian_chart(st_sph)
        assert isinstance(st_cart.spatial_chart, cxc.Cart3D)

    def test_spacetime_euclidean_cartesian_chart(self) -> None:
        """cartesian_chart should convert SpaceTimeEuclidean spatial to Cart3D."""
        ste_sph = cxc.SpaceTimeEuclidean(cxc.sph3d)
        ste_cart = cxc.cartesian_chart(ste_sph)
        assert isinstance(ste_cart.spatial_chart, cxc.Cart3D)

    def test_namespaced_product_cartesian_chart(self) -> None:
        """cartesian_chart should convert factors while preserving factor_names."""
        phase_sph = cxc.CartesianProductChart((cxc.sph3d, cxc.sph3d), ("q", "p"))
        phase_cart = cxc.cartesian_chart(phase_sph)
        # Should have Cart3D factors
        assert isinstance(phase_cart.factors[0], cxc.Cart3D)
        assert isinstance(phase_cart.factors[1], cxc.Cart3D)
        # Should preserve factor_names
        assert phase_cart.factor_names == ("q", "p")

    def test_cartesian_chart_idempotent(self) -> None:
        """cartesian_chart applied twice should return same object."""
        phase_sph = cxc.CartesianProductChart((cxc.sph3d, cxc.sph3d), ("q", "p"))
        cart1 = cxc.cartesian_chart(phase_sph)
        cart2 = cxc.cartesian_chart(cart1)
        assert cart1 is cart2
