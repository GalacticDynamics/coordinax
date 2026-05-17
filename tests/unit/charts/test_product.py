"""Tests for Cartesian product charts (product.py).

CartesianProductChart: namespaced and flat component key variants.
"""

import pytest
from hypothesis import given, settings

import coordinax.charts as cxc
import coordinax.hypothesis.main as cxst

# =============================================================================
# CartesianProductChart construction
# =============================================================================


class TestCartesianProductChartConstruction:
    """Test CartesianProductChart construction with factor_names."""

    def test_namespaced_construction(self) -> None:
        """CartesianProductChart requires both factors and factor_names."""
        cart3d = cxc.cart3d
        product = cxc.CartesianProductChart((cart3d, cart3d), ("q", "p"))
        assert product.factors == (cart3d, cart3d)
        assert product.factor_names == ("q", "p")

    def test_components_are_namespaced_tuples(self) -> None:
        """Components should be 'factor_name.component_name' dot-delimited strings."""
        cart3d = cxc.cart3d
        product = cxc.CartesianProductChart((cart3d, cart3d), ("q", "p"))
        expected = ("q.x", "q.y", "q.z", "p.x", "p.y", "p.z")
        assert product.components == expected

    def test_ndim_is_sum_of_factors(self) -> None:
        """Ndim should equal sum of factor dimensions."""
        cart3d = cxc.cart3d
        cart2d = cxc.cart2d
        product = cxc.CartesianProductChart((cart3d, cart2d), ("a", "b"))
        assert product.ndim == 3 + 2

    def test_factors(self) -> None:
        """Product chart stores factors."""
        product = cxc.CartesianProductChart((cxc.cart3d, cxc.sph3d), ("a", "b"))
        assert product.factors == (cxc.cart3d, cxc.sph3d)

    def test_factor_names(self) -> None:
        """Product chart stores factor names."""
        product = cxc.CartesianProductChart((cxc.cart3d, cxc.sph3d), ("a", "b"))
        assert product.factor_names == ("a", "b")

    def test_factor_names_count_must_match_factors(self) -> None:
        """factor_names length must match factors length."""
        with pytest.raises(ValueError, match="same length"):
            cxc.CartesianProductChart((cxc.cart3d, cxc.cart3d), ("q",))

    def test_factor_names_must_be_unique(self) -> None:
        """factor_names must be unique."""
        with pytest.raises(ValueError, match="unique"):
            cxc.CartesianProductChart((cxc.cart3d, cxc.cart3d), ("q", "q"))


# =============================================================================
# split_components and merge_components
# =============================================================================


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
        p = {"q.x": 1, "q.y": 2, "q.z": 3, "p.x": 4, "p.y": 5, "p.z": 6}
        parts = phase_space.split_components(p)
        assert len(parts) == 2
        assert parts[0] == {"x": 1, "y": 2, "z": 3}
        assert parts[1] == {"x": 4, "y": 5, "z": 6}

    def test_merge_components_reattaches_prefix(
        self, phase_space: cxc.CartesianProductChart
    ) -> None:
        """merge_components should re-add dot-delimited prefix."""
        parts = ({"x": 1, "y": 2, "z": 3}, {"x": 4, "y": 5, "z": 6})
        merged = phase_space.merge_components(parts)
        expected = {"q.x": 1, "q.y": 2, "q.z": 3, "p.x": 4, "p.y": 5, "p.z": 6}
        assert merged == expected

    def test_split_merge_roundtrip(
        self, phase_space: cxc.CartesianProductChart
    ) -> None:
        """Split followed by merge should recover original dict."""
        original = {"q.x": 1, "q.y": 2, "q.z": 3, "p.x": 4, "p.y": 5, "p.z": 6}
        parts = phase_space.split_components(original)
        recovered = phase_space.merge_components(parts)
        assert recovered == original


# =============================================================================
# Property tests using coordinax-hypothesis
# =============================================================================


class TestCartesianProductChartPropertyTests:
    """Property tests for CartesianProductChart."""

    @given(chart=cxst.charts(cxc.AbstractCartesianProductChart))
    @settings(deadline=None)
    def test_product_ndim_is_sum_of_factor_ndims(
        self, chart: cxc.CartesianProductChart
    ) -> None:
        """Property: product chart ndim is sum of factor ndims."""
        expected_ndim = sum(f.ndim for f in chart.factors)
        assert chart.ndim == expected_ndim

    @given(chart=cxst.charts(cxc.AbstractCartesianProductChart))
    @settings(deadline=None)
    def test_product_factors_match_factor_names_length(
        self, chart: cxc.CartesianProductChart
    ) -> None:
        """Property: factors and factor_names have same length."""
        assert len(chart.factors) == len(chart.factor_names)

    @given(chart=cxst.charts(cxc.AbstractCartesianProductChart))
    @settings(deadline=None)
    def test_product_factor_names_are_unique(
        self, chart: cxc.CartesianProductChart
    ) -> None:
        """Property: factor_names are unique."""
        assert len(set(chart.factor_names)) == len(chart.factor_names)

    @given(chart=cxst.charts(cxc.AbstractCartesianProductChart))
    @settings(deadline=None)
    def test_split_merge_roundtrip(self, chart: cxc.CartesianProductChart) -> None:
        """Property: split then merge is identity."""
        data = {comp: float(i) for i, comp in enumerate(chart.components)}
        parts = chart.split_components(data)
        merged = chart.merge_components(parts)
        assert merged == data
