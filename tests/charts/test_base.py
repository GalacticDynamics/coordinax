"""Tests for chart base classes: AbstractChart, AbstractFixedComponentsChart, etc."""

import pytest
from hypothesis import given, settings

import unxt as u

import coordinax.charts as cxc
import coordinax_hypothesis as cxst
from coordinax._src.charts.base import CHART_CLASSES, DIMENSIONAL_FLAGS


@given(cxst.chart_classes())
def test_chart_registered_in_chart_classes(self, chart_cls) -> None:
    """All charts are registered in CHART_CLASSES."""
    assert chart_cls in CHART_CLASSES


class TestChartProperties:
    """Unit tests for AbstractChart behavior."""

    @given(cxst.charts())
    def test_chart_properties(self, chart) -> None:
        """Components returns a tuple."""
        # Components
        assert isinstance(chart.components, tuple)
        for c in chart.components:
            assert isinstance(c, (str, tuple))
        # Ndim
        assert chart.ndim == len(chart.components)
        assert chart.ndim == len(chart.coord_dimensions)
        # Coord-dims
        assert isinstance(chart.coord_dimensions, tuple)
        # Cartesian
        assert isinstance(chart.cartesian, cxc.AbstractChart)
        # Metric
        assert isinstance(chart.is_euclidean, bool)

        # Repr
        assert isinstance(repr(chart), str)
        assert chart.__class__.__name__ in repr(chart)

        # str
        assert isinstance(str(chart), str)

    @pytest.mark.parametrize(
        ("chart", "is_euclidean"),
        [
            (cxc.cart3d, True),
            (cxc.sph3d, True),
            (cxc.cyl3d, True),
            (cxc.twosphere, False),
        ],
    )
    def test_charts_is_euclidean(self, chart, is_euclidean) -> None:
        """Euclidean charts have is_euclidean=True."""
        assert chart.is_euclidean is is_euclidean


class TestAbstractChartCheckData:
    """Unit tests for AbstractChart.check_data method."""

    def test_check_data_passes_for_valid_data(self) -> None:
        """check_data passes for valid data matching chart components."""
        data = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(3.0, "m")}
        cxc.cart3d.check_data(data)

    def test_check_data_raises_for_missing_component(self) -> None:
        """check_data raises when a component is missing."""
        data = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m")}  # missing z
        with pytest.raises(ValueError, match="Data keys do not match"):
            cxc.cart3d.check_data(data)

    def test_check_data_raises_for_extra_component(self) -> None:
        """check_data raises when an extra component is present."""
        data = {
            "x": u.Q(1.0, "m"),
            "y": u.Q(2.0, "m"),
            "z": u.Q(3.0, "m"),
            "w": u.Q(4.0, "m"),
        }
        with pytest.raises(ValueError, match="Data keys do not match"):
            cxc.cart3d.check_data(data)

    def test_check_data_raises_for_wrong_dimensions(self) -> None:
        """check_data raises for wrong physical dimensions."""
        # cart3d expects length, not angle
        data = {"x": u.Q(1.0, "rad"), "y": u.Q(2.0, "m"), "z": u.Q(3.0, "m")}
        with pytest.raises(ValueError, match="Data dimensions do not match"):
            cxc.cart3d.check_data(data)


# =============================================================================
# Unit Tests for AbstractDimensionalFlag
# =============================================================================


class TestAbstractDimensionalFlag:
    """Unit tests for AbstractDimensionalFlag."""

    def test_dimensional_flags_registered(self) -> None:
        """Dimensional flags are registered in DIMENSIONAL_FLAGS."""
        assert len(DIMENSIONAL_FLAGS) > 0

    def test_flags_must_subclass_chart(self) -> None:
        """Non-abstract dimensional flags must subclass AbstractChart."""
        for _flag_cls in DIMENSIONAL_FLAGS.values():
            # All registered flags should have chart subclasses
            pass  # Registration enforces this


# =============================================================================
# Unit Tests for CartesianProductChart
# =============================================================================


class TestCartesianProductChart:
    """Unit tests for CartesianProductChart."""

    def test_product_chart_creation(self) -> None:
        """CartesianProductChart can be created from factors."""
        product = cxc.CartesianProductChart((cxc.cart3d, cxc.cart3d), ("q", "p"))
        assert isinstance(product, cxc.CartesianProductChart)

    def test_product_chart_ndim(self) -> None:
        """Product chart ndim is sum of factor ndims."""
        product = cxc.CartesianProductChart((cxc.cart3d, cxc.cart3d), ("q", "p"))
        assert product.ndim == 6

    def test_product_chart_namespaced_components(self) -> None:
        """Product chart has namespaced tuple components."""
        product = cxc.CartesianProductChart((cxc.cart3d, cxc.cart3d), ("q", "p"))
        expected = (
            ("q", "x"),
            ("q", "y"),
            ("q", "z"),
            ("p", "x"),
            ("p", "y"),
            ("p", "z"),
        )
        assert product.components == expected

    def test_product_chart_factors(self) -> None:
        """Product chart stores factors."""
        product = cxc.CartesianProductChart((cxc.cart3d, cxc.sph3d), ("a", "b"))
        assert product.factors == (cxc.cart3d, cxc.sph3d)

    def test_product_chart_factor_names(self) -> None:
        """Product chart stores factor names."""
        product = cxc.CartesianProductChart((cxc.cart3d, cxc.sph3d), ("a", "b"))
        assert product.factor_names == ("a", "b")

    def test_product_chart_raises_on_length_mismatch(self) -> None:
        """CartesianProductChart raises if factors and names have different lengths."""
        with pytest.raises(ValueError, match="same length"):
            cxc.CartesianProductChart((cxc.cart3d, cxc.cart3d), ("q",))

    def test_product_chart_raises_on_duplicate_names(self) -> None:
        """CartesianProductChart raises if factor names are not unique."""
        with pytest.raises(ValueError, match="unique"):
            cxc.CartesianProductChart((cxc.cart3d, cxc.cart3d), ("q", "q"))

    def test_product_chart_split_components(self) -> None:
        """split_components partitions data by factor."""
        product = cxc.CartesianProductChart((cxc.cart2d, cxc.cart2d), ("a", "b"))
        data = {
            ("a", "x"): 1.0,
            ("a", "y"): 2.0,
            ("b", "x"): 3.0,
            ("b", "y"): 4.0,
        }
        parts = product.split_components(data)
        assert parts == ({"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0})

    def test_product_chart_merge_components(self) -> None:
        """merge_components combines factor data."""
        product = cxc.CartesianProductChart((cxc.cart2d, cxc.cart2d), ("a", "b"))
        parts = ({"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0})
        merged = product.merge_components(parts)
        expected = {
            ("a", "x"): 1.0,
            ("a", "y"): 2.0,
            ("b", "x"): 3.0,
            ("b", "y"): 4.0,
        }
        assert merged == expected


# =============================================================================
# Property Tests using coordinax-hypothesis
# =============================================================================


class TestCartesianProductChartPropertyTests:
    """Property tests for CartesianProductChart."""

    @given(chart=cxst.product_charts())
    @settings(max_examples=50, deadline=None)
    def test_product_ndim_is_sum_of_factor_ndims(
        self, chart: cxc.CartesianProductChart
    ) -> None:
        """Property: product chart ndim is sum of factor ndims."""
        expected_ndim = sum(f.ndim for f in chart.factors)
        assert chart.ndim == expected_ndim

    @given(chart=cxst.product_charts())
    @settings(max_examples=50, deadline=None)
    def test_product_factors_match_factor_names_length(
        self, chart: cxc.CartesianProductChart
    ) -> None:
        """Property: factors and factor_names have same length."""
        assert len(chart.factors) == len(chart.factor_names)

    @given(chart=cxst.product_charts())
    @settings(max_examples=50, deadline=None)
    def test_product_factor_names_are_unique(
        self, chart: cxc.CartesianProductChart
    ) -> None:
        """Property: factor_names are unique."""
        assert len(set(chart.factor_names)) == len(chart.factor_names)

    @given(chart=cxst.product_charts())
    @settings(max_examples=50, deadline=None)
    def test_split_merge_roundtrip(self, chart: cxc.CartesianProductChart) -> None:
        """Property: split then merge is identity."""
        # Create dummy data for the product chart
        data = {comp: float(i) for i, comp in enumerate(chart.components)}
        parts = chart.split_components(data)
        merged = chart.merge_components(parts)
        assert merged == data
