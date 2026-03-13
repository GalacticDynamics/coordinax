"""Tests for coordinax.hypothesis.product_charts strategy."""

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings
from hypothesis.errors import Unsatisfiable

import coordinax.charts as cxc
import coordinax.hypothesis.main as cxst


class TestProductChartsBasic:
    """Basic tests for product_charts strategy."""

    @given(chart=cxst.product_charts())
    def test_generates_product_chart(
        self, chart: cxc.AbstractCartesianProductChart
    ) -> None:
        """Generated objects are CartesianProductChart instances."""
        assert isinstance(chart, cxc.AbstractCartesianProductChart)
        assert hasattr(chart, "factors")
        assert hasattr(chart, "factor_names")
        assert len(chart.factors) >= 1

    @given(chart=cxst.product_charts())
    def test_factor_names_never_none(
        self, chart: cxc.AbstractCartesianProductChart
    ) -> None:
        """factor_names is always a tuple, never None."""
        factor_names = chart.factor_names
        assert isinstance(factor_names, tuple)
        assert all(isinstance(name, str) for name in factor_names)
        assert len(factor_names) == len(chart.factors)

    @given(chart=cxst.product_charts())
    def test_factor_names_unique(
        self, chart: cxc.AbstractCartesianProductChart
    ) -> None:
        """All factor names are unique."""
        # Only check for namespaced products (CartesianProductChart)
        if isinstance(chart, cxc.CartesianProductChart):
            factor_names = chart.factor_names
            assert len(factor_names) == len(set(factor_names))

    @given(chart=cxst.product_charts())
    def test_has_dispitive_dimension(
        self, chart: cxc.AbstractCartesianProductChart
    ) -> None:
        """Product chart dimension is positive (sum of factor dimensions)."""
        assert chart.ndim > 0
        expected_ndim = sum(f.ndim for f in chart.factors)
        assert chart.ndim == expected_ndim


class TestProductChartsWithFixedFactors:
    """Tests for product_charts with specific factor_charts."""

    @given(
        chart=cxst.product_charts(
            factor_charts=(cxc.cart3d, cxc.cart2d),
            factor_names=("position", "velocity"),
        )
    )
    def test_fixed_factors(self, chart: cxc.AbstractCartesianProductChart) -> None:
        """Can generate product with fixed factors and names."""
        assert len(chart.factors) == 2
        assert chart.factors[0] is cxc.cart3d
        assert chart.factors[1] is cxc.cart2d
        assert chart.ndim == 5  # 3 + 2

    @given(
        chart=cxst.product_charts(
            factor_charts=st.just((cxc.cart1d, cxc.cart1d, cxc.cart1d)),
            factor_names=st.just(("x", "y", "z")),
        )
    )
    def test_strategy_factor_charts(
        self, chart: cxc.AbstractCartesianProductChart
    ) -> None:
        """Can use strategies for factor_charts and factor_names."""
        assert len(chart.factors) == 3
        assert all(f is cxc.cart1d for f in chart.factors)
        assert chart.factor_names == ("x", "y", "z")

    @given(
        chart=cxst.product_charts(
            factor_charts=(cxc.sph3d, cxc.cart2d),
        )
    )
    def test_mixed_chart_types(self, chart: cxc.AbstractCartesianProductChart) -> None:
        """Can mix different chart types as factors."""
        assert len(chart.factors) == 2
        assert chart.factors[0] is cxc.sph3d
        assert chart.factors[1] is cxc.cart2d
        # Default names should be generated
        assert len(chart.factor_names) == 2


class TestProductChartsTypes:
    """Tests for different product chart types."""

    @given(chart=cxst.product_charts())
    def test_generates_product_charts(
        self, chart: cxc.AbstractCartesianProductChart
    ) -> None:
        """Strategy generates valid product charts."""
        # Can be either CartesianProductChart or specialized types
        assert isinstance(chart, cxc.AbstractCartesianProductChart)
        assert len(chart.factors) >= 1

    @given(
        chart=cxst.product_charts(
            factor_charts=(cxc.cart3d, cxc.cart3d),
            factor_names=("q", "p"),
        )
    )
    def test_explicit_factors_generates_namespaced(
        self, chart: cxc.AbstractCartesianProductChart
    ) -> None:
        """factor_charts generates CartesianProductChart with dot-delimited keys."""
        assert isinstance(chart, cxc.CartesianProductChart)
        # Components should be dot-delimited strings "factor_name.component_name"
        assert all(isinstance(c, str) and "." in c for c in chart.components)
        assert len(chart.factors) == 2
        assert chart.factor_names == ("q", "p")


class TestProductChartsFactorCount:
    """Tests for min_factors and max_factors parameters."""

    @given(chart=cxst.product_charts(min_factors=2, max_factors=2))
    def test_exact_factor_count(self, chart: cxc.AbstractCartesianProductChart) -> None:
        """Can generate products with exact factor count."""
        # SpaceTime classes always have conceptually 2 factors
        # CartesianProductChart should have exactly 2
        if isinstance(chart, cxc.CartesianProductChart):
            assert len(chart.factors) == 2

    @given(chart=cxst.product_charts(min_factors=3, max_factors=5))
    def test_factor_count_range(self, chart: cxc.AbstractCartesianProductChart) -> None:
        """Generated factor count respects min/max bounds."""
        # Skip SpaceTime specializations which always have 2 conceptual factors
        if isinstance(chart, cxc.CartesianProductChart):
            assert 3 <= len(chart.factors) <= 5

    @given(chart=cxst.product_charts(min_factors=1, max_factors=1))
    def test_single_factor_product(
        self, chart: cxc.AbstractCartesianProductChart
    ) -> None:
        """Can generate product with single factor."""
        # Skip SpaceTime specializations which always have 2 conceptual factors
        if isinstance(chart, cxc.CartesianProductChart):
            assert len(chart.factors) == 1
            # Still should have factor_names
            assert len(chart.factor_names) == 1


class TestProductChartsIntegration:
    """Integration tests with other strategies."""

    @given(
        chart=cxst.product_charts(
            factor_charts=(cxc.cart3d, cxc.cart2d),
            factor_names=("q", "p"),
        ),
        pdict=st.data(),
    )
    def test_with_cdicts_strategy(
        self, chart: cxc.CartesianProductChart, pdict: st.DataObject
    ) -> None:
        """Generated product charts work with cdicts strategy."""
        # Generate a matching pdict for this chart
        data = pdict.draw(cxst.cdicts(chart=chart))

        # Verify keys match chart components
        assert set(data.keys()) == set(chart.components)

        # For namespaced products, components are dot-delimited strings
        assert all(isinstance(k, str) and "." in k for k in data)

    @given(
        chart=cxst.product_charts(
            factor_charts=(cxc.cart3d, cxc.cart2d),
            factor_names=("q", "p"),
        )
    )
    def test_split_merge_roundtrip(self, chart: cxc.CartesianProductChart) -> None:
        """split_components and merge_components are inverses."""
        # Create test data matching chart structure
        test_data = {comp: float(i) for i, comp in enumerate(chart.components)}

        # Split into factor-specific dicts
        factor_dicts = chart.split_components(test_data)

        # Should have one dict per factor
        assert len(factor_dicts) == len(chart.factors)

        # Merge back together
        merged = chart.merge_components(factor_dicts)

        # Should recover original data
        assert merged == test_data


class TestProductChartsEdgeCases:
    """Edge cases and error conditions."""

    def test_mismatched_lengths_rejected(self) -> None:
        """Hypothesis rejects examples with mismatched factor/name counts."""

        # This should be caught by assume() in the strategy
        @given(
            chart=cxst.product_charts(
                factor_charts=st.just((cxc.cart3d, cxc.cart2d)),
                factor_names=st.just(("only_one",)),  # Wrong length!
            )
        )
        def test_func(chart: cxc.AbstractCartesianProductChart) -> None:
            # Should never reach here due to assume(len(factors) == len(names))
            pytest.fail("Should have been filtered by assume()")

        # This test expects Unsatisfiable when no valid examples exist
        with pytest.raises(Unsatisfiable):
            test_func()

    @given(
        chart=cxst.product_charts(
            factor_charts=st.lists(
                cxst.charts(exclude=(cxc.Abstract0D,)),
                min_size=1,
                max_size=3,
            ).map(tuple)
        )
    )
    def test_random_factor_lists(
        self, chart: cxc.AbstractCartesianProductChart
    ) -> None:
        """Can handle randomly generated factor lists."""
        assert len(chart.factors) >= 1
        # No 0D charts should appear (excluded in strategy)
        assert all(f.ndim > 0 for f in chart.factors)


class TestProductChartsDocExamples:
    """Test examples from docstring."""

    @given(chart=cxst.product_charts())
    def test_basic_example(self, chart: cxc.AbstractCartesianProductChart) -> None:
        """Basic usage example from docstring."""
        assert isinstance(chart, cxc.AbstractCartesianProductChart)
        assert len(chart.factors) >= 1

    @given(
        chart=cxst.product_charts(
            factor_charts=(cxc.cart3d, cxc.cart3d),
            factor_names=("q", "p"),
        )
    )
    def test_phase_space_example(
        self, chart: cxc.AbstractCartesianProductChart
    ) -> None:
        """Phase space example from docstring."""
        assert chart.ndim == 6
        assert len(chart.factors) == 2

    @given(chart=cxst.product_charts())
    def test_can_generate_both_types(
        self, chart: cxc.AbstractCartesianProductChart
    ) -> None:
        """Strategy can generate both namespaced and flat-key products."""
        # Just verify it's a valid product chart
        assert isinstance(chart, cxc.AbstractCartesianProductChart)
        # Components are always strings (dot-delimited for namespaced, plain for
        # flat-key)
        if isinstance(chart, cxc.CartesianProductChart):
            assert all(isinstance(c, str) and "." in c for c in chart.components)
        else:  # Flat-key specializations
            assert all(isinstance(c, str) for c in chart.components)


@given(
    ndim=st.integers(min_value=2, max_value=6),
    chart=st.data(),
)
@settings(max_examples=5, deadline=None)
def test_product_charts_with_ndim(ndim: int, chart: st.DataObject) -> None:
    """Test that product_charts respects ndim parameter."""
    # Draw chart with specified ndim
    product_chart = chart.draw(cxst.product_charts(ndim=ndim))

    # Verify total dimensionality matches
    assert product_chart.ndim == ndim
    assert isinstance(product_chart, cxc.AbstractCartesianProductChart)

    # Verify factors sum to total dimension
    total_factor_dim = sum(f.ndim for f in product_chart.factors)
    assert total_factor_dim == ndim


@given(kwargs=cxst.chart_init_kwargs(cxc.CartesianProductChart))
@settings(max_examples=5, deadline=None)
def test_cartesian_product_chart_init_kwargs(kwargs: dict) -> None:
    """Test chart_init_kwargs for CartesianProductChart."""
    # Verify kwargs structure
    assert "factors" in kwargs
    assert "factor_names" in kwargs

    factors = kwargs["factors"]
    names = kwargs["factor_names"]

    # Verify structure
    assert isinstance(factors, tuple)
    assert isinstance(names, tuple)
    assert len(factors) == len(names)
    assert len(factors) >= 1

    # Verify each factor is a chart
    for factor in factors:
        assert isinstance(factor, cxc.AbstractChart)
        assert factor.ndim >= 1

    # Verify names are strings
    for name in names:
        assert isinstance(name, str)

    # Verify we can create the chart
    chart = cxc.CartesianProductChart(**kwargs)
    assert isinstance(chart, cxc.CartesianProductChart)

    # Verify total dimension
    expected_dim = sum(f.ndim for f in factors)
    assert chart.ndim == expected_dim


@given(
    data=st.data(),
    factor_charts=st.lists(
        st.sampled_from([cxc.cart1d, cxc.polar2d, cxc.cart3d]),
        min_size=2,
        max_size=3,
    ),
)
@settings(max_examples=5)
def test_product_charts_with_fixed_factors_and_ndim(
    data: st.DataObject, factor_charts: list[cxc.AbstractChart]
) -> None:
    """Test product_charts with fixed factors (ndim is derived)."""
    # Create product chart with fixed factors
    chart = data.draw(cxst.product_charts(factor_charts=tuple(factor_charts)))

    # Verify structure
    assert isinstance(chart, cxc.CartesianProductChart)
    assert len(chart.factors) == len(factor_charts)

    # Verify total dimension
    expected_dim = sum(f.ndim for f in factor_charts)
    assert chart.ndim == expected_dim
