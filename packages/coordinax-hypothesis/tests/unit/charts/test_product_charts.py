"""Tests for coordinax_hypothesis.product_charts strategy."""

import hypothesis.strategies as st
import pytest
from hypothesis import given
from hypothesis.errors import Unsatisfiable

import coordinax.charts as cxc
import coordinax_hypothesis.core as cxst


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
        """factor_charts generates CartesianProductChart with namespaced keys."""
        assert isinstance(chart, cxc.CartesianProductChart)
        # Components should be tuples (factor_name, component_name)
        assert all(isinstance(c, tuple) and len(c) == 2 for c in chart.components)
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

        # For namespaced products, components are tuples
        assert all(isinstance(k, tuple) and len(k) == 2 for k in data)

    @given(
        chart=cxst.product_charts(
            factor_charts=(cxc.cart3d, cxc.cart2d),
            factor_names=("q", "p"),
        )
    )
    def test_split_merge_roundtrip(self, chart: cxc.CartesianProductChart) -> None:
        """split_components and merge_components are inverses."""
        # Create test data matching chart structure
        test_data = {
            (fname, cname): float(i)
            for i, (fname, cname) in enumerate(chart.components)
        }

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
        # Components can be either tuples (namespaced) or strings (flat-key)
        if isinstance(chart, cxc.CartesianProductChart):
            assert all(isinstance(c, tuple) for c in chart.components)
        else:  # Flat-key specializations
            assert all(isinstance(c, str) for c in chart.components)
