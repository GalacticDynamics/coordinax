"""Tests for product_charts ndim parameter."""

import hypothesis.strategies as st
from hypothesis import HealthCheck, given, settings

import coordinax.charts as cxc
import coordinax_hypothesis.core as cxst


@given(
    ndim=st.integers(min_value=2, max_value=6),
    chart=st.data(),
)
@settings(
    max_examples=5,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
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
@settings(max_examples=5, suppress_health_check=[HealthCheck.too_slow], deadline=None)
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
