"""Tests for the charts strategy (instance generation)."""

import hypothesis.strategies as st
from hypothesis import given, settings

import coordinax as cx
import coordinax_hypothesis as cxst


@given(chart=cxst.charts())
@settings(max_examples=50)
def test_basic_chart(chart: cx.charts.AbstractChart) -> None:
    """Test basic chart instance generation."""
    assert isinstance(chart, cx.charts.AbstractChart)


@given(chart=cxst.charts(filter=cx.charts.Abstract3D))
@settings(max_examples=50)
def test_3d_chart(chart: cx.charts.AbstractChart) -> None:
    """Test 3D chart instance generation."""
    assert isinstance(chart, cx.charts.Abstract3D)
    assert isinstance(chart, cx.charts.AbstractChart)


@given(chart=cxst.charts(filter=cx.charts.Abstract2D))
@settings(max_examples=50)
def test_2d_chart(chart: cx.charts.AbstractChart) -> None:
    """Test 2D chart instance generation."""
    assert isinstance(chart, cx.charts.Abstract2D)
    assert isinstance(chart, cx.charts.AbstractChart)


@given(chart=cxst.charts(filter=cx.charts.Abstract1D))
@settings(max_examples=50)
def test_1d_chart(chart: cx.charts.AbstractChart) -> None:
    """Test 1D chart instance generation."""
    assert isinstance(chart, cx.charts.Abstract1D)
    assert isinstance(chart, cx.charts.AbstractChart)


@given(chart=cxst.charts(filter=(cx.charts.Abstract3D, cx.charts.AbstractSpherical3D)))
@settings(max_examples=50)
def test_spherical_3d_chart(chart: cx.charts.AbstractChart) -> None:
    """Test chart instance generation with multiple types."""
    assert isinstance(chart, cx.charts.AbstractChart)
    assert isinstance(chart, cx.charts.Abstract3D)
    assert isinstance(chart, cx.charts.AbstractSpherical3D)


@given(
    chart=cxst.charts(
        filter=st.sampled_from([cx.charts.Abstract1D, cx.charts.Abstract2D])
    )
)
@settings(max_examples=50)
def test_dynamic_filter(chart: cx.charts.AbstractChart) -> None:
    """Test chart instance generation with dynamic union class."""
    assert isinstance(chart, cx.charts.AbstractChart)
    assert isinstance(chart, (cx.charts.Abstract1D, cx.charts.Abstract2D))


@given(chart=cxst.charts(dimensionality=None))
@settings(max_examples=50)
def test_dimensionality_none_allows_0d(chart: cx.charts.AbstractChart) -> None:
    """Test that dimensionality=None allows all dimensionalities."""
    assert isinstance(chart, cx.charts.AbstractChart)
    # Could be 0D or higher


@given(chart=cxst.charts(dimensionality=2))
@settings(max_examples=50)
def test_exact_dimensionality(chart: cx.charts.AbstractChart) -> None:
    """Test exact dimensionality match."""
    assert chart.ndim == 2


@given(chart=cxst.charts(dimensionality=st.integers(min_value=1, max_value=2)))
@settings(max_examples=50)
def test_dimensionality_strategy(chart: cx.charts.AbstractChart) -> None:
    """Test dimensionality as a strategy."""
    assert 1 <= chart.ndim <= 2


@given(chart=cxst.charts(filter=cx.charts.EmbeddedManifold))
@settings(max_examples=20)
def test_embedded_manifold_chartresentation(
    chart: cx.charts.AbstractChart,
) -> None:
    """Test EmbeddedManifold chart instance generation."""
    assert isinstance(chart, cx.charts.EmbeddedManifold)
    assert isinstance(chart.intrinsic_chart, cx.charts.TwoSphere)
    assert isinstance(chart.ambient_chart, cx.charts.Cart3D)
    assert "R" in chart.params
