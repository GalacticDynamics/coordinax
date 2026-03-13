"""Tests for the charts strategy (instance generation)."""

import hypothesis.strategies as st
from hypothesis import given, settings

import coordinax.charts as cxc
import coordinax.hypothesis.main as cxst


@given(chart=cxst.charts())
@settings(max_examples=50)
def test_basic_chart(chart: cxc.AbstractChart) -> None:
    """Test basic chart instance generation."""
    assert isinstance(chart, cxc.AbstractChart)


@given(chart=cxst.charts(filter=cxc.Abstract3D))
@settings(max_examples=50)
def test_3d_chart(chart: cxc.AbstractChart) -> None:
    """Test 3D chart instance generation."""
    assert isinstance(chart, cxc.Abstract3D)
    assert isinstance(chart, cxc.AbstractChart)


@given(chart=cxst.charts(filter=cxc.Abstract2D))
@settings(max_examples=50)
def test_2d_chart(chart: cxc.AbstractChart) -> None:
    """Test 2D chart instance generation."""
    assert isinstance(chart, cxc.Abstract2D)
    assert isinstance(chart, cxc.AbstractChart)


@given(chart=cxst.charts(filter=cxc.Abstract1D))
@settings(max_examples=50)
def test_1d_chart(chart: cxc.AbstractChart) -> None:
    """Test 1D chart instance generation."""
    assert isinstance(chart, cxc.Abstract1D)
    assert isinstance(chart, cxc.AbstractChart)


@given(chart=cxst.charts(filter=(cxc.Abstract3D, cxc.AbstractSpherical3D)))
@settings(max_examples=50)
def test_spherical_3d_chart(chart: cxc.AbstractChart) -> None:
    """Test chart instance generation with multiple types."""
    assert isinstance(chart, cxc.AbstractChart)
    assert isinstance(chart, cxc.Abstract3D)
    assert isinstance(chart, cxc.AbstractSpherical3D)


@given(chart=cxst.charts(filter=st.sampled_from([cxc.Abstract1D, cxc.Abstract2D])))
@settings(max_examples=50)
def test_dynamic_filter(chart: cxc.AbstractChart) -> None:
    """Test chart instance generation with dynamic union class."""
    assert isinstance(chart, cxc.AbstractChart)
    assert isinstance(chart, (cxc.Abstract1D, cxc.Abstract2D))


@given(chart=cxst.charts(ndim=None))
@settings(max_examples=50)
def test_ndim_none_allows_0d(chart: cxc.AbstractChart) -> None:
    """Test that ndim=None allows all dimensionalities."""
    assert isinstance(chart, cxc.AbstractChart)
    # Could be 0D or higher


@given(chart=cxst.charts(ndim=2))
@settings(max_examples=50)
def test_exact_ndim(chart: cxc.AbstractChart) -> None:
    """Test exact ndim match."""
    assert chart.ndim == 2


@given(chart=cxst.charts(ndim=st.integers(min_value=1, max_value=2)))
@settings(max_examples=50)
def test_ndim_strategy(chart: cxc.AbstractChart) -> None:
    """Test ndim as a strategy."""
    assert 1 <= chart.ndim <= 2
