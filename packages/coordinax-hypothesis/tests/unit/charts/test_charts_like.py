"""Tests for the charts_like strategy."""

from hypothesis import given, settings

import coordinax.charts as cxc
import coordinax_hypothesis.core as cxst


@given(chart=cxst.charts_like(cxc.cart3d))
@settings(max_examples=50)
def test_charts_like_cart3d(chart: cxc.AbstractChart) -> None:
    """Test charts_like with Cart3D template."""
    assert isinstance(chart, cxc.Abstract3D)
    assert chart.ndim == 3


@given(chart=cxst.charts_like(cxc.sph3d))
@settings(max_examples=50)
def test_charts_like_spherical3d(chart: cxc.AbstractChart) -> None:
    """Test charts_like with Spherical3D template."""
    assert isinstance(chart, cxc.Abstract3D)
    assert isinstance(chart, cxc.AbstractSpherical3D)
    assert chart.ndim == 3


@given(chart=cxst.charts_like(cxc.polar2d))
@settings(max_examples=50)
def test_charts_like_polar2d(chart: cxc.AbstractChart) -> None:
    """Test charts_like with Polar2D template."""
    assert isinstance(chart, cxc.Abstract2D)
    assert chart.ndim == 2


@given(chart=cxst.charts_like(cxc.radial1d))
@settings(max_examples=50)
def test_charts_like_radial1d(chart: cxc.AbstractChart) -> None:
    """Test charts_like with Radial1D template."""
    assert isinstance(chart, cxc.Abstract1D)
    assert chart.ndim == 1


@given(chart=cxst.charts_like(cxc.twosphere))
@settings(max_examples=50)
def test_charts_like_twosphere(chart: cxc.AbstractChart) -> None:
    """Test charts_like with TwoSphere template."""
    assert isinstance(chart, cxc.Abstract2D)
    assert chart.ndim == 2
