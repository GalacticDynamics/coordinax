"""Tests for the chart_time_chain strategy."""

from hypothesis import given, settings

import coordinax.charts as cxc
import coordinax.roles as cxr
import coordinax_hypothesis as cxst


@given(chain=cxst.chart_time_chain(cxr.Point, cxc.cart3d))
@settings(max_examples=30)
def test_position_chain_is_singleton(chain: tuple) -> None:
    """Test that position representations return single-element chains."""
    assert len(chain) == 1
    (rep,) = chain
    assert isinstance(rep, cxc.Abstract3D)


@given(chain=cxst.chart_time_chain(cxr.Point, cxc.sph3d))
@settings(max_examples=30)
def test_spherical_position_chain(chain: tuple) -> None:
    """Test spherical position chain structure."""
    assert len(chain) == 1
    (rep,) = chain
    assert isinstance(rep, cxc.Abstract3D)


@given(chain=cxst.chart_time_chain(cxr.Point, cxc.radial1d))
@settings(max_examples=30)
def test_radial_position_chain(chain: tuple) -> None:
    """Test 1D position chain structure."""
    assert len(chain) == 1
    (rep,) = chain
    assert isinstance(rep, cxc.Abstract1D)


@given(chain=cxst.chart_time_chain(cxr.PhysVel, cxc.cart3d))
@settings(max_examples=30)
def test_velocity_chain_has_two_elements(chain: tuple) -> None:
    """Test that velocity roles return two-element chains."""
    assert len(chain) == 2
    vel_chart, point_chart = chain
    assert isinstance(vel_chart, cxc.Abstract3D)
    assert isinstance(point_chart, cxc.Abstract3D)


@given(chain=cxst.chart_time_chain(cxr.PhysVel, cxc.polar2d))
@settings(max_examples=30)
def test_polar_velocity_chain(chain: tuple) -> None:
    """Test 2D velocity chain structure."""
    assert len(chain) == 2
    vel_chart, point_chart = chain
    assert isinstance(vel_chart, cxc.Abstract2D)
    assert isinstance(point_chart, cxc.Abstract2D)


@given(chain=cxst.chart_time_chain(cxr.PhysVel, cxc.radial1d))
@settings(max_examples=30)
def test_radial_velocity_chain(chain: tuple) -> None:
    """Test 1D velocity chain structure."""
    assert len(chain) == 2
    vel_chart, point_chart = chain
    assert isinstance(vel_chart, cxc.Abstract1D)
    assert isinstance(point_chart, cxc.Abstract1D)


@given(chain=cxst.chart_time_chain(cxr.PhysAcc, cxc.cart3d))
@settings(max_examples=30)
def test_acceleration_chain_has_three_elements(chain: tuple) -> None:
    """Test that acceleration roles return three-element chains."""
    assert len(chain) == 3
    acc_chart, vel_chart, point_chart = chain
    assert isinstance(acc_chart, cxc.Abstract3D)
    assert isinstance(vel_chart, cxc.Abstract3D)
    assert isinstance(point_chart, cxc.Abstract3D)


@given(chain=cxst.chart_time_chain(cxr.PhysAcc, cxc.sph3d))
@settings(max_examples=30)
def test_spherical_acceleration_chain(chain: tuple) -> None:
    """Test spherical acceleration chain structure."""
    assert len(chain) == 3
    acc_chart, vel_chart, point_chart = chain
    assert isinstance(acc_chart, cxc.Abstract3D)
    assert isinstance(vel_chart, cxc.Abstract3D)
    assert isinstance(point_chart, cxc.Abstract3D)


@given(chain=cxst.chart_time_chain(cxr.PhysAcc, cxc.radial1d))
@settings(max_examples=30)
def test_radial_acceleration_chain(chain: tuple) -> None:
    """Test 1D acceleration chain structure."""
    assert len(chain) == 3
    acc_chart, vel_chart, point_chart = chain
    assert isinstance(acc_chart, cxc.Abstract1D)
    assert isinstance(vel_chart, cxc.Abstract1D)
    assert isinstance(point_chart, cxc.Abstract1D)


@given(chain=cxst.chart_time_chain(cxr.PhysDisp, cxst.charts(filter=cxc.Abstract3D)))
@settings(max_examples=30)
def test_position_chain_from_strategy(chain: tuple) -> None:
    """Test position chains generated from a representation strategy."""
    assert len(chain) == 1
    (rep,) = chain
    assert isinstance(rep, cxc.Abstract3D)


@given(chain=cxst.chart_time_chain(cxr.PhysVel, cxst.charts(filter=cxc.Abstract3D)))
@settings(max_examples=30)
def test_velocity_chain_from_strategy(chain: tuple) -> None:
    """Test velocity chains generated from a representation strategy."""
    assert len(chain) == 2
    vel_chart, point_chart = chain
    assert isinstance(vel_chart, cxc.Abstract3D)
    assert isinstance(point_chart, cxc.Abstract3D)


@given(chain=cxst.chart_time_chain(cxr.PhysAcc, cxst.charts(filter=cxc.Abstract3D)))
@settings(max_examples=30)
def test_acceleration_chain_from_strategy(chain: tuple) -> None:
    """Test acceleration chains generated from a representation strategy."""
    assert len(chain) == 3
    acc_chart, vel_chart, point_chart = chain
    assert isinstance(acc_chart, cxc.Abstract3D)
    assert isinstance(vel_chart, cxc.Abstract3D)
    assert isinstance(point_chart, cxc.Abstract3D)


@given(chain=cxst.chart_time_chain(cxr.PhysAcc, cxst.charts(filter=cxc.Abstract2D)))
@settings(max_examples=30)
def test_2d_acceleration_chain_preserves_dimensionality(chain: tuple) -> None:
    """Test 2D acceleration chain preserves dimensionality."""
    assert len(chain) == 3
    acc_chart, vel_chart, point_chart = chain
    assert isinstance(acc_chart, cxc.Abstract2D)
    assert isinstance(vel_chart, cxc.Abstract2D)
    assert isinstance(point_chart, cxc.Abstract2D)
    assert acc_chart.ndim == 2
    assert vel_chart.ndim == 2
    assert point_chart.ndim == 2


@given(chain=cxst.chart_time_chain(cxr.PhysVel, cxst.charts(filter=cxc.Abstract3D)))
@settings(max_examples=30)
def test_3d_velocity_chain_preserves_dimensionality(chain: tuple) -> None:
    """Test that 3D velocity chains preserve dimensionality flags."""
    assert len(chain) == 2
    vel_chart, point_chart = chain
    assert isinstance(vel_chart, cxc.Abstract3D)
    assert isinstance(point_chart, cxc.Abstract3D)
    assert vel_chart.ndim == 3
    assert point_chart.ndim == 3
