"""Tests for the chart_time_chain strategy."""

from hypothesis import given, settings

import coordinax as cx
import coordinax_hypothesis as cxst


@given(chain=cxst.chart_time_chain(cx.roles.Pos, cx.charts.cart3d))
@settings(max_examples=30)
def test_position_chain_is_singleton(chain: tuple) -> None:
    """Test that position representations return single-element chains."""
    assert len(chain) == 1
    (rep,) = chain
    assert isinstance(rep, cx.charts.Abstract3D)


@given(chain=cxst.chart_time_chain(cx.roles.Pos, cx.charts.sph3d))
@settings(max_examples=30)
def test_spherical_position_chain(chain: tuple) -> None:
    """Test spherical position chain structure."""
    assert len(chain) == 1
    (rep,) = chain
    assert isinstance(rep, cx.charts.Abstract3D)


@given(chain=cxst.chart_time_chain(cx.roles.Pos, cx.charts.radial1d))
@settings(max_examples=30)
def test_radial_position_chain(chain: tuple) -> None:
    """Test 1D position chain structure."""
    assert len(chain) == 1
    (rep,) = chain
    assert isinstance(rep, cx.charts.Abstract1D)


@given(chain=cxst.chart_time_chain(cx.roles.Vel, cx.charts.cart3d))
@settings(max_examples=30)
def test_velocity_chain_has_two_elements(chain: tuple) -> None:
    """Test that velocity roles return two-element chains."""
    assert len(chain) == 2
    vel_rep, pos_rep = chain
    assert isinstance(vel_rep, cx.charts.Abstract3D)
    assert isinstance(pos_rep, cx.charts.Abstract3D)


@given(chain=cxst.chart_time_chain(cx.roles.Vel, cx.charts.polar2d))
@settings(max_examples=30)
def test_polar_velocity_chain(chain: tuple) -> None:
    """Test 2D velocity chain structure."""
    assert len(chain) == 2
    vel_rep, pos_rep = chain
    assert isinstance(vel_rep, cx.charts.Abstract2D)
    assert isinstance(pos_rep, cx.charts.Abstract2D)


@given(chain=cxst.chart_time_chain(cx.roles.Vel, cx.charts.radial1d))
@settings(max_examples=30)
def test_radial_velocity_chain(chain: tuple) -> None:
    """Test 1D velocity chain structure."""
    assert len(chain) == 2
    vel_rep, pos_rep = chain
    assert isinstance(vel_rep, cx.charts.Abstract1D)
    assert isinstance(pos_rep, cx.charts.Abstract1D)


@given(chain=cxst.chart_time_chain(cx.roles.Acc, cx.charts.cart3d))
@settings(max_examples=30)
def test_acceleration_chain_has_three_elements(chain: tuple) -> None:
    """Test that acceleration roles return three-element chains."""
    assert len(chain) == 3
    acc_rep, vel_rep, pos_rep = chain
    assert isinstance(acc_rep, cx.charts.Abstract3D)
    assert isinstance(vel_rep, cx.charts.Abstract3D)
    assert isinstance(pos_rep, cx.charts.Abstract3D)


@given(chain=cxst.chart_time_chain(cx.roles.Acc, cx.charts.sph3d))
@settings(max_examples=30)
def test_spherical_acceleration_chain(chain: tuple) -> None:
    """Test spherical acceleration chain structure."""
    assert len(chain) == 3
    acc_rep, vel_rep, pos_rep = chain
    assert isinstance(acc_rep, cx.charts.Abstract3D)
    assert isinstance(vel_rep, cx.charts.Abstract3D)
    assert isinstance(pos_rep, cx.charts.Abstract3D)


@given(chain=cxst.chart_time_chain(cx.roles.Acc, cx.charts.radial1d))
@settings(max_examples=30)
def test_radial_acceleration_chain(chain: tuple) -> None:
    """Test 1D acceleration chain structure."""
    assert len(chain) == 3
    acc_rep, vel_rep, pos_rep = chain
    assert isinstance(acc_rep, cx.charts.Abstract1D)
    assert isinstance(vel_rep, cx.charts.Abstract1D)
    assert isinstance(pos_rep, cx.charts.Abstract1D)


@given(
    chain=cxst.chart_time_chain(cx.roles.Pos, cxst.charts(filter=cx.charts.Abstract3D))
)
@settings(max_examples=30)
def test_position_chain_from_strategy(chain: tuple) -> None:
    """Test position chains generated from a representation strategy."""
    assert len(chain) == 1
    (rep,) = chain
    assert isinstance(rep, cx.charts.Abstract3D)


@given(
    chain=cxst.chart_time_chain(cx.roles.Vel, cxst.charts(filter=cx.charts.Abstract3D))
)
@settings(max_examples=30)
def test_velocity_chain_from_strategy(chain: tuple) -> None:
    """Test velocity chains generated from a representation strategy."""
    assert len(chain) == 2
    vel_rep, pos_rep = chain
    assert isinstance(vel_rep, cx.charts.Abstract3D)
    assert isinstance(pos_rep, cx.charts.Abstract3D)


@given(
    chain=cxst.chart_time_chain(cx.roles.Acc, cxst.charts(filter=cx.charts.Abstract3D))
)
@settings(max_examples=30)
def test_acceleration_chain_from_strategy(chain: tuple) -> None:
    """Test acceleration chains generated from a representation strategy."""
    assert len(chain) == 3
    acc_rep, vel_rep, pos_rep = chain
    assert isinstance(acc_rep, cx.charts.Abstract3D)
    assert isinstance(vel_rep, cx.charts.Abstract3D)
    assert isinstance(pos_rep, cx.charts.Abstract3D)


@given(
    chain=cxst.chart_time_chain(cx.roles.Acc, cxst.charts(filter=cx.charts.Abstract2D))
)
@settings(max_examples=30)
def test_2d_acceleration_chain_preserves_dimensionality(chain: tuple) -> None:
    """Test 2D acceleration chain preserves dimensionality."""
    assert len(chain) == 3
    acc_rep, vel_rep, pos_rep = chain
    assert isinstance(acc_rep, cx.charts.Abstract2D)
    assert isinstance(vel_rep, cx.charts.Abstract2D)
    assert isinstance(pos_rep, cx.charts.Abstract2D)
    assert acc_rep.ndim == 2
    assert vel_rep.ndim == 2
    assert pos_rep.ndim == 2


@given(
    chain=cxst.chart_time_chain(cx.roles.Vel, cxst.charts(filter=cx.charts.Abstract3D))
)
@settings(max_examples=30)
def test_3d_velocity_chain_preserves_dimensionality(chain: tuple) -> None:
    """Test that 3D velocity chains preserve dimensionality flags."""
    assert len(chain) == 2
    vel_rep, pos_rep = chain
    assert isinstance(vel_rep, cx.charts.Abstract3D)
    assert isinstance(pos_rep, cx.charts.Abstract3D)
    assert vel_rep.ndim == 3
    assert pos_rep.ndim == 3
