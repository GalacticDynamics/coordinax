"""Tests for the vectors strategy."""

import hypothesis.strategies as st
import jax.numpy as jnp
from hypothesis import given, settings

import coordinax.charts as cxc
import coordinax.objs as cxobj
import coordinax.roles as cxr
import coordinax_hypothesis as cxst

# =============================================================================
# Basic tests


@given(vec=cxst.vectors())
@settings(max_examples=50)
def test_vectors_basic(vec: cxobj.Vector) -> None:
    """Test basic vector instance generation."""
    assert isinstance(vec, cxobj.Vector)
    assert hasattr(vec, "data")
    assert hasattr(vec, "chart")
    assert hasattr(vec, "role")


@given(vec=cxst.vectors(chart=cxc.cart3d))
@settings(max_examples=20)
def test_vectors_specific_chart(vec: cxobj.Vector) -> None:
    """Test vector generation with a specific chart."""
    assert isinstance(vec, cxobj.Vector)
    assert vec.chart == cxc.cart3d


@given(vec=cxst.vectors(shape=(10,)))
@settings(max_examples=20)
def test_vectors_with_shape(vec: cxobj.Vector) -> None:
    """Test vector generation with a specific shape."""
    assert isinstance(vec, cxobj.Vector)
    assert vec.shape == (10,)


@given(vec=cxst.vectors(shape=(5, 3)))
@settings(max_examples=20)
def test_vectors_with_2d_shape(vec: cxobj.Vector) -> None:
    """Test vector generation with a 2D shape."""
    assert isinstance(vec, cxobj.Vector)
    assert vec.shape == (5, 3)


@given(vec=cxst.vectors(dtype=jnp.float32))
@settings(max_examples=20)
def test_vectors_with_dtype(vec: cxobj.Vector) -> None:
    """Test vector generation with a specific dtype."""
    assert isinstance(vec, cxobj.Vector)
    # Check that at least one component has the correct dtype
    dtypes = {v.dtype for v in vec.data.values()}
    assert jnp.float32 in dtypes or any(
        jnp.issubdtype(dt, jnp.float32) for dt in dtypes
    )


# =============================================================================
# Tests with position roles


@given(vec=cxst.vectors(role=cxr.PhysDisp))
@settings(max_examples=30)
def test_vectors_position(vec: cxobj.Vector) -> None:
    """Test vector generation with position roles."""
    assert isinstance(vec, cxobj.Vector)
    assert isinstance(vec.role, cxr.PhysDisp)


@given(vec=cxst.vectors(chart=cxst.charts(filter=cxc.Abstract1D), role=cxr.PhysDisp))
@settings(max_examples=20)
def test_vectors_position_1d(vec: cxobj.Vector) -> None:
    """Test 1D position vector generation."""
    assert isinstance(vec, cxobj.Vector)
    assert isinstance(vec.role, cxr.PhysDisp)
    assert isinstance(vec.chart, cxc.Abstract1D)


@given(vec=cxst.vectors(chart=cxst.charts(filter=cxc.Abstract2D), role=cxr.PhysDisp))
@settings(max_examples=20)
def test_vectors_position_2d(vec: cxobj.Vector) -> None:
    """Test 2D position vector generation."""
    assert isinstance(vec, cxobj.Vector)
    assert isinstance(vec.role, cxr.PhysDisp)
    assert isinstance(vec.chart, cxc.Abstract2D)


@given(vec=cxst.vectors(chart=cxst.charts(filter=cxc.Abstract3D), role=cxr.PhysDisp))
@settings(max_examples=20)
def test_vectors_position_3d(vec: cxobj.Vector) -> None:
    """Test 3D position vector generation."""
    assert isinstance(vec, cxobj.Vector)
    assert isinstance(vec.role, cxr.PhysDisp)
    assert isinstance(vec.chart, cxc.Abstract3D)


# =============================================================================
# Tests with velocity roles


@given(vec=cxst.vectors(role=cxr.PhysVel))
@settings(max_examples=30)
def test_vectors_velocity(vec: cxobj.Vector) -> None:
    """Test vector generation with velocity roles."""
    assert isinstance(vec, cxobj.Vector)
    assert isinstance(vec.role, cxr.PhysVel)


@given(vec=cxst.vectors(chart=cxst.charts(filter=cxc.Abstract1D), role=cxr.PhysVel))
@settings(max_examples=20)
def test_vectors_velocity_1d(vec: cxobj.Vector) -> None:
    """Test 1D velocity vector generation."""
    assert isinstance(vec, cxobj.Vector)
    assert isinstance(vec.role, cxr.PhysVel)
    assert isinstance(vec.chart, cxc.Abstract1D)


@given(vec=cxst.vectors(chart=cxst.charts(filter=cxc.Abstract3D), role=cxr.PhysVel))
@settings(max_examples=20)
def test_vectors_velocity_3d(vec: cxobj.Vector) -> None:
    """Test 3D velocity vector generation."""
    assert isinstance(vec, cxobj.Vector)
    assert isinstance(vec.role, cxr.PhysVel)
    assert isinstance(vec.chart, cxc.Abstract3D)


# =============================================================================
# Tests with acceleration roles


@given(vec=cxst.vectors(role=cxr.PhysAcc))
@settings(max_examples=30)
def test_vectors_acceleration(vec: cxobj.Vector) -> None:
    """Test vector generation with acceleration roles."""
    assert isinstance(vec, cxobj.Vector)
    assert isinstance(vec.role, cxr.PhysAcc)


@given(vec=cxst.vectors(chart=cxst.charts(filter=cxc.Abstract1D), role=cxr.PhysAcc))
@settings(max_examples=20)
def test_vectors_acceleration_1d(vec: cxobj.Vector) -> None:
    """Test 1D acceleration vector generation."""
    assert isinstance(vec, cxobj.Vector)
    assert isinstance(vec.role, cxr.PhysAcc)
    assert isinstance(vec.chart, cxc.Abstract1D)


@given(vec=cxst.vectors(chart=cxst.charts(filter=cxc.Abstract3D), role=cxr.PhysAcc))
@settings(max_examples=20)
def test_vectors_acceleration_3d(vec: cxobj.Vector) -> None:
    """Test 3D acceleration vector generation."""
    assert isinstance(vec, cxobj.Vector)
    assert isinstance(vec.role, cxr.PhysAcc)
    assert isinstance(vec.chart, cxc.Abstract3D)


# =============================================================================
# Test data dictionary structure


@given(vec=cxst.vectors())
@settings(max_examples=20)
def test_vectors_data_structure(vec: cxobj.Vector) -> None:
    """Test that generated vectors have the correct data structure."""
    # The data keys should match the chart components
    assert set(vec.data.keys()) == set(vec.chart.components)

    # All data values should have the same shape (broadcastable)
    shapes = [v.shape for v in vec.data.values()]
    if shapes:
        # Check that all shapes are broadcastable to vec.shape
        for shape in shapes:
            assert all(
                s1 in (s2, 1) or s2 == 1
                for s1, s2 in zip(reversed(shape), reversed(vec.shape), strict=False)
            )


# =============================================================================
# Test dynamic chart selection


@given(vec=cxst.vectors(chart=st.sampled_from([cxc.cart3d, cxc.sph3d, cxc.cyl3d])))
@settings(max_examples=30)
def test_vectors_dynamic_chart(vec: cxobj.Vector) -> None:
    """Test vector generation with dynamically selected chart."""
    assert isinstance(vec, cxobj.Vector)
    assert vec.chart in (
        cxc.cart3d,
        cxc.sph3d,
        cxc.cyl3d,
    )
