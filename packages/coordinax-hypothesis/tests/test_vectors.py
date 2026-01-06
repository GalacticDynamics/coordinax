"""Tests for the vectors strategy."""

import hypothesis.strategies as st
import jax.numpy as jnp
from coordinax_hypothesis import representations, vectors
from hypothesis import given, settings

import coordinax as cx

# =============================================================================
# Basic tests


@given(vec=vectors())
@settings(max_examples=50)
def test_vectors_basic(vec: cx.Vector) -> None:
    """Test basic vector instance generation."""
    assert isinstance(vec, cx.Vector)
    assert hasattr(vec, "data")
    assert hasattr(vec, "rep")
    assert hasattr(vec, "role")


@given(vec=vectors(rep=cx.r.cart3d))
@settings(max_examples=20)
def test_vectors_specific_representation(vec: cx.Vector) -> None:
    """Test vector generation with a specific representation."""
    assert isinstance(vec, cx.Vector)
    assert vec.rep == cx.r.cart3d


@given(vec=vectors(shape=(10,)))
@settings(max_examples=20)
def test_vectors_with_shape(vec: cx.Vector) -> None:
    """Test vector generation with a specific shape."""
    assert isinstance(vec, cx.Vector)
    assert vec.shape == (10,)


@given(vec=vectors(shape=(5, 3)))
@settings(max_examples=20)
def test_vectors_with_2d_shape(vec: cx.Vector) -> None:
    """Test vector generation with a 2D shape."""
    assert isinstance(vec, cx.Vector)
    assert vec.shape == (5, 3)


@given(vec=vectors(dtype=jnp.float32))
@settings(max_examples=20)
def test_vectors_with_dtype(vec: cx.Vector) -> None:
    """Test vector generation with a specific dtype."""
    assert isinstance(vec, cx.Vector)
    # Check that at least one component has the correct dtype
    dtypes = {v.dtype for v in vec.data.values()}
    assert jnp.float32 in dtypes or any(
        jnp.issubdtype(dt, jnp.float32) for dt in dtypes
    )


# =============================================================================
# Tests with position roles


@given(vec=vectors(role=cx.r.Pos))
@settings(max_examples=30)
def test_vectors_position(vec: cx.Vector) -> None:
    """Test vector generation with position roles."""
    assert isinstance(vec, cx.Vector)
    assert isinstance(vec.role, cx.r.Pos)


@given(vec=vectors(rep=representations(filter=cx.r.Abstract1D), role=cx.r.Pos))
@settings(max_examples=20)
def test_vectors_position_1d(vec: cx.Vector) -> None:
    """Test 1D position vector generation."""
    assert isinstance(vec, cx.Vector)
    assert isinstance(vec.role, cx.r.Pos)
    assert isinstance(vec.rep, cx.r.Abstract1D)


@given(vec=vectors(rep=representations(filter=cx.r.Abstract2D), role=cx.r.Pos))
@settings(max_examples=20)
def test_vectors_position_2d(vec: cx.Vector) -> None:
    """Test 2D position vector generation."""
    assert isinstance(vec, cx.Vector)
    assert isinstance(vec.role, cx.r.Pos)
    assert isinstance(vec.rep, cx.r.Abstract2D)


@given(vec=vectors(rep=representations(filter=cx.r.Abstract3D), role=cx.r.Pos))
@settings(max_examples=20)
def test_vectors_position_3d(vec: cx.Vector) -> None:
    """Test 3D position vector generation."""
    assert isinstance(vec, cx.Vector)
    assert isinstance(vec.role, cx.r.Pos)
    assert isinstance(vec.rep, cx.r.Abstract3D)


# =============================================================================
# Tests with velocity roles


@given(vec=vectors(role=cx.r.Vel))
@settings(max_examples=30)
def test_vectors_velocity(vec: cx.Vector) -> None:
    """Test vector generation with velocity roles."""
    assert isinstance(vec, cx.Vector)
    assert isinstance(vec.role, cx.r.Vel)


@given(vec=vectors(rep=representations(filter=cx.r.Abstract1D), role=cx.r.Vel))
@settings(max_examples=20)
def test_vectors_velocity_1d(vec: cx.Vector) -> None:
    """Test 1D velocity vector generation."""
    assert isinstance(vec, cx.Vector)
    assert isinstance(vec.role, cx.r.Vel)
    assert isinstance(vec.rep, cx.r.Abstract1D)


@given(vec=vectors(rep=representations(filter=cx.r.Abstract3D), role=cx.r.Vel))
@settings(max_examples=20)
def test_vectors_velocity_3d(vec: cx.Vector) -> None:
    """Test 3D velocity vector generation."""
    assert isinstance(vec, cx.Vector)
    assert isinstance(vec.role, cx.r.Vel)
    assert isinstance(vec.rep, cx.r.Abstract3D)


# =============================================================================
# Tests with acceleration roles


@given(vec=vectors(role=cx.r.Acc))
@settings(max_examples=30)
def test_vectors_acceleration(vec: cx.Vector) -> None:
    """Test vector generation with acceleration roles."""
    assert isinstance(vec, cx.Vector)
    assert isinstance(vec.role, cx.r.Acc)


@given(vec=vectors(rep=representations(filter=cx.r.Abstract1D), role=cx.r.Acc))
@settings(max_examples=20)
def test_vectors_acceleration_1d(vec: cx.Vector) -> None:
    """Test 1D acceleration vector generation."""
    assert isinstance(vec, cx.Vector)
    assert isinstance(vec.role, cx.r.Acc)
    assert isinstance(vec.rep, cx.r.Abstract1D)


@given(vec=vectors(rep=representations(filter=cx.r.Abstract3D), role=cx.r.Acc))
@settings(max_examples=20)
def test_vectors_acceleration_3d(vec: cx.Vector) -> None:
    """Test 3D acceleration vector generation."""
    assert isinstance(vec, cx.Vector)
    assert isinstance(vec.role, cx.r.Acc)
    assert isinstance(vec.rep, cx.r.Abstract3D)


# =============================================================================
# Test data dictionary structure


@given(vec=vectors())
@settings(max_examples=20)
def test_vectors_data_structure(vec: cx.Vector) -> None:
    """Test that generated vectors have the correct data structure."""
    # The data keys should match the representation components
    assert set(vec.data.keys()) == set(vec.rep.components)

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
# Test dynamic representation selection


@given(vec=vectors(rep=st.sampled_from([cx.r.cart3d, cx.r.sph3d, cx.r.cyl3d])))
@settings(max_examples=30)
def test_vectors_dynamic_representation(vec: cx.Vector) -> None:
    """Test vector generation with dynamically selected representation."""
    assert isinstance(vec, cx.Vector)
    assert vec.rep in (cx.r.cart3d, cx.r.sph3d, cx.r.cyl3d)
