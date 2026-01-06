"""Tests for the representations strategy (instance generation)."""

import hypothesis.strategies as st
from coordinax_hypothesis import representations
from hypothesis import given, settings

import coordinax as cx


@given(rep=representations())
@settings(max_examples=50)
def test_basic_representation(rep: cx.r.AbstractRep) -> None:
    """Test basic representation instance generation."""
    assert isinstance(rep, cx.r.AbstractRep)


@given(rep=representations(filter=cx.r.Abstract3D))
@settings(max_examples=50)
def test_3d_representations(rep: cx.r.AbstractRep) -> None:
    """Test 3D representation instance generation."""
    assert isinstance(rep, cx.r.Abstract3D)
    assert isinstance(rep, cx.r.AbstractRep)


@given(rep=representations(filter=cx.r.Abstract2D))
@settings(max_examples=50)
def test_2d_representations(rep: cx.r.AbstractRep) -> None:
    """Test 2D representation instance generation."""
    assert isinstance(rep, cx.r.Abstract2D)
    assert isinstance(rep, cx.r.AbstractRep)


@given(rep=representations(filter=cx.r.Abstract1D))
@settings(max_examples=50)
def test_1d_representations(rep: cx.r.AbstractRep) -> None:
    """Test 1D representation instance generation."""
    assert isinstance(rep, cx.r.Abstract1D)
    assert isinstance(rep, cx.r.AbstractRep)


@given(rep=representations(filter=(cx.r.Abstract3D, cx.r.AbstractSpherical3D)))
@settings(max_examples=50)
def test_spherical_3d_representations(rep: cx.r.AbstractRep) -> None:
    """Test representation instance generation with multiple types."""
    assert isinstance(rep, cx.r.AbstractRep)
    assert isinstance(rep, cx.r.Abstract3D)
    assert isinstance(rep, cx.r.AbstractSpherical3D)


@given(rep=representations(filter=st.sampled_from([cx.r.Abstract1D, cx.r.Abstract2D])))
@settings(max_examples=50)
def test_dynamic_filter(rep: cx.r.AbstractRep) -> None:
    """Test representation instance generation with dynamic union class."""
    assert isinstance(rep, cx.r.AbstractRep)
    assert isinstance(rep, (cx.r.Abstract1D, cx.r.Abstract2D))


@given(rep=representations(dimensionality=None))
@settings(max_examples=50)
def test_dimensionality_none_allows_0d(rep: cx.r.AbstractRep) -> None:
    """Test that dimensionality=None allows all dimensionalities."""
    assert isinstance(rep, cx.r.AbstractRep)
    # Could be 0D or higher


@given(rep=representations(dimensionality=2))
@settings(max_examples=50)
def test_exact_dimensionality(rep: cx.r.AbstractRep) -> None:
    """Test exact dimensionality match."""
    assert rep.dimensionality == 2


@given(rep=representations(dimensionality=st.integers(min_value=1, max_value=2)))
@settings(max_examples=50)
def test_dimensionality_strategy(rep: cx.r.AbstractRep) -> None:
    """Test dimensionality as a strategy."""
    assert 1 <= rep.dimensionality <= 2


@given(rep=representations(filter=cx.r.EmbeddedManifold))
@settings(max_examples=20)
def test_embedded_manifold_representation(rep: cx.r.AbstractRep) -> None:
    """Test EmbeddedManifold representation instance generation."""
    assert isinstance(rep, cx.r.EmbeddedManifold)
    assert isinstance(rep.chart_kind, cx.r.TwoSphere)
    assert isinstance(rep.ambient_kind, cx.r.Cart3D)
    assert "R" in rep.params
