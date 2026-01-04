"""Tests for the distances strategy."""

import hypothesis.strategies as st
from hypothesis import given, settings

import coordinax.distance as cxd
from coordinax_hypothesis import distances


@given(dist=distances())
@settings(max_examples=50)
def test_basic_distance(dist: cxd.Distance) -> None:
    """Test basic distance generation."""
    assert isinstance(dist, cxd.Distance)
    assert dist.shape == ()
    assert dist.value >= 0  # default check_negative=True


@given(dist=distances(check_negative=False))
@settings(max_examples=50)
def test_distance_allow_negative(dist: cxd.Distance) -> None:
    """Test distance generation with negative values allowed."""
    assert isinstance(dist, cxd.Distance)
    # Don't check sign when check_negative=False


@given(dist=distances(unit="kpc"))
@settings(max_examples=50)
def test_distance_with_units(dist: cxd.Distance) -> None:
    """Test distance generation with specific units."""
    assert isinstance(dist, cxd.Distance)
    assert dist.unit == "kpc"


@given(dist=distances(shape=5))
@settings(max_examples=30)
def test_distance_vector(dist: cxd.Distance) -> None:
    """Test vector distance generation."""
    assert isinstance(dist, cxd.Distance)
    assert dist.shape == (5,)
    assert all(dist.value >= 0)  # all elements should be non-negative


@given(dist=distances(shape=(2, 3)))
@settings(max_examples=30)
def test_distance_2d(dist: cxd.Distance) -> None:
    """Test 2D distance array generation."""
    assert isinstance(dist, cxd.Distance)
    assert dist.shape == (2, 3)


@given(dist=distances(check_negative=st.sampled_from([True, False])))
@settings(max_examples=50)
def test_distance_with_strategy_check_negative(dist: cxd.Distance) -> None:
    """Test distance with check_negative as a strategy."""
    assert isinstance(dist, cxd.Distance)
    # check_negative varies, so we can't assert about the sign


@given(dist=distances(elements=st.floats(min_value=1.0, max_value=100.0, width=32)))
@settings(max_examples=30)
def test_distance_with_custom_elements(dist: cxd.Distance) -> None:
    """Test distance with custom elements range."""
    assert isinstance(dist, cxd.Distance)
    assert 1.0 <= dist.value <= 100.0


@given(
    dist=distances(
        check_negative=True, elements=st.floats(min_value=0.0, max_value=10.0, width=32)
    )
)
@settings(max_examples=30)
def test_distance_check_negative_with_elements(dist: cxd.Distance) -> None:
    """Test that check_negative works with custom elements."""
    assert isinstance(dist, cxd.Distance)
    # When check_negative=True and elements provided, min_value should be adjusted
    assert dist.value >= 0
    assert dist.value <= 50.0
