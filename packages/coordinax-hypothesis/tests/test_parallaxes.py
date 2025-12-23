"""Tests for the parallaxes strategy."""

import hypothesis.strategies as st
from coordinax_hypothesis import parallaxes
from hypothesis import given, settings

import coordinax.distance as cxd


@given(plx=parallaxes())
@settings(max_examples=50)
def test_basic_parallax(plx: cxd.Parallax) -> None:
    """Test basic parallax generation."""
    assert isinstance(plx, cxd.Parallax)
    assert plx.shape == ()
    assert plx.value >= 0  # default check_negative=True


@given(plx=parallaxes(check_negative=False))
@settings(max_examples=50)
def test_parallax_allow_negative(plx: cxd.Parallax) -> None:
    """Test parallax generation with negative values allowed."""
    assert isinstance(plx, cxd.Parallax)
    # Don't check sign when check_negative=False


@given(plx=parallaxes(unit="mas"))
@settings(max_examples=50)
def test_parallax_with_units(plx: cxd.Parallax) -> None:
    """Test parallax generation with specific units."""
    assert isinstance(plx, cxd.Parallax)
    assert plx.unit == "mas"


@given(plx=parallaxes(unit="arcsec"))
@settings(max_examples=50)
def test_parallax_arcsec(plx: cxd.Parallax) -> None:
    """Test parallax generation in arcseconds."""
    assert isinstance(plx, cxd.Parallax)
    assert plx.unit == "arcsec"


@given(plx=parallaxes(shape=5))
@settings(max_examples=30)
def test_parallax_vector(plx: cxd.Parallax) -> None:
    """Test vector parallax generation."""
    assert isinstance(plx, cxd.Parallax)
    assert plx.shape == (5,)
    assert all(plx.value >= 0)  # all elements should be non-negative


@given(plx=parallaxes(shape=(2, 3)))
@settings(max_examples=30)
def test_parallax_2d(plx: cxd.Parallax) -> None:
    """Test 2D parallax array generation."""
    assert isinstance(plx, cxd.Parallax)
    assert plx.shape == (2, 3)


@given(plx=parallaxes(check_negative=st.sampled_from([True, False])))
@settings(max_examples=50)
def test_parallax_with_strategy_check_negative(plx: cxd.Parallax) -> None:
    """Test parallax with check_negative as a strategy."""
    assert isinstance(plx, cxd.Parallax)
    # check_negative varies, so we can't assert about the sign


@given(plx=parallaxes(elements=st.floats(min_value=1.0, max_value=100.0, width=32)))
@settings(max_examples=30)
def test_parallax_with_custom_elements(plx: cxd.Parallax) -> None:
    """Test parallax with custom elements range."""
    assert isinstance(plx, cxd.Parallax)
    assert 1.0 <= plx.value <= 100.0


@given(
    plx=parallaxes(
        check_negative=True, elements=st.floats(min_value=0.0, max_value=10.0, width=32)
    )
)
@settings(max_examples=30)
def test_parallax_check_negative_with_elements(plx: cxd.Parallax) -> None:
    """Test that check_negative works with custom elements."""
    assert isinstance(plx, cxd.Parallax)
    # When check_negative=True and elements provided, min_value should be adjusted
    assert plx.value >= 0
    assert plx.value <= 10.0
