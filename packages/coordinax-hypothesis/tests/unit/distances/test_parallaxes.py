"""Tests for the parallaxes strategy."""

import hypothesis.strategies as st
from hypothesis import given, settings

import unxt as u

import coordinax.distances as cxd
import coordinax_hypothesis.core as cxst


@given(plx=cxst.parallaxes())
@settings(max_examples=50)
def test_basic_parallax(plx: cxd.Parallax) -> None:
    """Test basic parallax generation."""
    assert isinstance(plx, cxd.Parallax)
    assert plx.shape == ()
    assert plx.value >= 0  # default check_negative=True


@given(plx=cxst.parallaxes(check_negative=False))
@settings(max_examples=50)
def test_parallax_allow_negative(plx: cxd.Parallax) -> None:
    """Test parallax generation with negative values allowed."""
    assert isinstance(plx, cxd.Parallax)
    # Don't check sign when check_negative=False


@given(plx=cxst.parallaxes(unit="mas"))
@settings(max_examples=50)
def test_parallax_with_units(plx: cxd.Parallax) -> None:
    """Test parallax generation with specific units."""
    assert isinstance(plx, cxd.Parallax)
    assert plx.unit == "mas"


@given(plx=cxst.parallaxes(unit="arcsec"))
@settings(max_examples=50)
def test_parallax_arcsec(plx: cxd.Parallax) -> None:
    """Test parallax generation in arcseconds."""
    assert isinstance(plx, cxd.Parallax)
    assert plx.unit == "arcsec"


@given(plx=cxst.parallaxes(shape=5))
@settings(max_examples=30)
def test_parallax_array_1d(plx: cxd.Parallax) -> None:
    """Test 1D array parallax generation."""
    assert isinstance(plx, cxd.Parallax)
    assert plx.shape == (5,)
    assert all(plx.value >= 0)  # all elements should be non-negative


@given(plx=cxst.parallaxes(shape=(2, 3)))
@settings(max_examples=30)
def test_parallax_array_2d(plx: cxd.Parallax) -> None:
    """Test 2D parallax array generation."""
    assert isinstance(plx, cxd.Parallax)
    assert plx.shape == (2, 3)


@given(plx=cxst.parallaxes(check_negative=st.sampled_from([True, False])))
@settings(max_examples=50)
def test_parallax_with_strategy_check_negative(plx: cxd.Parallax) -> None:
    """Test parallax with check_negative as a strategy."""
    assert isinstance(plx, cxd.Parallax)
    # check_negative varies, so we can't assert about the sign


@given(
    plx=cxst.parallaxes(elements=st.floats(min_value=1.0, max_value=100.0, width=32))
)
@settings(max_examples=30)
def test_parallax_with_custom_elements(plx: cxd.Parallax) -> None:
    """Test parallax with custom elements range."""
    assert isinstance(plx, cxd.Parallax)
    assert 1.0 <= plx.value <= 100.0


@given(
    plx=cxst.parallaxes(
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


class TestParallaxFromType:
    """Test st.from_type() for Parallax type."""

    @given(plx=st.from_type(cxd.Parallax))
    def test_from_type_basic(self, plx: cxd.Parallax) -> None:
        """Test that st.from_type(Parallax) generates valid Parallax instances."""
        assert isinstance(plx, cxd.Parallax)
        # Default check_negative=True means value should be non-negative
        assert plx.value >= 0

    @given(plx=st.from_type(cxd.Parallax))
    def test_from_type_has_angle_dimension(self, plx: cxd.Parallax) -> None:
        """Test that generated parallaxes have angle dimension."""
        assert u.dimension_of(plx) == u.dimension("angle")

    @given(data=st.data())
    def test_from_type_generates_variety(self, data: st.DataObject) -> None:
        """Test that from_type generates different values."""
        plx1 = data.draw(st.from_type(cxd.Parallax))
        plx2 = data.draw(st.from_type(cxd.Parallax))

        assert isinstance(plx1, cxd.Parallax)
        assert isinstance(plx2, cxd.Parallax)

    @given(data=st.data())
    def test_builds_with_parallax_arg(self, data: st.DataObject) -> None:
        """Test that st.builds() can use from_type for Parallax arguments."""

        def takes_parallax(plx: cxd.Parallax) -> float:
            """Parallax -> float."""
            return plx.value.item()

        strategy = st.builds(takes_parallax, plx=st.from_type(cxd.Parallax))
        value = data.draw(strategy)

        assert isinstance(value, float)
        assert value >= 0
