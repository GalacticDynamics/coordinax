"""Tests for the parallaxes strategy."""

import hypothesis.strategies as st
from hypothesis import given

import unxt as u

import coordinax.astro as cxastro
import coordinax.hypothesis.astro as cxastrost


@given(plx=cxastrost.parallaxes())
def test_basic_parallax(plx: cxastro.Parallax) -> None:
    """Test basic parallax generation."""
    assert isinstance(plx, cxastro.Parallax)
    assert plx.shape == ()
    assert plx.value >= 0  # default check_negative=True


@given(plx=cxastrost.parallaxes(check_negative=False))
def test_parallax_allow_negative(plx: cxastro.Parallax) -> None:
    """Test parallax generation with negative values allowed."""
    assert isinstance(plx, cxastro.Parallax)
    # Don't check sign when check_negative=False


@given(plx=cxastrost.parallaxes(unit="mas"))
def test_parallax_with_units(plx: cxastro.Parallax) -> None:
    """Test parallax generation with specific units."""
    assert isinstance(plx, cxastro.Parallax)
    assert plx.unit == "mas"


@given(plx=cxastrost.parallaxes(unit="arcsec"))
def test_parallax_arcsec(plx: cxastro.Parallax) -> None:
    """Test parallax generation in arcseconds."""
    assert isinstance(plx, cxastro.Parallax)
    assert plx.unit == "arcsec"


@given(plx=cxastrost.parallaxes(shape=5))
def test_parallax_array_1d(plx: cxastro.Parallax) -> None:
    """Test 1D array parallax generation."""
    assert isinstance(plx, cxastro.Parallax)
    assert plx.shape == (5,)
    assert all(plx.value >= 0)  # all elements should be non-negative


@given(plx=cxastrost.parallaxes(shape=(2, 3)))
def test_parallax_array_2d(plx: cxastro.Parallax) -> None:
    """Test 2D parallax array generation."""
    assert isinstance(plx, cxastro.Parallax)
    assert plx.shape == (2, 3)


@given(plx=cxastrost.parallaxes(check_negative=st.sampled_from([True, False])))
def test_parallax_with_strategy_check_negative(plx: cxastro.Parallax) -> None:
    """Test parallax with check_negative as a strategy."""
    assert isinstance(plx, cxastro.Parallax)
    # check_negative varies, so we can't assert about the sign


@given(
    plx=cxastrost.parallaxes(
        elements=st.floats(min_value=1.0, max_value=100.0, width=32)
    )
)
def test_parallax_with_custom_elements(plx: cxastro.Parallax) -> None:
    """Test parallax with custom elements range."""
    assert isinstance(plx, cxastro.Parallax)
    assert 1.0 <= plx.value <= 100.0


@given(
    plx=cxastrost.parallaxes(
        check_negative=True, elements=st.floats(min_value=0.0, max_value=10.0, width=32)
    )
)
def test_parallax_check_negative_with_elements(plx: cxastro.Parallax) -> None:
    """Test that check_negative works with custom elements."""
    assert isinstance(plx, cxastro.Parallax)
    # When check_negative=True and elements provided, min_value should be adjusted
    assert plx.value >= 0
    assert plx.value <= 10.0


class TestParallaxFromType:
    """Test st.from_type() for Parallax type."""

    @given(plx=st.from_type(cxastro.Parallax))
    def test_from_type_basic(self, plx: cxastro.Parallax) -> None:
        """Test that st.from_type(Parallax) generates valid Parallax instances."""
        assert isinstance(plx, cxastro.Parallax)
        # Default check_negative=True means value should be non-negative
        assert plx.value >= 0

    @given(plx=st.from_type(cxastro.Parallax))
    def test_from_type_has_angle_dimension(self, plx: cxastro.Parallax) -> None:
        """Test that generated parallaxes have angle dimension."""
        assert u.dimension_of(plx) == u.dimension("angle")

    @given(data=st.data())
    def test_from_type_generates_variety(self, data: st.DataObject) -> None:
        """Test that from_type generates different values."""
        plx1 = data.draw(st.from_type(cxastro.Parallax))
        plx2 = data.draw(st.from_type(cxastro.Parallax))

        assert isinstance(plx1, cxastro.Parallax)
        assert isinstance(plx2, cxastro.Parallax)

    @given(data=st.data())
    def test_builds_with_parallax_arg(self, data: st.DataObject) -> None:
        """Test that st.builds() can use from_type for Parallax arguments."""

        def takes_parallax(plx: cxastro.Parallax) -> float:
            """Parallax -> float."""
            return plx.value.item()

        strategy = st.builds(takes_parallax, plx=st.from_type(cxastro.Parallax))
        value = data.draw(strategy)

        assert isinstance(value, float)
        assert value >= 0
