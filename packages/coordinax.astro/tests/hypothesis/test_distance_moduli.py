"""Tests for the distance_moduli strategy."""

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

import coordinax.astro as cxastro
import coordinax.hypothesis.astro as cxastrost


@given(dm=cxastrost.distance_moduli())
def test_basic_distance_modulus(dm: cxastro.DistanceModulus) -> None:
    """Test basic distance modulus generation."""
    assert isinstance(dm, cxastro.DistanceModulus)
    assert dm.shape == ()
    assert dm.unit == "mag"


@given(dm=cxastrost.distance_moduli(shape=5))
def test_distance_modulus_vector(dm: cxastro.DistanceModulus) -> None:
    """Test vector distance modulus generation."""
    assert isinstance(dm, cxastro.DistanceModulus)
    assert dm.shape == (5,)
    assert dm.unit == "mag"


@given(dm=cxastrost.distance_moduli(shape=(2, 3)))
def test_distance_modulus_2d(dm: cxastro.DistanceModulus) -> None:
    """Test 2D distance modulus array generation."""
    assert isinstance(dm, cxastro.DistanceModulus)
    assert dm.shape == (2, 3)
    assert dm.unit == "mag"


@given(
    dm=cxastrost.distance_moduli(
        elements=st.floats(min_value=0, max_value=30, width=32)
    )
)
def test_distance_modulus_with_custom_elements(dm: cxastro.DistanceModulus) -> None:
    """Test distance modulus with custom elements range."""
    assert isinstance(dm, cxastro.DistanceModulus)
    assert 0 <= dm.value <= 30
    assert dm.unit == "mag"


@given(st.data())
@settings(max_examples=1)  # Only need one example to test the warning
def test_distance_modulus_unit_warning(data: st.DataObject) -> None:
    """Test that specifying a unit raises a warning."""
    with pytest.warns(UserWarning, match="DistanceModulus always uses 'mag' units"):
        dm = data.draw(cxastrost.distance_moduli(unit="pc"))

    # Verify it still creates valid DistanceModulus with 'mag' units
    assert isinstance(dm, cxastro.DistanceModulus)
    assert dm.unit == "mag"


class TestDistanceModulusFromType:
    """Test st.from_type() for DistanceModulus type."""

    @given(dm=st.from_type(cxastro.DistanceModulus))
    def test_from_type_basic(self, dm: cxastro.DistanceModulus) -> None:
        """Test that st.from_type(DistanceModulus) generates valid instances."""
        assert isinstance(dm, cxastro.DistanceModulus)

    @given(dm=st.from_type(cxastro.DistanceModulus))
    def test_from_type_has_mag_units(self, dm: cxastro.DistanceModulus) -> None:
        """Test that generated distance moduli have magnitude units."""
        assert dm.unit == "mag"

    @given(data=st.data())
    def test_from_type_generates_variety(self, data: st.DataObject) -> None:
        """Test that from_type generates different values."""
        dm1 = data.draw(st.from_type(cxastro.DistanceModulus))
        dm2 = data.draw(st.from_type(cxastro.DistanceModulus))

        assert isinstance(dm1, cxastro.DistanceModulus)
        assert isinstance(dm2, cxastro.DistanceModulus)

    @given(data=st.data())
    def test_builds_with_distance_modulus_arg(self, data: st.DataObject) -> None:
        """Test that st.builds() can use from_type for DistanceModulus arguments."""

        def takes_distance_modulus(dm: cxastro.DistanceModulus) -> float:
            """DistanceModulus -> float."""
            return dm.value.item()

        strategy = st.builds(
            takes_distance_modulus, dm=st.from_type(cxastro.DistanceModulus)
        )
        value = data.draw(strategy)

        assert isinstance(value, float)
