"""Tests for the distance_moduli strategy."""

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

import coordinax.distances as cxd
import coordinax_hypothesis.core as cxst


@given(dm=cxst.distance_moduli())
@settings(max_examples=50)
def test_basic_distance_modulus(dm: cxd.DistanceModulus) -> None:
    """Test basic distance modulus generation."""
    assert isinstance(dm, cxd.DistanceModulus)
    assert dm.shape == ()
    assert dm.unit == "mag"


@given(dm=cxst.distance_moduli(shape=5))
@settings(max_examples=30)
def test_distance_modulus_vector(dm: cxd.DistanceModulus) -> None:
    """Test vector distance modulus generation."""
    assert isinstance(dm, cxd.DistanceModulus)
    assert dm.shape == (5,)
    assert dm.unit == "mag"


@given(dm=cxst.distance_moduli(shape=(2, 3)))
@settings(max_examples=30)
def test_distance_modulus_2d(dm: cxd.DistanceModulus) -> None:
    """Test 2D distance modulus array generation."""
    assert isinstance(dm, cxd.DistanceModulus)
    assert dm.shape == (2, 3)
    assert dm.unit == "mag"


@given(dm=cxst.distance_moduli(elements=st.floats(min_value=0, max_value=30, width=32)))
@settings(max_examples=30)
def test_distance_modulus_with_custom_elements(dm: cxd.DistanceModulus) -> None:
    """Test distance modulus with custom elements range."""
    assert isinstance(dm, cxd.DistanceModulus)
    assert 0.0 <= dm.value <= 30.0
    assert dm.unit == "mag"


@given(st.data())
@settings(max_examples=1)
def test_distance_modulus_unit_warning(data: st.DataObject) -> None:
    """Test that specifying a unit raises a warning."""
    with pytest.warns(UserWarning, match="DistanceModulus always uses 'mag' units"):
        dm = data.draw(cxst.distance_moduli(unit="pc"))

    # Verify it still creates valid DistanceModulus with 'mag' units
    assert isinstance(dm, cxd.DistanceModulus)
    assert dm.unit == "mag"


class TestDistanceModulusFromType:
    """Test st.from_type() for DistanceModulus type."""

    @given(dm=st.from_type(cxd.DistanceModulus))
    def test_from_type_basic(self, dm: cxd.DistanceModulus) -> None:
        """Test that st.from_type(DistanceModulus) generates valid instances."""
        assert isinstance(dm, cxd.DistanceModulus)

    @given(dm=st.from_type(cxd.DistanceModulus))
    def test_from_type_has_mag_units(self, dm: cxd.DistanceModulus) -> None:
        """Test that generated distance moduli have magnitude units."""
        assert dm.unit == "mag"

    @given(data=st.data())
    def test_from_type_generates_variety(self, data: st.DataObject) -> None:
        """Test that from_type generates different values."""
        dm1 = data.draw(st.from_type(cxd.DistanceModulus))
        dm2 = data.draw(st.from_type(cxd.DistanceModulus))

        assert isinstance(dm1, cxd.DistanceModulus)
        assert isinstance(dm2, cxd.DistanceModulus)

    @given(data=st.data())
    def test_builds_with_distance_modulus_arg(self, data: st.DataObject) -> None:
        """Test that st.builds() can use from_type for DistanceModulus arguments."""

        def takes_distance_modulus(dm: cxd.DistanceModulus) -> float:
            """DistanceModulus -> float."""
            return dm.value.item()

        strategy = st.builds(
            takes_distance_modulus, dm=st.from_type(cxd.DistanceModulus)
        )
        value = data.draw(strategy)

        assert isinstance(value, float)
