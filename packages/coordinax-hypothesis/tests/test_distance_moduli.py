"""Tests for the distance_moduli strategy."""

import hypothesis.strategies as st
import pytest
from coordinax_hypothesis import distance_moduli
from hypothesis import given, settings

import coordinax.distance as cxd


@given(dm=distance_moduli())
@settings(max_examples=50)
def test_basic_distance_modulus(dm: cxd.DistanceModulus) -> None:
    """Test basic distance modulus generation."""
    assert isinstance(dm, cxd.DistanceModulus)
    assert dm.shape == ()
    assert dm.unit == "mag"


@given(dm=distance_moduli(shape=5))
@settings(max_examples=30)
def test_distance_modulus_vector(dm: cxd.DistanceModulus) -> None:
    """Test vector distance modulus generation."""
    assert isinstance(dm, cxd.DistanceModulus)
    assert dm.shape == (5,)
    assert dm.unit == "mag"


@given(dm=distance_moduli(shape=(2, 3)))
@settings(max_examples=30)
def test_distance_modulus_2d(dm: cxd.DistanceModulus) -> None:
    """Test 2D distance modulus array generation."""
    assert isinstance(dm, cxd.DistanceModulus)
    assert dm.shape == (2, 3)
    assert dm.unit == "mag"


@given(dm=distance_moduli(elements=st.floats(min_value=0, max_value=30, width=32)))
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
        dm = data.draw(distance_moduli(unit="pc"))

    # Verify it still creates valid DistanceModulus with 'mag' units
    assert isinstance(dm, cxd.DistanceModulus)
    assert dm.unit == "mag"
