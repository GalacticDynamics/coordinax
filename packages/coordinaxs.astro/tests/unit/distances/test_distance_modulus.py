"""Unit tests for coordinax.distances.DistanceModulus using hypothesis strategies."""

import jax.numpy as jnp
import plum
import pytest
from hypothesis import given, settings

import unxt as u

import coordinax.distances as cxd
import coordinaxs.astro as cxastro
import coordinaxs.hypothesis.astro as cxastrost


class TestDistanceModulusConstruction:
    """Tests for DistanceModulus construction and basic properties."""

    @given(dm=cxastrost.distance_moduli())
    def test_unit_is_mag(self, dm: cxastro.DistanceModulus) -> None:
        """Distance moduli always have unit 'mag'."""
        assert dm.unit == u.unit("mag")

    def test_invalid_unit_raises(self) -> None:
        """DistanceModulus with non-mag unit raises ValueError."""
        with pytest.raises(ValueError, match="magnitude"):
            cxastro.DistanceModulus(15, "kpc")


class TestDistanceModulusArithmetic:
    """Tests for arithmetic operations on DistanceModulus."""

    @given(dm=cxastrost.distance_moduli())
    def test_add_distance_moduli(self, dm: cxastro.DistanceModulus) -> None:
        """DistanceModulus + DistanceModulus returns DistanceModulus."""
        result = dm + dm
        assert isinstance(result, cxastro.DistanceModulus)

    @given(dm=cxastrost.distance_moduli())
    def test_sub_distance_moduli(self, dm: cxastro.DistanceModulus) -> None:
        """DistanceModulus - DistanceModulus returns DistanceModulus with zero."""
        result = dm - dm
        assert isinstance(result, cxastro.DistanceModulus)
        assert jnp.allclose(result.value, 0)

    @given(dm=cxastrost.distance_moduli())
    def test_scalar_mul(self, dm: cxastro.DistanceModulus) -> None:
        """Scalar multiplication returns DistanceModulus."""
        result = 2 * dm
        assert isinstance(result, cxastro.DistanceModulus)
        assert jnp.allclose(result.value, 2 * dm.value)


class TestDistanceModulusConversionProperties:
    """Tests for DistanceModulus conversion properties."""

    @given(dm=cxastrost.distance_moduli(elements={"min_value": -5, "max_value": 25}))
    @settings(deadline=None)
    def test_distance_property(self, dm: cxastro.DistanceModulus) -> None:
        """.distance property returns a Distance."""
        assert isinstance(dm.distance, cxd.Distance)


class TestDistanceModulusPlumConvert:
    """Tests for plum.convert with DistanceModulus."""

    @given(dm=cxastrost.distance_moduli())
    def test_convert_to_quantity(self, dm: cxastro.DistanceModulus) -> None:
        """Can convert DistanceModulus to Quantity."""
        q = plum.convert(dm, u.Q)
        assert isinstance(q, u.Q)
        assert q.unit is dm.unit
        assert q.value is dm.value

    @given(dm=cxastrost.distance_moduli(elements={"min_value": -5, "max_value": 25}))
    @settings(deadline=None)
    def test_convert_to_distance(self, dm: cxastro.DistanceModulus) -> None:
        """Can convert DistanceModulus to Distance."""
        d = plum.convert(dm, cxd.Distance)
        assert isinstance(d, cxd.Distance)

    @given(dm=cxastrost.distance_moduli(elements={"min_value": -5, "max_value": 25}))
    @settings(deadline=None)
    def test_convert_to_parallax(self, dm: cxastro.DistanceModulus) -> None:
        """Can convert DistanceModulus to Parallax."""
        plx = plum.convert(dm, cxastro.Parallax)
        assert isinstance(plx, cxastro.Parallax)
