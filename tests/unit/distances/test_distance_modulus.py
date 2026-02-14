"""Unit tests for coordinax.distances.DistanceModulus using hypothesis strategies."""

import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, settings
from plum import convert

import unxt as u

import coordinax.distances as cxd
import coordinax_hypothesis.core as cxst


class TestDistanceModulusConstruction:
    """Tests for DistanceModulus construction and basic properties."""

    @given(dm=cxst.distance_moduli())
    def test_is_distance_modulus(self, dm: cxd.DistanceModulus) -> None:
        """Generated distance moduli are DistanceModulus instances."""
        assert isinstance(dm, cxd.DistanceModulus)
        assert isinstance(dm, cxd.AbstractDistance)

    @given(dm=cxst.distance_moduli())
    def test_unit_is_mag(self, dm: cxd.DistanceModulus) -> None:
        """Distance moduli always have unit 'mag'."""
        assert dm.unit == u.unit("mag")

    @given(dm=cxst.distance_moduli())
    def test_scalar_default(self, dm: cxd.DistanceModulus) -> None:
        """Default distance moduli are scalar."""
        assert dm.shape == ()

    @given(dm=cxst.distance_moduli(shape=(3,)))
    def test_shape(self, dm: cxd.DistanceModulus) -> None:
        """Can generate distance moduli with a specific shape."""
        assert dm.shape == (3,)

    @given(dm=cxst.distance_moduli(shape=(2, 3)))
    def test_multidim_shape(self, dm: cxd.DistanceModulus) -> None:
        """Can generate distance moduli with multi-dimensional shape."""
        assert dm.shape == (2, 3)

    @given(dm=cxst.distance_moduli())
    def test_has_value_and_unit(self, dm: cxd.DistanceModulus) -> None:
        """Generated distance moduli have value and unit attributes."""
        assert hasattr(dm, "value")
        assert hasattr(dm, "unit")
        assert dm.value is not None
        assert dm.unit is not None

    @given(dm=cxst.distance_moduli())
    def test_can_be_negative(self, dm: cxd.DistanceModulus) -> None:
        """Distance modulus can be any real number (no non-negative constraint)."""
        # Just verify it's a valid float — DM can be negative for nearby objects
        assert isinstance(dm, cxd.DistanceModulus)

    def test_invalid_unit_raises(self) -> None:
        """DistanceModulus with non-mag unit raises ValueError."""
        with pytest.raises(ValueError, match="magnitude"):
            cxd.DistanceModulus(15.0, "kpc")


class TestDistanceModulusArithmetic:
    """Tests for arithmetic operations on DistanceModulus."""

    @given(dm=cxst.distance_moduli())
    def test_add_distance_moduli(self, dm: cxd.DistanceModulus) -> None:
        """DistanceModulus + DistanceModulus returns DistanceModulus."""
        result = dm + dm
        assert isinstance(result, cxd.DistanceModulus)

    @given(dm=cxst.distance_moduli())
    def test_sub_distance_moduli(self, dm: cxd.DistanceModulus) -> None:
        """DistanceModulus - DistanceModulus returns DistanceModulus with zero."""
        result = dm - dm
        assert isinstance(result, cxd.DistanceModulus)
        assert jnp.allclose(result.value, 0.0)

    @given(dm=cxst.distance_moduli())
    def test_scalar_mul(self, dm: cxd.DistanceModulus) -> None:
        """Scalar multiplication returns DistanceModulus."""
        result = 2 * dm
        assert isinstance(result, cxd.DistanceModulus)
        assert jnp.allclose(result.value, 2 * dm.value)


class TestDistanceModulusConversionProperties:
    """Tests for DistanceModulus conversion properties."""

    @given(dm=cxst.distance_moduli(elements={"min_value": -5.0, "max_value": 25.0}))
    @settings(deadline=None)
    def test_distance_property(self, dm: cxd.DistanceModulus) -> None:
        """.distance property returns a Distance."""
        assert isinstance(dm.distance, cxd.Distance)

    @given(dm=cxst.distance_moduli(elements={"min_value": -5.0, "max_value": 25.0}))
    @settings(deadline=None)
    def test_parallax_property(self, dm: cxd.DistanceModulus) -> None:
        """.parallax property returns a Parallax."""
        assert isinstance(dm.parallax, cxd.Parallax)

    @given(dm=cxst.distance_moduli(elements={"min_value": -5.0, "max_value": 25.0}))
    @settings(deadline=None)
    def test_distance_modulus_property_identity(self, dm: cxd.DistanceModulus) -> None:
        """.distance_modulus property returns a DistanceModulus."""
        assert isinstance(dm.distance_modulus, cxd.DistanceModulus)


class TestDistanceModulusJAX:
    """Tests for JAX compatibility."""

    @given(dm=cxst.distance_moduli())
    @settings(deadline=None)
    def test_pytree_roundtrip(self, dm: cxd.DistanceModulus) -> None:
        """DistanceModulus survives PyTree flatten/unflatten."""
        flat, tree = jax.tree.flatten(dm)
        restored = jax.tree.unflatten(tree, flat)
        assert type(restored) is type(dm)
        assert restored.unit == dm.unit
        assert jnp.array_equal(restored.value, dm.value)

    @given(dm=cxst.distance_moduli())
    @settings(deadline=None)
    def test_jit_identity(self, dm: cxd.DistanceModulus) -> None:
        """JIT-compiled identity preserves DistanceModulus."""
        result = jax.jit(lambda x: x)(dm)
        assert type(result) is type(dm)
        assert jnp.array_equal(result.value, dm.value)

    @given(dm=cxst.distance_moduli())
    @settings(deadline=None)
    def test_jit_add(self, dm: cxd.DistanceModulus) -> None:
        """JIT-compiled addition works on DistanceModulus."""
        result = jax.jit(lambda x: x + x)(dm)
        assert isinstance(result, cxd.DistanceModulus)
        assert jnp.allclose(result.value, 2 * dm.value)

    @given(dm=cxst.distance_moduli(shape=(3,)))
    @settings(deadline=None)
    def test_vmap(self, dm: cxd.DistanceModulus) -> None:
        """Vmap works on DistanceModulus arrays."""
        result = jax.vmap(lambda x: x + x)(dm)
        assert isinstance(result, cxd.DistanceModulus)
        assert result.shape == (3,)


class TestDistanceModulusPlumConvert:
    """Tests for plum.convert with DistanceModulus."""

    @given(dm=cxst.distance_moduli())
    def test_convert_to_quantity(self, dm: cxd.DistanceModulus) -> None:
        """Can convert DistanceModulus to Quantity."""
        q = convert(dm, u.Q)
        assert isinstance(q, u.Q)
        assert q.unit is dm.unit
        assert q.value is dm.value

    @given(dm=cxst.distance_moduli(elements={"min_value": -5.0, "max_value": 25.0}))
    @settings(deadline=None)
    def test_convert_to_distance(self, dm: cxd.DistanceModulus) -> None:
        """Can convert DistanceModulus to Distance."""
        d = convert(dm, cxd.Distance)
        assert isinstance(d, cxd.Distance)

    @given(dm=cxst.distance_moduli(elements={"min_value": -5.0, "max_value": 25.0}))
    @settings(deadline=None)
    def test_convert_to_parallax(self, dm: cxd.DistanceModulus) -> None:
        """Can convert DistanceModulus to Parallax."""
        plx = convert(dm, cxd.Parallax)
        assert isinstance(plx, cxd.Parallax)
