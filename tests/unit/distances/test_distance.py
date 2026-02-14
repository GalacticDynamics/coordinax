"""Unit tests for coordinax.distances.Distance using hypothesis strategies."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, settings
from plum import convert

import unxt as u

import coordinax.distances as cxd
import coordinax_hypothesis.core as cxst

LENGTH = u.dimension("length")


class TestDistanceConstruction:
    """Tests for Distance construction and basic properties."""

    @given(d=cxst.distances())
    def test_is_distance(self, d: cxd.Distance) -> None:
        """Generated distances are Distance instances."""
        assert isinstance(d, cxd.Distance)
        assert isinstance(d, cxd.AbstractDistance)

    @given(d=cxst.distances())
    def test_has_length_dimension(self, d: cxd.Distance) -> None:
        """Distances have length dimensions."""
        assert u.dimension_of(d) == LENGTH

    @given(d=cxst.distances())
    def test_non_negative_default(self, d: cxd.Distance) -> None:
        """Default distances are non-negative."""
        assert jnp.all(d.value >= 0)

    @given(d=cxst.distances(check_negative=False))
    def test_allow_negative(self, d: cxd.Distance) -> None:
        """Can generate distances that might be negative."""
        assert isinstance(d, cxd.Distance)

    @given(d=cxst.distances(unit="kpc"))
    def test_specific_unit(self, d: cxd.Distance) -> None:
        """Can generate distances in specific units."""
        assert d.unit == u.unit("kpc")

    @given(d=cxst.distances(unit="pc"))
    def test_specific_unit_pc(self, d: cxd.Distance) -> None:
        """Can generate distances in parsecs."""
        assert d.unit == u.unit("pc")

    @given(d=cxst.distances(shape=(3,)))
    def test_shape(self, d: cxd.Distance) -> None:
        """Can generate distances with a specific shape."""
        assert d.shape == (3,)

    @given(d=cxst.distances(shape=(2, 3)))
    def test_multidim_shape(self, d: cxd.Distance) -> None:
        """Can generate distances with multi-dimensional shape."""
        assert d.shape == (2, 3)

    @given(d=cxst.distances())
    def test_scalar_default(self, d: cxd.Distance) -> None:
        """Default distances are scalar."""
        assert d.shape == ()

    @given(d=cxst.distances())
    def test_has_value_and_unit(self, d: cxd.Distance) -> None:
        """Generated distances have value and unit attributes."""
        assert hasattr(d, "value")
        assert hasattr(d, "unit")
        assert d.value is not None
        assert d.unit is not None

    def test_invalid_unit_raises(self) -> None:
        """Distance with non-length unit raises ValueError."""
        with pytest.raises(ValueError, match="dimensions length"):
            cxd.Distance(1.0, "rad")

    def test_negative_raises(self) -> None:
        """Distance with negative value raises when check_negative=True."""
        with pytest.raises(eqx.EquinoxRuntimeError):
            cxd.Distance(-1.0, "kpc", check_negative=True)


class TestDistanceArithmetic:
    """Tests for arithmetic operations on Distance."""

    @given(d=cxst.distances())
    def test_add_distances(self, d: cxd.Distance) -> None:
        """Distance + Distance returns Distance."""
        result = d + d
        assert isinstance(result, cxd.Distance)

    @given(d=cxst.distances(elements={"allow_infinity": False}))
    def test_sub_distances(self, d: cxd.Distance) -> None:
        """Distance - Distance returns Distance with zero value."""
        result = d - d
        assert isinstance(result, cxd.Distance)
        assert jnp.allclose(result.value, 0.0)

    @given(d=cxst.distances())
    def test_neg_degrades_to_quantity(self, d: cxd.Distance) -> None:
        """Negation degrades Distance to Quantity (can't be non-negative)."""
        result = -d
        assert isinstance(result, u.AbstractQuantity)
        # Negation of Distance returns Quantity, not Distance
        assert not isinstance(result, cxd.Distance)

    @given(d=cxst.distances())
    def test_scalar_mul(self, d: cxd.Distance) -> None:
        """Scalar multiplication returns Distance."""
        result = 2 * d
        assert isinstance(result, cxd.Distance)
        assert jnp.allclose(result.value, 2 * d.value)

    @given(d=cxst.distances())
    def test_mul_quantity_promotes(self, d: cxd.Distance) -> None:
        """Distance * Quantity promotes to Quantity."""
        q = u.Q(2.0, "s")
        result = d * q
        assert isinstance(result, u.AbstractQuantity)
        assert not isinstance(result, cxd.Distance)


class TestDistanceConversionProperties:
    """Tests for Distance conversion properties."""

    @given(d=cxst.distances(unit="kpc"))
    @settings(deadline=None)
    def test_distance_property_identity(self, d: cxd.Distance) -> None:
        """.distance property returns a Distance."""
        assert isinstance(d.distance, cxd.Distance)

    @given(
        d=cxst.distances(unit="kpc", elements={"min_value": 0.125, "max_value": 1e6})
    )
    @settings(deadline=None)
    def test_parallax_property(self, d: cxd.Distance) -> None:
        """.parallax property returns a Parallax."""
        assert isinstance(d.parallax, cxd.Parallax)

    @given(
        d=cxst.distances(unit="kpc", elements={"min_value": 0.125, "max_value": 1e6})
    )
    @settings(deadline=None)
    def test_distance_modulus_property(self, d: cxd.Distance) -> None:
        """.distance_modulus property returns a DistanceModulus."""
        assert isinstance(d.distance_modulus, cxd.DistanceModulus)


class TestDistanceJAX:
    """Tests for JAX compatibility."""

    @given(d=cxst.distances())
    @settings(deadline=None)
    def test_pytree_roundtrip(self, d: cxd.Distance) -> None:
        """Distance survives PyTree flatten/unflatten."""
        flat, tree = jax.tree.flatten(d)
        restored = jax.tree.unflatten(tree, flat)
        assert type(restored) is type(d)
        assert restored.unit == d.unit
        assert jnp.array_equal(restored.value, d.value)

    @given(d=cxst.distances())
    @settings(deadline=None)
    def test_jit_identity(self, d: cxd.Distance) -> None:
        """JIT-compiled identity preserves Distance."""
        result = jax.jit(lambda x: x)(d)
        assert type(result) is type(d)
        assert jnp.array_equal(result.value, d.value)

    @given(d=cxst.distances())
    @settings(deadline=None)
    def test_jit_add(self, d: cxd.Distance) -> None:
        """JIT-compiled addition works on Distances."""
        result = jax.jit(lambda x: x + x)(d)
        assert isinstance(result, cxd.Distance)
        assert jnp.allclose(result.value, 2 * d.value)

    @given(d=cxst.distances(shape=(3,)))
    @settings(deadline=None)
    def test_vmap(self, d: cxd.Distance) -> None:
        """Vmap works on Distance arrays."""
        result = jax.vmap(lambda x: x + x)(d)
        assert isinstance(result, cxd.Distance)
        assert result.shape == (3,)
        assert jnp.allclose(result.value, 2 * d.value)


class TestDistancePlumConvert:
    """Tests for plum.convert with Distance."""

    @given(d=cxst.distances())
    def test_convert_to_quantity(self, d: cxd.Distance) -> None:
        """Can convert Distance to Quantity."""
        q = convert(d, u.Q)
        assert isinstance(q, u.Q)
        assert q.unit is d.unit
        assert q.value is d.value

    @given(
        d=cxst.distances(unit="kpc", elements={"min_value": 0.125, "max_value": 1e6})
    )
    @settings(deadline=None)
    def test_convert_to_distance_modulus(self, d: cxd.Distance) -> None:
        """Can convert Distance to DistanceModulus."""
        dm = convert(d, cxd.DistanceModulus)
        assert isinstance(dm, cxd.DistanceModulus)
        assert dm.unit == u.unit("mag")

    @given(
        d=cxst.distances(unit="kpc", elements={"min_value": 0.125, "max_value": 1e6})
    )
    @settings(deadline=None)
    def test_convert_to_parallax(self, d: cxd.Distance) -> None:
        """Can convert Distance to Parallax."""
        plx = convert(d, cxd.Parallax)
        assert isinstance(plx, cxd.Parallax)
