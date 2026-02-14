"""Unit tests for coordinax.distances.Parallax using hypothesis strategies."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, settings

import unxt as u

import coordinax.distances as cxd
import coordinax_hypothesis.core as cxst

ANGLE = u.dimension("angle")


class TestParallaxConstruction:
    """Tests for Parallax construction and basic properties."""

    @given(plx=cxst.parallaxes())
    def test_is_parallax(self, plx: cxd.Parallax) -> None:
        """Generated parallaxes are Parallax instances."""
        assert isinstance(plx, cxd.Parallax)

    @given(plx=cxst.parallaxes())
    def test_has_angular_dimension(self, plx: cxd.Parallax) -> None:
        """Parallaxes have angular dimensions."""
        assert u.dimension_of(plx) == ANGLE

    @given(plx=cxst.parallaxes())
    def test_non_negative_default(self, plx: cxd.Parallax) -> None:
        """Default parallaxes are non-negative."""
        assert jnp.all(plx.value >= 0)

    @given(plx=cxst.parallaxes(check_negative=False))
    def test_allow_negative(self, plx: cxd.Parallax) -> None:
        """Can generate parallaxes that might be negative (noisy data)."""
        assert isinstance(plx, cxd.Parallax)

    @given(plx=cxst.parallaxes(unit="mas"))
    def test_specific_unit(self, plx: cxd.Parallax) -> None:
        """Can generate parallaxes in specific units."""
        assert plx.unit == u.unit("mas")

    @given(plx=cxst.parallaxes(shape=(3,)))
    def test_shape(self, plx: cxd.Parallax) -> None:
        """Can generate parallaxes with a specific shape."""
        assert plx.shape == (3,)

    @given(plx=cxst.parallaxes())
    def test_scalar_default(self, plx: cxd.Parallax) -> None:
        """Default parallaxes are scalar."""
        assert plx.shape == ()

    def test_negative_raises(self) -> None:
        """Parallax with negative value raises when check_negative=True."""
        with pytest.raises(eqx.EquinoxRuntimeError):
            cxd.Parallax(-1.0, "mas", check_negative=True)

    @given(plx=cxst.parallaxes())
    def test_has_value_and_unit(self, plx: cxd.Parallax) -> None:
        """Generated parallaxes have value and unit attributes."""
        assert hasattr(plx, "value")
        assert hasattr(plx, "unit")


class TestParallaxJAX:
    """Tests for JAX compatibility of Parallax."""

    @given(plx=cxst.parallaxes())
    @settings(deadline=None)
    def test_pytree_roundtrip(self, plx: cxd.Parallax) -> None:
        """Parallax survives PyTree flatten/unflatten."""
        flat, tree = jax.tree.flatten(plx)
        restored = jax.tree.unflatten(tree, flat)
        assert type(restored) is type(plx)
        assert restored.unit == plx.unit
        assert jnp.array_equal(restored.value, plx.value)

    @given(plx=cxst.parallaxes())
    @settings(deadline=None)
    def test_jit_identity(self, plx: cxd.Parallax) -> None:
        """JIT-compiled identity preserves Parallax."""
        result = jax.jit(lambda x: x)(plx)
        assert type(result) is type(plx)
        assert jnp.array_equal(result.value, plx.value)
