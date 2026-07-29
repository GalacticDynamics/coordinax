"""Unit tests for coordinax.distances.Parallax using hypothesis strategies."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from hypothesis import given

import unxt as u

import coordinaxs.astro as cxastro
import coordinaxs.hypothesis.astro as cxastrost

ANGLE = u.dimension("angle")


class TestParallaxConstruction:
    """Tests for Parallax construction and basic properties."""

    @given(plx=cxastrost.parallaxes())
    def test_has_angular_dimension(self, plx: cxastro.Parallax) -> None:
        """Parallaxes have angular dimensions."""
        assert u.dimension_of(plx) == ANGLE

    @given(plx=cxastrost.parallaxes())
    def test_non_negative_default(self, plx: cxastro.Parallax) -> None:
        """Default parallaxes are non-negative."""
        assert jnp.all(plx.value >= 0)

    @given(plx=cxastrost.parallaxes(check_negative=False))
    def test_allow_negative(self, plx: cxastro.Parallax) -> None:
        """Can generate parallaxes that might be negative (noisy data)."""
        assert isinstance(plx, cxastro.Parallax)

    @given(plx=cxastrost.parallaxes(unit="mas"))
    def test_specific_unit(self, plx: cxastro.Parallax) -> None:
        """Can generate parallaxes in specific units."""
        assert plx.unit == u.unit("mas")

    def test_negative_raises(self) -> None:
        """Parallax with negative value raises when check_negative=True."""
        with pytest.raises(
            (eqx.EquinoxRuntimeError, ValueError), match="Parallax must be non-negative"
        ):
            cxastro.Parallax(-1, "mas", check_negative=True)

    def test_negative_raises_under_jit(self) -> None:
        """The non-negativity check is not dead-code-eliminated under jit."""

        @eqx.filter_jit
        def build(v: jax.Array) -> jax.Array:
            return cxastro.Parallax(v, "mas", check_negative=True).value

        with pytest.raises(
            eqx.EquinoxRuntimeError, match="Parallax must be non-negative"
        ):
            jax.block_until_ready(build(jnp.asarray(-1.0)))
