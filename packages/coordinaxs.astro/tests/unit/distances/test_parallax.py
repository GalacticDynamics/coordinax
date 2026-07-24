"""Unit tests for coordinax.distances.Parallax using hypothesis strategies."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, settings

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


class TestConversionEndpoints:
    """The guard-free conversions hold at the ends of their range.

    `Parallax.from_` builds via `_make`, which skips the non-negativity guard
    because `atan2(1 AU, d)` is never negative. That interval is *closed*:
    `atan2` returns exactly `0` at `d = +inf` and `pi/2` at `d = 0`, and the
    distance-modulus route reaches both by over/underflow of `10 ** x`. Zero
    is what the guard accepts, so these are sound -- asserted here because the
    comments at those call sites assert it in prose.
    """

    @pytest.mark.parametrize(
        ("q", "expected"),
        [
            (u.Q(jnp.inf, "pc"), 0.0),  # d -> inf: atan2 -> 0
            (u.Q(0.0, "pc"), jnp.pi / 2),  # d = 0: atan2 -> pi/2
            (u.Q(1e4, "mag"), 0.0),  # 10**x overflows to +inf
            (u.Q(-1e4, "mag"), jnp.pi / 2),  # 10**x underflows to 0
        ],
        ids=["d_inf", "d_zero", "dm_overflow", "dm_underflow"],
    )
    def test_endpoints_are_non_negative(
        self, q: u.AbstractQuantity, expected: float
    ) -> None:
        plx = cxastro.Parallax.from_(q)
        assert jnp.allclose(plx.ustrip("rad"), expected)
        assert jnp.all(plx.value >= 0)

    @given(plx=cxastrost.parallaxes())
    @settings(deadline=None)
    def test_pytree_roundtrip(self, plx: cxastro.Parallax) -> None:
        """Parallax survives PyTree flatten/unflatten."""
        flat, tree = jax.tree.flatten(plx)
        restored = jax.tree.unflatten(tree, flat)
        assert type(restored) is type(plx)
        assert restored.unit == plx.unit
        assert jnp.array_equal(restored.value, plx.value)

    @given(plx=cxastrost.parallaxes())
    @settings(deadline=None)
    def test_jit_identity(self, plx: cxastro.Parallax) -> None:
        """JIT-compiled identity preserves Parallax."""
        result = jax.jit(lambda x: x)(plx)
        assert type(result) is type(plx)
        assert jnp.array_equal(result.value, plx.value)


class TestParametricFromDispatch:
    """`from_` routes ParametricQuantity by type (optional unxts.parametric)."""

    @pytest.mark.parametrize(("value", "unit"), [(1, "mas"), (10, "pc"), (10, "mag")])
    def test_matches_quantity_path(self, value: float, unit: str) -> None:
        """A ParametricQuantity gives the same result as a plain Quantity."""
        pq = pytest.importorskip("unxts.parametric").PQ(value, unit)
        got = cxastro.Parallax.from_(pq, dtype=float)
        expected = cxastro.Parallax.from_(u.Q(value, unit), dtype=float)
        assert got.unit == expected.unit
        assert jnp.allclose(got.value, expected.value)
