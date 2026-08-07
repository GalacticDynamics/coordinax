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


def _resolved_from_(cls, arg, /):
    """Return the `from_` implementation plum selects for *arg*.

    `from_` is a `classmethod` wrapping a plum dispatcher, and how many
    wrappers sit between it and the `Function` that can `resolve_method` is not
    stable across Python versions: 3.12 nests
    ``method -> _BoundFunction -> Function``, 3.13+ drops the outer one. Walk
    the chain until something can resolve, rather than hardcoding the hops.

    plum exposes no public accessor here, and which overload gets selected is
    exactly what these tests exist to pin, so the private walk earns its keep.
    """
    f = cls.from_
    for _ in range(5):
        if hasattr(f, "resolve_method"):
            fn, _ = f.resolve_method((cls, arg))
            return fn
        f = getattr(f, "__func__", None) or getattr(f, "_f", None)
        if f is None:
            break
    msg = f"could not reach plum's dispatcher from {cls.__name__}.from_"
    raise AssertionError(msg)


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

    @pytest.mark.parametrize(("value", "unit"), [(1, "mas"), (10, "pc"), (10, "mag")])
    def test_dispatches_by_type_not_by_dimension(self, value: float, unit: str) -> None:
        """The parametric input selects a *different* overload than a plain one.

        Behavioural equivalence alone cannot see this: if the parametric
        registrations were dropped, a `PQ` would fall through to the
        `AbstractQuantity` overload, branch on `u.dimension_of` and return the
        very same answer -- the feature would be gone with every other
        assertion still green.

        Poisoning `u.dimension_of` does not work as a substitute.
        `__check_init__` calls it to validate the *constructed* object, so the
        call lands on both paths and the check fails even when dispatch is
        correct.
        """
        pq = pytest.importorskip("unxts.parametric").PQ(value, unit)
        parametric = _resolved_from_(cxastro.Parallax, pq)
        plain = _resolved_from_(cxastro.Parallax, u.Q(value, unit))
        assert parametric is not plain


class TestFromUnsupportedDimension:
    """`from_` rejects a dimension it has no branch for."""

    def test_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="cannot build a Parallax"):
            cxastro.Parallax.from_(u.Q(1.0, "s"))
