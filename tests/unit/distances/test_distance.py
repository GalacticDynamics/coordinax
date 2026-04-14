"""Unit tests for coordinax.distances.Distance.

Covers construction, unit conversion, arithmetic, and JAX compatibility.
Property-based tests use hypothesis strategies from ``coordinax.hypothesis``.

Key behavioral contracts:
* Distance is non-negative by default (``check_negative=True``).
* Negation degrades to a plain length ``Quantity`` — a negative Distance is
  not representable.
* Arithmetic between two Distances preserves the ``Distance`` type.
* Distance is a valid JAX pytree and works under JIT, vmap, and grad.
"""

__all__: tuple[str, ...] = ()

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, settings, strategies as st
from plum import convert

import quaxed.numpy as qnp
import unxt as u

import coordinax.distances as cxd
import coordinax.hypothesis.main as cxst

# ---------------------------------------------------------------------------
# Reusable hypothesis strategies
# ---------------------------------------------------------------------------

# Non-negative float32 values bounded away from overflow (safe to double, add)
_bounded_f32 = st.floats(min_value=0.0, max_value=1e10, width=32)

# Non-negative float32 values in [0, 1] — used for algebraic property tests
_unit_f32 = st.floats(min_value=0.0, max_value=1.0, width=32)

# ---------------------------------------------------------------------------
# Named constants for float32 overflow boundary
#
# float32 max ≈ 3.4e38, so any value > max/2 will overflow when doubled.
# ---------------------------------------------------------------------------
_F32_OVERFLOW_MIN = 1.9999999360571385e38
_F32_OVERFLOW_MAX = 3.3999999521443642e38

LENGTH = u.dimension("length")


class TestDistanceConstruction:
    """Distance construction accepts a variety of input types and validates units."""

    @given(d=cxst.distances())
    def test_is_distance(self, d: cxd.Distance) -> None:
        """Strategy always produces a valid Distance, satisfying the full MRO."""
        assert isinstance(d, cxd.Distance)
        assert isinstance(d, cxd.AbstractDistance)

    @given(d=cxst.distances())
    def test_has_length_dimension(self, d: cxd.Distance) -> None:
        """Every generated Distance carries length dimension."""
        assert u.dimension_of(d) == LENGTH

    @given(d=cxst.distances())
    def test_has_value_and_unit(self, d: cxd.Distance) -> None:
        """Every Distance exposes non-None ``.value`` and ``.unit`` attributes."""
        assert d.value is not None
        assert d.unit is not None

    @given(d=cxst.distances())
    def test_non_negative_default(self, d: cxd.Distance) -> None:
        """Distances generated with default settings are non-negative."""
        assert jnp.all(d.value >= 0)

    @given(d=cxst.distances(check_negative=False))
    def test_allow_negative(self, d: cxd.Distance) -> None:
        """check_negative=False permits negative values."""
        assert isinstance(d, cxd.Distance)

    # --- unit selection ---

    @pytest.mark.parametrize("unit_str", ["kpc", "pc", "m", "km"])
    @given(data=st.data())
    def test_unit_matches_requested(self, unit_str: str, data: st.DataObject) -> None:
        """The unit of a generated Distance matches the requested unit."""
        d = data.draw(cxst.distances(unit=unit_str))
        assert d.unit == u.unit(unit_str)

    # --- shape selection ---

    @given(d=cxst.distances())
    def test_scalar_default(self, d: cxd.Distance) -> None:
        """Distances generated without a shape argument are scalar."""
        assert d.shape == ()

    @pytest.mark.parametrize("shape", [(3,), (2, 3)])
    @given(data=st.data())
    def test_shape_matches_requested(
        self, shape: tuple[int, ...], data: st.DataObject
    ) -> None:
        """The shape of a generated Distance matches the requested shape."""
        d = data.draw(cxst.distances(shape=shape))
        assert d.shape == shape

    # --- construction from concrete Python / JAX values ---

    @pytest.mark.parametrize(
        ("value", "unit_str", "expected_shape"),
        [
            (1, "kpc", ()),
            ([1.0, 2.0, 3.0], "pc", (3,)),
        ],
    )
    def test_construct_from_python(
        self, value: object, unit_str: str, expected_shape: tuple[int, ...]
    ) -> None:
        """Distances can be constructed from Python ints and lists."""
        d = cxd.Distance(value, unit_str)
        assert isinstance(d, cxd.Distance)
        assert d.shape == expected_shape

    def test_construct_from_jnp_array(self) -> None:
        """Distances can be constructed from a JAX array."""
        arr = jnp.array([0.0, 1.0, 2.0])
        d = cxd.Distance(arr, "kpc")
        assert isinstance(d, cxd.Distance)
        assert d.shape == (3,)

    # --- validation ---

    def test_invalid_unit_raises(self) -> None:
        """Non-length units are rejected at construction time."""
        with pytest.raises(ValueError, match="dimensions length"):
            cxd.Distance(1.0, "rad")

    def test_negative_raises_when_checked(self) -> None:
        """Negative values raise when check_negative=True."""
        with pytest.raises(
            (eqx.EquinoxRuntimeError, ValueError),
            match="Distance must be non-negative",
        ):
            cxd.Distance(-1.0, "kpc", check_negative=True)


class TestDistanceConversion:
    """Unit conversion preserves the Distance type and produces correct values."""

    @pytest.mark.parametrize(
        ("value", "from_unit", "to_unit", "expected"),
        [
            (1.0, "kpc", "pc", 1000.0),
            (1000.0, "pc", "kpc", 1.0),
        ],
    )
    def test_unit_conversion(
        self, value: float, from_unit: str, to_unit: str, expected: float
    ) -> None:
        """Converting between length units gives the expected numeric value."""
        result = cxd.Distance(value, from_unit).to(u.unit(to_unit))
        assert isinstance(result, cxd.Distance)
        assert jnp.allclose(result.value, expected, atol=1e-3)

    @given(d=cxst.distances(unit="kpc", elements=_bounded_f32))
    def test_round_trip_kpc_pc_kpc(self, d: cxd.Distance) -> None:
        """Kpc → pc → kpc recovers the original value (up to float32 rounding).

        Uses _bounded_f32 (max 1e10) so that kpc → pc (x1000) stays well below
        float32 overflow (~3.4e38).
        """
        round_tripped = d.to(u.unit("pc")).to(u.unit("kpc"))
        assert isinstance(round_tripped, cxd.Distance)
        assert jnp.allclose(round_tripped.value, d.value, rtol=1e-4)


class TestDistanceArithmetic:
    """Arithmetic operators observe the non-negativity constraint of Distance."""

    @given(d=cxst.distances())
    def test_add_returns_distance(self, d: cxd.Distance) -> None:
        assert isinstance(d + d, cxd.Distance)

    @given(d=cxst.distances(elements={"allow_infinity": False}))
    def test_sub_self_is_zero(self, d: cxd.Distance) -> None:
        result = d - d
        assert isinstance(result, cxd.Distance)
        assert jnp.allclose(result.value, 0.0)

    @given(d=cxst.distances())
    def test_neg_degrades_to_quantity(self, d: cxd.Distance) -> None:
        """Negation cannot produce a valid Distance; result is a plain Quantity.

        This is the key behavioral difference from ``Angle``, where ``-angle``
        returns an ``Angle``.  A ``Distance`` is defined as non-negative, so
        negation must drop to the parent ``Quantity`` type.
        """
        result = -d
        assert isinstance(result, u.AbstractQuantity)
        assert not isinstance(result, cxd.Distance)
        assert jnp.allclose(result.value, -d.value)

    @given(d=cxst.distances())
    def test_scalar_mul_scales_value(self, d: cxd.Distance) -> None:
        result = 2 * d
        assert isinstance(result, cxd.Distance)
        assert jnp.allclose(result.value, 2 * d.value)

    @given(d=cxst.distances(elements=_bounded_f32))
    def test_add_sub_roundtrip(self, d: cxd.Distance) -> None:
        """(d + d) - d == d for values that don't overflow float32."""
        result = (d + d) - d
        assert isinstance(result, cxd.Distance)
        assert jnp.allclose(result.value, d.value, atol=1e-5)

    @given(
        d=cxst.distances(
            elements=st.floats(
                min_value=_F32_OVERFLOW_MIN,
                max_value=_F32_OVERFLOW_MAX,
                width=32,
            )
        )
    )
    def test_add_overflow_produces_inf(self, d: cxd.Distance) -> None:
        """Doubling a near-max float32 Distance produces infinity."""
        result = d + d
        assert isinstance(result, cxd.Distance)
        assert jnp.isinf(result.value)

    @given(d=cxst.distances())
    def test_mul_quantity_promotes(self, d: cxd.Distance) -> None:
        """Distance x dimensioned Quantity yields a plain Quantity, not a Distance."""
        result = d * u.Q(2.0, "s")
        assert isinstance(result, u.AbstractQuantity)
        assert not isinstance(result, cxd.Distance)

    # --- algebraic laws (bounded to avoid float32 overflow) ---

    @given(
        a=cxst.distances(unit="kpc", elements=_unit_f32),
        b=cxst.distances(unit="kpc", elements=_unit_f32),
    )
    def test_add_commutativity(self, a: cxd.Distance, b: cxd.Distance) -> None:
        """A + b == b + a."""
        assert jnp.allclose((a + b).value, (b + a).value, atol=1e-5)

    @given(
        a=cxst.distances(unit="kpc", elements=_unit_f32),
        b=cxst.distances(unit="kpc", elements=_unit_f32),
        c=cxst.distances(unit="kpc", elements=_unit_f32),
    )
    def test_add_associativity(
        self, a: cxd.Distance, b: cxd.Distance, c: cxd.Distance
    ) -> None:
        """(a + b) + c == a + (b + c)."""
        assert jnp.allclose(((a + b) + c).value, (a + (b + c)).value, atol=1e-4)

    def test_atan2_promotes_to_angle(self) -> None:
        """atan2(Distance, Distance) yields an angle-dimensioned Quantity."""
        result = qnp.atan2(cxd.Distance(1.0, "m"), cxd.Distance(3.0, "km"))
        assert isinstance(result, u.AbstractQuantity)
        assert not isinstance(result, cxd.Distance)
        assert u.dimension_of(result) == u.dimension("angle")
        assert jnp.allclose(result.ustrip("rad"), jnp.atan2(1.0, 3000.0))


class TestDistanceConversionProperties:
    """.distance property always returns a Distance."""

    @given(d=cxst.distances(unit="kpc"))
    @settings(deadline=None)
    def test_distance_property_identity(self, d: cxd.Distance) -> None:
        assert isinstance(d.distance, cxd.Distance)


class TestDistancePlumConvert:
    """plum.convert round-trips Distance to Quantity without copying data."""

    @given(d=cxst.distances())
    def test_convert_to_quantity(self, d: cxd.Distance) -> None:
        q = convert(d, u.Q)
        assert isinstance(q, u.Q)
        assert q.unit is d.unit
        assert q.value is d.value


class TestDistanceJAX:
    """Distance is a valid JAX pytree and works under JIT, vmap, and grad."""

    @given(d=cxst.distances())
    @settings(deadline=None)
    def test_pytree_roundtrip(self, d: cxd.Distance) -> None:
        """Flatten → unflatten recovers an identical Distance."""
        flat, tree = jax.tree.flatten(d)
        restored = jax.tree.unflatten(tree, flat)
        assert type(restored) is type(d)
        assert restored.unit == d.unit
        assert jnp.array_equal(restored.value, d.value)

    @given(d=cxst.distances())
    @settings(deadline=None)
    def test_jit_identity(self, d: cxd.Distance) -> None:
        """jax.jit of the identity function preserves the Distance unchanged."""
        result = jax.jit(lambda x: x)(d)
        assert type(result) is type(d)
        assert jnp.array_equal(result.value, d.value)

    @given(d=cxst.distances())
    @settings(deadline=None)
    def test_jit_add(self, d: cxd.Distance) -> None:
        """jax.jit works over Distance addition."""
        result = jax.jit(lambda x: x + x)(d)
        assert isinstance(result, cxd.Distance)
        assert jnp.allclose(result.value, 2 * d.value)

    @given(d=cxst.distances(shape=(3,)))
    @settings(deadline=None)
    def test_vmap(self, d: cxd.Distance) -> None:
        """jax.vmap maps a scalar op over a Distance array."""
        result = jax.vmap(lambda x: x + x)(d)
        assert isinstance(result, cxd.Distance)
        assert result.shape == (3,)
        assert jnp.allclose(result.value, 2 * d.value)

    @given(d=cxst.distances(unit="kpc", elements=_unit_f32))
    @settings(deadline=None)
    def test_grad_through_distance(self, d: cxd.Distance) -> None:
        """jax.grad differentiates through quaxed sum; d/dx sum(x) == 1."""
        g = jax.grad(lambda x: qnp.sum(x).value)(d)
        assert isinstance(g, cxd.Distance)
        assert jnp.allclose(g.value, 1.0, atol=1e-5)
