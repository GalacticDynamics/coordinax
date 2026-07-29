"""Angle-specific behaviour: wrapping, and the operators that keep the type.

Construction, unit conversion, the additive laws and JAX compatibility are
shared with `Distance` and asserted once in
`tests/unit/test_quantity_kind_contract.py`.
"""

__all__: tuple[str, ...] = ()

import jax.numpy as jnp
import pytest
from hypothesis import given, settings, strategies as st

import unxt as u
from unxt.quantity import AbstractAngle

import coordinax.angles as cxa
import coordinaxs.hypothesis.main as cxst

# ---------------------------------------------------------------------------
# Reusable hypothesis strategies
# ---------------------------------------------------------------------------

# Float32 values bounded away from overflow (safe to double, add, etc.)
_bounded_f32 = st.floats(min_value=-1e10, max_value=1e10, width=32)

# Float32 values in [0, 1] — used for algebraic property tests
_unit_f32 = st.floats(min_value=0, max_value=1, width=32)

# Float32 values in [-1, 1] — safe trig domain for grad tests
_trig_f32 = st.floats(min_value=-1, max_value=1, width=32)

# ---------------------------------------------------------------------------
# Named constants for float32 overflow boundary
#
# float32 max ≈ 3.4e38, so any value > max/2 will overflow when doubled.
# These exact values are used in tests to avoid subtle float representation
# issues that hypothesis would flag.
# ---------------------------------------------------------------------------
_F32_OVERFLOW_MIN = 1.9999999360571385e38
_F32_OVERFLOW_MAX = 3.3999999521443642e38


class TestAngleConstruction:
    """Angle construction accepts a variety of input types and validates units."""

    @given(angle=cxst.angles())
    def test_is_angle(self, angle: cxa.Angle) -> None:
        """Strategy always produces a valid Angle, satisfying the full MRO."""
        assert isinstance(angle, cxa.Angle)
        assert isinstance(angle, u.Angle)
        assert isinstance(angle, AbstractAngle)

    @given(angle=cxst.angles())
    def test_has_angular_dimension(self, angle: cxa.Angle) -> None:
        """Every generated Angle carries angular dimension."""
        assert u.dimension_of(angle) == u.dimension("angle")

    @given(angle=cxst.angles())
    def test_has_value_and_unit(self, angle: cxa.Angle) -> None:
        """Every Angle exposes non-None ``.value`` and ``.unit`` attributes."""
        assert angle.value is not None
        assert angle.unit is not None

    # --- unit selection ---

    @pytest.mark.parametrize("unit_str", ["deg", "rad"])
    @given(data=st.data())
    def test_unit_matches_requested(self, unit_str: str, data: st.DataObject) -> None:
        """The unit of a generated Angle matches the requested unit."""
        angle = data.draw(cxst.angles(unit=unit_str))
        assert angle.unit == u.unit(unit_str)

    # --- shape selection ---

    @given(angle=cxst.angles())
    def test_scalar_default(self, angle: cxa.Angle) -> None:
        """Angles generated without a shape argument are scalar."""
        assert angle.shape == ()

    @pytest.mark.parametrize("shape", [(3,), (2, 3)])
    @given(data=st.data())
    def test_shape_matches_requested(
        self, shape: tuple[int, ...], data: st.DataObject
    ) -> None:
        """The shape of a generated Angle matches the requested shape."""
        angle = data.draw(cxst.angles(shape=shape))
        assert angle.shape == shape

    # --- construction from concrete Python / JAX values ---

    @pytest.mark.parametrize(
        ("value", "unit_str", "expected_shape"),
        [(1, "rad", ()), ([1, 2, 3], "deg", (3,))],
    )
    def test_construct_from_python(
        self, value: object, unit_str: str, expected_shape: tuple[int, ...]
    ) -> None:
        """Angles can be constructed from Python ints and lists."""
        a = cxa.Angle(value, unit_str)
        assert isinstance(a, cxa.Angle)
        assert a.shape == expected_shape

    def test_construct_from_jnp_array(self) -> None:
        """Angles can be constructed from a JAX array."""
        arr = jnp.array([0, jnp.pi / 2, jnp.pi])
        a = cxa.Angle(arr, "rad")
        assert isinstance(a, cxa.Angle)
        assert a.shape == (3,)

    def test_invalid_unit_raises(self) -> None:
        """Non-angular units are rejected at construction time."""
        with pytest.raises(ValueError, match="angular dimensions"):
            cxa.Angle(1, "m")


class TestAngleConversion:
    """Unit conversion preserves the Angle type and produces correct values."""

    @pytest.mark.parametrize(
        ("value", "from_unit", "to_unit", "expected"),
        [(180, "deg", "rad", jnp.pi), (jnp.pi, "rad", "deg", 180)],
    )
    def test_unit_conversion(
        self, value: float, from_unit: str, to_unit: str, expected: float
    ) -> None:
        """Converting between deg and rad gives the expected numeric value."""
        result = cxa.Angle(value, from_unit).to(u.unit(to_unit))
        assert isinstance(result, cxa.Angle)
        assert jnp.allclose(result.value, expected, atol=1e-5)

    @given(angle=cxst.angles(unit="deg"))
    def test_round_trip_deg_rad_deg(self, angle: cxa.Angle) -> None:
        """Deg → rad → deg recovers the original value (up to float32 rounding)."""
        round_tripped = angle.to(u.unit("rad")).to(u.unit("deg"))
        assert isinstance(round_tripped, cxa.Angle)
        assert jnp.allclose(round_tripped.value, angle.value, atol=1e-4)


class TestAngleWrapTo:
    """``wrap_to`` maps angles into a half-open interval, preserving type/unit."""

    @pytest.mark.parametrize(
        ("unit_str", "lo", "hi"),
        [
            ("deg", u.Q(0, "deg"), u.Q(360, "deg")),
            ("rad", u.Q(-jnp.pi, "rad"), u.Q(jnp.pi, "rad")),
        ],
    )
    @given(data=st.data())
    # JAX traces/compiles `wrap_to` on the first call, which can exceed the
    # default per-example deadline when this test happens to run first under
    # random ordering (cf. `test_wrap_idempotent` below).
    @settings(deadline=None)
    def test_wrap_to_range(
        self, unit_str: str, lo: u.Q, hi: u.Q, data: st.DataObject
    ) -> None:
        """wrap_to places every value within [lo, hi) (up to float tolerance)."""
        angle = data.draw(cxst.angles(unit=unit_str))
        wrapped = angle.wrap_to(lo, hi)
        assert isinstance(wrapped, cxa.Angle)
        assert jnp.all(wrapped.value >= lo.value - 1e-6)
        assert jnp.all(wrapped.value <= hi.value + 1e-6)

    @given(angle=cxst.angles(unit="deg"))
    def test_wrap_preserves_type_and_unit(self, angle: cxa.Angle) -> None:
        """wrap_to does not change the Angle subtype or its unit."""
        wrapped = angle.wrap_to(u.Q(0, "deg"), u.Q(360, "deg"))
        assert type(wrapped) is type(angle)
        assert wrapped.unit == angle.unit

    @given(
        angle=cxst.angles(unit="deg", wrap_to=st.just((u.Q(0, "deg"), u.Q(360, "deg"))))
    )
    def test_strategy_wrap_to(self, angle: cxa.Angle) -> None:
        """The ``wrap_to`` strategy argument pre-wraps generated angles."""
        assert jnp.all(angle.value >= -1e-6)
        assert jnp.all(angle.value <= 360 + 1e-6)

    @given(angle=cxst.angles(unit="deg"))
    @settings(deadline=None)
    def test_wrap_idempotent(self, angle: cxa.Angle) -> None:
        """Applying wrap_to twice returns the same result as applying it once."""
        lo, hi = u.Q(0, "deg"), u.Q(360, "deg")
        once = angle.wrap_to(lo, hi)
        twice = once.wrap_to(lo, hi)
        diff = jnp.abs(once.value - twice.value)
        assert jnp.all((diff < 1e-4) | (jnp.abs(diff - 360) < 1e-4))

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (370, 10),  # one full turn above the range
            (-10, 350),  # one step below zero
        ],
    )
    def test_wrap_known_values(self, value: float, expected: float) -> None:
        """Spot-check wrap_to with concrete in/out pairs."""
        wrapped = cxa.Angle(value, "deg").wrap_to(u.Q(0, "deg"), u.Q(360, "deg"))
        assert jnp.allclose(wrapped.value, expected, atol=1e-4)

    @given(angle=cxst.angles(unit="deg", shape=(4,)))
    def test_wrap_array_angle(self, angle: cxa.Angle) -> None:
        """wrap_to acts element-wise on array-valued Angles."""
        wrapped = angle.wrap_to(u.Q(0, "deg"), u.Q(360, "deg"))
        assert isinstance(wrapped, cxa.Angle)
        assert wrapped.shape == (4,)
        assert jnp.all(wrapped.value >= -1e-6)
        assert jnp.all(wrapped.value <= 360 + 1e-6)


class TestAngleArithmetic:
    """Operators where Angle differs from a plain Quantity, or from Distance."""

    @given(angle=cxst.angles(unit="rad", elements=_bounded_f32))
    def test_neg_flips_sign_and_keeps_type(self, angle: cxa.Angle) -> None:
        """Unlike Distance, an Angle stays an Angle under negation."""
        negated = -angle
        assert isinstance(negated, cxa.Angle)
        assert jnp.allclose(negated.value, -angle.value)

    @given(
        angle=cxst.angles(
            unit="rad",
            elements=st.floats(
                min_value=_F32_OVERFLOW_MIN, max_value=_F32_OVERFLOW_MAX, width=32
            ),
        )
    )
    def test_add_overflow_produces_inf(self, angle: cxa.Angle) -> None:
        """Doubling past float32 max saturates to inf rather than wrapping."""
        assert jnp.all(jnp.isinf((angle + angle).value))

    @given(angle=cxst.angles())
    def test_mul_dimensioned_quantity_drops_the_angle_type(
        self, angle: cxa.Angle
    ) -> None:
        """Angle x dimensioned Quantity yields a plain Quantity, not an Angle."""
        result = angle * u.Q(2, "s")
        assert isinstance(result, u.AbstractQuantity)
        assert not isinstance(result, AbstractAngle)
