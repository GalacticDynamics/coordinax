"""Tests for chart check functions."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

import unxt as u
import unxt_hypothesis as ust

import coordinax.angles as cxa
from coordinax.charts._src.checks import geq, leq, polar_range, strictly_positive


# Use width=32 for float32 compatibility with JAX
# min/max values must also be representable in float32
def float32s(**kwargs):
    """Generate floats compatible with float32."""
    return st.floats(**kwargs, width=32, allow_subnormal=False)


angle_classes = st.sampled_from((u.Q, u.Angle, u.quantity.BareQuantity, cxa.Angle))

# Float32-representable constants
PI_F32 = float(np.float32(np.pi))  # 3.1415927...


class TestPolarRange:
    """Tests for polar_range check."""

    @given(
        data=st.data(),
        lower=float32s(min_value=0.0, max_value=1.0),
        upper=float32s(min_value=2.0, max_value=PI_F32),
        quantity_cls=angle_classes,
    )
    @settings(max_examples=100, deadline=None)
    def test_angular_within_bounds_passes(
        self, data, lower, upper, quantity_cls
    ) -> None:
        """Angular quantities in [lower, upper] pass through unchanged."""
        angle = data.draw(
            ust.quantities(
                "rad",
                elements=float32s(min_value=lower, max_value=upper),
                quantity_cls=data.draw(quantity_cls),
            ),
            label="angle",
        )
        result = polar_range(angle)
        assert jnp.array_equal(result.value, angle.value)

    @given(ust.quantities("m", elements=float32s(min_value=0.0, max_value=PI_F32)))
    @settings(max_examples=50, deadline=None)
    def test_non_angular_units_raises(self, x: u.AbstractQuantity) -> None:
        """Non-angular quantities always raise, regardless of value."""
        with pytest.raises(eqx.EquinoxTracetimeError, match="must be in angular units"):
            polar_range(x)

    @given(data=st.data())
    @settings(max_examples=50, deadline=None)
    def test_angular_outside_bounds_raises(self, data: st.DataObject) -> None:
        """Angular quantities outside [0, pi] raise an error."""
        # Either below 0 or above pi (use finite values to avoid -inf/inf issues)
        outside_bounds = data.draw(
            ust.quantities(
                "rad",
                elements=st.one_of(
                    float32s(min_value=-10.0, max_value=0.0, exclude_max=True),
                    float32s(min_value=PI_F32, max_value=10.0, exclude_min=True),
                ),
                quantity_cls=data.draw(angle_classes),
            ),
            label="outside_bounds",
        )
        with pytest.raises(
            (eqx.EquinoxRuntimeError, ValueError), match="must be in the range"
        ):
            polar_range(outside_bounds)


class TestStrictlyPositive:
    """Tests for strictly_positive check."""

    # Use 0.0625 (1/16) which is exactly representable in float32
    @given(
        ust.quantities(
            shape=(),
            elements=float32s(min_value=0.0625, max_value=1e9),
        )
    )
    @settings(max_examples=100)
    def test_positive_values_pass(self, x: u.AbstractQuantity) -> None:
        """Positive values should pass through unchanged."""
        result = strictly_positive(x)
        assert jnp.array_equal(result.value, x.value)

    def test_zero_raises(self) -> None:
        """Zero should raise an error."""
        x = u.Q(0.0, "m")
        with pytest.raises(
            (eqx.EquinoxRuntimeError, ValueError),
            match="must be non-negative and non-zero",
        ):
            strictly_positive(x)

    def test_negative_raises(self) -> None:
        """Negative values should raise an error."""
        x = u.Q(-1.0, "m")
        with pytest.raises(
            (eqx.EquinoxRuntimeError, ValueError),
            match="must be non-negative and non-zero",
        ):
            strictly_positive(x)

    def test_array_with_zero_raises(self) -> None:
        """Arrays containing zero should raise an error."""
        x = u.Q([1.0, 0.0, 2.0], "m")
        with pytest.raises(
            (eqx.EquinoxRuntimeError, ValueError),
            match="must be non-negative and non-zero",
        ):
            strictly_positive(x)

    def test_array_with_negative_raises(self) -> None:
        """Arrays containing negative values should raise an error."""
        x = u.Q([1.0, -1.0, 2.0], "m")
        with pytest.raises(
            (eqx.EquinoxRuntimeError, ValueError),
            match="must be non-negative and non-zero",
        ):
            strictly_positive(x)


class TestLeq:
    """Tests for leq (less than or equal) check."""

    @given(
        data=st.data(),
        max_val=float32s(min_value=1.0, max_value=100.0),
    )
    @settings(max_examples=100)
    def test_values_below_max_pass(self, data: st.DataObject, max_val: float) -> None:
        """Values <= max should pass through unchanged."""
        x = data.draw(
            ust.quantities(
                "m", shape=(), elements=float32s(min_value=0.0, max_value=max_val)
            )
        )
        max_q = u.Q(max_val, "m")
        result = leq(x, max_q)
        assert jnp.array_equal(result.value, x.value)

    def test_equal_to_max_passes(self) -> None:
        """Values equal to max should pass."""
        x = u.Q(5.0, "m")
        max_q = u.Q(5.0, "m")
        result = leq(x, max_q)
        assert jnp.array_equal(result.value, x.value)

    def test_above_max_raises(self) -> None:
        """Values above max should raise an error."""
        x = u.Q(6.0, "m")
        max_q = u.Q(5.0, "m")
        with pytest.raises(
            (eqx.EquinoxRuntimeError, ValueError), match="must be less than or equal to"
        ):
            leq(x, max_q)

    def test_array_with_value_above_max_raises(self) -> None:
        """Arrays with any value above max should raise an error."""
        x = u.Q([1.0, 5.0, 6.0], "m")
        max_q = u.Q(5.0, "m")
        with pytest.raises(
            (eqx.EquinoxRuntimeError, ValueError), match="must be less than or equal to"
        ):
            leq(x, max_q)


class TestGeq:
    """Tests for geq (greater than or equal) check."""

    @given(
        data=st.data(),
        min_val=float32s(min_value=0.0, max_value=10.0),
    )
    @settings(max_examples=100)
    def test_values_above_min_pass(self, data: st.DataObject, min_val: float) -> None:
        """Values >= min should pass through unchanged."""
        x = data.draw(
            ust.quantities(
                "m",
                shape=(),
                elements=float32s(min_value=min_val, max_value=100.0),
            )
        )
        min_q = u.Q(min_val, "m")
        result = geq(x, min_q)
        assert jnp.array_equal(result.value, x.value)

    def test_equal_to_min_passes(self) -> None:
        """Values equal to min should pass."""
        x = u.Q(5.0, "m")
        min_q = u.Q(5.0, "m")
        result = geq(x, min_q)
        assert jnp.array_equal(result.value, x.value)

    def test_below_min_raises(self) -> None:
        """Values below min should raise an error."""
        x = u.Q(4.0, "m")
        min_q = u.Q(5.0, "m")
        with pytest.raises(
            (eqx.EquinoxRuntimeError, ValueError),
            match="must be greater than or equal to",
        ):
            geq(x, min_q)

    def test_array_with_value_below_min_raises(self) -> None:
        """Arrays with any value below min should raise an error."""
        x = u.Q([4.0, 5.0, 6.0], "m")
        min_q = u.Q(5.0, "m")
        with pytest.raises(
            (eqx.EquinoxRuntimeError, ValueError),
            match="must be greater than or equal to",
        ):
            geq(x, min_q)
