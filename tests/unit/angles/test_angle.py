"""Unit tests for coordinax.angles.Angle using hypothesis strategies."""

import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, settings, strategies as st

import unxt as u
from unxt.quantity import AbstractAngle

import coordinax_hypothesis.core as cxst


class TestAngleConstruction:
    """Tests for Angle construction and basic properties."""

    @given(angle=cxst.angles())
    def test_is_angle(self, angle: u.Angle) -> None:
        """Generated angles are Angle instances."""
        assert isinstance(angle, u.Angle)
        assert isinstance(angle, AbstractAngle)

    @given(angle=cxst.angles())
    def test_has_angular_dimension(self, angle: u.Angle) -> None:
        """Generated angles have angular dimensions."""
        assert u.dimension_of(angle) == u.dimension("angle")

    @given(angle=cxst.angles())
    def test_has_value_and_unit(self, angle: u.Angle) -> None:
        """Generated angles have value and unit attributes."""
        assert hasattr(angle, "value")
        assert hasattr(angle, "unit")
        assert angle.value is not None
        assert angle.unit is not None

    @given(angle=cxst.angles(unit="deg"))
    def test_specific_unit(self, angle: u.Angle) -> None:
        """Can generate angles with a specific unit."""
        assert angle.unit == u.unit("deg")

    @given(angle=cxst.angles(unit="rad"))
    def test_specific_unit_rad(self, angle: u.Angle) -> None:
        """Can generate angles in radians."""
        assert angle.unit == u.unit("rad")

    @given(angle=cxst.angles(shape=(3,)))
    def test_shape(self, angle: u.Angle) -> None:
        """Can generate angles with a specific shape."""
        assert angle.shape == (3,)

    @given(angle=cxst.angles(shape=(2, 3)))
    def test_multidim_shape(self, angle: u.Angle) -> None:
        """Can generate angles with multi-dimensional shape."""
        assert angle.shape == (2, 3)

    @given(angle=cxst.angles())
    def test_scalar_default(self, angle: u.Angle) -> None:
        """Default angles are scalar."""
        assert angle.shape == ()

    def test_invalid_unit_raises(self) -> None:
        """Angle with non-angular unit raises ValueError."""
        with pytest.raises(ValueError, match="angular dimensions"):
            u.Angle(1.0, "m")


class TestAngleWrapTo:
    """Tests for angle wrapping behavior."""

    @given(angle=cxst.angles(unit="deg"))
    def test_wrap_to_0_360(self, angle: u.Angle) -> None:
        """Wrapping to [0, 360) produces values in range."""
        wrapped = angle.wrap_to(u.Q(0, "deg"), u.Q(360, "deg"))
        assert isinstance(wrapped, AbstractAngle)
        assert jnp.all(wrapped.value >= -1e-6)
        assert jnp.all(wrapped.value <= 360 + 1e-6)

    @given(angle=cxst.angles(unit="rad"))
    def test_wrap_to_neg_pi_pi(self, angle: u.Angle) -> None:
        """Wrapping to [-pi, pi) produces values in range."""
        wrapped = angle.wrap_to(u.Q(-jnp.pi, "rad"), u.Q(jnp.pi, "rad"))
        assert isinstance(wrapped, AbstractAngle)
        assert jnp.all(wrapped.value >= -jnp.pi - 1e-6)
        assert jnp.all(wrapped.value <= jnp.pi + 1e-6)

    @given(angle=cxst.angles(unit="deg"))
    def test_wrap_preserves_type(self, angle: u.Angle) -> None:
        """Wrapping preserves the Angle type."""
        wrapped = angle.wrap_to(u.Q(0, "deg"), u.Q(360, "deg"))
        assert type(wrapped) is type(angle)

    @given(angle=cxst.angles(unit="deg"))
    def test_wrap_preserves_unit(self, angle: u.Angle) -> None:
        """Wrapping preserves the unit."""
        wrapped = angle.wrap_to(u.Q(0, "deg"), u.Q(360, "deg"))
        assert wrapped.unit == angle.unit

    @given(
        angle=cxst.angles(
            unit="deg",
            wrap_to=st.just((u.Q(0, "deg"), u.Q(360, "deg"))),
        )
    )
    def test_strategy_wrap_to(self, angle: u.Angle) -> None:
        """Strategy-level wrap_to generates wrapped angles."""
        assert isinstance(angle, u.Angle)
        # The angle should already be approximately in [0, 360)
        assert jnp.all(angle.value >= -1e-6)
        assert jnp.all(angle.value <= 360 + 1e-6)

    @given(angle=cxst.angles(unit="deg"))
    def test_wrap_idempotent(self, angle: u.Angle) -> None:
        """Wrapping an already-wrapped angle is idempotent."""
        lo, hi = u.Q(0, "deg"), u.Q(360, "deg")
        once = angle.wrap_to(lo, hi)
        twice = once.wrap_to(lo, hi)
        # Values should be equivalent modulo the period
        diff = jnp.abs(once.value - twice.value)
        assert jnp.all((diff < 1e-4) | (jnp.abs(diff - 360) < 1e-4))


class TestAngleArithmetic:
    """Tests for arithmetic operations on Angles."""

    @given(angle=cxst.angles())
    def test_add_self(self, angle: u.Angle) -> None:
        """Angle + Angle returns an Angle."""
        result = angle + angle
        assert isinstance(result, u.Angle)

    @given(angle=cxst.angles())
    def test_sub_self(self, angle: u.Angle) -> None:
        """Angle - Angle returns an Angle."""
        result = angle - angle
        assert isinstance(result, u.Angle)
        assert jnp.allclose(result.value, 0.0)

    @given(angle=cxst.angles())
    def test_neg(self, angle: u.Angle) -> None:
        """Negation returns an Angle."""
        result = -angle
        assert isinstance(result, u.Angle)
        assert jnp.allclose(result.value, -angle.value)

    @given(angle=cxst.angles())
    def test_scalar_mul(self, angle: u.Angle) -> None:
        """Scalar multiplication returns an Angle."""
        result = 2 * angle
        assert isinstance(result, u.Angle)
        assert jnp.allclose(result.value, 2 * angle.value)

    @given(angle=cxst.angles())
    def test_add_sub_roundtrip(self, angle: u.Angle) -> None:
        """(a + a) - a == a."""
        result = (angle + angle) - angle
        assert isinstance(result, u.Angle)
        assert jnp.allclose(result.value, angle.value, atol=1e-5)

    @given(angle=cxst.angles())
    def test_mul_quantity_promotes(self, angle: u.Angle) -> None:
        """Angle * Quantity promotes to Quantity."""
        q = u.Q(2.0, "s")
        result = angle * q
        assert isinstance(result, u.AbstractQuantity)
        assert not isinstance(result, AbstractAngle)


class TestAngleJAX:
    """Tests for JAX compatibility."""

    @given(angle=cxst.angles())
    @settings(deadline=None)
    def test_pytree_roundtrip(self, angle: u.Angle) -> None:
        """Angle survives PyTree flatten/unflatten."""
        flat, tree = jax.tree.flatten(angle)
        restored = jax.tree.unflatten(tree, flat)
        assert type(restored) is type(angle)
        assert restored.unit == angle.unit
        assert jnp.array_equal(restored.value, angle.value)

    @given(angle=cxst.angles())
    @settings(deadline=None)
    def test_jit_identity(self, angle: u.Angle) -> None:
        """JIT-compiled identity preserves Angle."""
        result = jax.jit(lambda x: x)(angle)
        assert type(result) is type(angle)
        assert jnp.array_equal(result.value, angle.value)

    @given(angle=cxst.angles())
    @settings(deadline=None)
    def test_jit_add(self, angle: u.Angle) -> None:
        """JIT-compiled addition works on Angles."""
        result = jax.jit(lambda x: x + x)(angle)
        assert isinstance(result, u.Angle)
        assert jnp.allclose(result.value, 2 * angle.value)

    @given(angle=cxst.angles(shape=(3,)))
    @settings(deadline=None)
    def test_vmap(self, angle: u.Angle) -> None:
        """Vmap works on Angle arrays."""
        result = jax.vmap(lambda x: x + x)(angle)
        assert isinstance(result, u.Angle)
        assert result.shape == (3,)
        assert jnp.allclose(result.value, 2 * angle.value)
