"""Tests for AbstractTangentSemanticKind and its subclasses."""

__all__: tuple[str, ...] = ()

import dataclasses

from typing import final

import jax
import jax.tree_util as jtu
import pytest

import coordinax.main as cx
import coordinax.representations as cxr
from coordinax.representations._src.semantics import _TANGENT_TIME_ORDER_LADDER

# ===================================================================


class TestAbstractTangentSemanticKind:
    """AbstractTangentSemanticKind is a proper subclass of AbstractSemanticKind."""

    def test_displacement_inherits(self) -> None:
        """Displacement is a subclass of AbstractTangentSemanticKind."""
        assert issubclass(cxr.Displacement, cxr.AbstractTangentSemanticKind)

    def test_velocity_inherits(self) -> None:
        """Velocity is a subclass of AbstractTangentSemanticKind."""
        assert issubclass(cxr.Velocity, cxr.AbstractTangentSemanticKind)

    def test_acceleration_inherits(self) -> None:
        """Acceleration is a subclass of AbstractTangentSemanticKind."""
        assert issubclass(cxr.Acceleration, cxr.AbstractTangentSemanticKind)

    def test_abstracttangent_is_abstractsemantic(self) -> None:
        """AbstractTangentSemanticKind is a subclass of AbstractSemanticKind."""
        assert issubclass(cxr.AbstractTangentSemanticKind, cxr.AbstractSemanticKind)

    def test_location_not_abstracttangent(self) -> None:
        """Location is NOT a subclass of AbstractTangentSemanticKind."""
        assert not issubclass(cxr.Location, cxr.AbstractTangentSemanticKind)


class TestDisplacement:
    """Displacement construction, equality, exports."""

    def test_construction(self) -> None:
        """Displacement() can be constructed."""
        d = cxr.Displacement()
        assert isinstance(d, cxr.Displacement)

    def test_singleton_is_displacement(self) -> None:
        """`dpl` is the canonical Displacement() instance."""
        assert isinstance(cxr.dpl, cxr.Displacement)

    def test_canonical_name(self) -> None:
        """`dpl` has the correct canonical name."""
        assert cxr.Displacement.canonical_name == "dpl"

    def test_equality(self) -> None:
        """Two Displacement() instances are equal."""
        assert cxr.Displacement() == cxr.Displacement()

    def test_inequality_with_velocity(self) -> None:
        """Displacement is not equal to Velocity."""
        assert cxr.dpl != cxr.vel

    def test_inequality_with_acceleration(self) -> None:
        """Displacement is not equal to Acceleration."""
        assert cxr.dpl != cxr.acc

    def test_inequality_with_location(self) -> None:
        """Displacement is not equal to Location."""
        assert cxr.dpl != cxr.loc

    def test_jax_static(self) -> None:
        """Displacement is a valid JAX static value."""

        @jax.jit
        def identity(x):
            return x

        result = identity(cxr.dpl)
        assert result == cxr.dpl

    def test_exported_from_main(self) -> None:
        """Displacement and dpl exported from coordinax.main."""
        assert hasattr(cx, "Displacement")
        assert hasattr(cx, "dpl")
        assert hasattr(cx, "AbstractTangentSemanticKind")


class TestVelocity:
    """Velocity construction, equality, exports."""

    def test_construction(self) -> None:
        """Velocity() can be constructed."""
        v = cxr.Velocity()
        assert isinstance(v, cxr.Velocity)

    def test_singleton_is_velocity(self) -> None:
        """`vel` is the canonical Velocity() instance."""
        assert isinstance(cxr.vel, cxr.Velocity)

    def test_canonical_name(self) -> None:
        """`vel` has the correct canonical name."""
        assert cxr.Velocity.canonical_name == "vel"

    def test_equality(self) -> None:
        """Two Velocity() instances are equal."""
        assert cxr.Velocity() == cxr.Velocity()

    def test_inequality_with_displacement(self) -> None:
        """Velocity is not equal to Displacement."""
        assert cxr.vel != cxr.dpl

    def test_inequality_with_acceleration(self) -> None:
        """Velocity is not equal to Acceleration."""
        assert cxr.vel != cxr.acc

    def test_jax_static(self) -> None:
        """Velocity is a valid JAX static value."""

        @jax.jit
        def identity(x):
            return x

        result = identity(cxr.vel)
        assert result == cxr.vel

    def test_exported_from_main(self) -> None:
        """Velocity and vel exported from coordinax.main."""
        assert hasattr(cx, "Velocity")
        assert hasattr(cx, "vel")


class TestAcceleration:
    """Acceleration construction, equality, exports."""

    def test_construction(self) -> None:
        """Acceleration() can be constructed."""
        a = cxr.Acceleration()
        assert isinstance(a, cxr.Acceleration)

    def test_singleton_is_acceleration(self) -> None:
        """`acc` is the canonical Acceleration() instance."""
        assert isinstance(cxr.acc, cxr.Acceleration)

    def test_canonical_name(self) -> None:
        """`acc` has the correct canonical name."""
        assert cxr.Acceleration.canonical_name == "acc"

    def test_equality(self) -> None:
        """Two Acceleration() instances are equal."""
        assert cxr.Acceleration() == cxr.Acceleration()

    def test_inequality_with_velocity(self) -> None:
        """Acceleration is not equal to Velocity."""
        assert cxr.acc != cxr.vel

    def test_jax_static(self) -> None:
        """Acceleration is a valid JAX static value."""

        @jax.jit
        def identity(x):
            return x

        result = identity(cxr.acc)
        assert result == cxr.acc

    def test_exported_from_main(self) -> None:
        """Acceleration and acc exported from coordinax.main."""
        assert hasattr(cx, "Acceleration")
        assert hasattr(cx, "acc")


# ===================================================================


class TestTangentTimeOrderLadder:
    """Internal registry mapping time-derivative order to tangent semantic kinds."""

    def test_not_exported_from_representations(self) -> None:
        """TANGENT_TIME_ORDER_LADDER is NOT exported from coordinax.representations."""
        assert not hasattr(cxr, "TANGENT_TIME_ORDER_LADDER")

    def test_is_dict(self) -> None:
        """The internal registry is a dict."""
        assert isinstance(_TANGENT_TIME_ORDER_LADDER, dict)

    def test_has_three_entries(self) -> None:
        """The internal registry has exactly three entries (0, 1, 2)."""
        assert set(_TANGENT_TIME_ORDER_LADDER.keys()) == {0, 1, 2}

    def test_displacement_registered_at_zero(self) -> None:
        """Displacement class is registered at order 0."""
        assert _TANGENT_TIME_ORDER_LADDER[0] is cxr.Displacement

    def test_velocity_registered_at_one(self) -> None:
        """Velocity class is registered at order 1."""
        assert _TANGENT_TIME_ORDER_LADDER[1] is cxr.Velocity

    def test_acceleration_registered_at_two(self) -> None:
        """Acceleration class is registered at order 2."""
        assert _TANGENT_TIME_ORDER_LADDER[2] is cxr.Acceleration

    def test_new_subclass_auto_registers(self) -> None:
        """A new concrete subclass with a new order auto-registers."""

        @final
        @jtu.register_static
        @dataclasses.dataclass(frozen=True, slots=True)
        class Jerk(cxr.AbstractTangentSemanticKind):
            canonical_name = "jrk"
            order = 3

        assert 3 in _TANGENT_TIME_ORDER_LADDER
        assert _TANGENT_TIME_ORDER_LADDER[3] is Jerk
        # Acceleration.derivative() now works because the registry has order 3
        assert isinstance(cxr.Acceleration().derivative(), Jerk)

        # Clean up so we don't pollute other tests
        del _TANGENT_TIME_ORDER_LADDER[3]

    def test_duplicate_order_raises_type_error(self) -> None:
        """Defining a subclass at an already-occupied order raises TypeError."""
        with pytest.raises(TypeError, match="already occupied"):

            @final
            @jtu.register_static
            @dataclasses.dataclass(frozen=True, slots=True)
            class DuplicateVelocity(cxr.AbstractTangentSemanticKind):
                canonical_name = "vel2"
                order = 1  # already occupied by Velocity

    def test_absement_registration_enables_displacement_antiderivative(self) -> None:
        """Registering Absement at order -1 makes Displacement.antiderivative() work."""

        @final
        @jtu.register_static
        @dataclasses.dataclass(frozen=True, slots=True)
        class Absement(cxr.AbstractTangentSemanticKind):
            canonical_name = "abs"
            order = -1

        assert -1 in _TANGENT_TIME_ORDER_LADDER
        assert _TANGENT_TIME_ORDER_LADDER[-1] is Absement
        # Displacement.antiderivative() now resolves via the internal registry
        result = cxr.Displacement().antiderivative()
        assert isinstance(result, Absement)

        # Clean up so we don't pollute other tests
        del _TANGENT_TIME_ORDER_LADDER[-1]


# ===================================================================


class TestOrder:
    """order ClassVar on each concrete tangent semantic kind."""

    def test_displacement_order_is_zero(self) -> None:
        """Displacement.order is 0."""
        assert cxr.Displacement.order == 0

    def test_velocity_order_is_one(self) -> None:
        """Velocity.order is 1."""
        assert cxr.Velocity.order == 1

    def test_acceleration_order_is_two(self) -> None:
        """Acceleration.order is 2."""
        assert cxr.Acceleration.order == 2

    def test_instance_order_matches_class(self) -> None:
        """Instances expose the same order as the class."""
        assert cxr.dpl.order == 0
        assert cxr.vel.order == 1
        assert cxr.acc.order == 2


# ===================================================================


class TestDerivative:
    """derivative() method steps up the time-order ladder."""

    def test_displacement_derivative_is_velocity(self) -> None:
        """Displacement.derivative() returns a Velocity instance."""
        result = cxr.Displacement().derivative()
        assert isinstance(result, cxr.Velocity)

    def test_velocity_derivative_is_acceleration(self) -> None:
        """Velocity.derivative() returns an Acceleration instance."""
        result = cxr.Velocity().derivative()
        assert isinstance(result, cxr.Acceleration)

    def test_acceleration_derivative_raises_value_error(self) -> None:
        """Acceleration.derivative() raises ValueError (no order-3 kind defined)."""
        with pytest.raises(ValueError, match="No tangent semantic kind"):
            cxr.Acceleration().derivative()

    def test_derivative_chain_dpl_to_acc(self) -> None:
        """derivative() chains: dpl -> vel -> acc."""
        assert isinstance(cxr.dpl.derivative(), cxr.Velocity)
        assert isinstance(cxr.dpl.derivative().derivative(), cxr.Acceleration)

    def test_derivative_return_is_instance(self) -> None:
        """derivative() returns a new instance, not the class."""
        result = cxr.dpl.derivative()
        assert isinstance(result, cxr.Velocity)
        assert result == cxr.vel


# ===================================================================


class TestAntiderivative:
    """antiderivative() method steps down the time-order ladder."""

    def test_acceleration_antiderivative_is_velocity(self) -> None:
        """Acceleration.antiderivative() returns a Velocity instance."""
        result = cxr.Acceleration().antiderivative()
        assert isinstance(result, cxr.Velocity)

    def test_velocity_antiderivative_is_displacement(self) -> None:
        """Velocity.antiderivative() returns a Displacement instance."""
        result = cxr.Velocity().antiderivative()
        assert isinstance(result, cxr.Displacement)

    def test_displacement_antiderivative_raises_value_error(self) -> None:
        """Displacement.antiderivative() raises ValueError (no order -1)."""
        with pytest.raises(ValueError, match="No tangent semantic kind"):
            cxr.Displacement().antiderivative()

    def test_antiderivative_chain_acc_to_dpl(self) -> None:
        """antiderivative() chains: acc -> vel -> dpl."""
        assert isinstance(cxr.acc.antiderivative(), cxr.Velocity)
        assert isinstance(cxr.acc.antiderivative().antiderivative(), cxr.Displacement)

    def test_antiderivative_return_is_instance(self) -> None:
        """antiderivative() returns a new instance, not the class."""
        result = cxr.acc.antiderivative()
        assert isinstance(result, cxr.Velocity)
        assert result == cxr.vel


# ===================================================================


class TestDerivativeAntiderivativeRoundtrip:
    """derivative() and antiderivative() are mutual inverses on the interior."""

    def test_dpl_derivative_antiderivative_roundtrip(self) -> None:
        """dpl.derivative().antiderivative() equals dpl."""
        assert cxr.dpl.derivative().antiderivative() == cxr.dpl

    def test_vel_derivative_antiderivative_roundtrip(self) -> None:
        """vel.derivative().antiderivative() equals vel."""
        assert cxr.vel.derivative().antiderivative() == cxr.vel

    def test_acc_antiderivative_derivative_roundtrip(self) -> None:
        """acc.antiderivative().derivative() equals acc."""
        assert cxr.acc.antiderivative().derivative() == cxr.acc

    def test_vel_antiderivative_derivative_roundtrip(self) -> None:
        """vel.antiderivative().derivative() equals vel."""
        assert cxr.vel.antiderivative().derivative() == cxr.vel
