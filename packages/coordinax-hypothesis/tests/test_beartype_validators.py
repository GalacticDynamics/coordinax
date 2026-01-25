"""Test Beartype validator support in hypothesis strategies."""

from jaxtyping import Array, Float
from typing import Annotated

import hypothesis.strategies as st
import jax.numpy as jnp
import pytest
from beartype.vale import Is
from hypothesis import given

import unxt as u

from coordinax_hypothesis._src.utils import (
    strategy_for_annotation,
    wrap_if_not_inspectable,
)


@given(st.data())
def test_beartype_validator_extraction_quantity(data):
    """Test that Beartype validators are extracted from Annotated quantities."""
    # Define an annotated type with a Beartype validator
    PositiveLength = Annotated[u.Q["length"], Is[lambda x: x.value > 0]]

    # Build a strategy for this type
    strategy = strategy_for_annotation(
        wrap_if_not_inspectable(PositiveLength),
        meta={"dtype": jnp.float64, "shape": ()},
    )

    # Generate some examples and verify they're all positive
    value = data.draw(strategy)
    assert value.value > 0, f"Expected positive value, got {value.value}"


@given(st.data())
def test_beartype_validator_extraction_array(data):
    """Test that Beartype validators are extracted from Annotated arrays."""
    # Define an annotated type with a Beartype validator
    PositiveArray = Annotated[Float[Array, ""], Is[lambda x: (x > 0).all()]]

    # Build a strategy for this type
    strategy = strategy_for_annotation(
        wrap_if_not_inspectable(PositiveArray),
        meta={"dtype": jnp.float64, "shape": ()},
    )

    # Generate some examples and verify they're all positive
    value = data.draw(strategy)
    assert value > 0, f"Expected positive value, got {value}"


@given(st.data())
def test_multiple_validators(data):
    """Test that multiple Beartype validators can be combined."""
    # Define an annotated type with multiple validators
    BoundedLength = Annotated[
        u.Q["length"],
        Is[lambda x: x.value > 0],
        Is[lambda x: x.value < 100],
    ]

    # Build a strategy for this type
    strategy = strategy_for_annotation(
        wrap_if_not_inspectable(BoundedLength),
        meta={"dtype": jnp.float64, "shape": ()},
    )

    # Generate some examples and verify they're all in bounds
    value = data.draw(strategy)
    assert 0 < value.value < 100, f"Expected value in (0, 100), got {value.value}"


@given(st.data())
def test_beartype_validator_with_hypothesis(data):
    """Test that Beartype validators work with Hypothesis' @given decorator."""
    PositiveLength = Annotated[u.Q["length"], Is[lambda x: x.value > 0]]

    strategy = strategy_for_annotation(
        wrap_if_not_inspectable(PositiveLength),
        meta={"dtype": jnp.float64, "shape": ()},
    )

    value = data.draw(strategy)
    assert value.value > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
