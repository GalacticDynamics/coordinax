"""Test :mod:`vector._utils`."""

import astropy.units as u
import jax.numpy as jnp
import pytest
from jax_quantity import Quantity

from vector._utils import converter_quantity_array


def test_converter_quantity_array_with_unsupported_type():
    """Test that an unsupported type raises a NotImplementedError."""
    with pytest.raises(NotImplementedError):
        converter_quantity_array("unsupported type")


def test_converter_quantity_array_with_quantity():
    """Test that a Quantity is returned as is."""
    q = Quantity(jnp.array([1.0, 2.0, 3.0]), "m")
    result = converter_quantity_array(q)
    assert isinstance(result, Quantity)
    assert jnp.all(result.value == q.value)
    assert result.unit == q.unit


def test_converter_quantity_array_with_jax_array():
    """Test that a JAX array is converted to a Quantity."""
    arr = jnp.array([1.0, 2.0, 3.0])
    result = converter_quantity_array(arr)
    assert isinstance(result, Quantity)
    assert jnp.all(result.value == arr)
    assert result.unit == u.dimensionless_unscaled
