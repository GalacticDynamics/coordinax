"""Test using Jax operations."""

from functools import partial

import jax
import pytest

from dataclassish import field_items
from unxt import AbstractQuantity

import coordinax as cx
from coordinax._coordinax.base_pos import VECTOR_CLASSES

VECTOR_CLASSES_3D = [c for c in VECTOR_CLASSES if issubclass(c, cx.AbstractPosition3D)]


# TODO: cycle through all representations
@pytest.fixture(params=VECTOR_CLASSES_3D)
def q(request) -> cx.AbstractPosition:
    """Fixture for 3D Vectors."""
    q = cx.CartesianPosition3D.constructor([1, 2, 3], "kpc")
    return q.represent_as(request.param)


@partial(jax.jit, static_argnums=(1,))
def func(
    q: cx.AbstractPosition, target: type[cx.AbstractPosition]
) -> cx.AbstractPosition:
    return q.represent_as(target)


@pytest.mark.parametrize("target", VECTOR_CLASSES_3D)
def test_jax_through_representation(
    q: cx.AbstractPosition, target: type[cx.AbstractPosition]
) -> None:
    """Test using Jax operations through representation."""
    newq = func(q, target)

    assert isinstance(newq, cx.AbstractPosition)
    for k, f in field_items(newq):
        assert isinstance(f, AbstractQuantity), k
