"""Test using Jax operations."""

import equinox as eqx
import pytest

from dataclassish import field_items
from unxt import AbstractQuantity

import coordinax as cx
from coordinax._src.base.base_pos import POSITION_CLASSES

POSITION_CLASSES_3D = [
    c for c in POSITION_CLASSES if issubclass(c, cx.AbstractPosition3D)
]


# TODO: cycle through all representations
@pytest.fixture(params=POSITION_CLASSES_3D)
def q(request) -> cx.AbstractPosition:
    """Fixture for 3D Vectors."""
    q = cx.CartesianPosition3D.from_([1, 2, 3], "kpc")
    return q.represent_as(request.param)


@eqx.filter_jit
def func(
    q: cx.AbstractPosition, target: type[cx.AbstractPosition]
) -> cx.AbstractPosition:
    return q.represent_as(target)


@pytest.mark.parametrize("target", POSITION_CLASSES_3D)
def test_jax_through_representation(
    q: cx.AbstractPosition, target: type[cx.AbstractPosition]
) -> None:
    """Test using Jax operations through representation."""
    newq = func(q, target)

    assert isinstance(newq, cx.AbstractPosition)
    for k, f in field_items(newq):
        assert isinstance(f, AbstractQuantity), k
