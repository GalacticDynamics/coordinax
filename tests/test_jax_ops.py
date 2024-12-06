"""Test using Jax operations."""

import equinox as eqx
import pytest

from dataclassish import field_items
from unxt.quantity import AbstractQuantity, Quantity

import coordinax as cx
from coordinax._src.vectors.base.base_pos import POSITION_CLASSES

POSITION_CLASSES_3D = [
    c for c in POSITION_CLASSES if issubclass(c, cx.vecs.AbstractPos3D)
]


# TODO: cycle through all representations
@pytest.fixture(params=POSITION_CLASSES_3D)
def q(request) -> cx.vecs.AbstractPos:
    """Fixture for 3D Vectors."""
    q = cx.CartesianPos3D.from_([1, 2, 3], "kpc")

    # Special case ProlateSpheroidalPos, which requires a value of Delta to define the
    # coordinate system
    kwargs = (
        {}
        if request.param is not cx.vecs.ProlateSpheroidalPos
        else {"Delta": Quantity(1.0, "kpc")}
    )

    return q.represent_as(request.param, **kwargs)


@eqx.filter_jit
def func(
    q: cx.vecs.AbstractPos, target: type[cx.vecs.AbstractPos]
) -> cx.vecs.AbstractPos:
    # Special case ProlateSpheroidalPos, which requires a value of Delta to define the
    # coordinate system
    kwargs = (
        {}
        if target is not cx.vecs.ProlateSpheroidalPos
        else {"Delta": Quantity(1.0, "kpc")}
    )

    return q.represent_as(target, **kwargs)


@pytest.mark.parametrize("target", POSITION_CLASSES_3D)
def test_jax_through_representation(
    q: cx.vecs.AbstractPos, target: type[cx.vecs.AbstractPos]
) -> None:
    """Test using Jax operations through representation."""
    newq = func(q, target)

    assert isinstance(newq, cx.vecs.AbstractPos)
    for k, f in field_items(newq):
        assert isinstance(f, AbstractQuantity), k
