"""Test using Jax operations."""

from functools import partial

import astropy.units as u
import jax
import pytest

from unxt import AbstractQuantity, Quantity

import coordinax as cx
from coordinax._base_pos import VECTOR_CLASSES
from coordinax._utils import dataclass_items

VECTOR_CLASSES_3D = [c for c in VECTOR_CLASSES if issubclass(c, cx.AbstractPosition3D)]


# TODO: cycle through all representations
@pytest.fixture(params=VECTOR_CLASSES_3D)
def q(request) -> cx.AbstractPosition:
    """Fixture for 3D Vectors."""
    q = cx.CartesianPosition3D.constructor(Quantity([1, 2, 3], unit=u.kpc))
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
    for k, f in dataclass_items(newq):
        assert isinstance(f, AbstractQuantity), k
