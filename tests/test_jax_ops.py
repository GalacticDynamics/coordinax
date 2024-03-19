"""Test using Jax operations."""

from functools import partial

import astropy.units as u
import jax
import pytest

from unxt import Quantity

import coordinax as cx
from coordinax._base_vec import VECTOR_CLASSES
from coordinax._utils import dataclass_items

VECTOR_CLASSES_3D = [c for c in VECTOR_CLASSES if issubclass(c, cx.Abstract3DVector)]


# TODO: cycle through all representations
@pytest.fixture(params=VECTOR_CLASSES_3D)
def q(request) -> cx.AbstractVector:
    """Fixture for 3D Vectors."""
    q = cx.Cartesian3DVector.constructor(Quantity([1, 2, 3], unit=u.kpc))
    return q.represent_as(request.param)


@partial(jax.jit, static_argnums=(1,))
def func(q: cx.AbstractVector, target: type[cx.AbstractVector]) -> cx.AbstractVector:
    return q.represent_as(target)


@pytest.mark.parametrize("target", VECTOR_CLASSES_3D)
def test_jax_through_representation(
    q: cx.AbstractVector, target: type[cx.AbstractVector]
) -> None:
    """Test using Jax operations through representation."""
    newq = func(q, target)

    assert isinstance(newq, cx.AbstractVector)
    for k, f in dataclass_items(newq):
        assert isinstance(f, Quantity), k
