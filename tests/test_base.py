"""Test :mod:`coordinax.utils`."""

from collections import UserDict
from contextlib import AbstractContextManager, nullcontext
from dataclasses import replace
from types import MappingProxyType
from typing import Any

import pytest

import quaxed.numpy as jnp
from dataclassish import field_items
from unxt.quantity import AbstractQuantity

import coordinax as cx

BUILTIN_VECTORS = [
    # 1D
    cx.CartesianPos1D,
    cx.RadialPos,
    # 2D
    cx.CartesianPos2D,
    cx.PolarPos,
    # 3D
    cx.CartesianPos3D,
    cx.SphericalPos,
    cx.CylindricalPos,
]

BUILTIN_DIFFERENTIALS = [
    # 1D
    cx.CartesianVel1D,
    cx.RadialVel,
    # 2D
    cx.CartesianVel2D,
    cx.PolarVel,
    # LnPolarVel,
    # Log10PolarVel,
    # 3D
    cx.CartesianVel3D,
    cx.SphericalVel,
    cx.CylindricalVel,
]


def context_dimension_reduction(
    vector: cx.AbstractPos, target: type[cx.AbstractPos]
) -> AbstractContextManager[Any]:
    """Return a context manager that checks for dimensionality reduction."""
    context: AbstractContextManager[Any]
    if (
        isinstance(vector, cx.AbstractPos2D) and issubclass(target, cx.AbstractPos1D)
    ) or (
        isinstance(vector, cx.AbstractPos3D)
        and issubclass(target, cx.AbstractPos2D | cx.AbstractPos1D)
    ):
        context = pytest.warns(cx.IrreversibleDimensionChange)
    else:
        context = nullcontext()
    return context


class AbstractVectorTest:
    """Test :class:`coordinax.AbstractVector`."""

    # ===============================================================
    # Array

    def test_shape(self, vector):
        """Test :meth:`AbstractVector.shape`."""
        shape = vector.shape
        assert isinstance(shape, tuple)
        assert all(isinstance(s, int) for s in shape)
        assert shape == jnp.broadcast_shapes(
            *(getattr(vector, c).shape for c in vector.components)
        )

    def test_flatten(self, vector):
        """Test :meth:`AbstractVector.flatten`."""
        # Test input vector
        flat = vector.flatten()
        assert isinstance(flat, type(vector))
        assert all(
            jnp.array_equal(getattr(flat, c), getattr(vector, c).flatten())
            for c in vector.components
        )

        # Test an explicitly shaped vector
        vec = replace(
            vector,
            **{k: replace(v, value=jnp.ones((2, 4))) for k, v in field_items(vector)},
        )
        flat = vec.flatten()
        assert isinstance(flat, type(vec))
        assert all(
            jnp.array_equal(getattr(flat, c).value, jnp.ones(8)) for c in vec.components
        )

    def test_reshape(self, vector):
        """Test :meth:`AbstractVector.reshape`."""
        # Test input vector
        reshaped = vector.reshape(2, -1)
        assert isinstance(reshaped, type(vector))
        assert all(
            jnp.array_equal(getattr(reshaped, c), getattr(vector, c).reshape(2, -1))
            for c in vector.components
        )

        # Test an explicitly shaped vector
        vec = replace(
            vector,
            **{k: replace(v, value=jnp.ones((2, 4))) for k, v in field_items(vector)},
        )
        reshaped = vec.reshape(1, 8)
        assert isinstance(reshaped, type(vec))
        assert all(
            jnp.array_equal(getattr(reshaped, c).value, jnp.ones((1, 8)))
            for c in vec.components
        )

    # ===============================================================
    # Collection

    def test_asdict(self, vector):
        """Test :meth:`AbstractVector.asdict`."""
        # Simple test
        adict = vector.asdict()
        assert isinstance(adict, dict)
        for k, v in adict.items():
            assert isinstance(k, str)
            assert isinstance(v, AbstractQuantity)
            assert jnp.array_equal(v, getattr(vector, k))

        # Test with a different dict_factory
        adict = vector.asdict(dict_factory=UserDict)
        assert isinstance(adict, UserDict)
        assert all(jnp.array_equal(v, getattr(vector, k)) for k, v in adict.items())

    def test_components(self, vector):
        """Test :meth:`AbstractVector.components`."""
        # Simple test
        components = vector.components
        assert isinstance(components, tuple)
        assert all(isinstance(c, str) for c in components)
        assert all(hasattr(vector, c) for c in components)

    def test_shapes(self, vector):
        """Test :meth:`AbstractVector.shapes`."""
        # Simple test
        shapes = vector.shapes
        assert isinstance(shapes, MappingProxyType)
        assert set(shapes.keys()) == set(vector.components)
        assert all(v == getattr(vector, k).shape for k, v in shapes.items())


class AbstractPosTest(AbstractVectorTest):
    """Test :class:`coordinax.AbstractPos`."""

    @pytest.fixture(scope="class")
    def vector(self) -> cx.AbstractPos:
        """Return a vector."""
        raise NotImplementedError

    @pytest.mark.parametrize("target", BUILTIN_VECTORS)
    def test_represent_as(self, vector, target):
        """Test :meth:`AbstractPos.represent_as`.

        This just tests that the machiner works.
        """
        # Perform the conversion.
        # Detecting whether the conversion reduces the dimensionality.
        with context_dimension_reduction(vector, target):
            newvec = vector.represent_as(target)

        # Test
        assert isinstance(newvec, target)


class AbstractVelTest(AbstractVectorTest):
    """Test :class:`coordinax.AbstractVel`."""

    @pytest.fixture(scope="class")
    def vector(self) -> cx.AbstractPos:
        """Return a vector."""
        raise NotImplementedError

    @pytest.fixture(scope="class")
    def difntl(self) -> cx.AbstractVel:
        """Return a vector."""
        raise NotImplementedError

    @pytest.mark.parametrize("target", BUILTIN_DIFFERENTIALS)
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_represent_as(self, difntl, target, vector):
        """Test :meth:`AbstractPos.represent_as`.

        This just tests that the machiner works.
        """
        # TODO: have all the conversions
        if (
            (
                isinstance(difntl, cx.AbstractVel1D)
                and not issubclass(target, cx.AbstractVel1D)
            )
            or (
                isinstance(difntl, cx.AbstractVel2D)
                and not issubclass(target, cx.AbstractVel2D)
            )
            or (
                isinstance(difntl, cx.AbstractVel3D)
                and not issubclass(target, cx.AbstractVel3D)
            )
        ):
            pytest.xfail("Not implemented yet")

        # Perform the conversion.
        # Detecting whether the conversion reduces the dimensionality.
        with context_dimension_reduction(vector, target.integral_cls):
            newdif = difntl.represent_as(target, vector)

        # Test
        assert isinstance(newdif, target)
