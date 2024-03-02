"""Test :mod:`coordinax._utils`."""

from collections import UserDict
from contextlib import AbstractContextManager, nullcontext
from dataclasses import replace
from types import MappingProxyType
from typing import Any

import jax.numpy as jnp
import pytest
from quax import quaxify

import array_api_jax_compat as xp
from jax_quantity import Quantity

from coordinax import (
    Abstract1DVector,
    Abstract1DVectorDifferential,
    Abstract2DVector,
    Abstract2DVectorDifferential,
    Abstract3DVector,
    Abstract3DVectorDifferential,
    AbstractVector,
    AbstractVectorDifferential,
    Cartesian1DVector,
    Cartesian2DVector,
    Cartesian3DVector,
    CartesianDifferential1D,
    CartesianDifferential2D,
    CartesianDifferential3D,
    CylindricalDifferential,
    CylindricalVector,
    IrreversibleDimensionChange,
    PolarDifferential,
    PolarVector,
    RadialDifferential,
    RadialVector,
    SphericalDifferential,
    SphericalVector,
)
from coordinax._utils import dataclass_items

BUILTIN_VECTORS = [
    # 1D
    Cartesian1DVector,
    RadialVector,
    # 2D
    Cartesian2DVector,
    PolarVector,
    # LnPolarVector,
    # Log10PolarVector,
    # 3D
    Cartesian3DVector,
    SphericalVector,
    CylindricalVector,
]

BUILTIN_DIFFERENTIALS = [
    # 1D
    CartesianDifferential1D,
    RadialDifferential,
    # 2D
    CartesianDifferential2D,
    PolarDifferential,
    # LnPolarDifferential,
    # Log10PolarDifferential,
    # 3D
    CartesianDifferential3D,
    SphericalDifferential,
    CylindricalDifferential,
]

array_equal = quaxify(jnp.array_equal)


def context_dimension_reduction(
    vector: AbstractVector, target: type[AbstractVector]
) -> AbstractContextManager[Any]:
    """Return a context manager that checks for dimensionality reduction."""
    context: AbstractContextManager[Any]
    if (
        isinstance(vector, Abstract2DVector) and issubclass(target, Abstract1DVector)
    ) or (
        isinstance(vector, Abstract3DVector)
        and issubclass(target, Abstract2DVector | Abstract1DVector)
    ):
        context = pytest.warns(IrreversibleDimensionChange)
    else:
        context = nullcontext()
    return context


class AbstractVectorBaseTest:
    """Test :class:`coordinax.AbstractVectorBase`."""

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
            array_equal(getattr(flat, c), getattr(vector, c).flatten())
            for c in vector.components
        )

        # Test an explicitly shaped vector
        vec = replace(
            vector,
            **{
                k: replace(v, value=xp.ones((2, 4))) for k, v in dataclass_items(vector)
            },
        )
        flat = vec.flatten()
        assert isinstance(flat, type(vec))
        assert all(
            array_equal(getattr(flat, c).value, xp.ones(8)) for c in vec.components
        )

    def test_reshape(self, vector):
        """Test :meth:`AbstractVector.reshape`."""
        # Test input vector
        reshaped = vector.reshape(2, -1)
        assert isinstance(reshaped, type(vector))
        assert all(
            array_equal(getattr(reshaped, c), getattr(vector, c).reshape(2, -1))
            for c in vector.components
        )

        # Test an explicitly shaped vector
        vec = replace(
            vector,
            **{
                k: replace(v, value=xp.ones((2, 4))) for k, v in dataclass_items(vector)
            },
        )
        reshaped = vec.reshape(1, 8)
        assert isinstance(reshaped, type(vec))
        assert all(
            array_equal(getattr(reshaped, c).value, xp.ones((1, 8)))
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
            assert isinstance(v, Quantity)
            assert array_equal(v, getattr(vector, k))

        # Test with a different dict_factory
        adict = vector.asdict(dict_factory=UserDict)
        assert isinstance(adict, UserDict)
        assert all(array_equal(v, getattr(vector, k)) for k, v in adict.items())

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


class AbstractVectorTest(AbstractVectorBaseTest):
    """Test :class:`coordinax.AbstractVector`."""

    @pytest.fixture(scope="class")
    def vector(self) -> AbstractVector:  # noqa: PT004
        """Return a vector."""
        raise NotImplementedError

    @pytest.mark.parametrize("target", BUILTIN_VECTORS)
    def test_represent_as(self, vector, target):
        """Test :meth:`AbstractVector.represent_as`.

        This just tests that the machiner works.
        """
        # Perform the conversion.
        # Detecting whether the conversion reduces the dimensionality.
        with context_dimension_reduction(vector, target):
            newvec = vector.represent_as(target)

        # Test
        assert isinstance(newvec, target)


class AbstractVectorDifferentialTest(AbstractVectorBaseTest):
    """Test :class:`coordinax.AbstractVectorDifferential`."""

    @pytest.fixture(scope="class")
    def vector(self) -> AbstractVector:  # noqa: PT004
        """Return a vector."""
        raise NotImplementedError

    @pytest.fixture(scope="class")
    def difntl(self) -> AbstractVectorDifferential:  # noqa: PT004
        """Return a vector."""
        raise NotImplementedError

    @pytest.mark.parametrize("target", BUILTIN_DIFFERENTIALS)
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_represent_as(self, difntl, target, vector):
        """Test :meth:`AbstractVector.represent_as`.

        This just tests that the machiner works.
        """
        # TODO: have all the convertsions
        if (
            (
                isinstance(difntl, Abstract1DVectorDifferential)
                and not issubclass(target, Abstract1DVectorDifferential)
            )
            or (
                isinstance(difntl, Abstract2DVectorDifferential)
                and not issubclass(target, Abstract2DVectorDifferential)
            )
            or (
                isinstance(difntl, Abstract3DVectorDifferential)
                and not issubclass(target, Abstract3DVectorDifferential)
            )
        ):
            pytest.xfail("Not implemented yet")

        # Perform the conversion.
        # Detecting whether the conversion reduces the dimensionality.
        with context_dimension_reduction(vector, target.integral_cls):
            newdif = difntl.represent_as(target, vector)

        # Test
        assert isinstance(newdif, target)
