"""Test :mod:`vector._utils`."""

from contextlib import AbstractContextManager, nullcontext

import pytest

from vector import (
    Abstract1DVector,
    Abstract2DVector,
    Abstract3DVector,
    AbstractVector,
    Cartesian1DVector,
    Cartesian2DVector,
    Cartesian3DVector,
    CylindricalVector,
    IrreversibleDimensionChange,
    # LnPolarVector,
    # Log10PolarVector,
    PolarVector,
    RadialVector,
    SphericalVector,
)

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


def context_dimension_reduction(vector, target) -> AbstractContextManager:
    context: AbstractContextManager
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


class AbstractVectorTest:
    """Test :class:`vector.AbstractVector`."""

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


class Abstract1DVectorTest(AbstractVectorTest):
    """Test :class:`vector.Abstract1DVector`."""


class Abstract2DVectorTest(AbstractVectorTest):
    """Test :class:`vector.Abstract2DVector`."""


class Abstract3DVectorTest(AbstractVectorTest):
    """Test :class:`vector.Abstract3DVector`."""
