"""Test :mod:`vector._utils`."""

from contextlib import AbstractContextManager, nullcontext
from typing import Any

import pytest

from vector import (
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


class AbstractVectorDifferentialTest:
    """Test :class:`vector.AbstractVectorDifferential`."""

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
        with context_dimension_reduction(vector, target.vector_cls):
            newdif = difntl.represent_as(target, vector)

        # Test
        assert isinstance(newdif, target)
