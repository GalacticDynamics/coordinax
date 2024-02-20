"""Test :mod:`vector._builtin`."""

import astropy.units as u
import pytest
from jax_quantity import Quantity

from vector import (
    AbstractVector,
    Cartesian1DVector,
    Cartesian2DVector,
    Cartesian3DVector,
    CylindricalVector,
    PolarVector,
    RadialVector,
    SphericalVector,
)

from .test_base import AbstractVectorTest


class Abstract3DVectorTest(AbstractVectorTest):
    """Test :class:`vector.Abstract3DVector`."""


class TestCartesian3DVector:
    """Test :class:`vector.Cartesian3DVector`."""

    @pytest.fixture(scope="class")
    def vector(self) -> AbstractVector:
        """Return a vector."""
        from vector import Cartesian3DVector

        return Cartesian3DVector(
            x=Quantity([1, 2, 3, 4], u.kpc),
            y=Quantity([5, 6, 7, 8], u.km),
            z=Quantity([9, 10, 11, 12], u.m),
        )

    # ==========================================================================
    # represent_as

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cartesian3d_to_cartesian1d(self, vector):
        """Test ``vector.represent_as(Cartesian1DVector)``."""
        cart1d = vector.represent_as(Cartesian1DVector)

        assert isinstance(cart1d, Cartesian1DVector)
        assert cart1d.x == Quantity([1, 2, 3, 4], u.kpc)

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cartesian3d_to_radial(self, vector):
        """Test ``vector.represent_as(RadialVector)``."""
        radial = vector.represent_as(RadialVector)

        assert isinstance(radial, RadialVector)
        assert radial.r == Quantity([1, 2, 3, 4], u.kpc)

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cartesian3d_to_cartesian2d(self, vector):
        """Test ``vector.represent_as(Cartesian2DVector)``."""
        cart2d = vector.represent_as(Cartesian2DVector, y=Quantity([5, 6, 7, 8], u.km))

        assert isinstance(cart2d, Cartesian2DVector)
        assert cart2d.x == Quantity([1, 2, 3, 4], u.kpc)
        assert cart2d.y == Quantity([5, 6, 7, 8], u.km)

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cartesian3d_to_polar(self, vector):
        """Test ``vector.represent_as(PolarVector)``."""
        polar = vector.represent_as(PolarVector, phi=Quantity([0, 1, 2, 3], u.rad))

        assert isinstance(polar, PolarVector)
        assert polar.r == Quantity([1, 2, 3, 4], u.kpc)
        assert polar.phi == Quantity([0, 1, 2, 3], u.rad)

    # @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    # def test_cartesian3d_to_lnpolar(self, vector):
    #     """Test ``vector.represent_as(LnPolarVector)``."""
    #     assert False

    # @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    # def test_cartesian3d_to_log10polar(self, vector):
    #     """Test ``vector.represent_as(Log10PolarVector)``."""
    #     assert False

    def test_cartesian3d_to_cartesian3d(self, vector):
        """Test ``vector.represent_as(Cartesian3DVector)``."""
        newvec = vector.represent_as(Cartesian3DVector)
        assert newvec is vector

    def test_cartesian3d_to_spherical(self, vector):
        """Test ``vector.represent_as(SphericalVector)``."""
        spherical = vector.represent_as(
            SphericalVector,
            phi=Quantity([0, 1, 2, 3], u.rad),
            theta=Quantity([4, 5, 6, 7], u.rad),
        )

        assert isinstance(spherical, SphericalVector)
        assert spherical.r == Quantity([1, 2, 3, 4], u.kpc)
        assert spherical.phi == Quantity([0, 1, 2, 3], u.rad)
        assert spherical.theta == Quantity([4, 5, 6, 7], u.rad)

    def test_cartesian3d_to_cylindrical(self, vector):
        """Test ``vector.represent_as(CylindricalVector)``."""
        cylindrical = vector.represent_as(
            CylindricalVector,
            phi=Quantity([0, 1, 2, 3], u.rad),
            z=Quantity([4, 5, 6, 7], u.m),
        )

        assert isinstance(cylindrical, CylindricalVector)
        assert cylindrical.rho == Quantity([1, 2, 3, 4], u.kpc)
        assert cylindrical.phi == Quantity([0, 1, 2, 3], u.rad)
        assert cylindrical.z == Quantity([4, 5, 6, 7], u.m)


class TestSphericalVector:
    """Test :class:`vector.SphericalVector`."""

    @pytest.fixture(scope="class")
    def vector(self) -> AbstractVector:
        """Return a vector."""
        from vector import SphericalVector

        return SphericalVector(
            r=Quantity([1, 2, 3, 4], u.kpc),
            phi=Quantity([0, 1, 2, 3], u.rad),
            theta=Quantity([4, 5, 6, 7], u.rad),
        )

    # ==========================================================================
    # represent_as

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_spherical_to_cartesian1d(self, vector):
        """Test ``vector.represent_as(Cartesian1DVector)``."""
        cart1d = vector.represent_as(Cartesian1DVector)

        assert isinstance(cart1d, Cartesian1DVector)
        assert cart1d.x == Quantity([1, 2, 3, 4], u.kpc)

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_spherical_to_radial(self, vector):
        """Test ``vector.represent_as(RadialVector)``."""
        radial = vector.represent_as(RadialVector)

        assert isinstance(radial, RadialVector)
        assert radial.r == Quantity([1, 2, 3, 4], u.kpc)

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_spherical_to_cartesian2d(self, vector):
        """Test ``vector.represent_as(Cartesian2DVector)``."""
        cart2d = vector.represent_as(Cartesian2DVector, y=Quantity([5, 6, 7, 8], u.km))

        assert isinstance(cart2d, Cartesian2DVector)
        assert cart2d.x == Quantity([1, 2, 3, 4], u.kpc)
        assert cart2d.y == Quantity([5, 6, 7, 8], u.km)

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_spherical_to_polar(self, vector):
        """Test ``vector.represent_as(PolarVector)``."""
        polar = vector.represent_as(PolarVector, phi=Quantity([0, 1, 2, 3], u.rad))

        assert isinstance(polar, PolarVector)
        assert polar.r == Quantity([1, 2, 3, 4], u.kpc)
        assert polar.phi == Quantity([0, 1, 2, 3], u.rad)

    # @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    # def test_spherical_to_lnpolar(self, vector):
    #     """Test ``vector.represent_as(LnPolarVector)``."""
    #     assert False

    # @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    # def test_spherical_to_log10polar(self, vector):
    #     """Test ``vector.represent_as(Log10PolarVector)``."""
    #     assert False

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_spherical_to_cartesian3d(self, vector):
        """Test ``vector.represent_as(Cartesian3DVector)``."""
        cart3d = vector.represent_as(
            Cartesian3DVector, z=Quantity([9, 10, 11, 12], u.m)
        )

        assert isinstance(cart3d, Cartesian3DVector)
        assert cart3d.x == Quantity([1, 2, 3, 4], u.kpc)
        assert cart3d.y == Quantity([5, 6, 7, 8], u.km)
        assert cart3d.z == Quantity([9, 10, 11, 12], u.m)

    def test_spherical_to_spherical(self, vector):
        """Test ``vector.represent_as(SphericalVector)``."""
        newvec = vector.represent_as(SphericalVector)
        assert newvec is vector

    def test_spherical_to_cylindrical(self, vector):
        """Test ``vector.represent_as(CylindricalVector)``."""
        cylindrical = vector.represent_as(
            CylindricalVector, z=Quantity([9, 10, 11, 12], u.m)
        )

        assert isinstance(cylindrical, CylindricalVector)
        assert cylindrical.rho == Quantity([1, 2, 3, 4], u.kpc)
        assert cylindrical.phi == Quantity([0, 1, 2, 3], u.rad)
        assert cylindrical.z == Quantity([9, 10, 11, 12], u.m)


class TestCylindricalVector:
    """Test :class:`vector.CylindricalVector`."""

    @pytest.fixture(scope="class")
    def vector(self) -> AbstractVector:
        """Return a vector."""
        from vector import CylindricalVector

        return CylindricalVector(
            rho=Quantity([1, 2, 3, 4], u.kpc),
            phi=Quantity([0, 1, 2, 3], u.rad),
            z=Quantity([9, 10, 11, 12], u.m),
        )

    # ==========================================================================
    # represent_as

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cylindrical_to_cartesian1d(self, vector):
        """Test ``vector.represent_as(Cartesian1DVector)``."""
        cart1d = vector.represent_as(Cartesian1DVector)

        assert isinstance(cart1d, Cartesian1DVector)
        assert cart1d.x == Quantity([1, 2, 3, 4], u.kpc)

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cylindrical_to_radial(self, vector):
        """Test ``vector.represent_as(RadialVector)``."""
        radial = vector.represent_as(RadialVector)

        assert isinstance(radial, RadialVector)
        assert radial.r == Quantity([1, 2, 3, 4], u.kpc)

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cylindrical_to_cartesian2d(self, vector):
        """Test ``vector.represent_as(Cartesian2DVector)``."""
        cart2d = vector.represent_as(Cartesian2DVector, y=Quantity([5, 6, 7, 8], u.km))

        assert isinstance(cart2d, Cartesian2DVector)
        assert cart2d.x == Quantity([1, 2, 3, 4], u.kpc)
        assert cart2d.y == Quantity([5, 6, 7, 8], u.km)

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cylindrical_to_polar(self, vector):
        """Test ``vector.represent_as(PolarVector)``."""
        polar = vector.represent_as(PolarVector, phi=Quantity([0, 1, 2, 3], u.rad))

        assert isinstance(polar, PolarVector)
        assert polar.r == Quantity([1, 2, 3, 4], u.kpc)
        assert polar.phi == Quantity([0, 1, 2, 3], u.rad)

    # @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    # def test_cylindrical_to_lnpolar(self, vector):
    #     """Test ``vector.represent_as(LnPolarVector)``."""
    #     assert False

    # @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    # def test_cylindrical_to_log10polar(self, vector):
    #     """Test ``vector.represent_as(Log10PolarVector)``."""
    #     assert False

    def test_cylindrical_to_cartesian3d(self, vector):
        """Test ``vector.represent_as(Cartesian3DVector)``."""
        cart3d = vector.represent_as(Cartesian3DVector, y=Quantity([5, 6, 7, 8], u.km))

        assert isinstance(cart3d, Cartesian3DVector)
        assert cart3d.x == Quantity([1, 2, 3, 4], u.kpc)
        assert cart3d.y == Quantity([5, 6, 7, 8], u.km)
        assert cart3d.z == Quantity([9, 10, 11, 12], u.m)

    def test_cylindrical_to_spherical(self, vector):
        """Test ``vector.represent_as(SphericalVector)``."""
        spherical = vector.represent_as(
            SphericalVector, theta=Quantity([4, 5, 6, 7], u.rad)
        )

        assert isinstance(spherical, SphericalVector)
        assert spherical.r == Quantity([1, 2, 3, 4], u.kpc)
        assert spherical.phi == Quantity([0, 1, 2, 3], u.rad)
        assert spherical.theta == Quantity([4, 5, 6, 7], u.rad)

    def test_cylindrical_to_cylindrical(self, vector):
        """Test ``vector.represent_as(CylindricalVector)``."""
        newvec = vector.represent_as(CylindricalVector)
        assert newvec is vector
