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
    represent_as,
)
from vector._d1.builtin import CartesianDifferential1D
from vector._d2.builtin import CartesianDifferential2D
from vector._d3.builtin import (
    CartesianDifferential3D,
    CylindricalDifferential,
    SphericalDifferential,
)

from .test_base import AbstractVectorDifferentialTest, AbstractVectorTest


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
        # Jit can copy
        newvec = vector.represent_as(Cartesian3DVector)
        assert newvec == vector

        # The normal `represent_as` method should return the same object
        newvec = represent_as(vector, Cartesian3DVector)
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
        # Jit can copy
        newvec = vector.represent_as(SphericalVector)
        assert newvec == vector

        # The normal `represent_as` method should return the same object
        newvec = represent_as(vector, SphericalVector)
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
        # Jit can copy
        newvec = vector.represent_as(CylindricalVector)
        assert newvec == vector

        # The normal `represent_as` method should return the same object
        newvec = represent_as(vector, CylindricalVector)
        assert newvec is vector


class Abstract3DVectorDifferentialTest(AbstractVectorDifferentialTest):
    """Test :class:`vector.Abstract2DVectorDifferential`."""


class TestCartesianDifferential3D(Abstract3DVectorDifferentialTest):
    """Test :class:`vector.CartesianDifferential3D`."""

    @pytest.fixture(scope="class")
    def difntl(self) -> CartesianDifferential3D:
        """Return a differential."""
        return CartesianDifferential3D(
            d_x=Quantity([1, 2, 3, 4], u.km / u.s),
            d_y=Quantity([5, 6, 7, 8], u.km / u.s),
            d_z=Quantity([9, 10, 11, 12], u.km / u.s),
        )

    @pytest.fixture(scope="class")
    def vector(self) -> Cartesian3DVector:
        """Return a vector."""
        return Cartesian3DVector(
            x=Quantity([1, 2, 3, 4], u.kpc),
            y=Quantity([5, 6, 7, 8], u.kpc),
            z=Quantity([9, 10, 11, 12], u.kpc),
        )

    # ==========================================================================

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian3d_to_cartesian1d(self, difntl, vector):
        """Test ``vector.represent_as(Cartesian1DVector)``."""
        cart1d = difntl.represent_as(CartesianDifferential1D, vector)

        assert isinstance(cart1d, CartesianDifferential1D)
        assert cart1d.d_x == Quantity([1, 2, 3, 4], u.km / u.s)

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian3d_to_radial(self, difntl, vector):
        """Test ``vector.represent_as(RadialVector)``."""
        radial = difntl.represent_as(RadialVector, vector)

        assert isinstance(radial, RadialVector)
        assert radial.d_r == Quantity([1, 2, 3, 4], u.km / u.s)

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian3d_to_cartesian2d(self, difntl, vector):
        """Test ``vector.represent_as(Cartesian2DVector)``."""
        cart2d = difntl.represent_as(CartesianDifferential2D, vector)

        assert isinstance(cart2d, CartesianDifferential2D)
        assert cart2d.d_x == Quantity([1, 2, 3, 4], u.km / u.s)
        assert cart2d.d_y == Quantity([5, 6, 7, 8], u.km / u.s)

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian3d_to_polar(self, difntl, vector):
        """Test ``vector.represent_as(PolarVector)``."""
        polar = difntl.represent_as(PolarVector, vector)

        assert isinstance(polar, PolarVector)
        assert polar.d_r == Quantity([1, 2, 3, 4], u.km / u.s)
        assert polar.d_phi == Quantity([5, 6, 7, 8], u.mas / u.yr)

    def test_cartesian3d_to_cartesian3d(self, difntl, vector):
        """Test ``vector.represent_as(Cartesian3DVector)``."""
        # Jit can copy
        newvec = difntl.represent_as(CartesianDifferential3D, vector)
        assert newvec == difntl

        # The normal `represent_as` method should return the same object
        newvec = represent_as(difntl, CartesianDifferential3D, vector)
        assert newvec is difntl

    def test_cartesian3d_to_spherical(self, difntl, vector):
        """Test ``vector.represent_as(SphericalDifferential)``."""
        spherical = difntl.represent_as(SphericalDifferential, vector)

        assert isinstance(spherical, SphericalDifferential)
        assert spherical.d_r == Quantity([1, 2, 3, 4], u.km / u.s)
        assert spherical.d_phi == Quantity([5, 6, 7, 8], u.mas / u.yr)
        assert spherical.d_theta == Quantity([9, 10, 11, 12], u.mas / u.yr)

    def test_cartesian3d_to_cylindrical(self, difntl, vector):
        """Test ``vector.represent_as(CylindricalDifferential)``."""
        cylindrical = difntl.represent_as(CylindricalDifferential, vector)

        assert isinstance(cylindrical, CylindricalDifferential)
        assert cylindrical.d_rho == Quantity([1, 2, 3, 4], u.km / u.s)
        assert cylindrical.d_phi == Quantity([5, 6, 7, 8], u.mas / u.yr)
        assert cylindrical.d_z == Quantity([9, 10, 11, 12], u.km / u.s)


class TestSphericalDifferential(Abstract3DVectorDifferentialTest):
    """Test :class:`vector.SphericalDifferential`."""

    @pytest.fixture(scope="class")
    def difntl(self) -> SphericalDifferential:
        """Return a differential."""
        return SphericalDifferential(
            d_r=Quantity([1, 2, 3, 4], u.km / u.s),
            d_phi=Quantity([5, 6, 7, 8], u.mas / u.yr),
            d_theta=Quantity([9, 10, 11, 12], u.mas / u.yr),
        )

    @pytest.fixture(scope="class")
    def vector(self) -> SphericalVector:
        """Return a vector."""
        return SphericalVector(
            r=Quantity([1, 2, 3, 4], u.kpc),
            phi=Quantity([0, 1, 2, 3], u.rad),
            theta=Quantity([4, 5, 6, 7], u.rad),
        )

    # ==========================================================================

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_spherical_to_cartesian1d(self, difntl, vector):
        """Test ``vector.represent_as(Cartesian1DVector)``."""
        cart1d = difntl.represent_as(CartesianDifferential1D, vector)

        assert isinstance(cart1d, CartesianDifferential1D)
        assert cart1d.d_x == Quantity([1, 2, 3, 4], u.km / u.s)

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_spherical_to_radial(self, difntl, vector):
        """Test ``vector.represent_as(RadialVector)``."""
        radial = difntl.represent_as(RadialVector, vector)

        assert isinstance(radial, RadialVector)
        assert radial.d_r == Quantity([1, 2, 3, 4], u.km / u.s)

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_spherical_to_cartesian2d(self, difntl, vector):
        """Test ``vector.represent_as(Cartesian2DVector)``."""
        cart2d = difntl.represent_as(CartesianDifferential2D, vector)

        assert isinstance(cart2d, CartesianDifferential2D)
        assert cart2d.d_x == Quantity([1, 2, 3, 4], u.km / u.s)
        assert cart2d.d_y == Quantity([5, 6, 7, 8], u.km / u.s)

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_spherical_to_polar(self, difntl, vector):
        """Test ``vector.represent_as(PolarVector)``."""
        polar = difntl.represent_as(PolarVector, vector)

        assert isinstance(polar, PolarVector)
        assert polar.d_r == Quantity([1, 2, 3, 4], u.km / u.s)
        assert polar.d_phi == Quantity([5, 6, 7, 8], u.mas / u.yr)

    def test_spherical_to_cartesian3d(self, difntl, vector):
        """Test ``vector.represent_as(Cartesian3DVector)``."""
        cart3d = difntl.represent_as(CartesianDifferential3D, vector)

        assert isinstance(cart3d, CartesianDifferential3D)
        assert cart3d.d_x == Quantity([1, 2, 3, 4], u.km / u.s)
        assert cart3d.d_y == Quantity([5, 6, 7, 8], u.km / u.s)
        assert cart3d.d_z == Quantity([9, 10, 11, 12], u.km / u.s)

    def test_spherical_to_spherical(self, difntl, vector):
        """Test ``vector.represent_as(SphericalDifferential)``."""
        # Jit can copy
        newvec = difntl.represent_as(SphericalDifferential, vector)
        assert newvec == difntl

        # The normal `represent_as` method should return the same object
        newvec = represent_as(difntl, SphericalDifferential, vector)
        assert newvec is difntl

    def test_spherical_to_cylindrical(self, difntl, vector):
        """Test ``vector.represent_as(CylindricalDifferential)``."""
        cylindrical = difntl.represent_as(CylindricalDifferential, vector)

        assert isinstance(cylindrical, CylindricalDifferential)
        assert cylindrical.d_rho == Quantity([1, 2, 3, 4], u.km / u.s)
        assert cylindrical.d_phi == Quantity([5, 6, 7, 8], u.mas / u.yr)
        assert cylindrical.d_z == Quantity([9, 10, 11, 12], u.km / u.s)


class TestCylindricalDifferential(Abstract3DVectorDifferentialTest):
    """Test :class:`vector.CylindricalDifferential`."""

    @pytest.fixture(scope="class")
    def difntl(self) -> CylindricalDifferential:
        """Return a differential."""
        return CylindricalDifferential(
            d_rho=Quantity([1, 2, 3, 4], u.km / u.s),
            d_phi=Quantity([5, 6, 7, 8], u.mas / u.yr),
            d_z=Quantity([9, 10, 11, 12], u.km / u.s),
        )

    @pytest.fixture(scope="class")
    def vector(self) -> CylindricalVector:
        """Return a vector."""
        return CylindricalVector(
            rho=Quantity([1, 2, 3, 4], u.kpc),
            phi=Quantity([0, 1, 2, 3], u.rad),
            z=Quantity([9, 10, 11, 12], u.kpc),
        )

    # ==========================================================================

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cylindrical_to_cartesian1d(self, difntl, vector):
        """Test ``vector.represent_as(Cartesian1DVector)``."""
        cart1d = difntl.represent_as(CartesianDifferential1D, vector)

        assert isinstance(cart1d, CartesianDifferential1D)
        assert cart1d.d_x == Quantity([1, 2, 3, 4], u.km / u.s)

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cylindrical_to_radial(self, difntl, vector):
        """Test ``vector.represent_as(RadialVector)``."""
        radial = difntl.represent_as(RadialVector, vector)

        assert isinstance(radial, RadialVector)
        assert radial.d_r == Quantity([1, 2, 3, 4], u.km / u.s)

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cylindrical_to_cartesian2d(self, difntl, vector):
        """Test ``vector.represent_as(Cartesian2DVector)``."""
        cart2d = difntl.represent_as(CartesianDifferential2D, vector)

        assert isinstance(cart2d, CartesianDifferential2D)
        assert cart2d.d_x == Quantity([1, 2, 3, 4], u.km / u.s)
        assert cart2d.d_y == Quantity([5, 6, 7, 8], u.km / u.s)

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cylindrical_to_polar(self, difntl, vector):
        """Test ``vector.represent_as(PolarVector)``."""
        polar = difntl.represent_as(PolarVector, vector)

        assert isinstance(polar, PolarVector)
        assert polar.d_r == Quantity([1, 2, 3, 4], u.km / u.s)
        assert polar.d_phi == Quantity([5, 6, 7, 8], u.mas / u.yr)

    def test_cylindrical_to_cartesian3d(self, difntl, vector):
        """Test ``vector.represent_as(Cartesian3DVector)``."""
        cart3d = difntl.represent_as(CartesianDifferential3D, vector)

        assert isinstance(cart3d, CartesianDifferential3D)
        assert cart3d.d_x == Quantity([1, 2, 3, 4], u.km / u.s)
        assert cart3d.d_y == Quantity([5, 6, 7, 8], u.km / u.s)
        assert cart3d.d_z == Quantity([9, 10, 11, 12], u.km / u.s)

    def test_cylindrical_to_spherical(self, difntl, vector):
        """Test ``vector.represent_as(SphericalDifferential)``."""
        spherical = difntl.represent_as(SphericalDifferential, vector)

        assert isinstance(spherical, SphericalDifferential)
        assert spherical.d_r == Quantity([1, 2, 3, 4], u.km / u.s)
        assert spherical.d_phi == Quantity([5, 6, 7, 8], u.mas / u.yr)
        assert spherical.d_theta == Quantity([9, 10, 11, 12], u.mas / u.yr)

    def test_cylindrical_to_cylindrical(self, difntl, vector):
        """Test ``vector.represent_as(CylindricalDifferential)``."""
        # Jit can copy
        newvec = difntl.represent_as(CylindricalDifferential, vector)
        assert newvec == difntl

        # The normal `represent_as` method should return the same object
        newvec = represent_as(difntl, CylindricalDifferential, vector)
        assert newvec is difntl
