"""Test :mod:`vector._d1`."""

import astropy.units as u
import pytest
from jax_quantity import Quantity

from vector import (
    AbstractVector,
    Cartesian1DVector,
    Cartesian2DVector,
    Cartesian3DVector,
    CartesianDifferential1D,
    CartesianDifferential2D,
    CartesianDifferential3D,
    CylindricalDifferential,
    CylindricalVector,
    PolarDifferential,
    PolarVector,
    RadialDifferential,
    RadialVector,
    SphericalDifferential,
    SphericalVector,
)

from .test_base import AbstractVectorDifferentialTest, AbstractVectorTest


class Abstract1DVectorTest(AbstractVectorTest):
    """Test :class:`vector.Abstract1DVector`."""


class TestCartesian1DVector(Abstract1DVectorTest):
    """Test :class:`vector.Cartesian1DVector`."""

    @pytest.fixture(scope="class")
    def vector(self) -> AbstractVector:
        """Return a vector."""
        from vector import Cartesian1DVector

        return Cartesian1DVector(x=Quantity([1, 2, 3, 4], u.kpc))

    # ==========================================================================
    # represent_as

    def test_cartesian1d_to_cartesian1d(self, vector):
        """Test ``vector.represent_as(Cartesian1DVector)``."""
        newvec = vector.represent_as(Cartesian1DVector)
        assert newvec is vector

    def test_cartesian1d_to_radial(self, vector):
        """Test ``vector.represent_as(RadialVector)``."""
        radial = vector.represent_as(RadialVector)

        assert isinstance(radial, RadialVector)
        assert radial.r == Quantity([1, 2, 3, 4], u.kpc)

    def test_cartesian1d_to_cartesian2d(self, vector):
        """Test ``vector.represent_as(Cartesian2DVector)``."""
        cart2d = vector.represent_as(Cartesian2DVector, y=Quantity([5, 6, 7, 8], u.km))

        assert isinstance(cart2d, Cartesian2DVector)
        assert cart2d.x == Quantity([1, 2, 3, 4], u.kpc)
        assert cart2d.y == Quantity([5, 6, 7, 8], u.km)

    def test_cartesian1d_to_polar(self, vector):
        """Test ``vector.represent_as(PolarVector)``."""
        polar = vector.represent_as(PolarVector, phi=Quantity([0, 1, 2, 3], u.rad))

        assert isinstance(polar, PolarVector)
        assert polar.r == Quantity([1, 2, 3, 4], u.kpc)
        assert polar.phi == Quantity([0, 1, 2, 3], u.rad)

    # def test_cartesian1d_to_lnpolar(self, vector):
    #     """Test ``vector.represent_as(LnPolarVector)``."""
    #     lnpolar = vector.to_lnpolar(phi=Quantity([0, 1, 2, 3], u.rad))

    #     assert isinstance(lnpolar, LnPolarVector)
    #     assert lnpolar.lnr == xp.log(Quantity([1, 2, 3, 4], u.kpc))
    #     assert lnpolar.phi == Quantity([0, 1, 2, 3], u.rad)

    # def test_cartesian1d_to_log10polar(self, vector):
    #     """Test ``vector.represent_as(Log10PolarVector)``."""
    #     log10polar = vector.to_log10polar(phi=Quantity([0, 1, 2, 3], u.rad))

    #     assert isinstance(log10polar, Log10PolarVector)
    #     assert log10polar.log10r == xp.log10(Quantity([1, 2, 3, 4], u.kpc))
    #     assert log10polar.phi == Quantity([0, 1, 2, 3], u.rad)

    def test_cartesian1d_to_cartesian3d(self, vector):
        """Test ``vector.represent_as(Cartesian3DVector)``."""
        cart3d = vector.represent_as(
            Cartesian3DVector,
            y=Quantity([5, 6, 7, 8], u.km),
            z=Quantity([9, 10, 11, 12], u.m),
        )

        assert isinstance(cart3d, Cartesian3DVector)
        assert cart3d.x == Quantity([1, 2, 3, 4], u.kpc)
        assert cart3d.y == Quantity([5, 6, 7, 8], u.km)
        assert cart3d.z == Quantity([9, 10, 11, 12], u.m)

    def test_cartesian1d_to_spherical(self, vector):
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

    def test_cartesian1d_to_cylindrical(self, vector):
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


class TestRadialVector(Abstract1DVectorTest):
    """Test :class:`vector.RadialVector`."""

    @pytest.fixture(scope="class")
    def vector(self) -> AbstractVector:
        """Return a vector."""
        from vector import RadialVector

        return RadialVector(r=Quantity([1, 2, 3, 4], u.kpc))

    # ==========================================================================
    # represent_as

    def test_radial_to_cartesian1d(self, vector):
        """Test ``vector.represent_as(Cartesian1DVector)``."""
        cart1d = vector.represent_as(Cartesian1DVector)

        assert isinstance(cart1d, Cartesian1DVector)
        assert cart1d.x == Quantity([1, 2, 3, 4], u.kpc)

    def test_radial_to_radial(self, vector):
        """Test ``vector.represent_as(RadialVector)``."""
        newvec = vector.represent_as(RadialVector)
        assert newvec is vector

    def test_radial_to_cartesian2d(self, vector):
        """Test ``vector.represent_as(Cartesian2DVector)``."""
        cart2d = vector.represent_as(Cartesian2DVector, y=Quantity([5, 6, 7, 8], u.km))

        assert isinstance(cart2d, Cartesian2DVector)
        assert cart2d.x == Quantity([1, 2, 3, 4], u.kpc)
        assert cart2d.y == Quantity([5, 6, 7, 8], u.km)

    def test_radial_to_polar(self, vector):
        """Test ``vector.represent_as(PolarVector)``."""
        polar = vector.represent_as(PolarVector, phi=Quantity([0, 1, 2, 3], u.rad))

        assert isinstance(polar, PolarVector)
        assert polar.r == Quantity([1, 2, 3, 4], u.kpc)
        assert polar.phi == Quantity([0, 1, 2, 3], u.rad)

    # def test_radial_to_lnpolar(self, vector):
    #     """Test ``vector.represent_as(LnPolarVector)``."""
    #     assert False

    # def test_radial_to_log10polar(self, vector):
    #     """Test ``vector.represent_as(Log10PolarVector)``."""
    #     assert False

    def test_radial_to_cartesian3d(self, vector):
        """Test ``vector.represent_as(Cartesian3DVector)``."""
        cart3d = vector.represent_as(
            Cartesian3DVector,
            y=Quantity([5, 6, 7, 8], u.km),
            z=Quantity([9, 10, 11, 12], u.m),
        )

        assert isinstance(cart3d, Cartesian3DVector)
        assert cart3d.x == Quantity([1, 2, 3, 4], u.kpc)
        assert cart3d.y == Quantity([5, 6, 7, 8], u.km)
        assert cart3d.z == Quantity([9, 10, 11, 12], u.m)

    def test_radial_to_spherical(self, vector):
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

    def test_radial_to_cylindrical(self, vector):
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


class Abstract1DVectorDifferentialTest(AbstractVectorDifferentialTest):
    """Test :class:`vector.Abstract1DVectorDifferential`."""


class TestCartesianDifferential1D(Abstract1DVectorDifferentialTest):
    """Test :class:`vector.CartesianDifferential1D`."""

    @pytest.fixture(scope="class")
    def difntl(self) -> CartesianDifferential1D:
        """Return a vector."""
        return CartesianDifferential1D(d_x=Quantity([1.0, 2, 3, 4], u.km / u.s))

    @pytest.fixture(scope="class")
    def vector(self) -> Cartesian1DVector:
        """Return a vector."""
        return Cartesian1DVector(x=Quantity([1.0, 2, 3, 4], u.kpc))

    # ==========================================================================
    # represent_as

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_cartesian1d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianDifferential1D)``."""
        newvec = difntl.represent_as(CartesianDifferential1D, vector)
        assert newvec is difntl

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_radial(self, difntl, vector):
        """Test ``difntl.represent_as(RadialDifferential)``."""
        radial = difntl.represent_as(RadialDifferential, vector)

        assert isinstance(radial, RadialDifferential)
        assert radial.d_r == Quantity([1, 2, 3, 4], u.km / u.s)

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_cartesian2d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianDifferential2D)``."""
        cart2d = difntl.represent_as(
            CartesianDifferential2D, vector, dy=Quantity([5, 6, 7, 8], u.km)
        )

        assert isinstance(cart2d, CartesianDifferential2D)
        assert cart2d.d_x == Quantity([1, 2, 3, 4], u.km / u.s)
        assert cart2d.d_y == Quantity([5, 6, 7, 8], u.km / u.s)

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_polar(self, difntl, vector):
        """Test ``difntl.represent_as(PolarDifferential)``."""
        polar = difntl.represent_as(
            PolarDifferential, vector, dphi=Quantity([0, 1, 2, 3], u.rad)
        )

        assert isinstance(polar, PolarDifferential)
        assert polar.d_r == Quantity([1, 2, 3, 4], u.km / u.s)
        assert polar.d_phi == Quantity([0, 1, 2, 3], u.rad / u.s)

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_cartesian3d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianDifferential3D)``."""
        cart3d = difntl.represent_as(
            CartesianDifferential3D,
            vector,
            dy=Quantity([5, 6, 7, 8], u.km),
            dz=Quantity([9, 10, 11, 12], u.m),
        )

        assert isinstance(cart3d, CartesianDifferential3D)
        assert cart3d.d_x == Quantity([1, 2, 3, 4], u.kpc)
        assert cart3d.d_y == Quantity([5, 6, 7, 8], u.km)
        assert cart3d.d_z == Quantity([9, 10, 11, 12], u.m)

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_spherical(self, difntl, vector):
        """Test ``difntl.represent_as(SphericalDifferential)``."""
        spherical = difntl.represent_as(
            SphericalDifferential,
            vector,
            dphi=Quantity([0, 1, 2, 3], u.rad),
            dtheta=Quantity([4, 5, 6, 7], u.rad),
        )

        assert isinstance(spherical, SphericalDifferential)
        assert spherical.d_r == Quantity([1, 2, 3, 4], u.kpc)
        assert spherical.d_phi == Quantity([0, 1, 2, 3], u.rad)
        assert spherical.dtheta == Quantity([4, 5, 6, 7], u.rad)

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_cylindrical(self, difntl, vector):
        """Test ``difntl.represent_as(CylindricalDifferential)``."""
        cylindrical = difntl.represent_as(
            CylindricalDifferential,
            vector,
            dphi=Quantity([0, 1, 2, 3], u.rad),
            dz=Quantity([4, 5, 6, 7], u.m),
        )

        assert isinstance(cylindrical, CylindricalDifferential)
        assert cylindrical.d_rho == Quantity([1, 2, 3, 4], u.kpc)
        assert cylindrical.d_phi == Quantity([0, 1, 2, 3], u.rad)
        assert cylindrical.d_z == Quantity([4, 5, 6, 7], u.m)


class TestRadialDifferential(Abstract1DVectorDifferentialTest):
    """Test :class:`vector.RadialDifferential`."""

    @pytest.fixture(scope="class")
    def difntl(self) -> RadialDifferential:
        """Return a vector."""
        return RadialDifferential(d_r=Quantity([1, 2, 3, 4], u.km / u.s))

    @pytest.fixture(scope="class")
    def vector(self) -> RadialVector:
        """Return a vector."""
        return RadialVector(r=Quantity([1, 2, 3, 4], u.kpc))

    # ==========================================================================
    # represent_as

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_cartesian1d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianDifferential1D)``."""
        cart1d = difntl.represent_as(CartesianDifferential1D, vector)

        assert isinstance(cart1d, CartesianDifferential1D)
        assert cart1d.d_x == Quantity([1, 2, 3, 4], u.km / u.s)

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_radial(self, difntl, vector):
        """Test ``difntl.represent_as(RadialDifferential)``."""
        newvec = difntl.represent_as(RadialDifferential, vector)
        assert newvec is difntl

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_cartesian2d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianDifferential2D)``."""
        cart2d = difntl.represent_as(
            CartesianDifferential2D, vector, dy=Quantity([5, 6, 7, 8], u.km)
        )

        assert isinstance(cart2d, CartesianDifferential2D)
        assert cart2d.d_x == Quantity([1, 2, 3, 4], u.kpc)
        assert cart2d.d_y == Quantity([5, 6, 7, 8], u.km)

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_polar(self, difntl, vector):
        """Test ``difntl.represent_as(PolarDifferential)``."""
        polar = difntl.represent_as(
            PolarDifferential, vector, dphi=Quantity([0, 1, 2, 3], u.rad)
        )

        assert isinstance(polar, PolarDifferential)
        assert polar.d_r == Quantity([1, 2, 3, 4], u.kpc)
        assert polar.d_phi == Quantity([0, 1, 2, 3], u.rad)

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_cartesian3d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianDifferential3D)``."""
        cart3d = difntl.represent_as(
            CartesianDifferential3D,
            vector,
            dy=Quantity([5, 6, 7, 8], u.km),
            dz=Quantity([9, 10, 11, 12], u.m),
        )

        assert isinstance(cart3d, CartesianDifferential3D)
        assert cart3d.d_x == Quantity([1, 2, 3, 4], u.kpc)
        assert cart3d.d_y == Quantity([5, 6, 7, 8], u.km)
        assert cart3d.d_z == Quantity([9, 10, 11, 12], u.m)

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_spherical(self, difntl, vector):
        """Test ``difntl.represent_as(SphericalDifferential)``."""
        spherical = difntl.represent_as(
            SphericalDifferential,
            vector,
            dphi=Quantity([0, 1, 2, 3], u.rad),
            dtheta=Quantity([4, 5, 6, 7], u.rad),
        )

        assert isinstance(spherical, SphericalDifferential)
        assert spherical.d_r == Quantity([1, 2, 3, 4], u.kpc)
        assert spherical.d_phi == Quantity([0, 1, 2, 3], u.rad)
        assert spherical.dtheta == Quantity([4, 5, 6, 7], u.rad)

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_cylindrical(self, difntl, vector):
        """Test ``difntl.represent_as(CylindricalDifferential)``."""
        cylindrical = difntl.represent_as(
            CylindricalDifferential,
            vector,
            dphi=Quantity([0, 1, 2, 3], u.rad),
            dz=Quantity([4, 5, 6, 7], u.m),
        )

        assert isinstance(cylindrical, CylindricalDifferential)
        assert cylindrical.d_rho == Quantity([1, 2, 3, 4], u.kpc)
        assert cylindrical.d_phi == Quantity([0, 1, 2, 3], u.rad)
        assert cylindrical.d_z == Quantity([4, 5, 6, 7], u.m)