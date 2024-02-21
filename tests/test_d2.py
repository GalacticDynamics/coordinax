"""Test :mod:`vector._d2`."""

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
from vector._d1.builtin import CartesianDifferential1D, RadialDifferential
from vector._d2.builtin import CartesianDifferential2D, PolarDifferential
from vector._d3.builtin import (
    CartesianDifferential3D,
    CylindricalDifferential,
    SphericalDifferential,
)

from .test_base import AbstractVectorDifferentialTest, AbstractVectorTest


class Abstract2DVectorTest(AbstractVectorTest):
    """Test :class:`vector.Abstract2DVector`."""


class TestCartesian2DVector:
    """Test :class:`vector.Cartesian2DVector`."""

    @pytest.fixture(scope="class")
    def vector(self) -> AbstractVector:
        """Return a vector."""
        from vector import Cartesian2DVector

        return Cartesian2DVector(
            x=Quantity([1, 2, 3, 4], u.kpc), y=Quantity([5, 6, 7, 8], u.km)
        )

    # ==========================================================================
    # represent_as

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cartesian2d_to_cartesian1d(self, vector):
        """Test ``vector.represent_as(Cartesian1DVector)``."""
        cart1d = vector.represent_as(Cartesian1DVector)

        assert isinstance(cart1d, Cartesian1DVector)
        assert cart1d.x == Quantity([1, 2, 3, 4], u.kpc)

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cartesian2d_to_radial(self, vector):
        """Test ``vector.represent_as(RadialVector)``."""
        radial = vector.represent_as(RadialVector)

        assert isinstance(radial, RadialVector)
        assert radial.r == Quantity([1, 2, 3, 4], u.kpc)

    def test_cartesian2d_to_cartesian2d(self, vector):
        """Test ``vector.represent_as(Cartesian2DVector)``."""
        # Jit can copy
        newvec = vector.represent_as(Cartesian2DVector)
        assert newvec == vector

        # The normal `represent_as` method should return the same object
        newvec = represent_as(vector, Cartesian2DVector)
        assert newvec is vector

    def test_cartesian2d_to_polar(self, vector):
        """Test ``vector.represent_as(PolarVector)``."""
        polar = vector.represent_as(PolarVector, phi=Quantity([0, 1, 2, 3], u.rad))

        assert isinstance(polar, PolarVector)
        assert polar.r == Quantity([1, 2, 3, 4], u.kpc)
        assert polar.phi == Quantity([0, 1, 2, 3], u.rad)

    # def test_cartesian2d_to_lnpolar(self, vector):
    #     """Test ``vector.represent_as(LnPolarVector)``."""
    #     assert False

    # def test_cartesian2d_to_log10polar(self, vector):
    #     """Test ``vector.represent_as(Log10PolarVector)``."""
    #    assert False

    def test_cartesian2d_to_cartesian3d(self, vector):
        """Test ``vector.represent_as(Cartesian3DVector)``."""
        cart3d = vector.represent_as(
            Cartesian3DVector, z=Quantity([9, 10, 11, 12], u.m)
        )

        assert isinstance(cart3d, Cartesian3DVector)
        assert cart3d.x == Quantity([1, 2, 3, 4], u.kpc)
        assert cart3d.y == Quantity([5, 6, 7, 8], u.km)
        assert cart3d.z == Quantity([9, 10, 11, 12], u.m)

    def test_cartesian2d_to_spherical(self, vector):
        """Test ``vector.represent_as(SphericalVector)``."""
        spherical = vector.represent_as(
            SphericalVector, theta=Quantity([4, 5, 6, 7], u.rad)
        )

        assert isinstance(spherical, SphericalVector)
        assert spherical.r == Quantity([1, 2, 3, 4], u.kpc)
        assert spherical.phi == Quantity([5, 6, 7, 8], u.rad)
        assert spherical.theta == Quantity([4, 5, 6, 7], u.rad)

    def test_cartesian2d_to_cylindrical(self, vector):
        """Test ``vector.represent_as(CylindricalVector)``."""
        cylindrical = vector.represent_as(
            CylindricalVector, z=Quantity([9, 10, 11, 12], u.m)
        )

        assert isinstance(cylindrical, CylindricalVector)
        assert cylindrical.rho == Quantity([1, 2, 3, 4], u.kpc)
        assert cylindrical.phi == Quantity([5, 6, 7, 8], u.rad)
        assert cylindrical.z == Quantity([9, 10, 11, 12], u.m)


class TestPolarVector:
    """Test :class:`vector.PolarVector`."""

    @pytest.fixture(scope="class")
    def vector(self) -> AbstractVector:
        """Return a vector."""
        from vector import PolarVector

        return PolarVector(
            r=Quantity([1, 2, 3, 4], u.kpc), phi=Quantity([0, 1, 2, 3], u.rad)
        )

    # ==========================================================================
    # represent_as

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_polar_to_cartesian1d(self, vector):
        """Test ``vector.represent_as(Cartesian1DVector)``."""
        cart1d = vector.represent_as(Cartesian1DVector)

        assert isinstance(cart1d, Cartesian1DVector)
        assert cart1d.x == Quantity([1, 2, 3, 4], u.kpc)

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_polar_to_radial(self, vector):
        """Test ``vector.represent_as(RadialVector)``."""
        radial = vector.represent_as(RadialVector)

        assert isinstance(radial, RadialVector)
        assert radial.r == Quantity([1, 2, 3, 4], u.kpc)

    def test_polar_to_cartesian2d(self, vector):
        """Test ``vector.represent_as(Cartesian2DVector)``."""
        cart2d = vector.represent_as(Cartesian2DVector, y=Quantity([5, 6, 7, 8], u.km))

        assert isinstance(cart2d, Cartesian2DVector)
        assert cart2d.x == Quantity([1, 2, 3, 4], u.kpc)
        assert cart2d.y == Quantity([5, 6, 7, 8], u.km)

    def test_polar_to_polar(self, vector):
        """Test ``vector.represent_as(PolarVector)``."""
        # Jit can copy
        newvec = vector.represent_as(PolarVector)
        assert newvec == vector

        # The normal `represent_as` method should return the same object
        newvec = represent_as(vector, PolarVector)
        assert newvec is vector

    # def test_polar_to_lnpolar(self, vector):
    #     """Test ``vector.represent_as(LnPolarVector)``."""
    #     assert False

    # def test_polar_to_log10polar(self, vector):
    #     """Test ``vector.represent_as(Log10PolarVector)
    #     assert False

    def test_polar_to_cartesian3d(self, vector):
        """Test ``vector.represent_as(Cartesian3DVector)``."""
        cart3d = vector.represent_as(
            Cartesian3DVector, z=Quantity([9, 10, 11, 12], u.m)
        )

        assert isinstance(cart3d, Cartesian3DVector)
        assert cart3d.x == Quantity([1, 2, 3, 4], u.kpc)
        assert cart3d.y == Quantity([5, 6, 7, 8], u.km)
        assert cart3d.z == Quantity([9, 10, 11, 12], u.m)

    def test_polar_to_spherical(self, vector):
        """Test ``vector.represent_as(SphericalVector)``."""
        spherical = vector.represent_as(
            SphericalVector, theta=Quantity([4, 5, 6, 7], u.rad)
        )

        assert isinstance(spherical, SphericalVector)
        assert spherical.r == Quantity([1, 2, 3, 4], u.kpc)
        assert spherical.phi == Quantity([0, 1, 2, 3], u.rad)
        assert spherical.theta == Quantity([4, 5, 6, 7], u.rad)

    def test_polar_to_cylindrical(self, vector):
        """Test ``vector.represent_as(CylindricalVector)``."""
        cylindrical = vector.represent_as(
            CylindricalVector, z=Quantity([9, 10, 11, 12], u.m)
        )

        assert isinstance(cylindrical, CylindricalVector)
        assert cylindrical.rho == Quantity([1, 2, 3, 4], u.kpc)
        assert cylindrical.phi == Quantity([0, 1, 2, 3], u.rad)
        assert cylindrical.z == Quantity([9, 10, 11, 12], u.m)


class Abstract2DVectorDifferentialTest(AbstractVectorDifferentialTest):
    """Test :class:`vector.Abstract2DVectorDifferential`."""


class TestCartesianDifferential2D(Abstract2DVectorDifferentialTest):
    """Test :class:`vector.CartesianDifferential2D`."""

    @pytest.fixture(scope="class")
    def difntl(self) -> CartesianDifferential2D:
        """Return a differential."""
        return CartesianDifferential2D(
            d_x=Quantity([1, 2, 3, 4], u.km / u.s),
            d_y=Quantity([5, 6, 7, 8], u.km / u.s),
        )

    @pytest.fixture(scope="class")
    def vector(self) -> Cartesian2DVector:
        """Return a vector."""
        return Cartesian2DVector(
            x=Quantity([1, 2, 3, 4], u.kpc), y=Quantity([5, 6, 7, 8], u.km)
        )

    # ==========================================================================
    # represent_as

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_cartesian1d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianDifferential1D, vector)``."""
        cart1d = difntl.represent_as(CartesianDifferential1D, vector)

        assert isinstance(cart1d, CartesianDifferential1D)
        assert cart1d.d_x == Quantity([1, 2, 3, 4], u.km / u.s)

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_radial(self, difntl, vector):
        """Test ``difntl.represent_as(RadialDifferential, vector)``."""
        radial = difntl.represent_as(RadialDifferential, vector)

        assert isinstance(radial, RadialDifferential)
        assert radial.d_r == Quantity([1, 2, 3, 4], u.km / u.s)

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_cartesian2d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianDifferential2D, vector)``."""
        # Jit can copy
        newvec = difntl.represent_as(CartesianDifferential2D, vector)
        assert newvec == difntl

        # The normal `represent_as` method should return the same object
        newvec = represent_as(difntl, CartesianDifferential2D, vector)
        assert newvec is difntl

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_polar(self, difntl, vector):
        """Test ``difntl.represent_as(PolarDifferential, vector)``."""
        polar = difntl.represent_as(PolarDifferential, vector)

        assert isinstance(polar, PolarDifferential)
        assert polar.d_r == Quantity([1, 2, 3, 4], u.km / u.s)
        assert polar.d_phi == Quantity([5, 6, 7, 8], u.km * u.rad / (u.kpc * u.s))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_cartesian3d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianDifferential3D, vector)``."""
        cart3d = difntl.represent_as(
            CartesianDifferential3D, vector, d_z=Quantity([9, 10, 11, 12], u.m / u.s)
        )

        assert isinstance(cart3d, CartesianDifferential3D)
        assert cart3d.d_x == Quantity([1, 2, 3, 4], u.km / u.s)
        assert cart3d.d_y == Quantity([5, 6, 7, 8], u.km / u.s)
        assert cart3d.d_z == Quantity([9, 10, 11, 12], u.m / u.s)

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_spherical(self, difntl, vector):
        """Test ``difntl.represent_as(SphericalDifferential, vector)``."""
        spherical = difntl.represent_as(
            SphericalDifferential, vector, d_theta=Quantity([4, 5, 6, 7], u.rad)
        )

        assert isinstance(spherical, SphericalDifferential)
        assert spherical.d_r == Quantity([1, 2, 3, 4], u.km / u.s)
        assert spherical.d_phi == Quantity([5, 6, 7, 8], u.km / u.s)
        assert spherical.d_theta == Quantity([4, 5, 6, 7], u.rad)

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_cylindrical(self, difntl, vector):
        """Test ``difntl.represent_as(CylindricalDifferential, vector)``."""
        cylindrical = difntl.represent_as(
            CylindricalDifferential, vector, d_z=Quantity([9, 10, 11, 12], u.m / u.s)
        )

        assert isinstance(cylindrical, CylindricalDifferential)
        assert cylindrical.d_rho == Quantity([1, 2, 3, 4], u.km / u.s)
        assert cylindrical.d_phi == Quantity([5, 6, 7, 8], u.km / u.s)
        assert cylindrical.d_z == Quantity([9, 10, 11, 12], u.m / u.s)


class TestPolarDifferential(Abstract2DVectorDifferentialTest):
    """Test :class:`vector.PolarDifferential`."""

    @pytest.fixture(scope="class")
    def difntl(self) -> PolarDifferential:
        """Return a differential."""
        return PolarDifferential(
            d_r=Quantity([1, 2, 3, 4], u.km / u.s),
            d_phi=Quantity([5, 6, 7, 8], u.mas / u.yr),
        )

    @pytest.fixture(scope="class")
    def vector(self) -> PolarVector:
        """Return a vector."""
        return PolarVector(
            r=Quantity([1, 2, 3, 4], u.kpc), phi=Quantity([0, 1, 2, 3], u.rad)
        )

    # ==========================================================================
    # represent_as

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_cartesian1d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianDifferential1D, vector)``."""
        cart1d = difntl.represent_as(CartesianDifferential1D, vector)

        assert isinstance(cart1d, CartesianDifferential1D)
        assert cart1d.d_x == Quantity([1, 2, 3, 4], u.km / u.s)

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_radial(self, difntl, vector):
        """Test ``difntl.represent_as(RadialDifferential, vector)``."""
        radial = difntl.represent_as(RadialDifferential, vector)

        assert isinstance(radial, RadialDifferential)
        assert radial.d_r == Quantity([1, 2, 3, 4], u.km / u.s)

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_cartesian2d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianDifferential2D, vector)``."""
        cart2d = difntl.represent_as(CartesianDifferential2D, vector)

        assert isinstance(cart2d, CartesianDifferential2D)
        assert cart2d.d_x == Quantity([1, 2, 3, 4], u.km / u.s)
        assert cart2d.d_y == Quantity([5, 6, 7, 8], u.km / u.s)

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_polar(self, difntl, vector):
        """Test ``difntl.represent_as(PolarDifferential, vector)``."""
        # Jit can copy
        newvec = difntl.represent_as(PolarDifferential, vector)
        assert newvec == difntl

        # The normal `represent_as` method should return the same object
        newvec = represent_as(difntl, PolarDifferential, vector)
        assert newvec is difntl

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_cartesian3d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianDifferential3D, vector)``."""
        cart3d = difntl.represent_as(
            CartesianDifferential3D, vector, d_z=Quantity([9, 10, 11, 12], u.m / u.s)
        )

        assert isinstance(cart3d, CartesianDifferential3D)
        assert cart3d.d_x == Quantity([1, 2, 3, 4], u.km / u.s)
        assert cart3d.d_y == Quantity([5, 6, 7, 8], u.km / u.s)
        assert cart3d.d_z == Quantity([9, 10, 11, 12], u.m / u.s)

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_spherical(self, difntl, vector):
        """Test ``difntl.represent_as(SphericalDifferential, vector)``."""
        spherical = difntl.represent_as(
            SphericalDifferential, vector, d_theta=Quantity([4, 5, 6, 7], u.rad)
        )

        assert isinstance(spherical, SphericalDifferential)
        assert spherical.d_r == Quantity([1, 2, 3, 4], u.km / u.s)
        assert spherical.d_phi == Quantity([5, 6, 7, 8], u.km / u.s)
        assert spherical.d_theta == Quantity([4, 5, 6, 7], u.rad)

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_cylindrical(self, difntl, vector):
        """Test ``difntl.represent_as(CylindricalDifferential, vector)``."""
        cylindrical = difntl.represent_as(
            CylindricalDifferential, vector, d_z=Quantity([9, 10, 11, 12], u.m / u.s)
        )

        assert isinstance(cylindrical, CylindricalDifferential)
        assert cylindrical.d_rho == Quantity([1, 2, 3, 4], u.km / u.s)
        assert cylindrical.d_phi == Quantity([5, 6, 7, 8], u.km / u.s)
        assert cylindrical.d_z == Quantity([9, 10, 11, 12], u.m / u.s)
