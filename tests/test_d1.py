"""Test :mod:`coordinax._d1`."""

import pytest

import quaxed.numpy as qnp
from unxt import Quantity

import coordinax as cx
from .test_base import AbstractVectorDifferentialTest, AbstractVectorTest


class Abstract1DVectorTest(AbstractVectorTest):
    """Test :class:`coordinax.Abstract1DVector`."""

    # TODO: Add tests


class TestCartesian1DVector(Abstract1DVectorTest):
    """Test :class:`coordinax.Cartesian1DVector`."""

    @pytest.fixture(scope="class")
    def vector(self) -> cx.AbstractVector:
        """Return a vector."""
        return cx.Cartesian1DVector(x=Quantity([1, 2, 3, 4], "kpc"))

    # ==========================================================================
    # represent_as

    def test_cartesian1d_to_cartesian1d(self, vector):
        """Test ``coordinax.represent_as(Cartesian1DVector)``."""
        # Jit can copy
        newvec = vector.represent_as(cx.Cartesian1DVector)
        assert newvec == vector

        # The normal `represent_as` method should return the same object
        newvec = cx.represent_as(vector, cx.Cartesian1DVector)
        assert newvec is vector

    def test_cartesian1d_to_radial(self, vector):
        """Test ``coordinax.represent_as(RadialVector)``."""
        radial = vector.represent_as(cx.RadialVector)

        assert isinstance(radial, cx.RadialVector)
        assert qnp.array_equal(radial.r, Quantity([1, 2, 3, 4], "kpc"))

    def test_cartesian1d_to_cartesian2d(self, vector):
        """Test ``coordinax.represent_as(Cartesian2DVector)``."""
        cart2d = vector.represent_as(
            cx.Cartesian2DVector, y=Quantity([5, 6, 7, 8], "km")
        )

        assert isinstance(cart2d, cx.Cartesian2DVector)
        assert qnp.array_equal(cart2d.x, Quantity([1, 2, 3, 4], "kpc"))
        assert qnp.array_equal(cart2d.y, Quantity([5, 6, 7, 8], "km"))

    def test_cartesian1d_to_polar(self, vector):
        """Test ``coordinax.represent_as(PolarVector)``."""
        polar = vector.represent_as(cx.PolarVector, phi=Quantity([0, 1, 2, 3], "rad"))

        assert isinstance(polar, cx.PolarVector)
        assert qnp.array_equal(polar.r, Quantity([1, 2, 3, 4], "kpc"))
        assert qnp.array_equal(polar.phi, Quantity([0, 1, 2, 3], "rad"))

    # def test_cartesian1d_to_lnpolar(self, vector):
    #     """Test ``coordinax.represent_as(LnPolarVector)``."""
    #     lnpolar = vector.to_lnpolar(phi=Quantity([0, 1, 2, 3], "rad"))

    #     assert isinstance(lnpolar, LnPolarVector)
    #     assert lnpolar.lnr == xp.log(Quantity([1, 2, 3, 4], "kpc"))
    #     assert qnp.array_equal(lnpolar.phi, Quantity([0, 1, 2, 3], "rad"))

    # def test_cartesian1d_to_log10polar(self, vector):
    #     """Test ``coordinax.represent_as(Log10PolarVector)``."""
    #     log10polar = vector.to_log10polar(phi=Quantity([0, 1, 2, 3], "rad"))

    #     assert isinstance(log10polar, Log10PolarVector)
    #     assert log10polar.log10r == xp.log10(Quantity([1, 2, 3, 4], "kpc"))
    #     assert qnp.array_equal(log10polar.phi, Quantity([0, 1, 2, 3], "rad"))

    def test_cartesian1d_to_cartesian3d(self, vector):
        """Test ``coordinax.represent_as(Cartesian3DVector)``."""
        cart3d = vector.represent_as(
            cx.Cartesian3DVector,
            y=Quantity([5, 6, 7, 8], "km"),
            z=Quantity([9, 10, 11, 12], "m"),
        )

        assert isinstance(cart3d, cx.Cartesian3DVector)
        assert qnp.array_equal(cart3d.x, Quantity([1, 2, 3, 4], "kpc"))
        assert qnp.array_equal(cart3d.y, Quantity([5, 6, 7, 8], "km"))
        assert qnp.array_equal(cart3d.z, Quantity([9, 10, 11, 12], "m"))

    def test_cartesian1d_to_spherical(self, vector):
        """Test ``coordinax.represent_as(SphericalVector)``."""
        spherical = vector.represent_as(
            cx.SphericalVector,
            phi=Quantity([0, 1, 2, 3], "rad"),
            theta=Quantity([4, 5, 6, 7], "rad"),
        )

        assert isinstance(spherical, cx.SphericalVector)
        assert qnp.array_equal(spherical.r, Quantity([1, 2, 3, 4], "kpc"))
        assert qnp.array_equal(spherical.phi, Quantity([0, 1, 2, 3], "rad"))
        assert qnp.array_equal(spherical.theta, Quantity([4, 5, 6, 7], "rad"))

    def test_cartesian1d_to_cylindrical(self, vector):
        """Test ``coordinax.represent_as(CylindricalVector)``."""
        cylindrical = vector.represent_as(
            cx.CylindricalVector,
            phi=Quantity([0, 1, 2, 3], "rad"),
            z=Quantity([4, 5, 6, 7], "m"),
        )

        assert isinstance(cylindrical, cx.CylindricalVector)
        assert qnp.array_equal(cylindrical.rho, Quantity([1, 2, 3, 4], "kpc"))
        assert qnp.array_equal(cylindrical.phi, Quantity([0, 1, 2, 3], "rad"))
        assert qnp.array_equal(cylindrical.z, Quantity([4, 5, 6, 7], "m"))


class TestRadialVector(Abstract1DVectorTest):
    """Test :class:`coordinax.RadialVector`."""

    @pytest.fixture(scope="class")
    def vector(self) -> cx.AbstractVector:
        """Return a vector."""
        return cx.RadialVector(r=Quantity([1, 2, 3, 4], "kpc"))

    # ==========================================================================
    # represent_as

    def test_radial_to_cartesian1d(self, vector):
        """Test ``coordinax.represent_as(Cartesian1DVector)``."""
        cart1d = vector.represent_as(cx.Cartesian1DVector)

        assert isinstance(cart1d, cx.Cartesian1DVector)
        assert qnp.array_equal(cart1d.x, Quantity([1, 2, 3, 4], "kpc"))

    def test_radial_to_radial(self, vector):
        """Test ``coordinax.represent_as(RadialVector)``."""
        # Jit can copy
        newvec = vector.represent_as(cx.RadialVector)
        assert newvec == vector

        # The normal `represent_as` method should return the same object
        newvec = cx.represent_as(vector, cx.RadialVector)
        assert newvec is vector

    def test_radial_to_cartesian2d(self, vector):
        """Test ``coordinax.represent_as(Cartesian2DVector)``."""
        cart2d = vector.represent_as(
            cx.Cartesian2DVector, y=Quantity([5, 6, 7, 8], "km")
        )

        assert isinstance(cart2d, cx.Cartesian2DVector)
        assert qnp.array_equal(cart2d.x, Quantity([1, 2, 3, 4], "kpc"))
        assert qnp.array_equal(cart2d.y, Quantity([5, 6, 7, 8], "km"))

    def test_radial_to_polar(self, vector):
        """Test ``coordinax.represent_as(PolarVector)``."""
        polar = vector.represent_as(cx.PolarVector, phi=Quantity([0, 1, 2, 3], "rad"))

        assert isinstance(polar, cx.PolarVector)
        assert qnp.array_equal(polar.r, Quantity([1, 2, 3, 4], "kpc"))
        assert qnp.array_equal(polar.phi, Quantity([0, 1, 2, 3], "rad"))

    # def test_radial_to_lnpolar(self, vector):
    #     """Test ``coordinax.represent_as(LnPolarVector)``."""
    #     assert False

    # def test_radial_to_log10polar(self, vector):
    #     """Test ``coordinax.represent_as(Log10PolarVector)``."""
    #     assert False

    def test_radial_to_cartesian3d(self, vector):
        """Test ``coordinax.represent_as(Cartesian3DVector)``."""
        cart3d = vector.represent_as(
            cx.Cartesian3DVector,
            y=Quantity([5, 6, 7, 8], "km"),
            z=Quantity([9, 10, 11, 12], "m"),
        )

        assert isinstance(cart3d, cx.Cartesian3DVector)
        assert qnp.array_equal(cart3d.x, Quantity([1, 2, 3, 4], "kpc"))
        assert qnp.array_equal(cart3d.y, Quantity([5, 6, 7, 8], "km"))
        assert qnp.array_equal(cart3d.z, Quantity([9, 10, 11, 12], "m"))

    def test_radial_to_spherical(self, vector):
        """Test ``coordinax.represent_as(SphericalVector)``."""
        spherical = vector.represent_as(
            cx.SphericalVector,
            phi=Quantity([0, 1, 2, 3], "rad"),
            theta=Quantity([4, 5, 6, 7], "rad"),
        )

        assert isinstance(spherical, cx.SphericalVector)
        assert qnp.array_equal(spherical.r, Quantity([1, 2, 3, 4], "kpc"))
        assert qnp.array_equal(spherical.phi, Quantity([0, 1, 2, 3], "rad"))
        assert qnp.array_equal(spherical.theta, Quantity([4, 5, 6, 7], "rad"))

    def test_radial_to_cylindrical(self, vector):
        """Test ``coordinax.represent_as(CylindricalVector)``."""
        cylindrical = vector.represent_as(
            cx.CylindricalVector,
            phi=Quantity([0, 1, 2, 3], "rad"),
            z=Quantity([4, 5, 6, 7], "m"),
        )

        assert isinstance(cylindrical, cx.CylindricalVector)
        assert qnp.array_equal(cylindrical.rho, Quantity([1, 2, 3, 4], "kpc"))
        assert qnp.array_equal(cylindrical.phi, Quantity([0, 1, 2, 3], "rad"))
        assert qnp.array_equal(cylindrical.z, Quantity([4, 5, 6, 7], "m"))


class Abstract1DVectorDifferentialTest(AbstractVectorDifferentialTest):
    """Test :class:`coordinax.Abstract1DVectorDifferential`."""


class TestCartesianDifferential1D(Abstract1DVectorDifferentialTest):
    """Test :class:`coordinax.CartesianDifferential1D`."""

    @pytest.fixture(scope="class")
    def difntl(self) -> cx.CartesianDifferential1D:
        """Return a vector."""
        return cx.CartesianDifferential1D(d_x=Quantity([1.0, 2, 3, 4], "km/s"))

    @pytest.fixture(scope="class")
    def vector(self) -> cx.Cartesian1DVector:
        """Return a vector."""
        return cx.Cartesian1DVector(x=Quantity([1.0, 2, 3, 4], "kpc"))

    # ==========================================================================
    # represent_as

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_cartesian1d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianDifferential1D)``."""
        # Jit can copy
        newvec = difntl.represent_as(cx.CartesianDifferential1D, vector)
        assert newvec == difntl

        # The normal `represent_as` method should return the same object
        newvec = cx.represent_as(difntl, cx.CartesianDifferential1D, vector)
        assert newvec is difntl

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_radial(self, difntl, vector):
        """Test ``difntl.represent_as(RadialDifferential)``."""
        radial = difntl.represent_as(cx.RadialDifferential, vector)

        assert isinstance(radial, cx.RadialDifferential)
        assert qnp.array_equal(radial.d_r, Quantity([1, 2, 3, 4], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_cartesian2d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianDifferential2D)``."""
        cart2d = difntl.represent_as(
            cx.CartesianDifferential2D, vector, d_y=Quantity([5, 6, 7, 8], "km")
        )

        assert isinstance(cart2d, cx.CartesianDifferential2D)
        assert qnp.array_equal(cart2d.d_x, Quantity([1, 2, 3, 4], "km/s"))
        assert qnp.array_equal(cart2d.d_y, Quantity([5, 6, 7, 8], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_polar(self, difntl, vector):
        """Test ``difntl.represent_as(PolarDifferential)``."""
        polar = difntl.represent_as(
            cx.PolarDifferential, vector, d_phi=Quantity([0, 1, 2, 3], "rad")
        )

        assert isinstance(polar, cx.PolarDifferential)
        assert qnp.array_equal(polar.d_r, Quantity([1, 2, 3, 4], "km/s"))
        assert qnp.array_equal(polar.d_phi, Quantity([0, 1, 2, 3], "rad/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_cartesian3d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianDifferential3D)``."""
        cart3d = difntl.represent_as(
            cx.CartesianDifferential3D,
            vector,
            d_y=Quantity([5, 6, 7, 8], "km"),
            d_z=Quantity([9, 10, 11, 12], "m"),
        )

        assert isinstance(cart3d, cx.CartesianDifferential3D)
        assert qnp.array_equal(cart3d.d_x, Quantity([1, 2, 3, 4], "kpc"))
        assert qnp.array_equal(cart3d.d_y, Quantity([5, 6, 7, 8], "km"))
        assert qnp.array_equal(cart3d.d_z, Quantity([9, 10, 11, 12], "m"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_spherical(self, difntl, vector):
        """Test ``difntl.represent_as(SphericalDifferential)``."""
        spherical = difntl.represent_as(
            cx.SphericalDifferential,
            vector,
            d_phi=Quantity([0, 1, 2, 3], "rad"),
            d_theta=Quantity([4, 5, 6, 7], "rad"),
        )

        assert isinstance(spherical, cx.SphericalDifferential)
        assert qnp.array_equal(spherical.d_r, Quantity([1, 2, 3, 4], "kpc"))
        assert qnp.array_equal(spherical.d_phi, Quantity([0, 1, 2, 3], "rad"))
        assert spherical.dtheta == Quantity([4, 5, 6, 7], "rad")

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_cylindrical(self, difntl, vector):
        """Test ``difntl.represent_as(CylindricalDifferential)``."""
        cylindrical = difntl.represent_as(
            cx.CylindricalDifferential,
            vector,
            d_phi=Quantity([0, 1, 2, 3], "rad"),
            d_z=Quantity([4, 5, 6, 7], "m"),
        )

        assert isinstance(cylindrical, cx.CylindricalDifferential)
        assert qnp.array_equal(cylindrical.d_rho, Quantity([1, 2, 3, 4], "kpc"))
        assert qnp.array_equal(cylindrical.d_phi, Quantity([0, 1, 2, 3], "rad"))
        assert qnp.array_equal(cylindrical.d_z, Quantity([4, 5, 6, 7], "m"))


class TestRadialDifferential(Abstract1DVectorDifferentialTest):
    """Test :class:`coordinax.RadialDifferential`."""

    @pytest.fixture(scope="class")
    def difntl(self) -> cx.RadialDifferential:
        """Return a vector."""
        return cx.RadialDifferential(d_r=Quantity([1, 2, 3, 4], "km/s"))

    @pytest.fixture(scope="class")
    def vector(self) -> cx.RadialVector:
        """Return a vector."""
        return cx.RadialVector(r=Quantity([1, 2, 3, 4], "kpc"))

    # ==========================================================================
    # represent_as

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_cartesian1d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianDifferential1D)``."""
        cart1d = difntl.represent_as(cx.CartesianDifferential1D, vector)

        assert isinstance(cart1d, cx.CartesianDifferential1D)
        assert qnp.array_equal(cart1d.d_x, Quantity([1, 2, 3, 4], "km/s"))

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_radial(self, difntl, vector):
        """Test ``difntl.represent_as(RadialDifferential)``."""
        # Jit can copy
        newvec = difntl.represent_as(cx.RadialDifferential, vector)
        assert newvec == difntl

        # The normal `represent_as` method should return the same object
        newvec = cx.represent_as(difntl, cx.RadialDifferential, vector)
        assert newvec is difntl

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_cartesian2d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianDifferential2D)``."""
        cart2d = difntl.represent_as(
            cx.CartesianDifferential2D, vector, d_y=Quantity([5, 6, 7, 8], "km")
        )

        assert isinstance(cart2d, cx.CartesianDifferential2D)
        assert qnp.array_equal(cart2d.d_x, Quantity([1, 2, 3, 4], "kpc"))
        assert qnp.array_equal(cart2d.d_y, Quantity([5, 6, 7, 8], "km"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_polar(self, difntl, vector):
        """Test ``difntl.represent_as(PolarDifferential)``."""
        polar = difntl.represent_as(
            cx.PolarDifferential, vector, d_phi=Quantity([0, 1, 2, 3], "rad")
        )

        assert isinstance(polar, cx.PolarDifferential)
        assert qnp.array_equal(polar.d_r, Quantity([1, 2, 3, 4], "kpc"))
        assert qnp.array_equal(polar.d_phi, Quantity([0, 1, 2, 3], "rad"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_cartesian3d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianDifferential3D)``."""
        cart3d = difntl.represent_as(
            cx.CartesianDifferential3D,
            vector,
            d_y=Quantity([5, 6, 7, 8], "km"),
            d_z=Quantity([9, 10, 11, 12], "m"),
        )

        assert isinstance(cart3d, cx.CartesianDifferential3D)
        assert qnp.array_equal(cart3d.d_x, Quantity([1, 2, 3, 4], "kpc"))
        assert qnp.array_equal(cart3d.d_y, Quantity([5, 6, 7, 8], "km"))
        assert qnp.array_equal(cart3d.d_z, Quantity([9, 10, 11, 12], "m"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_spherical(self, difntl, vector):
        """Test ``difntl.represent_as(SphericalDifferential)``."""
        spherical = difntl.represent_as(
            cx.SphericalDifferential,
            vector,
            d_phi=Quantity([0, 1, 2, 3], "rad"),
            dtheta=Quantity([4, 5, 6, 7], "rad"),
        )

        assert isinstance(spherical, cx.SphericalDifferential)
        assert qnp.array_equal(spherical.d_r, Quantity([1, 2, 3, 4], "kpc"))
        assert qnp.array_equal(spherical.d_phi, Quantity([0, 1, 2, 3], "rad"))
        assert spherical.dtheta == Quantity([4, 5, 6, 7], "rad")

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_cylindrical(self, difntl, vector):
        """Test ``difntl.represent_as(CylindricalDifferential)``."""
        cylindrical = difntl.represent_as(
            cx.CylindricalDifferential,
            vector,
            d_phi=Quantity([0, 1, 2, 3], "rad"),
            d_z=Quantity([4, 5, 6, 7], "m"),
        )

        assert isinstance(cylindrical, cx.CylindricalDifferential)
        assert qnp.array_equal(cylindrical.d_rho, Quantity([1, 2, 3, 4], "kpc"))
        assert qnp.array_equal(cylindrical.d_phi, Quantity([0, 1, 2, 3], "rad"))
        assert qnp.array_equal(cylindrical.d_z, Quantity([4, 5, 6, 7], "m"))
