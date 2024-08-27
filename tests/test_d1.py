"""Test :mod:`coordinax.d1`."""

import pytest

import quaxed.numpy as jnp
from unxt import Quantity

import coordinax as cx
from .test_base import AbstractPositionTest, AbstractVelocityTest


class AbstractPosition1DTest(AbstractPositionTest):
    """Test :class:`coordinax.AbstractPosition1D`."""

    # TODO: Add tests


class TestCartesianPosition1D(AbstractPosition1DTest):
    """Test :class:`coordinax.CartesianPosition1D`."""

    @pytest.fixture(scope="class")
    def vector(self) -> cx.AbstractPosition:
        """Return a vector."""
        return cx.CartesianPosition1D(x=Quantity([1, 2, 3, 4], "kpc"))

    # ==========================================================================
    # represent_as

    def test_cartesian1d_to_cartesian1d(self, vector):
        """Test ``coordinax.represent_as(CartesianPosition1D)``."""
        # Jit can copy
        newvec = vector.represent_as(cx.CartesianPosition1D)
        assert newvec == vector

        # The normal `represent_as` method should return the same object
        newvec = cx.represent_as(vector, cx.CartesianPosition1D)
        assert newvec is vector

    def test_cartesian1d_to_radial(self, vector):
        """Test ``coordinax.represent_as(RadialPosition)``."""
        radial = vector.represent_as(cx.RadialPosition)

        assert isinstance(radial, cx.RadialPosition)
        assert jnp.array_equal(radial.r, Quantity([1, 2, 3, 4], "kpc"))

    def test_cartesian1d_to_cartesian2d(self, vector):
        """Test ``coordinax.represent_as(CartesianPosition2D)``."""
        cart2d = vector.represent_as(
            cx.CartesianPosition2D, y=Quantity([5, 6, 7, 8], "km")
        )

        assert isinstance(cart2d, cx.CartesianPosition2D)
        assert jnp.array_equal(cart2d.x, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cart2d.y, Quantity([5, 6, 7, 8], "km"))

    def test_cartesian1d_to_polar(self, vector):
        """Test ``coordinax.represent_as(PolarPosition)``."""
        polar = vector.represent_as(cx.PolarPosition, phi=Quantity([0, 1, 2, 3], "rad"))

        assert isinstance(polar, cx.PolarPosition)
        assert jnp.array_equal(polar.r, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(polar.phi, Quantity([0, 1, 2, 3], "rad"))

    def test_cartesian1d_to_cartesian3d(self, vector):
        """Test ``coordinax.represent_as(CartesianPosition3D)``."""
        cart3d = vector.represent_as(
            cx.CartesianPosition3D,
            y=Quantity([5, 6, 7, 8], "km"),
            z=Quantity([9, 10, 11, 12], "m"),
        )

        assert isinstance(cart3d, cx.CartesianPosition3D)
        assert jnp.array_equal(cart3d.x, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cart3d.y, Quantity([5, 6, 7, 8], "km"))
        assert jnp.array_equal(cart3d.z, Quantity([9, 10, 11, 12], "m"))

    def test_cartesian1d_to_spherical(self, vector):
        """Test ``coordinax.represent_as(SphericalPosition)``."""
        spherical = vector.represent_as(
            cx.SphericalPosition,
            theta=Quantity([4, 15, 60, 170], "deg"),
            phi=Quantity([0, 1, 2, 3], "rad"),
        )

        assert isinstance(spherical, cx.SphericalPosition)
        assert jnp.array_equal(spherical.r, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(spherical.theta, Quantity([4, 15, 60, 170], "deg"))
        assert jnp.array_equal(spherical.phi, Quantity([0, 1, 2, 3], "rad"))

    def test_cartesian1d_to_cylindrical(self, vector):
        """Test ``coordinax.represent_as(CylindricalPosition)``."""
        cylindrical = vector.represent_as(
            cx.CylindricalPosition,
            phi=Quantity([0, 1, 2, 3], "rad"),
            z=Quantity([4, 5, 6, 7], "m"),
        )

        assert isinstance(cylindrical, cx.CylindricalPosition)
        assert jnp.array_equal(cylindrical.rho, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cylindrical.phi, Quantity([0, 1, 2, 3], "rad"))
        assert jnp.array_equal(cylindrical.z, Quantity([4, 5, 6, 7], "m"))


class TestRadialPosition(AbstractPosition1DTest):
    """Test :class:`coordinax.RadialPosition`."""

    @pytest.fixture(scope="class")
    def vector(self) -> cx.AbstractPosition:
        """Return a vector."""
        return cx.RadialPosition(r=Quantity([1, 2, 3, 4], "kpc"))

    # ==========================================================================
    # represent_as

    def test_radial_to_cartesian1d(self, vector):
        """Test ``coordinax.represent_as(CartesianPosition1D)``."""
        cart1d = vector.represent_as(cx.CartesianPosition1D)

        assert isinstance(cart1d, cx.CartesianPosition1D)
        assert jnp.array_equal(cart1d.x, Quantity([1, 2, 3, 4], "kpc"))

    def test_radial_to_radial(self, vector):
        """Test ``coordinax.represent_as(RadialPosition)``."""
        # Jit can copy
        newvec = vector.represent_as(cx.RadialPosition)
        assert newvec == vector

        # The normal `represent_as` method should return the same object
        newvec = cx.represent_as(vector, cx.RadialPosition)
        assert newvec is vector

    def test_radial_to_cartesian2d(self, vector):
        """Test ``coordinax.represent_as(CartesianPosition2D)``."""
        cart2d = vector.represent_as(
            cx.CartesianPosition2D, y=Quantity([5, 6, 7, 8], "km")
        )

        assert isinstance(cart2d, cx.CartesianPosition2D)
        assert jnp.array_equal(cart2d.x, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cart2d.y, Quantity([5, 6, 7, 8], "km"))

    def test_radial_to_polar(self, vector):
        """Test ``coordinax.represent_as(PolarPosition)``."""
        polar = vector.represent_as(cx.PolarPosition, phi=Quantity([0, 1, 2, 3], "rad"))

        assert isinstance(polar, cx.PolarPosition)
        assert jnp.array_equal(polar.r, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(polar.phi, Quantity([0, 1, 2, 3], "rad"))

    def test_radial_to_cartesian3d(self, vector):
        """Test ``coordinax.represent_as(CartesianPosition3D)``."""
        cart3d = vector.represent_as(
            cx.CartesianPosition3D,
            y=Quantity([5, 6, 7, 8], "km"),
            z=Quantity([9, 10, 11, 12], "m"),
        )

        assert isinstance(cart3d, cx.CartesianPosition3D)
        assert jnp.array_equal(cart3d.x, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cart3d.y, Quantity([5, 6, 7, 8], "km"))
        assert jnp.array_equal(cart3d.z, Quantity([9, 10, 11, 12], "m"))

    def test_radial_to_spherical(self, vector):
        """Test ``coordinax.represent_as(SphericalPosition)``."""
        spherical = vector.represent_as(
            cx.SphericalPosition,
            theta=Quantity([4, 15, 60, 170], "deg"),
            phi=Quantity([0, 1, 2, 3], "rad"),
        )

        assert isinstance(spherical, cx.SphericalPosition)
        assert jnp.array_equal(spherical.r, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(spherical.theta, Quantity([4, 15, 60, 170], "deg"))
        assert jnp.array_equal(spherical.phi, Quantity([0, 1, 2, 3], "rad"))

    def test_radial_to_cylindrical(self, vector):
        """Test ``coordinax.represent_as(CylindricalPosition)``."""
        cylindrical = vector.represent_as(
            cx.CylindricalPosition,
            phi=Quantity([0, 1, 2, 3], "rad"),
            z=Quantity([4, 5, 6, 7], "m"),
        )

        assert isinstance(cylindrical, cx.CylindricalPosition)
        assert jnp.array_equal(cylindrical.rho, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cylindrical.phi, Quantity([0, 1, 2, 3], "rad"))
        assert jnp.array_equal(cylindrical.z, Quantity([4, 5, 6, 7], "m"))


class AbstractVelocity1DTest(AbstractVelocityTest):
    """Test :class:`coordinax.AbstractVelocity1D`."""


class TestCartesianVelocity1D(AbstractVelocity1DTest):
    """Test :class:`coordinax.CartesianVelocity1D`."""

    @pytest.fixture(scope="class")
    def difntl(self) -> cx.CartesianVelocity1D:
        """Return a vector."""
        return cx.CartesianVelocity1D(d_x=Quantity([1.0, 2, 3, 4], "km/s"))

    @pytest.fixture(scope="class")
    def vector(self) -> cx.CartesianPosition1D:
        """Return a vector."""
        return cx.CartesianPosition1D(x=Quantity([1.0, 2, 3, 4], "kpc"))

    # ==========================================================================
    # represent_as

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_cartesian1d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianVelocity1D)``."""
        # Jit can copy
        newvec = difntl.represent_as(cx.CartesianVelocity1D, vector)
        assert newvec == difntl

        # The normal `represent_as` method should return the same object
        newvec = cx.represent_as(difntl, cx.CartesianVelocity1D, vector)
        assert newvec is difntl

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_radial(self, difntl, vector):
        """Test ``difntl.represent_as(RadialVelocity)``."""
        radial = difntl.represent_as(cx.RadialVelocity, vector)

        assert isinstance(radial, cx.RadialVelocity)
        assert jnp.array_equal(radial.d_r, Quantity([1, 2, 3, 4], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_cartesian2d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianVelocity2D)``."""
        cart2d = difntl.represent_as(
            cx.CartesianVelocity2D, vector, d_y=Quantity([5, 6, 7, 8], "km")
        )

        assert isinstance(cart2d, cx.CartesianVelocity2D)
        assert jnp.array_equal(cart2d.d_x, Quantity([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(cart2d.d_y, Quantity([5, 6, 7, 8], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_polar(self, difntl, vector):
        """Test ``difntl.represent_as(PolarVelocity)``."""
        polar = difntl.represent_as(
            cx.PolarVelocity, vector, d_phi=Quantity([0, 1, 2, 3], "rad")
        )

        assert isinstance(polar, cx.PolarVelocity)
        assert jnp.array_equal(polar.d_r, Quantity([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(polar.d_phi, Quantity([0, 1, 2, 3], "rad/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_cartesian3d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianVelocity3D)``."""
        cart3d = difntl.represent_as(
            cx.CartesianVelocity3D,
            vector,
            d_y=Quantity([5, 6, 7, 8], "km"),
            d_z=Quantity([9, 10, 11, 12], "m"),
        )

        assert isinstance(cart3d, cx.CartesianVelocity3D)
        assert jnp.array_equal(cart3d.d_x, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cart3d.d_y, Quantity([5, 6, 7, 8], "km"))
        assert jnp.array_equal(cart3d.d_z, Quantity([9, 10, 11, 12], "m"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_spherical(self, difntl, vector):
        """Test ``difntl.represent_as(SphericalVelocity)``."""
        spherical = difntl.represent_as(
            cx.SphericalVelocity,
            vector,
            d_theta=Quantity([4, 5, 6, 7], "rad/s"),
            d_phi=Quantity([0, 1, 2, 3], "rad/s"),
        )

        assert isinstance(spherical, cx.SphericalVelocity)
        assert jnp.array_equal(spherical.d_r, Quantity([1, 2, 3, 4], "kpc"))
        assert spherical.d_theta == Quantity([4, 5, 6, 7], "rad/s")
        assert jnp.array_equal(spherical.d_phi, Quantity([0, 1, 2, 3], "rad/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_cylindrical(self, difntl, vector):
        """Test ``difntl.represent_as(CylindricalVelocity)``."""
        cylindrical = difntl.represent_as(
            cx.CylindricalVelocity,
            vector,
            d_phi=Quantity([0, 1, 2, 3], "rad/s"),
            d_z=Quantity([4, 5, 6, 7], "m/s"),
        )

        assert isinstance(cylindrical, cx.CylindricalVelocity)
        assert jnp.array_equal(cylindrical.d_rho, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cylindrical.d_phi, Quantity([0, 1, 2, 3], "rad/s"))
        assert jnp.array_equal(cylindrical.d_z, Quantity([4, 5, 6, 7], "m/s"))


class TestRadialVelocity(AbstractVelocity1DTest):
    """Test :class:`coordinax.RadialVelocity`."""

    @pytest.fixture(scope="class")
    def difntl(self) -> cx.RadialVelocity:
        """Return a vector."""
        return cx.RadialVelocity(d_r=Quantity([1, 2, 3, 4], "km/s"))

    @pytest.fixture(scope="class")
    def vector(self) -> cx.RadialPosition:
        """Return a vector."""
        return cx.RadialPosition(r=Quantity([1, 2, 3, 4], "kpc"))

    # ==========================================================================
    # represent_as

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_cartesian1d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianVelocity1D)``."""
        cart1d = difntl.represent_as(cx.CartesianVelocity1D, vector)

        assert isinstance(cart1d, cx.CartesianVelocity1D)
        assert jnp.array_equal(cart1d.d_x, Quantity([1, 2, 3, 4], "km/s"))

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_radial(self, difntl, vector):
        """Test ``difntl.represent_as(RadialVelocity)``."""
        # Jit can copy
        newvec = difntl.represent_as(cx.RadialVelocity, vector)
        assert newvec == difntl

        # The normal `represent_as` method should return the same object
        newvec = cx.represent_as(difntl, cx.RadialVelocity, vector)
        assert newvec is difntl

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_cartesian2d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianVelocity2D)``."""
        cart2d = difntl.represent_as(
            cx.CartesianVelocity2D, vector, d_y=Quantity([5, 6, 7, 8], "km")
        )

        assert isinstance(cart2d, cx.CartesianVelocity2D)
        assert jnp.array_equal(cart2d.d_x, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cart2d.d_y, Quantity([5, 6, 7, 8], "km"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_polar(self, difntl, vector):
        """Test ``difntl.represent_as(PolarVelocity)``."""
        polar = difntl.represent_as(
            cx.PolarVelocity, vector, d_phi=Quantity([0, 1, 2, 3], "rad")
        )

        assert isinstance(polar, cx.PolarVelocity)
        assert jnp.array_equal(polar.d_r, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(polar.d_phi, Quantity([0, 1, 2, 3], "rad"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_cartesian3d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianVelocity3D)``."""
        cart3d = difntl.represent_as(
            cx.CartesianVelocity3D,
            vector,
            d_y=Quantity([5, 6, 7, 8], "km"),
            d_z=Quantity([9, 10, 11, 12], "m"),
        )

        assert isinstance(cart3d, cx.CartesianVelocity3D)
        assert jnp.array_equal(cart3d.d_x, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cart3d.d_y, Quantity([5, 6, 7, 8], "km"))
        assert jnp.array_equal(cart3d.d_z, Quantity([9, 10, 11, 12], "m"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_spherical(self, difntl, vector):
        """Test ``difntl.represent_as(SphericalVelocity)``."""
        spherical = difntl.represent_as(
            cx.SphericalVelocity,
            vector,
            d_theta=Quantity([4, 5, 6, 7], "rad"),
            d_phi=Quantity([0, 1, 2, 3], "rad"),
        )

        assert isinstance(spherical, cx.SphericalVelocity)
        assert jnp.array_equal(spherical.d_r, Quantity([1, 2, 3, 4], "kpc"))
        assert spherical.d_theta == Quantity([4, 5, 6, 7], "rad")
        assert jnp.array_equal(spherical.d_phi, Quantity([0, 1, 2, 3], "rad"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_cylindrical(self, difntl, vector):
        """Test ``difntl.represent_as(CylindricalVelocity)``."""
        cylindrical = difntl.represent_as(
            cx.CylindricalVelocity,
            vector,
            d_phi=Quantity([0, 1, 2, 3], "rad"),
            d_z=Quantity([4, 5, 6, 7], "m"),
        )

        assert isinstance(cylindrical, cx.CylindricalVelocity)
        assert jnp.array_equal(cylindrical.d_rho, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cylindrical.d_phi, Quantity([0, 1, 2, 3], "rad"))
        assert jnp.array_equal(cylindrical.d_z, Quantity([4, 5, 6, 7], "m"))
