"""Test :mod:`coordinax.d1`."""

import pytest

import quaxed.numpy as jnp
from unxt import Quantity

import coordinax as cx
from .test_base import AbstractPosTest, AbstractVelTest


class AbstractPos1DTest(AbstractPosTest):
    """Test :class:`coordinax.AbstractPos1D`."""

    # TODO: Add tests


class TestCartesianPos1D(AbstractPos1DTest):
    """Test :class:`coordinax.CartesianPos1D`."""

    @pytest.fixture(scope="class")
    def vector(self) -> cx.AbstractPos:
        """Return a vector."""
        return cx.CartesianPos1D(x=Quantity([1, 2, 3, 4], "kpc"))

    # ==========================================================================
    # represent_as

    def test_cartesian1d_to_cartesian1d(self, vector):
        """Test ``coordinax.represent_as(CartesianPos1D)``."""
        # Jit can copy
        newvec = vector.represent_as(cx.CartesianPos1D)
        assert jnp.array_equal(newvec, vector)

        # The normal `represent_as` method should return the same object
        newvec = cx.represent_as(vector, cx.CartesianPos1D)
        assert newvec is vector

    def test_cartesian1d_to_radial(self, vector):
        """Test ``coordinax.represent_as(RadialPos)``."""
        radial = vector.represent_as(cx.RadialPos)

        assert isinstance(radial, cx.RadialPos)
        assert jnp.array_equal(radial.r, Quantity([1, 2, 3, 4], "kpc"))

    def test_cartesian1d_to_cartesian2d(self, vector):
        """Test ``coordinax.represent_as(CartesianPos2D)``."""
        cart2d = vector.represent_as(cx.CartesianPos2D, y=Quantity([5, 6, 7, 8], "km"))

        assert isinstance(cart2d, cx.CartesianPos2D)
        assert jnp.array_equal(cart2d.x, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cart2d.y, Quantity([5, 6, 7, 8], "km"))

    def test_cartesian1d_to_polar(self, vector):
        """Test ``coordinax.represent_as(PolarPos)``."""
        polar = vector.represent_as(cx.PolarPos, phi=Quantity([0, 1, 2, 3], "rad"))

        assert isinstance(polar, cx.PolarPos)
        assert jnp.array_equal(polar.r, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(polar.phi, Quantity([0, 1, 2, 3], "rad"))

    def test_cartesian1d_to_cartesian3d(self, vector):
        """Test ``coordinax.represent_as(CartesianPos3D)``."""
        cart3d = vector.represent_as(
            cx.CartesianPos3D,
            y=Quantity([5, 6, 7, 8], "km"),
            z=Quantity([9, 10, 11, 12], "m"),
        )

        assert isinstance(cart3d, cx.CartesianPos3D)
        assert jnp.array_equal(cart3d.x, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cart3d.y, Quantity([5, 6, 7, 8], "km"))
        assert jnp.array_equal(cart3d.z, Quantity([9, 10, 11, 12], "m"))

    def test_cartesian1d_to_spherical(self, vector):
        """Test ``coordinax.represent_as(SphericalPos)``."""
        spherical = vector.represent_as(
            cx.SphericalPos,
            theta=Quantity([4, 15, 60, 170], "deg"),
            phi=Quantity([0, 1, 2, 3], "rad"),
        )

        assert isinstance(spherical, cx.SphericalPos)
        assert jnp.array_equal(spherical.r, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(spherical.theta, Quantity([4, 15, 60, 170], "deg"))
        assert jnp.array_equal(spherical.phi, Quantity([0, 1, 2, 3], "rad"))

    def test_cartesian1d_to_cylindrical(self, vector):
        """Test ``coordinax.represent_as(CylindricalPos)``."""
        cylindrical = vector.represent_as(
            cx.CylindricalPos,
            phi=Quantity([0, 1, 2, 3], "rad"),
            z=Quantity([4, 5, 6, 7], "m"),
        )

        assert isinstance(cylindrical, cx.CylindricalPos)
        assert jnp.array_equal(cylindrical.rho, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cylindrical.phi, Quantity([0, 1, 2, 3], "rad"))
        assert jnp.array_equal(cylindrical.z, Quantity([4, 5, 6, 7], "m"))


class TestRadialPos(AbstractPos1DTest):
    """Test :class:`coordinax.RadialPos`."""

    @pytest.fixture(scope="class")
    def vector(self) -> cx.AbstractPos:
        """Return a vector."""
        return cx.RadialPos(r=Quantity([1, 2, 3, 4], "kpc"))

    # ==========================================================================
    # represent_as

    def test_radial_to_cartesian1d(self, vector):
        """Test ``coordinax.represent_as(CartesianPos1D)``."""
        cart1d = vector.represent_as(cx.CartesianPos1D)

        assert isinstance(cart1d, cx.CartesianPos1D)
        assert jnp.array_equal(cart1d.x, Quantity([1, 2, 3, 4], "kpc"))

    def test_radial_to_radial(self, vector):
        """Test ``coordinax.represent_as(RadialPos)``."""
        # Jit can copy
        newvec = vector.represent_as(cx.RadialPos)
        assert jnp.array_equal(newvec, vector)

        # The normal `represent_as` method should return the same object
        newvec = cx.represent_as(vector, cx.RadialPos)
        assert newvec is vector

    def test_radial_to_cartesian2d(self, vector):
        """Test ``coordinax.represent_as(CartesianPos2D)``."""
        cart2d = vector.represent_as(cx.CartesianPos2D, y=Quantity([5, 6, 7, 8], "km"))

        assert isinstance(cart2d, cx.CartesianPos2D)
        assert jnp.array_equal(cart2d.x, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cart2d.y, Quantity([5, 6, 7, 8], "km"))

    def test_radial_to_polar(self, vector):
        """Test ``coordinax.represent_as(PolarPos)``."""
        polar = vector.represent_as(cx.PolarPos, phi=Quantity([0, 1, 2, 3], "rad"))

        assert isinstance(polar, cx.PolarPos)
        assert jnp.array_equal(polar.r, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(polar.phi, Quantity([0, 1, 2, 3], "rad"))

    def test_radial_to_cartesian3d(self, vector):
        """Test ``coordinax.represent_as(CartesianPos3D)``."""
        cart3d = vector.represent_as(
            cx.CartesianPos3D,
            y=Quantity([5, 6, 7, 8], "km"),
            z=Quantity([9, 10, 11, 12], "m"),
        )

        assert isinstance(cart3d, cx.CartesianPos3D)
        assert jnp.array_equal(cart3d.x, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cart3d.y, Quantity([5, 6, 7, 8], "km"))
        assert jnp.array_equal(cart3d.z, Quantity([9, 10, 11, 12], "m"))

    def test_radial_to_spherical(self, vector):
        """Test ``coordinax.represent_as(SphericalPos)``."""
        spherical = vector.represent_as(
            cx.SphericalPos,
            theta=Quantity([4, 15, 60, 170], "deg"),
            phi=Quantity([0, 1, 2, 3], "rad"),
        )

        assert isinstance(spherical, cx.SphericalPos)
        assert jnp.array_equal(spherical.r, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(spherical.theta, Quantity([4, 15, 60, 170], "deg"))
        assert jnp.array_equal(spherical.phi, Quantity([0, 1, 2, 3], "rad"))

    def test_radial_to_cylindrical(self, vector):
        """Test ``coordinax.represent_as(CylindricalPos)``."""
        cylindrical = vector.represent_as(
            cx.CylindricalPos,
            phi=Quantity([0, 1, 2, 3], "rad"),
            z=Quantity([4, 5, 6, 7], "m"),
        )

        assert isinstance(cylindrical, cx.CylindricalPos)
        assert jnp.array_equal(cylindrical.rho, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cylindrical.phi, Quantity([0, 1, 2, 3], "rad"))
        assert jnp.array_equal(cylindrical.z, Quantity([4, 5, 6, 7], "m"))


class AbstractVel1DTest(AbstractVelTest):
    """Test :class:`coordinax.AbstractVel1D`."""


class TestCartesianVel1D(AbstractVel1DTest):
    """Test :class:`coordinax.CartesianVel1D`."""

    @pytest.fixture(scope="class")
    def difntl(self) -> cx.CartesianVel1D:
        """Return a vector."""
        return cx.CartesianVel1D(d_x=Quantity([1.0, 2, 3, 4], "km/s"))

    @pytest.fixture(scope="class")
    def vector(self) -> cx.CartesianPos1D:
        """Return a vector."""
        return cx.CartesianPos1D(x=Quantity([1.0, 2, 3, 4], "kpc"))

    # ==========================================================================
    # represent_as

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_cartesian1d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianVel1D)``."""
        # Jit can copy
        newvec = difntl.represent_as(cx.CartesianVel1D, vector)
        assert jnp.array_equal(newvec, difntl)

        # The normal `represent_as` method should return the same object
        newvec = cx.represent_as(difntl, cx.CartesianVel1D, vector)
        assert newvec is difntl

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_radial(self, difntl, vector):
        """Test ``difntl.represent_as(RadialVel)``."""
        radial = difntl.represent_as(cx.RadialVel, vector)

        assert isinstance(radial, cx.RadialVel)
        assert jnp.array_equal(radial.d_r, Quantity([1, 2, 3, 4], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_cartesian2d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianVel2D)``."""
        cart2d = difntl.represent_as(
            cx.CartesianVel2D, vector, d_y=Quantity([5, 6, 7, 8], "km")
        )

        assert isinstance(cart2d, cx.CartesianVel2D)
        assert jnp.array_equal(cart2d.d_x, Quantity([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(cart2d.d_y, Quantity([5, 6, 7, 8], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_polar(self, difntl, vector):
        """Test ``difntl.represent_as(PolarVel)``."""
        polar = difntl.represent_as(
            cx.PolarVel, vector, d_phi=Quantity([0, 1, 2, 3], "rad")
        )

        assert isinstance(polar, cx.PolarVel)
        assert jnp.array_equal(polar.d_r, Quantity([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(polar.d_phi, Quantity([0, 1, 2, 3], "rad/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_cartesian3d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianVel3D)``."""
        cart3d = difntl.represent_as(
            cx.CartesianVel3D,
            vector,
            d_y=Quantity([5, 6, 7, 8], "km"),
            d_z=Quantity([9, 10, 11, 12], "m"),
        )

        assert isinstance(cart3d, cx.CartesianVel3D)
        assert jnp.array_equal(cart3d.d_x, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cart3d.d_y, Quantity([5, 6, 7, 8], "km"))
        assert jnp.array_equal(cart3d.d_z, Quantity([9, 10, 11, 12], "m"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_spherical(self, difntl, vector):
        """Test ``difntl.represent_as(SphericalVel)``."""
        spherical = difntl.represent_as(
            cx.SphericalVel,
            vector,
            d_theta=Quantity([4, 5, 6, 7], "rad/s"),
            d_phi=Quantity([0, 1, 2, 3], "rad/s"),
        )

        assert isinstance(spherical, cx.SphericalVel)
        assert jnp.array_equal(spherical.d_r, Quantity([1, 2, 3, 4], "kpc"))
        assert spherical.d_theta == Quantity([4, 5, 6, 7], "rad/s")
        assert jnp.array_equal(spherical.d_phi, Quantity([0, 1, 2, 3], "rad/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_cylindrical(self, difntl, vector):
        """Test ``difntl.represent_as(CylindricalVel)``."""
        cylindrical = difntl.represent_as(
            cx.CylindricalVel,
            vector,
            d_phi=Quantity([0, 1, 2, 3], "rad/s"),
            d_z=Quantity([4, 5, 6, 7], "m/s"),
        )

        assert isinstance(cylindrical, cx.CylindricalVel)
        assert jnp.array_equal(cylindrical.d_rho, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cylindrical.d_phi, Quantity([0, 1, 2, 3], "rad/s"))
        assert jnp.array_equal(cylindrical.d_z, Quantity([4, 5, 6, 7], "m/s"))


class TestRadialVel(AbstractVel1DTest):
    """Test :class:`coordinax.RadialVel`."""

    @pytest.fixture(scope="class")
    def difntl(self) -> cx.RadialVel:
        """Return a vector."""
        return cx.RadialVel(d_r=Quantity([1, 2, 3, 4], "km/s"))

    @pytest.fixture(scope="class")
    def vector(self) -> cx.RadialPos:
        """Return a vector."""
        return cx.RadialPos(r=Quantity([1, 2, 3, 4], "kpc"))

    # ==========================================================================
    # represent_as

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_cartesian1d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianVel1D)``."""
        cart1d = difntl.represent_as(cx.CartesianVel1D, vector)

        assert isinstance(cart1d, cx.CartesianVel1D)
        assert jnp.array_equal(cart1d.d_x, Quantity([1, 2, 3, 4], "km/s"))

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_radial(self, difntl, vector):
        """Test ``difntl.represent_as(RadialVel)``."""
        # Jit can copy
        newvec = difntl.represent_as(cx.RadialVel, vector)
        assert jnp.array_equal(newvec, difntl)

        # The normal `represent_as` method should return the same object
        newvec = cx.represent_as(difntl, cx.RadialVel, vector)
        assert newvec is difntl

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_cartesian2d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianVel2D)``."""
        cart2d = difntl.represent_as(
            cx.CartesianVel2D, vector, d_y=Quantity([5, 6, 7, 8], "km")
        )

        assert isinstance(cart2d, cx.CartesianVel2D)
        assert jnp.array_equal(cart2d.d_x, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cart2d.d_y, Quantity([5, 6, 7, 8], "km"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_polar(self, difntl, vector):
        """Test ``difntl.represent_as(PolarVel)``."""
        polar = difntl.represent_as(
            cx.PolarVel, vector, d_phi=Quantity([0, 1, 2, 3], "rad")
        )

        assert isinstance(polar, cx.PolarVel)
        assert jnp.array_equal(polar.d_r, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(polar.d_phi, Quantity([0, 1, 2, 3], "rad"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_cartesian3d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianVel3D)``."""
        cart3d = difntl.represent_as(
            cx.CartesianVel3D,
            vector,
            d_y=Quantity([5, 6, 7, 8], "km"),
            d_z=Quantity([9, 10, 11, 12], "m"),
        )

        assert isinstance(cart3d, cx.CartesianVel3D)
        assert jnp.array_equal(cart3d.d_x, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cart3d.d_y, Quantity([5, 6, 7, 8], "km"))
        assert jnp.array_equal(cart3d.d_z, Quantity([9, 10, 11, 12], "m"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_spherical(self, difntl, vector):
        """Test ``difntl.represent_as(SphericalVel)``."""
        spherical = difntl.represent_as(
            cx.SphericalVel,
            vector,
            d_theta=Quantity([4, 5, 6, 7], "rad"),
            d_phi=Quantity([0, 1, 2, 3], "rad"),
        )

        assert isinstance(spherical, cx.SphericalVel)
        assert jnp.array_equal(spherical.d_r, Quantity([1, 2, 3, 4], "kpc"))
        assert spherical.d_theta == Quantity([4, 5, 6, 7], "rad")
        assert jnp.array_equal(spherical.d_phi, Quantity([0, 1, 2, 3], "rad"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_cylindrical(self, difntl, vector):
        """Test ``difntl.represent_as(CylindricalVel)``."""
        cylindrical = difntl.represent_as(
            cx.CylindricalVel,
            vector,
            d_phi=Quantity([0, 1, 2, 3], "rad"),
            d_z=Quantity([4, 5, 6, 7], "m"),
        )

        assert isinstance(cylindrical, cx.CylindricalVel)
        assert jnp.array_equal(cylindrical.d_rho, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cylindrical.d_phi, Quantity([0, 1, 2, 3], "rad"))
        assert jnp.array_equal(cylindrical.d_z, Quantity([4, 5, 6, 7], "m"))
