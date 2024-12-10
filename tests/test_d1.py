"""Test :mod:`coordinax.d1`."""

import pytest

import quaxed.numpy as jnp
import unxt as u

import coordinax as cx
from .test_base import AbstractPosTest, AbstractVelTest


class AbstractPos1DTest(AbstractPosTest):
    """Test :class:`coordinax.AbstractPos1D`."""

    # TODO: Add tests


class TestCartesianPos1D(AbstractPos1DTest):
    """Test :class:`coordinax.CartesianPos1D`."""

    @pytest.fixture(scope="class")
    def vector(self) -> cx.vecs.AbstractPos:
        """Return a vector."""
        return cx.vecs.CartesianPos1D(x=u.Quantity([1, 2, 3, 4], "kpc"))

    # ==========================================================================
    # vconvert

    def test_cartesian1d_to_cartesian1d(self, vector):
        """Test ``coordinax.vconvert(CartesianPos1D)``."""
        # Jit can copy
        newvec = vector.vconvert(cx.vecs.CartesianPos1D)
        assert jnp.array_equal(newvec, vector)

        # The normal `vconvert` method should return the same object
        newvec = cx.vconvert(cx.vecs.CartesianPos1D, vector)
        assert newvec is vector

    def test_cartesian1d_to_radial(self, vector):
        """Test ``coordinax.vconvert(RadialPos)``."""
        radial = vector.vconvert(cx.vecs.RadialPos)

        assert isinstance(radial, cx.vecs.RadialPos)
        assert jnp.array_equal(radial.r, u.Quantity([1, 2, 3, 4], "kpc"))

    def test_cartesian1d_to_cartesian2d(self, vector):
        """Test ``coordinax.vconvert(CartesianPos2D)``."""
        cart2d = vector.vconvert(
            cx.vecs.CartesianPos2D, y=u.Quantity([5, 6, 7, 8], "km")
        )

        assert isinstance(cart2d, cx.vecs.CartesianPos2D)
        assert jnp.array_equal(cart2d.x, u.Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cart2d.y, u.Quantity([5, 6, 7, 8], "km"))

    def test_cartesian1d_to_polar(self, vector):
        """Test ``coordinax.vconvert(PolarPos)``."""
        polar = vector.vconvert(cx.vecs.PolarPos, phi=u.Quantity([0, 1, 2, 3], "rad"))

        assert isinstance(polar, cx.vecs.PolarPos)
        assert jnp.array_equal(polar.r, u.Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(polar.phi, u.Quantity([0, 1, 2, 3], "rad"))

    def test_cartesian1d_to_cartesian3d(self, vector):
        """Test ``coordinax.vconvert(CartesianPos3D)``."""
        cart3d = vector.vconvert(
            cx.CartesianPos3D,
            y=u.Quantity([5, 6, 7, 8], "km"),
            z=u.Quantity([9, 10, 11, 12], "m"),
        )

        assert isinstance(cart3d, cx.CartesianPos3D)
        assert jnp.array_equal(cart3d.x, u.Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cart3d.y, u.Quantity([5, 6, 7, 8], "km"))
        assert jnp.array_equal(cart3d.z, u.Quantity([9, 10, 11, 12], "m"))

    def test_cartesian1d_to_spherical(self, vector):
        """Test ``coordinax.vconvert(SphericalPos)``."""
        spherical = vector.vconvert(
            cx.SphericalPos,
            theta=u.Quantity([4, 15, 60, 170], "deg"),
            phi=u.Quantity([0, 1, 2, 3], "rad"),
        )

        assert isinstance(spherical, cx.SphericalPos)
        assert jnp.array_equal(spherical.r, u.Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(spherical.theta, u.Quantity([4, 15, 60, 170], "deg"))
        assert jnp.array_equal(spherical.phi, u.Quantity([0, 1, 2, 3], "rad"))

    def test_cartesian1d_to_cylindrical(self, vector):
        """Test ``coordinax.vconvert(CylindricalPos)``."""
        cylindrical = vector.vconvert(
            cx.vecs.CylindricalPos,
            phi=u.Quantity([0, 1, 2, 3], "rad"),
            z=u.Quantity([4, 5, 6, 7], "m"),
        )

        assert isinstance(cylindrical, cx.vecs.CylindricalPos)
        assert jnp.array_equal(cylindrical.rho, u.Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cylindrical.phi, u.Quantity([0, 1, 2, 3], "rad"))
        assert jnp.array_equal(cylindrical.z, u.Quantity([4, 5, 6, 7], "m"))


class TestRadialPos(AbstractPos1DTest):
    """Test :class:`coordinax.RadialPos`."""

    @pytest.fixture(scope="class")
    def vector(self) -> cx.vecs.AbstractPos:
        """Return a vector."""
        return cx.vecs.RadialPos(r=u.Quantity([1, 2, 3, 4], "kpc"))

    # ==========================================================================
    # vconvert

    def test_radial_to_cartesian1d(self, vector):
        """Test ``coordinax.vconvert(CartesianPos1D)``."""
        cart1d = vector.vconvert(cx.vecs.CartesianPos1D)

        assert isinstance(cart1d, cx.vecs.CartesianPos1D)
        assert jnp.array_equal(cart1d.x, u.Quantity([1, 2, 3, 4], "kpc"))

    def test_radial_to_radial(self, vector):
        """Test ``coordinax.vconvert(RadialPos)``."""
        # Jit can copy
        newvec = vector.vconvert(cx.vecs.RadialPos)
        assert jnp.array_equal(newvec, vector)

        # The normal `vconvert` method should return the same object
        newvec = cx.vconvert(cx.vecs.RadialPos, vector)
        assert newvec is vector

    def test_radial_to_cartesian2d(self, vector):
        """Test ``coordinax.vconvert(CartesianPos2D)``."""
        cart2d = vector.vconvert(
            cx.vecs.CartesianPos2D, y=u.Quantity([5, 6, 7, 8], "km")
        )

        assert isinstance(cart2d, cx.vecs.CartesianPos2D)
        assert jnp.array_equal(cart2d.x, u.Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cart2d.y, u.Quantity([5, 6, 7, 8], "km"))

    def test_radial_to_polar(self, vector):
        """Test ``coordinax.vconvert(PolarPos)``."""
        polar = vector.vconvert(cx.vecs.PolarPos, phi=u.Quantity([0, 1, 2, 3], "rad"))

        assert isinstance(polar, cx.vecs.PolarPos)
        assert jnp.array_equal(polar.r, u.Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(polar.phi, u.Quantity([0, 1, 2, 3], "rad"))

    def test_radial_to_cartesian3d(self, vector):
        """Test ``coordinax.vconvert(CartesianPos3D)``."""
        cart3d = vector.vconvert(
            cx.CartesianPos3D,
            y=u.Quantity([5, 6, 7, 8], "km"),
            z=u.Quantity([9, 10, 11, 12], "m"),
        )

        assert isinstance(cart3d, cx.CartesianPos3D)
        assert jnp.array_equal(cart3d.x, u.Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cart3d.y, u.Quantity([5, 6, 7, 8], "km"))
        assert jnp.array_equal(cart3d.z, u.Quantity([9, 10, 11, 12], "m"))

    def test_radial_to_spherical(self, vector):
        """Test ``coordinax.vconvert(SphericalPos)``."""
        spherical = vector.vconvert(
            cx.SphericalPos,
            theta=u.Quantity([4, 15, 60, 170], "deg"),
            phi=u.Quantity([0, 1, 2, 3], "rad"),
        )

        assert isinstance(spherical, cx.SphericalPos)
        assert jnp.array_equal(spherical.r, u.Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(spherical.theta, u.Quantity([4, 15, 60, 170], "deg"))
        assert jnp.array_equal(spherical.phi, u.Quantity([0, 1, 2, 3], "rad"))

    def test_radial_to_cylindrical(self, vector):
        """Test ``coordinax.vconvert(CylindricalPos)``."""
        cylindrical = vector.vconvert(
            cx.vecs.CylindricalPos,
            phi=u.Quantity([0, 1, 2, 3], "rad"),
            z=u.Quantity([4, 5, 6, 7], "m"),
        )

        assert isinstance(cylindrical, cx.vecs.CylindricalPos)
        assert jnp.array_equal(cylindrical.rho, u.Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cylindrical.phi, u.Quantity([0, 1, 2, 3], "rad"))
        assert jnp.array_equal(cylindrical.z, u.Quantity([4, 5, 6, 7], "m"))


class AbstractVel1DTest(AbstractVelTest):
    """Test :class:`coordinax.AbstractVel1D`."""


class TestCartesianVel1D(AbstractVel1DTest):
    """Test :class:`coordinax.CartesianVel1D`."""

    @pytest.fixture(scope="class")
    def difntl(self) -> cx.vecs.CartesianVel1D:
        """Return a vector."""
        return cx.vecs.CartesianVel1D(d_x=u.Quantity([1.0, 2, 3, 4], "km/s"))

    @pytest.fixture(scope="class")
    def vector(self) -> cx.vecs.CartesianPos1D:
        """Return a vector."""
        return cx.vecs.CartesianPos1D(x=u.Quantity([1.0, 2, 3, 4], "kpc"))

    # ==========================================================================
    # vconvert

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_cartesian1d(self, difntl, vector):
        """Test ``difntl.vconvert(CartesianVel1D)``."""
        # Jit can copy
        newvec = difntl.vconvert(cx.vecs.CartesianVel1D, vector)
        assert jnp.array_equal(newvec, difntl)

        # The normal `vconvert` method should return the same object
        newvec = cx.vconvert(cx.vecs.CartesianVel1D, difntl, vector)
        assert newvec is difntl

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_radial(self, difntl, vector):
        """Test ``difntl.vconvert(RadialVel)``."""
        radial = difntl.vconvert(cx.vecs.RadialVel, vector)

        assert isinstance(radial, cx.vecs.RadialVel)
        assert jnp.array_equal(radial.d_r, u.Quantity([1, 2, 3, 4], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_cartesian2d(self, difntl, vector):
        """Test ``difntl.vconvert(CartesianVel2D)``."""
        cart2d = difntl.vconvert(
            cx.vecs.CartesianVel2D, vector, d_y=u.Quantity([5, 6, 7, 8], "km")
        )

        assert isinstance(cart2d, cx.vecs.CartesianVel2D)
        assert jnp.array_equal(cart2d.d_x, u.Quantity([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(cart2d.d_y, u.Quantity([5, 6, 7, 8], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_polar(self, difntl, vector):
        """Test ``difntl.vconvert(PolarVel)``."""
        polar = difntl.vconvert(
            cx.vecs.PolarVel, vector, d_phi=u.Quantity([0, 1, 2, 3], "rad")
        )

        assert isinstance(polar, cx.vecs.PolarVel)
        assert jnp.array_equal(polar.d_r, u.Quantity([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(polar.d_phi, u.Quantity([0, 1, 2, 3], "rad/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_cartesian3d(self, difntl, vector):
        """Test ``difntl.vconvert(CartesianVel3D)``."""
        cart3d = difntl.vconvert(
            cx.CartesianVel3D,
            vector,
            d_y=u.Quantity([5, 6, 7, 8], "km"),
            d_z=u.Quantity([9, 10, 11, 12], "m"),
        )

        assert isinstance(cart3d, cx.CartesianVel3D)
        assert jnp.array_equal(cart3d.d_x, u.Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cart3d.d_y, u.Quantity([5, 6, 7, 8], "km"))
        assert jnp.array_equal(cart3d.d_z, u.Quantity([9, 10, 11, 12], "m"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_spherical(self, difntl, vector):
        """Test ``difntl.vconvert(SphericalVel)``."""
        spherical = difntl.vconvert(
            cx.SphericalVel,
            vector,
            d_theta=u.Quantity([4, 5, 6, 7], "rad/s"),
            d_phi=u.Quantity([0, 1, 2, 3], "rad/s"),
        )

        assert isinstance(spherical, cx.SphericalVel)
        assert jnp.array_equal(spherical.d_r, u.Quantity([1, 2, 3, 4], "kpc"))
        assert spherical.d_theta == u.Quantity([4, 5, 6, 7], "rad/s")
        assert jnp.array_equal(spherical.d_phi, u.Quantity([0, 1, 2, 3], "rad/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_cylindrical(self, difntl, vector):
        """Test ``difntl.vconvert(CylindricalVel)``."""
        cylindrical = difntl.vconvert(
            cx.vecs.CylindricalVel,
            vector,
            d_phi=u.Quantity([0, 1, 2, 3], "rad/s"),
            d_z=u.Quantity([4, 5, 6, 7], "m/s"),
        )

        assert isinstance(cylindrical, cx.vecs.CylindricalVel)
        assert jnp.array_equal(cylindrical.d_rho, u.Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cylindrical.d_phi, u.Quantity([0, 1, 2, 3], "rad/s"))
        assert jnp.array_equal(cylindrical.d_z, u.Quantity([4, 5, 6, 7], "m/s"))


class TestRadialVel(AbstractVel1DTest):
    """Test :class:`coordinax.RadialVel`."""

    @pytest.fixture(scope="class")
    def difntl(self) -> cx.vecs.RadialVel:
        """Return a vector."""
        return cx.vecs.RadialVel(d_r=u.Quantity([1, 2, 3, 4], "km/s"))

    @pytest.fixture(scope="class")
    def vector(self) -> cx.vecs.RadialPos:
        """Return a vector."""
        return cx.vecs.RadialPos(r=u.Quantity([1, 2, 3, 4], "kpc"))

    # ==========================================================================
    # vconvert

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_cartesian1d(self, difntl, vector):
        """Test ``difntl.vconvert(CartesianVel1D)``."""
        cart1d = difntl.vconvert(cx.vecs.CartesianVel1D, vector)

        assert isinstance(cart1d, cx.vecs.CartesianVel1D)
        assert jnp.array_equal(cart1d.d_x, u.Quantity([1, 2, 3, 4], "km/s"))

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_radial(self, difntl, vector):
        """Test ``difntl.vconvert(RadialVel)``."""
        # Jit can copy
        newvec = difntl.vconvert(cx.vecs.RadialVel, vector)
        assert jnp.array_equal(newvec, difntl)

        # The normal `vconvert` method should return the same object
        newvec = cx.vconvert(cx.vecs.RadialVel, difntl, vector)
        assert newvec is difntl

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_cartesian2d(self, difntl, vector):
        """Test ``difntl.vconvert(CartesianVel2D)``."""
        cart2d = difntl.vconvert(
            cx.vecs.CartesianVel2D, vector, d_y=u.Quantity([5, 6, 7, 8], "km")
        )

        assert isinstance(cart2d, cx.vecs.CartesianVel2D)
        assert jnp.array_equal(cart2d.d_x, u.Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cart2d.d_y, u.Quantity([5, 6, 7, 8], "km"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_polar(self, difntl, vector):
        """Test ``difntl.vconvert(PolarVel)``."""
        polar = difntl.vconvert(
            cx.vecs.PolarVel, vector, d_phi=u.Quantity([0, 1, 2, 3], "rad")
        )

        assert isinstance(polar, cx.vecs.PolarVel)
        assert jnp.array_equal(polar.d_r, u.Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(polar.d_phi, u.Quantity([0, 1, 2, 3], "rad"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_cartesian3d(self, difntl, vector):
        """Test ``difntl.vconvert(CartesianVel3D)``."""
        cart3d = difntl.vconvert(
            cx.CartesianVel3D,
            vector,
            d_y=u.Quantity([5, 6, 7, 8], "km"),
            d_z=u.Quantity([9, 10, 11, 12], "m"),
        )

        assert isinstance(cart3d, cx.CartesianVel3D)
        assert jnp.array_equal(cart3d.d_x, u.Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cart3d.d_y, u.Quantity([5, 6, 7, 8], "km"))
        assert jnp.array_equal(cart3d.d_z, u.Quantity([9, 10, 11, 12], "m"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_spherical(self, difntl, vector):
        """Test ``difntl.vconvert(SphericalVel)``."""
        spherical = difntl.vconvert(
            cx.SphericalVel,
            vector,
            d_theta=u.Quantity([4, 5, 6, 7], "rad"),
            d_phi=u.Quantity([0, 1, 2, 3], "rad"),
        )

        assert isinstance(spherical, cx.SphericalVel)
        assert jnp.array_equal(spherical.d_r, u.Quantity([1, 2, 3, 4], "kpc"))
        assert spherical.d_theta == u.Quantity([4, 5, 6, 7], "rad")
        assert jnp.array_equal(spherical.d_phi, u.Quantity([0, 1, 2, 3], "rad"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_cylindrical(self, difntl, vector):
        """Test ``difntl.vconvert(CylindricalVel)``."""
        cylindrical = difntl.vconvert(
            cx.vecs.CylindricalVel,
            vector,
            d_phi=u.Quantity([0, 1, 2, 3], "rad"),
            d_z=u.Quantity([4, 5, 6, 7], "m"),
        )

        assert isinstance(cylindrical, cx.vecs.CylindricalVel)
        assert jnp.array_equal(cylindrical.d_rho, u.Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cylindrical.d_phi, u.Quantity([0, 1, 2, 3], "rad"))
        assert jnp.array_equal(cylindrical.d_z, u.Quantity([4, 5, 6, 7], "m"))
