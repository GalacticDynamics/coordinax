"""Test :mod:`coordinax.d1`."""

import pytest
from test_base import AbstractPosTest, AbstractVelTest

import quaxed.numpy as jnp
import unxt as u

import coordinax.vecs as cxv


class AbstractPos1DTest(AbstractPosTest):
    """Test `coordinax.AbstractPos1D`."""

    # TODO: Add tests


class TestCartesianPos1D(AbstractPos1DTest):
    """Test `coordinax.CartesianPos1D`."""

    @pytest.fixture(scope="class")
    def vector(self) -> cxv.AbstractPos:
        """Return a vector."""
        return cxv.CartesianPos1D(x=u.Q([1, 2, 3, 4], "kpc"))

    # ==========================================================================
    # vconvert

    def test_cartesian1d_to_cartesian1d(self, vector):
        """Test ``coordinax.vconvert(CartesianPos1D)``."""
        # Jit can copy
        newvec = vector.vconvert(cxv.CartesianPos1D)
        assert jnp.array_equal(newvec, vector)

        # The normal `vconvert` method should return the same object
        newvec = cxv.vconvert(cxv.CartesianPos1D, vector)
        assert newvec is vector

    def test_cartesian1d_to_radial(self, vector):
        """Test ``coordinax.vconvert(RadialPos)``."""
        radial = vector.vconvert(cxv.RadialPos)

        assert isinstance(radial, cxv.RadialPos)
        assert jnp.array_equal(radial.r, u.Q([1, 2, 3, 4], "kpc"))

    def test_cartesian1d_to_cartesian2d(self, vector):
        """Test ``coordinax.vconvert(CartesianPos2D)``."""
        cart2d = vector.vconvert(cxv.CartesianPos2D, y=u.Q([5, 6, 7, 8], "km"))

        assert isinstance(cart2d, cxv.CartesianPos2D)
        assert jnp.array_equal(cart2d.x, u.Q([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cart2d.y, u.Q([5, 6, 7, 8], "km"))

    def test_cartesian1d_to_polar(self, vector):
        """Test ``coordinax.vconvert(PolarPos)``."""
        polar = vector.vconvert(cxv.PolarPos, phi=u.Q([0, 1, 2, 3], "rad"))

        assert isinstance(polar, cxv.PolarPos)
        assert jnp.array_equal(polar.r, u.Q([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(polar.phi, u.Q([0, 1, 2, 3], "rad"))

    def test_cartesian1d_to_cartesian3d(self, vector):
        """Test ``coordinax.vconvert(CartesianPos3D)``."""
        cart3d = vector.vconvert(
            cxv.CartesianPos3D,
            y=u.Q([5, 6, 7, 8], "km"),
            z=u.Q([9, 10, 11, 12], "m"),
        )

        assert isinstance(cart3d, cxv.CartesianPos3D)
        assert jnp.array_equal(cart3d.x, u.Q([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cart3d.y, u.Q([5, 6, 7, 8], "km"))
        assert jnp.array_equal(cart3d.z, u.Q([9, 10, 11, 12], "m"))

    def test_cartesian1d_to_spherical(self, vector):
        """Test ``coordinax.vconvert(SphericalPos)``."""
        spherical = vector.vconvert(
            cxv.SphericalPos,
            theta=u.Q([4, 15, 60, 170], "deg"),
            phi=u.Q([0, 1, 2, 3], "rad"),
        )

        assert isinstance(spherical, cxv.SphericalPos)
        assert jnp.array_equal(spherical.r, u.Q([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(spherical.theta, u.Q([4, 15, 60, 170], "deg"))
        assert jnp.array_equal(spherical.phi, u.Q([0, 1, 2, 3], "rad"))

    def test_cartesian1d_to_cylindrical(self, vector):
        """Test ``coordinax.vconvert(CylindricalPos)``."""
        cylindrical = vector.vconvert(
            cxv.CylindricalPos,
            phi=u.Q([0, 1, 2, 3], "rad"),
            z=u.Q([4, 5, 6, 7], "m"),
        )

        assert isinstance(cylindrical, cxv.CylindricalPos)
        assert jnp.array_equal(cylindrical.rho, u.Q([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cylindrical.phi, u.Q([0, 1, 2, 3], "rad"))
        assert jnp.array_equal(cylindrical.z, u.Q([4, 5, 6, 7], "m"))


class TestRadialPos(AbstractPos1DTest):
    """Test `coordinax.RadialPos`."""

    @pytest.fixture(scope="class")
    def vector(self) -> cxv.AbstractPos:
        """Return a vector."""
        return cxv.RadialPos(r=u.Q([1, 2, 3, 4], "kpc"))

    # ==========================================================================
    # vconvert

    def test_radial_to_cartesian1d(self, vector):
        """Test ``coordinax.vconvert(CartesianPos1D)``."""
        cart1d = vector.vconvert(cxv.CartesianPos1D)

        assert isinstance(cart1d, cxv.CartesianPos1D)
        assert jnp.array_equal(cart1d.x, u.Q([1, 2, 3, 4], "kpc"))

    def test_radial_to_radial(self, vector):
        """Test ``coordinax.vconvert(RadialPos)``."""
        # Jit can copy
        newvec = vector.vconvert(cxv.RadialPos)
        assert jnp.array_equal(newvec, vector)

        # The normal `vconvert` method should return the same object
        newvec = cxv.vconvert(cxv.RadialPos, vector)
        assert newvec is vector

    def test_radial_to_cartesian2d(self, vector):
        """Test ``coordinax.vconvert(CartesianPos2D)``."""
        cart2d = vector.vconvert(cxv.CartesianPos2D, y=u.Q([5, 6, 7, 8], "km"))

        assert isinstance(cart2d, cxv.CartesianPos2D)
        assert jnp.array_equal(cart2d.x, u.Q([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cart2d.y, u.Q([5, 6, 7, 8], "km"))

    def test_radial_to_polar(self, vector):
        """Test ``coordinax.vconvert(PolarPos)``."""
        polar = vector.vconvert(cxv.PolarPos, phi=u.Q([0, 1, 2, 3], "rad"))

        assert isinstance(polar, cxv.PolarPos)
        assert jnp.array_equal(polar.r, u.Q([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(polar.phi, u.Q([0, 1, 2, 3], "rad"))

    def test_radial_to_cartesian3d(self, vector):
        """Test ``coordinax.vconvert(CartesianPos3D)``."""
        cart3d = vector.vconvert(
            cxv.CartesianPos3D,
            y=u.Q([5, 6, 7, 8], "km"),
            z=u.Q([9, 10, 11, 12], "m"),
        )

        assert isinstance(cart3d, cxv.CartesianPos3D)
        assert jnp.array_equal(cart3d.x, u.Q([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cart3d.y, u.Q([5, 6, 7, 8], "km"))
        assert jnp.array_equal(cart3d.z, u.Q([9, 10, 11, 12], "m"))

    def test_radial_to_spherical(self, vector):
        """Test ``coordinax.vconvert(SphericalPos)``."""
        spherical = vector.vconvert(
            cxv.SphericalPos,
            theta=u.Q([4, 15, 60, 170], "deg"),
            phi=u.Q([0, 1, 2, 3], "rad"),
        )

        assert isinstance(spherical, cxv.SphericalPos)
        assert jnp.array_equal(spherical.r, u.Q([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(spherical.theta, u.Q([4, 15, 60, 170], "deg"))
        assert jnp.array_equal(spherical.phi, u.Q([0, 1, 2, 3], "rad"))

    def test_radial_to_cylindrical(self, vector):
        """Test ``coordinax.vconvert(CylindricalPos)``."""
        cylindrical = vector.vconvert(
            cxv.CylindricalPos,
            phi=u.Q([0, 1, 2, 3], "rad"),
            z=u.Q([4, 5, 6, 7], "m"),
        )

        assert isinstance(cylindrical, cxv.CylindricalPos)
        assert jnp.array_equal(cylindrical.rho, u.Q([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cylindrical.phi, u.Q([0, 1, 2, 3], "rad"))
        assert jnp.array_equal(cylindrical.z, u.Q([4, 5, 6, 7], "m"))


class AbstractVel1DTest(AbstractVelTest):
    """Test `coordinax.AbstractVel1D`."""


class TestCartesianVel1D(AbstractVel1DTest):
    """Test `coordinax.CartesianVel1D`."""

    @pytest.fixture(scope="class")
    def difntl(self) -> cxv.CartesianVel1D:
        """Return a vector."""
        return cxv.CartesianVel1D(x=u.Q([1.0, 2, 3, 4], "km/s"))

    @pytest.fixture(scope="class")
    def vector(self) -> cxv.CartesianPos1D:
        """Return a vector."""
        return cxv.CartesianPos1D(x=u.Q([1.0, 2, 3, 4], "kpc"))

    # ==========================================================================
    # vconvert

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_cartesian1d(self, difntl, vector):
        """Test ``difntl.vconvert(CartesianVel1D)``."""
        # Jit can copy
        newvec = difntl.vconvert(cxv.CartesianVel1D, vector)
        assert jnp.array_equal(newvec, difntl)

        # The normal `vconvert` method should return the same object
        newvec = cxv.vconvert(cxv.CartesianVel1D, difntl, vector)
        assert newvec is difntl

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_radial(self, difntl, vector):
        """Test ``difntl.vconvert(RadialVel)``."""
        radial = difntl.vconvert(cxv.RadialVel, vector)

        assert isinstance(radial, cxv.RadialVel)
        assert jnp.array_equal(radial.r, u.Q([1, 2, 3, 4], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_cartesian2d(self, difntl, vector):
        """Test ``difntl.vconvert(CartesianVel2D)``."""
        cart2d = difntl.vconvert(cxv.CartesianVel2D, vector, y=u.Q([5, 6, 7, 8], "km"))

        assert isinstance(cart2d, cxv.CartesianVel2D)
        assert jnp.array_equal(cart2d.x, u.Q([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(cart2d.y, u.Q([5, 6, 7, 8], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_polar(self, difntl, vector):
        """Test ``difntl.vconvert(PolarVel)``."""
        polar = difntl.vconvert(cxv.PolarVel, vector, phi=u.Q([0, 1, 2, 3], "rad"))

        assert isinstance(polar, cxv.PolarVel)
        assert jnp.array_equal(polar.r, u.Q([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(polar.phi, u.Q([0, 1, 2, 3], "rad/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_cartesian3d(self, difntl, vector):
        """Test ``difntl.vconvert(CartesianVel3D)``."""
        cart3d = difntl.vconvert(
            cxv.CartesianVel3D,
            vector,
            y=u.Q([5, 6, 7, 8], "km"),
            z=u.Q([9, 10, 11, 12], "m"),
        )

        assert isinstance(cart3d, cxv.CartesianVel3D)
        assert jnp.array_equal(cart3d.x, u.Q([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cart3d.y, u.Q([5, 6, 7, 8], "km"))
        assert jnp.array_equal(cart3d.z, u.Q([9, 10, 11, 12], "m"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_spherical(self, difntl, vector):
        """Test ``difntl.vconvert(SphericalVel)``."""
        spherical = difntl.vconvert(
            cxv.SphericalVel,
            vector,
            d_theta=u.Q([4, 5, 6, 7], "rad/s"),
            d_phi=u.Q([0, 1, 2, 3], "rad/s"),
        )

        assert isinstance(spherical, cxv.SphericalVel)
        assert jnp.array_equal(spherical.d_r, u.Q([1, 2, 3, 4], "kpc"))
        assert spherical.d_theta == u.Q([4, 5, 6, 7], "rad/s")
        assert jnp.array_equal(spherical.d_phi, u.Q([0, 1, 2, 3], "rad/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian1d_to_cylindrical(self, difntl, vector):
        """Test ``difntl.vconvert(CylindricalVel)``."""
        cylindrical = difntl.vconvert(
            cxv.CylindricalVel,
            vector,
            d_phi=u.Q([0, 1, 2, 3], "rad/s"),
            d_z=u.Q([4, 5, 6, 7], "m/s"),
        )

        assert isinstance(cylindrical, cxv.CylindricalVel)
        assert jnp.array_equal(cylindrical.d_rho, u.Q([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cylindrical.d_phi, u.Q([0, 1, 2, 3], "rad/s"))
        assert jnp.array_equal(cylindrical.d_z, u.Q([4, 5, 6, 7], "m/s"))


class TestRadialVel(AbstractVel1DTest):
    """Test `coordinax.RadialVel`."""

    @pytest.fixture(scope="class")
    def difntl(self) -> cxv.RadialVel:
        """Return a vector."""
        return cxv.RadialVel(r=u.Q([1, 2, 3, 4], "km/s"))

    @pytest.fixture(scope="class")
    def vector(self) -> cxv.RadialPos:
        """Return a vector."""
        return cxv.RadialPos(r=u.Q([1, 2, 3, 4], "kpc"))

    # ==========================================================================
    # vconvert

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_cartesian1d(self, difntl, vector):
        """Test ``difntl.vconvert(CartesianVel1D)``."""
        cart1d = difntl.vconvert(cxv.CartesianVel1D, vector)

        assert isinstance(cart1d, cxv.CartesianVel1D)
        assert jnp.array_equal(cart1d.x, u.Q([1, 2, 3, 4], "km/s"))

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_radial(self, difntl, vector):
        """Test ``difntl.vconvert(RadialVel)``."""
        # Jit can copy
        newvec = difntl.vconvert(cxv.RadialVel, vector)
        assert jnp.array_equal(newvec, difntl)

        # The normal `vconvert` method should return the same object
        newvec = cxv.vconvert(cxv.RadialVel, difntl, vector)
        assert newvec is difntl

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_cartesian2d(self, difntl, vector):
        """Test ``difntl.vconvert(CartesianVel2D)``."""
        cart2d = difntl.vconvert(cxv.CartesianVel2D, vector, y=u.Q([5, 6, 7, 8], "km"))

        assert isinstance(cart2d, cxv.CartesianVel2D)
        assert jnp.array_equal(cart2d.x, u.Q([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cart2d.y, u.Q([5, 6, 7, 8], "km"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_polar(self, difntl, vector):
        """Test ``difntl.vconvert(PolarVel)``."""
        polar = difntl.vconvert(cxv.PolarVel, vector, phi=u.Q([0, 1, 2, 3], "rad"))

        assert isinstance(polar, cxv.PolarVel)
        assert jnp.array_equal(polar.r, u.Q([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(polar.phi, u.Q([0, 1, 2, 3], "rad"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_cartesian3d(self, difntl, vector):
        """Test ``difntl.vconvert(CartesianVel3D)``."""
        cart3d = difntl.vconvert(
            cxv.CartesianVel3D,
            vector,
            y=u.Q([5, 6, 7, 8], "km"),
            z=u.Q([9, 10, 11, 12], "m"),
        )

        assert isinstance(cart3d, cxv.CartesianVel3D)
        assert jnp.array_equal(cart3d.x, u.Q([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cart3d.y, u.Q([5, 6, 7, 8], "km"))
        assert jnp.array_equal(cart3d.z, u.Q([9, 10, 11, 12], "m"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_spherical(self, difntl, vector):
        """Test ``difntl.vconvert(SphericalVel)``."""
        spherical = difntl.vconvert(
            cxv.SphericalVel,
            vector,
            theta=u.Q([4, 5, 6, 7], "rad"),
            phi=u.Q([0, 1, 2, 3], "rad"),
        )

        assert isinstance(spherical, cxv.SphericalVel)
        assert jnp.array_equal(spherical.r, u.Q([1, 2, 3, 4], "kpc"))
        assert spherical.theta == u.Q([4, 5, 6, 7], "rad")
        assert jnp.array_equal(spherical.phi, u.Q([0, 1, 2, 3], "rad"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_radial_to_cylindrical(self, difntl, vector):
        """Test ``difntl.vconvert(CylindricalVel)``."""
        cylindrical = difntl.vconvert(
            cxv.CylindricalVel,
            vector,
            phi=u.Q([0, 1, 2, 3], "rad"),
            z=u.Q([4, 5, 6, 7], "m"),
        )

        assert isinstance(cylindrical, cxv.CylindricalVel)
        assert jnp.array_equal(cylindrical.rho, u.Q([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cylindrical.phi, u.Q([0, 1, 2, 3], "rad"))
        assert jnp.array_equal(cylindrical.z, u.Q([4, 5, 6, 7], "m"))
