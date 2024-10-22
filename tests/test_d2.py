"""Test :mod:`coordinax.d2`."""

import pytest

import quaxed.numpy as jnp
from unxt import Quantity

import coordinax as cx
from .test_base import AbstractPosTest, AbstractVelTest


class AbstractPos2DTest(AbstractPosTest):
    """Test :class:`coordinax.AbstractPos2D`."""


class TestCartesianPos2D:
    """Test :class:`coordinax.CartesianPos2D`."""

    @pytest.fixture(scope="class")
    def vector(self) -> cx.CartesianPos2D:
        """Return a vector."""
        return cx.CartesianPos2D(
            x=Quantity([1, 2, 3, 4], "kpc"), y=Quantity([5, 6, 7, 8], "kpc")
        )

    # ==========================================================================
    # represent_as

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cartesian2d_to_cartesian1d(self, vector):
        """Test ``coordinax.represent_as(CartesianPos1D)``."""
        cart1d = vector.represent_as(cx.CartesianPos1D)

        assert isinstance(cart1d, cx.CartesianPos1D)
        assert jnp.array_equal(cart1d.x, Quantity([1, 2, 3, 4], "kpc"))

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cartesian2d_to_radial(self, vector):
        """Test ``coordinax.represent_as(RadialPos)``."""
        radial = vector.represent_as(cx.RadialPos)

        assert isinstance(radial, cx.RadialPos)
        assert jnp.array_equal(radial.r, jnp.hypot(vector.x, vector.y))

    def test_cartesian2d_to_cartesian2d(self, vector):
        """Test ``coordinax.represent_as(CartesianPos2D)``."""
        newvec = vector.represent_as(cx.CartesianPos2D)
        assert newvec is vector

    def test_cartesian2d_to_cartesian2d(self, vector):
        """Test ``coordinax.represent_as(CartesianPos2D)``."""
        # Jit can copy
        newvec = vector.represent_as(cx.CartesianPos2D)
        assert jnp.array_equal(newvec, vector)

        # The normal `represent_as` method should return the same object
        newvec = cx.represent_as(vector, cx.CartesianPos2D)
        assert newvec is vector

    def test_cartesian2d_to_polar(self, vector):
        """Test ``coordinax.represent_as(PolarPos)``."""
        polar = vector.represent_as(cx.PolarPos)

        assert isinstance(polar, cx.PolarPos)
        assert jnp.array_equal(polar.r, jnp.hypot(vector.x, vector.y))
        assert jnp.allclose(
            polar.phi,
            Quantity([1.3734008, 1.2490457, 1.1659045, 1.1071488], "rad"),
            atol=Quantity(1e-8, "deg"),
        )

    def test_cartesian2d_to_cartesian3d(self, vector):
        """Test ``coordinax.represent_as(CartesianPos3D)``."""
        cart3d = vector.represent_as(
            cx.CartesianPos3D, z=Quantity([9, 10, 11, 12], "m")
        )

        assert isinstance(cart3d, cx.CartesianPos3D)
        assert jnp.array_equal(cart3d.x, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cart3d.y, Quantity([5, 6, 7, 8], "kpc"))
        assert jnp.array_equal(cart3d.z, Quantity([9, 10, 11, 12], "m"))

    def test_cartesian2d_to_spherical(self, vector):
        """Test ``coordinax.represent_as(SphericalPos)``."""
        spherical = vector.represent_as(
            cx.SphericalPos, theta=Quantity([4, 5, 6, 7], "rad")
        )

        assert isinstance(spherical, cx.SphericalPos)
        assert jnp.array_equal(spherical.r, jnp.hypot(vector.x, vector.y))
        assert jnp.allclose(
            spherical.phi,
            Quantity([1.3734008, 1.2490457, 1.1659045, 1.1071488], "rad"),
            atol=Quantity(1e-8, "rad"),
        )
        assert jnp.array_equal(
            spherical.theta, Quantity(jnp.full(4, fill_value=jnp.pi / 2), "rad")
        )

    def test_cartesian2d_to_cylindrical(self, vector):
        """Test ``coordinax.represent_as(CylindricalPos)``."""
        cylindrical = vector.represent_as(
            cx.CylindricalPos, z=Quantity([9, 10, 11, 12], "m")
        )

        assert isinstance(cylindrical, cx.CylindricalPos)
        assert jnp.array_equal(cylindrical.rho, jnp.hypot(vector.x, vector.y))
        assert jnp.array_equal(
            cylindrical.phi,
            Quantity([1.3734008, 1.2490457, 1.1659045, 1.1071488], "rad"),
        )
        assert jnp.array_equal(cylindrical.z, Quantity([9, 10, 11, 12], "m"))


class TestPolarPos:
    """Test :class:`coordinax.PolarPos`."""

    @pytest.fixture(scope="class")
    def vector(self) -> cx.AbstractPos:
        """Return a vector."""
        return cx.PolarPos(
            r=Quantity([1, 2, 3, 4], "kpc"), phi=Quantity([0, 1, 2, 3], "rad")
        )

    # ==========================================================================
    # represent_as

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_polar_to_cartesian1d(self, vector):
        """Test ``coordinax.represent_as(CartesianPos1D)``."""
        cart1d = vector.represent_as(cx.CartesianPos1D)

        assert isinstance(cart1d, cx.CartesianPos1D)
        assert jnp.allclose(
            cart1d.x,
            Quantity([1.0, 1.0806047, -1.2484405, -3.95997], "kpc"),
            atol=Quantity(1e-8, "kpc"),
        )
        assert jnp.array_equal(cart1d.x, vector.r * jnp.cos(vector.phi))

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_polar_to_radial(self, vector):
        """Test ``coordinax.represent_as(RadialPos)``."""
        radial = vector.represent_as(cx.RadialPos)

        assert isinstance(radial, cx.RadialPos)
        assert jnp.array_equal(radial.r, Quantity([1, 2, 3, 4], "kpc"))

    def test_polar_to_cartesian2d(self, vector):
        """Test ``coordinax.represent_as(CartesianPos2D)``."""
        cart2d = vector.represent_as(cx.CartesianPos2D, y=Quantity([5, 6, 7, 8], "km"))

        assert isinstance(cart2d, cx.CartesianPos2D)
        assert jnp.array_equal(
            cart2d.x, Quantity([1.0, 1.0806046, -1.2484405, -3.95997], "kpc")
        )
        assert jnp.allclose(
            cart2d.x, (vector.r * jnp.cos(vector.phi)), atol=Quantity(1e-8, "kpc")
        )
        assert jnp.array_equal(
            cart2d.y, Quantity([0.0, 1.6829419, 2.7278922, 0.56448], "kpc")
        )
        assert jnp.allclose(
            cart2d.y, (vector.r * jnp.sin(vector.phi)), atol=Quantity(1e-8, "kpc")
        )

    def test_polar_to_polar(self, vector):
        """Test ``coordinax.represent_as(PolarPos)``."""
        # Jit can copy
        newvec = vector.represent_as(cx.PolarPos)
        assert jnp.array_equal(newvec, vector)

        # The normal `represent_as` method should return the same object
        newvec = cx.represent_as(vector, cx.PolarPos)
        assert newvec is vector

    def test_polar_to_cartesian3d(self, vector):
        """Test ``coordinax.represent_as(CartesianPos3D)``."""
        cart3d = vector.represent_as(
            cx.CartesianPos3D, z=Quantity([9, 10, 11, 12], "m")
        )

        assert isinstance(cart3d, cx.CartesianPos3D)
        assert jnp.array_equal(
            cart3d.x, Quantity([1.0, 1.0806046, -1.2484405, -3.95997], "kpc")
        )
        assert jnp.array_equal(
            cart3d.y, Quantity([0.0, 1.6829419, 2.7278922, 0.56448], "kpc")
        )
        assert jnp.array_equal(cart3d.z, Quantity([9, 10, 11, 12], "m"))

    def test_polar_to_spherical(self, vector):
        """Test ``coordinax.represent_as(SphericalPos)``."""
        spherical = vector.represent_as(
            cx.SphericalPos, theta=Quantity([4, 15, 60, 170], "deg")
        )

        assert isinstance(spherical, cx.SphericalPos)
        assert jnp.array_equal(spherical.r, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(spherical.theta, Quantity([4, 15, 60, 170], "deg"))
        assert jnp.array_equal(spherical.phi, Quantity([0, 1, 2, 3], "rad"))

    def test_polar_to_cylindrical(self, vector):
        """Test ``coordinax.represent_as(CylindricalPos)``."""
        cylindrical = vector.represent_as(
            cx.CylindricalPos, z=Quantity([9, 10, 11, 12], "m")
        )

        assert isinstance(cylindrical, cx.CylindricalPos)
        assert jnp.array_equal(cylindrical.rho, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cylindrical.phi, Quantity([0, 1, 2, 3], "rad"))
        assert jnp.array_equal(cylindrical.z, Quantity([9, 10, 11, 12], "m"))


class AbstractVel2DTest(AbstractVelTest):
    """Test :class:`coordinax.AbstractVel2D`."""


class TestCartesianVel2D(AbstractVel2DTest):
    """Test :class:`coordinax.CartesianVel2D`."""

    @pytest.fixture(scope="class")
    def difntl(self) -> cx.CartesianVel2D:
        """Return a differential."""
        return cx.CartesianVel2D(
            d_x=Quantity([1, 2, 3, 4], "km/s"),
            d_y=Quantity([5, 6, 7, 8], "km/s"),
        )

    @pytest.fixture(scope="class")
    def vector(self) -> cx.CartesianPos2D:
        """Return a vector."""
        return cx.CartesianPos2D(
            x=Quantity([1, 2, 3, 4], "kpc"), y=Quantity([5, 6, 7, 8], "km")
        )

    # ==========================================================================
    # represent_as

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_cartesian1d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianVel1D, vector)``."""
        cart1d = difntl.represent_as(cx.CartesianVel1D, vector)

        assert isinstance(cart1d, cx.CartesianVel1D)
        assert jnp.array_equal(cart1d.d_x, Quantity([1, 2, 3, 4], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_radial(self, difntl, vector):
        """Test ``difntl.represent_as(RadialVel, vector)``."""
        radial = difntl.represent_as(cx.RadialVel, vector)

        assert isinstance(radial, cx.RadialVel)
        assert jnp.array_equal(radial.d_r, Quantity([1, 2, 3, 4], "km/s"))

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_cartesian2d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianVel2D, vector)``."""
        # Jit can copy
        newvec = difntl.represent_as(cx.CartesianVel2D, vector)
        assert jnp.array_equal(newvec, difntl)

        # The normal `represent_as` method should return the same object
        newvec = cx.represent_as(difntl, cx.CartesianVel2D, vector)
        assert newvec is difntl

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_polar(self, difntl, vector):
        """Test ``difntl.represent_as(PolarVel, vector)``."""
        polar = difntl.represent_as(cx.PolarVel, vector)

        assert isinstance(polar, cx.PolarVel)
        assert jnp.array_equal(polar.d_r, Quantity([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(
            polar.d_phi,
            Quantity([5.0, 3.0, 2.3333335, 1.9999999], "km rad / (kpc s)"),
        )

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_cartesian3d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianVel3D, vector)``."""
        cart3d = difntl.represent_as(
            cx.CartesianVel3D, vector, d_z=Quantity([9, 10, 11, 12], "m/s")
        )

        assert isinstance(cart3d, cx.CartesianVel3D)
        assert jnp.array_equal(cart3d.d_x, Quantity([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(cart3d.d_y, Quantity([5, 6, 7, 8], "km/s"))
        assert jnp.array_equal(cart3d.d_z, Quantity([9, 10, 11, 12], "m/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_spherical(self, difntl, vector):
        """Test ``difntl.represent_as(SphericalVel, vector)``."""
        spherical = difntl.represent_as(
            cx.SphericalVel, vector, d_theta=Quantity([4, 5, 6, 7], "rad")
        )

        assert isinstance(spherical, cx.SphericalVel)
        assert jnp.array_equal(spherical.d_r, Quantity([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(spherical.d_theta, Quantity([4, 5, 6, 7], "rad"))
        assert jnp.array_equal(spherical.d_phi, Quantity([5, 6, 7, 8], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_cylindrical(self, difntl, vector):
        """Test ``difntl.represent_as(CylindricalVel, vector)``."""
        cylindrical = difntl.represent_as(
            cx.CylindricalVel, vector, d_z=Quantity([9, 10, 11, 12], "m/s")
        )

        assert isinstance(cylindrical, cx.CylindricalVel)
        assert jnp.array_equal(cylindrical.d_rho, Quantity([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(cylindrical.d_phi, Quantity([5, 6, 7, 8], "km/s"))
        assert jnp.array_equal(cylindrical.d_z, Quantity([9, 10, 11, 12], "m/s"))


class TestPolarVel(AbstractVel2DTest):
    """Test :class:`coordinax.PolarVel`."""

    @pytest.fixture(scope="class")
    def difntl(self) -> cx.PolarVel:
        """Return a differential."""
        return cx.PolarVel(
            d_r=Quantity([1, 2, 3, 4], "km/s"),
            d_phi=Quantity([5, 6, 7, 8], "mas/yr"),
        )

    @pytest.fixture(scope="class")
    def vector(self) -> cx.PolarPos:
        """Return a vector."""
        return cx.PolarPos(
            r=Quantity([1, 2, 3, 4], "kpc"), phi=Quantity([0, 1, 2, 3], "rad")
        )

    # ==========================================================================
    # represent_as

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_cartesian1d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianVel1D, vector)``."""
        cart1d = difntl.represent_as(cx.CartesianVel1D, vector)

        assert isinstance(cart1d, cx.CartesianVel1D)
        assert jnp.array_equal(cart1d.d_x, Quantity([1, 2, 3, 4], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_radial(self, difntl, vector):
        """Test ``difntl.represent_as(RadialVel, vector)``."""
        radial = difntl.represent_as(cx.RadialVel, vector)

        assert isinstance(radial, cx.RadialVel)
        assert jnp.array_equal(radial.d_r, Quantity([1, 2, 3, 4], "km/s"))

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_cartesian2d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianVel2D, vector)``."""
        cart2d = difntl.represent_as(cx.CartesianVel2D, vector)

        assert isinstance(cart2d, cx.CartesianVel2D)
        assert jnp.array_equal(
            cart2d.d_x, Quantity([1.0, -46.787014, -91.76889, -25.367176], "km/s")
        )
        assert jnp.array_equal(
            cart2d.d_y,
            Quantity([23.702353, 32.418385, -38.69947, -149.61249], "km/s"),
        )

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_polar(self, difntl, vector):
        """Test ``difntl.represent_as(PolarVel, vector)``."""
        # Jit can copy
        newvec = difntl.represent_as(cx.PolarVel, vector)
        assert all(newvec == difntl)

        # The normal `represent_as` method should return the same object
        newvec = cx.represent_as(difntl, cx.PolarVel, vector)
        assert newvec is difntl

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_cartesian3d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianVel3D, vector)``."""
        cart3d = difntl.represent_as(
            cx.CartesianVel3D, vector, d_z=Quantity([9, 10, 11, 12], "m/s")
        )

        assert isinstance(cart3d, cx.CartesianVel3D)
        assert jnp.array_equal(cart3d.d_x, Quantity([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(cart3d.d_y, Quantity([5, 6, 7, 8], "km/s"))
        assert jnp.array_equal(cart3d.d_z, Quantity([9, 10, 11, 12], "m/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_spherical(self, difntl, vector):
        """Test ``difntl.represent_as(SphericalVel, vector)``."""
        spherical = difntl.represent_as(
            cx.SphericalVel, vector, d_theta=Quantity([4, 5, 6, 7], "rad")
        )

        assert isinstance(spherical, cx.SphericalVel)
        assert jnp.array_equal(spherical.d_r, Quantity([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(spherical.d_theta, Quantity([4, 5, 6, 7], "rad"))
        assert jnp.array_equal(spherical.d_phi, Quantity([5, 6, 7, 8], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_cylindrical(self, difntl, vector):
        """Test ``difntl.represent_as(CylindricalVel, vector)``."""
        cylindrical = difntl.represent_as(
            cx.CylindricalVel, vector, d_z=Quantity([9, 10, 11, 12], "m/s")
        )

        assert isinstance(cylindrical, cx.CylindricalVel)
        assert jnp.array_equal(cylindrical.d_rho, Quantity([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(cylindrical.d_phi, Quantity([5, 6, 7, 8], "km/s"))
        assert jnp.array_equal(cylindrical.d_z, Quantity([9, 10, 11, 12], "m/s"))
