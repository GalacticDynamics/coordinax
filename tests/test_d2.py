"""Test :mod:`coordinax.d2`."""

import pytest

import quaxed.array_api as xp
import quaxed.numpy as jnp
from unxt import Quantity

import coordinax as cx
from .test_base import AbstractPositionTest, AbstractVelocityTest


class AbstractPosition2DTest(AbstractPositionTest):
    """Test :class:`coordinax.AbstractPosition2D`."""


class TestCartesianPosition2D:
    """Test :class:`coordinax.CartesianPosition2D`."""

    @pytest.fixture(scope="class")
    def vector(self) -> cx.CartesianPosition2D:
        """Return a vector."""
        return cx.CartesianPosition2D(
            x=Quantity([1, 2, 3, 4], "kpc"), y=Quantity([5, 6, 7, 8], "kpc")
        )

    # ==========================================================================
    # represent_as

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cartesian2d_to_cartesian1d(self, vector):
        """Test ``coordinax.represent_as(CartesianPosition1D)``."""
        cart1d = vector.represent_as(cx.CartesianPosition1D)

        assert isinstance(cart1d, cx.CartesianPosition1D)
        assert jnp.array_equal(cart1d.x, Quantity([1, 2, 3, 4], "kpc"))

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cartesian2d_to_radial(self, vector):
        """Test ``coordinax.represent_as(RadialPosition)``."""
        radial = vector.represent_as(cx.RadialPosition)

        assert isinstance(radial, cx.RadialPosition)
        assert jnp.array_equal(radial.r, jnp.hypot(vector.x, vector.y))

    def test_cartesian2d_to_cartesian2d(self, vector):
        """Test ``coordinax.represent_as(CartesianPosition2D)``."""
        newvec = vector.represent_as(cx.CartesianPosition2D)
        assert newvec is vector

    def test_cartesian2d_to_cartesian2d(self, vector):
        """Test ``coordinax.represent_as(CartesianPosition2D)``."""
        # Jit can copy
        newvec = vector.represent_as(cx.CartesianPosition2D)
        assert newvec == vector

        # The normal `represent_as` method should return the same object
        newvec = cx.represent_as(vector, cx.CartesianPosition2D)
        assert newvec is vector

    def test_cartesian2d_to_polar(self, vector):
        """Test ``coordinax.represent_as(PolarPosition)``."""
        polar = vector.represent_as(cx.PolarPosition)

        assert isinstance(polar, cx.PolarPosition)
        assert jnp.array_equal(polar.r, jnp.hypot(vector.x, vector.y))
        assert jnp.allclose(
            polar.phi,
            Quantity([1.3734008, 1.2490457, 1.1659045, 1.1071488], "rad"),
            atol=Quantity(1e-8, "deg"),
        )

    def test_cartesian2d_to_cartesian3d(self, vector):
        """Test ``coordinax.represent_as(CartesianPosition3D)``."""
        cart3d = vector.represent_as(
            cx.CartesianPosition3D, z=Quantity([9, 10, 11, 12], "m")
        )

        assert isinstance(cart3d, cx.CartesianPosition3D)
        assert jnp.array_equal(cart3d.x, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cart3d.y, Quantity([5, 6, 7, 8], "kpc"))
        assert jnp.array_equal(cart3d.z, Quantity([9, 10, 11, 12], "m"))

    def test_cartesian2d_to_spherical(self, vector):
        """Test ``coordinax.represent_as(SphericalPosition)``."""
        spherical = vector.represent_as(
            cx.SphericalPosition, theta=Quantity([4, 5, 6, 7], "rad")
        )

        assert isinstance(spherical, cx.SphericalPosition)
        assert jnp.array_equal(spherical.r, jnp.hypot(vector.x, vector.y))
        assert jnp.allclose(
            spherical.phi,
            Quantity([1.3734008, 1.2490457, 1.1659045, 1.1071488], "rad"),
            atol=Quantity(1e-8, "rad"),
        )
        assert jnp.array_equal(
            spherical.theta, Quantity(xp.full(4, fill_value=xp.pi / 2), "rad")
        )

    def test_cartesian2d_to_cylindrical(self, vector):
        """Test ``coordinax.represent_as(CylindricalPosition)``."""
        cylindrical = vector.represent_as(
            cx.CylindricalPosition, z=Quantity([9, 10, 11, 12], "m")
        )

        assert isinstance(cylindrical, cx.CylindricalPosition)
        assert jnp.array_equal(cylindrical.rho, jnp.hypot(vector.x, vector.y))
        assert jnp.array_equal(
            cylindrical.phi,
            Quantity([1.3734008, 1.2490457, 1.1659045, 1.1071488], "rad"),
        )
        assert jnp.array_equal(cylindrical.z, Quantity([9, 10, 11, 12], "m"))


class TestPolarPosition:
    """Test :class:`coordinax.PolarPosition`."""

    @pytest.fixture(scope="class")
    def vector(self) -> cx.AbstractPosition:
        """Return a vector."""
        return cx.PolarPosition(
            r=Quantity([1, 2, 3, 4], "kpc"), phi=Quantity([0, 1, 2, 3], "rad")
        )

    # ==========================================================================
    # represent_as

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_polar_to_cartesian1d(self, vector):
        """Test ``coordinax.represent_as(CartesianPosition1D)``."""
        cart1d = vector.represent_as(cx.CartesianPosition1D)

        assert isinstance(cart1d, cx.CartesianPosition1D)
        assert jnp.allclose(
            cart1d.x,
            Quantity([1.0, 1.0806047, -1.2484405, -3.95997], "kpc"),
            atol=Quantity(1e-8, "kpc"),
        )
        assert jnp.array_equal(cart1d.x, vector.r * xp.cos(vector.phi))

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_polar_to_radial(self, vector):
        """Test ``coordinax.represent_as(RadialPosition)``."""
        radial = vector.represent_as(cx.RadialPosition)

        assert isinstance(radial, cx.RadialPosition)
        assert jnp.array_equal(radial.r, Quantity([1, 2, 3, 4], "kpc"))

    def test_polar_to_cartesian2d(self, vector):
        """Test ``coordinax.represent_as(CartesianPosition2D)``."""
        cart2d = vector.represent_as(
            cx.CartesianPosition2D, y=Quantity([5, 6, 7, 8], "km")
        )

        assert isinstance(cart2d, cx.CartesianPosition2D)
        assert jnp.array_equal(
            cart2d.x, Quantity([1.0, 1.0806046, -1.2484405, -3.95997], "kpc")
        )
        assert jnp.allclose(
            cart2d.x, (vector.r * xp.cos(vector.phi)), atol=Quantity(1e-8, "kpc")
        )
        assert jnp.array_equal(
            cart2d.y, Quantity([0.0, 1.6829419, 2.7278922, 0.56448], "kpc")
        )
        assert jnp.allclose(
            cart2d.y, (vector.r * xp.sin(vector.phi)), atol=Quantity(1e-8, "kpc")
        )

    def test_polar_to_polar(self, vector):
        """Test ``coordinax.represent_as(PolarPosition)``."""
        # Jit can copy
        newvec = vector.represent_as(cx.PolarPosition)
        assert newvec == vector

        # The normal `represent_as` method should return the same object
        newvec = cx.represent_as(vector, cx.PolarPosition)
        assert newvec is vector

    def test_polar_to_cartesian3d(self, vector):
        """Test ``coordinax.represent_as(CartesianPosition3D)``."""
        cart3d = vector.represent_as(
            cx.CartesianPosition3D, z=Quantity([9, 10, 11, 12], "m")
        )

        assert isinstance(cart3d, cx.CartesianPosition3D)
        assert jnp.array_equal(
            cart3d.x, Quantity([1.0, 1.0806046, -1.2484405, -3.95997], "kpc")
        )
        assert jnp.array_equal(
            cart3d.y, Quantity([0.0, 1.6829419, 2.7278922, 0.56448], "kpc")
        )
        assert jnp.array_equal(cart3d.z, Quantity([9, 10, 11, 12], "m"))

    def test_polar_to_spherical(self, vector):
        """Test ``coordinax.represent_as(SphericalPosition)``."""
        spherical = vector.represent_as(
            cx.SphericalPosition, theta=Quantity([4, 15, 60, 170], "deg")
        )

        assert isinstance(spherical, cx.SphericalPosition)
        assert jnp.array_equal(spherical.r, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(spherical.theta, Quantity([4, 15, 60, 170], "deg"))
        assert jnp.array_equal(spherical.phi, Quantity([0, 1, 2, 3], "rad"))

    def test_polar_to_cylindrical(self, vector):
        """Test ``coordinax.represent_as(CylindricalPosition)``."""
        cylindrical = vector.represent_as(
            cx.CylindricalPosition, z=Quantity([9, 10, 11, 12], "m")
        )

        assert isinstance(cylindrical, cx.CylindricalPosition)
        assert jnp.array_equal(cylindrical.rho, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cylindrical.phi, Quantity([0, 1, 2, 3], "rad"))
        assert jnp.array_equal(cylindrical.z, Quantity([9, 10, 11, 12], "m"))


class AbstractVelocity2DTest(AbstractVelocityTest):
    """Test :class:`coordinax.AbstractVelocity2D`."""


class TestCartesianVelocity2D(AbstractVelocity2DTest):
    """Test :class:`coordinax.CartesianVelocity2D`."""

    @pytest.fixture(scope="class")
    def difntl(self) -> cx.CartesianVelocity2D:
        """Return a differential."""
        return cx.CartesianVelocity2D(
            d_x=Quantity([1, 2, 3, 4], "km/s"),
            d_y=Quantity([5, 6, 7, 8], "km/s"),
        )

    @pytest.fixture(scope="class")
    def vector(self) -> cx.CartesianPosition2D:
        """Return a vector."""
        return cx.CartesianPosition2D(
            x=Quantity([1, 2, 3, 4], "kpc"), y=Quantity([5, 6, 7, 8], "km")
        )

    # ==========================================================================
    # represent_as

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_cartesian1d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianVelocity1D, vector)``."""
        cart1d = difntl.represent_as(cx.CartesianVelocity1D, vector)

        assert isinstance(cart1d, cx.CartesianVelocity1D)
        assert jnp.array_equal(cart1d.d_x, Quantity([1, 2, 3, 4], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_radial(self, difntl, vector):
        """Test ``difntl.represent_as(RadialVelocity, vector)``."""
        radial = difntl.represent_as(cx.RadialVelocity, vector)

        assert isinstance(radial, cx.RadialVelocity)
        assert jnp.array_equal(radial.d_r, Quantity([1, 2, 3, 4], "km/s"))

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_cartesian2d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianVelocity2D, vector)``."""
        # Jit can copy
        newvec = difntl.represent_as(cx.CartesianVelocity2D, vector)
        assert newvec == difntl

        # The normal `represent_as` method should return the same object
        newvec = cx.represent_as(difntl, cx.CartesianVelocity2D, vector)
        assert newvec is difntl

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_polar(self, difntl, vector):
        """Test ``difntl.represent_as(PolarVelocity, vector)``."""
        polar = difntl.represent_as(cx.PolarVelocity, vector)

        assert isinstance(polar, cx.PolarVelocity)
        assert jnp.array_equal(polar.d_r, Quantity([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(
            polar.d_phi,
            Quantity([5.0, 3.0, 2.3333335, 1.9999999], "km rad / (kpc s)"),
        )

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_cartesian3d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianVelocity3D, vector)``."""
        cart3d = difntl.represent_as(
            cx.CartesianVelocity3D, vector, d_z=Quantity([9, 10, 11, 12], "m/s")
        )

        assert isinstance(cart3d, cx.CartesianVelocity3D)
        assert jnp.array_equal(cart3d.d_x, Quantity([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(cart3d.d_y, Quantity([5, 6, 7, 8], "km/s"))
        assert jnp.array_equal(cart3d.d_z, Quantity([9, 10, 11, 12], "m/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_spherical(self, difntl, vector):
        """Test ``difntl.represent_as(SphericalVelocity, vector)``."""
        spherical = difntl.represent_as(
            cx.SphericalVelocity, vector, d_theta=Quantity([4, 5, 6, 7], "rad")
        )

        assert isinstance(spherical, cx.SphericalVelocity)
        assert jnp.array_equal(spherical.d_r, Quantity([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(spherical.d_theta, Quantity([4, 5, 6, 7], "rad"))
        assert jnp.array_equal(spherical.d_phi, Quantity([5, 6, 7, 8], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_cylindrical(self, difntl, vector):
        """Test ``difntl.represent_as(CylindricalVelocity, vector)``."""
        cylindrical = difntl.represent_as(
            cx.CylindricalVelocity, vector, d_z=Quantity([9, 10, 11, 12], "m/s")
        )

        assert isinstance(cylindrical, cx.CylindricalVelocity)
        assert jnp.array_equal(cylindrical.d_rho, Quantity([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(cylindrical.d_phi, Quantity([5, 6, 7, 8], "km/s"))
        assert jnp.array_equal(cylindrical.d_z, Quantity([9, 10, 11, 12], "m/s"))


class TestPolarVelocity(AbstractVelocity2DTest):
    """Test :class:`coordinax.PolarVelocity`."""

    @pytest.fixture(scope="class")
    def difntl(self) -> cx.PolarVelocity:
        """Return a differential."""
        return cx.PolarVelocity(
            d_r=Quantity([1, 2, 3, 4], "km/s"),
            d_phi=Quantity([5, 6, 7, 8], "mas/yr"),
        )

    @pytest.fixture(scope="class")
    def vector(self) -> cx.PolarPosition:
        """Return a vector."""
        return cx.PolarPosition(
            r=Quantity([1, 2, 3, 4], "kpc"), phi=Quantity([0, 1, 2, 3], "rad")
        )

    # ==========================================================================
    # represent_as

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_cartesian1d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianVelocity1D, vector)``."""
        cart1d = difntl.represent_as(cx.CartesianVelocity1D, vector)

        assert isinstance(cart1d, cx.CartesianVelocity1D)
        assert jnp.array_equal(cart1d.d_x, Quantity([1, 2, 3, 4], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_radial(self, difntl, vector):
        """Test ``difntl.represent_as(RadialVelocity, vector)``."""
        radial = difntl.represent_as(cx.RadialVelocity, vector)

        assert isinstance(radial, cx.RadialVelocity)
        assert jnp.array_equal(radial.d_r, Quantity([1, 2, 3, 4], "km/s"))

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_cartesian2d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianVelocity2D, vector)``."""
        cart2d = difntl.represent_as(cx.CartesianVelocity2D, vector)

        assert isinstance(cart2d, cx.CartesianVelocity2D)
        assert jnp.array_equal(
            cart2d.d_x, Quantity([1.0, -46.787014, -91.76889, -25.367176], "km/s")
        )
        assert jnp.array_equal(
            cart2d.d_y,
            Quantity([23.702353, 32.418385, -38.69947, -149.61249], "km/s"),
        )

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_polar(self, difntl, vector):
        """Test ``difntl.represent_as(PolarVelocity, vector)``."""
        # Jit can copy
        newvec = difntl.represent_as(cx.PolarVelocity, vector)
        assert newvec == difntl

        # The normal `represent_as` method should return the same object
        newvec = cx.represent_as(difntl, cx.PolarVelocity, vector)
        assert newvec is difntl

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_cartesian3d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianVelocity3D, vector)``."""
        cart3d = difntl.represent_as(
            cx.CartesianVelocity3D, vector, d_z=Quantity([9, 10, 11, 12], "m/s")
        )

        assert isinstance(cart3d, cx.CartesianVelocity3D)
        assert jnp.array_equal(cart3d.d_x, Quantity([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(cart3d.d_y, Quantity([5, 6, 7, 8], "km/s"))
        assert jnp.array_equal(cart3d.d_z, Quantity([9, 10, 11, 12], "m/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_spherical(self, difntl, vector):
        """Test ``difntl.represent_as(SphericalVelocity, vector)``."""
        spherical = difntl.represent_as(
            cx.SphericalVelocity, vector, d_theta=Quantity([4, 5, 6, 7], "rad")
        )

        assert isinstance(spherical, cx.SphericalVelocity)
        assert jnp.array_equal(spherical.d_r, Quantity([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(spherical.d_theta, Quantity([4, 5, 6, 7], "rad"))
        assert jnp.array_equal(spherical.d_phi, Quantity([5, 6, 7, 8], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_cylindrical(self, difntl, vector):
        """Test ``difntl.represent_as(CylindricalVelocity, vector)``."""
        cylindrical = difntl.represent_as(
            cx.CylindricalVelocity, vector, d_z=Quantity([9, 10, 11, 12], "m/s")
        )

        assert isinstance(cylindrical, cx.CylindricalVelocity)
        assert jnp.array_equal(cylindrical.d_rho, Quantity([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(cylindrical.d_phi, Quantity([5, 6, 7, 8], "km/s"))
        assert jnp.array_equal(cylindrical.d_z, Quantity([9, 10, 11, 12], "m/s"))
