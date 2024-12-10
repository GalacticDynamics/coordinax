"""Test :mod:`coordinax.d2`."""

import pytest

import quaxed.numpy as jnp
import unxt as u

import coordinax as cx
from .test_base import AbstractPosTest, AbstractVelTest


class AbstractPos2DTest(AbstractPosTest):
    """Test :class:`coordinax.AbstractPos2D`."""


class TestCartesianPos2D:
    """Test :class:`coordinax.vecs.CartesianPos2D`."""

    @pytest.fixture(scope="class")
    def vector(self) -> cx.vecs.CartesianPos2D:
        """Return a vector."""
        return cx.vecs.CartesianPos2D(
            x=u.Quantity([1, 2, 3, 4], "kpc"), y=u.Quantity([5, 6, 7, 8], "kpc")
        )

    # ==========================================================================
    # vconvert

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cartesian2d_to_cartesian1d(self, vector):
        """Test ``coordinax.vconvert(CartesianPos1D)``."""
        cart1d = vector.vconvert(cx.vecs.CartesianPos1D)

        assert isinstance(cart1d, cx.vecs.CartesianPos1D)
        assert jnp.array_equal(cart1d.x, u.Quantity([1, 2, 3, 4], "kpc"))

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cartesian2d_to_radial(self, vector):
        """Test ``coordinax.vconvert(RadialPos)``."""
        radial = vector.vconvert(cx.vecs.RadialPos)

        assert isinstance(radial, cx.vecs.RadialPos)
        assert jnp.array_equal(radial.r, jnp.hypot(vector.x, vector.y))

    def test_cartesian2d_to_cartesian2d(self, vector):
        """Test ``coordinax.vconvert(CartesianPos2D)``."""
        newvec = vector.vconvert(cx.vecs.CartesianPos2D)
        assert newvec is vector

    def test_cartesian2d_to_cartesian2d(self, vector):
        """Test ``coordinax.vconvert(CartesianPos2D)``."""
        # Jit can copy
        newvec = vector.vconvert(cx.vecs.CartesianPos2D)
        assert jnp.array_equal(newvec, vector)

        # The normal `vconvert` method should return the same object
        newvec = cx.vconvert(cx.vecs.CartesianPos2D, vector)
        assert newvec is vector

    def test_cartesian2d_to_polar(self, vector):
        """Test ``coordinax.vconvert(PolarPos)``."""
        polar = vector.vconvert(cx.vecs.PolarPos)

        assert isinstance(polar, cx.vecs.PolarPos)
        assert jnp.array_equal(polar.r, jnp.hypot(vector.x, vector.y))
        assert jnp.allclose(
            polar.phi,
            u.Quantity([1.3734008, 1.2490457, 1.1659045, 1.1071488], "rad"),
            atol=u.Quantity(1e-8, "deg"),
        )

    def test_cartesian2d_to_cartesian3d(self, vector):
        """Test ``coordinax.vconvert(CartesianPos3D)``."""
        cart3d = vector.vconvert(cx.CartesianPos3D, z=u.Quantity([9, 10, 11, 12], "m"))

        assert isinstance(cart3d, cx.CartesianPos3D)
        assert jnp.array_equal(cart3d.x, u.Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cart3d.y, u.Quantity([5, 6, 7, 8], "kpc"))
        assert jnp.array_equal(cart3d.z, u.Quantity([9, 10, 11, 12], "m"))

    def test_cartesian2d_to_spherical(self, vector):
        """Test ``coordinax.vconvert(SphericalPos)``."""
        spherical = vector.vconvert(
            cx.SphericalPos, theta=u.Quantity([4, 5, 6, 7], "rad")
        )

        assert isinstance(spherical, cx.SphericalPos)
        assert jnp.array_equal(spherical.r, jnp.hypot(vector.x, vector.y))
        assert jnp.allclose(
            spherical.phi,
            u.Quantity([1.3734008, 1.2490457, 1.1659045, 1.1071488], "rad"),
            atol=u.Quantity(1e-8, "rad"),
        )
        assert jnp.array_equal(
            spherical.theta, u.Quantity(jnp.full(4, fill_value=jnp.pi / 2), "rad")
        )

    def test_cartesian2d_to_cylindrical(self, vector):
        """Test ``coordinax.vconvert(CylindricalPos)``."""
        cylindrical = vector.vconvert(
            cx.vecs.CylindricalPos, z=u.Quantity([9, 10, 11, 12], "m")
        )

        assert isinstance(cylindrical, cx.vecs.CylindricalPos)
        assert jnp.array_equal(cylindrical.rho, jnp.hypot(vector.x, vector.y))
        assert jnp.array_equal(
            cylindrical.phi,
            u.Quantity([1.3734008, 1.2490457, 1.1659045, 1.1071488], "rad"),
        )
        assert jnp.array_equal(cylindrical.z, u.Quantity([9, 10, 11, 12], "m"))


class TestPolarPos:
    """Test :class:`coordinax.PolarPos`."""

    @pytest.fixture(scope="class")
    def vector(self) -> cx.vecs.AbstractPos:
        """Return a vector."""
        return cx.vecs.PolarPos(
            r=u.Quantity([1, 2, 3, 4], "kpc"), phi=u.Quantity([0, 1, 2, 3], "rad")
        )

    # ==========================================================================
    # vconvert

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_polar_to_cartesian1d(self, vector):
        """Test ``coordinax.vconvert(CartesianPos1D)``."""
        cart1d = vector.vconvert(cx.vecs.CartesianPos1D)

        assert isinstance(cart1d, cx.vecs.CartesianPos1D)
        assert jnp.allclose(
            cart1d.x,
            u.Quantity([1.0, 1.0806047, -1.2484405, -3.95997], "kpc"),
            atol=u.Quantity(1e-8, "kpc"),
        )
        assert jnp.array_equal(cart1d.x, vector.r * jnp.cos(vector.phi))

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_polar_to_radial(self, vector):
        """Test ``coordinax.vconvert(RadialPos)``."""
        radial = vector.vconvert(cx.vecs.RadialPos)

        assert isinstance(radial, cx.vecs.RadialPos)
        assert jnp.array_equal(radial.r, u.Quantity([1, 2, 3, 4], "kpc"))

    def test_polar_to_cartesian2d(self, vector):
        """Test ``coordinax.vconvert(CartesianPos2D)``."""
        cart2d = vector.vconvert(
            cx.vecs.CartesianPos2D, y=u.Quantity([5, 6, 7, 8], "km")
        )

        assert isinstance(cart2d, cx.vecs.CartesianPos2D)
        assert jnp.array_equal(
            cart2d.x, u.Quantity([1.0, 1.0806046, -1.2484405, -3.95997], "kpc")
        )
        assert jnp.allclose(
            cart2d.x, (vector.r * jnp.cos(vector.phi)), atol=u.Quantity(1e-8, "kpc")
        )
        assert jnp.array_equal(
            cart2d.y, u.Quantity([0.0, 1.6829419, 2.7278922, 0.56448], "kpc")
        )
        assert jnp.allclose(
            cart2d.y, (vector.r * jnp.sin(vector.phi)), atol=u.Quantity(1e-8, "kpc")
        )

    def test_polar_to_polar(self, vector):
        """Test ``coordinax.vconvert(PolarPos)``."""
        # Jit can copy
        newvec = vector.vconvert(cx.vecs.PolarPos)
        assert jnp.array_equal(newvec, vector)

        # The normal `vconvert` method should return the same object
        newvec = cx.vconvert(cx.vecs.PolarPos, vector)
        assert newvec is vector

    def test_polar_to_cartesian3d(self, vector):
        """Test ``coordinax.vconvert(CartesianPos3D)``."""
        cart3d = vector.vconvert(cx.CartesianPos3D, z=u.Quantity([9, 10, 11, 12], "m"))

        assert isinstance(cart3d, cx.CartesianPos3D)
        assert jnp.array_equal(
            cart3d.x, u.Quantity([1.0, 1.0806046, -1.2484405, -3.95997], "kpc")
        )
        assert jnp.array_equal(
            cart3d.y, u.Quantity([0.0, 1.6829419, 2.7278922, 0.56448], "kpc")
        )
        assert jnp.array_equal(cart3d.z, u.Quantity([9, 10, 11, 12], "m"))

    def test_polar_to_spherical(self, vector):
        """Test ``coordinax.vconvert(SphericalPos)``."""
        spherical = vector.vconvert(
            cx.SphericalPos, theta=u.Quantity([4, 15, 60, 170], "deg")
        )

        assert isinstance(spherical, cx.SphericalPos)
        assert jnp.array_equal(spherical.r, u.Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(spherical.theta, u.Quantity([4, 15, 60, 170], "deg"))
        assert jnp.array_equal(spherical.phi, u.Quantity([0, 1, 2, 3], "rad"))

    def test_polar_to_cylindrical(self, vector):
        """Test ``coordinax.vconvert(CylindricalPos)``."""
        cylindrical = vector.vconvert(
            cx.vecs.CylindricalPos, z=u.Quantity([9, 10, 11, 12], "m")
        )

        assert isinstance(cylindrical, cx.vecs.CylindricalPos)
        assert jnp.array_equal(cylindrical.rho, u.Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cylindrical.phi, u.Quantity([0, 1, 2, 3], "rad"))
        assert jnp.array_equal(cylindrical.z, u.Quantity([9, 10, 11, 12], "m"))


class AbstractVel2DTest(AbstractVelTest):
    """Test :class:`coordinax.AbstractVel2D`."""


class TestCartesianVel2D(AbstractVel2DTest):
    """Test :class:`coordinax.CartesianVel2D`."""

    @pytest.fixture(scope="class")
    def difntl(self) -> cx.vecs.CartesianVel2D:
        """Return a differential."""
        return cx.vecs.CartesianVel2D(
            d_x=u.Quantity([1, 2, 3, 4], "km/s"),
            d_y=u.Quantity([5, 6, 7, 8], "km/s"),
        )

    @pytest.fixture(scope="class")
    def vector(self) -> cx.vecs.CartesianPos2D:
        """Return a vector."""
        return cx.vecs.CartesianPos2D(
            x=u.Quantity([1, 2, 3, 4], "kpc"), y=u.Quantity([5, 6, 7, 8], "km")
        )

    # ==========================================================================
    # vconvert

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_cartesian1d(self, difntl, vector):
        """Test ``difntl.vconvert(CartesianVel1D, vector)``."""
        cart1d = difntl.vconvert(cx.vecs.CartesianVel1D, vector)

        assert isinstance(cart1d, cx.vecs.CartesianVel1D)
        assert jnp.array_equal(cart1d.d_x, u.Quantity([1, 2, 3, 4], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_radial(self, difntl, vector):
        """Test ``difntl.vconvert(RadialVel, vector)``."""
        radial = difntl.vconvert(cx.vecs.RadialVel, vector)

        assert isinstance(radial, cx.vecs.RadialVel)
        assert jnp.array_equal(radial.d_r, u.Quantity([1, 2, 3, 4], "km/s"))

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_cartesian2d(self, difntl, vector):
        """Test ``difntl.vconvert(CartesianVel2D, vector)``."""
        # Jit can copy
        newvec = difntl.vconvert(cx.vecs.CartesianVel2D, vector)
        assert jnp.array_equal(newvec, difntl)

        # The normal `vconvert` method should return the same object
        newvec = cx.vconvert(cx.vecs.CartesianVel2D, difntl, vector)
        assert newvec is difntl

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_polar(self, difntl, vector):
        """Test ``difntl.vconvert(PolarVel, vector)``."""
        polar = difntl.vconvert(cx.vecs.PolarVel, vector)

        assert isinstance(polar, cx.vecs.PolarVel)
        assert jnp.array_equal(polar.d_r, u.Quantity([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(
            polar.d_phi,
            u.Quantity([5.0, 3.0, 2.3333335, 1.9999999], "km rad / (kpc s)"),
        )

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_cartesian3d(self, difntl, vector):
        """Test ``difntl.vconvert(CartesianVel3D, vector)``."""
        cart3d = difntl.vconvert(
            cx.CartesianVel3D, vector, d_z=u.Quantity([9, 10, 11, 12], "m/s")
        )

        assert isinstance(cart3d, cx.CartesianVel3D)
        assert jnp.array_equal(cart3d.d_x, u.Quantity([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(cart3d.d_y, u.Quantity([5, 6, 7, 8], "km/s"))
        assert jnp.array_equal(cart3d.d_z, u.Quantity([9, 10, 11, 12], "m/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_spherical(self, difntl, vector):
        """Test ``difntl.vconvert(SphericalVel, vector)``."""
        spherical = difntl.vconvert(
            cx.SphericalVel, vector, d_theta=u.Quantity([4, 5, 6, 7], "rad")
        )

        assert isinstance(spherical, cx.SphericalVel)
        assert jnp.array_equal(spherical.d_r, u.Quantity([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(spherical.d_theta, u.Quantity([4, 5, 6, 7], "rad"))
        assert jnp.array_equal(spherical.d_phi, u.Quantity([5, 6, 7, 8], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_cylindrical(self, difntl, vector):
        """Test ``difntl.vconvert(CylindricalVel, vector)``."""
        cylindrical = difntl.vconvert(
            cx.vecs.CylindricalVel, vector, d_z=u.Quantity([9, 10, 11, 12], "m/s")
        )

        assert isinstance(cylindrical, cx.vecs.CylindricalVel)
        assert jnp.array_equal(cylindrical.d_rho, u.Quantity([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(cylindrical.d_phi, u.Quantity([5, 6, 7, 8], "km/s"))
        assert jnp.array_equal(cylindrical.d_z, u.Quantity([9, 10, 11, 12], "m/s"))


class TestPolarVel(AbstractVel2DTest):
    """Test :class:`coordinax.PolarVel`."""

    @pytest.fixture(scope="class")
    def difntl(self) -> cx.vecs.PolarVel:
        """Return a differential."""
        return cx.vecs.PolarVel(
            d_r=u.Quantity([1, 2, 3, 4], "km/s"),
            d_phi=u.Quantity([5, 6, 7, 8], "mas/yr"),
        )

    @pytest.fixture(scope="class")
    def vector(self) -> cx.vecs.PolarPos:
        """Return a vector."""
        return cx.vecs.PolarPos(
            r=u.Quantity([1, 2, 3, 4], "kpc"), phi=u.Quantity([0, 1, 2, 3], "rad")
        )

    # ==========================================================================
    # vconvert

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_cartesian1d(self, difntl, vector):
        """Test ``difntl.vconvert(CartesianVel1D, vector)``."""
        cart1d = difntl.vconvert(cx.vecs.CartesianVel1D, vector)

        assert isinstance(cart1d, cx.vecs.CartesianVel1D)
        assert jnp.array_equal(cart1d.d_x, u.Quantity([1, 2, 3, 4], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_radial(self, difntl, vector):
        """Test ``difntl.vconvert(RadialVel, vector)``."""
        radial = difntl.vconvert(cx.vecs.RadialVel, vector)

        assert isinstance(radial, cx.vecs.RadialVel)
        assert jnp.array_equal(radial.d_r, u.Quantity([1, 2, 3, 4], "km/s"))

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_cartesian2d(self, difntl, vector):
        """Test ``difntl.vconvert(CartesianVel2D, vector)``."""
        cart2d = difntl.vconvert(cx.vecs.CartesianVel2D, vector)

        assert isinstance(cart2d, cx.vecs.CartesianVel2D)
        assert jnp.array_equal(
            cart2d.d_x, u.Quantity([1.0, -46.787014, -91.76889, -25.367176], "km/s")
        )
        assert jnp.array_equal(
            cart2d.d_y,
            u.Quantity([23.702353, 32.418385, -38.69947, -149.61249], "km/s"),
        )

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_polar(self, difntl, vector):
        """Test ``difntl.vconvert(PolarVel, vector)``."""
        # Jit can copy
        newvec = difntl.vconvert(cx.vecs.PolarVel, vector)
        assert all(newvec == difntl)

        # The normal `vconvert` method should return the same object
        newvec = cx.vconvert(cx.vecs.PolarVel, difntl, vector)
        assert newvec is difntl

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_cartesian3d(self, difntl, vector):
        """Test ``difntl.vconvert(CartesianVel3D, vector)``."""
        cart3d = difntl.vconvert(
            cx.CartesianVel3D, vector, d_z=u.Quantity([9, 10, 11, 12], "m/s")
        )

        assert isinstance(cart3d, cx.CartesianVel3D)
        assert jnp.array_equal(cart3d.d_x, u.Quantity([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(cart3d.d_y, u.Quantity([5, 6, 7, 8], "km/s"))
        assert jnp.array_equal(cart3d.d_z, u.Quantity([9, 10, 11, 12], "m/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_spherical(self, difntl, vector):
        """Test ``difntl.vconvert(SphericalVel, vector)``."""
        spherical = difntl.vconvert(
            cx.SphericalVel, vector, d_theta=u.Quantity([4, 5, 6, 7], "rad")
        )

        assert isinstance(spherical, cx.SphericalVel)
        assert jnp.array_equal(spherical.d_r, u.Quantity([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(spherical.d_theta, u.Quantity([4, 5, 6, 7], "rad"))
        assert jnp.array_equal(spherical.d_phi, u.Quantity([5, 6, 7, 8], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_cylindrical(self, difntl, vector):
        """Test ``difntl.vconvert(CylindricalVel, vector)``."""
        cylindrical = difntl.vconvert(
            cx.vecs.CylindricalVel, vector, d_z=u.Quantity([9, 10, 11, 12], "m/s")
        )

        assert isinstance(cylindrical, cx.vecs.CylindricalVel)
        assert jnp.array_equal(cylindrical.d_rho, u.Quantity([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(cylindrical.d_phi, u.Quantity([5, 6, 7, 8], "km/s"))
        assert jnp.array_equal(cylindrical.d_z, u.Quantity([9, 10, 11, 12], "m/s"))
