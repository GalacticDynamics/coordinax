"""Test :mod:`coordinax._d2`."""

import pytest

import quaxed.array_api as xp
import quaxed.numpy as qnp
from unxt import Quantity

import coordinax as cx
from .test_base import AbstractPositionTest, AbstractVelocityTest


class Abstract2DVectorTest(AbstractPositionTest):
    """Test :class:`coordinax.Abstract2DVector`."""


class TestCartesian2DVector:
    """Test :class:`coordinax.Cartesian2DVector`."""

    @pytest.fixture(scope="class")
    def vector(self) -> cx.Cartesian2DVector:
        """Return a vector."""
        return cx.Cartesian2DVector(
            x=Quantity([1, 2, 3, 4], "kpc"), y=Quantity([5, 6, 7, 8], "kpc")
        )

    # ==========================================================================
    # represent_as

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cartesian2d_to_cartesian1d(self, vector):
        """Test ``coordinax.represent_as(Cartesian1DVector)``."""
        cart1d = vector.represent_as(cx.Cartesian1DVector)

        assert isinstance(cart1d, cx.Cartesian1DVector)
        assert qnp.array_equal(cart1d.x, Quantity([1, 2, 3, 4], "kpc"))

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cartesian2d_to_radial(self, vector):
        """Test ``coordinax.represent_as(RadialVector)``."""
        radial = vector.represent_as(cx.RadialVector)

        assert isinstance(radial, cx.RadialVector)
        assert qnp.array_equal(radial.r, qnp.hypot(vector.x, vector.y))

    def test_cartesian2d_to_cartesian2d(self, vector):
        """Test ``coordinax.represent_as(Cartesian2DVector)``."""
        newvec = vector.represent_as(cx.Cartesian2DVector)
        assert newvec is vector

    def test_cartesian2d_to_cartesian2d(self, vector):
        """Test ``coordinax.represent_as(Cartesian2DVector)``."""
        # Jit can copy
        newvec = vector.represent_as(cx.Cartesian2DVector)
        assert newvec == vector

        # The normal `represent_as` method should return the same object
        newvec = cx.represent_as(vector, cx.Cartesian2DVector)
        assert newvec is vector

    def test_cartesian2d_to_polar(self, vector):
        """Test ``coordinax.represent_as(PolarVector)``."""
        polar = vector.represent_as(cx.PolarVector)

        assert isinstance(polar, cx.PolarVector)
        assert qnp.array_equal(polar.r, qnp.hypot(vector.x, vector.y))
        assert qnp.allclose(
            polar.phi,
            Quantity([1.3734008, 1.2490457, 1.1659045, 1.1071488], "rad"),
            atol=Quantity(1e-8, "deg"),
        )

    def test_cartesian2d_to_cartesian3d(self, vector):
        """Test ``coordinax.represent_as(Cartesian3DVector)``."""
        cart3d = vector.represent_as(
            cx.Cartesian3DVector, z=Quantity([9, 10, 11, 12], "m")
        )

        assert isinstance(cart3d, cx.Cartesian3DVector)
        assert qnp.array_equal(cart3d.x, Quantity([1, 2, 3, 4], "kpc"))
        assert qnp.array_equal(cart3d.y, Quantity([5, 6, 7, 8], "kpc"))
        assert qnp.array_equal(cart3d.z, Quantity([9, 10, 11, 12], "m"))

    def test_cartesian2d_to_spherical(self, vector):
        """Test ``coordinax.represent_as(SphericalVector)``."""
        spherical = vector.represent_as(
            cx.SphericalVector, theta=Quantity([4, 5, 6, 7], "rad")
        )

        assert isinstance(spherical, cx.SphericalVector)
        assert qnp.array_equal(spherical.r, qnp.hypot(vector.x, vector.y))
        assert qnp.allclose(
            spherical.phi,
            Quantity([1.3734008, 1.2490457, 1.1659045, 1.1071488], "rad"),
            atol=Quantity(1e-8, "rad"),
        )
        assert qnp.array_equal(
            spherical.theta, Quantity(xp.full(4, fill_value=xp.pi / 2), "rad")
        )

    def test_cartesian2d_to_cylindrical(self, vector):
        """Test ``coordinax.represent_as(CylindricalVector)``."""
        cylindrical = vector.represent_as(
            cx.CylindricalVector, z=Quantity([9, 10, 11, 12], "m")
        )

        assert isinstance(cylindrical, cx.CylindricalVector)
        assert qnp.array_equal(cylindrical.rho, qnp.hypot(vector.x, vector.y))
        assert qnp.array_equal(
            cylindrical.phi,
            Quantity([1.3734008, 1.2490457, 1.1659045, 1.1071488], "rad"),
        )
        assert qnp.array_equal(cylindrical.z, Quantity([9, 10, 11, 12], "m"))


class TestPolarVector:
    """Test :class:`coordinax.PolarVector`."""

    @pytest.fixture(scope="class")
    def vector(self) -> cx.AbstractPosition:
        """Return a vector."""
        return cx.PolarVector(
            r=Quantity([1, 2, 3, 4], "kpc"), phi=Quantity([0, 1, 2, 3], "rad")
        )

    # ==========================================================================
    # represent_as

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_polar_to_cartesian1d(self, vector):
        """Test ``coordinax.represent_as(Cartesian1DVector)``."""
        cart1d = vector.represent_as(cx.Cartesian1DVector)

        assert isinstance(cart1d, cx.Cartesian1DVector)
        assert qnp.allclose(
            cart1d.x,
            Quantity([1.0, 1.0806047, -1.2484405, -3.95997], "kpc"),
            atol=Quantity(1e-8, "kpc"),
        )
        assert qnp.array_equal(cart1d.x, vector.r * xp.cos(vector.phi))

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_polar_to_radial(self, vector):
        """Test ``coordinax.represent_as(RadialVector)``."""
        radial = vector.represent_as(cx.RadialVector)

        assert isinstance(radial, cx.RadialVector)
        assert qnp.array_equal(radial.r, Quantity([1, 2, 3, 4], "kpc"))

    def test_polar_to_cartesian2d(self, vector):
        """Test ``coordinax.represent_as(Cartesian2DVector)``."""
        cart2d = vector.represent_as(
            cx.Cartesian2DVector, y=Quantity([5, 6, 7, 8], "km")
        )

        assert isinstance(cart2d, cx.Cartesian2DVector)
        assert qnp.array_equal(
            cart2d.x, Quantity([1.0, 1.0806046, -1.2484405, -3.95997], "kpc")
        )
        assert qnp.allclose(
            cart2d.x, (vector.r * xp.cos(vector.phi)), atol=Quantity(1e-8, "kpc")
        )
        assert qnp.array_equal(
            cart2d.y, Quantity([0.0, 1.6829419, 2.7278922, 0.56448], "kpc")
        )
        assert qnp.allclose(
            cart2d.y, (vector.r * xp.sin(vector.phi)), atol=Quantity(1e-8, "kpc")
        )

    def test_polar_to_polar(self, vector):
        """Test ``coordinax.represent_as(PolarVector)``."""
        # Jit can copy
        newvec = vector.represent_as(cx.PolarVector)
        assert newvec == vector

        # The normal `represent_as` method should return the same object
        newvec = cx.represent_as(vector, cx.PolarVector)
        assert newvec is vector

    def test_polar_to_cartesian3d(self, vector):
        """Test ``coordinax.represent_as(Cartesian3DVector)``."""
        cart3d = vector.represent_as(
            cx.Cartesian3DVector, z=Quantity([9, 10, 11, 12], "m")
        )

        assert isinstance(cart3d, cx.Cartesian3DVector)
        assert qnp.array_equal(
            cart3d.x, Quantity([1.0, 1.0806046, -1.2484405, -3.95997], "kpc")
        )
        assert qnp.array_equal(
            cart3d.y, Quantity([0.0, 1.6829419, 2.7278922, 0.56448], "kpc")
        )
        assert qnp.array_equal(cart3d.z, Quantity([9, 10, 11, 12], "m"))

    def test_polar_to_spherical(self, vector):
        """Test ``coordinax.represent_as(SphericalVector)``."""
        spherical = vector.represent_as(
            cx.SphericalVector, theta=Quantity([4, 5, 6, 7], "rad")
        )

        assert isinstance(spherical, cx.SphericalVector)
        assert qnp.array_equal(spherical.r, Quantity([1, 2, 3, 4], "kpc"))
        assert qnp.array_equal(spherical.theta, Quantity([4, 5, 6, 7], "rad"))
        assert qnp.array_equal(spherical.phi, Quantity([0, 1, 2, 3], "rad"))

    def test_polar_to_cylindrical(self, vector):
        """Test ``coordinax.represent_as(CylindricalVector)``."""
        cylindrical = vector.represent_as(
            cx.CylindricalVector, z=Quantity([9, 10, 11, 12], "m")
        )

        assert isinstance(cylindrical, cx.CylindricalVector)
        assert qnp.array_equal(cylindrical.rho, Quantity([1, 2, 3, 4], "kpc"))
        assert qnp.array_equal(cylindrical.phi, Quantity([0, 1, 2, 3], "rad"))
        assert qnp.array_equal(cylindrical.z, Quantity([9, 10, 11, 12], "m"))


class Abstract2DVectorDifferentialTest(AbstractVelocityTest):
    """Test :class:`coordinax.Abstract2DVectorDifferential`."""


class TestCartesianDifferential2D(Abstract2DVectorDifferentialTest):
    """Test :class:`coordinax.CartesianDifferential2D`."""

    @pytest.fixture(scope="class")
    def difntl(self) -> cx.CartesianDifferential2D:
        """Return a differential."""
        return cx.CartesianDifferential2D(
            d_x=Quantity([1, 2, 3, 4], "km/s"),
            d_y=Quantity([5, 6, 7, 8], "km/s"),
        )

    @pytest.fixture(scope="class")
    def vector(self) -> cx.Cartesian2DVector:
        """Return a vector."""
        return cx.Cartesian2DVector(
            x=Quantity([1, 2, 3, 4], "kpc"), y=Quantity([5, 6, 7, 8], "km")
        )

    # ==========================================================================
    # represent_as

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_cartesian1d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianDifferential1D, vector)``."""
        cart1d = difntl.represent_as(cx.CartesianDifferential1D, vector)

        assert isinstance(cart1d, cx.CartesianDifferential1D)
        assert qnp.array_equal(cart1d.d_x, Quantity([1, 2, 3, 4], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_radial(self, difntl, vector):
        """Test ``difntl.represent_as(RadialDifferential, vector)``."""
        radial = difntl.represent_as(cx.RadialDifferential, vector)

        assert isinstance(radial, cx.RadialDifferential)
        assert qnp.array_equal(radial.d_r, Quantity([1, 2, 3, 4], "km/s"))

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_cartesian2d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianDifferential2D, vector)``."""
        # Jit can copy
        newvec = difntl.represent_as(cx.CartesianDifferential2D, vector)
        assert newvec == difntl

        # The normal `represent_as` method should return the same object
        newvec = cx.represent_as(difntl, cx.CartesianDifferential2D, vector)
        assert newvec is difntl

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_polar(self, difntl, vector):
        """Test ``difntl.represent_as(PolarDifferential, vector)``."""
        polar = difntl.represent_as(cx.PolarDifferential, vector)

        assert isinstance(polar, cx.PolarDifferential)
        assert qnp.array_equal(polar.d_r, Quantity([1, 2, 3, 4], "km/s"))
        assert qnp.array_equal(
            polar.d_phi,
            Quantity([5.0, 3.0, 2.3333335, 1.9999999], "km rad / (kpc s)"),
        )

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_cartesian3d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianDifferential3D, vector)``."""
        cart3d = difntl.represent_as(
            cx.CartesianDifferential3D, vector, d_z=Quantity([9, 10, 11, 12], "m/s")
        )

        assert isinstance(cart3d, cx.CartesianDifferential3D)
        assert qnp.array_equal(cart3d.d_x, Quantity([1, 2, 3, 4], "km/s"))
        assert qnp.array_equal(cart3d.d_y, Quantity([5, 6, 7, 8], "km/s"))
        assert qnp.array_equal(cart3d.d_z, Quantity([9, 10, 11, 12], "m/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_spherical(self, difntl, vector):
        """Test ``difntl.represent_as(SphericalDifferential, vector)``."""
        spherical = difntl.represent_as(
            cx.SphericalDifferential, vector, d_theta=Quantity([4, 5, 6, 7], "rad")
        )

        assert isinstance(spherical, cx.SphericalDifferential)
        assert qnp.array_equal(spherical.d_r, Quantity([1, 2, 3, 4], "km/s"))
        assert qnp.array_equal(spherical.d_theta, Quantity([4, 5, 6, 7], "rad"))
        assert qnp.array_equal(spherical.d_phi, Quantity([5, 6, 7, 8], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_cylindrical(self, difntl, vector):
        """Test ``difntl.represent_as(CylindricalDifferential, vector)``."""
        cylindrical = difntl.represent_as(
            cx.CylindricalDifferential, vector, d_z=Quantity([9, 10, 11, 12], "m/s")
        )

        assert isinstance(cylindrical, cx.CylindricalDifferential)
        assert qnp.array_equal(cylindrical.d_rho, Quantity([1, 2, 3, 4], "km/s"))
        assert qnp.array_equal(cylindrical.d_phi, Quantity([5, 6, 7, 8], "km/s"))
        assert qnp.array_equal(cylindrical.d_z, Quantity([9, 10, 11, 12], "m/s"))


class TestPolarDifferential(Abstract2DVectorDifferentialTest):
    """Test :class:`coordinax.PolarDifferential`."""

    @pytest.fixture(scope="class")
    def difntl(self) -> cx.PolarDifferential:
        """Return a differential."""
        return cx.PolarDifferential(
            d_r=Quantity([1, 2, 3, 4], "km/s"),
            d_phi=Quantity([5, 6, 7, 8], "mas/yr"),
        )

    @pytest.fixture(scope="class")
    def vector(self) -> cx.PolarVector:
        """Return a vector."""
        return cx.PolarVector(
            r=Quantity([1, 2, 3, 4], "kpc"), phi=Quantity([0, 1, 2, 3], "rad")
        )

    # ==========================================================================
    # represent_as

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_cartesian1d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianDifferential1D, vector)``."""
        cart1d = difntl.represent_as(cx.CartesianDifferential1D, vector)

        assert isinstance(cart1d, cx.CartesianDifferential1D)
        assert qnp.array_equal(cart1d.d_x, Quantity([1, 2, 3, 4], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_radial(self, difntl, vector):
        """Test ``difntl.represent_as(RadialDifferential, vector)``."""
        radial = difntl.represent_as(cx.RadialDifferential, vector)

        assert isinstance(radial, cx.RadialDifferential)
        assert qnp.array_equal(radial.d_r, Quantity([1, 2, 3, 4], "km/s"))

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_cartesian2d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianDifferential2D, vector)``."""
        cart2d = difntl.represent_as(cx.CartesianDifferential2D, vector)

        assert isinstance(cart2d, cx.CartesianDifferential2D)
        assert qnp.array_equal(
            cart2d.d_x, Quantity([1.0, -46.787014, -91.76889, -25.367176], "km/s")
        )
        assert qnp.array_equal(
            cart2d.d_y,
            Quantity([23.702353, 32.418385, -38.69947, -149.61249], "km/s"),
        )

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_polar(self, difntl, vector):
        """Test ``difntl.represent_as(PolarDifferential, vector)``."""
        # Jit can copy
        newvec = difntl.represent_as(cx.PolarDifferential, vector)
        assert newvec == difntl

        # The normal `represent_as` method should return the same object
        newvec = cx.represent_as(difntl, cx.PolarDifferential, vector)
        assert newvec is difntl

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_cartesian3d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianDifferential3D, vector)``."""
        cart3d = difntl.represent_as(
            cx.CartesianDifferential3D, vector, d_z=Quantity([9, 10, 11, 12], "m/s")
        )

        assert isinstance(cart3d, cx.CartesianDifferential3D)
        assert qnp.array_equal(cart3d.d_x, Quantity([1, 2, 3, 4], "km/s"))
        assert qnp.array_equal(cart3d.d_y, Quantity([5, 6, 7, 8], "km/s"))
        assert qnp.array_equal(cart3d.d_z, Quantity([9, 10, 11, 12], "m/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_spherical(self, difntl, vector):
        """Test ``difntl.represent_as(SphericalDifferential, vector)``."""
        spherical = difntl.represent_as(
            cx.SphericalDifferential, vector, d_theta=Quantity([4, 5, 6, 7], "rad")
        )

        assert isinstance(spherical, cx.SphericalDifferential)
        assert qnp.array_equal(spherical.d_r, Quantity([1, 2, 3, 4], "km/s"))
        assert qnp.array_equal(spherical.d_theta, Quantity([4, 5, 6, 7], "rad"))
        assert qnp.array_equal(spherical.d_phi, Quantity([5, 6, 7, 8], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_cylindrical(self, difntl, vector):
        """Test ``difntl.represent_as(CylindricalDifferential, vector)``."""
        cylindrical = difntl.represent_as(
            cx.CylindricalDifferential, vector, d_z=Quantity([9, 10, 11, 12], "m/s")
        )

        assert isinstance(cylindrical, cx.CylindricalDifferential)
        assert qnp.array_equal(cylindrical.d_rho, Quantity([1, 2, 3, 4], "km/s"))
        assert qnp.array_equal(cylindrical.d_phi, Quantity([5, 6, 7, 8], "km/s"))
        assert qnp.array_equal(cylindrical.d_z, Quantity([9, 10, 11, 12], "m/s"))
