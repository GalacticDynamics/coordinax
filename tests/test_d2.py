"""Test :mod:`coordinax._d2`."""

import astropy.units as u
import jax.numpy as jnp
import pytest
from quax import quaxify

import array_api_jax_compat as xp
from jax_quantity import Quantity

from .test_base import AbstractVectorDifferentialTest, AbstractVectorTest, array_equal
from coordinax import (
    AbstractVector,
    Cartesian1DVector,
    Cartesian2DVector,
    Cartesian3DVector,
    CartesianDifferential1D,
    CartesianDifferential2D,
    CartesianDifferential3D,
    CylindricalDifferential,
    CylindricalVector,
    PolarDifferential,
    PolarVector,
    RadialDifferential,
    RadialVector,
    SphericalDifferential,
    SphericalVector,
    represent_as,
)

hypot = quaxify(jnp.hypot)
allclose = quaxify(jnp.allclose)


class Abstract2DVectorTest(AbstractVectorTest):
    """Test :class:`coordinax.Abstract2DVector`."""


class TestCartesian2DVector:
    """Test :class:`coordinax.Cartesian2DVector`."""

    @pytest.fixture(scope="class")
    def vector(self) -> Cartesian2DVector:
        """Return a vector."""
        return Cartesian2DVector(
            x=Quantity([1, 2, 3, 4], u.kpc), y=Quantity([5, 6, 7, 8], u.kpc)
        )

    # ==========================================================================
    # represent_as

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cartesian2d_to_cartesian1d(self, vector):
        """Test ``coordinax.represent_as(Cartesian1DVector)``."""
        cart1d = vector.represent_as(Cartesian1DVector)

        assert isinstance(cart1d, Cartesian1DVector)
        assert array_equal(cart1d.x, Quantity([1, 2, 3, 4], u.kpc))

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cartesian2d_to_radial(self, vector):
        """Test ``coordinax.represent_as(RadialVector)``."""
        radial = vector.represent_as(RadialVector)

        assert isinstance(radial, RadialVector)
        assert array_equal(radial.r, hypot(vector.x, vector.y))

    def test_cartesian2d_to_cartesian2d(self, vector):
        """Test ``coordinax.represent_as(Cartesian2DVector)``."""
        newvec = vector.represent_as(Cartesian2DVector)
        assert newvec is vector

    def test_cartesian2d_to_cartesian2d(self, vector):
        """Test ``coordinax.represent_as(Cartesian2DVector)``."""
        # Jit can copy
        newvec = vector.represent_as(Cartesian2DVector)
        assert newvec == vector

        # The normal `represent_as` method should return the same object
        newvec = represent_as(vector, Cartesian2DVector)
        assert newvec is vector

    def test_cartesian2d_to_polar(self, vector):
        """Test ``coordinax.represent_as(PolarVector)``."""
        polar = vector.represent_as(PolarVector)

        assert isinstance(polar, PolarVector)
        assert array_equal(polar.r, hypot(vector.x, vector.y))
        assert jnp.allclose(
            polar.phi.value,
            xp.asarray([1.3734008, 1.2490457, 1.1659045, 1.1071488]),
        )

    # def test_cartesian2d_to_lnpolar(self, vector):
    #     """Test ``coordinax.represent_as(LnPolarVector)``."""
    #     assert False

    # def test_cartesian2d_to_log10polar(self, vector):
    #     """Test ``coordinax.represent_as(Log10PolarVector)``."""
    #    assert False

    def test_cartesian2d_to_cartesian3d(self, vector):
        """Test ``coordinax.represent_as(Cartesian3DVector)``."""
        cart3d = vector.represent_as(
            Cartesian3DVector, z=Quantity([9, 10, 11, 12], u.m)
        )

        assert isinstance(cart3d, Cartesian3DVector)
        assert array_equal(cart3d.x, Quantity([1, 2, 3, 4], u.kpc))
        assert array_equal(cart3d.y, Quantity([5, 6, 7, 8], u.kpc))
        assert array_equal(cart3d.z, Quantity([9, 10, 11, 12], u.m))

    def test_cartesian2d_to_spherical(self, vector):
        """Test ``coordinax.represent_as(SphericalVector)``."""
        spherical = vector.represent_as(
            SphericalVector, theta=Quantity([4, 5, 6, 7], u.rad)
        )

        assert isinstance(spherical, SphericalVector)
        assert array_equal(spherical.r, hypot(vector.x, vector.y))
        assert jnp.allclose(
            spherical.phi.to_value(u.rad),
            xp.asarray([1.3734008, 1.2490457, 1.1659045, 1.1071488]),
        )
        assert array_equal(
            spherical.theta, Quantity(xp.full(4, fill_value=xp.pi / 2), u.rad)
        )

    def test_cartesian2d_to_cylindrical(self, vector):
        """Test ``coordinax.represent_as(CylindricalVector)``."""
        cylindrical = vector.represent_as(
            CylindricalVector, z=Quantity([9, 10, 11, 12], u.m)
        )

        assert isinstance(cylindrical, CylindricalVector)
        assert array_equal(cylindrical.rho, hypot(vector.x, vector.y))
        assert array_equal(
            cylindrical.phi,
            Quantity([1.3734008, 1.2490457, 1.1659045, 1.1071488], u.rad),
        )
        assert array_equal(cylindrical.z, Quantity([9, 10, 11, 12], u.m))


class TestPolarVector:
    """Test :class:`coordinax.PolarVector`."""

    @pytest.fixture(scope="class")
    def vector(self) -> AbstractVector:
        """Return a vector."""
        from coordinax import PolarVector

        return PolarVector(
            r=Quantity([1, 2, 3, 4], u.kpc), phi=Quantity([0, 1, 2, 3], u.rad)
        )

    # ==========================================================================
    # represent_as

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_polar_to_cartesian1d(self, vector):
        """Test ``coordinax.represent_as(Cartesian1DVector)``."""
        cart1d = vector.represent_as(Cartesian1DVector)

        assert isinstance(cart1d, Cartesian1DVector)
        assert jnp.allclose(
            cart1d.x.to_value(u.kpc), xp.asarray([1.0, 1.0806047, -1.2484405, -3.95997])
        )
        assert array_equal(cart1d.x, vector.r * xp.cos(vector.phi))

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_polar_to_radial(self, vector):
        """Test ``coordinax.represent_as(RadialVector)``."""
        radial = vector.represent_as(RadialVector)

        assert isinstance(radial, RadialVector)
        assert array_equal(radial.r, Quantity([1, 2, 3, 4], u.kpc))

    def test_polar_to_cartesian2d(self, vector):
        """Test ``coordinax.represent_as(Cartesian2DVector)``."""
        cart2d = vector.represent_as(Cartesian2DVector, y=Quantity([5, 6, 7, 8], u.km))

        assert isinstance(cart2d, Cartesian2DVector)
        assert array_equal(
            cart2d.x, Quantity([1.0, 1.0806046, -1.2484405, -3.95997], u.kpc)
        )
        assert jnp.allclose(cart2d.x.value, (vector.r * xp.cos(vector.phi)).value)
        assert array_equal(
            cart2d.y, Quantity([0.0, 1.6829419, 2.7278922, 0.56448], u.kpc)
        )
        assert jnp.allclose(cart2d.y.value, (vector.r * xp.sin(vector.phi)).value)

    def test_polar_to_polar(self, vector):
        """Test ``coordinax.represent_as(PolarVector)``."""
        # Jit can copy
        newvec = vector.represent_as(PolarVector)
        assert newvec == vector

        # The normal `represent_as` method should return the same object
        newvec = represent_as(vector, PolarVector)
        assert newvec is vector

    # def test_polar_to_lnpolar(self, vector):
    #     """Test ``coordinax.represent_as(LnPolarVector)``."""
    #     assert False

    # def test_polar_to_log10polar(self, vector):
    #     """Test ``coordinax.represent_as(Log10PolarVector)
    #     assert False

    def test_polar_to_cartesian3d(self, vector):
        """Test ``coordinax.represent_as(Cartesian3DVector)``."""
        cart3d = vector.represent_as(
            Cartesian3DVector, z=Quantity([9, 10, 11, 12], u.m)
        )

        assert isinstance(cart3d, Cartesian3DVector)
        assert array_equal(
            cart3d.x, Quantity([1.0, 1.0806046, -1.2484405, -3.95997], u.kpc)
        )
        assert array_equal(
            cart3d.y, Quantity([0.0, 1.6829419, 2.7278922, 0.56448], u.kpc)
        )
        assert array_equal(cart3d.z, Quantity([9, 10, 11, 12], u.m))

    def test_polar_to_spherical(self, vector):
        """Test ``coordinax.represent_as(SphericalVector)``."""
        spherical = vector.represent_as(
            SphericalVector, theta=Quantity([4, 5, 6, 7], u.rad)
        )

        assert isinstance(spherical, SphericalVector)
        assert array_equal(spherical.r, Quantity([1, 2, 3, 4], u.kpc))
        assert array_equal(spherical.phi, Quantity([0, 1, 2, 3], u.rad))
        assert array_equal(spherical.theta, Quantity([4, 5, 6, 7], u.rad))

    def test_polar_to_cylindrical(self, vector):
        """Test ``coordinax.represent_as(CylindricalVector)``."""
        cylindrical = vector.represent_as(
            CylindricalVector, z=Quantity([9, 10, 11, 12], u.m)
        )

        assert isinstance(cylindrical, CylindricalVector)
        assert array_equal(cylindrical.rho, Quantity([1, 2, 3, 4], u.kpc))
        assert array_equal(cylindrical.phi, Quantity([0, 1, 2, 3], u.rad))
        assert array_equal(cylindrical.z, Quantity([9, 10, 11, 12], u.m))


class Abstract2DVectorDifferentialTest(AbstractVectorDifferentialTest):
    """Test :class:`coordinax.Abstract2DVectorDifferential`."""


class TestCartesianDifferential2D(Abstract2DVectorDifferentialTest):
    """Test :class:`coordinax.CartesianDifferential2D`."""

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
        assert array_equal(cart1d.d_x, Quantity([1, 2, 3, 4], u.km / u.s))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_radial(self, difntl, vector):
        """Test ``difntl.represent_as(RadialDifferential, vector)``."""
        radial = difntl.represent_as(RadialDifferential, vector)

        assert isinstance(radial, RadialDifferential)
        assert array_equal(radial.d_r, Quantity([1, 2, 3, 4], u.km / u.s))

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
        assert array_equal(polar.d_r, Quantity([1, 2, 3, 4], u.km / u.s))
        assert array_equal(
            polar.d_phi,
            Quantity([5.0, 3.0, 2.3333335, 1.9999999], (u.km * u.rad) / (u.kpc * u.s)),
        )

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_cartesian3d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianDifferential3D, vector)``."""
        cart3d = difntl.represent_as(
            CartesianDifferential3D, vector, d_z=Quantity([9, 10, 11, 12], u.m / u.s)
        )

        assert isinstance(cart3d, CartesianDifferential3D)
        assert array_equal(cart3d.d_x, Quantity([1, 2, 3, 4], u.km / u.s))
        assert array_equal(cart3d.d_y, Quantity([5, 6, 7, 8], u.km / u.s))
        assert array_equal(cart3d.d_z, Quantity([9, 10, 11, 12], u.m / u.s))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_spherical(self, difntl, vector):
        """Test ``difntl.represent_as(SphericalDifferential, vector)``."""
        spherical = difntl.represent_as(
            SphericalDifferential, vector, d_theta=Quantity([4, 5, 6, 7], u.rad)
        )

        assert isinstance(spherical, SphericalDifferential)
        assert array_equal(spherical.d_r, Quantity([1, 2, 3, 4], u.km / u.s))
        assert array_equal(spherical.d_phi, Quantity([5, 6, 7, 8], u.km / u.s))
        assert array_equal(spherical.d_theta, Quantity([4, 5, 6, 7], u.rad))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian2d_to_cylindrical(self, difntl, vector):
        """Test ``difntl.represent_as(CylindricalDifferential, vector)``."""
        cylindrical = difntl.represent_as(
            CylindricalDifferential, vector, d_z=Quantity([9, 10, 11, 12], u.m / u.s)
        )

        assert isinstance(cylindrical, CylindricalDifferential)
        assert array_equal(cylindrical.d_rho, Quantity([1, 2, 3, 4], u.km / u.s))
        assert array_equal(cylindrical.d_phi, Quantity([5, 6, 7, 8], u.km / u.s))
        assert array_equal(cylindrical.d_z, Quantity([9, 10, 11, 12], u.m / u.s))


class TestPolarDifferential(Abstract2DVectorDifferentialTest):
    """Test :class:`coordinax.PolarDifferential`."""

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
        assert array_equal(cart1d.d_x, Quantity([1, 2, 3, 4], u.km / u.s))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_radial(self, difntl, vector):
        """Test ``difntl.represent_as(RadialDifferential, vector)``."""
        radial = difntl.represent_as(RadialDifferential, vector)

        assert isinstance(radial, RadialDifferential)
        assert array_equal(radial.d_r, Quantity([1, 2, 3, 4], u.km / u.s))

    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_cartesian2d(self, difntl, vector):
        """Test ``difntl.represent_as(CartesianDifferential2D, vector)``."""
        cart2d = difntl.represent_as(CartesianDifferential2D, vector)

        assert isinstance(cart2d, CartesianDifferential2D)
        assert array_equal(
            cart2d.d_x, Quantity([1.0, -46.787014, -91.76889, -25.367176], u.km / u.s)
        )
        assert array_equal(
            cart2d.d_y,
            Quantity([23.702353, 32.418385, -38.69947, -149.61249], u.km / u.s),
        )

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
        assert array_equal(cart3d.d_x, Quantity([1, 2, 3, 4], u.km / u.s))
        assert array_equal(cart3d.d_y, Quantity([5, 6, 7, 8], u.km / u.s))
        assert array_equal(cart3d.d_z, Quantity([9, 10, 11, 12], u.m / u.s))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_spherical(self, difntl, vector):
        """Test ``difntl.represent_as(SphericalDifferential, vector)``."""
        spherical = difntl.represent_as(
            SphericalDifferential, vector, d_theta=Quantity([4, 5, 6, 7], u.rad)
        )

        assert isinstance(spherical, SphericalDifferential)
        assert array_equal(spherical.d_r, Quantity([1, 2, 3, 4], u.km / u.s))
        assert array_equal(spherical.d_phi, Quantity([5, 6, 7, 8], u.km / u.s))
        assert array_equal(spherical.d_theta, Quantity([4, 5, 6, 7], u.rad))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_polar_to_cylindrical(self, difntl, vector):
        """Test ``difntl.represent_as(CylindricalDifferential, vector)``."""
        cylindrical = difntl.represent_as(
            CylindricalDifferential, vector, d_z=Quantity([9, 10, 11, 12], u.m / u.s)
        )

        assert isinstance(cylindrical, CylindricalDifferential)
        assert array_equal(cylindrical.d_rho, Quantity([1, 2, 3, 4], u.km / u.s))
        assert array_equal(cylindrical.d_phi, Quantity([5, 6, 7, 8], u.km / u.s))
        assert array_equal(cylindrical.d_z, Quantity([9, 10, 11, 12], u.m / u.s))
