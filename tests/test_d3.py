"""Test :mod:`vector._builtin`."""

import array_api_jax_compat as xp
import astropy.units as u
import jax.numpy as jnp
import pytest
from jax_quantity import Quantity

from vector import (
    AbstractVector,
    Cartesian1DVector,
    Cartesian2DVector,
    Cartesian3DVector,
    CylindricalVector,
    PolarVector,
    RadialVector,
    SphericalVector,
    represent_as,
)
from vector._d1.builtin import CartesianDifferential1D
from vector._d2.builtin import CartesianDifferential2D
from vector._d3.builtin import (
    CartesianDifferential3D,
    CylindricalDifferential,
    SphericalDifferential,
)

from .test_base import AbstractVectorDifferentialTest, AbstractVectorTest, array_equal
from .test_d2 import hypot


class Abstract3DVectorTest(AbstractVectorTest):
    """Test :class:`vector.Abstract3DVector`."""


class TestCartesian3DVector:
    """Test :class:`vector.Cartesian3DVector`."""

    @pytest.fixture(scope="class")
    def vector(self) -> AbstractVector:
        """Return a vector."""
        from vector import Cartesian3DVector

        return Cartesian3DVector(
            x=Quantity([1, 2, 3, 4], u.kpc),
            y=Quantity([5, 6, 7, 8], u.kpc),
            z=Quantity([9, 10, 11, 12], u.kpc),
        )

    # ==========================================================================
    # represent_as

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cartesian3d_to_cartesian1d(self, vector):
        """Test ``vector.represent_as(Cartesian1DVector)``."""
        cart1d = vector.represent_as(Cartesian1DVector)

        assert isinstance(cart1d, Cartesian1DVector)
        assert array_equal(cart1d.x, Quantity([1, 2, 3, 4], u.kpc))

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cartesian3d_to_radial(self, vector):
        """Test ``vector.represent_as(RadialVector)``."""
        radial = vector.represent_as(RadialVector)

        assert isinstance(radial, RadialVector)
        assert array_equal(
            radial.r, Quantity([10.34408, 11.83216, 13.379088, 14.96663], u.kpc)
        )

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cartesian3d_to_cartesian2d(self, vector):
        """Test ``vector.represent_as(Cartesian2DVector)``."""
        cart2d = vector.represent_as(Cartesian2DVector, y=Quantity([5, 6, 7, 8], u.km))

        assert isinstance(cart2d, Cartesian2DVector)
        assert array_equal(cart2d.x, Quantity([1, 2, 3, 4], u.kpc))
        assert array_equal(cart2d.y, Quantity([5, 6, 7, 8], u.kpc))

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cartesian3d_to_polar(self, vector):
        """Test ``vector.represent_as(PolarVector)``."""
        polar = vector.represent_as(PolarVector, phi=Quantity([0, 1, 2, 3], u.rad))

        assert isinstance(polar, PolarVector)
        assert array_equal(polar.r, hypot(vector.x, vector.y))
        assert array_equal(
            polar.phi, Quantity([1.3734008, 1.2490457, 1.1659045, 1.1071488], u.rad)
        )

    # @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    # def test_cartesian3d_to_lnpolar(self, vector):
    #     """Test ``vector.represent_as(LnPolarVector)``."""
    #     assert False

    # @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    # def test_cartesian3d_to_log10polar(self, vector):
    #     """Test ``vector.represent_as(Log10PolarVector)``."""
    #     assert False

    def test_cartesian3d_to_cartesian3d(self, vector):
        """Test ``vector.represent_as(Cartesian3DVector)``."""
        # Jit can copy
        newvec = vector.represent_as(Cartesian3DVector)
        assert newvec == vector

        # The normal `represent_as` method should return the same object
        newvec = represent_as(vector, Cartesian3DVector)
        assert newvec is vector

    def test_cartesian3d_to_spherical(self, vector):
        """Test ``vector.represent_as(SphericalVector)``."""
        spherical = vector.represent_as(
            SphericalVector,
            phi=Quantity([0, 1, 2, 3], u.rad),
            theta=Quantity([4, 5, 6, 7], u.rad),
        )

        assert isinstance(spherical, SphericalVector)
        assert array_equal(
            spherical.r, Quantity([10.34408, 11.83216, 13.379088, 14.96663], u.kpc)
        )
        assert array_equal(
            spherical.phi, Quantity([1.3734008, 1.2490457, 1.1659045, 1.1071488], u.rad)
        )
        assert array_equal(
            spherical.theta,
            Quantity([0.51546645, 0.5639427, 0.6055685, 0.64052236], u.rad),
        )

    def test_cartesian3d_to_cylindrical(self, vector):
        """Test ``vector.represent_as(CylindricalVector)``."""
        cylindrical = vector.represent_as(
            CylindricalVector,
            phi=Quantity([0, 1, 2, 3], u.rad),
        )

        assert isinstance(cylindrical, CylindricalVector)
        assert array_equal(cylindrical.rho, hypot(vector.x, vector.y))
        assert array_equal(
            cylindrical.phi,
            Quantity([1.3734008, 1.2490457, 1.1659045, 1.1071488], u.rad),
        )
        assert array_equal(cylindrical.z, Quantity([9.0, 10, 11, 12], u.kpc))


class TestSphericalVector:
    """Test :class:`vector.SphericalVector`."""

    @pytest.fixture(scope="class")
    def vector(self) -> AbstractVector:
        """Return a vector."""
        from vector import SphericalVector

        return SphericalVector(
            r=Quantity([1, 2, 3, 4], u.kpc),
            phi=Quantity([0, 1, 2, 3], u.rad),
            theta=Quantity([4, 5, 6, 7], u.rad),
        )

    # ==========================================================================
    # represent_as

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_spherical_to_cartesian1d(self, vector):
        """Test ``vector.represent_as(Cartesian1DVector)``."""
        cart1d = vector.represent_as(Cartesian1DVector)

        assert isinstance(cart1d, Cartesian1DVector)
        assert array_equal(
            cart1d.x, Quantity([-0.7568025, -1.036218, 0.34883362, -2.6016471], u.kpc)
        )

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_spherical_to_radial(self, vector):
        """Test ``vector.represent_as(RadialVector)``."""
        radial = vector.represent_as(RadialVector)

        assert isinstance(radial, RadialVector)
        assert array_equal(radial.r, Quantity([1, 2, 3, 4], u.kpc))

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_spherical_to_cartesian2d(self, vector):
        """Test ``vector.represent_as(Cartesian2DVector)``."""
        cart2d = vector.represent_as(Cartesian2DVector, y=Quantity([5, 6, 7, 8], u.km))

        assert isinstance(cart2d, Cartesian2DVector)
        assert array_equal(
            cart2d.x, Quantity([-0.7568025, -1.0362179, 0.34883362, -2.6016471], u.kpc)
        )
        assert array_equal(
            cart2d.y, Quantity([-0.0, -1.6138139, -0.7622153, 0.3708558], u.kpc)
        )

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_spherical_to_polar(self, vector):
        """Test ``vector.represent_as(PolarVector)``."""
        polar = vector.represent_as(PolarVector, phi=Quantity([0, 1, 2, 3], u.rad))

        assert isinstance(polar, PolarVector)
        assert array_equal(
            polar.r, Quantity([-0.7568025, -1.9178486, -0.83824646, 2.6279464], u.kpc)
        )
        assert array_equal(polar.phi, Quantity([0.0, 1, 2, 3], u.rad))

    # @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    # def test_spherical_to_lnpolar(self, vector):
    #     """Test ``vector.represent_as(LnPolarVector)``."""
    #     assert False

    # @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    # def test_spherical_to_log10polar(self, vector):
    #     """Test ``vector.represent_as(Log10PolarVector)``."""
    #     assert False

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_spherical_to_cartesian3d(self, vector):
        """Test ``vector.represent_as(Cartesian3DVector)``."""
        cart3d = vector.represent_as(
            Cartesian3DVector, z=Quantity([9, 10, 11, 12], u.m)
        )

        assert isinstance(cart3d, Cartesian3DVector)
        assert array_equal(
            cart3d.x, Quantity([-0.7568025, -1.0362179, 0.34883362, -2.6016471], u.kpc)
        )
        assert array_equal(
            cart3d.y, Quantity([-0.0, -1.6138139, -0.7622153, 0.3708558], u.kpc)
        )
        assert array_equal(
            cart3d.z, Quantity([-0.6536436, 0.5673244, 2.8805108, 3.015609], u.kpc)
        )

    def test_spherical_to_spherical(self, vector):
        """Test ``vector.represent_as(SphericalVector)``."""
        # Jit can copy
        newvec = vector.represent_as(SphericalVector)
        assert newvec == vector

        # The normal `represent_as` method should return the same object
        newvec = represent_as(vector, SphericalVector)
        assert newvec is vector

    def test_spherical_to_cylindrical(self, vector):
        """Test ``vector.represent_as(CylindricalVector)``."""
        cylindrical = vector.represent_as(
            CylindricalVector, z=Quantity([9, 10, 11, 12], u.m)
        )

        assert isinstance(cylindrical, CylindricalVector)
        assert array_equal(
            cylindrical.rho,
            Quantity([-0.7568025, -1.9178486, -0.83824646, 2.6279464], u.kpc),
        )
        assert array_equal(cylindrical.phi, Quantity([0, 1, 2, 3], u.rad))
        assert array_equal(
            cylindrical.z, Quantity([-0.6536436, 0.5673244, 2.8805108, 3.015609], u.kpc)
        )


class TestCylindricalVector:
    """Test :class:`vector.CylindricalVector`."""

    @pytest.fixture(scope="class")
    def vector(self) -> AbstractVector:
        """Return a vector."""
        from vector import CylindricalVector

        return CylindricalVector(
            rho=Quantity([1, 2, 3, 4], u.kpc),
            phi=Quantity([0, 1, 2, 3], u.rad),
            z=Quantity([9, 10, 11, 12], u.m),
        )

    # ==========================================================================
    # represent_as

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cylindrical_to_cartesian1d(self, vector):
        """Test ``vector.represent_as(Cartesian1DVector)``."""
        cart1d = vector.represent_as(Cartesian1DVector)

        assert isinstance(cart1d, Cartesian1DVector)
        assert array_equal(
            cart1d.x, Quantity([1.0, 1.0806047, -1.2484405, -3.95997], u.kpc)
        )

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cylindrical_to_radial(self, vector):
        """Test ``vector.represent_as(RadialVector)``."""
        radial = vector.represent_as(RadialVector)

        assert isinstance(radial, RadialVector)
        assert array_equal(radial.r, Quantity([1, 2, 3, 4], u.kpc))

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cylindrical_to_cartesian2d(self, vector):
        """Test ``vector.represent_as(Cartesian2DVector)``."""
        cart2d = vector.represent_as(Cartesian2DVector)

        assert isinstance(cart2d, Cartesian2DVector)
        assert array_equal(
            cart2d.x, Quantity([1.0, 1.0806046, -1.2484405, -3.95997], u.kpc)
        )
        assert array_equal(
            cart2d.y, Quantity([0.0, 1.6829419, 2.7278922, 0.56448], u.kpc)
        )

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cylindrical_to_polar(self, vector):
        """Test ``vector.represent_as(PolarVector)``."""
        polar = vector.represent_as(PolarVector)

        assert isinstance(polar, PolarVector)
        assert array_equal(polar.r, Quantity([1, 2, 3, 4], u.kpc))
        assert array_equal(polar.phi, Quantity([0, 1, 2, 3], u.rad))

    # @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    # def test_cylindrical_to_lnpolar(self, vector):
    #     """Test ``vector.represent_as(LnPolarVector)``."""
    #     assert False

    # @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    # def test_cylindrical_to_log10polar(self, vector):
    #     """Test ``vector.represent_as(Log10PolarVector)``."""
    #     assert False

    def test_cylindrical_to_cartesian3d(self, vector):
        """Test ``vector.represent_as(Cartesian3DVector)``."""
        cart3d = vector.represent_as(Cartesian3DVector)

        assert isinstance(cart3d, Cartesian3DVector)
        assert array_equal(
            cart3d.x, Quantity([1.0, 1.0806046, -1.2484405, -3.95997], u.kpc)
        )
        assert array_equal(
            cart3d.y, Quantity([0.0, 1.6829419, 2.7278922, 0.56448], u.kpc)
        )
        assert array_equal(cart3d.z, vector.z)

    def test_cylindrical_to_spherical(self, vector):
        """Test ``vector.represent_as(SphericalVector)``."""
        spherical = vector.represent_as(SphericalVector)

        assert isinstance(spherical, SphericalVector)
        assert array_equal(spherical.r, Quantity([1, 2, 3, 4], u.kpc))
        assert array_equal(spherical.phi, Quantity([0, 1, 2, 3], u.rad))
        assert array_equal(spherical.theta, Quantity(xp.full(4, xp.pi / 2), u.rad))

    def test_cylindrical_to_cylindrical(self, vector):
        """Test ``vector.represent_as(CylindricalVector)``."""
        # Jit can copy
        newvec = vector.represent_as(CylindricalVector)
        assert newvec == vector

        # The normal `represent_as` method should return the same object
        newvec = represent_as(vector, CylindricalVector)
        assert newvec is vector


class Abstract3DVectorDifferentialTest(AbstractVectorDifferentialTest):
    """Test :class:`vector.Abstract2DVectorDifferential`."""


class TestCartesianDifferential3D(Abstract3DVectorDifferentialTest):
    """Test :class:`vector.CartesianDifferential3D`."""

    @pytest.fixture(scope="class")
    def difntl(self) -> CartesianDifferential3D:
        """Return a differential."""
        return CartesianDifferential3D(
            d_x=Quantity([1, 2, 3, 4], u.km / u.s),
            d_y=Quantity([5, 6, 7, 8], u.km / u.s),
            d_z=Quantity([9, 10, 11, 12], u.km / u.s),
        )

    @pytest.fixture(scope="class")
    def vector(self) -> Cartesian3DVector:
        """Return a vector."""
        return Cartesian3DVector(
            x=Quantity([1, 2, 3, 4], u.kpc),
            y=Quantity([5, 6, 7, 8], u.kpc),
            z=Quantity([9, 10, 11, 12], u.kpc),
        )

    # ==========================================================================

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian3d_to_cartesian1d(self, difntl, vector):
        """Test ``vector.represent_as(Cartesian1DVector)``."""
        cart1d = difntl.represent_as(CartesianDifferential1D, vector)

        assert isinstance(cart1d, CartesianDifferential1D)
        assert array_equal(cart1d.d_x, Quantity([1, 2, 3, 4], u.km / u.s))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian3d_to_radial(self, difntl, vector):
        """Test ``vector.represent_as(RadialVector)``."""
        radial = difntl.represent_as(RadialVector, vector)

        assert isinstance(radial, RadialVector)
        assert array_equal(radial.d_r, Quantity([1, 2, 3, 4], u.km / u.s))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian3d_to_cartesian2d(self, difntl, vector):
        """Test ``vector.represent_as(Cartesian2DVector)``."""
        cart2d = difntl.represent_as(CartesianDifferential2D, vector)

        assert isinstance(cart2d, CartesianDifferential2D)
        assert array_equal(cart2d.d_x, Quantity([1, 2, 3, 4], u.km / u.s))
        assert array_equal(cart2d.d_y, Quantity([5, 6, 7, 8], u.km / u.s))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian3d_to_polar(self, difntl, vector):
        """Test ``vector.represent_as(PolarVector)``."""
        polar = difntl.represent_as(PolarVector, vector)

        assert isinstance(polar, PolarVector)
        assert array_equal(polar.d_r, Quantity([1, 2, 3, 4], u.km / u.s))
        assert array_equal(polar.d_phi, Quantity([5, 6, 7, 8], u.mas / u.yr))

    def test_cartesian3d_to_cartesian3d(self, difntl, vector):
        """Test ``vector.represent_as(Cartesian3DVector)``."""
        # Jit can copy
        newvec = difntl.represent_as(CartesianDifferential3D, vector)
        assert newvec == difntl

        # The normal `represent_as` method should return the same object
        newvec = represent_as(difntl, CartesianDifferential3D, vector)
        assert newvec is difntl

    def test_cartesian3d_to_spherical(self, difntl, vector):
        """Test ``vector.represent_as(SphericalDifferential)``."""
        spherical = difntl.represent_as(SphericalDifferential, vector)

        assert isinstance(spherical, SphericalDifferential)
        assert array_equal(
            spherical.d_r,
            Quantity([10.344081, 11.832159, 13.379088, 14.966629], u.km / u.s),
        )
        assert jnp.allclose(spherical.d_phi.value, xp.asarray([0, 0, 2e-8, 0]))
        assert jnp.allclose(
            spherical.d_theta.to_value(u.mas / u.Myr),
            xp.asarray([0.03221978, -0.05186598, -0.01964621, -0.01886036]),
        )

    def test_cartesian3d_to_cylindrical(self, difntl, vector):
        """Test ``vector.represent_as(CylindricalDifferential)``."""
        cylindrical = difntl.represent_as(CylindricalDifferential, vector)

        assert isinstance(cylindrical, CylindricalDifferential)
        assert array_equal(
            cylindrical.d_rho,
            Quantity([5.0990195, 6.324555, 7.6157727, 8.944272], u.km / u.s),
        )
        assert jnp.allclose(
            cylindrical.d_phi.to_value(u.mas / u.Myr), xp.asarray([0, 0, 0.00471509, 0])
        )
        assert array_equal(cylindrical.d_z, Quantity([9, 10, 11, 12], u.km / u.s))


class TestSphericalDifferential(Abstract3DVectorDifferentialTest):
    """Test :class:`vector.SphericalDifferential`."""

    @pytest.fixture(scope="class")
    def difntl(self) -> SphericalDifferential:
        """Return a differential."""
        return SphericalDifferential(
            d_r=Quantity([1, 2, 3, 4], u.km / u.s),
            d_phi=Quantity([5, 6, 7, 8], u.mas / u.yr),
            d_theta=Quantity([9, 10, 11, 12], u.mas / u.yr),
        )

    @pytest.fixture(scope="class")
    def vector(self) -> SphericalVector:
        """Return a vector."""
        return SphericalVector(
            r=Quantity([1, 2, 3, 4], u.kpc),
            phi=Quantity([0, 1, 2, 3], u.rad),
            theta=Quantity([4, 5, 6, 7], u.rad),
        )

    # ==========================================================================

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_spherical_to_cartesian1d(self, difntl, vector):
        """Test ``vector.represent_as(Cartesian1DVector)``."""
        cart1d = difntl.represent_as(CartesianDifferential1D, vector)

        assert isinstance(cart1d, CartesianDifferential1D)
        assert array_equal(cart1d.d_x, Quantity([1, 2, 3, 4], u.km / u.s))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_spherical_to_radial(self, difntl, vector):
        """Test ``vector.represent_as(RadialVector)``."""
        radial = difntl.represent_as(RadialVector, vector)

        assert isinstance(radial, RadialVector)
        assert array_equal(radial.d_r, Quantity([1, 2, 3, 4], u.km / u.s))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_spherical_to_cartesian2d(self, difntl, vector):
        """Test ``vector.represent_as(Cartesian2DVector)``."""
        cart2d = difntl.represent_as(CartesianDifferential2D, vector)

        assert isinstance(cart2d, CartesianDifferential2D)
        assert array_equal(cart2d.d_x, Quantity([1, 2, 3, 4], u.km / u.s))
        assert array_equal(cart2d.d_y, Quantity([5, 6, 7, 8], u.km / u.s))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_spherical_to_polar(self, difntl, vector):
        """Test ``vector.represent_as(PolarVector)``."""
        polar = difntl.represent_as(PolarVector, vector)

        assert isinstance(polar, PolarVector)
        assert array_equal(polar.d_r, Quantity([1, 2, 3, 4], u.km / u.s))
        assert array_equal(polar.d_phi, Quantity([5, 6, 7, 8], u.mas / u.yr))

    def test_spherical_to_cartesian3d(self, difntl, vector):
        """Test ``vector.represent_as(Cartesian3DVector)``."""
        cart3d = difntl.represent_as(CartesianDifferential3D, vector)

        assert isinstance(cart3d, CartesianDifferential3D)
        assert array_equal(
            cart3d.d_x,
            Quantity([-28.644005, 59.39601, -36.865578, -186.49405], u.km / u.s),
        )
        assert array_equal(
            cart3d.d_y,
            Quantity([-17.938, -8.456386, 147.39401, -74.084984], u.km / u.s),
        )
        assert array_equal(
            cart3d.d_z,
            Quantity([31.634754, 91.48237, 46.59102, -146.47682], u.km / u.s),
        )

    def test_spherical_to_spherical(self, difntl, vector):
        """Test ``vector.represent_as(SphericalDifferential)``."""
        # Jit can copy
        newvec = difntl.represent_as(SphericalDifferential, vector)
        assert newvec == difntl

        # The normal `represent_as` method should return the same object
        newvec = represent_as(difntl, SphericalDifferential, vector)
        assert newvec is difntl

    def test_spherical_to_cylindrical(self, difntl, vector):
        """Test ``vector.represent_as(CylindricalDifferential)``."""
        cylindrical = difntl.represent_as(CylindricalDifferential, vector)

        assert isinstance(cylindrical, CylindricalDifferential)
        assert array_equal(
            cylindrical.d_rho,
            Quantity([-28.644005, 24.975996, 149.3665, 174.17282], u.km / u.s),
        )
        assert array_equal(cylindrical.d_phi, Quantity([5, 6, 7, 8], u.mas / u.yr))
        assert array_equal(
            cylindrical.d_z,
            Quantity([31.634754, 91.48237, 46.59102, -146.47682], u.km / u.s),
        )


class TestCylindricalDifferential(Abstract3DVectorDifferentialTest):
    """Test :class:`vector.CylindricalDifferential`."""

    @pytest.fixture(scope="class")
    def difntl(self) -> CylindricalDifferential:
        """Return a differential."""
        return CylindricalDifferential(
            d_rho=Quantity([1, 2, 3, 4], u.km / u.s),
            d_phi=Quantity([5, 6, 7, 8], u.mas / u.yr),
            d_z=Quantity([9, 10, 11, 12], u.km / u.s),
        )

    @pytest.fixture(scope="class")
    def vector(self) -> CylindricalVector:
        """Return a vector."""
        return CylindricalVector(
            rho=Quantity([1, 2, 3, 4], u.kpc),
            phi=Quantity([0, 1, 2, 3], u.rad),
            z=Quantity([9, 10, 11, 12], u.kpc),
        )

    # ==========================================================================

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cylindrical_to_cartesian1d(self, difntl, vector):
        """Test ``vector.represent_as(Cartesian1DVector)``."""
        cart1d = difntl.represent_as(CartesianDifferential1D, vector)

        assert isinstance(cart1d, CartesianDifferential1D)
        assert array_equal(cart1d.d_x, Quantity([1, 2, 3, 4], u.km / u.s))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cylindrical_to_radial(self, difntl, vector):
        """Test ``vector.represent_as(RadialVector)``."""
        radial = difntl.represent_as(RadialVector, vector)

        assert isinstance(radial, RadialVector)
        assert array_equal(radial.d_r, Quantity([1, 2, 3, 4], u.km / u.s))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cylindrical_to_cartesian2d(self, difntl, vector):
        """Test ``vector.represent_as(Cartesian2DVector)``."""
        cart2d = difntl.represent_as(CartesianDifferential2D, vector)

        assert isinstance(cart2d, CartesianDifferential2D)
        assert array_equal(cart2d.d_x, Quantity([1, 2, 3, 4], u.km / u.s))
        assert array_equal(cart2d.d_y, Quantity([5, 6, 7, 8], u.km / u.s))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cylindrical_to_polar(self, difntl, vector):
        """Test ``vector.represent_as(PolarVector)``."""
        polar = difntl.represent_as(PolarVector, vector)

        assert isinstance(polar, PolarVector)
        assert array_equal(polar.d_r, Quantity([1, 2, 3, 4], u.km / u.s))
        assert array_equal(polar.d_phi, Quantity([5, 6, 7, 8], u.mas / u.yr))

    def test_cylindrical_to_cartesian3d(self, difntl, vector):
        """Test ``vector.represent_as(Cartesian3DVector)``."""
        cart3d = difntl.represent_as(CartesianDifferential3D, vector)

        assert isinstance(cart3d, CartesianDifferential3D)
        assert array_equal(
            cart3d.d_x, Quantity([1.0, -46.787014, -91.76889, -25.367176], u.km / u.s)
        )
        assert array_equal(
            cart3d.d_y,
            Quantity([23.702353, 32.418385, -38.69947, -149.61249], u.km / u.s),
        )
        assert array_equal(cart3d.d_z, Quantity([9, 10, 11, 12], u.km / u.s))

    def test_cylindrical_to_spherical(self, difntl, vector):
        """Test ``vector.represent_as(SphericalDifferential)``."""
        spherical = difntl.represent_as(SphericalDifferential, vector)

        assert isinstance(spherical, SphericalDifferential)
        assert array_equal(
            spherical.d_r,
            Quantity([9.055385, 10.198039, 11.401753, 12.64911], u.km / u.s),
        )
        assert array_equal(spherical.d_phi, Quantity([5, 6, 7, 8], u.mas / u.yr))
        assert jnp.allclose(
            spherical.d_theta.to_value(u.mas / u.Myr),
            xp.asarray([-0.08428223, 0.07544143, -0.0326127, -0.01571696]),
        )

    def test_cylindrical_to_cylindrical(self, difntl, vector):
        """Test ``vector.represent_as(CylindricalDifferential)``."""
        # Jit can copy
        newvec = difntl.represent_as(CylindricalDifferential, vector)
        assert newvec == difntl

        # The normal `represent_as` method should return the same object
        newvec = represent_as(difntl, CylindricalDifferential, vector)
        assert newvec is difntl
