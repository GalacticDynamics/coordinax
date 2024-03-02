"""Test :mod:`coordinax._builtin`."""

import astropy.coordinates as apyc
import astropy.units as u
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.coordinates.tests.test_representation import representation_equal
from plum import convert

import array_api_jax_compat as xp
from jax_quantity import Quantity

from .test_base import AbstractVectorDifferentialTest, AbstractVectorTest, array_equal
from .test_d2 import hypot
from coordinax import (
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
from coordinax._d1.builtin import CartesianDifferential1D
from coordinax._d2.builtin import CartesianDifferential2D
from coordinax._d3.builtin import (
    CartesianDifferential3D,
    CylindricalDifferential,
    SphericalDifferential,
)


class Abstract3DVectorTest(AbstractVectorTest):
    """Test :class:`coordinax.Abstract3DVector`."""

    # ==========================================================================
    # Unary operations

    def test_neg_compare_apy(
        self, vector: AbstractVector, apyvector: apyc.BaseRepresentation
    ):
        """Test negation."""
        # To take the negative, Vector converts to Cartesian coordinates, takes
        # the negative, then converts back to the original representation.
        # This can result in equivalent but different angular coordinates than
        # Astropy. AFAIK this only happens at the poles.
        cart = convert(-vector, type(apyvector)).represent_as(
            apyc.CartesianRepresentation
        )
        apycart = -apyvector.represent_as(apyc.CartesianRepresentation)
        assert np.allclose(cart.x, apycart.x, atol=5e-7)
        assert np.allclose(cart.y, apycart.y, atol=5e-7)
        assert np.allclose(cart.z, apycart.z, atol=5e-7)

        # # Try finding the poles
        # if hasattr(vector, "theta"):
        #     sel = (vector.theta.to_value("deg") != 0) & (
        #         vector.theta.to_value("deg") != 180
        #     )
        # else:
        #     sel = slice(None)
        # vecsel = convert(-vector[sel], type(apyvector))
        # apyvecsel = -apyvector[sel]
        # for c in vecsel.components:
        #     unit = getattr(apyvecsel, c).unit
        #     assert np.allclose(
        #         getattr(vecsel, c).to_value(unit),
        #         getattr(apyvecsel, c).to_value(unit),
        #         atol=5e-7,
        #     )


class TestCartesian3DVector(Abstract3DVectorTest):
    """Test :class:`coordinax.Cartesian3DVector`."""

    @pytest.fixture(scope="class")
    def vector(self) -> AbstractVector:
        """Return a vector."""
        from coordinax import Cartesian3DVector

        return Cartesian3DVector(
            x=Quantity([1, 2, 3, 4], u.kpc),
            y=Quantity([5, 6, 7, 8], u.kpc),
            z=Quantity([9, 10, 11, 12], u.kpc),
        )

    @pytest.fixture(scope="class")
    def apyvector(self, vector: AbstractVector) -> apyc.CartesianRepresentation:
        """Return an Astropy vector."""
        return convert(vector, apyc.CartesianRepresentation)

    # ==========================================================================
    # represent_as

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cartesian3d_to_cartesian1d(self, vector):
        """Test ``coordinax.represent_as(Cartesian1DVector)``."""
        cart1d = vector.represent_as(Cartesian1DVector)

        assert isinstance(cart1d, Cartesian1DVector)
        assert array_equal(cart1d.x, Quantity([1, 2, 3, 4], u.kpc))

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cartesian3d_to_radial(self, vector):
        """Test ``coordinax.represent_as(RadialVector)``."""
        radial = vector.represent_as(RadialVector)

        assert isinstance(radial, RadialVector)
        assert array_equal(
            radial.r, Quantity([10.34408, 11.83216, 13.379088, 14.96663], u.kpc)
        )

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cartesian3d_to_cartesian2d(self, vector):
        """Test ``coordinax.represent_as(Cartesian2DVector)``."""
        cart2d = vector.represent_as(Cartesian2DVector, y=Quantity([5, 6, 7, 8], u.km))

        assert isinstance(cart2d, Cartesian2DVector)
        assert array_equal(cart2d.x, Quantity([1, 2, 3, 4], u.kpc))
        assert array_equal(cart2d.y, Quantity([5, 6, 7, 8], u.kpc))

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cartesian3d_to_polar(self, vector):
        """Test ``coordinax.represent_as(PolarVector)``."""
        polar = vector.represent_as(PolarVector, phi=Quantity([0, 1, 2, 3], u.rad))

        assert isinstance(polar, PolarVector)
        assert array_equal(polar.r, hypot(vector.x, vector.y))
        assert array_equal(
            polar.phi, Quantity([1.3734008, 1.2490457, 1.1659045, 1.1071488], u.rad)
        )

    # @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    # def test_cartesian3d_to_lnpolar(self, vector):
    #     """Test ``coordinax.represent_as(LnPolarVector)``."""
    #     assert False

    # @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    # def test_cartesian3d_to_log10polar(self, vector):
    #     """Test ``coordinax.represent_as(Log10PolarVector)``."""
    #     assert False

    def test_cartesian3d_to_cartesian3d(self, vector):
        """Test ``coordinax.represent_as(Cartesian3DVector)``."""
        # Jit can copy
        newvec = vector.represent_as(Cartesian3DVector)
        assert newvec == vector

        # The normal `represent_as` method should return the same object
        newvec = represent_as(vector, Cartesian3DVector)
        assert newvec is vector

    def test_cartesian3d_to_cartesian3d_astropy(self, vector, apyvector):
        """Test Astropy equivalence."""
        newvec = vector.represent_as(Cartesian3DVector)

        assert np.allclose(convert(newvec.x, u.Quantity), apyvector.x)
        assert np.allclose(convert(newvec.y, u.Quantity), apyvector.y)
        assert np.allclose(convert(newvec.z, u.Quantity), apyvector.z)

    def test_cartesian3d_to_spherical(self, vector):
        """Test ``coordinax.represent_as(SphericalVector)``."""
        spherical = vector.represent_as(SphericalVector)

        assert isinstance(spherical, SphericalVector)
        assert array_equal(
            spherical.r, Quantity([10.34408, 11.83216, 13.379088, 14.96663], u.kpc)
        )
        assert array_equal(
            spherical.phi, Quantity([1.3734008, 1.2490457, 1.1659045, 1.1071488], u.rad)
        )
        assert jnp.allclose(
            spherical.theta.to_value(u.rad),
            xp.asarray([0.51546645, 0.5639427, 0.6055685, 0.64052236]),
        )

    def test_cartesian3d_to_spherical_astropy(self, vector, apyvector):
        """Test Astropy equivalence."""
        sph = vector.represent_as(SphericalVector)

        apysph = apyvector.represent_as(apyc.PhysicsSphericalRepresentation)
        assert np.allclose(convert(sph.r, u.Quantity), apysph.r)
        assert np.allclose(convert(sph.theta, u.Quantity), apysph.theta)
        assert np.allclose(convert(sph.phi, u.Quantity), apysph.phi)

    def test_cartesian3d_to_cylindrical(self, vector):
        """Test ``coordinax.represent_as(CylindricalVector)``."""
        cylindrical = vector.represent_as(CylindricalVector)

        assert isinstance(cylindrical, CylindricalVector)
        assert array_equal(cylindrical.rho, hypot(vector.x, vector.y))
        assert array_equal(
            cylindrical.phi,
            Quantity([1.3734008, 1.2490457, 1.1659045, 1.1071488], u.rad),
        )
        assert array_equal(cylindrical.z, Quantity([9.0, 10, 11, 12], u.kpc))

    def test_cartesian3d_to_cylindrical_astropy(self, vector, apyvector):
        """Test Astropy equivalence."""
        cyl = vector.represent_as(CylindricalVector)

        apycyl = apyvector.represent_as(apyc.CylindricalRepresentation)
        assert np.allclose(convert(cyl.rho, u.Quantity), apycyl.rho)
        assert np.allclose(convert(cyl.z, u.Quantity), apycyl.z)
        assert np.allclose(convert(cyl.phi, u.Quantity), apycyl.phi)


class TestSphericalVector(Abstract3DVectorTest):
    """Test :class:`coordinax.SphericalVector`."""

    @pytest.fixture(scope="class")
    def vector(self) -> SphericalVector:
        """Return a vector."""
        return SphericalVector(
            r=Quantity([1, 2, 3, 4], u.kpc),
            phi=Quantity([0, 65, 135, 270], u.deg),
            theta=Quantity([0, 36, 142, 180], u.deg),
        )

    @pytest.fixture(scope="class")
    def apyvector(self, vector: AbstractVector):
        """Return an Astropy vector."""
        return convert(vector, apyc.PhysicsSphericalRepresentation)

    # ==========================================================================
    # represent_as

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_spherical_to_cartesian1d(self, vector):
        """Test ``coordinax.represent_as(Cartesian1DVector)``."""
        cart1d = vector.represent_as(Cartesian1DVector)

        assert isinstance(cart1d, Cartesian1DVector)
        assert jnp.allclose(
            cart1d.x.to_value(u.kpc),
            xp.asarray([0, 0.49681753, -1.3060151, -4.1700245e-15]),
        )

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_spherical_to_radial(self, vector):
        """Test ``coordinax.represent_as(RadialVector)``."""
        radial = vector.represent_as(RadialVector)

        assert isinstance(radial, RadialVector)
        assert array_equal(radial.r, Quantity([1, 2, 3, 4], u.kpc))

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_spherical_to_cartesian2d(self, vector):
        """Test ``coordinax.represent_as(Cartesian2DVector)``."""
        cart2d = vector.represent_as(Cartesian2DVector, y=Quantity([5, 6, 7, 8], u.km))

        assert isinstance(cart2d, Cartesian2DVector)
        assert array_equal(
            cart2d.x,
            Quantity([0, 0.49681753, -1.3060151, -4.1700245e-15], u.kpc),
        )
        assert array_equal(
            cart2d.y, Quantity([0.0, 1.0654287, 1.3060151, 3.4969111e-07], u.kpc)
        )

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_spherical_to_polar(self, vector):
        """Test ``coordinax.represent_as(PolarVector)``."""
        polar = vector.represent_as(PolarVector, phi=Quantity([0, 1, 2, 3], u.rad))

        assert isinstance(polar, PolarVector)
        assert array_equal(
            polar.r,
            Quantity([0.0, 1.1755705, 1.8469844, -3.4969111e-07], u.kpc),
        )
        assert array_equal(polar.phi, Quantity([0.0, 65.0, 135.0, 270.0], u.deg))

    # @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    # def test_spherical_to_lnpolar(self, vector):
    #     """Test ``coordinax.represent_as(LnPolarVector)``."""
    #     assert False

    # @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    # def test_spherical_to_log10polar(self, vector):
    #     """Test ``coordinax.represent_as(Log10PolarVector)``."""
    #     assert False

    def test_spherical_to_cartesian3d(self, vector):
        """Test ``coordinax.represent_as(Cartesian3DVector)``."""
        cart3d = vector.represent_as(Cartesian3DVector)

        assert isinstance(cart3d, Cartesian3DVector)
        assert array_equal(
            cart3d.x, Quantity([0, 0.49681753, -1.3060151, -4.1700245e-15], u.kpc)
        )
        assert array_equal(
            cart3d.y, Quantity([0.0, 1.0654287, 1.3060151, 3.4969111e-07], u.kpc)
        )
        assert array_equal(cart3d.z, Quantity([1.0, 1.618034, -2.3640323, -4.0], u.kpc))

    def test_spherical_to_cartesian3d_astropy(self, vector, apyvector):
        """Test Astropy equivalence."""
        cart3d = vector.represent_as(Cartesian3DVector)

        apycart3 = apyvector.represent_as(apyc.CartesianRepresentation)
        assert np.allclose(convert(cart3d.x, u.Quantity), apycart3.x)
        assert np.allclose(convert(cart3d.y, u.Quantity), apycart3.y)
        assert np.allclose(convert(cart3d.z, u.Quantity), apycart3.z)

    def test_spherical_to_spherical(self, vector):
        """Test ``coordinax.represent_as(SphericalVector)``."""
        # Jit can copy
        newvec = vector.represent_as(SphericalVector)
        assert newvec == vector

        # The normal `represent_as` method should return the same object
        newvec = represent_as(vector, SphericalVector)
        assert newvec is vector

    def test_spherical_to_spherical_astropy(self, vector, apyvector):
        """Test Astropy equivalence."""
        sph = vector.represent_as(SphericalVector)

        apysph = apyvector.represent_as(apyc.PhysicsSphericalRepresentation)
        assert np.allclose(convert(sph.r, u.Quantity), apysph.r)
        assert np.allclose(convert(sph.theta, u.Quantity), apysph.theta)
        assert np.allclose(convert(sph.phi, u.Quantity), apysph.phi)

    def test_spherical_to_cylindrical(self, vector):
        """Test ``coordinax.represent_as(CylindricalVector)``."""
        cylindrical = vector.represent_as(
            CylindricalVector, z=Quantity([9, 10, 11, 12], u.m)
        )

        assert isinstance(cylindrical, CylindricalVector)
        assert array_equal(
            cylindrical.rho,
            Quantity([0.0, 1.1755705, 1.8469844, 3.4969111e-07], u.kpc),
        )
        assert array_equal(cylindrical.phi, Quantity([0.0, 65.0, 135.0, 270.0], u.deg))
        assert array_equal(
            cylindrical.z, Quantity([1.0, 1.618034, -2.3640323, -4.0], u.kpc)
        )

    def test_spherical_to_cylindrical_astropy(self, vector, apyvector):
        """Test ``coordinax.represent_as(CylindricalVector)``."""
        cyl = vector.represent_as(CylindricalVector, z=Quantity([9, 10, 11, 12], u.m))

        apycyl = apyvector.represent_as(apyc.CylindricalRepresentation)
        assert np.allclose(convert(cyl.rho, u.Quantity), apycyl.rho)
        assert np.allclose(convert(cyl.z, u.Quantity), apycyl.z)

        with pytest.raises(AssertionError):  # TODO: Fix this
            assert np.allclose(convert(cyl.phi, u.Quantity), apycyl.phi)


class TestCylindricalVector(Abstract3DVectorTest):
    """Test :class:`coordinax.CylindricalVector`."""

    @pytest.fixture(scope="class")
    def vector(self) -> AbstractVector:
        """Return a vector."""
        from coordinax import CylindricalVector

        return CylindricalVector(
            rho=Quantity([1, 2, 3, 4], u.kpc),
            phi=Quantity([0, 1, 2, 3], u.rad),
            z=Quantity([9, 10, 11, 12], u.m),
        )

    @pytest.fixture(scope="class")
    def apyvector(self, vector: AbstractVector):
        """Return an Astropy vector."""
        return convert(vector, apyc.CylindricalRepresentation)

    # ==========================================================================
    # represent_as

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cylindrical_to_cartesian1d(self, vector):
        """Test ``coordinax.represent_as(Cartesian1DVector)``."""
        cart1d = vector.represent_as(Cartesian1DVector)

        assert isinstance(cart1d, Cartesian1DVector)
        assert jnp.allclose(
            cart1d.x.to_value(u.kpc), xp.asarray([1.0, 1.0806047, -1.2484405, -3.95997])
        )

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cylindrical_to_radial(self, vector):
        """Test ``coordinax.represent_as(RadialVector)``."""
        radial = vector.represent_as(RadialVector)

        assert isinstance(radial, RadialVector)
        assert array_equal(radial.r, Quantity([1, 2, 3, 4], u.kpc))

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cylindrical_to_cartesian2d(self, vector):
        """Test ``coordinax.represent_as(Cartesian2DVector)``."""
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
        """Test ``coordinax.represent_as(PolarVector)``."""
        polar = vector.represent_as(PolarVector)

        assert isinstance(polar, PolarVector)
        assert array_equal(polar.r, Quantity([1, 2, 3, 4], u.kpc))
        assert array_equal(polar.phi, Quantity([0, 1, 2, 3], u.rad))

    # @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    # def test_cylindrical_to_lnpolar(self, vector):
    #     """Test ``coordinax.represent_as(LnPolarVector)``."""
    #     assert False

    # @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    # def test_cylindrical_to_log10polar(self, vector):
    #     """Test ``coordinax.represent_as(Log10PolarVector)``."""
    #     assert False

    def test_cylindrical_to_cartesian3d(self, vector):
        """Test ``coordinax.represent_as(Cartesian3DVector)``."""
        cart3d = vector.represent_as(Cartesian3DVector)

        assert isinstance(cart3d, Cartesian3DVector)
        assert array_equal(
            cart3d.x, Quantity([1.0, 1.0806046, -1.2484405, -3.95997], u.kpc)
        )
        assert array_equal(
            cart3d.y, Quantity([0.0, 1.6829419, 2.7278922, 0.56448], u.kpc)
        )
        assert array_equal(cart3d.z, vector.z)

    def test_cylindrical_to_cartesian3d_astropy(self, vector, apyvector):
        """Test Astropy equivalence."""
        cart3d = vector.represent_as(Cartesian3DVector)

        apycart3 = apyvector.represent_as(apyc.CartesianRepresentation)
        assert np.allclose(convert(cart3d.x, u.Quantity), apycart3.x)
        assert np.allclose(convert(cart3d.y, u.Quantity), apycart3.y)
        assert np.allclose(convert(cart3d.z, u.Quantity), apycart3.z)

    def test_cylindrical_to_spherical(self, vector):
        """Test ``coordinax.represent_as(SphericalVector)``."""
        spherical = vector.represent_as(SphericalVector)

        assert isinstance(spherical, SphericalVector)
        assert array_equal(spherical.r, Quantity([1, 2, 3, 4], u.kpc))
        assert array_equal(spherical.phi, Quantity([0, 1, 2, 3], u.rad))
        assert array_equal(spherical.theta, Quantity(xp.full(4, xp.pi / 2), u.rad))

    def test_cylindrical_to_spherical_astropy(self, vector, apyvector):
        """Test Astropy equivalence."""
        sph = vector.represent_as(SphericalVector)
        apysph = apyvector.represent_as(apyc.PhysicsSphericalRepresentation)
        assert np.allclose(convert(sph.r, u.Quantity), apysph.r)
        assert np.allclose(convert(sph.theta, u.Quantity), apysph.theta)
        assert np.allclose(convert(sph.phi, u.Quantity), apysph.phi)

    def test_cylindrical_to_cylindrical(self, vector):
        """Test ``coordinax.represent_as(CylindricalVector)``."""
        # Jit can copy
        newvec = vector.represent_as(CylindricalVector)
        assert newvec == vector

        # The normal `represent_as` method should return the same object
        newvec = represent_as(vector, CylindricalVector)
        assert newvec is vector

    def test_cylindrical_to_cylindrical_astropy(self, vector, apyvector):
        """Test Astropy equivalence."""
        cyl = vector.represent_as(CylindricalVector)

        apycyl = apyvector.represent_as(apyc.CylindricalRepresentation)
        assert np.allclose(convert(cyl.rho, u.Quantity), apycyl.rho)
        assert np.allclose(convert(cyl.z, u.Quantity), apycyl.z)
        assert np.allclose(convert(cyl.phi, u.Quantity), apycyl.phi)


class Abstract3DVectorDifferentialTest(AbstractVectorDifferentialTest):
    """Test :class:`coordinax.Abstract2DVectorDifferential`."""

    # ==========================================================================
    # Unary operations

    def test_neg_compare_apy(
        self, difntl: AbstractVector, apydifntl: apyc.BaseRepresentation
    ):
        """Test negation."""
        assert all(representation_equal(convert(-difntl, type(apydifntl)), -apydifntl))


class TestCartesianDifferential3D(Abstract3DVectorDifferentialTest):
    """Test :class:`coordinax.CartesianDifferential3D`."""

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

    @pytest.fixture(scope="class")
    def apydifntl(self, difntl: CartesianDifferential3D):
        """Return an Astropy differential."""
        return convert(difntl, apyc.CartesianDifferential)

    @pytest.fixture(scope="class")
    def apyvector(self, vector: Cartesian3DVector):
        """Return an Astropy vector."""
        return convert(vector, apyc.CartesianRepresentation)

    # ==========================================================================

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian3d_to_cartesian1d(self, difntl, vector):
        """Test ``coordinax.represent_as(Cartesian1DVector)``."""
        cart1d = difntl.represent_as(CartesianDifferential1D, vector)

        assert isinstance(cart1d, CartesianDifferential1D)
        assert array_equal(cart1d.d_x, Quantity([1, 2, 3, 4], u.km / u.s))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian3d_to_radial(self, difntl, vector):
        """Test ``coordinax.represent_as(RadialVector)``."""
        radial = difntl.represent_as(RadialVector, vector)

        assert isinstance(radial, RadialVector)
        assert array_equal(radial.d_r, Quantity([1, 2, 3, 4], u.km / u.s))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian3d_to_cartesian2d(self, difntl, vector):
        """Test ``coordinax.represent_as(Cartesian2DVector)``."""
        cart2d = difntl.represent_as(CartesianDifferential2D, vector)

        assert isinstance(cart2d, CartesianDifferential2D)
        assert array_equal(cart2d.d_x, Quantity([1, 2, 3, 4], u.km / u.s))
        assert array_equal(cart2d.d_y, Quantity([5, 6, 7, 8], u.km / u.s))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian3d_to_polar(self, difntl, vector):
        """Test ``coordinax.represent_as(PolarVector)``."""
        polar = difntl.represent_as(PolarVector, vector)

        assert isinstance(polar, PolarVector)
        assert array_equal(polar.d_r, Quantity([1, 2, 3, 4], u.km / u.s))
        assert array_equal(polar.d_phi, Quantity([5, 6, 7, 8], u.mas / u.yr))

    def test_cartesian3d_to_cartesian3d(self, difntl, vector):
        """Test ``coordinax.represent_as(Cartesian3DVector)``."""
        # Jit can copy
        newvec = difntl.represent_as(CartesianDifferential3D, vector)
        assert newvec == difntl

        # The normal `represent_as` method should return the same object
        newvec = represent_as(difntl, CartesianDifferential3D, vector)
        assert newvec is difntl

    def test_cartesian3d_to_cartesian3d_astropy(
        self, difntl, vector, apydifntl, apyvector
    ):
        """Test Astropy equivalence."""
        cart3 = difntl.represent_as(CartesianDifferential3D, vector)

        apycart3 = apydifntl.represent_as(apyc.CartesianDifferential, apyvector)
        assert np.allclose(convert(cart3.d_x, u.Quantity), apycart3.d_x)
        assert np.allclose(convert(cart3.d_y, u.Quantity), apycart3.d_y)
        assert np.allclose(convert(cart3.d_z, u.Quantity), apycart3.d_z)

    def test_cartesian3d_to_spherical(self, difntl, vector):
        """Test ``coordinax.represent_as(SphericalDifferential)``."""
        spherical = difntl.represent_as(SphericalDifferential, vector)

        assert isinstance(spherical, SphericalDifferential)
        assert jnp.allclose(
            spherical.d_r.to_value(u.km / u.s),
            xp.asarray([10.344081, 11.832159, 13.379088, 14.966629]),
        )
        assert jnp.allclose(
            spherical.d_phi.to_value(u.mas / u.Myr), xp.asarray([0, 0, 0.00471509, 0])
        )
        assert jnp.allclose(
            spherical.d_theta.to_value(u.mas / u.Myr),
            xp.asarray([0.03221978, -0.05186598, -0.01964621, -0.01886036]),
        )

    def test_cartesian3d_to_spherical_astropy(
        self, difntl, vector, apydifntl, apyvector
    ):
        """Test Astropy equivalence."""
        sph = difntl.represent_as(SphericalDifferential, vector)

        apysph = apydifntl.represent_as(apyc.PhysicsSphericalDifferential, apyvector)
        assert np.allclose(convert(sph.d_r, u.Quantity), apysph.d_r)
        with pytest.raises(AssertionError):  # TODO: fixme
            assert np.allclose(
                convert(sph.d_theta, u.Quantity).to(u.mas / u.Myr),
                apysph.d_theta.to(u.mas / u.Myr),
                atol=1e-9,
            )
        with pytest.raises(AssertionError):  # TODO: fixme
            assert np.allclose(
                convert(sph.d_phi, u.Quantity).to(u.mas / u.Myr),
                apysph.d_phi.to(u.mas / u.Myr),
                atol=1e-7,
            )

    def test_cartesian3d_to_cylindrical(self, difntl, vector):
        """Test ``coordinax.represent_as(CylindricalDifferential)``."""
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

    def test_cartesian3d_to_cylindrical_astropy(
        self, difntl, vector, apydifntl, apyvector
    ):
        """Test Astropy equivalence."""
        cyl = difntl.represent_as(CylindricalDifferential, vector)
        apycyl = apydifntl.represent_as(apyc.CylindricalDifferential, apyvector)
        assert np.allclose(convert(cyl.d_rho, u.Quantity), apycyl.d_rho)
        assert np.allclose(convert(cyl.d_z, u.Quantity), apycyl.d_z)
        with pytest.raises(AssertionError):  # TODO: fixme
            assert np.allclose(convert(cyl.d_phi, u.Quantity), apycyl.d_phi)


class TestSphericalDifferential(Abstract3DVectorDifferentialTest):
    """Test :class:`coordinax.SphericalDifferential`."""

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
            phi=Quantity([0, 42, 160, 270], u.deg),
            theta=Quantity([3, 63, 90, 179.5], u.deg),
        )

    @pytest.fixture(scope="class")
    def apydifntl(
        self, difntl: SphericalDifferential
    ) -> apyc.PhysicsSphericalDifferential:
        """Return an Astropy differential."""
        return convert(difntl, apyc.PhysicsSphericalDifferential)

    @pytest.fixture(scope="class")
    def apyvector(self, vector: SphericalVector) -> apyc.PhysicsSphericalRepresentation:
        """Return an Astropy vector."""
        return convert(vector, apyc.PhysicsSphericalRepresentation)

    # ==========================================================================

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_spherical_to_cartesian1d(self, difntl, vector):
        """Test ``coordinax.represent_as(Cartesian1DVector)``."""
        cart1d = difntl.represent_as(CartesianDifferential1D, vector)

        assert isinstance(cart1d, CartesianDifferential1D)
        assert array_equal(cart1d.d_x, Quantity([1, 2, 3, 4], u.km / u.s))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_spherical_to_radial(self, difntl, vector):
        """Test ``coordinax.represent_as(RadialVector)``."""
        radial = difntl.represent_as(RadialVector, vector)

        assert isinstance(radial, RadialVector)
        assert array_equal(radial.d_r, Quantity([1, 2, 3, 4], u.km / u.s))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_spherical_to_cartesian2d(self, difntl, vector):
        """Test ``coordinax.represent_as(Cartesian2DVector)``."""
        cart2d = difntl.represent_as(CartesianDifferential2D, vector)

        assert isinstance(cart2d, CartesianDifferential2D)
        assert array_equal(cart2d.d_x, Quantity([1, 2, 3, 4], u.km / u.s))
        assert array_equal(cart2d.d_y, Quantity([5, 6, 7, 8], u.km / u.s))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_spherical_to_polar(self, difntl, vector):
        """Test ``coordinax.represent_as(PolarVector)``."""
        polar = difntl.represent_as(PolarVector, vector)

        assert isinstance(polar, PolarVector)
        assert array_equal(polar.d_r, Quantity([1, 2, 3, 4], u.km / u.s))
        assert array_equal(polar.d_phi, Quantity([5, 6, 7, 8], u.mas / u.yr))

    def test_spherical_to_cartesian3d(self, difntl, vector):
        """Test ``coordinax.represent_as(Cartesian3DVector)``."""
        cart3d = difntl.represent_as(CartesianDifferential3D, vector)

        assert isinstance(cart3d, CartesianDifferential3D)
        assert array_equal(
            cart3d.d_x,
            Quantity([42.658096, -0.6040496, -36.867138, 1.323785], u.km / u.s),
        )
        assert array_equal(
            cart3d.d_y,
            Quantity([1.2404853, 67.66016, -92.52022, 227.499], u.km / u.s),
        )
        assert array_equal(
            cart3d.d_z,
            Quantity([-1.2342439, -83.56782, -156.43553, -5.985529], u.km / u.s),
        )

    def test_spherical_to_cartesian3d_astropy(
        self, difntl, vector, apydifntl, apyvector
    ):
        """Test Astropy equivalence."""
        cart3d = difntl.represent_as(CartesianDifferential3D, vector)

        apycart3 = apydifntl.represent_as(apyc.CartesianDifferential, apyvector)
        assert np.allclose(convert(cart3d.d_x, u.Quantity), apycart3.d_x)
        assert np.allclose(convert(cart3d.d_y, u.Quantity), apycart3.d_y)
        assert np.allclose(convert(cart3d.d_z, u.Quantity), apycart3.d_z)

    def test_spherical_to_spherical(self, difntl, vector):
        """Test ``coordinax.represent_as(SphericalDifferential)``."""
        # Jit can copy
        newvec = difntl.represent_as(SphericalDifferential, vector)
        assert newvec == difntl

        # The normal `represent_as` method should return the same object
        newvec = represent_as(difntl, SphericalDifferential, vector)
        assert newvec is difntl

    def test_spherical_to_spherical_astropy(self, difntl, vector, apydifntl, apyvector):
        """Test Astropy equivalence."""
        sph = difntl.represent_as(SphericalDifferential, vector)
        apysph = apydifntl.represent_as(apyc.PhysicsSphericalDifferential, apyvector)
        assert np.allclose(convert(sph.d_r, u.Quantity), apysph.d_r)
        assert np.allclose(convert(sph.d_theta, u.Quantity), apysph.d_theta)
        assert np.allclose(convert(sph.d_phi, u.Quantity), apysph.d_phi)

    def test_spherical_to_cylindrical(self, difntl, vector):
        """Test ``coordinax.represent_as(CylindricalDifferential)``."""
        cylindrical = difntl.represent_as(CylindricalDifferential, vector)

        assert isinstance(cylindrical, CylindricalDifferential)
        assert array_equal(
            cylindrical.d_rho,
            Quantity([42.658096, 44.824585, 2.999993, -227.499], u.km / u.s),
        )
        assert array_equal(cylindrical.d_phi, Quantity([5, 6, 7, 8], u.mas / u.yr))
        assert array_equal(
            cylindrical.d_z,
            Quantity([-1.2342439, -83.56782, -156.43553, -5.985529], u.km / u.s),
        )

    def test_spherical_to_cylindrical_astropy(
        self, difntl, vector, apydifntl, apyvector
    ):
        """Test Astropy equivalence."""
        cyl = difntl.represent_as(CylindricalDifferential, vector)
        apycyl = apydifntl.represent_as(apyc.CylindricalDifferential, apyvector)
        assert np.allclose(convert(cyl.d_rho, u.Quantity), apycyl.d_rho)
        assert np.allclose(convert(cyl.d_z, u.Quantity), apycyl.d_z)
        assert np.allclose(convert(cyl.d_phi, u.Quantity), apycyl.d_phi)


class TestCylindricalDifferential(Abstract3DVectorDifferentialTest):
    """Test :class:`coordinax.CylindricalDifferential`."""

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

    @pytest.fixture(scope="class")
    def apydifntl(self, difntl: CylindricalDifferential):
        """Return an Astropy differential."""
        return convert(difntl, apyc.CylindricalDifferential)

    @pytest.fixture(scope="class")
    def apyvector(self, vector: CylindricalVector) -> apyc.CylindricalRepresentation:
        """Return an Astropy vector."""
        return convert(vector, apyc.CylindricalRepresentation)

    # ==========================================================================

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cylindrical_to_cartesian1d(self, difntl, vector):
        """Test ``coordinax.represent_as(Cartesian1DVector)``."""
        cart1d = difntl.represent_as(CartesianDifferential1D, vector)

        assert isinstance(cart1d, CartesianDifferential1D)
        assert array_equal(cart1d.d_x, Quantity([1, 2, 3, 4], u.km / u.s))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cylindrical_to_radial(self, difntl, vector):
        """Test ``coordinax.represent_as(RadialVector)``."""
        radial = difntl.represent_as(RadialVector, vector)

        assert isinstance(radial, RadialVector)
        assert array_equal(radial.d_r, Quantity([1, 2, 3, 4], u.km / u.s))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cylindrical_to_cartesian2d(self, difntl, vector):
        """Test ``coordinax.represent_as(Cartesian2DVector)``."""
        cart2d = difntl.represent_as(CartesianDifferential2D, vector)

        assert isinstance(cart2d, CartesianDifferential2D)
        assert array_equal(cart2d.d_x, Quantity([1, 2, 3, 4], u.km / u.s))
        assert array_equal(cart2d.d_y, Quantity([5, 6, 7, 8], u.km / u.s))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cylindrical_to_polar(self, difntl, vector):
        """Test ``coordinax.represent_as(PolarVector)``."""
        polar = difntl.represent_as(PolarVector, vector)

        assert isinstance(polar, PolarVector)
        assert array_equal(polar.d_r, Quantity([1, 2, 3, 4], u.km / u.s))
        assert array_equal(polar.d_phi, Quantity([5, 6, 7, 8], u.mas / u.yr))

    def test_cylindrical_to_cartesian3d(self, difntl, vector, apydifntl, apyvector):
        """Test ``coordinax.represent_as(Cartesian3DVector)``."""
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

        apycart3 = apydifntl.represent_as(apyc.CartesianDifferential, apyvector)
        assert np.allclose(convert(cart3d.d_x, u.Quantity), apycart3.d_x)
        assert np.allclose(convert(cart3d.d_y, u.Quantity), apycart3.d_y)
        assert np.allclose(convert(cart3d.d_z, u.Quantity), apycart3.d_z)

    def test_cylindrical_to_spherical(self, difntl, vector):
        """Test ``coordinax.represent_as(SphericalDifferential)``."""
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

    def test_cylindrical_to_spherical_astropy(
        self, difntl, vector, apydifntl, apyvector
    ):
        """Test Astropy equivalence."""
        sph = difntl.represent_as(SphericalDifferential, vector)
        apysph = apydifntl.represent_as(apyc.PhysicsSphericalDifferential, apyvector)
        assert np.allclose(convert(sph.d_r, u.Quantity), apysph.d_r)
        with pytest.raises(AssertionError):
            assert np.allclose(convert(sph.d_theta, u.Quantity), apysph.d_theta)
        assert np.allclose(convert(sph.d_phi, u.Quantity), apysph.d_phi)

    def test_cylindrical_to_cylindrical(self, difntl, vector):
        """Test ``coordinax.represent_as(CylindricalDifferential)``."""
        # Jit can copy
        newvec = difntl.represent_as(CylindricalDifferential, vector)
        assert newvec == difntl

        # The normal `represent_as` method should return the same object
        newvec = represent_as(difntl, CylindricalDifferential, vector)
        assert newvec is difntl

    def test_cylindrical_to_cylindrical(self, difntl, vector, apydifntl, apyvector):
        """Test Astropy equivalence."""
        cyl = difntl.represent_as(CylindricalDifferential, vector)
        apycyl = apydifntl.represent_as(apyc.CylindricalDifferential, apyvector)
        assert np.allclose(convert(cyl.d_rho, u.Quantity), apycyl.d_rho)
        assert np.allclose(convert(cyl.d_z, u.Quantity), apycyl.d_z)
        assert np.allclose(convert(cyl.d_phi, u.Quantity), apycyl.d_phi)
