"""Test :mod:`coordinax._builtin`."""

import astropy.coordinates as apyc
import numpy as np
import pytest
from astropy.coordinates.tests.test_representation import representation_equal
from astropy.units import Quantity as APYQuantity
from plum import convert

import quaxed.array_api as xp
import quaxed.numpy as qnp
from unxt import Quantity

import coordinax as cx
from .test_base import AbstractVectorDifferentialTest, AbstractVectorTest


class Abstract3DVectorTest(AbstractVectorTest):
    """Test :class:`coordinax.Abstract3DVector`."""

    # ==========================================================================
    # Unary operations

    def test_neg_compare_apy(
        self, vector: cx.AbstractVector, apyvector: apyc.BaseRepresentation
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
    def vector(self) -> cx.AbstractVector:
        """Return a vector."""
        return cx.Cartesian3DVector(
            x=Quantity([1, 2, 3, 4], "kpc"),
            y=Quantity([5, 6, 7, 8], "kpc"),
            z=Quantity([9, 10, 11, 12], "kpc"),
        )

    @pytest.fixture(scope="class")
    def apyvector(self, vector: cx.AbstractVector) -> apyc.CartesianRepresentation:
        """Return an Astropy vector."""
        return convert(vector, apyc.CartesianRepresentation)

    # ==========================================================================
    # represent_as

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cartesian3d_to_cartesian1d(self, vector):
        """Test ``coordinax.represent_as(Cartesian1DVector)``."""
        cart1d = vector.represent_as(cx.Cartesian1DVector)

        assert isinstance(cart1d, cx.Cartesian1DVector)
        assert qnp.array_equal(cart1d.x, Quantity([1, 2, 3, 4], "kpc"))

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cartesian3d_to_radial(self, vector):
        """Test ``coordinax.represent_as(RadialVector)``."""
        radial = vector.represent_as(cx.RadialVector)

        assert isinstance(radial, cx.RadialVector)
        assert qnp.array_equal(
            radial.r, Quantity([10.34408, 11.83216, 13.379088, 14.96663], "kpc")
        )

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cartesian3d_to_cartesian2d(self, vector):
        """Test ``coordinax.represent_as(Cartesian2DVector)``."""
        cart2d = vector.represent_as(
            cx.Cartesian2DVector, y=Quantity([5, 6, 7, 8], "km")
        )

        assert isinstance(cart2d, cx.Cartesian2DVector)
        assert qnp.array_equal(cart2d.x, Quantity([1, 2, 3, 4], "kpc"))
        assert qnp.array_equal(cart2d.y, Quantity([5, 6, 7, 8], "kpc"))

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cartesian3d_to_polar(self, vector):
        """Test ``coordinax.represent_as(PolarVector)``."""
        polar = vector.represent_as(cx.PolarVector, phi=Quantity([0, 1, 2, 3], "rad"))

        assert isinstance(polar, cx.PolarVector)
        assert qnp.array_equal(polar.r, qnp.hypot(vector.x, vector.y))
        assert qnp.array_equal(
            polar.phi, Quantity([1.3734008, 1.2490457, 1.1659045, 1.1071488], "rad")
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
        newvec = vector.represent_as(cx.Cartesian3DVector)
        assert newvec == vector

        # The normal `represent_as` method should return the same object
        newvec = cx.represent_as(vector, cx.Cartesian3DVector)
        assert newvec is vector

    def test_cartesian3d_to_cartesian3d_astropy(self, vector, apyvector):
        """Test Astropy equivalence."""
        newvec = vector.represent_as(cx.Cartesian3DVector)

        assert np.allclose(convert(newvec.x, APYQuantity), apyvector.x)
        assert np.allclose(convert(newvec.y, APYQuantity), apyvector.y)
        assert np.allclose(convert(newvec.z, APYQuantity), apyvector.z)

    def test_cartesian3d_to_spherical(self, vector):
        """Test ``coordinax.represent_as(SphericalVector)``."""
        spherical = vector.represent_as(cx.SphericalVector)

        assert isinstance(spherical, cx.SphericalVector)
        assert qnp.array_equal(
            spherical.r, Quantity([10.34408, 11.83216, 13.379088, 14.96663], "kpc")
        )
        assert qnp.array_equal(
            spherical.phi, Quantity([1.3734008, 1.2490457, 1.1659045, 1.1071488], "rad")
        )
        assert qnp.allclose(
            spherical.theta,
            Quantity([0.51546645, 0.5639427, 0.6055685, 0.64052236], "rad"),
            atol=Quantity(1e-8, "rad"),
        )

    def test_cartesian3d_to_spherical_astropy(self, vector, apyvector):
        """Test Astropy equivalence."""
        sph = vector.represent_as(cx.SphericalVector)

        apysph = apyvector.represent_as(apyc.PhysicsSphericalRepresentation)
        assert np.allclose(convert(sph.r, APYQuantity), apysph.r)
        assert np.allclose(convert(sph.theta, APYQuantity), apysph.theta)
        assert np.allclose(convert(sph.phi, APYQuantity), apysph.phi)

    def test_cartesian3d_to_cylindrical(self, vector):
        """Test ``coordinax.represent_as(CylindricalVector)``."""
        cylindrical = vector.represent_as(cx.CylindricalVector)

        assert isinstance(cylindrical, cx.CylindricalVector)
        assert qnp.array_equal(cylindrical.rho, qnp.hypot(vector.x, vector.y))
        assert qnp.array_equal(
            cylindrical.phi,
            Quantity([1.3734008, 1.2490457, 1.1659045, 1.1071488], "rad"),
        )
        assert qnp.array_equal(cylindrical.z, Quantity([9.0, 10, 11, 12], "kpc"))

    def test_cartesian3d_to_cylindrical_astropy(self, vector, apyvector):
        """Test Astropy equivalence."""
        cyl = vector.represent_as(cx.CylindricalVector)

        apycyl = apyvector.represent_as(apyc.CylindricalRepresentation)
        assert np.allclose(convert(cyl.rho, APYQuantity), apycyl.rho)
        assert np.allclose(convert(cyl.z, APYQuantity), apycyl.z)
        assert np.allclose(convert(cyl.phi, APYQuantity), apycyl.phi)


class TestSphericalVector(Abstract3DVectorTest):
    """Test :class:`coordinax.SphericalVector`."""

    @pytest.fixture(scope="class")
    def vector(self) -> cx.SphericalVector:
        """Return a vector."""
        return cx.SphericalVector(
            r=Quantity([1, 2, 3, 4], "kpc"),
            phi=Quantity([0, 65, 135, 270], "deg"),
            theta=Quantity([0, 36, 142, 180], "deg"),
        )

    @pytest.fixture(scope="class")
    def apyvector(self, vector: cx.AbstractVector):
        """Return an Astropy vector."""
        return convert(vector, apyc.PhysicsSphericalRepresentation)

    # ==========================================================================
    # represent_as

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_spherical_to_cartesian1d(self, vector):
        """Test ``coordinax.represent_as(Cartesian1DVector)``."""
        cart1d = vector.represent_as(cx.Cartesian1DVector)

        assert isinstance(cart1d, cx.Cartesian1DVector)
        assert qnp.allclose(
            cart1d.x,
            Quantity([0, 0.49681753, -1.3060151, -4.1700245e-15], "kpc"),
            atol=Quantity(1e-8, "kpc"),
        )

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_spherical_to_radial(self, vector):
        """Test ``coordinax.represent_as(RadialVector)``."""
        radial = vector.represent_as(cx.RadialVector)

        assert isinstance(radial, cx.RadialVector)
        assert qnp.array_equal(radial.r, Quantity([1, 2, 3, 4], "kpc"))

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_spherical_to_cartesian2d(self, vector):
        """Test ``coordinax.represent_as(Cartesian2DVector)``."""
        cart2d = vector.represent_as(
            cx.Cartesian2DVector, y=Quantity([5, 6, 7, 8], "km")
        )

        assert isinstance(cart2d, cx.Cartesian2DVector)
        assert qnp.array_equal(
            cart2d.x,
            Quantity([0, 0.49681753, -1.3060151, -4.1700245e-15], "kpc"),
        )
        assert qnp.array_equal(
            cart2d.y, Quantity([0.0, 1.0654287, 1.3060151, 3.4969111e-07], "kpc")
        )

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_spherical_to_polar(self, vector):
        """Test ``coordinax.represent_as(PolarVector)``."""
        polar = vector.represent_as(cx.PolarVector, phi=Quantity([0, 1, 2, 3], "rad"))

        assert isinstance(polar, cx.PolarVector)
        assert qnp.array_equal(
            polar.r,
            Quantity([0.0, 1.1755705, 1.8469844, -3.4969111e-07], "kpc"),
        )
        assert qnp.array_equal(polar.phi, Quantity([0.0, 65.0, 135.0, 270.0], "deg"))

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
        cart3d = vector.represent_as(cx.Cartesian3DVector)

        assert isinstance(cart3d, cx.Cartesian3DVector)
        assert qnp.array_equal(
            cart3d.x, Quantity([0, 0.49681753, -1.3060151, -4.1700245e-15], "kpc")
        )
        assert qnp.array_equal(
            cart3d.y, Quantity([0.0, 1.0654287, 1.3060151, 3.4969111e-07], "kpc")
        )
        assert qnp.array_equal(
            cart3d.z, Quantity([1.0, 1.618034, -2.3640323, -4.0], "kpc")
        )

    def test_spherical_to_cartesian3d_astropy(self, vector, apyvector):
        """Test Astropy equivalence."""
        cart3d = vector.represent_as(cx.Cartesian3DVector)

        apycart3 = apyvector.represent_as(apyc.CartesianRepresentation)
        assert np.allclose(convert(cart3d.x, APYQuantity), apycart3.x)
        assert np.allclose(convert(cart3d.y, APYQuantity), apycart3.y)
        assert np.allclose(convert(cart3d.z, APYQuantity), apycart3.z)

    def test_spherical_to_spherical(self, vector):
        """Test ``coordinax.represent_as(SphericalVector)``."""
        # Jit can copy
        newvec = vector.represent_as(cx.SphericalVector)
        assert newvec == vector

        # The normal `represent_as` method should return the same object
        newvec = cx.represent_as(vector, cx.SphericalVector)
        assert newvec is vector

    def test_spherical_to_spherical_astropy(self, vector, apyvector):
        """Test Astropy equivalence."""
        sph = vector.represent_as(cx.SphericalVector)

        apysph = apyvector.represent_as(apyc.PhysicsSphericalRepresentation)
        assert np.allclose(convert(sph.r, APYQuantity), apysph.r)
        assert np.allclose(convert(sph.theta, APYQuantity), apysph.theta)
        assert np.allclose(convert(sph.phi, APYQuantity), apysph.phi)

    def test_spherical_to_cylindrical(self, vector):
        """Test ``coordinax.represent_as(CylindricalVector)``."""
        cylindrical = vector.represent_as(
            cx.CylindricalVector, z=Quantity([9, 10, 11, 12], "m")
        )

        assert isinstance(cylindrical, cx.CylindricalVector)
        assert qnp.array_equal(
            cylindrical.rho,
            Quantity([0.0, 1.1755705, 1.8469844, 3.4969111e-07], "kpc"),
        )
        assert qnp.array_equal(
            cylindrical.phi, Quantity([0.0, 65.0, 135.0, 270.0], "deg")
        )
        assert qnp.array_equal(
            cylindrical.z, Quantity([1.0, 1.618034, -2.3640323, -4.0], "kpc")
        )

    def test_spherical_to_cylindrical_astropy(self, vector, apyvector):
        """Test ``coordinax.represent_as(CylindricalVector)``."""
        cyl = vector.represent_as(
            cx.CylindricalVector, z=Quantity([9, 10, 11, 12], "m")
        )

        apycyl = apyvector.represent_as(apyc.CylindricalRepresentation)
        assert np.allclose(convert(cyl.rho, APYQuantity), apycyl.rho)
        assert np.allclose(convert(cyl.z, APYQuantity), apycyl.z)

        assert np.allclose(convert(cyl.phi[:-1], APYQuantity), apycyl.phi[:-1])
        # There's a 'bug' in Astropy where at the origin phi is always 90, or at
        # least doesn't keep its value.
        with pytest.raises(AssertionError):  # TODO: Fix this
            assert np.allclose(convert(cyl.phi[-1], APYQuantity), apycyl.phi[-1])


class TestCylindricalVector(Abstract3DVectorTest):
    """Test :class:`coordinax.CylindricalVector`."""

    @pytest.fixture(scope="class")
    def vector(self) -> cx.AbstractVector:
        """Return a vector."""
        return cx.CylindricalVector(
            rho=Quantity([1, 2, 3, 4], "kpc"),
            phi=Quantity([0, 1, 2, 3], "rad"),
            z=Quantity([9, 10, 11, 12], "m"),
        )

    @pytest.fixture(scope="class")
    def apyvector(self, vector: cx.AbstractVector):
        """Return an Astropy vector."""
        return convert(vector, apyc.CylindricalRepresentation)

    # ==========================================================================
    # represent_as

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cylindrical_to_cartesian1d(self, vector):
        """Test ``coordinax.represent_as(Cartesian1DVector)``."""
        cart1d = vector.represent_as(cx.Cartesian1DVector)

        assert isinstance(cart1d, cx.Cartesian1DVector)
        assert qnp.allclose(
            cart1d.x,
            Quantity([1.0, 1.0806047, -1.2484405, -3.95997], "kpc"),
            atol=Quantity(1e-8, "kpc"),
        )

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cylindrical_to_radial(self, vector):
        """Test ``coordinax.represent_as(RadialVector)``."""
        radial = vector.represent_as(cx.RadialVector)

        assert isinstance(radial, cx.RadialVector)
        assert qnp.array_equal(radial.r, Quantity([1, 2, 3, 4], "kpc"))

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cylindrical_to_cartesian2d(self, vector):
        """Test ``coordinax.represent_as(Cartesian2DVector)``."""
        cart2d = vector.represent_as(cx.Cartesian2DVector)

        assert isinstance(cart2d, cx.Cartesian2DVector)
        assert qnp.array_equal(
            cart2d.x, Quantity([1.0, 1.0806046, -1.2484405, -3.95997], "kpc")
        )
        assert qnp.array_equal(
            cart2d.y, Quantity([0.0, 1.6829419, 2.7278922, 0.56448], "kpc")
        )

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cylindrical_to_polar(self, vector):
        """Test ``coordinax.represent_as(PolarVector)``."""
        polar = vector.represent_as(cx.PolarVector)

        assert isinstance(polar, cx.PolarVector)
        assert qnp.array_equal(polar.r, Quantity([1, 2, 3, 4], "kpc"))
        assert qnp.array_equal(polar.phi, Quantity([0, 1, 2, 3], "rad"))

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
        cart3d = vector.represent_as(cx.Cartesian3DVector)

        assert isinstance(cart3d, cx.Cartesian3DVector)
        assert qnp.array_equal(
            cart3d.x, Quantity([1.0, 1.0806046, -1.2484405, -3.95997], "kpc")
        )
        assert qnp.array_equal(
            cart3d.y, Quantity([0.0, 1.6829419, 2.7278922, 0.56448], "kpc")
        )
        assert qnp.array_equal(cart3d.z, vector.z)

    def test_cylindrical_to_cartesian3d_astropy(self, vector, apyvector):
        """Test Astropy equivalence."""
        cart3d = vector.represent_as(cx.Cartesian3DVector)

        apycart3 = apyvector.represent_as(apyc.CartesianRepresentation)
        assert np.allclose(convert(cart3d.x, APYQuantity), apycart3.x)
        assert np.allclose(convert(cart3d.y, APYQuantity), apycart3.y)
        assert np.allclose(convert(cart3d.z, APYQuantity), apycart3.z)

    def test_cylindrical_to_spherical(self, vector):
        """Test ``coordinax.represent_as(SphericalVector)``."""
        spherical = vector.represent_as(cx.SphericalVector)

        assert isinstance(spherical, cx.SphericalVector)
        assert qnp.array_equal(spherical.r, Quantity([1, 2, 3, 4], "kpc"))
        assert qnp.array_equal(spherical.phi, Quantity([0, 1, 2, 3], "rad"))
        assert qnp.array_equal(spherical.theta, Quantity(xp.full(4, xp.pi / 2), "rad"))

    def test_cylindrical_to_spherical_astropy(self, vector, apyvector):
        """Test Astropy equivalence."""
        sph = vector.represent_as(cx.SphericalVector)
        apysph = apyvector.represent_as(apyc.PhysicsSphericalRepresentation)
        assert np.allclose(convert(sph.r, APYQuantity), apysph.r)
        assert np.allclose(convert(sph.theta, APYQuantity), apysph.theta)
        assert np.allclose(convert(sph.phi, APYQuantity), apysph.phi)

    def test_cylindrical_to_cylindrical(self, vector):
        """Test ``coordinax.represent_as(CylindricalVector)``."""
        # Jit can copy
        newvec = vector.represent_as(cx.CylindricalVector)
        assert newvec == vector

        # The normal `represent_as` method should return the same object
        newvec = cx.represent_as(vector, cx.CylindricalVector)
        assert newvec is vector

    def test_cylindrical_to_cylindrical_astropy(self, vector, apyvector):
        """Test Astropy equivalence."""
        cyl = vector.represent_as(cx.CylindricalVector)

        apycyl = apyvector.represent_as(apyc.CylindricalRepresentation)
        assert np.allclose(convert(cyl.rho, APYQuantity), apycyl.rho)
        assert np.allclose(convert(cyl.z, APYQuantity), apycyl.z)
        assert np.allclose(convert(cyl.phi, APYQuantity), apycyl.phi)


class Abstract3DVectorDifferentialTest(AbstractVectorDifferentialTest):
    """Test :class:`coordinax.Abstract2DVectorDifferential`."""

    # ==========================================================================
    # Unary operations

    def test_neg_compare_apy(
        self, difntl: cx.AbstractVector, apydifntl: apyc.BaseRepresentation
    ):
        """Test negation."""
        assert all(representation_equal(convert(-difntl, type(apydifntl)), -apydifntl))


class TestCartesianDifferential3D(Abstract3DVectorDifferentialTest):
    """Test :class:`coordinax.CartesianDifferential3D`."""

    @pytest.fixture(scope="class")
    def difntl(self) -> cx.CartesianDifferential3D:
        """Return a differential."""
        return cx.CartesianDifferential3D(
            d_x=Quantity([5, 6, 7, 8], "km/s"),
            d_y=Quantity([9, 10, 11, 12], "km/s"),
            d_z=Quantity([13, 14, 15, 16], "km/s"),
        )

    @pytest.fixture(scope="class")
    def vector(self) -> cx.Cartesian3DVector:
        """Return a vector."""
        return cx.Cartesian3DVector(
            x=Quantity([1, 2, 3, 4], "kpc"),
            y=Quantity([5, 6, 7, 8], "kpc"),
            z=Quantity([9, 10, 11, 12], "kpc"),
        )

    @pytest.fixture(scope="class")
    def apydifntl(self, difntl: cx.CartesianDifferential3D):
        """Return an Astropy differential."""
        return convert(difntl, apyc.CartesianDifferential)

    @pytest.fixture(scope="class")
    def apyvector(self, vector: cx.Cartesian3DVector):
        """Return an Astropy vector."""
        return convert(vector, apyc.CartesianRepresentation)

    # ==========================================================================

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian3d_to_cartesian1d(self, difntl, vector):
        """Test ``coordinax.represent_as(Cartesian1DVector)``."""
        cart1d = difntl.represent_as(cx.CartesianDifferential1D, vector)

        assert isinstance(cart1d, cx.CartesianDifferential1D)
        assert qnp.array_equal(cart1d.d_x, Quantity([1, 2, 3, 4], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian3d_to_radial(self, difntl, vector):
        """Test ``coordinax.represent_as(RadialVector)``."""
        radial = difntl.represent_as(cx.RadialVector, vector)

        assert isinstance(radial, cx.RadialVector)
        assert qnp.array_equal(radial.d_r, Quantity([1, 2, 3, 4], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian3d_to_cartesian2d(self, difntl, vector):
        """Test ``coordinax.represent_as(Cartesian2DVector)``."""
        cart2d = difntl.represent_as(cx.CartesianDifferential2D, vector)

        assert isinstance(cart2d, cx.CartesianDifferential2D)
        assert qnp.array_equal(cart2d.d_x, Quantity([1, 2, 3, 4], "km/s"))
        assert qnp.array_equal(cart2d.d_y, Quantity([5, 6, 7, 8], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian3d_to_polar(self, difntl, vector):
        """Test ``coordinax.represent_as(PolarVector)``."""
        polar = difntl.represent_as(cx.PolarVector, vector)

        assert isinstance(polar, cx.PolarVector)
        assert qnp.array_equal(polar.d_r, Quantity([1, 2, 3, 4], "km/s"))
        assert qnp.array_equal(polar.d_phi, Quantity([5, 6, 7, 8], "mas/yr"))

    def test_cartesian3d_to_cartesian3d(self, difntl, vector):
        """Test ``coordinax.represent_as(Cartesian3DVector)``."""
        # Jit can copy
        newvec = difntl.represent_as(cx.CartesianDifferential3D, vector)
        assert newvec == difntl

        # The normal `represent_as` method should return the same object
        newvec = cx.represent_as(difntl, cx.CartesianDifferential3D, vector)
        assert newvec is difntl

    def test_cartesian3d_to_cartesian3d_astropy(
        self, difntl, vector, apydifntl, apyvector
    ):
        """Test Astropy equivalence."""
        cart3 = difntl.represent_as(cx.CartesianDifferential3D, vector)

        apycart3 = apydifntl.represent_as(apyc.CartesianDifferential, apyvector)
        assert np.allclose(convert(cart3.d_x, APYQuantity), apycart3.d_x)
        assert np.allclose(convert(cart3.d_y, APYQuantity), apycart3.d_y)
        assert np.allclose(convert(cart3.d_z, APYQuantity), apycart3.d_z)

    def test_cartesian3d_to_spherical(self, difntl, vector):
        """Test ``coordinax.represent_as(SphericalDifferential)``."""
        spherical = difntl.represent_as(cx.SphericalDifferential, vector)

        assert isinstance(spherical, cx.SphericalDifferential)
        assert qnp.allclose(
            spherical.d_r,
            Quantity([16.1445, 17.917269, 19.657543, 21.380898], "km/s"),
            atol=Quantity(1e-8, "km/s"),
        )
        assert qnp.allclose(
            spherical.d_phi,
            Quantity(
                [-0.61538464, -0.40000004, -0.275862, -0.19999999], "km rad / (kpc s)"
            ),
            atol=Quantity(1e-8, "mas/Myr"),
        )
        assert qnp.allclose(
            spherical.d_theta,
            Quantity(
                [0.2052807, 0.1807012, 0.15257944, 0.12777519], "km rad / (kpc s)"
            ),
            atol=Quantity(1e-8, "mas/Myr"),
        )

    def test_cartesian3d_to_spherical_astropy(
        self, difntl, vector, apydifntl, apyvector
    ):
        """Test Astropy equivalence."""
        sph = difntl.represent_as(cx.SphericalDifferential, vector)

        apysph = apydifntl.represent_as(apyc.PhysicsSphericalDifferential, apyvector)
        assert np.allclose(convert(sph.d_r, APYQuantity), apysph.d_r)
        assert np.allclose(convert(sph.d_theta, APYQuantity), apysph.d_theta, atol=1e-9)
        assert np.allclose(convert(sph.d_phi, APYQuantity), apysph.d_phi, atol=1e-7)

    def test_cartesian3d_to_cylindrical(self, difntl, vector):
        """Test ``coordinax.represent_as(CylindricalDifferential)``."""
        cylindrical = difntl.represent_as(cx.CylindricalDifferential, vector)

        assert isinstance(cylindrical, cx.CylindricalDifferential)
        assert qnp.array_equal(
            cylindrical.d_rho,
            Quantity([9.805807, 11.384199, 12.86803, 14.310835], "km/s"),
        )
        assert qnp.allclose(
            cylindrical.d_phi,
            Quantity(
                [-0.61538464, -0.40000004, -0.275862, -0.19999999], "km rad / (kpc s)"
            ),
            atol=Quantity(1e-8, "mas/Myr"),
        )
        assert qnp.array_equal(
            cylindrical.d_z, Quantity([13.0, 14.0, 15.0, 16], "km/s")
        )

    def test_cartesian3d_to_cylindrical_astropy(
        self, difntl, vector, apydifntl, apyvector
    ):
        """Test Astropy equivalence."""
        cyl = difntl.represent_as(cx.CylindricalDifferential, vector)
        apycyl = apydifntl.represent_as(apyc.CylindricalDifferential, apyvector)
        assert np.allclose(convert(cyl.d_rho, APYQuantity), apycyl.d_rho)
        assert np.allclose(convert(cyl.d_z, APYQuantity), apycyl.d_z)
        assert np.allclose(convert(cyl.d_phi, APYQuantity), apycyl.d_phi)


class TestSphericalDifferential(Abstract3DVectorDifferentialTest):
    """Test :class:`coordinax.SphericalDifferential`."""

    @pytest.fixture(scope="class")
    def difntl(self) -> cx.SphericalDifferential:
        """Return a differential."""
        return cx.SphericalDifferential(
            d_r=Quantity([5, 6, 7, 8], "km/s"),
            d_phi=Quantity([9, 10, 11, 12], "mas/yr"),
            d_theta=Quantity([13, 14, 15, 16], "mas/yr"),
        )

    @pytest.fixture(scope="class")
    def vector(self) -> cx.SphericalVector:
        """Return a vector."""
        return cx.SphericalVector(
            r=Quantity([1, 2, 3, 4], "kpc"),
            phi=Quantity([0, 42, 160, 270], "deg"),
            theta=Quantity([3, 63, 90, 179.5], "deg"),
        )

    @pytest.fixture(scope="class")
    def apydifntl(
        self, difntl: cx.SphericalDifferential
    ) -> apyc.PhysicsSphericalDifferential:
        """Return an Astropy differential."""
        return convert(difntl, apyc.PhysicsSphericalDifferential)

    @pytest.fixture(scope="class")
    def apyvector(
        self, vector: cx.SphericalVector
    ) -> apyc.PhysicsSphericalRepresentation:
        """Return an Astropy vector."""
        return convert(vector, apyc.PhysicsSphericalRepresentation)

    # ==========================================================================

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_spherical_to_cartesian1d(self, difntl, vector):
        """Test ``coordinax.represent_as(Cartesian1DVector)``."""
        cart1d = difntl.represent_as(cx.CartesianDifferential1D, vector)

        assert isinstance(cart1d, cx.CartesianDifferential1D)
        assert qnp.array_equal(cart1d.d_x, Quantity([1, 2, 3, 4], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_spherical_to_radial(self, difntl, vector):
        """Test ``coordinax.represent_as(RadialVector)``."""
        radial = difntl.represent_as(cx.RadialVector, vector)

        assert isinstance(radial, cx.RadialVector)
        assert qnp.array_equal(radial.d_r, Quantity([1, 2, 3, 4], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_spherical_to_cartesian2d(self, difntl, vector):
        """Test ``coordinax.represent_as(Cartesian2DVector)``."""
        cart2d = difntl.represent_as(cx.CartesianDifferential2D, vector)

        assert isinstance(cart2d, cx.CartesianDifferential2D)
        assert qnp.array_equal(cart2d.d_x, Quantity([1, 2, 3, 4], "km/s"))
        assert qnp.array_equal(cart2d.d_y, Quantity([5, 6, 7, 8], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_spherical_to_polar(self, difntl, vector):
        """Test ``coordinax.represent_as(PolarVector)``."""
        polar = difntl.represent_as(cx.PolarVector, vector)

        assert isinstance(polar, cx.PolarVector)
        assert qnp.array_equal(polar.d_r, Quantity([1, 2, 3, 4], "km/s"))
        assert qnp.array_equal(polar.d_phi, Quantity([5, 6, 7, 8], "mas/yr"))

    def test_spherical_to_cartesian3d(self, difntl, vector):
        """Test ``coordinax.represent_as(Cartesian3DVector)``."""
        cart3d = difntl.represent_as(cx.CartesianDifferential3D, vector)

        assert isinstance(cart3d, cx.CartesianDifferential3D)
        assert qnp.allclose(
            cart3d.d_x,
            Quantity([61.803337, -7.770853, -60.081947, 1.985678], "km/s"),
            atol=Quantity(1e-8, "km/s"),
        )
        assert qnp.allclose(
            cart3d.d_y,
            Quantity([2.2328734, 106.6765, -144.60716, 303.30875], "km/s"),
            atol=Quantity(1e-8, "km/s"),
        )
        assert qnp.allclose(
            cart3d.d_z,
            Quantity([1.7678856, -115.542175, -213.32118, -10.647271], "km/s"),
            atol=Quantity(1e-8, "km/s"),
        )

    def test_spherical_to_cartesian3d_astropy(
        self, difntl, vector, apydifntl, apyvector
    ):
        """Test Astropy equivalence."""
        cart3d = difntl.represent_as(cx.CartesianDifferential3D, vector)

        apycart3 = apydifntl.represent_as(apyc.CartesianDifferential, apyvector)
        assert np.allclose(convert(cart3d.d_x, APYQuantity), apycart3.d_x)
        assert np.allclose(convert(cart3d.d_y, APYQuantity), apycart3.d_y)
        assert np.allclose(convert(cart3d.d_z, APYQuantity), apycart3.d_z)

    def test_spherical_to_spherical(self, difntl, vector):
        """Test ``coordinax.represent_as(SphericalDifferential)``."""
        # Jit can copy
        newvec = difntl.represent_as(cx.SphericalDifferential, vector)
        assert newvec == difntl

        # The normal `represent_as` method should return the same object
        newvec = cx.represent_as(difntl, cx.SphericalDifferential, vector)
        assert newvec is difntl

    def test_spherical_to_spherical_astropy(self, difntl, vector, apydifntl, apyvector):
        """Test Astropy equivalence."""
        sph = difntl.represent_as(cx.SphericalDifferential, vector)
        apysph = apydifntl.represent_as(apyc.PhysicsSphericalDifferential, apyvector)
        assert np.allclose(convert(sph.d_r, APYQuantity), apysph.d_r)
        assert np.allclose(convert(sph.d_theta, APYQuantity), apysph.d_theta)
        assert np.allclose(convert(sph.d_phi, APYQuantity), apysph.d_phi)

    def test_spherical_to_cylindrical(self, difntl, vector):
        """Test ``coordinax.represent_as(CylindricalDifferential)``."""
        cylindrical = difntl.represent_as(cx.CylindricalDifferential, vector)

        assert isinstance(cylindrical, cx.CylindricalDifferential)
        assert qnp.allclose(
            cylindrical.d_rho,
            Quantity([61.803337, 65.60564, 6.9999905, -303.30875], "km/s"),
            atol=Quantity(1e-8, "km/s"),
        )
        assert qnp.allclose(
            cylindrical.d_phi,
            Quantity([2444.4805, 2716.0894, 2987.6985, 3259.3074], "deg km / (kpc s)"),
            atol=Quantity(1e-8, "mas/yr"),
        )
        assert qnp.allclose(
            cylindrical.d_z,
            Quantity([1.7678856, -115.542175, -213.32118, -10.647271], "km/s"),
            atol=Quantity(1e-8, "km/s"),
        )

    def test_spherical_to_cylindrical_astropy(
        self, difntl, vector, apydifntl, apyvector
    ):
        """Test Astropy equivalence."""
        cyl = difntl.represent_as(cx.CylindricalDifferential, vector)
        apycyl = apydifntl.represent_as(apyc.CylindricalDifferential, apyvector)
        assert np.allclose(convert(cyl.d_rho, APYQuantity), apycyl.d_rho)
        assert np.allclose(convert(cyl.d_z, APYQuantity), apycyl.d_z)
        assert np.allclose(convert(cyl.d_phi, APYQuantity), apycyl.d_phi)


class TestCylindricalDifferential(Abstract3DVectorDifferentialTest):
    """Test :class:`coordinax.CylindricalDifferential`."""

    @pytest.fixture(scope="class")
    def difntl(self) -> cx.CylindricalDifferential:
        """Return a differential."""
        return cx.CylindricalDifferential(
            d_rho=Quantity([5, 6, 7, 8], "km/s"),
            d_phi=Quantity([9, 10, 11, 12], "mas/yr"),
            d_z=Quantity([13, 14, 15, 16], "km/s"),
        )

    @pytest.fixture(scope="class")
    def vector(self) -> cx.CylindricalVector:
        """Return a vector."""
        return cx.CylindricalVector(
            rho=Quantity([1, 2, 3, 4], "kpc"),
            phi=Quantity([0, 1, 2, 3], "rad"),
            z=Quantity([9, 10, 11, 12], "kpc"),
        )

    @pytest.fixture(scope="class")
    def apydifntl(self, difntl: cx.CylindricalDifferential):
        """Return an Astropy differential."""
        return convert(difntl, apyc.CylindricalDifferential)

    @pytest.fixture(scope="class")
    def apyvector(self, vector: cx.CylindricalVector) -> apyc.CylindricalRepresentation:
        """Return an Astropy vector."""
        return convert(vector, apyc.CylindricalRepresentation)

    # ==========================================================================

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cylindrical_to_cartesian1d(self, difntl, vector):
        """Test ``coordinax.represent_as(Cartesian1DVector)``."""
        cart1d = difntl.represent_as(cx.CartesianDifferential1D, vector)

        assert isinstance(cart1d, cx.CartesianDifferential1D)
        assert qnp.array_equal(cart1d.d_x, Quantity([1, 2, 3, 4], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cylindrical_to_radial(self, difntl, vector):
        """Test ``coordinax.represent_as(RadialVector)``."""
        radial = difntl.represent_as(cx.RadialVector, vector)

        assert isinstance(radial, cx.RadialVector)
        assert qnp.array_equal(radial.d_r, Quantity([1, 2, 3, 4], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cylindrical_to_cartesian2d(self, difntl, vector):
        """Test ``coordinax.represent_as(Cartesian2DVector)``."""
        cart2d = difntl.represent_as(cx.CartesianDifferential2D, vector)

        assert isinstance(cart2d, cx.CartesianDifferential2D)
        assert qnp.array_equal(cart2d.d_x, Quantity([1, 2, 3, 4], "km/s"))
        assert qnp.array_equal(cart2d.d_y, Quantity([5, 6, 7, 8], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cylindrical_to_polar(self, difntl, vector):
        """Test ``coordinax.represent_as(PolarVector)``."""
        polar = difntl.represent_as(cx.PolarVector, vector)

        assert isinstance(polar, cx.PolarVector)
        assert qnp.array_equal(polar.d_r, Quantity([1, 2, 3, 4], "km/s"))
        assert qnp.array_equal(polar.d_phi, Quantity([5, 6, 7, 8], "mas/yr"))

    def test_cylindrical_to_cartesian3d(self, difntl, vector, apydifntl, apyvector):
        """Test ``coordinax.represent_as(Cartesian3DVector)``."""
        cart3d = difntl.represent_as(cx.CartesianDifferential3D, vector)

        assert isinstance(cart3d, cx.CartesianDifferential3D)
        assert qnp.array_equal(
            cart3d.d_x, Quantity([5.0, -76.537544, -145.15944, -40.03075], "km/s")
        )
        assert qnp.array_equal(
            cart3d.d_y,
            Quantity([42.664234, 56.274563, -58.73506, -224.13647], "km/s"),
        )
        assert qnp.array_equal(cart3d.d_z, Quantity([13.0, 14.0, 15.0, 16.0], "km/s"))

        apycart3 = apydifntl.represent_as(apyc.CartesianDifferential, apyvector)
        assert np.allclose(convert(cart3d.d_x, APYQuantity), apycart3.d_x)
        assert np.allclose(convert(cart3d.d_y, APYQuantity), apycart3.d_y)
        assert np.allclose(convert(cart3d.d_z, APYQuantity), apycart3.d_z)

    def test_cylindrical_to_spherical(self, difntl, vector):
        """Test ``coordinax.represent_as(SphericalDifferential)``."""
        dsph = difntl.represent_as(cx.SphericalDifferential, vector)

        assert isinstance(dsph, cx.SphericalDifferential)
        assert qnp.array_equal(
            dsph.d_r,
            Quantity([13.472646, 14.904826, 16.313278, 17.708754], "km/s"),
        )
        assert qnp.array_equal(
            dsph.d_phi,
            Quantity([42.664234, 47.404705, 52.145176, 56.885643], "km rad / (kpc s)"),
        )
        assert qnp.allclose(
            dsph.d_theta,
            Quantity(
                [0.3902412, 0.30769292, 0.24615361, 0.19999981], "km rad / (kpc s)"
            ),
            atol=Quantity(5e-7, "km rad / (kpc s)"),
        )

    def test_cylindrical_to_spherical_astropy(
        self, difntl, vector, apydifntl, apyvector
    ):
        """Test Astropy equivalence."""
        sph = difntl.represent_as(cx.SphericalDifferential, vector)
        apysph = apydifntl.represent_as(apyc.PhysicsSphericalDifferential, apyvector)
        assert np.allclose(convert(sph.d_r, APYQuantity), apysph.d_r)
        assert np.allclose(convert(sph.d_theta, APYQuantity), apysph.d_theta)
        assert np.allclose(convert(sph.d_phi, APYQuantity), apysph.d_phi)

    def test_cylindrical_to_cylindrical(self, difntl, vector):
        """Test ``coordinax.represent_as(CylindricalDifferential)``."""
        # Jit can copy
        newvec = difntl.represent_as(cx.CylindricalDifferential, vector)
        assert newvec == difntl

        # The normal `represent_as` method should return the same object
        newvec = cx.represent_as(difntl, cx.CylindricalDifferential, vector)
        assert newvec is difntl

    def test_cylindrical_to_cylindrical(self, difntl, vector, apydifntl, apyvector):
        """Test Astropy equivalence."""
        cyl = difntl.represent_as(cx.CylindricalDifferential, vector)
        apycyl = apydifntl.represent_as(apyc.CylindricalDifferential, apyvector)
        assert np.allclose(convert(cyl.d_rho, APYQuantity), apycyl.d_rho)
        assert np.allclose(convert(cyl.d_z, APYQuantity), apycyl.d_z)
        assert np.allclose(convert(cyl.d_phi, APYQuantity), apycyl.d_phi)
