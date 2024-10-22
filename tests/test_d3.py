"""Test :mod:`coordinax._builtin`."""

import astropy.coordinates as apyc
import numpy as np
import pytest
from astropy.coordinates.tests.test_representation import representation_equal
from astropy.units import Quantity as APYQuantity
from plum import convert

import quaxed.numpy as jnp
from unxt import Quantity

import coordinax as cx
from .test_base import AbstractPosTest, AbstractVelTest


class AbstractPos3DTest(AbstractPosTest):
    """Test :class:`coordinax.AbstractPos3D`."""

    # ==========================================================================
    # Unary operations

    def test_neg_compare_apy(
        self, vector: cx.AbstractPos, apyvector: apyc.BaseRepresentation
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
        assert np.allclose(cart.x, apycart.x, atol=1e-6)  # TODO: better agreement
        assert np.allclose(cart.y, apycart.y, atol=1e-5)  # TODO: better agreement
        assert np.allclose(cart.z, apycart.z, atol=5e-7)

        # # Try finding the poles
        # if hasattr(vector, "theta"):
        #     sel = (vector.theta.to_units_value("deg") != 0) & (
        #         vector.theta.to_units_value("deg") != 180
        #     )
        # else:
        #     sel = slice(None)
        # vecsel = convert(-vector[sel], type(apyvector))
        # apyvecsel = -apyvector[sel]
        # for c in vecsel.components:
        #     unit_ = units(getattr(apyvecsel, c))
        #     assert np.allclose(
        #         ustrip(unit_, getattr(vecsel, c)),
        #         ustrip(unit_, getattr(apyvecsel, c)),
        #         atol=5e-7,
        #     )


class TestCartesianPos3D(AbstractPos3DTest):
    """Test :class:`coordinax.CartesianPos3D`."""

    @pytest.fixture(scope="class")
    def vector(self) -> cx.AbstractPos:
        """Return a vector."""
        return cx.CartesianPos3D(
            x=Quantity([1, 2, 3, 4], "kpc"),
            y=Quantity([5, 6, 7, 8], "kpc"),
            z=Quantity([9, 10, 11, 12], "kpc"),
        )

    @pytest.fixture(scope="class")
    def apyvector(self, vector: cx.AbstractPos) -> apyc.CartesianRepresentation:
        """Return an Astropy vector."""
        return convert(vector, apyc.CartesianRepresentation)

    # ==========================================================================
    # represent_as

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cartesian3d_to_cartesian1d(self, vector):
        """Test ``coordinax.represent_as(CartesianPos1D)``."""
        cart1d = vector.represent_as(cx.CartesianPos1D)

        assert isinstance(cart1d, cx.CartesianPos1D)
        assert jnp.array_equal(cart1d.x, Quantity([1, 2, 3, 4], "kpc"))

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cartesian3d_to_radial(self, vector):
        """Test ``coordinax.represent_as(RadialPos)``."""
        radial = vector.represent_as(cx.RadialPos)

        assert isinstance(radial, cx.RadialPos)
        assert jnp.array_equal(
            radial.r, Quantity([10.34408, 11.83216, 13.379088, 14.96663], "kpc")
        )

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cartesian3d_to_cartesian2d(self, vector):
        """Test ``coordinax.represent_as(CartesianPos2D)``."""
        cart2d = vector.represent_as(cx.CartesianPos2D, y=Quantity([5, 6, 7, 8], "km"))

        assert isinstance(cart2d, cx.CartesianPos2D)
        assert jnp.array_equal(cart2d.x, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cart2d.y, Quantity([5, 6, 7, 8], "kpc"))

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cartesian3d_to_polar(self, vector):
        """Test ``coordinax.represent_as(PolarPos)``."""
        polar = vector.represent_as(cx.PolarPos, phi=Quantity([0, 1, 2, 3], "rad"))

        assert isinstance(polar, cx.PolarPos)
        assert jnp.array_equal(polar.r, jnp.hypot(vector.x, vector.y))
        assert jnp.array_equal(
            polar.phi, Quantity([1.3734008, 1.2490457, 1.1659045, 1.1071488], "rad")
        )

    def test_cartesian3d_to_cartesian3d(self, vector):
        """Test ``coordinax.represent_as(CartesianPos3D)``."""
        # Jit can copy
        newvec = vector.represent_as(cx.CartesianPos3D)
        assert jnp.array_equal(newvec, vector)

        # The normal `represent_as` method should return the same object
        newvec = cx.represent_as(vector, cx.CartesianPos3D)
        assert newvec is vector

    def test_cartesian3d_to_cartesian3d_astropy(self, vector, apyvector):
        """Test Astropy equivalence."""
        newvec = vector.represent_as(cx.CartesianPos3D)

        assert np.allclose(convert(newvec.x, APYQuantity), apyvector.x)
        assert np.allclose(convert(newvec.y, APYQuantity), apyvector.y)
        assert np.allclose(convert(newvec.z, APYQuantity), apyvector.z)

    def test_cartesian3d_to_spherical(self, vector):
        """Test ``coordinax.represent_as(SphericalPos)``."""
        spherical = vector.represent_as(cx.SphericalPos)

        assert isinstance(spherical, cx.SphericalPos)
        assert jnp.array_equal(
            spherical.r, Quantity([10.34408, 11.83216, 13.379088, 14.96663], "kpc")
        )
        assert jnp.array_equal(
            spherical.phi, Quantity([1.3734008, 1.2490457, 1.1659045, 1.1071488], "rad")
        )
        assert jnp.allclose(
            spherical.theta,
            Quantity([0.51546645, 0.5639427, 0.6055685, 0.64052236], "rad"),
            atol=Quantity(1e-8, "rad"),
        )

    def test_cartesian3d_to_spherical_astropy(self, vector, apyvector):
        """Test Astropy equivalence."""
        sph = vector.represent_as(cx.SphericalPos)

        apysph = apyvector.represent_as(apyc.PhysicsSphericalRepresentation)
        assert np.allclose(convert(sph.r, APYQuantity), apysph.r)
        assert np.allclose(convert(sph.theta, APYQuantity), apysph.theta)
        assert np.allclose(convert(sph.phi, APYQuantity), apysph.phi)

    def test_cartesian3d_to_cylindrical(self, vector):
        """Test ``coordinax.represent_as(CylindricalPos)``."""
        cylindrical = vector.represent_as(cx.CylindricalPos)

        assert isinstance(cylindrical, cx.CylindricalPos)
        assert jnp.array_equal(cylindrical.rho, jnp.hypot(vector.x, vector.y))
        assert jnp.array_equal(
            cylindrical.phi,
            Quantity([1.3734008, 1.2490457, 1.1659045, 1.1071488], "rad"),
        )
        assert jnp.array_equal(cylindrical.z, Quantity([9.0, 10, 11, 12], "kpc"))

    def test_cartesian3d_to_cylindrical_astropy(self, vector, apyvector):
        """Test Astropy equivalence."""
        cyl = vector.represent_as(cx.CylindricalPos)

        apycyl = apyvector.represent_as(apyc.CylindricalRepresentation)
        assert np.allclose(convert(cyl.rho, APYQuantity), apycyl.rho)
        assert np.allclose(convert(cyl.phi, APYQuantity), apycyl.phi)
        assert np.allclose(convert(cyl.z, APYQuantity), apycyl.z)


class TestCylindricalPos(AbstractPos3DTest):
    """Test :class:`coordinax.CylindricalPos`."""

    @pytest.fixture(scope="class")
    def vector(self) -> cx.AbstractPos:
        """Return a vector."""
        return cx.CylindricalPos(
            rho=Quantity([1, 2, 3, 4], "kpc"),
            phi=Quantity([0, 1, 2, 3], "rad"),
            z=Quantity([9, 10, 11, 12], "m"),
        )

    @pytest.fixture(scope="class")
    def apyvector(self, vector: cx.AbstractPos):
        """Return an Astropy vector."""
        return convert(vector, apyc.CylindricalRepresentation)

    # ==========================================================================
    # represent_as

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cylindrical_to_cartesian1d(self, vector):
        """Test ``coordinax.represent_as(CartesianPos1D)``."""
        cart1d = vector.represent_as(cx.CartesianPos1D)

        assert isinstance(cart1d, cx.CartesianPos1D)
        assert jnp.allclose(
            cart1d.x,
            Quantity([1.0, 1.0806047, -1.2484405, -3.95997], "kpc"),
            atol=Quantity(1e-8, "kpc"),
        )

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cylindrical_to_radial(self, vector):
        """Test ``coordinax.represent_as(RadialPos)``."""
        radial = vector.represent_as(cx.RadialPos)

        assert isinstance(radial, cx.RadialPos)
        assert jnp.array_equal(radial.r, Quantity([1, 2, 3, 4], "kpc"))

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cylindrical_to_cartesian2d(self, vector):
        """Test ``coordinax.represent_as(CartesianPos2D)``."""
        cart2d = vector.represent_as(cx.CartesianPos2D)

        assert isinstance(cart2d, cx.CartesianPos2D)
        assert jnp.array_equal(
            cart2d.x, Quantity([1.0, 1.0806046, -1.2484405, -3.95997], "kpc")
        )
        assert jnp.array_equal(
            cart2d.y, Quantity([0.0, 1.6829419, 2.7278922, 0.56448], "kpc")
        )

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cylindrical_to_polar(self, vector):
        """Test ``coordinax.represent_as(PolarPos)``."""
        polar = vector.represent_as(cx.PolarPos)

        assert isinstance(polar, cx.PolarPos)
        assert jnp.array_equal(polar.r, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(polar.phi, Quantity([0, 1, 2, 3], "rad"))

    def test_cylindrical_to_cartesian3d(self, vector):
        """Test ``coordinax.represent_as(CartesianPos3D)``."""
        cart3d = vector.represent_as(cx.CartesianPos3D)

        assert isinstance(cart3d, cx.CartesianPos3D)
        assert jnp.array_equal(
            cart3d.x, Quantity([1.0, 1.0806046, -1.2484405, -3.95997], "kpc")
        )
        assert jnp.array_equal(
            cart3d.y, Quantity([0.0, 1.6829419, 2.7278922, 0.56448], "kpc")
        )
        assert jnp.array_equal(cart3d.z, vector.z)

    def test_cylindrical_to_cartesian3d_astropy(self, vector, apyvector):
        """Test Astropy equivalence."""
        cart3d = vector.represent_as(cx.CartesianPos3D)

        apycart3 = apyvector.represent_as(apyc.CartesianRepresentation)
        assert np.allclose(convert(cart3d.x, APYQuantity), apycart3.x)
        assert np.allclose(convert(cart3d.y, APYQuantity), apycart3.y)
        assert np.allclose(convert(cart3d.z, APYQuantity), apycart3.z)

    def test_cylindrical_to_spherical(self, vector):
        """Test ``coordinax.represent_as(SphericalPos)``."""
        spherical = vector.represent_as(cx.SphericalPos)

        assert isinstance(spherical, cx.SphericalPos)
        assert jnp.array_equal(spherical.r, Quantity([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(
            spherical.theta, Quantity(jnp.full(4, jnp.pi / 2), "rad")
        )
        assert jnp.array_equal(spherical.phi, Quantity([0, 1, 2, 3], "rad"))

    def test_cylindrical_to_spherical_astropy(self, vector, apyvector):
        """Test Astropy equivalence."""
        sph = vector.represent_as(cx.SphericalPos)
        apysph = apyvector.represent_as(apyc.PhysicsSphericalRepresentation)
        assert np.allclose(convert(sph.r, APYQuantity), apysph.r)
        assert np.allclose(convert(sph.theta, APYQuantity), apysph.theta)
        assert np.allclose(convert(sph.phi, APYQuantity), apysph.phi)

    def test_cylindrical_to_cylindrical(self, vector):
        """Test ``coordinax.represent_as(CylindricalPos)``."""
        # Jit can copy
        newvec = vector.represent_as(cx.CylindricalPos)
        assert jnp.array_equal(newvec, vector)

        # The normal `represent_as` method should return the same object
        newvec = cx.represent_as(vector, cx.CylindricalPos)
        assert newvec is vector

    def test_cylindrical_to_cylindrical_astropy(self, vector, apyvector):
        """Test Astropy equivalence."""
        cyl = vector.represent_as(cx.CylindricalPos)

        apycyl = apyvector.represent_as(apyc.CylindricalRepresentation)
        assert np.allclose(convert(cyl.rho, APYQuantity), apycyl.rho)
        assert np.allclose(convert(cyl.phi, APYQuantity), apycyl.phi)
        assert np.allclose(convert(cyl.z, APYQuantity), apycyl.z)


class TestSphericalPos(AbstractPos3DTest):
    """Test :class:`coordinax.SphericalPos`."""

    @pytest.fixture(scope="class")
    def vector(self) -> cx.SphericalPos:
        """Return a vector."""
        return cx.SphericalPos(
            r=Quantity([1, 2, 3, 4], "kpc"),
            theta=Quantity([1, 36, 142, 180 - 1e-4], "deg"),
            phi=Quantity([0, 65, 135, 270], "deg"),
        )

    @pytest.fixture(scope="class")
    def apyvector(self, vector: cx.AbstractPos):
        """Return an Astropy vector."""
        return convert(vector, apyc.PhysicsSphericalRepresentation)

    # ==========================================================================
    # represent_as

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_spherical_to_cartesian1d(self, vector):
        """Test ``coordinax.represent_as(CartesianPos1D)``."""
        cart1d = vector.represent_as(cx.CartesianPos1D)

        assert isinstance(cart1d, cx.CartesianPos1D)
        assert jnp.allclose(
            cart1d.x,
            Quantity(
                [1.7452406e-02, 4.9681753e-01, -1.3060151e00, 8.6809595e-14], "kpc"
            ),
            atol=Quantity(1e-8, "kpc"),
        )

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_spherical_to_radial(self, vector):
        """Test ``coordinax.represent_as(RadialPos)``."""
        radial = vector.represent_as(cx.RadialPos)

        assert isinstance(radial, cx.RadialPos)
        assert jnp.array_equal(radial.r, Quantity([1, 2, 3, 4], "kpc"))

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_spherical_to_cartesian2d(self, vector):
        """Test ``coordinax.represent_as(CartesianPos2D)``."""
        cart2d = vector.represent_as(cx.CartesianPos2D, y=Quantity([5, 6, 7, 8], "km"))

        assert isinstance(cart2d, cx.CartesianPos2D)
        assert jnp.array_equal(
            cart2d.x,
            Quantity(
                [1.7452406e-02, 4.9681753e-01, -1.3060151e00, 8.6809595e-14], "kpc"
            ),
        )
        assert jnp.array_equal(
            cart2d.y,
            Quantity([0.0000000e00, 1.0654287e00, 1.3060151e00, -7.2797034e-06], "kpc"),
        )

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_spherical_to_polar(self, vector):
        """Test ``coordinax.represent_as(PolarPos)``."""
        polar = vector.represent_as(cx.PolarPos, phi=Quantity([0, 1, 2, 3], "rad"))

        assert isinstance(polar, cx.PolarPos)
        assert jnp.array_equal(
            polar.r,
            Quantity([1.7452406e-02, 1.1755705e00, 1.8469844e00, 7.2797034e-06], "kpc"),
        )
        assert jnp.array_equal(polar.phi, Quantity([0.0, 65.0, 135.0, 270.0], "deg"))

    def test_spherical_to_cartesian3d(self, vector):
        """Test ``coordinax.represent_as(CartesianPos3D)``."""
        cart3d = vector.represent_as(cx.CartesianPos3D)

        assert isinstance(cart3d, cx.CartesianPos3D)
        assert jnp.array_equal(
            cart3d.x,
            Quantity(
                [1.7452406e-02, 4.9681753e-01, -1.3060151e00, 8.6809595e-14], "kpc"
            ),
        )
        assert jnp.array_equal(
            cart3d.y,
            Quantity([0.0, 1.0654287e00, 1.3060151e00, -7.2797034e-06], "kpc"),
        )
        assert jnp.array_equal(
            cart3d.z, Quantity([0.9998477, 1.618034, -2.3640323, -4.0], "kpc")
        )

    def test_spherical_to_cartesian3d_astropy(self, vector, apyvector):
        """Test Astropy equivalence."""
        cart3d = vector.represent_as(cx.CartesianPos3D)

        apycart3 = apyvector.represent_as(apyc.CartesianRepresentation)
        assert np.allclose(convert(cart3d.x, APYQuantity), apycart3.x)
        assert np.allclose(convert(cart3d.y, APYQuantity), apycart3.y)
        assert np.allclose(convert(cart3d.z, APYQuantity), apycart3.z)

    def test_spherical_to_cylindrical(self, vector):
        """Test ``coordinax.represent_as(CylindricalPos)``."""
        cyl = vector.represent_as(cx.CylindricalPos, z=Quantity([9, 10, 11, 12], "m"))

        assert isinstance(cyl, cx.CylindricalPos)
        assert jnp.array_equal(
            cyl.rho,
            Quantity([1.7452406e-02, 1.1755705e00, 1.8469844e00, 7.2797034e-06], "kpc"),
        )
        assert jnp.array_equal(cyl.phi, Quantity([0.0, 65.0, 135.0, 270.0], "deg"))
        assert jnp.array_equal(
            cyl.z, Quantity([0.9998477, 1.618034, -2.3640323, -4.0], "kpc")
        )

    def test_spherical_to_cylindrical_astropy(self, vector, apyvector):
        """Test ``coordinax.represent_as(CylindricalPos)``."""
        cyl = vector.represent_as(cx.CylindricalPos, z=Quantity([9, 10, 11, 12], "m"))

        apycyl = apyvector.represent_as(apyc.CylindricalRepresentation)
        # There's a 'bug' in Astropy where rho can be negative.
        assert convert(cyl.rho[-1], APYQuantity) == apycyl.rho[-1]
        assert np.allclose(convert(cyl.rho, APYQuantity), np.abs(apycyl.rho))

        assert np.allclose(convert(cyl.z, APYQuantity), apycyl.z)
        # TODO: not require a modulus
        mod = Quantity(360, "deg")
        assert np.allclose(convert(cyl.phi, APYQuantity) % mod, apycyl.phi % mod)

    def test_spherical_to_spherical(self, vector):
        """Test ``coordinax.represent_as(SphericalPos)``."""
        # Jit can copy
        newvec = vector.represent_as(cx.SphericalPos)
        assert jnp.array_equal(newvec, vector)

        # The normal `represent_as` method should return the same object
        newvec = cx.represent_as(vector, cx.SphericalPos)
        assert newvec is vector

    def test_spherical_to_spherical_astropy(self, vector, apyvector):
        """Test Astropy equivalence."""
        sph = vector.represent_as(cx.SphericalPos)

        apysph = apyvector.represent_as(apyc.PhysicsSphericalRepresentation)
        assert np.allclose(convert(sph.r, APYQuantity), apysph.r)
        assert np.allclose(convert(sph.theta, APYQuantity), apysph.theta)
        assert np.allclose(convert(sph.phi, APYQuantity), apysph.phi)

    def test_spherical_to_mathspherical(self, vector):
        """Test ``coordinax.represent_as(MathSphericalPos)``."""
        newvec = cx.represent_as(vector, cx.MathSphericalPos)
        assert jnp.array_equal(newvec.r, vector.r)
        assert jnp.array_equal(newvec.theta, vector.phi)
        assert jnp.array_equal(newvec.phi, vector.theta)

    def test_spherical_to_lonlatspherical(self, vector):
        """Test ``coordinax.represent_as(LonLatSphericalPos)``."""
        llsph = vector.represent_as(
            cx.LonLatSphericalPos, z=Quantity([9, 10, 11, 12], "m")
        )

        assert isinstance(llsph, cx.LonLatSphericalPos)
        assert jnp.array_equal(llsph.lon, vector.phi)
        assert jnp.array_equal(llsph.lat, Quantity(90, "deg") - vector.theta)
        assert jnp.array_equal(llsph.distance, vector.r)

    def test_spherical_to_lonlatspherical_astropy(self, vector, apyvector):
        """Test Astropy equivalence."""
        llsph = vector.represent_as(cx.LonLatSphericalPos)

        apycart3 = apyvector.represent_as(apyc.SphericalRepresentation)
        assert np.allclose(convert(llsph.distance, APYQuantity), apycart3.distance)
        assert np.allclose(convert(llsph.lon, APYQuantity), apycart3.lon)
        assert np.allclose(convert(llsph.lat, APYQuantity), apycart3.lat)


class AbstractVel3DTest(AbstractVelTest):
    """Test :class:`coordinax.AbstractVel2D`."""

    # ==========================================================================
    # Unary operations

    def test_neg_compare_apy(
        self, difntl: cx.AbstractVel, apydifntl: apyc.BaseDifferential
    ):
        """Test negation."""
        assert all(representation_equal(convert(-difntl, type(apydifntl)), -apydifntl))


class TestCartesianVel3D(AbstractVel3DTest):
    """Test :class:`coordinax.CartesianVel3D`."""

    @pytest.fixture(scope="class")
    def difntl(self) -> cx.CartesianVel3D:
        """Return a differential."""
        return cx.CartesianVel3D(
            d_x=Quantity([5, 6, 7, 8], "km/s"),
            d_y=Quantity([9, 10, 11, 12], "km/s"),
            d_z=Quantity([13, 14, 15, 16], "km/s"),
        )

    @pytest.fixture(scope="class")
    def vector(self) -> cx.CartesianPos3D:
        """Return a vector."""
        return cx.CartesianPos3D(
            x=Quantity([1, 2, 3, 4], "kpc"),
            y=Quantity([5, 6, 7, 8], "kpc"),
            z=Quantity([9, 10, 11, 12], "kpc"),
        )

    @pytest.fixture(scope="class")
    def apydifntl(self, difntl: cx.CartesianVel3D):
        """Return an Astropy differential."""
        return convert(difntl, apyc.CartesianDifferential)

    @pytest.fixture(scope="class")
    def apyvector(self, vector: cx.CartesianPos3D):
        """Return an Astropy vector."""
        return convert(vector, apyc.CartesianRepresentation)

    # ==========================================================================

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian3d_to_cartesian1d(self, difntl, vector):
        """Test ``coordinax.represent_as(CartesianPos1D)``."""
        cart1d = difntl.represent_as(cx.CartesianVel1D, vector)

        assert isinstance(cart1d, cx.CartesianVel1D)
        assert jnp.array_equal(cart1d.d_x, Quantity([1, 2, 3, 4], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian3d_to_radial(self, difntl, vector):
        """Test ``coordinax.represent_as(RadialPos)``."""
        radial = difntl.represent_as(cx.RadialPos, vector)

        assert isinstance(radial, cx.RadialPos)
        assert jnp.array_equal(radial.d_r, Quantity([1, 2, 3, 4], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian3d_to_cartesian2d(self, difntl, vector):
        """Test ``coordinax.represent_as(CartesianPos2D)``."""
        cart2d = difntl.represent_as(cx.CartesianVel2D, vector)

        assert isinstance(cart2d, cx.CartesianVel2D)
        assert jnp.array_equal(cart2d.d_x, Quantity([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(cart2d.d_y, Quantity([5, 6, 7, 8], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian3d_to_polar(self, difntl, vector):
        """Test ``coordinax.represent_as(PolarPos)``."""
        polar = difntl.represent_as(cx.PolarPos, vector)

        assert isinstance(polar, cx.PolarPos)
        assert jnp.array_equal(polar.d_r, Quantity([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(polar.d_phi, Quantity([5, 6, 7, 8], "mas/yr"))

    def test_cartesian3d_to_cartesian3d(self, difntl, vector):
        """Test ``coordinax.represent_as(CartesianPos3D)``."""
        # Jit can copy
        newvec = difntl.represent_as(cx.CartesianVel3D, vector)
        assert jnp.array_equal(newvec, difntl)

        # The normal `represent_as` method should return the same object
        newvec = cx.represent_as(difntl, cx.CartesianVel3D, vector)
        assert newvec is difntl

    def test_cartesian3d_to_cartesian3d_astropy(
        self, difntl, vector, apydifntl, apyvector
    ):
        """Test Astropy equivalence."""
        cart3 = difntl.represent_as(cx.CartesianVel3D, vector)

        apycart3 = apydifntl.represent_as(apyc.CartesianDifferential, apyvector)
        assert np.allclose(convert(cart3.d_x, APYQuantity), apycart3.d_x)
        assert np.allclose(convert(cart3.d_y, APYQuantity), apycart3.d_y)
        assert np.allclose(convert(cart3.d_z, APYQuantity), apycart3.d_z)

    def test_cartesian3d_to_spherical(self, difntl, vector):
        """Test ``coordinax.represent_as(SphericalVel)``."""
        spherical = difntl.represent_as(cx.SphericalVel, vector)

        assert isinstance(spherical, cx.SphericalVel)
        assert jnp.allclose(
            spherical.d_r,
            Quantity([16.1445, 17.917269, 19.657543, 21.380898], "km/s"),
            atol=Quantity(1e-8, "km/s"),
        )
        assert jnp.allclose(
            spherical.d_phi,
            Quantity(
                [-0.61538464, -0.40000004, -0.275862, -0.19999999], "km rad / (kpc s)"
            ),
            atol=Quantity(1e-8, "mas/Myr"),
        )
        assert jnp.allclose(
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
        sph = difntl.represent_as(cx.SphericalVel, vector)

        apysph = apydifntl.represent_as(apyc.PhysicsSphericalDifferential, apyvector)
        assert np.allclose(convert(sph.d_r, APYQuantity), apysph.d_r)
        assert np.allclose(convert(sph.d_theta, APYQuantity), apysph.d_theta, atol=1e-9)
        assert np.allclose(convert(sph.d_phi, APYQuantity), apysph.d_phi, atol=1e-7)

    def test_cartesian3d_to_cylindrical(self, difntl, vector):
        """Test ``coordinax.represent_as(CylindricalVel)``."""
        cylindrical = difntl.represent_as(cx.CylindricalVel, vector)

        assert isinstance(cylindrical, cx.CylindricalVel)
        assert jnp.array_equal(
            cylindrical.d_rho,
            Quantity([9.805806, 11.384199, 12.86803, 14.310835], "km/s"),
        )
        assert jnp.allclose(
            cylindrical.d_phi,
            Quantity(
                [-0.61538464, -0.40000004, -0.275862, -0.19999999], "km rad / (kpc s)"
            ),
            atol=Quantity(1e-8, "mas/Myr"),
        )
        assert jnp.array_equal(
            cylindrical.d_z, Quantity([13.0, 14.0, 15.0, 16], "km/s")
        )

    def test_cartesian3d_to_cylindrical_astropy(
        self, difntl, vector, apydifntl, apyvector
    ):
        """Test Astropy equivalence."""
        cyl = difntl.represent_as(cx.CylindricalVel, vector)
        apycyl = apydifntl.represent_as(apyc.CylindricalDifferential, apyvector)
        assert np.allclose(convert(cyl.d_rho, APYQuantity), apycyl.d_rho)
        assert np.allclose(convert(cyl.d_phi, APYQuantity), apycyl.d_phi)
        assert np.allclose(convert(cyl.d_z, APYQuantity), apycyl.d_z)


class TestCylindricalVel(AbstractVel3DTest):
    """Test :class:`coordinax.CylindricalVel`."""

    @pytest.fixture(scope="class")
    def difntl(self) -> cx.CylindricalVel:
        """Return a differential."""
        return cx.CylindricalVel(
            d_rho=Quantity([5, 6, 7, 8], "km/s"),
            d_phi=Quantity([9, 10, 11, 12], "mas/yr"),
            d_z=Quantity([13, 14, 15, 16], "km/s"),
        )

    @pytest.fixture(scope="class")
    def vector(self) -> cx.CylindricalPos:
        """Return a vector."""
        return cx.CylindricalPos(
            rho=Quantity([1, 2, 3, 4], "kpc"),
            phi=Quantity([0, 1, 2, 3], "rad"),
            z=Quantity([9, 10, 11, 12], "kpc"),
        )

    @pytest.fixture(scope="class")
    def apydifntl(self, difntl: cx.CylindricalVel):
        """Return an Astropy differential."""
        return convert(difntl, apyc.CylindricalDifferential)

    @pytest.fixture(scope="class")
    def apyvector(self, vector: cx.CylindricalPos) -> apyc.CylindricalRepresentation:
        """Return an Astropy vector."""
        return convert(vector, apyc.CylindricalRepresentation)

    # ==========================================================================

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cylindrical_to_cartesian1d(self, difntl, vector):
        """Test ``coordinax.represent_as(CartesianPos1D)``."""
        cart1d = difntl.represent_as(cx.CartesianVel1D, vector)

        assert isinstance(cart1d, cx.CartesianVel1D)
        assert jnp.array_equal(cart1d.d_x, Quantity([1, 2, 3, 4], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cylindrical_to_radial(self, difntl, vector):
        """Test ``coordinax.represent_as(RadialPos)``."""
        radial = difntl.represent_as(cx.RadialPos, vector)

        assert isinstance(radial, cx.RadialPos)
        assert jnp.array_equal(radial.d_r, Quantity([1, 2, 3, 4], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cylindrical_to_cartesian2d(self, difntl, vector):
        """Test ``coordinax.represent_as(CartesianPos2D)``."""
        cart2d = difntl.represent_as(cx.CartesianVel2D, vector)

        assert isinstance(cart2d, cx.CartesianVel2D)
        assert jnp.array_equal(cart2d.d_x, Quantity([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(cart2d.d_y, Quantity([5, 6, 7, 8], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cylindrical_to_polar(self, difntl, vector):
        """Test ``coordinax.represent_as(PolarPos)``."""
        polar = difntl.represent_as(cx.PolarPos, vector)

        assert isinstance(polar, cx.PolarPos)
        assert jnp.array_equal(polar.d_r, Quantity([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(polar.d_phi, Quantity([5, 6, 7, 8], "mas/yr"))

    def test_cylindrical_to_cartesian3d(self, difntl, vector, apydifntl, apyvector):
        """Test ``coordinax.represent_as(CartesianPos3D)``."""
        cart3d = difntl.represent_as(cx.CartesianVel3D, vector)

        assert isinstance(cart3d, cx.CartesianVel3D)
        assert jnp.array_equal(
            cart3d.d_x, Quantity([5.0, -76.537544, -145.15944, -40.03075], "km/s")
        )
        assert jnp.array_equal(
            cart3d.d_y,
            Quantity([42.664234, 56.274563, -58.73506, -224.13647], "km/s"),
        )
        assert jnp.array_equal(cart3d.d_z, Quantity([13.0, 14.0, 15.0, 16.0], "km/s"))

        apycart3 = apydifntl.represent_as(apyc.CartesianDifferential, apyvector)
        assert np.allclose(convert(cart3d.d_x, APYQuantity), apycart3.d_x)
        assert np.allclose(convert(cart3d.d_y, APYQuantity), apycart3.d_y)
        assert np.allclose(convert(cart3d.d_z, APYQuantity), apycart3.d_z)

    def test_cylindrical_to_spherical(self, difntl, vector):
        """Test ``coordinax.represent_as(SphericalVel)``."""
        dsph = difntl.represent_as(cx.SphericalVel, vector)

        assert isinstance(dsph, cx.SphericalVel)
        assert jnp.array_equal(
            dsph.d_r,
            Quantity([13.472646, 14.904826, 16.313278, 17.708754], "km/s"),
        )
        assert jnp.allclose(
            dsph.d_theta,
            Quantity(
                [0.3902412, 0.30769292, 0.24615361, 0.19999981], "km rad / (kpc s)"
            ),
            atol=Quantity(5e-7, "km rad / (kpc s)"),
        )
        assert jnp.array_equal(
            dsph.d_phi,
            Quantity([42.664234, 47.404705, 52.145176, 56.885643], "km rad / (kpc s)"),
        )

    def test_cylindrical_to_spherical_astropy(
        self, difntl, vector, apydifntl, apyvector
    ):
        """Test Astropy equivalence."""
        sph = difntl.represent_as(cx.SphericalVel, vector)
        apysph = apydifntl.represent_as(apyc.PhysicsSphericalDifferential, apyvector)
        assert np.allclose(convert(sph.d_r, APYQuantity), apysph.d_r)
        assert np.allclose(convert(sph.d_theta, APYQuantity), apysph.d_theta)
        assert np.allclose(convert(sph.d_phi, APYQuantity), apysph.d_phi)

    def test_cylindrical_to_cylindrical(self, difntl, vector):
        """Test ``coordinax.represent_as(CylindricalVel)``."""
        # Jit can copy
        newvec = difntl.represent_as(cx.CylindricalVel, vector)
        assert newvec == difntl

        # The normal `represent_as` method should return the same object
        newvec = cx.represent_as(difntl, cx.CylindricalVel, vector)
        assert newvec is difntl

    def test_cylindrical_to_cylindrical(self, difntl, vector, apydifntl, apyvector):
        """Test Astropy equivalence."""
        cyl = difntl.represent_as(cx.CylindricalVel, vector)
        apycyl = apydifntl.represent_as(apyc.CylindricalDifferential, apyvector)
        assert np.allclose(convert(cyl.d_rho, APYQuantity), apycyl.d_rho)
        assert np.allclose(convert(cyl.d_phi, APYQuantity), apycyl.d_phi)
        assert np.allclose(convert(cyl.d_z, APYQuantity), apycyl.d_z)


class TestSphericalVel(AbstractVel3DTest):
    """Test :class:`coordinax.SphericalVel`."""

    @pytest.fixture(scope="class")
    def difntl(self) -> cx.SphericalVel:
        """Return a differential."""
        return cx.SphericalVel(
            d_r=Quantity([5, 6, 7, 8], "km/s"),
            d_theta=Quantity([13, 14, 15, 16], "mas/yr"),
            d_phi=Quantity([9, 10, 11, 12], "mas/yr"),
        )

    @pytest.fixture(scope="class")
    def vector(self) -> cx.SphericalPos:
        """Return a vector."""
        return cx.SphericalPos(
            r=Quantity([1, 2, 3, 4], "kpc"),
            theta=Quantity([3, 63, 90, 179.5], "deg"),
            phi=Quantity([0, 42, 160, 270], "deg"),
        )

    @pytest.fixture(scope="class")
    def apydifntl(self, difntl: cx.SphericalVel) -> apyc.PhysicsSphericalDifferential:
        """Return an Astropy differential."""
        return convert(difntl, apyc.PhysicsSphericalDifferential)

    @pytest.fixture(scope="class")
    def apyvector(self, vector: cx.SphericalPos) -> apyc.PhysicsSphericalRepresentation:
        """Return an Astropy vector."""
        return convert(vector, apyc.PhysicsSphericalRepresentation)

    # ==========================================================================

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_spherical_to_cartesian1d(self, difntl, vector):
        """Test ``coordinax.represent_as(CartesianPos1D)``."""
        cart1d = difntl.represent_as(cx.CartesianVel1D, vector)

        assert isinstance(cart1d, cx.CartesianVel1D)
        assert jnp.array_equal(cart1d.d_x, Quantity([1, 2, 3, 4], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_spherical_to_radial(self, difntl, vector):
        """Test ``coordinax.represent_as(RadialPos)``."""
        radial = difntl.represent_as(cx.RadialPos, vector)

        assert isinstance(radial, cx.RadialPos)
        assert jnp.array_equal(radial.d_r, Quantity([1, 2, 3, 4], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_spherical_to_cartesian2d(self, difntl, vector):
        """Test ``coordinax.represent_as(CartesianPos2D)``."""
        cart2d = difntl.represent_as(cx.CartesianVel2D, vector)

        assert isinstance(cart2d, cx.CartesianVel2D)
        assert jnp.array_equal(cart2d.d_x, Quantity([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(cart2d.d_y, Quantity([5, 6, 7, 8], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_spherical_to_polar(self, difntl, vector):
        """Test ``coordinax.represent_as(PolarPos)``."""
        polar = difntl.represent_as(cx.PolarPos, vector)

        assert isinstance(polar, cx.PolarPos)
        assert jnp.array_equal(polar.d_r, Quantity([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(polar.d_phi, Quantity([5, 6, 7, 8], "mas/yr"))

    def test_spherical_to_cartesian3d(self, difntl, vector):
        """Test ``coordinax.represent_as(CartesianPos3D)``."""
        cart3d = difntl.represent_as(cx.CartesianVel3D, vector)

        assert isinstance(cart3d, cx.CartesianVel3D)
        assert jnp.allclose(
            cart3d.d_x,
            Quantity([61.803337, -7.770853, -60.081947, 1.985678], "km/s"),
            atol=Quantity(1e-8, "km/s"),
        )
        assert jnp.allclose(
            cart3d.d_y,
            Quantity([2.2328734, 106.6765, -144.60716, 303.30875], "km/s"),
            atol=Quantity(1e-8, "km/s"),
        )
        assert jnp.allclose(
            cart3d.d_z,
            Quantity([1.7678856, -115.542175, -213.32118, -10.647271], "km/s"),
            atol=Quantity(1e-8, "km/s"),
        )

    def test_spherical_to_cartesian3d_astropy(
        self, difntl, vector, apydifntl, apyvector
    ):
        """Test Astropy equivalence."""
        cart3d = difntl.represent_as(cx.CartesianVel3D, vector)

        apycart3 = apydifntl.represent_as(apyc.CartesianDifferential, apyvector)
        assert np.allclose(convert(cart3d.d_x, APYQuantity), apycart3.d_x)
        assert np.allclose(convert(cart3d.d_y, APYQuantity), apycart3.d_y)
        assert np.allclose(convert(cart3d.d_z, APYQuantity), apycart3.d_z)

    def test_spherical_to_cylindrical(self, difntl, vector):
        """Test ``coordinax.represent_as(CylindricalVel)``."""
        cylindrical = difntl.represent_as(cx.CylindricalVel, vector)

        assert isinstance(cylindrical, cx.CylindricalVel)
        assert jnp.allclose(
            cylindrical.d_rho,
            Quantity([61.803337, 65.60564, 6.9999905, -303.30875], "km/s"),
            atol=Quantity(1e-8, "km/s"),
        )
        assert jnp.allclose(
            cylindrical.d_phi,
            Quantity([2444.4805, 2716.0894, 2987.6985, 3259.3074], "deg km / (kpc s)"),
            atol=Quantity(1e-8, "mas/yr"),
        )
        assert jnp.allclose(
            cylindrical.d_z,
            Quantity([1.7678856, -115.542175, -213.32118, -10.647271], "km/s"),
            atol=Quantity(1e-8, "km/s"),
        )

    def test_spherical_to_cylindrical_astropy(
        self, difntl, vector, apydifntl, apyvector
    ):
        """Test Astropy equivalence."""
        cyl = difntl.represent_as(cx.CylindricalVel, vector)
        apycyl = apydifntl.represent_as(apyc.CylindricalDifferential, apyvector)
        assert np.allclose(convert(cyl.d_rho, APYQuantity), apycyl.d_rho)
        assert np.allclose(convert(cyl.d_phi, APYQuantity), apycyl.d_phi)
        assert np.allclose(convert(cyl.d_z, APYQuantity), apycyl.d_z)

    def test_spherical_to_spherical(self, difntl, vector):
        """Test ``coordinax.represent_as(SphericalVel)``."""
        # Jit can copy
        newvec = difntl.represent_as(cx.SphericalVel, vector)
        assert all(newvec == difntl)

        # The normal `represent_as` method should return the same object
        newvec = cx.represent_as(difntl, cx.SphericalVel, vector)
        assert newvec is difntl

    def test_spherical_to_spherical_astropy(self, difntl, vector, apydifntl, apyvector):
        """Test Astropy equivalence."""
        sph = difntl.represent_as(cx.SphericalVel, vector)
        apysph = apydifntl.represent_as(apyc.PhysicsSphericalDifferential, apyvector)
        assert np.allclose(convert(sph.d_r, APYQuantity), apysph.d_r)
        assert np.allclose(convert(sph.d_theta, APYQuantity), apysph.d_theta)
        assert np.allclose(convert(sph.d_phi, APYQuantity), apysph.d_phi)

    def test_spherical_to_lonlatspherical(self, difntl, vector):
        """Test ``coordinax.represent_as(LonLatSphericalVel)``."""
        llsph = difntl.represent_as(cx.LonLatSphericalVel, vector)

        assert isinstance(llsph, cx.LonLatSphericalVel)
        assert jnp.array_equal(llsph.d_distance, difntl.d_r)
        assert jnp.array_equal(llsph.d_lon, difntl.d_phi)
        assert jnp.allclose(
            llsph.d_lat,
            Quantity([-13.0, -14.0, -15.0, -16.0], "mas/yr"),
            atol=Quantity(1e-8, "mas/yr"),
        )

    def test_spherical_to_lonlatspherical_astropy(
        self, difntl, vector, apydifntl, apyvector
    ):
        """Test Astropy equivalence."""
        cart3d = difntl.represent_as(cx.LonLatSphericalVel, vector)

        apycart3 = apydifntl.represent_as(apyc.SphericalDifferential, apyvector)
        assert np.allclose(convert(cart3d.d_distance, APYQuantity), apycart3.d_distance)
        assert np.allclose(convert(cart3d.d_lon, APYQuantity), apycart3.d_lon)
        assert np.allclose(convert(cart3d.d_lat, APYQuantity), apycart3.d_lat)

    def test_spherical_to_mathspherical(self, difntl, vector):
        """Test ``coordinax.represent_as(MathSpherical)``."""
        llsph = difntl.represent_as(cx.MathSphericalVel, vector)

        assert isinstance(llsph, cx.MathSphericalVel)
        assert jnp.array_equal(llsph.d_r, difntl.d_r)
        assert jnp.array_equal(llsph.d_phi, difntl.d_theta)
        assert jnp.array_equal(llsph.d_theta, difntl.d_phi)
