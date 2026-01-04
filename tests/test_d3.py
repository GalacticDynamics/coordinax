"""Test :mod:`coordinax._builtin`."""

import pytest
from test_base import AbstractPosTest, AbstractVelTest

import quaxed.numpy as jnp
import unxt as u

import coordinax as cx
import coordinax.vecs as cxv
from coordinax.distance import Distance


class AbstractPos3DTest(AbstractPosTest):
    """Test `coordinax.AbstractPos3D`."""


##############################################################################


class TestCartesianPos3D(AbstractPos3DTest):
    """Test `coordinax.CartesianPos3D`."""

    @pytest.fixture(scope="class")
    def vector(self) -> cxv.AbstractPos:
        """Return a vector."""
        return cx.CartesianPos3D(
            x=u.Q([1, 2, 3, 4], "kpc"),
            y=u.Q([5, 6, 7, 8], "kpc"),
            z=u.Q([9, 10, 11, 12], "kpc"),
        )

    # ==========================================================================
    # vconvert

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cartesian3d_to_cartesian1d(self, vector):
        """Test ``coordinax.vconvert(CartesianPos1D)``."""
        cart1d = vector.vconvert(cxv.CartesianPos1D)

        assert isinstance(cart1d, cxv.CartesianPos1D)
        assert jnp.array_equal(cart1d.x, u.Q([1, 2, 3, 4], "kpc"))

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cartesian3d_to_radial(self, vector):
        """Test ``coordinax.vconvert(RadialPos)``."""
        radial = vector.vconvert(cxv.RadialPos)

        assert isinstance(radial, cxv.RadialPos)
        assert jnp.array_equal(
            radial.r, u.Q([10.34408, 11.83216, 13.379088, 14.96663], "kpc")
        )

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cartesian3d_to_cartesian2d(self, vector):
        """Test ``coordinax.vconvert(CartesianPos2D)``."""
        cart2d = vector.vconvert(cxv.CartesianPos2D)

        assert isinstance(cart2d, cxv.CartesianPos2D)
        assert jnp.array_equal(cart2d.x, u.Q([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(cart2d.y, u.Q([5, 6, 7, 8], "kpc"))

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cartesian3d_to_polar(self, vector):
        """Test ``coordinax.vconvert(PolarPos)``."""
        polar = vector.vconvert(cxv.PolarPos, phi=u.Q([0, 1, 2, 3], "rad"))

        assert isinstance(polar, cxv.PolarPos)
        assert jnp.array_equal(polar.r, jnp.hypot(vector.x, vector.y))
        assert jnp.array_equal(
            polar.phi, u.Q([1.3734008, 1.2490457, 1.1659045, 1.1071488], "rad")
        )

    def test_cartesian3d_to_cartesian3d(self, vector):
        """Test ``coordinax.vconvert(CartesianPos3D)``."""
        # Jit can copy
        newvec = vector.vconvert(cx.CartesianPos3D)
        assert jnp.array_equal(newvec, vector)

        # The normal `vconvert` method should return the same object
        newvec = cx.vconvert(cx.CartesianPos3D, vector)
        assert newvec is vector

    def test_cartesian3d_to_spherical(self, vector):
        """Test ``coordinax.vconvert(SphericalPos)``."""
        spherical = vector.vconvert(cx.SphericalPos)

        assert isinstance(spherical, cx.SphericalPos)
        assert jnp.array_equal(
            spherical.r, u.Q([10.34408, 11.83216, 13.379088, 14.96663], "kpc")
        )
        assert jnp.array_equal(
            spherical.phi,
            u.Q([1.3734008, 1.2490457, 1.1659045, 1.1071488], "rad"),
        )
        assert jnp.allclose(
            spherical.theta,
            u.Q([0.51546645, 0.5639427, 0.6055685, 0.64052236], "rad"),
            atol=u.Q(1e-8, "rad"),
        )

    def test_cartesian3d_to_cylindrical(self, vector):
        """Test ``coordinax.vconvert(CylindricalPos)``."""
        cylindrical = vector.vconvert(cxv.CylindricalPos)

        assert isinstance(cylindrical, cxv.CylindricalPos)
        assert jnp.array_equal(cylindrical.rho, jnp.hypot(vector.x, vector.y))
        assert jnp.array_equal(
            cylindrical.phi,
            u.Q([1.3734008, 1.2490457, 1.1659045, 1.1071488], "rad"),
        )
        assert jnp.array_equal(cylindrical.z, u.Q([9.0, 10, 11, 12], "kpc"))


class TestCylindricalPos(AbstractPos3DTest):
    """Test `coordinax.CylindricalPos`."""

    @pytest.fixture(scope="class")
    def vector(self) -> cxv.AbstractPos:
        """Return a vector."""
        return cxv.CylindricalPos(
            rho=u.Q([1, 2, 3, 4], "kpc"),
            phi=u.Q([0, 1, 2, 3], "rad"),
            z=u.Q([9, 10, 11, 12], "m"),
        )

    # ==========================================================================
    # vconvert

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cylindrical_to_cartesian1d(self, vector):
        """Test ``coordinax.vconvert(CartesianPos1D)``."""
        cart1d = vector.vconvert(cxv.CartesianPos1D)

        assert isinstance(cart1d, cxv.CartesianPos1D)
        assert jnp.allclose(
            cart1d.x,
            u.Q([1.0, 1.0806047, -1.2484405, -3.95997], "kpc"),
            atol=u.Q(1e-8, "kpc"),
        )

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cylindrical_to_radial(self, vector):
        """Test ``coordinax.vconvert(RadialPos)``."""
        radial = vector.vconvert(cxv.RadialPos)

        assert isinstance(radial, cxv.RadialPos)
        assert jnp.array_equal(radial.r, u.Q([1, 2, 3, 4], "kpc"))

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cylindrical_to_cartesian2d(self, vector):
        """Test ``coordinax.vconvert(CartesianPos2D)``."""
        cart2d = vector.vconvert(cxv.CartesianPos2D)

        assert isinstance(cart2d, cxv.CartesianPos2D)
        assert jnp.array_equal(
            cart2d.x, u.Q([1.0, 1.0806046, -1.2484405, -3.95997], "kpc")
        )
        assert jnp.array_equal(
            cart2d.y, u.Q([0.0, 1.6829419, 2.7278922, 0.56448], "kpc")
        )

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_cylindrical_to_polar(self, vector):
        """Test ``coordinax.vconvert(PolarPos)``."""
        polar = vector.vconvert(cxv.PolarPos)

        assert isinstance(polar, cxv.PolarPos)
        assert jnp.array_equal(polar.r, u.Q([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(polar.phi, u.Q([0, 1, 2, 3], "rad"))

    def test_cylindrical_to_cartesian3d(self, vector):
        """Test ``coordinax.vconvert(CartesianPos3D)``."""
        cart3d = vector.vconvert(cx.CartesianPos3D)

        assert isinstance(cart3d, cx.CartesianPos3D)
        assert jnp.array_equal(
            cart3d.x, u.Q([1.0, 1.0806046, -1.2484405, -3.95997], "kpc")
        )
        assert jnp.array_equal(
            cart3d.y, u.Q([0.0, 1.6829419, 2.7278922, 0.56448], "kpc")
        )
        assert jnp.array_equal(cart3d.z, vector.z)

    def test_cylindrical_to_spherical(self, vector):
        """Test ``coordinax.vconvert(SphericalPos)``."""
        spherical = vector.vconvert(cx.SphericalPos)

        assert isinstance(spherical, cx.SphericalPos)
        assert jnp.array_equal(spherical.r, u.Q([1, 2, 3, 4], "kpc"))
        assert jnp.array_equal(spherical.theta, u.Q(jnp.full(4, jnp.pi / 2), "rad"))
        assert jnp.array_equal(spherical.phi, u.Q([0, 1, 2, 3], "rad"))

    def test_cylindrical_to_cylindrical(self, vector):
        """Test ``coordinax.vconvert(CylindricalPos)``."""
        # Jit can copy
        newvec = vector.vconvert(cxv.CylindricalPos)
        assert jnp.array_equal(newvec, vector)

        # The normal `vconvert` method should return the same object
        newvec = cx.vconvert(cxv.CylindricalPos, vector)
        assert newvec is vector


class TestSphericalPos(AbstractPos3DTest):
    """Test `coordinax.SphericalPos`."""

    @pytest.fixture(scope="class")
    def vector(self) -> cx.SphericalPos:
        """Return a vector."""
        return cx.SphericalPos(
            r=u.Q([1, 2, 3, 4], "kpc"),
            theta=u.Q([1, 36, 142, 180 - 1e-4], "deg"),
            phi=u.Q([0, 65, 135, 270], "deg"),
        )

    # ==========================================================================
    # vconvert

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_spherical_to_cartesian1d(self, vector):
        """Test ``coordinax.vconvert(CartesianPos1D)``."""
        cart1d = vector.vconvert(cxv.CartesianPos1D)

        assert isinstance(cart1d, cxv.CartesianPos1D)
        assert jnp.allclose(
            cart1d.x,
            u.Q([1.7452406e-02, 4.9681753e-01, -1.3060151e00, 8.6809595e-14], "kpc"),
            atol=u.Q(1e-8, "kpc"),
        )

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_spherical_to_radial(self, vector):
        """Test ``coordinax.vconvert(RadialPos)``."""
        radial = vector.vconvert(cxv.RadialPos)

        assert isinstance(radial, cxv.RadialPos)
        assert jnp.array_equal(radial.r, u.Q([1, 2, 3, 4], "kpc"))

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_spherical_to_cartesian2d(self, vector):
        """Test ``coordinax.vconvert(CartesianPos2D)``."""
        cart2d = vector.vconvert(cxv.CartesianPos2D, y=u.Q([5, 6, 7, 8], "km"))

        assert isinstance(cart2d, cxv.CartesianPos2D)
        assert jnp.array_equal(
            cart2d.x,
            u.Q([1.7452406e-02, 4.9681753e-01, -1.3060151e00, 8.6809595e-14], "kpc"),
        )
        assert jnp.array_equal(
            cart2d.y,
            u.Q([0.0000000e00, 1.0654287e00, 1.3060151e00, -7.2797034e-06], "kpc"),
        )

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_spherical_to_polar(self, vector):
        """Test ``coordinax.vconvert(PolarPos)``."""
        polar = vector.vconvert(cxv.PolarPos)

        assert isinstance(polar, cxv.PolarPos)
        assert jnp.array_equal(
            polar.r,
            u.Q([1.7452406e-02, 1.1755705e00, 1.8469844e00, 7.2797034e-06], "kpc"),
        )
        assert jnp.array_equal(polar.phi, u.Q([0.0, 65.0, 135.0, 270.0], "deg"))

    def test_spherical_to_cartesian3d(self, vector):
        """Test ``coordinax.vconvert(CartesianPos3D)``."""
        cart3d = vector.vconvert(cx.CartesianPos3D)

        assert isinstance(cart3d, cx.CartesianPos3D)
        assert jnp.array_equal(
            cart3d.x,
            u.Q([1.7452406e-02, 4.9681753e-01, -1.3060151e00, 8.6809595e-14], "kpc"),
        )
        assert jnp.array_equal(
            cart3d.y,
            u.Q([0.0, 1.0654287e00, 1.3060151e00, -7.2797034e-06], "kpc"),
        )
        assert jnp.array_equal(
            cart3d.z, u.Q([0.9998477, 1.618034, -2.3640323, -4.0], "kpc")
        )

    def test_spherical_to_cylindrical(self, vector):
        """Test ``coordinax.vconvert(CylindricalPos)``."""
        cyl = vector.vconvert(cxv.CylindricalPos)

        assert isinstance(cyl, cxv.CylindricalPos)
        assert jnp.array_equal(
            cyl.rho,
            u.Q([1.7452406e-02, 1.1755705e00, 1.8469844e00, 7.2797034e-06], "kpc"),
        )
        assert jnp.array_equal(cyl.phi, u.Q([0.0, 65.0, 135.0, 270.0], "deg"))
        assert jnp.array_equal(
            cyl.z, u.Q([0.9998477, 1.618034, -2.3640323, -4.0], "kpc")
        )

    def test_spherical_to_spherical(self, vector):
        """Test ``coordinax.vconvert(SphericalPos)``."""
        # Jit can copy
        newvec = vector.vconvert(cx.SphericalPos)
        assert jnp.array_equal(newvec, vector)

        # The normal `vconvert` method should return the same object
        newvec = cx.vconvert(cx.SphericalPos, vector)
        assert newvec is vector

    def test_spherical_to_mathspherical(self, vector):
        """Test ``coordinax.vconvert(MathSphericalPos)``."""
        newvec = cx.vconvert(cxv.MathSphericalPos, vector)
        assert jnp.array_equal(newvec.r, vector.r)
        assert jnp.array_equal(newvec.theta, vector.phi)
        assert jnp.array_equal(newvec.phi, vector.theta)

    def test_spherical_to_lonlatspherical(self, vector):
        """Test ``coordinax.vconvert(LonLatSphericalPos)``."""
        llsph = vector.vconvert(cxv.LonLatSphericalPos)

        assert isinstance(llsph, cxv.LonLatSphericalPos)
        assert jnp.array_equal(llsph.lon, vector.phi)
        assert jnp.array_equal(llsph.lat, u.Q(90, "deg") - vector.theta)
        assert jnp.array_equal(llsph.distance, vector.r)


class TestProlateSpheroidalPos(AbstractPos3DTest):
    """Test `coordinax.ProlateSpheroidalPos`."""

    @pytest.fixture(scope="class")
    def vector(self) -> cxv.AbstractPos:
        """Return a vector."""
        return cxv.ProlateSpheroidalPos(
            mu=u.Q([1, 2, 3, 4], "kpc2"),
            nu=u.Q([0.1, 0.2, 0.3, 0.4], "kpc2"),
            phi=u.Q([0, 1, 2, 3], "rad"),
            Delta=u.Q(1.0, "kpc"),
        )

    # ==========================================================================
    # vconvert

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_prolatespheroidal_to_cartesian1d(self, vector):
        """Test ``coordinax.vconvert(CartesianPos1D)``."""
        cart1d = vector.vconvert(cxv.CartesianPos1D)

        assert isinstance(cart1d, cxv.CartesianPos1D)
        assert jnp.allclose(
            cart1d.x,
            u.Q([0.0, 0.48326105, -0.4923916, -1.3282144], "kpc"),
            atol=u.Q(1e-8, "kpc"),
        )

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_prolatespheroidal_to_radial(self, vector):
        """Test ``coordinax.vconvert(RadialPos)``."""
        radial = vector.vconvert(cxv.RadialPos)

        assert isinstance(radial, cxv.RadialPos)
        exp = u.Q([0.0, 0.8944272, 1.183216, 1.3416408], "kpc")
        assert jnp.array_equal(radial.r, exp)

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_prolatespheroidal_to_cartesian2d(self, vector):
        """Test ``coordinax.vconvert(CartesianPos2D)``."""
        cart2d = vector.vconvert(cxv.CartesianPos2D)

        assert isinstance(cart2d, cxv.CartesianPos2D)
        assert jnp.array_equal(
            cart2d.x, u.Q([0.0, 0.48326105, -0.4923916, -1.3282144], "kpc")
        )
        assert jnp.array_equal(
            cart2d.y, u.Q([0.0, 0.75263447, 1.0758952, 0.18933235], "kpc")
        )

    @pytest.mark.filterwarnings("ignore:Irreversible dimension change")
    def test_prolatespheroidal_to_polar(self, vector):
        """Test ``coordinax.vconvert(PolarPos)``."""
        polar = vector.vconvert(cxv.PolarPos)

        assert isinstance(polar, cxv.PolarPos)
        exp = Distance([0.0, 0.8944272, 1.183216, 1.3416408], "kpc")
        assert jnp.array_equal(polar.r, exp)

        exp = u.Q([0, 1, 2, 3], "rad")
        assert jnp.allclose(polar.phi, exp, atol=u.Q(1e-8, "rad"))

    def test_prolatespheroidal_to_cartesian3d(self, vector):
        """Test ``coordinax.vconvert(CartesianPos3D)``."""
        cart3d = vector.vconvert(cx.CartesianPos3D)

        assert isinstance(cart3d, cx.CartesianPos3D)
        assert jnp.array_equal(
            cart3d.x, u.Q([0.0, 0.48326105, -0.4923916, -1.3282144], "kpc")
        )
        assert jnp.array_equal(
            cart3d.y, u.Q([0.0, 0.75263447, 1.0758952, 0.18933235], "kpc")
        )
        assert jnp.array_equal(
            cart3d.z, u.Q([0.31622776, 0.6324555, 0.9486833, 1.264911], "kpc")
        )

    def test_prolatespheroidal_to_cylindrical(self, vector):
        """Test ``coordinax.vconvert(CylindricalPos)``."""
        cyl = vector.vconvert(cxv.CylindricalPos)

        assert isinstance(cyl, cxv.CylindricalPos)
        assert jnp.array_equal(
            cyl.rho, u.Q([0.0, 0.8944272, 1.183216, 1.3416408], "kpc")
        )
        assert jnp.array_equal(cyl.phi, vector.phi)
        assert jnp.array_equal(
            cyl.z, u.Q([0.31622776, 0.6324555, 0.9486833, 1.264911], "kpc")
        )

    def test_prolatespheroidal_to_spherical(self, vector):
        """Test ``coordinax.vconvert(SphericalPos)``."""
        spherical = vector.vconvert(cx.SphericalPos)

        assert isinstance(spherical, cx.SphericalPos)
        assert jnp.array_equal(
            spherical.r,
            u.Q([0.31622776, 1.0954452, 1.5165752, 1.8439089], "kpc"),
        )
        assert jnp.allclose(spherical.phi, vector.phi, atol=u.Q(1e-8, "rad"))
        assert jnp.allclose(
            spherical.theta,
            u.Q([0.0, 0.95531654, 0.89496875, 0.8148269], "rad"),
            atol=u.Q(1e-8, "rad"),
        )

    def test_prolatespheroidal_to_prolatespheroidal(self, vector):
        """Test ``coordinax.vconvert(ProlateSpheroidalPos)``."""
        # Jit can copy
        newvec = vector.vconvert(cxv.ProlateSpheroidalPos, Delta=vector.Delta)
        assert jnp.allclose(newvec.mu.value, vector.mu.value)
        assert jnp.allclose(newvec.nu.value, vector.nu.value)
        assert jnp.array_equal(newvec.phi, vector.phi)

        # With a different focal length, should not be the same:
        newvec = vector.vconvert(cxv.ProlateSpheroidalPos, Delta=u.Q(0.5, "kpc"))
        assert not jnp.allclose(newvec.mu.value, vector.mu.value)
        assert not jnp.allclose(newvec.nu.value, vector.nu.value)
        assert jnp.array_equal(newvec.phi, vector.phi)

        # The normal `vconvert` method should return the same object
        newvec = cx.vconvert(cxv.ProlateSpheroidalPos, vector)
        # TODO: re-enable when equality is fixed for array-valued vectors
        # assert newvec == vector


class AbstractVel3DTest(AbstractVelTest):
    """Test `coordinax.AbstractVel2D`."""


class TestCartesianVel3D(AbstractVel3DTest):
    """Test `coordinax.CartesianVel3D`."""

    @pytest.fixture(scope="class")
    def difntl(self) -> cx.CartesianVel3D:
        """Return a differential."""
        return cx.CartesianVel3D(
            x=u.Q([5, 6, 7, 8], "km/s"),
            y=u.Q([9, 10, 11, 12], "km/s"),
            z=u.Q([13, 14, 15, 16], "km/s"),
        )

    @pytest.fixture(scope="class")
    def vector(self) -> cx.CartesianPos3D:
        """Return a vector."""
        return cx.CartesianPos3D(
            x=u.Q([1, 2, 3, 4], "kpc"),
            y=u.Q([5, 6, 7, 8], "kpc"),
            z=u.Q([9, 10, 11, 12], "kpc"),
        )

    # ==========================================================================

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian3d_to_cartesian1d(self, difntl, vector):
        """Test ``coordinax.vconvert(CartesianPos1D)``."""
        cart1d = difntl.vconvert(cxv.CartesianVel1D, vector)

        assert isinstance(cart1d, cxv.CartesianVel1D)
        assert jnp.array_equal(cart1d.x, u.Q([1, 2, 3, 4], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian3d_to_radial(self, difntl, vector):
        """Test ``coordinax.vconvert(RadialPos)``."""
        radial = difntl.vconvert(cxv.RadialPos, vector)

        assert isinstance(radial, cxv.RadialPos)
        assert jnp.array_equal(radial.r, u.Q([1, 2, 3, 4], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian3d_to_cartesian2d(self, difntl, vector):
        """Test ``coordinax.vconvert(CartesianPos2D)``."""
        cart2d = difntl.vconvert(cxv.CartesianVel2D, vector)

        assert isinstance(cart2d, cxv.CartesianVel2D)
        assert jnp.array_equal(cart2d.x, u.Q([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(cart2d.y, u.Q([5, 6, 7, 8], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cartesian3d_to_polar(self, difntl, vector):
        """Test ``coordinax.vconvert(PolarPos)``."""
        polar = difntl.vconvert(cxv.PolarPos, vector)

        assert isinstance(polar, cxv.PolarPos)
        assert jnp.array_equal(polar.r, u.Q([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(polar.phi, u.Q([5, 6, 7, 8], "mas/yr"))

    def test_cartesian3d_to_cartesian3d(self, difntl, vector):
        """Test ``coordinax.vconvert(CartesianPos3D)``."""
        # Jit can copy
        newvec = difntl.vconvert(cx.CartesianVel3D, vector)
        assert jnp.array_equal(newvec, difntl)

        # The normal `vconvert` method should return the same object
        newvec = cx.vconvert(cx.CartesianVel3D, difntl, vector)
        assert newvec is difntl

    def test_cartesian3d_to_spherical(self, difntl, vector):
        """Test ``coordinax.vconvert(SphericalVel)``."""
        spherical = difntl.vconvert(cx.SphericalVel, vector)

        assert isinstance(spherical, cx.SphericalVel)
        assert jnp.allclose(
            spherical.r,
            u.Q([16.1445, 17.917269, 19.657543, 21.380898], "km/s"),
            atol=u.Q(1e-8, "km/s"),
        )
        assert jnp.allclose(
            spherical.phi,
            u.Q([-0.61538464, -0.40000004, -0.275862, -0.19999999], "km rad / (kpc s)"),
            atol=u.Q(1e-8, "mas/Myr"),
        )
        assert jnp.allclose(
            spherical.theta,
            u.Q([0.2052807, 0.1807012, 0.15257944, 0.12777519], "km rad / (kpc s)"),
            atol=u.Q(1e-8, "mas/Myr"),
        )

    def test_cartesian3d_to_cylindrical(self, difntl, vector):
        """Test ``coordinax.vconvert(CylindricalVel)``."""
        cylindrical = difntl.vconvert(cxv.CylindricalVel, vector)

        assert isinstance(cylindrical, cxv.CylindricalVel)

        exp = u.Q([9.805806, 11.384199, 12.868031, 14.310835], "km/s")
        assert jnp.array_equal(cylindrical.rho, exp)

        exp = u.Q([-129815.1, -84379.82, -58192.97, -42189.902], "mas/Myr")
        assert jnp.allclose(cylindrical.phi, exp, atol=u.Q(1e-8, "mas/Myr"))

        exp = u.Q([13.0, 14.0, 15.0, 16], "km/s")
        assert jnp.array_equal(cylindrical.z, exp)

    def test_add_cartesian3d_diff_units(self, difntl):
        """Test that you can add CartesianVel3D with different units."""
        total = difntl + cx.CartesianVel3D(
            x=u.Q([1, 1, 1, 1], "m/s"),
            y=u.Q([0, 0, 0, 0], "m/s"),
            z=u.Q([0, 0, 0, 0], "m/s"),
        )
        correct_x = u.Q([5, 6, 7, 8], "km/s") + u.Q([1, 1, 1, 1], "m/s")
        assert jnp.allclose(total.x, correct_x, atol=u.Q(1e-8, "km/s"))


class TestCylindricalVel(AbstractVel3DTest):
    """Test `coordinax.CylindricalVel`."""

    @pytest.fixture(scope="class")
    def difntl(self) -> cxv.CylindricalVel:
        """Return a differential."""
        return cxv.CylindricalVel(
            rho=u.Q([5, 6, 7, 8], "km/s"),
            phi=u.Q([9, 10, 11, 12], "mas/yr"),
            z=u.Q([13, 14, 15, 16], "km/s"),
        )

    @pytest.fixture(scope="class")
    def vector(self) -> cxv.CylindricalPos:
        """Return a vector."""
        return cxv.CylindricalPos(
            rho=u.Q([1, 2, 3, 4], "kpc"),
            phi=u.Q([0, 1, 2, 3], "rad"),
            z=u.Q([9, 10, 11, 12], "kpc"),
        )

    # ==========================================================================

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cylindrical_to_cartesian1d(self, difntl, vector):
        """Test ``coordinax.vconvert(CartesianPos1D)``."""
        cart1d = difntl.vconvert(cxv.CartesianVel1D, vector)

        assert isinstance(cart1d, cxv.CartesianVel1D)
        assert jnp.array_equal(cart1d.x, u.Q([1, 2, 3, 4], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cylindrical_to_radial(self, difntl, vector):
        """Test ``coordinax.vconvert(RadialPos)``."""
        radial = difntl.vconvert(cxv.RadialPos, vector)

        assert isinstance(radial, cxv.RadialPos)
        assert jnp.array_equal(radial.r, u.Q([1, 2, 3, 4], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cylindrical_to_cartesian2d(self, difntl, vector):
        """Test ``coordinax.vconvert(CartesianPos2D)``."""
        cart2d = difntl.vconvert(cxv.CartesianVel2D, vector)

        assert isinstance(cart2d, cxv.CartesianVel2D)
        assert jnp.array_equal(cart2d.x, u.Q([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(cart2d.y, u.Q([5, 6, 7, 8], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_cylindrical_to_polar(self, difntl, vector):
        """Test ``coordinax.vconvert(PolarPos)``."""
        polar = difntl.vconvert(cxv.PolarPos, vector)

        assert isinstance(polar, cxv.PolarPos)
        assert jnp.array_equal(polar.r, u.Q([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(polar.phi, u.Q([5, 6, 7, 8], "mas/yr"))

    def test_cylindrical_to_spherical(self, difntl, vector):
        """Test ``coordinax.vconvert(SphericalVel)``."""
        dsph = difntl.vconvert(cx.SphericalVel, vector)

        assert isinstance(dsph, cx.SphericalVel)

        exp = u.Q([13.472647, 14.904824, 16.313278, 17.708754], "km/s")
        assert jnp.allclose(dsph.r, exp, atol=u.Q(1e-8, "km/s"))

        exp = u.Q([0.3902412, 0.30769292, 0.24615361, 0.19999981], "km rad / (kpc s)")
        assert jnp.allclose(dsph.theta, exp, atol=u.Q(5e-7, "km rad / (kpc s)"))

        exp = u.Q([9, 10, 11, 12], "mas / yr")
        assert jnp.array_equal(dsph.phi, exp)

    def test_cylindrical_to_cylindrical(self, difntl, vector):
        """Test ``coordinax.vconvert(CylindricalVel)``."""
        # Jit can copy
        newvec = difntl.vconvert(cxv.CylindricalVel, vector)
        assert jnp.array_equal(newvec, difntl)

        # The normal `vconvert` method should return the same object
        newvec = cx.vconvert(cxv.CylindricalVel, difntl, vector)
        assert newvec is difntl


class TestSphericalVel(AbstractVel3DTest):
    """Test `coordinax.SphericalVel`."""

    @pytest.fixture(scope="class")
    def difntl(self) -> cx.SphericalVel:
        """Return a differential."""
        return cx.SphericalVel(
            r=u.Q([5, 6, 7, 8], "km/s"),
            theta=u.Q([13, 14, 15, 16], "mas/yr"),
            phi=u.Q([9, 10, 11, 12], "mas/yr"),
        )

    @pytest.fixture(scope="class")
    def vector(self) -> cx.SphericalPos:
        """Return a vector."""
        return cx.SphericalPos(
            r=u.Q([1, 2, 3, 4], "kpc"),
            theta=u.Q([3, 63, 90, 179.5], "deg"),
            phi=u.Q([0, 42, 160, 270], "deg"),
        )

    # ==========================================================================

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_spherical_to_cartesian1d(self, difntl, vector):
        """Test ``coordinax.vconvert(CartesianPos1D)``."""
        cart1d = difntl.vconvert(cxv.CartesianVel1D, vector)

        assert isinstance(cart1d, cxv.CartesianVel1D)
        assert jnp.array_equal(cart1d.x, u.Q([1, 2, 3, 4], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_spherical_to_radial(self, difntl, vector):
        """Test ``coordinax.vconvert(RadialPos)``."""
        radial = difntl.vconvert(cxv.RadialPos, vector)

        assert isinstance(radial, cxv.RadialPos)
        assert jnp.array_equal(radial.r, u.Q([1, 2, 3, 4], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_spherical_to_cartesian2d(self, difntl, vector):
        """Test ``coordinax.vconvert(CartesianPos2D)``."""
        cart2d = difntl.vconvert(cxv.CartesianVel2D, vector)

        assert isinstance(cart2d, cxv.CartesianVel2D)
        assert jnp.array_equal(cart2d.x, u.Q([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(cart2d.y, u.Q([5, 6, 7, 8], "km/s"))

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
    def test_spherical_to_polar(self, difntl, vector):
        """Test ``coordinax.vconvert(PolarPos)``."""
        polar = difntl.vconvert(cxv.PolarPos, vector)

        assert isinstance(polar, cxv.PolarPos)
        assert jnp.array_equal(polar.r, u.Q([1, 2, 3, 4], "km/s"))
        assert jnp.array_equal(polar.phi, u.Q([5, 6, 7, 8], "mas/yr"))

    def test_spherical_to_cartesian3d(self, difntl, vector):
        """Test ``coordinax.vconvert(CartesianPos3D)``."""
        cart3d = difntl.vconvert(cx.CartesianVel3D, vector)

        assert isinstance(cart3d, cx.CartesianVel3D)
        assert jnp.allclose(
            cart3d.x,
            u.Q([61.803337, -7.770853, -60.081947, 1.985678], "km/s"),
            atol=u.Q(1e-8, "km/s"),
        )
        assert jnp.allclose(
            cart3d.y,
            u.Q([2.2328734, 106.6765, -144.60716, 303.30875], "km/s"),
            atol=u.Q(1e-8, "km/s"),
        )
        assert jnp.allclose(
            cart3d.z,
            u.Q([1.7678856, -115.542175, -213.32118, -10.647271], "km/s"),
            atol=u.Q(1e-8, "km/s"),
        )

    def test_spherical_to_cylindrical(self, difntl, vector):
        """Test ``coordinax.vconvert(CylindricalVel)``."""
        cylindrical = difntl.vconvert(cxv.CylindricalVel, vector)

        assert isinstance(cylindrical, cxv.CylindricalVel)
        assert jnp.allclose(
            cylindrical.rho,
            u.Q([61.803337, 65.60564, 6.9999905, -303.30875], "km/s"),
            atol=u.Q(1e-8, "km/s"),
        )
        assert jnp.allclose(
            cylindrical.phi,
            u.Q([2444.4805, 2716.0894, 2987.6985, 3259.3074], "deg km / (kpc s)"),
            atol=u.Q(1e-8, "mas/yr"),
        )
        assert jnp.allclose(
            cylindrical.z,
            u.Q([1.7678856, -115.542175, -213.32118, -10.647271], "km/s"),
            atol=u.Q(1e-8, "km/s"),
        )

    def test_spherical_to_spherical(self, difntl, vector):
        """Test ``coordinax.vconvert(SphericalVel)``."""
        # Jit can copy
        newvec = difntl.vconvert(cx.SphericalVel, vector)
        assert all(newvec == difntl)

        # The normal `vconvert` method should return the same object
        newvec = cx.vconvert(cx.SphericalVel, difntl, vector)
        assert newvec is difntl

    def test_spherical_to_lonlatspherical(self, difntl, vector):
        """Test ``coordinax.vconvert(LonLatSphericalVel)``."""
        llsph = difntl.vconvert(cxv.LonLatSphericalVel, vector)

        assert isinstance(llsph, cxv.LonLatSphericalVel)
        assert jnp.array_equal(llsph.distance, difntl.r)
        assert jnp.array_equal(llsph.lon, difntl.phi)
        assert jnp.allclose(
            llsph.lat,
            u.Q([-13.0, -14.0, -15.0, -16.0], "mas/yr"),
            atol=u.Q(1e-8, "mas/yr"),
        )

    def test_spherical_to_mathspherical(self, difntl, vector):
        """Test ``coordinax.vconvert(MathSpherical)``."""
        llsph = difntl.vconvert(cxv.MathSphericalVel, vector)

        assert isinstance(llsph, cxv.MathSphericalVel)
        assert jnp.array_equal(llsph.r, difntl.r)
        assert jnp.array_equal(llsph.phi, difntl.theta)
        assert jnp.array_equal(llsph.theta, difntl.phi)
