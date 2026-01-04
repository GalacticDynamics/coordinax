"""Test conversions between Astropy and coordinax-astro frames."""

import astropy.coordinates as apyc
import astropy.units as apyu
import numpy as np
import pytest
from plum import convert

import unxt as u

import coordinax.vecs as cxv

cart = cxv.CartesianPos3D.from_([[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]], "kpc")
apycart = convert(cart, apyc.CartesianRepresentation)

cyl = cxv.CylindricalPos(
    rho=u.Q([1, 2, 3, 4], "kpc"),
    phi=u.Q([0, 1, 2, 3], "rad"),
    z=u.Q([9, 10, 11, 12], "m"),
)
apycyl = convert(cyl, apyc.CylindricalRepresentation)

sph = cxv.SphericalPos(
    r=u.Q([1, 2, 3, 4], "kpc"),
    theta=u.Q([1, 36, 142, 180 - 1e-4], "deg"),
    phi=u.Q([0, 65, 135, 270], "deg"),
)
apysph = convert(sph, apyc.PhysicsSphericalRepresentation)

prolatesph = cxv.ProlateSpheroidalPos(
    mu=u.Q([1, 2, 3, 4], "kpc2"),
    nu=u.Q([0.1, 0.2, 0.3, 0.4], "kpc2"),
    phi=u.Q([0, 1, 2, 3], "rad"),
    Delta=u.Q(1.0, "kpc"),
)
apyprolatesph = None  # No corresponding Astropy representation


cartvel = cxv.CartesianVel3D(
    x=u.Q([5, 6, 7, 8], "km/s"),
    y=u.Q([9, 10, 11, 12], "km/s"),
    z=u.Q([13, 14, 15, 16], "km/s"),
)
apycartvel = convert(cartvel, apyc.CartesianDifferential)

cylvel = cxv.CylindricalVel(
    rho=u.Q([5, 6, 7, 8], "km/s"),
    phi=u.Q([9, 10, 11, 12], "mas/yr"),
    z=u.Q([13, 14, 15, 16], "km/s"),
)
apycylvel = convert(cylvel, apyc.CylindricalDifferential)

sphvel = cxv.SphericalVel(
    r=u.Q([5, 6, 7, 8], "km/s"),
    theta=u.Q([13, 14, 15, 16], "mas/yr"),
    phi=u.Q([9, 10, 11, 12], "mas/yr"),
)
apysphvel = convert(sphvel, apyc.PhysicsSphericalDifferential)


# TODO: rewrite this test to use hypothesis
@pytest.mark.parametrize(
    ("v", "apyv_cls"),
    [
        (cart, apyc.CartesianRepresentation),
        (cyl, apyc.CylindricalRepresentation),
        (sph, apyc.PhysicsSphericalRepresentation),
        (prolatesph, None),
    ],
)
def test_negation_astropy_pos_roundtrip(
    v: cxv.AbstractVector, apyv_cls: type[apyc.BaseRepresentation] | None
) -> None:
    """Test negation."""
    if apyv_cls is None:
        pytest.xfail("No corresponding Astropy representation class.")

    # To take the negative, Vector converts to Cartesian coordinates, takes
    # the negative, then converts back to the original representation.
    # This can result in equivalent but different angular coordinates than
    # Astropy. AFAIK this only happens at the poles.
    negcart = convert(-v, apyv_cls).represent_as(apyc.CartesianRepresentation)
    negapycart = -convert(v, apyv_cls).represent_as(apyc.CartesianRepresentation)
    assert np.allclose(negcart.x, negapycart.x, atol=1e-6)
    assert np.allclose(negcart.y, negapycart.y, atol=1e-6)
    assert np.allclose(negcart.z, negapycart.z, atol=5e-7)
    # TODO: use representation_equal_up_to_angular_type


# TODO: rewrite this test to use hypothesis
@pytest.mark.parametrize(
    ("vel", "pos", "apyv_cls"),
    [
        (cartvel, cart, apyc.CartesianDifferential),
        (cylvel, cyl, apyc.CylindricalDifferential),
        (sphvel, convert(cart, cxv.SphericalPos), apyc.PhysicsSphericalDifferential),
    ],
)
def test_negation_astropy_vel_roundtrip(
    vel: cxv.AbstractVector,
    pos: cxv.AbstractVector,
    apyv_cls: type[apyc.BaseRepresentation] | None,
) -> None:
    """Test negation."""
    if apyv_cls is None:
        pytest.xfail("No corresponding Astropy representation class.")

    # To take the negative, Vector converts to Cartesian coordinates, takes
    # the negative, then converts back to the original representation.
    # This can result in equivalent but different angular coordinates than
    # Astropy. AFAIK this only happens at the poles.
    apypos = convert(pos, apyc.CartesianRepresentation)
    negcart = convert(-vel, apyv_cls).to_cartesian(apypos)
    negapycart = -convert(vel, apyv_cls).to_cartesian(apypos)
    assert np.allclose(negcart.x, negapycart.x, atol=1e-6)
    assert np.allclose(negcart.y, negapycart.y, atol=1e-6)
    assert np.allclose(negcart.z, negapycart.z, atol=5e-7)
    # TODO: use representation_equal_up_to_angular_type


# =============================================================================


# TODO: rewrite this test to use hypothesis
def test_cartesian3d_to_astropy_cartesianrepresentation():
    """Test Astropy equivalence."""
    vec = cart.vconvert(cxv.CartesianPos3D)
    apyvector = apycart.represent_as(apyc.CartesianRepresentation)

    assert np.allclose(convert(vec.x, apyu.Quantity), apyvector.x)
    assert np.allclose(convert(vec.y, apyu.Quantity), apyvector.y)
    assert np.allclose(convert(vec.z, apyu.Quantity), apyvector.z)


def test_cartesian3d_to_spherical_astropy():
    """Test Astropy equivalence."""
    vec = cart.vconvert(cxv.SphericalPos)
    apyvec = apycart.represent_as(apyc.PhysicsSphericalRepresentation)

    assert np.allclose(convert(vec.r, apyu.Quantity), apyvec.r)
    assert np.allclose(convert(vec.theta, apyu.Quantity), apyvec.theta)
    assert np.allclose(convert(vec.phi, apyu.Quantity), apyvec.phi)


def test_cartesian3d_to_cylindrical_astropy():
    """Test Astropy equivalence."""
    vec = cart.vconvert(cxv.CylindricalPos)
    apyvec = apycart.represent_as(apyc.CylindricalRepresentation)

    assert np.allclose(convert(vec.rho, apyu.Quantity), apyvec.rho)
    assert np.allclose(convert(vec.phi, apyu.Quantity), apyvec.phi)
    assert np.allclose(convert(vec.z, apyu.Quantity), apyvec.z)


def test_cylindrical_to_cartesian3d_astropy():
    """Test Astropy equivalence."""
    vec = cyl.vconvert(cxv.CartesianPos3D)
    apyvec = apycyl.represent_as(apyc.CartesianRepresentation)

    assert np.allclose(convert(vec.x, apyu.Quantity), apyvec.x)
    assert np.allclose(convert(vec.y, apyu.Quantity), apyvec.y)
    assert np.allclose(convert(vec.z, apyu.Quantity), apyvec.z)


def test_cylindrical_to_spherical_astropy():
    """Test Astropy equivalence."""
    vec = cyl.vconvert(cxv.SphericalPos)
    apyvec = apycyl.represent_as(apyc.PhysicsSphericalRepresentation)

    assert np.allclose(convert(vec.r, apyu.Quantity), apyvec.r)
    assert np.allclose(convert(vec.theta, apyu.Quantity), apyvec.theta)
    assert np.allclose(convert(vec.phi, apyu.Quantity), apyvec.phi)


def test_cylindrical_to_cylindrical_astropy():
    """Test Astropy equivalence."""
    vec = cyl.vconvert(cxv.CylindricalPos)
    apyvec = apycyl.represent_as(apyc.CylindricalRepresentation)

    assert np.allclose(convert(vec.rho, apyu.Quantity), apyvec.rho)
    assert np.allclose(convert(vec.phi, apyu.Quantity), apyvec.phi)
    assert np.allclose(convert(vec.z, apyu.Quantity), apyvec.z)


def test_spherical_to_cartesian3d_astropy():
    """Test Astropy equivalence."""
    vec = sph.vconvert(cxv.CartesianPos3D)
    apyvec = apysph.represent_as(apyc.CartesianRepresentation)

    assert np.allclose(convert(vec.x, apyu.Quantity), apyvec.x)
    assert np.allclose(convert(vec.y, apyu.Quantity), apyvec.y)
    assert np.allclose(convert(vec.z, apyu.Quantity), apyvec.z)


@pytest.mark.xfail(reason="FIXME")
def test_spherical_to_cylindrical_astropy():
    """Test ``coordinax.vconvert(CylindricalPos)``."""
    vec = sph.vconvert(cxv.CylindricalPos)
    apyvec = apysph.represent_as(apyc.CylindricalRepresentation)

    # There's a 'bug' in Astropy where rho can be negative.
    assert convert(vec.rho[-1], apyu.Quantity) == apyvec.rho[-1]
    assert np.allclose(convert(vec.rho, apyu.Quantity), np.abs(apyvec.rho))

    assert np.allclose(convert(vec.z, apyu.Quantity), apyvec.z)
    # TODO: not require a modulus
    mod = u.Q(360, "deg")
    assert np.allclose(convert(vec.phi, apyu.Quantity) % mod, apycyl.phi % mod)


def test_spherical_to_spherical_astropy():
    """Test Astropy equivalence."""
    vec = sph.vconvert(cxv.SphericalPos)
    apyvec = apysph.represent_as(apyc.PhysicsSphericalRepresentation)

    assert np.allclose(convert(vec.r, apyu.Quantity), apyvec.r)
    assert np.allclose(convert(vec.theta, apyu.Quantity), apyvec.theta)
    assert np.allclose(convert(vec.phi, apyu.Quantity), apyvec.phi)


def test_spherical_to_lonlatspherical_astropy():
    """Test Astropy equivalence."""
    vec = sph.vconvert(cxv.LonLatSphericalPos)
    apyvec = apysph.represent_as(apyc.SphericalRepresentation)

    assert np.allclose(convert(vec.distance, apyu.Quantity), apyvec.distance)
    assert np.allclose(convert(vec.lon, apyu.Quantity), apyvec.lon)
    assert np.allclose(convert(vec.lat, apyu.Quantity), apyvec.lat)


def test_cartesianvel3d_to_astropy_cartesiandifferential():
    """Test Astropy equivalence."""
    vec = cartvel.vconvert(cxv.CartesianVel3D, cart)
    apyvec = apycartvel.represent_as(apyc.CartesianDifferential, apycart)

    assert np.allclose(convert(vec.x, apyu.Quantity), apyvec.d_x)
    assert np.allclose(convert(vec.y, apyu.Quantity), apyvec.d_y)
    assert np.allclose(convert(vec.z, apyu.Quantity), apyvec.d_z)


def test_sphericalvel_to_astropy_physicssphericaldifferential():
    """Test Astropy equivalence."""
    vec = sphvel.vconvert(cxv.SphericalVel, sph)
    apyvec = apysphvel.represent_as(apyc.PhysicsSphericalDifferential, apysph)

    assert np.allclose(convert(vec.r, apyu.Quantity), apyvec.d_r)
    assert np.allclose(convert(vec.theta, apyu.Quantity), apyvec.d_theta, atol=1e-9)
    assert np.allclose(convert(vec.phi, apyu.Quantity), apyvec.d_phi, atol=1e-7)


def test_cartesianvel3d_to_astropy_cylindricaldifferential():
    """Test Astropy equivalence."""
    vec = cartvel.vconvert(cxv.CylindricalVel, cart)
    apyvec = apycartvel.represent_as(apyc.CylindricalDifferential, apycart)

    assert np.allclose(convert(vec.rho, apyu.Quantity), apyvec.d_rho)
    assert np.allclose(convert(vec.phi, apyu.Quantity), apyvec.d_phi)
    assert np.allclose(convert(vec.z, apyu.Quantity), apyvec.d_z)


@pytest.mark.xfail(reason="FIXME")
def test_cylindricalvel_to_astropy_cartesianvel3d():
    """Test ``coordinax.vconvert(CartesianPos3D)``."""
    vec = cylvel.vconvert(cxv.CartesianVel3D, cart)
    apyvec = apycylvel.represent_as(apyc.CartesianDifferential, apycyl)

    assert apyu.allclose(convert(vec.x, apyu.Quantity), apyvec.d_x)
    assert apyu.allclose(convert(vec.y, apyu.Quantity), apyvec.d_y)
    assert apyu.allclose(convert(vec.z, apyu.Quantity), apyvec.d_z)


def test_cylindricalvel_to_astropy_physicssphericaldifferential():
    """Test Astropy equivalence."""
    vec = cylvel.vconvert(cxv.SphericalVel, cyl)
    apyvec = apycylvel.represent_as(apyc.PhysicsSphericalDifferential, apycyl)

    assert np.allclose(convert(vec.r, apyu.Quantity), apyvec.d_r)
    assert np.allclose(convert(vec.theta, apyu.Quantity), apyvec.d_theta)
    assert np.allclose(convert(vec.phi, apyu.Quantity), apyvec.d_phi)


def test_cylindricalvel_to_astropy_cylindricaldifferential():
    """Test Astropy equivalence."""
    vec = cylvel.vconvert(cxv.CylindricalVel, cyl)
    apyvec = apycylvel.represent_as(apyc.CylindricalDifferential, apycyl)

    assert np.allclose(convert(vec.rho, apyu.Quantity), apyvec.d_rho)
    assert np.allclose(convert(vec.phi, apyu.Quantity), apyvec.d_phi)
    assert np.allclose(convert(vec.z, apyu.Quantity), apyvec.d_z)


@pytest.mark.xfail(reason="FIXME")
def test_sphericalvel_to_astropy_cartesiandifferential():
    """Test Astropy equivalence."""
    vec = sphvel.vconvert(cxv.CartesianVel3D, sph)
    apyvec = apysphvel.represent_as(apyc.CartesianDifferential, apysph)

    assert np.allclose(convert(vec.x, apyu.Quantity), apyvec.d_x)
    assert np.allclose(convert(vec.y, apyu.Quantity), apyvec.d_y)
    assert np.allclose(convert(vec.z, apyu.Quantity), apyvec.d_z)


def test_sphericalvel_to_astropy_cylindricaldifferential():
    """Test Astropy equivalence."""
    vec = sphvel.vconvert(cxv.CylindricalVel, sph)
    apyvec = apysphvel.represent_as(apyc.CylindricalDifferential, apysph)

    assert np.allclose(convert(vec.rho, apyu.Quantity), apyvec.d_rho)
    assert np.allclose(convert(vec.phi, apyu.Quantity), apyvec.d_phi)
    assert np.allclose(convert(vec.z, apyu.Quantity), apyvec.d_z)


def test_sphericalvel_to_astropy_sphericaldifferential():
    """Test Astropy equivalence."""
    vec = sphvel.vconvert(cxv.SphericalVel, sph)
    apyvec = apysphvel.represent_as(apyc.PhysicsSphericalDifferential, apysph)

    assert np.allclose(convert(vec.r, apyu.Quantity), apyvec.d_r)
    assert np.allclose(convert(vec.theta, apyu.Quantity), apyvec.d_theta)
    assert np.allclose(convert(vec.phi, apyu.Quantity), apyvec.d_phi)


def test_sphericalvel_to_astropy_sphericaldifferential():
    """Test Astropy equivalence."""
    vec = sphvel.vconvert(cxv.LonLatSphericalVel, sph)
    apyvec = apysphvel.represent_as(apyc.SphericalDifferential, apysph)

    assert np.allclose(convert(vec.distance, apyu.Quantity), apyvec.d_distance)
    assert np.allclose(convert(vec.lon, apyu.Quantity), apyvec.d_lon)
    assert np.allclose(convert(vec.lat, apyu.Quantity), apyvec.d_lat)
