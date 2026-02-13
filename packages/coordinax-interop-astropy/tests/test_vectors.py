"""Test conversions between Astropy and coordinax-astro frames."""

import astropy.coordinates as apyc
import astropy.units as apyu
import numpy as np
import pytest
from plum import convert

import unxt as u

import coordinax as cx

cart = cx.Vector(
    {
        "x": u.Q([1, 2, 3, 4], "kpc"),
        "y": u.Q([5, 6, 7, 8], "kpc"),
        "z": u.Q([9, 10, 11, 12], "kpc"),
    },
    cxc.cart3d,
    cxr.phys_disp,
)
apycart = convert(cart, apyc.CartesianRepresentation)

cyl = cx.Vector(
    {
        "rho": u.Q([1, 2, 3, 4], "kpc"),
        "phi": u.Q([0, 1, 2, 3], "rad"),
        "z": u.Q([9, 10, 11, 12], "m"),
    },
    cxc.cyl3d,
    cxr.phys_disp,
)
apycyl = convert(cyl, apyc.CylindricalRepresentation)

sph = cx.Vector(
    {
        "r": u.Q([1, 2, 3, 4], "kpc"),
        "theta": u.Q([1, 36, 142, 180 - 1e-4], "deg"),
        "phi": u.Q([0, 65, 135, 270], "deg"),
    },
    cxc.sph3d,
    cxr.phys_disp,
)
apysph = convert(sph, apyc.PhysicsSphericalRepresentation)

prolatesph = cx.Vector(
    {
        "mu": u.Q([1, 2, 3, 4], "kpc2"),
        "nu": u.Q([0.1, 0.2, 0.3, 0.4], "kpc2"),
        "phi": u.Q([0, 1, 2, 3], "rad"),
    },
    cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(1.0, "kpc")),
    cxr.phys_disp,
)
apyprolatesph = None  # No corresponding Astropy representation


cartvel = cx.Vector(
    {
        "x": u.Q([5, 6, 7, 8], "km/s"),
        "y": u.Q([9, 10, 11, 12], "km/s"),
        "z": u.Q([13, 14, 15, 16], "km/s"),
    },
    cxc.cart3d,
    cxr.phys_vel,
)
apycartvel = convert(cartvel, apyc.CartesianDifferential)

cylvel = cx.Vector(
    {
        "rho": u.Q([5, 6, 7, 8], "km/s"),
        "phi": u.Q([9, 10, 11, 12], "mas/yr"),
        "z": u.Q([13, 14, 15, 16], "km/s"),
    },
    cxc.cyl3d,
    cxr.phys_vel,
)
apycylvel = convert(cylvel, apyc.CylindricalDifferential)

sphvel = cx.Vector(
    {
        "r": u.Q([5, 6, 7, 8], "km/s"),
        "theta": u.Q([13, 14, 15, 16], "mas/yr"),
        "phi": u.Q([9, 10, 11, 12], "mas/yr"),
    },
    cxc.sph3d,
    cxr.phys_vel,
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
    v: cx.Vector, apyv_cls: type[apyc.BaseRepresentation] | None
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
    ("phys_vel", "phys_disp", "apyv_cls"),
    [
        (cartvel, cart, apyc.CartesianDifferential),
        (cylvel, cyl, apyc.CylindricalDifferential),
        (sphvel, sph, apyc.PhysicsSphericalDifferential),
    ],
)
def test_negation_astropy_vel_roundtrip(
    vel: cx.Vector,
    pos: cx.Vector,
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
    vec = cart.vconvert(cxc.cart3d)
    apyvector = apycart.represent_as(apyc.CartesianRepresentation)

    assert np.allclose(convert(vec.x, apyu.Q), apyvector.x)
    assert np.allclose(convert(vec.y, apyu.Q), apyvector.y)
    assert np.allclose(convert(vec.z, apyu.Q), apyvector.z)


def test_cartesian3d_to_spherical_astropy():
    """Test Astropy equivalence."""
    vec = cart.vconvert(cxc.sph3d)
    apyvec = apycart.represent_as(apyc.PhysicsSphericalRepresentation)

    assert np.allclose(convert(vec.r, apyu.Q), apyvec.r)
    assert np.allclose(convert(vec.theta, apyu.Q), apyvec.theta)
    assert np.allclose(convert(vec.phi, apyu.Q), apyvec.phi)


def test_cartesian3d_to_cylindrical_astropy():
    """Test Astropy equivalence."""
    vec = cart.vconvert(cxc.cyl3d)
    apyvec = apycart.represent_as(apyc.CylindricalRepresentation)

    assert np.allclose(convert(vec.rho, apyu.Q), apyvec.rho)
    assert np.allclose(convert(vec.phi, apyu.Q), apyvec.phi)
    assert np.allclose(convert(vec.z, apyu.Q), apyvec.z)


def test_cylindrical_to_cartesian3d_astropy():
    """Test Astropy equivalence."""
    vec = cyl.vconvert(cxc.cart3d)
    apyvec = apycyl.represent_as(apyc.CartesianRepresentation)

    assert np.allclose(convert(vec.x, apyu.Q), apyvec.x)
    assert np.allclose(convert(vec.y, apyu.Q), apyvec.y)
    assert np.allclose(convert(vec.z, apyu.Q), apyvec.z)


def test_cylindrical_to_spherical_astropy():
    """Test Astropy equivalence."""
    vec = cyl.vconvert(cxc.sph3d)
    apyvec = apycyl.represent_as(apyc.PhysicsSphericalRepresentation)

    assert np.allclose(convert(vec.r, apyu.Q), apyvec.r)
    assert np.allclose(convert(vec.theta, apyu.Q), apyvec.theta)
    assert np.allclose(convert(vec.phi, apyu.Q), apyvec.phi)


def test_cylindrical_to_cylindrical_astropy():
    """Test Astropy equivalence."""
    vec = cyl.vconvert(cxc.cyl3d)
    apyvec = apycyl.represent_as(apyc.CylindricalRepresentation)

    assert np.allclose(convert(vec.rho, apyu.Q), apyvec.rho)
    assert np.allclose(convert(vec.phi, apyu.Q), apyvec.phi)
    assert np.allclose(convert(vec.z, apyu.Q), apyvec.z)


def test_spherical_to_cartesian3d_astropy():
    """Test Astropy equivalence."""
    vec = sph.vconvert(cxc.cart3d)
    apyvec = apysph.represent_as(apyc.CartesianRepresentation)

    assert np.allclose(convert(vec.x, apyu.Q), apyvec.x)
    assert np.allclose(convert(vec.y, apyu.Q), apyvec.y)
    assert np.allclose(convert(vec.z, apyu.Q), apyvec.z)


@pytest.mark.xfail(reason="FIXME")
def test_spherical_to_cylindrical_astropy():
    """Test ``coordinax.vconvert(CylindricalPos)``."""
    vec = sph.vconvert(cxc.cyl3d)
    apyvec = apysph.represent_as(apyc.CylindricalRepresentation)

    # There's a 'bug' in Astropy where rho can be negative.
    assert convert(vec.rho[-1], apyu.Q) == apyvec.rho[-1]
    assert np.allclose(convert(vec.rho, apyu.Q), np.abs(apyvec.rho))

    assert np.allclose(convert(vec.z, apyu.Q), apyvec.z)
    # TODO: not require a modulus
    mod = u.Q(360, "deg")
    assert np.allclose(convert(vec.phi, apyu.Q) % mod, apycyl.phi % mod)


def test_spherical_to_spherical_astropy():
    """Test Astropy equivalence."""
    vec = sph.vconvert(cxc.sph3d)
    apyvec = apysph.represent_as(apyc.PhysicsSphericalRepresentation)

    assert np.allclose(convert(vec.r, apyu.Q), apyvec.r)
    assert np.allclose(convert(vec.theta, apyu.Q), apyvec.theta)
    assert np.allclose(convert(vec.phi, apyu.Q), apyvec.phi)


def test_spherical_to_lonlatspherical_astropy():
    """Test Astropy equivalence."""
    vec = sph.vconvert(cxc.lonlatsph3d)
    apyvec = apysph.represent_as(apyc.SphericalRepresentation)

    assert np.allclose(convert(vec.distance, apyu.Q), apyvec.distance)
    assert np.allclose(convert(vec.lon, apyu.Q), apyvec.lon)
    assert np.allclose(convert(vec.lat, apyu.Q), apyvec.lat)


def test_cartesianvel3d_to_astropy_cartesiandifferential():
    """Test Astropy equivalence."""
    vec = cartvel.vconvert(cxc.cart3d, cart)
    apyvec = apycartvel.represent_as(apyc.CartesianDifferential, apycart)

    assert np.allclose(convert(vec.x, apyu.Q), apyvec.d_x)
    assert np.allclose(convert(vec.y, apyu.Q), apyvec.d_y)
    assert np.allclose(convert(vec.z, apyu.Q), apyvec.d_z)


def test_sphericalvel_to_astropy_physicssphericaldifferential():
    """Test Astropy equivalence."""
    vec = sphvel.vconvert(cxc.sph3d, sph)
    apyvec = apysphvel.represent_as(apyc.PhysicsSphericalDifferential, apysph)

    assert np.allclose(convert(vec.r, apyu.Q), apyvec.d_r)
    assert np.allclose(convert(vec.theta, apyu.Q), apyvec.d_theta, atol=1e-9)
    assert np.allclose(convert(vec.phi, apyu.Q), apyvec.d_phi, atol=1e-7)


def test_cartesianvel3d_to_astropy_cylindricaldifferential():
    """Test Astropy equivalence."""
    vec = cartvel.vconvert(cxc.cyl3d, cart)
    apyvec = apycartvel.represent_as(apyc.CylindricalDifferential, apycart)

    assert np.allclose(convert(vec.rho, apyu.Q), apyvec.d_rho)
    assert np.allclose(convert(vec.phi, apyu.Q), apyvec.d_phi)
    assert np.allclose(convert(vec.z, apyu.Q), apyvec.d_z)


@pytest.mark.xfail(reason="FIXME")
def test_cylindricalvel_to_astropy_cartesianvel3d():
    """Test ``coordinax.vconvert(CartesianPos3D)``."""
    vec = cylvel.vconvert(cxc.cart3d, cart)
    apyvec = apycylvel.represent_as(apyc.CartesianDifferential, apycyl)

    assert apyu.allclose(convert(vec.x, apyu.Q), apyvec.d_x)
    assert apyu.allclose(convert(vec.y, apyu.Q), apyvec.d_y)
    assert apyu.allclose(convert(vec.z, apyu.Q), apyvec.d_z)


def test_cylindricalvel_to_astropy_physicssphericaldifferential():
    """Test Astropy equivalence."""
    vec = cylvel.vconvert(cxc.sph3d, cyl)
    apyvec = apycylvel.represent_as(apyc.PhysicsSphericalDifferential, apycyl)

    assert np.allclose(convert(vec.r, apyu.Q), apyvec.d_r)
    assert np.allclose(convert(vec.theta, apyu.Q), apyvec.d_theta)
    assert np.allclose(convert(vec.phi, apyu.Q), apyvec.d_phi)


def test_cylindricalvel_to_astropy_cylindricaldifferential():
    """Test Astropy equivalence."""
    vec = cylvel.vconvert(cxc.cyl3d, cyl)
    apyvec = apycylvel.represent_as(apyc.CylindricalDifferential, apycyl)

    assert np.allclose(convert(vec.rho, apyu.Q), apyvec.d_rho)
    assert np.allclose(convert(vec.phi, apyu.Q), apyvec.d_phi)
    assert np.allclose(convert(vec.z, apyu.Q), apyvec.d_z)


@pytest.mark.xfail(reason="FIXME")
def test_sphericalvel_to_astropy_cartesiandifferential():
    """Test Astropy equivalence."""
    vec = sphvel.vconvert(cxc.cart3d, sph)
    apyvec = apysphvel.represent_as(apyc.CartesianDifferential, apysph)

    assert np.allclose(convert(vec.x, apyu.Q), apyvec.d_x)
    assert np.allclose(convert(vec.y, apyu.Q), apyvec.d_y)
    assert np.allclose(convert(vec.z, apyu.Q), apyvec.d_z)


def test_sphericalvel_to_astropy_cylindricaldifferential():
    """Test Astropy equivalence."""
    vec = sphvel.vconvert(cxc.cyl3d, sph)
    apyvec = apysphvel.represent_as(apyc.CylindricalDifferential, apysph)

    assert np.allclose(convert(vec.rho, apyu.Q), apyvec.d_rho)
    assert np.allclose(convert(vec.phi, apyu.Q), apyvec.d_phi)
    assert np.allclose(convert(vec.z, apyu.Q), apyvec.d_z)


def test_sphericalvel_to_astropy_sphericaldifferential():
    """Test Astropy equivalence."""
    vec = sphvel.vconvert(cxc.sph3d, sph)
    apyvec = apysphvel.represent_as(apyc.PhysicsSphericalDifferential, apysph)

    assert np.allclose(convert(vec.r, apyu.Q), apyvec.d_r)
    assert np.allclose(convert(vec.theta, apyu.Q), apyvec.d_theta)
    assert np.allclose(convert(vec.phi, apyu.Q), apyvec.d_phi)


def test_sphericalvel_to_astropy_sphericaldifferential():
    """Test Astropy equivalence."""
    vec = sphvel.vconvert(cxc.lonlatsph3d, sph)
    apyvec = apysphvel.represent_as(apyc.SphericalDifferential, apysph)

    assert np.allclose(convert(vec.distance, apyu.Q), apyvec.d_distance)
    assert np.allclose(convert(vec.lon, apyu.Q), apyvec.d_lon)
    assert np.allclose(convert(vec.lat, apyu.Q), apyvec.d_lat)
