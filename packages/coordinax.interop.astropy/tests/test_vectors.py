"""Test conversions between Astropy and coordinax-astro frames."""

import astropy.coordinates as apyc
import astropy.units as apyu
import numpy as np
import plum
import pytest

import unxt as u

import coordinax.charts as cxc
import coordinax.main as cx

cart = cx.Point.from_(
    {
        "x": u.Q([1, 2, 3, 4], "kpc"),
        "y": u.Q([5, 6, 7, 8], "kpc"),
        "z": u.Q([9, 10, 11, 12], "kpc"),
    },
    cx.cart3d,
)
apycart = plum.convert(cart, apyc.CartesianRepresentation)

cyl = cx.Point.from_(
    {
        "rho": u.Q([1, 2, 3, 4], "kpc"),
        "phi": u.Q([0, 1, 2, 3], "rad"),
        "z": u.Q([9, 10, 11, 12], "m"),
    },
    cxc.cyl3d,
)
apycyl = plum.convert(cyl, apyc.CylindricalRepresentation)

sph = cx.Point.from_(
    {
        "r": u.Q([1, 2, 3, 4], "kpc"),
        "theta": u.Q([1, 36, 142, 180 - 1e-4], "deg"),
        "phi": u.Q([0, 65, 135, 270], "deg"),
    },
    cxc.sph3d,
)
apysph = plum.convert(sph, apyc.PhysicsSphericalRepresentation)

prolatesph = cx.Point.from_(
    {
        "mu": u.Q([1, 2, 3, 4], "kpc2"),
        "nu": u.Q([0.1, 0.2, 0.3, 0.4], "kpc2"),
        "phi": u.Q([0, 1, 2, 3], "rad"),
    },
    cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(1, "kpc")),
)
apyprolatesph = None  # No corresponding Astropy representation


# TODO: rewrite this test to use hypothesis
@pytest.mark.parametrize(
    ("v", "apy_cls"),
    [
        (cart, apyc.CartesianRepresentation),
        (cyl, apyc.CylindricalRepresentation),
        (sph, apyc.PhysicsSphericalRepresentation),
        (prolatesph, None),
    ],
)
def test_negation_astropy_point_roundtrip(
    v: cx.Point, apy_cls: type[apyc.BaseRepresentation] | None
) -> None:
    """Test negation."""
    if apy_cls is None:
        pytest.xfail("No corresponding Astropy representation class.")

    # To take the negative, Point converts to Cartesian coordinates, takes
    # the negative, then converts back to the original representation.
    # This can result in equivalent but different angular coordinates than
    # Astropy. AFAIK this only happens at the poles.
    negcart = plum.convert(-v, apy_cls).represent_as(apyc.CartesianRepresentation)
    negapycart = -plum.convert(v, apy_cls).represent_as(apyc.CartesianRepresentation)
    assert np.allclose(negcart.x, negapycart.x, atol=1e-6)
    assert np.allclose(negcart.y, negapycart.y, atol=1e-6)
    assert np.allclose(negcart.z, negapycart.z, atol=5e-7)
    # TODO: use representation_equal_up_to_angular_type


# =============================================================================


# TODO: rewrite this test to use hypothesis
def test_cartesian3d_to_astropy_cartesianrepresentation():
    """Test Astropy equivalence."""
    vec = cart.cconvert(cxc.cart3d)
    apyvector = apycart.represent_as(apyc.CartesianRepresentation)

    assert np.allclose(plum.convert(vec["x"], apyu.Quantity), apyvector.x)
    assert np.allclose(plum.convert(vec["y"], apyu.Quantity), apyvector.y)
    assert np.allclose(plum.convert(vec["z"], apyu.Quantity), apyvector.z)


def test_cartesian3d_to_spherical_astropy():
    """Test Astropy equivalence."""
    vec = cart.cconvert(cxc.sph3d)
    apyvec = apycart.represent_as(apyc.PhysicsSphericalRepresentation)

    assert np.allclose(plum.convert(vec["r"], apyu.Quantity), apyvec.r)
    assert np.allclose(plum.convert(vec["theta"], apyu.Quantity), apyvec.theta)
    assert np.allclose(plum.convert(vec["phi"], apyu.Quantity), apyvec.phi)


def test_cartesian3d_to_cylindrical_astropy():
    """Test Astropy equivalence."""
    vec = cart.cconvert(cxc.cyl3d)
    apyvec = apycart.represent_as(apyc.CylindricalRepresentation)

    assert np.allclose(plum.convert(vec["rho"], apyu.Quantity), apyvec.rho)
    assert np.allclose(plum.convert(vec["phi"], apyu.Quantity), apyvec.phi)
    assert np.allclose(plum.convert(vec["z"], apyu.Quantity), apyvec.z)


def test_cylindrical_to_cartesian3d_astropy():
    """Test Astropy equivalence."""
    vec = cyl.cconvert(cxc.cart3d)
    apyvec = apycyl.represent_as(apyc.CartesianRepresentation)

    assert np.allclose(plum.convert(vec["x"], apyu.Quantity), apyvec.x)
    assert np.allclose(plum.convert(vec["y"], apyu.Quantity), apyvec.y)
    assert np.allclose(plum.convert(vec["z"], apyu.Quantity), apyvec.z)


def test_cylindrical_to_spherical_astropy():
    """Test Astropy equivalence."""
    vec = cyl.cconvert(cxc.sph3d)
    apyvec = apycyl.represent_as(apyc.PhysicsSphericalRepresentation)

    assert np.allclose(plum.convert(vec["r"], apyu.Quantity), apyvec.r)
    assert np.allclose(plum.convert(vec["theta"], apyu.Quantity), apyvec.theta)
    assert np.allclose(plum.convert(vec["phi"], apyu.Quantity), apyvec.phi)


def test_cylindrical_to_cylindrical_astropy():
    """Test Astropy equivalence."""
    vec = cyl.cconvert(cxc.cyl3d)
    apyvec = apycyl.represent_as(apyc.CylindricalRepresentation)

    assert np.allclose(plum.convert(vec["rho"], apyu.Quantity), apyvec.rho)
    assert np.allclose(plum.convert(vec["phi"], apyu.Quantity), apyvec.phi)
    assert np.allclose(plum.convert(vec["z"], apyu.Quantity), apyvec.z)


def test_spherical_to_cartesian3d_astropy():
    """Test Astropy equivalence."""
    vec = sph.cconvert(cxc.cart3d)
    apyvec = apysph.represent_as(apyc.CartesianRepresentation)

    assert np.allclose(plum.convert(vec["x"], apyu.Quantity), apyvec.x)
    assert np.allclose(plum.convert(vec["y"], apyu.Quantity), apyvec.y)
    assert np.allclose(plum.convert(vec["z"], apyu.Quantity), apyvec.z)


def test_spherical_to_cylindrical_astropy():
    """Test ``coordinax.cconvert(CylindricalPos)``."""
    vec = sph.cconvert(cxc.cyl3d)
    apyvec = apysph.represent_as(apyc.CylindricalRepresentation)

    # There's a 'bug' in Astropy where rho can be negative.
    assert plum.convert(vec["rho"][-1], apyu.Quantity) == apyvec.rho[-1]
    assert np.allclose(plum.convert(vec["rho"], apyu.Quantity), np.abs(apyvec.rho))

    assert np.allclose(plum.convert(vec["z"], apyu.Quantity), apyvec.z)
    assert np.allclose(
        plum.convert(vec["phi"], apyu.Quantity), apyu.Quantity(apyvec.phi)
    )


def test_spherical_to_spherical_astropy():
    """Test Astropy equivalence."""
    vec = sph.cconvert(cxc.sph3d)
    apyvec = apysph.represent_as(apyc.PhysicsSphericalRepresentation)

    assert np.allclose(plum.convert(vec["r"], apyu.Quantity), apyvec.r)
    assert np.allclose(plum.convert(vec["theta"], apyu.Quantity), apyvec.theta)
    assert np.allclose(plum.convert(vec["phi"], apyu.Quantity), apyvec.phi)


def test_spherical_to_lonlatspherical_astropy():
    """Test Astropy equivalence."""
    vec = sph.cconvert(cxc.lonlat_sph3d)
    apyvec = apysph.represent_as(apyc.SphericalRepresentation)

    assert np.allclose(plum.convert(vec["distance"], apyu.Quantity), apyvec.distance)
    assert np.allclose(plum.convert(vec["lon"], apyu.Quantity), apyvec.lon)
    assert np.allclose(plum.convert(vec["lat"], apyu.Quantity), apyvec.lat)
