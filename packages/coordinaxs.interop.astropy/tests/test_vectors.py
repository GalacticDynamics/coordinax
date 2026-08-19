"""Conversions between Astropy representations and `coordinax.vectors.Point`.

`Point.cconvert(target)` must agree with Astropy's `represent_as(target)` for
every supported chart pair; that correspondence is the `CCONVERT_CASES` table.
CDict-level agreement is covered separately in ``test_ptmap_cdict.py``.
"""

__all__: tuple[str, ...] = ()

import astropy.coordinates as apyc
import astropy.units as apyu
import numpy as np
import plum
import pytest

import unxt as u

import coordinax as cx
import coordinax.charts as cxc

# ---------------------------------------------------------------------------
# Source points, and their Astropy counterparts

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


# ---------------------------------------------------------------------------
# cconvert vs represent_as

#: (source Point, source Astropy rep, target chart, target Astropy rep).
#: The chart's own `components` supply the keys to compare, and the Astropy
#: attribute names match them one-for-one.
CCONVERT_CASES = [
    (cart, apycart, cxc.cart3d, apyc.CartesianRepresentation),
    (cart, apycart, cxc.sph3d, apyc.PhysicsSphericalRepresentation),
    (cart, apycart, cxc.cyl3d, apyc.CylindricalRepresentation),
    (cyl, apycyl, cxc.cart3d, apyc.CartesianRepresentation),
    (cyl, apycyl, cxc.sph3d, apyc.PhysicsSphericalRepresentation),
    (cyl, apycyl, cxc.cyl3d, apyc.CylindricalRepresentation),
    (sph, apysph, cxc.cart3d, apyc.CartesianRepresentation),
    (sph, apysph, cxc.sph3d, apyc.PhysicsSphericalRepresentation),
    (sph, apysph, cxc.lonlat_sph3d, apyc.SphericalRepresentation),
]

CCONVERT_IDS = [
    f"{src.chart.__class__.__name__}->{target.__class__.__name__}"
    for src, _, target, _ in CCONVERT_CASES
]


@pytest.mark.parametrize(
    ("point", "apy_point", "target_chart", "apy_rep"), CCONVERT_CASES, ids=CCONVERT_IDS
)
def test_cconvert_matches_astropy(point, apy_point, target_chart, apy_rep) -> None:
    """`Point.cconvert(target)` equals Astropy's `represent_as(target)`."""
    got = point.cconvert(target_chart)
    expected = apy_point.represent_as(apy_rep)

    for key in target_chart.components:
        assert np.allclose(
            plum.convert(got[key], apyu.Quantity), getattr(expected, key)
        ), key


def test_spherical_to_cylindrical_astropy() -> None:
    """sph3d -> cyl3d, where Astropy's own rho can come out negative.

    Kept out of `CCONVERT_CASES` because the comparison is not the plain
    component-wise one: coordinax always returns rho >= 0, so the reference
    needs `abs`.
    """
    vec = sph.cconvert(cxc.cyl3d)
    apyvec = apysph.represent_as(apyc.CylindricalRepresentation)

    assert plum.convert(vec["rho"][-1], apyu.Quantity) == apyvec.rho[-1]
    assert np.allclose(plum.convert(vec["rho"], apyu.Quantity), np.abs(apyvec.rho))
    assert np.allclose(plum.convert(vec["z"], apyu.Quantity), apyvec.z)
    assert np.allclose(
        plum.convert(vec["phi"], apyu.Quantity), apyu.Quantity(apyvec.phi)
    )


# ---------------------------------------------------------------------------
# Negation


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
    """Negation agrees with Astropy once both are taken to Cartesian."""
    if apy_cls is None:
        pytest.xfail("No corresponding Astropy representation class.")

    # To negate, Point converts to Cartesian, negates, and converts back. That
    # can yield equivalent-but-different angular coordinates than Astropy --
    # AFAIK only at the poles -- so compare in Cartesian.
    negcart = plum.convert(-v, apy_cls).represent_as(apyc.CartesianRepresentation)
    negapycart = -plum.convert(v, apy_cls).represent_as(apyc.CartesianRepresentation)
    assert np.allclose(negcart.x, negapycart.x, atol=1e-6)
    assert np.allclose(negcart.y, negapycart.y, atol=1e-6)
    assert np.allclose(negcart.z, negapycart.z, atol=5e-7)
    # TODO: use representation_equal_up_to_angular_type


# ---------------------------------------------------------------------------
# Point (with frame) -> Astropy frame-with-data / SkyCoord


@pytest.mark.parametrize(
    ("frame", "kw"),
    [
        (
            apyc.ICRS,
            {"ra": 90 * apyu.deg, "dec": 45 * apyu.deg, "distance": 1 * apyu.kpc},
        ),
        (
            apyc.Galactic,
            {"l": 30 * apyu.deg, "b": 20 * apyu.deg, "distance": 2 * apyu.kpc},
        ),
        (
            apyc.Galactocentric,
            {"x": 1 * apyu.kpc, "y": 2 * apyu.kpc, "z": 3 * apyu.kpc},
        ),
    ],
)
def test_point_to_astropy_frame_roundtrip(frame, kw) -> None:
    """Astropy frame-with-data -> Point -> astropy frame-with-data is identity."""
    orig = frame(**kw)
    point = cx.Point.from_(orig)
    back = plum.convert(point, apyc.BaseCoordinateFrame)

    assert isinstance(back, frame)
    assert back.has_data
    d = (back.cartesian.xyz - orig.cartesian.xyz).to(apyu.pc).value
    assert np.allclose(d, 0.0, atol=1e-6)


def test_point_to_astropy_skycoord_roundtrip() -> None:
    """Astropy SkyCoord -> Point -> SkyCoord preserves the sky position."""
    orig = apyc.SkyCoord(ra=10 * apyu.deg, dec=-5 * apyu.deg, distance=5 * apyu.kpc)
    point = cx.Point.from_(orig)
    back = plum.convert(point, apyc.SkyCoord)

    assert isinstance(back, apyc.SkyCoord)
    assert back.separation_3d(orig).to(apyu.pc).value < 1e-6


def test_point_without_frame_to_astropy_frame_raises() -> None:
    """A Point with no reference frame cannot become an astropy frame."""
    point = cx.Point.from_([1, 2, 3], "kpc")  # noframe
    with pytest.raises(ValueError, match="no reference frame"):
        plum.convert(point, apyc.BaseCoordinateFrame)
