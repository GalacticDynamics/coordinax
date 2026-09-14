"""Conversions between Astropy representations and `coordinax.vectors.Point`.

`Point.cconvert(target)` must agree with Astropy's `represent_as(target)` for
every supported chart pair; that correspondence is the `CCONVERT_CASES` table.
CDict-level agreement is covered separately in ``test_ptmap_cdict.py``.
"""

__all__: tuple[str, ...] = ()

from typing import ClassVar

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


class TestVelocityConversion:
    """`Tangent` <-> astropy differentials.

    The Cartesian pair has existed untested; the spherical pair is new. Both
    are asserted against astropy rather than against themselves, since astropy
    is the independent implementation here.
    """

    CART: ClassVar = {
        "x": u.Q(12.9, "km/s"),
        "y": u.Q(245.6, "km/s"),
        "z": u.Q(7.78, "km/s"),
    }
    LONLAT: ClassVar = {
        "lon": u.Q(1.5, "mas/yr"),
        "lat": u.Q(-2.5, "mas/yr"),
        "distance": u.Q(30.0, "km/s"),
    }

    def test_cartesian_round_trips_exactly(self):
        vel = cx.Tangent.from_(self.CART, cxc.cart3d)
        back = plum.convert(plum.convert(vel, apyc.CartesianDifferential), cx.Tangent)
        for k, want in self.CART.items():
            assert np.asarray(back[k].value) == np.asarray(want.value)

    def test_cartesian_values_reach_astropy(self):
        """Not just round-tripping: the numbers land in the right attributes."""
        got = plum.convert(
            cx.Tangent.from_(self.CART, cxc.cart3d), apyc.CartesianDifferential
        )
        assert got.d_x.to_value(apyu.km / apyu.s) == 12.9
        assert got.d_y.to_value(apyu.km / apyu.s) == 245.6
        assert got.d_z.to_value(apyu.km / apyu.s) == 7.78

    @pytest.mark.parametrize("unit", ["m/s", "km/s", "pc/Myr"])
    def test_the_unit_is_carried_not_assumed(self, unit):
        vel = cx.Tangent.from_({k: u.Q(1.0, unit) for k in "xyz"}, cxc.cart3d)
        got = plum.convert(vel, apyc.CartesianDifferential)
        assert got.d_x.to_value(apyu.Unit(unit)) == 1.0

    def test_a_non_cartesian_chart_is_refused_with_advice(self):
        """A tangent cannot change chart without its base point, so it says so."""
        vel = cx.Tangent.from_(self.LONLAT, cxc.lonlat_sph3d)
        with pytest.raises(ValueError, match="cconvert"):
            plum.convert(vel, apyc.CartesianDifferential)

    def test_a_non_lonlat_chart_is_refused_with_advice(self):
        """The spherical converter refuses the same way the Cartesian one does."""
        vel = cx.Tangent.from_(self.CART, cxc.cart3d)
        with pytest.raises(ValueError, match="cconvert"):
            plum.convert(vel, apyc.SphericalDifferential)

    def test_an_acceleration_is_refused(self):
        acc = cx.Tangent.from_({k: u.Q(1.0, "km/s2") for k in "xyz"}, cxc.cart3d)
        with pytest.raises(TypeError, match="Velocity"):
            plum.convert(acc, apyc.CartesianDifferential)

    def test_lonlat_round_trips_and_keeps_its_chart(self):
        vel = cx.Tangent.from_(self.LONLAT, cxc.lonlat_sph3d)
        back = plum.convert(plum.convert(vel, apyc.SphericalDifferential), cx.Tangent)
        assert back.chart == cxc.lonlat_sph3d
        for k, want in self.LONLAT.items():
            assert np.asarray(back[k].value) == np.asarray(want.value)

    def test_lonlat_needs_no_convention_change(self):
        """`d_lon` is the same quantity on both sides -- no cos(lat) anywhere."""
        got = plum.convert(
            cx.Tangent.from_(self.LONLAT, cxc.lonlat_sph3d), apyc.SphericalDifferential
        )
        assert got.d_lon.to_value(apyu.mas / apyu.yr) == 1.5
        assert got.d_lat.to_value(apyu.mas / apyu.yr) == -2.5
        assert got.d_distance.to_value(apyu.km / apyu.s) == 30.0


class TestTheCosLatConventionIsNotTheCosLatChart:
    """astropy's cos(lat) *rate* is not coordinax's lon*cos(lat) *coordinate*.

    `SphericalCosLatDifferential` carries `cos(lat) d_lon` on an ordinary
    (lon, lat, distance) base. `loncoslat_sph3d` is a chart whose coordinate
    is `lon cos(lat)`, so a tangent in it carries
    `cos(lat) d_lon - lon sin(lat) d_lat`. Matching them by component name
    would look right and be wrong by a term that grows toward the poles.
    """

    @pytest.mark.parametrize("lat_deg", [30.0, 60.0, 85.0])
    def test_the_two_genuinely_differ_away_from_the_equator(self, lat_deg):
        """The reason the conversion is refused rather than written."""
        lon, lat = np.radians(10.0), np.radians(lat_deg)
        d_lon, d_lat = 1.0, 2.0
        astropy_rate = np.cos(lat) * d_lon
        coordinax_rate = np.cos(lat) * d_lon - lon * np.sin(lat) * d_lat
        assert abs(astropy_rate - coordinax_rate) > 1e-3

    def test_they_agree_on_the_equator_where_the_extra_term_vanishes(self):
        lon, lat = np.radians(10.0), 0.0
        assert np.cos(lat) * 1.0 == np.cos(lat) * 1.0 - lon * np.sin(lat) * 2.0

    def test_converting_to_the_astropy_type_is_refused_by_name(self):
        vel = cx.Tangent.from_(
            {
                "lon_coslat": u.Q(1.0, "mas/yr"),
                "lat": u.Q(2.0, "mas/yr"),
                "distance": u.Q(3.0, "km/s"),
            },
            cxc.loncoslat_sph3d,
        )
        with pytest.raises(ValueError, match="rate convention"):
            plum.convert(vel, apyc.SphericalCosLatDifferential)

    def test_converting_from_the_astropy_type_is_refused_by_name(self):
        dif = apyc.SphericalCosLatDifferential(
            d_lon_coslat=1.0 * apyu.mas / apyu.yr,
            d_lat=2.0 * apyu.mas / apyu.yr,
            d_distance=3.0 * apyu.km / apyu.s,
        )
        with pytest.raises(ValueError, match="base point"):
            plum.convert(dif, cx.Tangent)

    def test_the_supported_route_works(self):
        """What the message tells you to do: give astropy the base, then convert."""
        dif = apyc.SphericalCosLatDifferential(
            d_lon_coslat=1.0 * apyu.mas / apyu.yr,
            d_lat=2.0 * apyu.mas / apyu.yr,
            d_distance=3.0 * apyu.km / apyu.s,
        )
        base = apyc.SphericalRepresentation(
            lon=10.0 * apyu.deg, lat=60.0 * apyu.deg, distance=1.0 * apyu.kpc
        )
        plain = dif.represent_as(apyc.SphericalDifferential, base=base)
        got = plum.convert(plain, cx.Tangent)
        assert got.chart == cxc.lonlat_sph3d
        # d_lon = d_lon_coslat / cos(lat); astropy did that, we just carried it.
        assert np.isclose(
            float(np.asarray(got["lon"].ustrip("mas/yr"))),
            1.0 / np.cos(np.radians(60.0)),
        )


def test_point_from_a_frame_without_data_names_that_frame() -> None:
    """Not "ICRS" whatever it was handed."""
    with pytest.raises(ValueError, match="Galactic frame has no data"):
        cx.Point.from_(apyc.Galactic())


class TestVelocityOnAstropyFramesAndSkyCoords:
    """`Point.from_` takes the position; `Tangent.from_` takes the velocity.

    An astropy frame holds both in one object. Converting one to a `Point`
    therefore leaves the velocity behind, and `Tangent.from_` is how it is
    picked up.
    """

    @staticmethod
    def galactocentric_with_velocity() -> apyc.Galactocentric:
        return apyc.Galactocentric(
            x=1.0 * apyu.kpc,
            y=2.0 * apyu.kpc,
            z=3.0 * apyu.kpc,
            v_x=4.0 * apyu.km / apyu.s,
            v_y=5.0 * apyu.km / apyu.s,
            v_z=6.0 * apyu.km / apyu.s,
        )

    def test_point_takes_the_position(self):
        point = cx.Point.from_(self.galactocentric_with_velocity())
        assert sorted(point.data) == ["x", "y", "z"]
        assert np.allclose(point["x"].ustrip("kpc"), 1.0)

    def test_tangent_takes_the_velocity(self):
        vel = cx.Tangent.from_(self.galactocentric_with_velocity())
        assert vel.chart == cxc.cart3d
        assert np.allclose(
            [float(np.asarray(vel[k].ustrip("km/s"))) for k in ("x", "y", "z")],
            [4.0, 5.0, 6.0],
        )

    def test_angular_rates_come_across(self):
        frame = apyc.ICRS(
            ra=90.0 * apyu.deg,
            dec=45.0 * apyu.deg,
            distance=1.0 * apyu.kpc,
            pm_ra=3.0 * apyu.mas / apyu.yr,
            pm_dec=2.0 * apyu.mas / apyu.yr,
            radial_velocity=10.0 * apyu.km / apyu.s,
            differential_type=apyc.SphericalDifferential,
        )
        vel = cx.Tangent.from_(frame)
        assert vel.chart == cxc.lonlat_sph3d
        assert np.allclose(vel["lon"].ustrip("mas/yr"), 3.0)

    def test_a_frame_without_velocity_says_so(self):
        frame = apyc.ICRS(ra=1.0 * apyu.deg, dec=2.0 * apyu.deg)
        with pytest.raises(ValueError, match="carries no velocity"):
            cx.Tangent.from_(frame)

    def test_a_skycoord_velocity_comes_across(self):
        sc = apyc.SkyCoord(
            x=1.0 * apyu.kpc,
            y=2.0 * apyu.kpc,
            z=3.0 * apyu.kpc,
            v_x=4.0 * apyu.km / apyu.s,
            v_y=5.0 * apyu.km / apyu.s,
            v_z=6.0 * apyu.km / apyu.s,
            representation_type="cartesian",
            differential_type="cartesian",
        )
        assert np.allclose(cx.Tangent.from_(sc)["x"].ustrip("km/s"), 4.0)

    def test_the_skycoord_default_proper_motion_convention_is_refused(self):
        """A `SkyCoord`'s default is the cos(lat)-scaled form, which has no chart."""
        sc = apyc.SkyCoord(
            ra=90.0 * apyu.deg,
            dec=45.0 * apyu.deg,
            distance=1.0 * apyu.kpc,
            pm_ra_cosdec=3.0 * apyu.mas / apyu.yr,
            pm_dec=2.0 * apyu.mas / apyu.yr,
            radial_velocity=10.0 * apyu.km / apyu.s,
        )
        with pytest.raises(ValueError, match="rate convention"):
            cx.Tangent.from_(sc)
