"""Astropy's representations that are not three-dimensional.

A sky position with no distance (`UnitSphericalRepresentation`) and a distance
with no direction (`RadialRepresentation`) live on the two-sphere and on the
line, not in R^3. `coordinax.charts.guess_chart` has always named the charts
for them; these check that a point can actually be carried across, both ways.
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


class TestSkyPositionWithoutDistance:
    """The commonest astropy coordinate of all: `ra`/`dec` and nothing else."""

    def test_a_skycoord_becomes_a_two_sphere_point(self):
        sc = apyc.SkyCoord(ra=90.0 * apyu.deg, dec=45.0 * apyu.deg)
        point = cx.Point.from_(sc)
        assert point.chart == cxc.lonlat_sph2
        assert sorted(point.data) == ["lat", "lon"]
        assert np.allclose(point["lon"].ustrip("deg"), 90.0)

    def test_the_frame_comes_across_too(self):
        point = cx.Point.from_(apyc.ICRS(ra=1.0 * apyu.deg, dec=2.0 * apyu.deg))
        assert point.chart == cxc.lonlat_sph2
        assert plum.convert(point.frame, apyc.BaseCoordinateFrame).name == "icrs"

    def test_the_representation_round_trips(self):
        rep = apyc.UnitSphericalRepresentation(lon=2.0 * apyu.deg, lat=3.0 * apyu.deg)
        back = plum.convert(plum.convert(rep, cx.Point), apyc.BaseRepresentation)
        assert isinstance(back, apyc.UnitSphericalRepresentation)
        assert np.allclose(back.lon.to_value("deg"), 2.0)
        assert np.allclose(back.lat.to_value("deg"), 3.0)

    def test_dropping_a_distance_is_not_a_change_of_chart(self):
        """A 3D point is refused rather than silently projected."""
        point = cx.Point.from_(
            {
                "lon": u.Q(1.0, "deg"),
                "lat": u.Q(2.0, "deg"),
                "distance": u.Q(3.0, "km"),
            },
            cxc.lonlat_sph3d,
        )
        with pytest.raises(ValueError, match="two-sphere chart"):
            plum.convert(point, apyc.UnitSphericalRepresentation)


class TestDistanceWithoutDirection:
    """`RadialRepresentation` <-> `coordinax.charts.radial1d`."""

    def test_the_component_is_renamed(self):
        """Astropy spells it ``distance``; `radial1d` spells it ``r``."""
        data = cxc.cdict(apyc.RadialRepresentation(distance=1.0 * apyu.kpc))
        assert list(data) == ["r"]
        assert np.allclose(data["r"].ustrip("kpc"), 1.0)

    def test_a_representation_round_trips(self):
        rep = apyc.RadialRepresentation(distance=1.0 * apyu.kpc)
        point = cx.Point.from_(rep)
        assert point.chart == cxc.radial1d
        back = plum.convert(point, apyc.BaseRepresentation)
        assert isinstance(back, apyc.RadialRepresentation)
        assert np.allclose(back.distance.to_value("kpc"), 1.0)

    def test_keeping_only_the_radius_is_not_a_change_of_chart(self):
        point = cx.Point.from_([1.0, 2.0, 3.0], "km")
        with pytest.raises(ValueError, match="radial chart"):
            plum.convert(point, apyc.RadialRepresentation)
