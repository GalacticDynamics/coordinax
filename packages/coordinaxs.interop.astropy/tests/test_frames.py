"""Frame interop: the guards and the shape of ``galcen_coord``.

Two defects, both in ``coordinaxs.interop.astropy._src.frames``: `plum.convert`
used to return a bare frame for an astropy frame carrying data, dropping it
silently, while the paired ``from_`` raised; and the cx -> astropy
Galactocentric direction attached a distance to ``galcen_coord``, which astropy
itself cannot compare or transform.
"""

__all__: tuple[str, ...] = ()

import astropy.coordinates as apyc
import astropy.units as apyu
import numpy as np
import plum
import pytest

import coordinaxs.astro as cxastro


@pytest.mark.parametrize(
    ("apy_frame", "cx_frame_cls"),
    [
        (
            apyc.ICRS(ra=10 * apyu.deg, dec=20 * apyu.deg, distance=1 * apyu.kpc),
            cxastro.ICRS,
        ),
        (apyc.Galactic(l=1 * apyu.deg, b=2 * apyu.deg), cxastro.Galactic),
        (
            apyc.Galactocentric(x=1 * apyu.kpc, y=2 * apyu.kpc, z=3 * apyu.kpc),
            cxastro.Galactocentric,
        ),
    ],
    ids=["icrs", "galactic", "galactocentric"],
)
def test_convert_refuses_a_frame_carrying_data(apy_frame, cx_frame_cls):
    """`plum.convert` refuses data rather than dropping it, as ``from_`` does."""
    with pytest.raises(ValueError, match="must not have data"):
        plum.convert(apy_frame, cx_frame_cls)

    with pytest.raises(ValueError, match="must not have data"):
        cx_frame_cls.from_(apy_frame)


class TestGalactocentricGalcenCoord:
    """``galcen_coord`` is a direction; the distance is ``galcen_distance``."""

    @staticmethod
    def round_trip(apy_frame):
        return plum.convert(
            cxastro.Galactocentric.from_(apy_frame), apyc.Galactocentric
        )

    def test_galcen_coord_is_unit_spherical(self):
        a1 = self.round_trip(apyc.Galactocentric())
        assert isinstance(a1.galcen_coord.data, apyc.UnitSphericalRepresentation)

    def test_the_round_tripped_frame_is_equivalent_to_astropys_own(self):
        """A `SphericalRepresentation` here made this a `TypeError`."""
        a0 = apyc.Galactocentric()
        assert a0.is_equivalent_frame(self.round_trip(a0))

    def test_the_transform_is_unchanged(self):
        """Dropping the distance off ``galcen_coord`` does not move the answer.

        The bound is headroom, not observed error: on this machine the two
        transforms agree bit-for-bit. 1e-6 pc is ~1e6x above the double-
        precision floor for a ~9 kpc vector (~2e-12 pc), so a libm or Astropy
        version difference cannot trip it -- while still far below any real
        regression: stripping ``galcen_distance`` by mistake (rather than the
        distance off ``galcen_coord``) moves the point by ~7 kpc, and the
        bound catches a ``galcen_coord`` direction wrong by a quarter of a
        milliarcsecond.

        ``separation_3d`` compares directly rather than transforming only
        because the two frames are equivalent -- which the test above pins.
        """
        atol_pc = 1e-6
        a0 = apyc.Galactocentric()
        sc = apyc.SkyCoord(ra=90 * apyu.deg, dec=45 * apyu.deg, distance=1 * apyu.kpc)
        g0 = sc.transform_to(a0)
        g1 = sc.transform_to(self.round_trip(a0))
        assert np.allclose(
            g0.cartesian.xyz.to_value("pc"),
            g1.cartesian.xyz.to_value("pc"),
            rtol=0.0,
            atol=atol_pc,
        )
        assert g0.separation_3d(g1).to_value("pc") < atol_pc

    def test_the_frame_parameters_survive_the_round_trip(self):
        """`roll` comes back a float rather than a weak int, hence `allclose`."""
        a0 = apyc.Galactocentric(
            galcen_coord=apyc.ICRS(ra=1 * apyu.deg, dec=2 * apyu.deg),
            galcen_distance=8.0 * apyu.kpc,
            z_sun=15.0 * apyu.pc,
            roll=3 * apyu.deg,
        )
        a1 = self.round_trip(a0)
        assert np.allclose(a1.galcen_coord.ra.to_value("deg"), 1.0)
        assert np.allclose(a1.galcen_coord.dec.to_value("deg"), 2.0)
        assert np.allclose(a1.galcen_distance.to_value("kpc"), 8.0)
        assert np.allclose(a1.z_sun.to_value("pc"), 15.0)
        assert np.allclose(a1.roll.to_value("deg"), 3.0)
