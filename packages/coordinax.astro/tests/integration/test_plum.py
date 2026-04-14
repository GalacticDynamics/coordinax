"""Test `coordinax.astro`."""

import plum
from hypothesis import given, settings

import quaxed.numpy as jnp
import unxt as u

import coordinax.astro as cxastro
import coordinax.distances as cxd
import coordinax.hypothesis.astro as cxastrost
import coordinax.hypothesis.distances as cxdst


@given(cxastrost.parallaxes())
def test_promotion_rule(a):
    """Test the promotion rule for angles."""
    # Quantities
    q = u.Q(1.0, "rad")

    # Explicit promotion test
    a_p, q_p = plum.promote(a, q)
    assert isinstance(a_p, u.Q)
    assert isinstance(q_p, u.Q)

    # Implicit promotion test
    assert isinstance(a * q, u.Q)
    assert isinstance(q * a, u.Q)


@given(cxastrost.parallaxes())
def test_convert_angle_to_quantity(a):
    """Test converting angle types to general quantity types.

    These conversions should be covered under rules defined in `unxt`.

    """
    q = plum.convert(a, u.Q)

    assert isinstance(q, u.Q)
    assert q.unit is a.unit
    assert q.value is a.value


@given(cxastrost.distance_moduli())
def test_convert_distance_to_quantity(d):
    """Test converting distance types to general quantity types.

    These conversions should be covered under rules defined in `unxt`.

    """
    q = plum.convert(d, u.Q)

    assert isinstance(q, u.Q)
    assert q.unit is d.unit
    assert q.value is d.value


class TestDistanceParallaxRoundtrip:
    """Tests for Distance <-> Parallax roundtrip conversions."""

    @given(
        d=cxdst.distances(unit="kpc", elements={"min_value": 0.125, "max_value": 1e4})
    )
    @settings(deadline=None)
    def test_distance_to_parallax_to_distance(self, d: cxd.Distance) -> None:
        """Distance -> Parallax -> Distance roundtrip is consistent."""
        plx = plum.convert(d, cxastro.Parallax)
        d_back = plum.convert(plx, cxd.Distance)
        # Compare in the same unit (pc)
        d_pc = u.ustrip("pc", d)
        d_back_pc = u.ustrip("pc", d_back)
        assert jnp.allclose(d_pc, d_back_pc, rtol=1e-4)

    @given(
        plx=cxastrost.parallaxes(
            unit="mas", elements={"min_value": 0.125, "max_value": 1e4}
        )
    )
    @settings(deadline=None)
    def test_parallax_to_distance_to_parallax(self, plx: cxastro.Parallax) -> None:
        """Parallax -> Distance -> Parallax roundtrip is consistent."""
        d = plum.convert(plx, cxd.Distance)
        plx_back = plum.convert(d, cxastro.Parallax)
        plx_rad = u.ustrip("rad", plx)
        plx_back_rad = u.ustrip("rad", plx_back)
        assert jnp.allclose(plx_rad, plx_back_rad, rtol=1e-4)


class TestDistanceDistanceModulusRoundtrip:
    """Tests for Distance <-> DistanceModulus roundtrip conversions."""

    @given(
        d=cxdst.distances(unit="kpc", elements={"min_value": 0.125, "max_value": 1e4})
    )
    @settings(deadline=None)
    def test_distance_to_dm_to_distance(self, d: cxd.Distance) -> None:
        """Distance -> DM -> Distance roundtrip is consistent."""
        dm = plum.convert(d, cxastro.DistanceModulus)
        d_back = plum.convert(dm, cxd.Distance)
        d_pc = u.ustrip("pc", d)
        d_back_pc = u.ustrip("pc", d_back)
        assert jnp.allclose(d_pc, d_back_pc, rtol=1e-4)

    @given(dm=cxastrost.distance_moduli(elements={"min_value": 1.0, "max_value": 25.0}))
    @settings(deadline=None)
    def test_dm_to_distance_to_dm(self, dm: cxastro.DistanceModulus) -> None:
        """DM -> Distance -> DM roundtrip is consistent."""
        d = plum.convert(dm, cxd.Distance)
        dm_back = plum.convert(d, cxastro.DistanceModulus)
        assert jnp.allclose(dm.value, dm_back.value, rtol=1e-3)


class TestParallaxDistanceModulusRoundtrip:
    """Tests for Parallax <-> DistanceModulus roundtrip conversions."""

    @given(
        plx=cxastrost.parallaxes(
            unit="mas", elements={"min_value": 0.125, "max_value": 1e4}
        )
    )
    @settings(deadline=None)
    def test_parallax_to_dm_to_parallax(self, plx: cxastro.Parallax) -> None:
        """Parallax -> DM -> Parallax roundtrip is consistent."""
        dm = plum.convert(plx, cxastro.DistanceModulus)
        plx_back = plum.convert(dm, cxastro.Parallax)
        plx_rad = u.ustrip("rad", plx)
        plx_back_rad = u.ustrip("rad", plx_back)
        assert jnp.allclose(plx_rad, plx_back_rad, rtol=1e-3)


class TestDistancePlumConvert:
    """Tests for plum.convert with Distance."""

    @given(
        d=cxdst.distances(unit="kpc", elements={"min_value": 0.125, "max_value": 1e6})
    )
    @settings(deadline=None)
    def test_convert_to_distance_modulus(self, d: cxd.Distance) -> None:
        """Can convert Distance to DistanceModulus."""
        dm = plum.convert(d, cxastro.DistanceModulus)
        assert isinstance(dm, cxastro.DistanceModulus)
        assert dm.unit == u.unit("mag")

    @given(
        d=cxdst.distances(unit="kpc", elements={"min_value": 0.125, "max_value": 1e6})
    )
    @settings(deadline=None)
    def test_convert_to_parallax(self, d: cxd.Distance) -> None:
        """Can convert Distance to Parallax."""
        plx = plum.convert(d, cxastro.Parallax)
        assert isinstance(plx, cxastro.Parallax)
