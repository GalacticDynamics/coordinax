"""Unit tests for cross-type distance conversions using hypothesis strategies."""

import jax.numpy as jnp
from hypothesis import given, settings

import unxt as u

import coordinax.distances as cxd
import coordinax_hypothesis.core as cxst


class TestDistanceParallaxRoundtrip:
    """Tests for Distance <-> Parallax roundtrip conversions."""

    @given(
        d=cxst.distances(unit="kpc", elements={"min_value": 0.125, "max_value": 1e4})
    )
    @settings(deadline=None)
    def test_distance_to_parallax_to_distance(self, d: cxd.Distance) -> None:
        """Distance -> Parallax -> Distance roundtrip is consistent."""
        plx = d.parallax
        d_back = plx.distance
        # Compare in the same unit (pc)
        d_pc = u.ustrip("pc", d)
        d_back_pc = u.ustrip("pc", d_back)
        assert jnp.allclose(d_pc, d_back_pc, rtol=1e-4)

    @given(
        plx=cxst.parallaxes(unit="mas", elements={"min_value": 0.125, "max_value": 1e4})
    )
    @settings(deadline=None)
    def test_parallax_to_distance_to_parallax(self, plx: cxd.Parallax) -> None:
        """Parallax -> Distance -> Parallax roundtrip is consistent."""
        d = plx.distance
        plx_back = d.parallax
        plx_rad = u.ustrip("rad", plx)
        plx_back_rad = u.ustrip("rad", plx_back)
        assert jnp.allclose(plx_rad, plx_back_rad, rtol=1e-4)


class TestDistanceDistanceModulusRoundtrip:
    """Tests for Distance <-> DistanceModulus roundtrip conversions."""

    @given(
        d=cxst.distances(unit="kpc", elements={"min_value": 0.125, "max_value": 1e4})
    )
    @settings(deadline=None)
    def test_distance_to_dm_to_distance(self, d: cxd.Distance) -> None:
        """Distance -> DM -> Distance roundtrip is consistent."""
        dm = d.distance_modulus
        d_back = dm.distance
        d_pc = u.ustrip("pc", d)
        d_back_pc = u.ustrip("pc", d_back)
        assert jnp.allclose(d_pc, d_back_pc, rtol=1e-4)

    @given(dm=cxst.distance_moduli(elements={"min_value": 1.0, "max_value": 25.0}))
    @settings(deadline=None)
    def test_dm_to_distance_to_dm(self, dm: cxd.DistanceModulus) -> None:
        """DM -> Distance -> DM roundtrip is consistent."""
        d = dm.distance
        dm_back = d.distance_modulus
        assert jnp.allclose(dm.value, dm_back.value, rtol=1e-3)


class TestParallaxDistanceModulusRoundtrip:
    """Tests for Parallax <-> DistanceModulus roundtrip conversions."""

    @given(
        plx=cxst.parallaxes(unit="mas", elements={"min_value": 0.125, "max_value": 1e4})
    )
    @settings(deadline=None)
    def test_parallax_to_dm_to_parallax(self, plx: cxd.Parallax) -> None:
        """Parallax -> DM -> Parallax roundtrip is consistent."""
        dm = plx.distance_modulus
        plx_back = dm.parallax
        plx_rad = u.ustrip("rad", plx)
        plx_back_rad = u.ustrip("rad", plx_back)
        assert jnp.allclose(plx_rad, plx_back_rad, rtol=1e-3)
