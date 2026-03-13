"""Tests for coordinax-hypothesis strategies."""

from hypothesis import given

import unxt as u

import coordinax.charts as cxc
import coordinax.hypothesis.main as cxst


class TestCDictStrategy:
    """Test pdict strategy for generating valid CDict objects."""

    @given(p=cxst.cdicts(cxc.cart3d))
    def test_cdict_keys_match_chart(self, p):
        """CDict keys must exactly match chart.components."""
        assert set(p.keys()) == set(cxc.cart3d.components)

    @given(p=cxst.cdicts(cxc.cart3d))
    def test_cdict_all_quantities(self, p):
        """All values must be quantity-like."""
        for v in p.values():
            assert hasattr(v, "unit") or isinstance(v, (int, float))

    @given(p=cxst.cdicts(cxc.sph3d))
    def test_cdict_mixed_dimensions(self, p):
        """Point role allows mixed dimensions from chart.coord_dimensions."""
        # Spherical has (length, angle, angle)
        assert set(p.keys()) == {"r", "theta", "phi"}
        assert u.dimension_of(p["r"]) == u.dimension("length")
        assert u.dimension_of(p["theta"]) == u.dimension("angle")
        assert u.dimension_of(p["phi"]) == u.dimension("angle")

    @given(p=cxst.cdicts(cxst.charts(filter=cxc.Abstract3D)))
    def test_cdict_with_chart_strategy(self, p):
        """Cdicts accepts chart as a strategy, drawing chart then building CDict."""
        # All 3D charts have exactly 3 components
        assert len(p) == 3
        # All keys should be strings
        assert all(isinstance(k, str) for k in p)
        # All values should be quantity-like
        assert all(hasattr(v, "unit") for v in p.values())
