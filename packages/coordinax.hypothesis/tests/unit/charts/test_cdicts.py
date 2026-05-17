"""Tests for coordinax-hypothesis strategies."""

import unxt as u
from hypothesis import given, strategies as st

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


class TestCDictValueControl:
    """Tests showing how to control the values (e.g. quadrant) of generated CDicts."""

    @given(
        p=cxst.cdicts(
            cxc.cart3d, elements=st.floats(min_value=1, max_value=100, width=32)
        )
    )
    def test_first_octant_via_elements(self, p):
        """elements= constrains all components to positive values (first octant)."""
        assert float(p["x"].value) > 0
        assert float(p["y"].value) > 0
        assert float(p["z"].value) > 0

    @given(
        p=cxst.cdicts(
            cxc.cart3d, elements=st.floats(min_value=-100, max_value=-1, width=32)
        )
    )
    def test_negative_octant_via_elements(self, p):
        """elements= constrains all Cartesian components to negative values."""
        assert float(p["x"].value) < 0
        assert float(p["y"].value) < 0
        assert float(p["z"].value) < 0

    @given(
        p=cxst.cdicts(
            cxc.cart2d,
            elements=st.floats(min_value=-10, max_value=10, allow_nan=False, width=32),
        )
    )
    def test_bounded_range(self, p):
        """elements= with explicit bounds keeps all component magnitudes in range."""
        for key in ("x", "y"):
            val = float(p[key].value)
            assert -10 <= val <= 10

    @given(data=st.data())
    def test_second_quadrant_per_component(self, data):
        """Use st.data() to draw different element ranges per component.

        Second quadrant in 2D: x < 0, y > 0.
        """
        p_x = data.draw(
            cxst.cdicts(
                cxc.cart2d,
                elements=st.floats(min_value=-100, max_value=-1, width=32),
            )
        )
        p_y = data.draw(
            cxst.cdicts(
                cxc.cart2d, elements=st.floats(min_value=1, max_value=100, width=32)
            )
        )

        assert float(p_x["x"].value) < 0
        assert float(p_y["y"].value) > 0

    @given(
        p=cxst.cdicts(
            cxc.sph3d, elements=st.floats(min_value=1, max_value=100, width=32)
        )
    )
    def test_spherical_positive_elements(self, p):
        """elements= applies to all component values including angles in a sphere chart."""
        # r, theta, phi all have positive values when elements is positive
        assert float(p["r"].value) > 0
        assert float(p["theta"].value) > 0
        assert float(p["phi"].value) > 0
