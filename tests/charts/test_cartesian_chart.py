"""Tests for chart APIs: AbstractFixedComponentsChart and cartesian_chart."""

import plum
import pytest
from hypothesis import given, settings

import coordinax.charts as cxc
import coordinax_hypothesis as cxst


class TestCartesianChartFunction:
    """Tests for cartesian_chart function."""

    @pytest.mark.parametrize(
        ("chart", "expected_cartesian"),
        [
            (cxc.cart0d, cxc.cart0d),
            (cxc.cart1d, cxc.cart1d),
            (cxc.radial1d, cxc.cart1d),
            (cxc.cart2d, cxc.cart2d),
            (cxc.polar2d, cxc.cart2d),
            (cxc.cart3d, cxc.cart3d),
            (cxc.sph3d, cxc.cart3d),
            (cxc.lonlatsph3d, cxc.cart3d),
            (cxc.loncoslatsph3d, cxc.cart3d),
            (cxc.cyl3d, cxc.cart3d),
            (cxc.cartnd, cxc.cartnd),
        ],
    )
    def test_cartesian_chart_examples(self, chart, expected_cartesian):
        """Test that cartesian_chart returns the expected cartesian chart."""
        assert cxc.cartesian_chart(chart) == expected_cartesian

    @given(chart=cxst.charts())
    @settings(max_examples=50)
    def test_cartesian_chart_idempotent(self, chart):
        """Property test: cartesian_chart is idempotent."""
        try:
            cart1 = cxc.cartesian_chart(chart)
            cart2 = cxc.cartesian_chart(cart1)
        except (NotImplementedError, plum.NotFoundLookupError):
            # Some charts may not have a cartesian_chart implementation yet
            # (raises NotImplementedError or NotFoundLookupError from plum)
            pass
        else:
            # Applying cartesian_chart again should give same result
            assert cart1 == cart2
