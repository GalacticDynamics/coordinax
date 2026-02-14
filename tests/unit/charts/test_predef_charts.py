"""Tests for chart APIs: AbstractFixedComponentsChart and cartesian_chart."""

import coordinax.charts as cxc


def test_predefined_charts_have_expected_components():
    """Components remain consistent across instances."""
    for name in cxc.__dir__():
        chart = getattr(cxc, name)

        # TODO: also work with non-fixed charts
        if not isinstance(chart, cxc.AbstractFixedComponentsChart):
            continue

        assert chart.components == type(chart)._components
