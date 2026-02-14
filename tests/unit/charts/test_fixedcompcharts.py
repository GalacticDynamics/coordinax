"""Tests for chart APIs: AbstractFixedComponentsChart and cartesian_chart."""

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

import coordinax.charts as cxc
import coordinax_hypothesis.core as cxst

fixedchart_classes = cxst.chart_classes(
    filter=cxc.AbstractFixedComponentsChart, exclude_abstract=True
)
fixedcharts = cxst.charts(filter=cxc.AbstractFixedComponentsChart)


class TestFixedComponentsChart:
    """Behavior of charts with fixed components."""

    @given(data=st.data(), chart_class=fixedchart_classes)
    @settings(max_examples=50)
    def test_all_fixed_component_charts_have_consistent_components(
        self, data, chart_class
    ):
        """Property test: all chart instances have consistent components."""
        kwargs = data.draw(cxst.chart_init_kwargs(chart_class=chart_class))

        chart1 = chart_class(**kwargs)
        chart2 = chart_class(**kwargs)

        assert chart1.components == chart2.components
        assert len(chart1.components) == chart1.ndim

    def test_instances_share_components(self) -> None:
        """All instances of a chart class share the same components."""
        chart1 = cxc.Cart3D()
        chart2 = cxc.Cart3D()
        assert chart1.components is chart2.components

    @pytest.mark.parametrize(
        ("chart", "components"),
        [
            (cxc.cart3d, ("x", "y", "z")),
            (cxc.sph3d, ("r", "theta", "phi")),
            (cxc.polar2d, ("r", "theta")),
        ],
    )
    def test_excpected_components(self, chart, components) -> None:
        """Cart3D has expected components."""
        assert chart.components == components
