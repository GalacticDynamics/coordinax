"""Tests for chart APIs: AbstractFixedComponentsChart and cartesian_chart."""

import hypothesis.strategies as st
from hypothesis import given, settings

import coordinax.charts as cxc
import coordinax_hypothesis as cxst

fixedchart_classes = cxst.chart_classes(
    filter=cxc.AbstractFixedComponentsChart, exclude_abstract=True
)
fixedcharts = cxst.charts(filter=cxc.AbstractFixedComponentsChart)


class TestAbstractFixedComponentsChart:
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
