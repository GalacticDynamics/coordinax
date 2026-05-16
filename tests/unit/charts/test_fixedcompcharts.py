"""Tests for AbstractFixedComponentsChart."""

import hypothesis.strategies as st
from hypothesis import given, settings

import coordinax.charts as cxc
import coordinax.hypothesis.main as cxst

# =============================================================================
# AbstractFixedComponentsChart
# =============================================================================

_fixedchart_classes = cxst.chart_classes(
    filter=cxc.AbstractFixedComponentsChart, exclude_abstract=True
)
_fixedcharts = cxst.charts(filter=cxc.AbstractFixedComponentsChart)


class TestFixedComponentsChart:
    """Behavior of charts with fixed components."""

    @settings(deadline=None)
    @given(data=st.data(), chart_class=_fixedchart_classes)
    def test_instances_from_same_class_have_same_component_schema(
        self, data, chart_class
    ) -> None:
        """Any two instances of a fixed chart class share component schema."""
        kwargs1 = data.draw(cxst.chart_init_kwargs(chart_class))
        kwargs2 = data.draw(cxst.chart_init_kwargs(chart_class))

        chart1 = chart_class(**kwargs1)
        chart2 = chart_class(**kwargs2)

        assert chart1.components == chart2.components
        assert chart1.coord_dimensions == chart2.coord_dimensions
        assert len(chart1.components) == chart1.ndim
        assert len(chart1.coord_dimensions) == chart1.ndim

    @given(chart_class=_fixedchart_classes)
    def test_fixed_chart_classes_expose_tuple_class_schema(self, chart_class) -> None:
        """Fixed chart classes define tuple-valued class-level schema."""
        assert isinstance(chart_class._components, tuple)
        assert isinstance(chart_class._coord_dimensions, tuple)
        assert len(chart_class._components) == len(chart_class._coord_dimensions)

    @given(chart=_fixedcharts)
    def test_instances_match_class_level_schema(self, chart) -> None:
        """Instance schema equals class-level fixed schema."""
        chart_cls = type(chart)
        assert chart.components == chart_cls._components
        assert chart.coord_dimensions == chart_cls._coord_dimensions

    @given(chart=_fixedcharts)
    def test_fixed_chart_component_names_are_strings(self, chart) -> None:
        """Fixed charts expose string component names."""
        assert all(isinstance(component, str) for component in chart.components)

    @given(chart=_fixedcharts)
    def test_fixed_chart_dimension_labels_are_str_or_none(self, chart) -> None:
        """Fixed chart dimension labels are strings or None."""
        assert all(
            isinstance(dimension, str | None) for dimension in chart.coord_dimensions
        )
