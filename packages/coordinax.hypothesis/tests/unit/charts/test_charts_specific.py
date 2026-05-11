"""Tests for charts() dispatch on explicit chart-class inputs."""

import hypothesis.strategies as st
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis.errors import Unsatisfiable

import coordinax.charts as cxc

import coordinax.hypothesis.main as cxst


@given(chart=cxst.charts(cxc.Cart3D))
def test_concrete_chart_class_generates_requested_class(
    chart: cxc.AbstractChart,
) -> None:
    """Passing a concrete chart class generates instances of that class."""
    assert isinstance(chart, cxc.Cart3D)


@given(chart=cxst.charts(cxc.Cart2D, ndim=2))
def test_concrete_chart_class_accepts_matching_ndim(chart: cxc.AbstractChart) -> None:
    """Passing a concrete chart class accepts a compatible ndim constraint."""
    assert isinstance(chart, cxc.Cart2D)
    assert chart.ndim == 2


@given(chart=cxst.charts(cxc.AbstractCartesianProductChart))
def test_abstract_chart_class_filters_generated_classes(
    chart: cxc.AbstractChart,
) -> None:
    """Passing an abstract chart class acts as a filter over matching charts."""
    assert isinstance(chart, cxc.AbstractCartesianProductChart)


@given(chart=cxst.charts(filter=cxc.Abstract3D))
def test_filter_keyword_still_handles_dimensional_flags(
    chart: cxc.AbstractChart,
) -> None:
    """Dimensional flag classes remain valid via the filter keyword."""
    assert isinstance(chart, cxc.Abstract3D)
    assert chart.ndim == 3


@given(chart=cxst.charts(st.sampled_from((cxc.Cart1D, cxc.Cart2D))))
def test_chart_class_strategy_dispatches_to_drawn_class(
    chart: cxc.AbstractChart,
) -> None:
    """Passing a chart-class strategy redispatches to the selected class."""
    assert isinstance(chart, (cxc.Cart1D, cxc.Cart2D))


def test_concrete_chart_with_impossible_ndim_is_unsatisfiable() -> None:
    """Concrete class requests remain unsatisfiable when ndim cannot match."""

    @given(chart=cxst.charts(cxc.Cart1D, ndim=2))
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def impossible(chart: cxc.AbstractChart) -> None:
        del chart

    with pytest.raises(Unsatisfiable):
        impossible()
