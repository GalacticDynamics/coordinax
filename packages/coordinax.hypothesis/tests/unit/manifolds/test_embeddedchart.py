"""Tests for charts() dispatch on explicit chart-class inputs."""

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings
from hypothesis.errors import Unsatisfiable

import coordinax.charts as cxc
import coordinax.manifolds as cxm

import coordinax.hypothesis.main as cxst


@given(chart=cxst.charts(cxm.EmbeddedChart))
def test_embedded_chart_dispatch_generates_embedded_chart(
    chart: cxc.AbstractChart,
) -> None:
    """EmbeddedChart overload generates embedded 2-D chart instances."""
    assert isinstance(chart, cxm.EmbeddedChart)
    assert isinstance(chart.embed_map, cxm.TwoSphereIn3D)
    assert chart.ndim == 2


@given(chart=cxst.charts(cxm.EmbeddedChart, ndim=2))
def test_embedded_chart_dispatch_accepts_matching_ndim(
    chart: cxc.AbstractChart,
) -> None:
    """EmbeddedChart overload accepts ndim=2."""
    assert isinstance(chart, cxm.EmbeddedChart)
    assert chart.ndim == 2


def test_embedded_chart_dispatch_rejects_non_matching_ndim() -> None:
    """EmbeddedChart overload is unsatisfiable for ndim values other than 2."""

    @given(chart=cxst.charts(cxm.EmbeddedChart, ndim=3))
    @settings(max_examples=10)
    def impossible(chart: cxc.AbstractChart) -> None:
        del chart

    with pytest.raises(Unsatisfiable):
        impossible()


@given(data=st.data())
def test_embedded_chart_dispatch_rejects_filter_and_exclude(
    data: st.DataObject,
) -> None:
    """Explicit EmbeddedChart dispatch rejects filter/exclude kwargs."""
    with pytest.raises(ValueError, match="filter and exclude"):
        data.draw(cxst.charts(cxm.EmbeddedChart, filter=cxc.Abstract2D))
