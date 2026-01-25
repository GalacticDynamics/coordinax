"""Tests for the chart_classes strategy."""

from typing import final

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

import coordinax as cx
import coordinax_hypothesis as cxst
from coordinax_hypothesis._src.utils import get_all_subclasses


@given(rep_class=cxst.chart_classes())
@settings(max_examples=50)
def test_basic_chart_class(
    rep_class: type[cx.charts.AbstractChart],
) -> None:
    """Test basic chart class generation."""
    assert issubclass(rep_class, cx.charts.AbstractChart)
    assert not issubclass(rep_class, type)  # Not a metaclass


@given(rep_class=cxst.chart_classes(filter=cx.charts.Abstract3D))
@settings(max_examples=50)
def test_3d_chart_classes(
    rep_class: type[cx.charts.AbstractChart],
) -> None:
    """Test 3D chart class generation."""
    assert issubclass(rep_class, cx.charts.Abstract3D)
    assert issubclass(rep_class, cx.charts.AbstractChart)


@given(rep_class=cxst.chart_classes(filter=cx.charts.Abstract1D))
@settings(max_examples=50)
def test_1d_chart_classes(
    rep_class: type[cx.charts.AbstractChart],
) -> None:
    """Test 1D chart class generation."""
    assert issubclass(rep_class, cx.charts.Abstract1D)
    assert issubclass(rep_class, cx.charts.AbstractChart)


@given(rep_class=cxst.chart_classes(exclude_abstract=True))
@settings(max_examples=50)
def test_concrete_classes_only(rep_class: type[cx.charts.AbstractChart]) -> None:
    """Test that only concrete classes are generated when exclude_abstract=True."""
    assert issubclass(rep_class, cx.charts.AbstractChart)
    # Should be a concrete class (can be instantiated)
    assert (
        not hasattr(rep_class, "__abstractmethods__")
        or not rep_class.__abstractmethods__
    )


@given(
    rep_class=cxst.chart_classes(
        filter=st.sampled_from([cx.charts.Abstract1D, cx.charts.Abstract2D])
    )
)
@settings(max_examples=50)
def test_dynamic_filter(rep_class: type[cx.charts.AbstractChart]) -> None:
    """Test chart class generation with dynamic union class."""
    assert issubclass(rep_class, cx.charts.AbstractChart)
    assert issubclass(rep_class, (cx.charts.Abstract1D, cx.charts.Abstract2D))


def test_warning_when_no_subclasses_found() -> None:
    """Test that a warning is raised when no subclasses match the criteria."""

    # Create a fake class that nothing will inherit from
    @final
    class _FakeClass:
        pass

    # The warning is raised when get_all_subclasses finds no matching classes
    with pytest.warns(UserWarning, match="No subclasses found"):
        result = get_all_subclasses(
            cx.charts.AbstractChart,
            filter=_FakeClass,
            exclude_abstract=True,
        )

    # Should return an empty tuple
    assert result == ()
