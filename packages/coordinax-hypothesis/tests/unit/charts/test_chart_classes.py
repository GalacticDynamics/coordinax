"""Tests for the chart_classes strategy."""

from typing import final

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

import coordinax.charts as cxc
import coordinax_hypothesis.core as cxst
from coordinax_hypothesis.utils import get_all_subclasses


@given(chart_cls=cxst.chart_classes())
# @settings(max_examples=50)
def test_basic_chart_class(chart_cls: type[cxc.AbstractChart]) -> None:
    """Test basic chart class generation."""
    # Verify it's a subclass of AbstractChart
    assert issubclass(chart_cls, cxc.AbstractChart)

    # Verify it's not AbstractChart itself
    assert chart_cls is not cxc.AbstractChart

    # Verify it's not a metaclass
    assert not issubclass(chart_cls, type)  # Not a metaclass


@given(
    data=st.data(),
    filter=st.sampled_from(
        [cxc.Abstract1D, cxc.Abstract2D, cxc.Abstract3D, cxc.AbstractSpherical3D]
    ),
)
def test_chart_class_filter(data, filter) -> None:
    """Test chart class generation with filter."""
    chart_cls = data.draw(cxst.chart_classes(filter=filter))

    assert issubclass(chart_cls, filter)
    assert issubclass(chart_cls, cxc.AbstractChart)


@given(
    data=st.data(),
    exclude=st.sampled_from([(cxc.Cart3D,), (cxc.Cart3D, cxc.Spherical3D)]),
)
@settings(max_examples=50)
def test_chart_class_filter_exclude(data, exclude) -> None:
    """Test that a specific class is excluded."""
    chart_cls = data.draw(cxst.chart_classes(exclude=exclude))
    for excluded_cls in exclude:
        assert chart_cls is not excluded_cls


@given(chart_cls=cxst.chart_classes(exclude_abstract=True))
@settings(max_examples=50)
def test_concrete_classes_only(chart_cls: type[cxc.AbstractChart]) -> None:
    """Test that only concrete classes are generated when exclude_abstract=True."""
    assert issubclass(chart_cls, cxc.AbstractChart)
    assert (
        not hasattr(chart_cls, "__abstractmethods__")
        or not chart_cls.__abstractmethods__
    )


# ============================================================
# Combining filter + exclude


@given(
    chart_cls=cxst.chart_classes(filter=cxc.Abstract3D, exclude=(cxc.Cart3D,)),
)
@settings(max_examples=50)
def test_filter_with_exclude(chart_cls: type[cxc.AbstractChart]) -> None:
    """Test filter and exclude combined."""
    assert issubclass(chart_cls, cxc.Abstract3D)
    assert chart_cls is not cxc.Cart3D


# ============================================================
# Dynamic / strategy-valued parameters


@given(
    chart_cls=cxst.chart_classes(
        filter=st.sampled_from([cxc.Abstract1D, cxc.Abstract2D])
    )
)
@settings(max_examples=50)
def test_dynamic_filter(chart_cls: type[cxc.AbstractChart]) -> None:
    """Test chart class generation with dynamic filter strategy."""
    assert issubclass(chart_cls, cxc.AbstractChart)
    assert issubclass(chart_cls, (cxc.Abstract1D, cxc.Abstract2D))


@given(chart_cls=cxst.chart_classes(exclude_abstract=st.sampled_from([True, False])))
@settings(max_examples=50)
def test_dynamic_exclude_abstract(chart_cls: type[cxc.AbstractChart]) -> None:
    """Test chart class generation with dynamic exclude_abstract strategy."""
    assert issubclass(chart_cls, cxc.AbstractChart)


# ============================================================
# Edge cases


def test_warning_when_no_subclasses_found() -> None:
    """Test that a warning is raised when no subclasses match the criteria."""

    # Create a fake class that nothing will inherit from
    @final
    class _FakeClass:
        pass

    # The warning is raised when get_all_subclasses finds no matching classes
    with pytest.warns(UserWarning, match="No subclasses found"):
        result = get_all_subclasses(
            cxc.AbstractChart, filter=_FakeClass, exclude_abstract=True
        )

    # Should return an empty tuple
    assert result == ()
