"""Tests for chart-specific strategy registration and lookup."""

from collections.abc import Callable

import hypothesis.strategies as st
import pytest
from hypothesis import assume, given, settings
from hypothesis.errors import Unsatisfiable

import coordinax.charts as cxc
from coordinax.hypothesis.charts._src.charts_specific import (
    CHART_STRATEGIES,
    any_charts,
    find_chart_strategy,
    register_chart_strategy,
)


def test_chart_strategies_mapping_contract() -> None:
    """CHART_STRATEGIES is a mapping from chart classes to strategy callables."""
    assert isinstance(CHART_STRATEGIES, dict)

    for key, value in CHART_STRATEGIES.items():
        assert isinstance(key, type)
        assert issubclass(key, cxc.AbstractChart)
        assert isinstance(value, Callable)


def test_find_chart_strategy_defaults_to_any_charts() -> None:
    """Unregistered chart classes fall back to any_charts strategy."""
    find_chart_strategy.cache_clear()

    strategy = find_chart_strategy(cxc.Cart3D)

    assert strategy is any_charts


def test_find_chart_strategy_is_cached() -> None:
    """Repeated lookup for the same class uses functools cache."""
    find_chart_strategy.cache_clear()

    first = find_chart_strategy(cxc.Cart2D)
    second = find_chart_strategy(cxc.Cart2D)

    assert first is second


def test_register_chart_strategy_returns_original_function() -> None:
    """Decorator does not wrap or replace the original strategy function."""

    class LocalChart(cxc.Cart3D):
        pass

    @st.composite
    def local_chart_strategy(
        draw: st.DrawFn,
        chart_cls: type[LocalChart],
        /,
        *,
        ndim: int | None = None,
    ):
        draw(st.just(None))
        chart = chart_cls()
        if ndim is not None:
            assume(chart.ndim == ndim)
        return chart

    decorator = register_chart_strategy(LocalChart)
    returned = decorator(local_chart_strategy)

    assert returned is local_chart_strategy


def test_register_chart_strategy_affects_find_chart_strategy() -> None:
    """Registered strategy is resolved for the target chart class."""

    class LocalChart(cxc.Cart3D):
        pass

    @register_chart_strategy(LocalChart)
    @st.composite
    def local_chart_strategy(
        draw: st.DrawFn,
        chart_cls: type[LocalChart],
        /,
        *,
        ndim: int | None = None,
    ):
        draw(st.just(None))
        chart = chart_cls()
        if ndim is not None:
            assume(chart.ndim == ndim)
        return chart

    find_chart_strategy.cache_clear()
    resolved = find_chart_strategy(LocalChart)

    assert resolved is local_chart_strategy


@given(chart=any_charts(cxc.Cart3D))
@settings(max_examples=25)
def test_any_charts_generates_requested_chart_class(chart: cxc.AbstractChart) -> None:
    """any_charts creates instances of the requested chart class."""
    assert isinstance(chart, cxc.Cart3D)


@given(chart=any_charts(cxc.Cart2D, ndim=2))
@settings(max_examples=25)
def test_any_charts_respects_matching_ndim(chart: cxc.AbstractChart) -> None:
    """any_charts accepts examples matching the requested ndim."""
    assert isinstance(chart, cxc.Cart2D)
    assert chart.ndim == 2


def test_any_charts_with_impossible_ndim_is_unsatisfiable() -> None:
    """any_charts rejects all examples when ndim cannot match the class."""

    @given(chart=any_charts(cxc.Cart1D, ndim=2))
    @settings(max_examples=10)
    def impossible(chart: cxc.AbstractChart) -> None:
        del chart

    with pytest.raises(Unsatisfiable):
        impossible()
