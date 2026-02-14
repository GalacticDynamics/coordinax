"""Tests for the build_init_kwargs_strategy function."""

import hypothesis.strategies as st
from hypothesis import given

import coordinax.charts as cxc
import coordinax_hypothesis.core as cxst
from coordinax_hypothesis.charts import build_init_kwargs_strategy


@given(data=st.data(), chart_class=cxst.chart_classes())
def test_build_init_kwargs_strategy_exists(data, chart_class) -> None:
    """Test build_init_kwargs_strategy returns a strategy that generates dicts.

    This test verifies that the function produces a valid Hypothesis strategy
    for any concrete chart class, and that drawing from it yields a dict.
    """
    strategy = build_init_kwargs_strategy(chart_class, dim=None)
    assert isinstance(strategy, st.SearchStrategy)
    kwargs = data.draw(strategy)
    assert isinstance(kwargs, dict)


@given(data=st.data(), chart_class=cxst.chart_classes())
def test_build_init_kwargs_strategy_instantiates(data, chart_class) -> None:
    """Test build_init_kwargs_strategy produces kwargs that instantiate charts."""
    strategy = build_init_kwargs_strategy(chart_class, dim=None)
    kwargs = data.draw(strategy)
    chart = chart_class(**kwargs)
    assert isinstance(chart, cxc.AbstractChart)
    assert isinstance(chart, chart_class)
