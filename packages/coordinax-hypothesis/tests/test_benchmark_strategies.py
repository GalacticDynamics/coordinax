"""Benchmark tests for coordinax-hypothesis strategies."""

from hypothesis import given, settings
from hypothesis.strategies import data as st_data

import coordinax as cx
import coordinax_hypothesis as cxst
from coordinax_hypothesis._src.utils import get_all_subclasses

# =============================================================================
# Benchmark: chart_classes strategy


def test_benchmark_chart_classes_simple(benchmark):
    """Benchmark drawing a representation class."""

    @given(d=st_data())
    @settings(max_examples=1, deadline=None)
    def draw_one(d):
        return d.draw(cxst.chart_classes())

    # Use data() to draw from strategy
    def run_draw():
        result = []

        @given(d=st_data())
        @settings(max_examples=1, deadline=None)
        def inner(d):
            result.append(d.draw(cxst.chart_classes()))

        inner()
        return result[0]

    result = benchmark(run_draw)
    assert issubclass(result, cx.charts.AbstractChart)


# =============================================================================
# Benchmark: representations strategy - using direct strategy building


def test_benchmark_representations_build_strategy(benchmark):
    """Benchmark building the representations strategy."""

    def build_strategy():
        return cxst.charts()

    strategy = benchmark(build_strategy)
    assert strategy is not None


def test_benchmark_representations_1d_build(benchmark):
    """Benchmark building the representations strategy for 1D."""

    def build_strategy():
        return cxst.charts(dimensionality=1)

    strategy = benchmark(build_strategy)
    assert strategy is not None


def test_benchmark_representations_3d_build(benchmark):
    """Benchmark building the representations strategy for 3D."""

    def build_strategy():
        return cxst.charts(dimensionality=3)

    strategy = benchmark(build_strategy)
    assert strategy is not None


# =============================================================================
# Benchmark: get_all_subclasses (the likely bottleneck)


def test_benchmark_get_all_subclasses(benchmark):
    """Benchmark the get_all_subclasses function."""

    def run_get_subclasses():
        return get_all_subclasses(
            cx.charts.AbstractChart,
            filter=object,
            exclude_abstract=True,
            exclude=(),
        )

    result = benchmark(run_get_subclasses)
    assert len(result) > 0


def test_benchmark_get_all_subclasses_filtered(benchmark):
    """Benchmark get_all_subclasses with filter."""

    def run_get_subclasses():
        return get_all_subclasses(
            cx.charts.AbstractChart,
            filter=cx.charts.Abstract3D,
            exclude_abstract=True,
            exclude=(),
        )

    result = benchmark(run_get_subclasses)
    assert len(result) > 0


# =============================================================================
# Benchmark: build_init_kwargs_strategy


def test_benchmark_build_init_kwargs_cart3d(benchmark):
    """Benchmark building init kwargs strategy for Cart3D."""

    def build_kwargs():
        return cxst.build_init_kwargs_strategy(cx.charts.Cart3D, dim=3)

    strategy = benchmark(build_kwargs)
    assert strategy is not None


def test_benchmark_build_init_kwargs_spacetimect(benchmark):
    """Benchmark building init kwargs strategy for SpaceTimeCT (recursive case)."""

    def build_kwargs():
        return cxst.build_init_kwargs_strategy(cx.charts.SpaceTimeCT, dim=4)

    strategy = benchmark(build_kwargs)
    assert strategy is not None
