"""Hypothesis strategies for coordinax representations."""

__all__ = (
    "can_point_transform",
    "chart_classes",
    "chart_init_kwargs",
    "build_init_kwargs_strategy",
    "charts",
    "charts_like",
    "chart_time_chain",
    "product_charts",
)

import hypothesis.strategies as st

import coordinax.charts as cxc
from .chart_kwargs import build_init_kwargs_strategy, chart_init_kwargs
from .classes import chart_classes
from .core import charts, product_charts
from .extend import chart_time_chain, charts_like
from .utils import can_point_transform
from coordinax_hypothesis._src.utils import get_all_subclasses

# Register type strategy for Hypothesis's st.from_type()
# Note: Pass the callable, not an invoked strategy
st.register_type_strategy(cxc.AbstractChart, lambda _: charts())
st.register_type_strategy(cxc.CartesianProductChart, lambda _: product_charts())

for flag_cls in get_all_subclasses(cxc.AbstractDimensionalFlag, exclude_abstract=False):
    # Skip representation base classes
    if issubclass(flag_cls, cxc.AbstractChart):
        continue

    st.register_type_strategy(flag_cls, lambda typ: charts(typ))
