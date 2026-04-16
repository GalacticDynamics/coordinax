"""Hypothesis strategies for coordinax representations."""

import hypothesis.strategies as st

import coordinax.charts as cxc

from .cdict import *
from .chart_kwargs import *
from .charts import *
from .charts import charts
from .charts_product import *
from .classes import *
from .extend import *
from .utils import *
from coordinax.hypothesis.utils import get_all_subclasses

# Register type strategy for Hypothesis's st.from_type()
# Note: Pass the callable, not an invoked strategy
st.register_type_strategy(cxc.AbstractChart, lambda _: charts())  # ty: ignore[missing-argument]
st.register_type_strategy(
    cxc.CartesianProductChart,
    lambda _: charts(cxc.CartesianProductChart),  # ty: ignore[missing-argument]
)

for flag_cls in get_all_subclasses(cxc.AbstractDimensionalFlag, exclude_abstract=False):
    # Skip representation base classes
    if issubclass(flag_cls, cxc.AbstractChart):
        continue

    st.register_type_strategy(flag_cls, lambda typ: charts(typ))  # ty: ignore[missing-argument]
