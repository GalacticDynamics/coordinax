"""Hypothesis strategies for coordinax representations."""

__all__: tuple[str, ...] = ()

import hypothesis.strategies as st

import coordinax.charts as cxc
from .cdict import *
from .chart_kwargs import *
from .charts import *
from .charts import charts
from .charts_product import *
from .charts_product import product_charts
from .charts_specific import *
from .classes import *
from .extend import *
from .utils import *
from coordinax.hypothesis.utils import get_all_subclasses

# Register type strategy for Hypothesis's st.from_type()
# Note: Pass the callable, not an invoked strategy
st.register_type_strategy(cxc.AbstractChart, lambda _: charts())
st.register_type_strategy(cxc.CartesianProductChart, lambda _: product_charts())

for flag_cls in get_all_subclasses(cxc.AbstractDimensionalFlag, exclude_abstract=False):
    # Skip representation base classes
    if issubclass(flag_cls, cxc.AbstractChart):
        continue

    st.register_type_strategy(flag_cls, lambda typ: charts(typ))
