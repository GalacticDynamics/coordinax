"""Hypothesis strategies for coordinax representations."""

from importlib.metadata import entry_points

from typing import Final

import hypothesis.strategies as st

import coordinax.charts as cxc
from coordinax._src.optional_exports import load_exports

from .cdict import *
from .chart_kwargs import *
from .charts import *
from .charts import charts
from .charts_product import *
from .classes import *
from .domains import *
from .extend import *
from .utils import *
from coordinaxs.hypothesis.utils import get_all_subclasses

# Register type strategy for Hypothesis's st.from_type()
# Note: Pass the callable, not an invoked strategy
st.register_type_strategy(cxc.AbstractChart, lambda _: charts())  # ty: ignore[missing-argument]
st.register_type_strategy(
    cxc.CartesianProductChart, lambda _: charts(cxc.CartesianProductChart)
)

for flag_cls in get_all_subclasses(cxc.AbstractDimensionalFlag, exclude_abstract=False):
    # Skip representation base classes
    if issubclass(flag_cls, cxc.AbstractChart):
        continue

    st.register_type_strategy(flag_cls, lambda typ: charts(typ))


# ============================================================================
# Optional strategy modules from other distributions
#
# A satellite package (e.g. `coordinaxs.curveframes`) registers a `charts()`
# overload for its own `AbstractChart` subclass under the
# `coordinaxs.hypothesis` entry-point group (mirroring `coordinax.frames`'s
# use of `coordinaxs.frames`). Load it unconditionally here so the overload
# exists before `chart_classes()` can enumerate the class -- otherwise a
# chart can appear via `__subclasses__()` before its strategy does,
# depending on import order.
_HYPOTHESIS_STRATEGY_ENTRYPOINT_GROUP: Final = "coordinaxs.hypothesis"
_OPTIONAL_HYPOTHESIS_STRATEGY_STATE: dict[str, bool] = {"loading": False}


def _load_optional_hypothesis_strategies() -> None:
    """Import strategy modules registered under the `coordinaxs.hypothesis` group."""
    # Guard against recursive entry-point loading during import-time cycles.
    if _OPTIONAL_HYPOTHESIS_STRATEGY_STATE["loading"]:
        return

    _OPTIONAL_HYPOTHESIS_STRATEGY_STATE["loading"] = True
    try:
        eps = sorted(
            entry_points(group=_HYPOTHESIS_STRATEGY_ENTRYPOINT_GROUP),
            key=lambda ep: ep.name,
        )
        exported = load_exports(
            eps,
            group=_HYPOTHESIS_STRATEGY_ENTRYPOINT_GROUP,
            noun="hypothesis strategy export",
        )
        globals().update(exported)
    finally:
        _OPTIONAL_HYPOTHESIS_STRATEGY_STATE["loading"] = False


_load_optional_hypothesis_strategies()
