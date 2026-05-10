"""Hypothesis strategies for coordinax representations."""

__all__ = ("can_pt_map", "get_init_params")


import functools as ft
import inspect
from collections.abc import Mapping
from typing import Any, Final, TypeVar

import coordinax.charts as cxc

T = TypeVar("T")
EMPTY: Final = inspect.Parameter.empty


def can_pt_map(
    from_chart: cxc.AbstractChart[Any, Any], to_chart: cxc.AbstractChart[Any, Any], /
) -> bool:
    """Return ``pt_map`` can convert between the two charts."""
    if type(to_chart) is type(from_chart):
        return True
    try:
        _ = to_chart.cartesian
        _ = from_chart.cartesian
    except (NotImplementedError, ValueError):
        return False
    return True


# ====================================================================

VAR_INPUT: Final = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


def _param_filter(name: str, param: inspect.Parameter, /) -> bool:
    """Filter function to identify parameters."""
    if name == "self" or param.kind in VAR_INPUT:
        return False
    # Skip keyword-only params with defaults (e.g., the inherited `manifold`
    # field on AbstractChart) — these are metadata, not mathematical parameters.
    if param.kind == inspect.Parameter.KEYWORD_ONLY and param.default is not EMPTY:
        return False
    return True


@ft.lru_cache(maxsize=256)
def get_init_params(cls: type, /) -> Mapping[str, inspect.Parameter]:
    """Get ``__init__`` parameters for a class (excluding 'self').

    This function is cached for performance.

    Parameters
    ----------
    cls : type
        The class to inspect.

    Returns
    -------
    Mapping[str, inspect.Parameter]
        Dictionary mapping parameter names to their Parameter objects.
        Only includes parameters without defaults.

    """
    try:
        sig = inspect.signature(cls.__init__)
    except (ValueError, TypeError):
        return {}

    return {n: p for n, p in sig.parameters.items() if _param_filter(n, p)}
