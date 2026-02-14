"""Hypothesis strategies for coordinax representations."""

__all__ = ("can_point_transform",)


import functools as ft
import inspect

from collections.abc import Mapping
from typing import Any

import coordinax.charts as cxc


def can_point_transform(
    to_chart: cxc.AbstractChart[Any, Any], from_chart: cxc.AbstractChart[Any, Any], /
) -> bool:
    """Return True if ``point_transform`` can convert between the two reps."""
    if type(to_chart) is type(from_chart):
        return True
    try:
        _ = to_chart.cartesian
        _ = from_chart.cartesian
    except (NotImplementedError, ValueError):
        return False
    return True


VAR_INPUT = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


def _param_filter(name: str, param: inspect.Parameter, /) -> bool:
    """Filter function to identify parameters."""
    return name != "self" and param.kind not in VAR_INPUT


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
        sig = inspect.signature(cls.__init__)  # type: ignore[misc]
    except (ValueError, TypeError):
        return {}

    return {n: p for n, p in sig.parameters.items() if _param_filter(n, p)}
