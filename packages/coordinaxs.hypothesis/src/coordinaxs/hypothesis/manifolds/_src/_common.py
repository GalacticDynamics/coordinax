"""Shared helpers for the atlas and manifold strategies.

Lives here rather than in either module because ``manifold`` imports ``atlas``,
so the helper cannot sit in ``manifold`` without a cycle.
"""

__all__: tuple[str, ...] = ()

import functools as ft

from typing import Any, cast

import coordinax.charts as cxc

from coordinaxs.hypothesis.utils import get_all_subclasses


def _is_zero_arg_constructible(
    chart_cls: type[cxc.AbstractChart[Any, Any, Any]], /
) -> bool:
    """Return True if chart_cls can be instantiated with no arguments."""
    try:
        chart_cls()
    except TypeError:
        return False
    return True


@ft.cache
def matching_chart_classes_for_ndim(
    ndim: int, /
) -> tuple[type[cxc.AbstractChart[Any, Any, Any]], ...]:
    """Return zero-arg chart classes whose default instance has ``ndim``.

    Cached: this instantiates every concrete chart class, and both the
    ``CustomAtlas`` strategy and the ``CustomManifold`` ndim predicate call it
    on every draw.
    """
    classes: list[type[cxc.AbstractChart[Any, Any, Any]]] = []
    for cls in get_all_subclasses(cxc.AbstractChart, exclude_abstract=True):
        cls = cast("type[cxc.AbstractChart[Any, Any, Any]]", cls)
        if not _is_zero_arg_constructible(cls):
            continue
        if cls().ndim == ndim:
            classes.append(cls)
    return tuple(classes)
